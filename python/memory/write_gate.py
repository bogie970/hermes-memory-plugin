"""Write gate — single entry point for ALL memory writes.

Enforces the three-tier pipeline:
    candidate (subconscious-only) -> probationary -> verified

Per-write checks:
    1. content + source_ref required
    2. tier assigned by writer (subconscious_haiku ALWAYS candidate)
    3. filesystem grounding: candidate-demote on missing code refs
    4. cosine dedup: identical/near-duplicate content bumps seen_count
    5. audit log entry recorded

Concurrency: relies on MemoryStore's existing FileLock.
"""

from __future__ import annotations

import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from memory.grounding import extract_code_refs, filesystem_exists
from memory.schema import MemoryRecord, MemoryType
from memory.secret_scrub import scrub_with_count
from memory.store import AUDIT_TABLE_NAME, MemoryStore


# ---- Tier assignment rules ----

CANDIDATE_ONLY_WRITERS = {"subconscious_haiku"}
USER_WRITERS = {"user", "manual"}
KNOWN_WRITERS = CANDIDATE_ONLY_WRITERS | USER_WRITERS | {
    "sonnet_promoter", "opus_auditor", "system",
}

DEDUP_COSINE_THRESHOLD = 0.92  # cosine sim above which we treat as duplicate


class WriteRejected(Exception):
    """Raised when a memory write fails validation."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_id(value: str) -> str:
    """Escape single quotes for safe LanceDB SQL where-clause use."""
    return str(value).replace("'", "''")


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _audit(store: MemoryStore, *, memory_id: str, op: str, who: str, why: str,
           before: str = "", after: str = "") -> None:
    """Append to memory_audit table. Idempotent — ensures table exists."""
    try:
        audit_table = store.db.open_table(AUDIT_TABLE_NAME)
    except (FileNotFoundError, ValueError):
        from memory.store import LANCE_AUDIT_SCHEMA
        audit_table = store.db.create_table(AUDIT_TABLE_NAME, schema=LANCE_AUDIT_SCHEMA)

    audit_table.add([{
        "id": str(uuid.uuid4()),
        "memory_id": memory_id,
        "op": op,
        "who": who,
        "when": _now_iso(),
        "why": why,
        "before": before,
        "after": after,
    }])


def _assign_tier(writer: str, provenance: str) -> str:
    """Tier based on writer + provenance.

    Per injection-audit (2026-05-19): unknown writers MUST NOT achieve
    verified tier via provenance='user_stated'. Only known writers can
    claim user_stated; unknown writers default to probationary regardless.
    """
    if writer in CANDIDATE_ONLY_WRITERS:
        return "candidate"
    if writer in USER_WRITERS:
        return "verified"
    if writer in KNOWN_WRITERS and provenance == "user_stated":
        return "verified"
    return "probationary"


def _find_dedup_candidate(store: MemoryStore, vector: list[float]) -> dict | None:
    """Native LanceDB vector search for a near-duplicate. Returns the row or None.

    Considers verified/probationary/candidate tiers (excludes only tombstoned).
    Uses LanceDB's vector index (or linear scan for small stores) — drops
    write-time dedup from O(N) Python cosine to ~O(log N) at scale.

    Embeddings are L2-normalized (per EmbeddingService), so LanceDB's
    default cosine distance == 1 - cos(theta). We threshold on distance.
    """
    distance_threshold = 1.0 - DEDUP_COSINE_THRESHOLD  # 0.08

    try:
        results = (
            store.table.search(vector)
            .where("tier != 'tombstoned'")
            .limit(1)
            .to_list()
        )
    except Exception:
        # If the where-clause or search fails, fall back to no-dedup
        # (write proceeds; duplicate may be caught later by cleanup).
        return None

    if not results:
        return None
    top = results[0]
    distance = top.get("_distance", 1.0)
    if distance > distance_threshold:
        return None
    return top


def write_memory(
    store: MemoryStore,
    *,
    content: str,
    writer: str,
    provenance: str,
    source_ref: str,
    confidence: float,
    tags: list[str] | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    category: str = "general",
    source: str = "hermes",
) -> str:
    """Write a memory through the gate. Returns the resulting record id.

    On dedup: returns the existing record's id and bumps its seen_count.
    On rejection: raises WriteRejected.
    """
    # 1. Validate input
    if not content or not content.strip():
        raise WriteRejected("content required")
    if not source_ref:
        raise WriteRejected("source_ref required")
    if writer not in KNOWN_WRITERS:
        # Unknown writers cannot claim user_stated provenance — that path
        # would bypass _assign_tier's known-writer check and produce a
        # verified-tier record. Reject explicitly per injection-audit.
        if provenance == "user_stated":
            raise WriteRejected(
                f"unknown writer {writer!r} cannot claim provenance='user_stated'"
            )
        # Otherwise allow; _assign_tier will return probationary.

    # 1a. Scrub credentials before they hit L2 — pasted API keys / tokens
    #     would otherwise resurface via retrieval months later.
    content, secrets_redacted = scrub_with_count(content)

    # 2. Tier assignment
    tier = _assign_tier(writer, provenance)
    effective_confidence = confidence

    # 3. Filesystem grounding for code refs (demotion)
    code_refs = extract_code_refs(content)
    missing_refs = [r for r in code_refs if not filesystem_exists(r)]
    if missing_refs and tier == "verified":
        tier = "candidate"
        effective_confidence *= 0.5

    # 4. Compute embedding (slow; do this outside the critical section)
    vector = store._embedder.embed_one(content)

    # 5. Critical section: dedup-check + insert/update must be atomic to prevent
    #    races where N concurrent threads all miss-dedup and create N copies.
    with store.lock:
        existing = _find_dedup_candidate(store, vector)
        if existing is not None:
            existing_id = _sanitize_id(existing["id"])
            existing_tier = existing.get("tier", "")

            # Trust-aware seen_count: subconscious writers cannot boost the
            # seen_count of an already-verified/probationary row, since that
            # would let hallucinations accelerate promotion of unrelated trusted
            # memories. Audit the dedup hit either way.
            should_bump = not (
                writer in CANDIDATE_ONLY_WRITERS
                and existing_tier in ("verified", "probationary")
            )

            if should_bump:
                new_seen = int(existing.get("seen_count", 1)) + 1
                store.table.update(
                    where=f"id = '{existing_id}'",
                    values={"seen_count": new_seen, "last_seen_at": _now_iso()},
                )
                _audit(
                    store, memory_id=existing_id, op="seen_bump", who=writer,
                    why=f"dedup hit on {source_ref}",
                    before=str(new_seen - 1), after=str(new_seen),
                )
            else:
                _audit(
                    store, memory_id=existing_id, op="seen_bump_blocked",
                    who=writer,
                    why=f"trust-block: {writer} cannot bump verified row from {source_ref}",
                    before=str(existing.get("seen_count", 1)),
                    after=str(existing.get("seen_count", 1)),
                )
            return existing_id

        # New record path
        rec = MemoryRecord(
            content=content,
            memory_type=memory_type,
            category=category,
            source=source,
            importance=effective_confidence,
            tags=tags or [],
        )
        rec.metadata["_v2_overrides"] = {
            "tier": tier,
            "provenance": provenance,
            "source_ref": source_ref,
            "writer": writer,
            "confidence": effective_confidence,
        }
        row = rec.to_lance_dict(vector, include_v2=True)
        store.table.add([row])

    _audit(
        store, memory_id=rec.id, op="create", who=writer,
        why=f"new memory at tier={tier} from {source_ref}"
            + (f" (scrubbed {secrets_redacted} secrets)" if secrets_redacted else ""),
        after=json.dumps({"tier": tier, "provenance": provenance, "writer": writer,
                          "secrets_redacted": secrets_redacted}),
    )

    return rec.id
