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
    """Tier based on writer + provenance."""
    if writer in CANDIDATE_ONLY_WRITERS:
        return "candidate"
    if writer in USER_WRITERS or provenance == "user_stated":
        return "verified"
    return "probationary"


def _find_dedup_candidate(store: MemoryStore, vector: list[float]) -> dict | None:
    """Cosine-search the store for a near-duplicate. Returns the row or None.

    Considers verified/probationary/candidate tiers (excludes only tombstoned).
    Including candidate prevents subconscious from spamming the same hallucination
    as N separate candidate rows — instead, seen_count is bumped on the existing one.
    """
    rows = store.scan_v2()
    best: tuple[float, dict] | None = None
    for row in rows:
        if row.get("tier") == "tombstoned":
            continue
        row_vec = list(row["vector"]) if "vector" in row else None
        if not row_vec:
            continue
        score = _cosine(vector, row_vec)
        if score >= DEDUP_COSINE_THRESHOLD:
            if best is None or score > best[0]:
                best = (score, row)
    return best[1] if best else None


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
        # We allow unknown writers but they default to probationary
        pass

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
        why=f"new memory at tier={tier} from {source_ref}",
        after=json.dumps({"tier": tier, "provenance": provenance, "writer": writer}),
    )

    return rec.id
