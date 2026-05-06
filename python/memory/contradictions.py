"""Contradiction detection + bitemporal adjudication.

Bitemporal model (Zep-inspired): never destructive delete.
- SUPERSEDES: B replaces A. Set A.valid_to = now, A.superseded_by = B.id, B.supersedes = A.id
- REFINES: B adds detail to A. Merge into single richer memory (manual confirm in user_review)
- COEXIST: Both true in different scopes/times. No state change.
- CONTRADICTS: Genuine conflict. Escalate to user_review.
- DUPLICATE: Same claim restated. Tombstone B.

Auto-apply only when Sonnet's confidence >= 0.85 AND decision is in
{SUPERSEDES, DUPLICATE, COEXIST}. Lower confidence or REFINES/CONTRADICTS
get contradiction_state=user_review for explicit human resolution.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from memory.store import MemoryStore


# ---- Result ----

@dataclass
class AdjudicationResult:
    decision: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    applied: bool = False  # True if auto-applied; False if escalated to user_review


# ---- Auto-apply rules ----

AUTO_APPLY_THRESHOLD = 0.85
AUTO_APPLY_DECISIONS = {"SUPERSEDES", "DUPLICATE", "COEXIST"}
ESCALATE_DECISIONS = {"CONTRADICTS", "REFINES"}


# ---- Detection ----

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def detect_candidates(
    store: MemoryStore,
    content: str,
    threshold: float = 0.55,
    k: int = 10,
) -> list[dict]:
    """Find existing memories that might contradict the new content.

    Uses cosine similarity on embeddings; returns rows above threshold.
    """
    if not content:
        return []

    vector = store._embedder.embed_one(content)
    rows = store.scan_v2()
    candidates: list[tuple[float, dict]] = []

    for row in rows:
        if row.get("tier") not in ("verified", "probationary"):
            continue
        row_vec = list(row["vector"]) if "vector" in row else None
        if not row_vec:
            continue
        score = _cosine(vector, row_vec)
        if score >= threshold:
            candidates.append((score, row))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in candidates[:k]]


# ---- Sonnet adjudication ----

ADJUDICATION_PROMPT = """You are a memory adjudicator. Two stored claims may conflict or relate.

CLAIM_A (id={a_id}, valid_from={a_valid_from}):
"{a_content}"

CLAIM_B (id={b_id}, valid_from={b_valid_from}):
"{b_content}"

Decide ONE of:
- SUPERSEDES: B replaces A (newer/correct version)
- REFINES: B adds detail to A; should merge
- COEXIST: Both true in different scopes or times (no conflict)
- CONTRADICTS: Genuine conflict; cannot auto-resolve
- DUPLICATE: Same claim restated; drop B

Return ONLY JSON:
{{"decision": "...", "reasoning": "...", "confidence": 0.0-1.0,
  "merged_text": "..." or null}}
"""


def adjudicate_pair(
    store: MemoryStore,
    *,
    a_id: str,
    b_id: str,
    chat_fn=None,
    model: str = "sonnet",
) -> AdjudicationResult:
    """Send a pair of memories to Sonnet, apply verdict to the store."""
    if chat_fn is None:
        from memory.l1_manager import _default_chat_fn
        chat_fn = _default_chat_fn

    # Fetch the rows
    rows = store.scan_v2()
    by_id = {r["id"]: r for r in rows}
    if a_id not in by_id or b_id not in by_id:
        return AdjudicationResult(decision="MISSING")

    a, b = by_id[a_id], by_id[b_id]

    # Bitemporal guard: if A is already superseded, reject any new SUPERSEDES
    # verdict to preserve the chain (avoid overwriting valid_to / superseded_by).
    a_already_superseded = bool(a.get("valid_to")) or bool(a.get("superseded_by"))

    prompt = ADJUDICATION_PROMPT.format(
        a_id=a_id,
        a_valid_from=a.get("valid_from", ""),
        a_content=a.get("content", ""),
        b_id=b_id,
        b_valid_from=b.get("valid_from", ""),
        b_content=b.get("content", ""),
    )

    try:
        response = chat_fn(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            timeout=30,
        )
    except Exception:
        return AdjudicationResult(decision="ERROR")

    if response is None:
        return AdjudicationResult(decision="ERROR")

    text = response.get("content", "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return AdjudicationResult(decision="PARSE_ERROR")

    decision = (parsed.get("decision") or "").upper()
    confidence = float(parsed.get("confidence") or 0.0)
    reasoning = (parsed.get("reasoning") or "")[:300]

    result = AdjudicationResult(
        decision=decision,
        confidence=confidence,
        reasoning=reasoning,
        applied=False,
    )

    # Reject SUPERSEDES against an already-superseded A (preserves bitemporal chain).
    if decision == "SUPERSEDES" and a_already_superseded:
        result.decision = "ALREADY_SUPERSEDED"
        result.applied = False
        from memory.write_gate import _audit
        _audit(store, memory_id=a_id, op="adjudicate",
               who="sonnet_promoter",
               why="rejected: A already superseded; preserving chain",
               before=a_id, after="rejected")
        return result

    auto_apply = (
        decision in AUTO_APPLY_DECISIONS
        and confidence >= AUTO_APPLY_THRESHOLD
    )

    now_iso = datetime.now(timezone.utc).isoformat()

    if auto_apply:
        if decision == "SUPERSEDES":
            with store.lock:
                store.table.update(
                    where=f"id = '{a_id}'",
                    values={"valid_to": now_iso, "superseded_by": b_id},
                )
                store.table.update(
                    where=f"id = '{b_id}'",
                    values={"supersedes": a_id},
                )
        elif decision == "DUPLICATE":
            with store.lock:
                store.table.update(
                    where=f"id = '{b_id}'",
                    values={"tier": "tombstoned"},
                )
        # COEXIST: no state change, just record audit

        from memory.write_gate import _audit
        _audit(store, memory_id=b_id, op="adjudicate", who="sonnet_promoter",
               why=f"{decision}: {reasoning}",
               before=str(a_id), after=str(b_id))
        result.applied = True
    else:
        # Escalate: set contradiction_state=user_review on BOTH
        with store.lock:
            store.table.update(
                where=f"id = '{a_id}'",
                values={"contradiction_state": "user_review"},
            )
            store.table.update(
                where=f"id = '{b_id}'",
                values={"contradiction_state": "user_review"},
            )
        from memory.write_gate import _audit
        _audit(store, memory_id=b_id, op="adjudicate", who="sonnet_promoter",
               why=f"{decision} (low confidence {confidence}): {reasoning}",
               before=str(a_id), after="user_review")

    return result
