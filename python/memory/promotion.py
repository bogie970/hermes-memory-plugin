"""Promotion job — runs daily via /schedule.

Sonnet adjudicates candidate memories:
  - seen_count >= 2 (re-encountered) OR
  - confidence >= 0.8 (initially high-confidence)
  -> consult Sonnet for {promote, reject, hold}
  -> promote: tier candidate -> probationary
  -> reject: stays candidate (downstream may tombstone)
  -> hold: stays candidate, will be re-evaluated next day

Probationary memories survive 7 days without contradiction -> verified
(handled by run_weekly() — Phase F.2).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from memory.store import MemoryStore


# ---- Result dataclass ----

@dataclass
class PromotionResult:
    processed: int = 0
    promoted: int = 0
    rejected: int = 0
    held: int = 0
    errors: int = 0


# ---- Eligibility ----

SEEN_COUNT_THRESHOLD = 2
CONFIDENCE_THRESHOLD = 0.8


def _eligible_for_review(row: dict) -> bool:
    if row.get("tier") != "candidate":
        return False
    seen = int(row.get("seen_count") or 1)
    conf = float(row.get("confidence") or 0.0)
    return seen >= SEEN_COUNT_THRESHOLD or conf >= CONFIDENCE_THRESHOLD


# ---- Sonnet prompt ----

ADJUDICATION_PROMPT = """You are a memory adjudicator. The candidate claim below was proposed
by a smaller model from a transcript and needs your verdict before being
promoted to the verified memory tier.

Candidate (id={id}, confidence={confidence}, seen_count={seen_count}):
"{content}"

Decide ONE of:
- promote: claim is well-supported, factual, and useful long-term
- reject: claim contradicts known facts or is clearly wrong
- hold: insufficient signal yet, leave as candidate

Return ONLY JSON:
{{"verdict": "...", "rationale": "..."}}
"""


def _audit(store: MemoryStore, *, memory_id: str, op: str, who: str, why: str,
           before: str = "", after: str = "") -> None:
    """Append to memory_audit table."""
    from memory.write_gate import _audit as _write_gate_audit
    _write_gate_audit(store, memory_id=memory_id, op=op, who=who, why=why,
                       before=before, after=after)


# ---- Main entry point ----

def run_daily(
    store: MemoryStore,
    chat_fn=None,
    model: str = "sonnet",
) -> PromotionResult:
    """Run the daily promotion pass.

    For each eligible candidate, ask Sonnet to adjudicate.
    Apply verdict; record audit entry.

    Args:
        store: MemoryStore (must be v2 schema)
        chat_fn: callable(messages, model, **kw) -> response dict
        model: which model handle to use ("sonnet" by default)

    Returns:
        PromotionResult with counts.
    """
    if chat_fn is None:
        from memory.l1_manager import _default_chat_fn
        chat_fn = _default_chat_fn

    result = PromotionResult()

    # Snapshot eligible candidates under a brief lock — don't hold the
    # store lock across slow Sonnet API calls (which can take seconds each)
    # since that would starve concurrent write_memory calls from l1_watch.
    with store.lock:
        rows = store.scan_v2_lean()
        candidates = [r for r in rows if _eligible_for_review(r)]

    now_iso = datetime.now(timezone.utc).isoformat()

    for row in candidates:
        result.processed += 1
        memory_id = row["id"]
        prompt = ADJUDICATION_PROMPT.format(
            id=memory_id,
            confidence=row.get("confidence", 0.0),
            seen_count=row.get("seen_count", 1),
            content=row.get("content", ""),
        )

        try:
            response = chat_fn(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                timeout=30,
            )
        except Exception:
            result.errors += 1
            continue

        if response is None:
            result.errors += 1
            continue

        text = response.get("content", "").strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            result.errors += 1
            continue

        verdict = parsed.get("verdict", "hold").lower()
        rationale = parsed.get("rationale", "")[:200]

        if verdict == "promote":
            # Acquire lock briefly. Where-clause guard prevents double-promote
            # if a parallel promotion run already moved this row.
            with store.lock:
                store.table.update(
                    where=f"id = '{memory_id}' AND tier = 'candidate'",
                    values={"tier": "probationary", "promoted_at": now_iso},
                )
            _audit(store, memory_id=memory_id, op="promote",
                   who="sonnet_promoter",
                   why=f"sonnet verdict: {rationale}",
                   before="candidate", after="probationary")
            result.promoted += 1
        elif verdict == "reject":
            _audit(store, memory_id=memory_id, op="reject",
                   who="sonnet_promoter",
                   why=f"sonnet verdict: {rationale}",
                   before="candidate", after="candidate")
            result.rejected += 1
        else:
            result.held += 1

    return result
