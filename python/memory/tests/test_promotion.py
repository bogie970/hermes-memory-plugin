"""Phase F — promotion job tests.

Sonnet daily promotes candidates -> probationary based on:
  - seen_count >= 2 (re-encountered) OR
  - confidence >= 0.8 (initially high-confidence)

Probationary memories survive 7 days without contradiction -> verified.

Run: pytest aisys/memory/tests/test_promotion.py -v
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


@pytest.fixture
def v2_store(lance_store_v1):
    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)
    return lance_store_v1


def _create_candidate(store, content: str, confidence: float = 0.5,
                       seen_count: int = 1) -> str:
    """Helper: insert a candidate-tier memory and bump seen_count if needed."""
    from memory.write_gate import write_memory
    rec_id = write_memory(
        store=store,
        content=content,
        writer="subconscious_haiku",
        provenance="llm_inferred",
        source_ref=f"test:{content[:20]}",
        confidence=confidence,
    )
    if seen_count > 1:
        with store.lock:
            store.table.update(
                where=f"id = '{rec_id}'",
                values={"seen_count": seen_count},
            )
    return rec_id


# ----- Promotion criteria -----

def test_seen_count_2_with_promote_verdict_promotes(v2_store):
    """Candidate with seen_count>=2 + Sonnet 'promote' verdict -> probationary."""
    cid = _create_candidate(v2_store, "decision X", confidence=0.5, seen_count=2)

    def fake_sonnet(messages, model="sonnet", **kw):
        return {"role": "assistant", "content": json.dumps({
            "verdict": "promote",
            "rationale": "re-encountered, content stable",
        })}

    from memory.promotion import run_daily
    result = run_daily(v2_store, chat_fn=fake_sonnet)
    assert result.promoted == 1
    rec = next(r for r in v2_store.scan_v2() if r["id"] == cid)
    assert rec["tier"] == "probationary"


def test_low_confidence_single_encounter_not_promoted(v2_store):
    """Single-encounter, low confidence stays in candidate."""
    cid = _create_candidate(v2_store, "speculative thing", confidence=0.3,
                              seen_count=1)

    def fake_sonnet(messages, **kw):
        return {"role": "assistant", "content": json.dumps({
            "verdict": "hold",
            "rationale": "not enough signal",
        })}

    from memory.promotion import run_daily
    result = run_daily(v2_store, chat_fn=fake_sonnet)
    assert result.promoted == 0
    rec = next(r for r in v2_store.scan_v2() if r["id"] == cid)
    assert rec["tier"] == "candidate"


def test_high_confidence_single_encounter_eligible_for_review(v2_store):
    """High-confidence single-encounter goes to Sonnet for review."""
    cid = _create_candidate(v2_store, "important fact",
                              confidence=0.85, seen_count=1)

    seen_prompts = []

    def fake_sonnet(messages, **kw):
        seen_prompts.append(messages)
        return {"role": "assistant", "content": json.dumps({
            "verdict": "promote", "rationale": "high confidence",
        })}

    from memory.promotion import run_daily
    result = run_daily(v2_store, chat_fn=fake_sonnet)
    assert len(seen_prompts) == 1  # Sonnet was consulted
    assert result.promoted == 1


def test_sonnet_reject_keeps_candidate(v2_store):
    """If Sonnet says 'reject', the memory stays at candidate (or could be tombstoned)."""
    cid = _create_candidate(v2_store, "wrong thing", confidence=0.9, seen_count=2)

    def fake_sonnet(messages, **kw):
        return {"role": "assistant", "content": json.dumps({
            "verdict": "reject",
            "rationale": "contradicts known facts",
        })}

    from memory.promotion import run_daily
    result = run_daily(v2_store, chat_fn=fake_sonnet)
    assert result.promoted == 0
    assert result.rejected == 1
    rec = next(r for r in v2_store.scan_v2() if r["id"] == cid)
    assert rec["tier"] == "candidate"


# ----- Audit log -----

def test_promotion_records_audit_entry(v2_store):
    """Successful promotion writes a 'promote' audit entry."""
    cid = _create_candidate(v2_store, "stable claim", confidence=0.6, seen_count=2)

    def fake_sonnet(messages, **kw):
        return {"role": "assistant", "content": json.dumps({
            "verdict": "promote", "rationale": "ok"
        })}

    from memory.promotion import run_daily
    run_daily(v2_store, chat_fn=fake_sonnet)
    audits = v2_store.audit_scan()
    promote_audits = [a for a in audits if a["op"] == "promote" and a["memory_id"] == cid]
    assert len(promote_audits) == 1


# ----- Result dataclass -----

def test_run_daily_returns_summary(v2_store):
    """run_daily returns counts of processed/promoted/rejected/held."""
    _create_candidate(v2_store, "thing 1", confidence=0.85, seen_count=1)
    _create_candidate(v2_store, "thing 2", confidence=0.5, seen_count=2)
    _create_candidate(v2_store, "thing 3", confidence=0.3, seen_count=1)

    def fake_sonnet(messages, **kw):
        return {"role": "assistant", "content": json.dumps({
            "verdict": "promote", "rationale": "ok"
        })}

    from memory.promotion import run_daily
    result = run_daily(v2_store, chat_fn=fake_sonnet)
    # thing 3 is below all thresholds -> not even consulted; processed = 2
    assert result.processed >= 2
    assert result.promoted >= 1


def test_run_daily_skips_when_no_eligible_candidates(v2_store):
    """Empty/all-low-confidence stores produce 0 processed."""
    _create_candidate(v2_store, "tiny", confidence=0.1, seen_count=1)

    from memory.promotion import run_daily
    result = run_daily(v2_store, chat_fn=lambda *a, **k: None)
    assert result.processed == 0
