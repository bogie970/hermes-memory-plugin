"""Phase G — contradiction detection tests.

Bitemporal model (Zep-inspired): never delete, set valid_to.
Sonnet adjudicates {SUPERSEDES, REFINES, COEXIST, CONTRADICTS, DUPLICATE}.

Run: pytest aisys/memory/tests/test_contradictions.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


@pytest.fixture
def v2_store(lance_store_v1):
    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)
    return lance_store_v1


# ----- detect_contradictions -----

def test_detect_finds_high_similarity_candidates(v2_store):
    """Identical content matches at cosine 1.0 (mechanism verified).

    Note: semantic similarity testing requires real embeddings; with the
    deterministic FakeEmbedder, only exact matches produce high cosine.
    Production semantic accuracy is verified by real-embedder eval suite.
    """
    from memory.write_gate import write_memory
    write_memory(store=v2_store, content="Atlas uses 4x A100 GPUs",
                  writer="user", provenance="user_stated",
                  source_ref="t:1", confidence=1.0)

    from memory.contradictions import detect_candidates
    # Query identical to existing -> cosine 1.0
    candidates = detect_candidates(v2_store,
                                    content="Atlas uses 4x A100 GPUs",
                                    threshold=0.5)
    assert len(candidates) >= 1


def test_detect_returns_empty_when_no_overlap(v2_store):
    from memory.contradictions import detect_candidates
    candidates = detect_candidates(v2_store, content="totally novel content",
                                    threshold=0.95)
    assert candidates == []


# ----- adjudicate via Sonnet -----

def test_adjudicate_supersedes_decision(v2_store):
    """SUPERSEDES verdict sets valid_to on old, supersedes link on new."""
    from memory.write_gate import write_memory
    old_id = write_memory(store=v2_store, content="Atlas has 4x A100",
                           writer="user", provenance="user_stated",
                           source_ref="t:old", confidence=1.0)
    new_id = write_memory(store=v2_store, content="Atlas has 8x H100",
                           writer="user", provenance="user_stated",
                           source_ref="t:new", confidence=1.0)

    def fake_sonnet(messages, **kw):
        return {"role": "assistant", "content": json.dumps({
            "decision": "SUPERSEDES",
            "reasoning": "newer hardware replaces older",
            "merged_text": None,
            "confidence": 0.92,
        })}

    from memory.contradictions import adjudicate_pair
    result = adjudicate_pair(v2_store, a_id=old_id, b_id=new_id,
                              chat_fn=fake_sonnet)
    assert result.decision == "SUPERSEDES"
    assert result.applied  # high confidence -> auto-applied

    rows = v2_store.scan_v2()
    old_row = next(r for r in rows if r["id"] == old_id)
    new_row = next(r for r in rows if r["id"] == new_id)
    assert old_row["valid_to"]  # set to a timestamp
    assert old_row["superseded_by"] == new_id
    assert new_row["supersedes"] == old_id


def test_adjudicate_low_confidence_escalates_to_user_review(v2_store):
    """Low confidence (<0.85) sets contradiction_state=user_review, doesn't apply."""
    from memory.write_gate import write_memory
    a = write_memory(store=v2_store, content="claim A",
                      writer="user", provenance="user_stated",
                      source_ref="t:1", confidence=1.0)
    b = write_memory(store=v2_store, content="claim B",
                      writer="user", provenance="user_stated",
                      source_ref="t:2", confidence=1.0)

    def fake_sonnet(messages, **kw):
        return {"role": "assistant", "content": json.dumps({
            "decision": "CONTRADICTS",
            "reasoning": "unclear",
            "confidence": 0.5,
        })}

    from memory.contradictions import adjudicate_pair
    result = adjudicate_pair(v2_store, a_id=a, b_id=b, chat_fn=fake_sonnet)
    assert not result.applied
    rows = v2_store.scan_v2()
    a_row = next(r for r in rows if r["id"] == a)
    assert a_row["contradiction_state"] == "user_review"


def test_adjudicate_duplicate_decision_tombstones_b(v2_store):
    """DUPLICATE verdict at high confidence tombstones the second record."""
    from memory.write_gate import write_memory
    a = write_memory(store=v2_store, content="Hermes is the AI co-pilot",
                      writer="user", provenance="user_stated",
                      source_ref="t:1", confidence=1.0)
    b = write_memory(store=v2_store, content="Hermes serves as Jacob's co-pilot",
                      writer="user", provenance="user_stated",
                      source_ref="t:2", confidence=1.0)

    def fake_sonnet(messages, **kw):
        return {"role": "assistant", "content": json.dumps({
            "decision": "DUPLICATE",
            "reasoning": "same claim restated",
            "confidence": 0.95,
        })}

    from memory.contradictions import adjudicate_pair
    result = adjudicate_pair(v2_store, a_id=a, b_id=b, chat_fn=fake_sonnet)
    assert result.applied
    rows = v2_store.scan_v2()
    b_row = next(r for r in rows if r["id"] == b)
    assert b_row["tier"] == "tombstoned"


def test_adjudicate_coexist_keeps_both_active(v2_store):
    """COEXIST verdict leaves both records active."""
    from memory.write_gate import write_memory
    a = write_memory(store=v2_store, content="Hermes does physics",
                      writer="user", provenance="user_stated",
                      source_ref="t:1", confidence=1.0)
    b = write_memory(store=v2_store, content="Atlas does GPU rental",
                      writer="user", provenance="user_stated",
                      source_ref="t:2", confidence=1.0)

    def fake_sonnet(messages, **kw):
        return {"role": "assistant", "content": json.dumps({
            "decision": "COEXIST",
            "reasoning": "different scopes",
            "confidence": 0.9,
        })}

    from memory.contradictions import adjudicate_pair
    result = adjudicate_pair(v2_store, a_id=a, b_id=b, chat_fn=fake_sonnet)
    assert result.applied  # decision applied (audit recorded)
    rows = v2_store.scan_v2()
    a_row = next(r for r in rows if r["id"] == a)
    b_row = next(r for r in rows if r["id"] == b)
    # Both still verified, neither superseded
    assert a_row["tier"] == "verified"
    assert b_row["tier"] == "verified"


def test_adjudicate_blocks_supersedes_when_already_superseded(v2_store):
    """If A is already superseded (valid_to set), don't overwrite it.

    Bitemporal correctness: the chain A -> B -> C must remain intact.
    A new SUPERSEDES verdict against an already-superseded A is rejected.
    """
    from memory.write_gate import write_memory
    a = write_memory(store=v2_store, content="claim A v1",
                      writer="user", provenance="user_stated",
                      source_ref="t:a", confidence=1.0)
    b = write_memory(store=v2_store, content="claim B replaces A",
                      writer="user", provenance="user_stated",
                      source_ref="t:b", confidence=1.0)
    c = write_memory(store=v2_store, content="claim C tries to also replace A",
                      writer="user", provenance="user_stated",
                      source_ref="t:c", confidence=1.0)

    def fake_sonnet(messages, **kw):
        return {"role": "assistant", "content": json.dumps({
            "decision": "SUPERSEDES", "reasoning": "ok", "confidence": 0.95,
        })}

    from memory.contradictions import adjudicate_pair
    # First: B supersedes A — applies normally
    r1 = adjudicate_pair(v2_store, a_id=a, b_id=b, chat_fn=fake_sonnet)
    assert r1.applied
    rows = v2_store.scan_v2()
    a_after_first = next(r for r in rows if r["id"] == a)
    first_valid_to = a_after_first["valid_to"]
    assert first_valid_to  # set
    assert a_after_first["superseded_by"] == b

    # Second: C tries to supersede A — should be REJECTED to preserve chain
    r2 = adjudicate_pair(v2_store, a_id=a, b_id=c, chat_fn=fake_sonnet)
    assert r2.decision == "ALREADY_SUPERSEDED"
    assert not r2.applied
    rows2 = v2_store.scan_v2()
    a_after_second = next(r for r in rows2 if r["id"] == a)
    # valid_to and superseded_by unchanged
    assert a_after_second["valid_to"] == first_valid_to
    assert a_after_second["superseded_by"] == b


def test_adjudicate_records_audit_entry(v2_store):
    from memory.write_gate import write_memory
    a = write_memory(store=v2_store, content="A", writer="user",
                      provenance="user_stated", source_ref="t:1", confidence=1.0)
    b = write_memory(store=v2_store, content="B", writer="user",
                      provenance="user_stated", source_ref="t:2", confidence=1.0)

    def fake_sonnet(messages, **kw):
        return {"role": "assistant", "content": json.dumps({
            "decision": "SUPERSEDES", "reasoning": "x", "confidence": 0.9,
        })}

    from memory.contradictions import adjudicate_pair
    adjudicate_pair(v2_store, a_id=a, b_id=b, chat_fn=fake_sonnet)
    audits = v2_store.audit_scan()
    adj_audits = [aa for aa in audits if aa["op"] == "adjudicate"]
    assert len(adj_audits) >= 1
