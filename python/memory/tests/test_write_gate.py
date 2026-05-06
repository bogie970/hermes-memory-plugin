"""Phase C — write gate tests.

Verifies the three-tier write pipeline:
  candidate (subconscious-only) -> probationary (Sonnet promotion) -> verified

Run: pytest aisys/memory/tests/test_write_gate.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


# Helper: store with v2 schema ready
@pytest.fixture
def v2_store(lance_store_v1, sample_v1_records):
    """Store with v2 schema migrated."""
    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)
    return lance_store_v1


# ----- Tier assignment by writer -----

def test_subconscious_writes_to_candidate_only(v2_store):
    """subconscious_haiku writer is forced to candidate tier."""
    from memory.write_gate import write_memory
    rec_id = write_memory(
        store=v2_store,
        content="some claim about state of code",
        writer="subconscious_haiku",
        provenance="llm_inferred",
        source_ref="transcript:42",
        confidence=0.95,  # even high confidence stays candidate
    )
    rec = v2_store.scan_v2()
    matched = [r for r in rec if r["id"] == rec_id]
    assert len(matched) == 1
    assert matched[0]["tier"] == "candidate"


def test_user_stated_goes_to_verified(v2_store):
    """User-stated facts bypass probation."""
    from memory.write_gate import write_memory
    rec_id = write_memory(
        store=v2_store,
        content="Jacob has ADHD; needs numbered lists",
        writer="user",
        provenance="user_stated",
        source_ref="prompt:turn_3",
        confidence=1.0,
    )
    rec = next(r for r in v2_store.scan_v2() if r["id"] == rec_id)
    assert rec["tier"] == "verified"


def test_unknown_writer_defaults_to_probationary(v2_store):
    """Writers we don't recognize default to probationary tier."""
    from memory.write_gate import write_memory
    rec_id = write_memory(
        store=v2_store,
        content="some fact",
        writer="some_unknown_agent",
        provenance="llm_inferred",
        source_ref="t:1",
        confidence=0.7,
    )
    rec = next(r for r in v2_store.scan_v2() if r["id"] == rec_id)
    assert rec["tier"] == "probationary"


# ----- Source ref required -----

def test_missing_source_ref_rejected(v2_store):
    """source_ref is required."""
    from memory.write_gate import write_memory, WriteRejected
    with pytest.raises(WriteRejected, match="source_ref"):
        write_memory(
            store=v2_store, content="x", writer="user",
            provenance="user_stated", source_ref="", confidence=0.5,
        )


def test_empty_content_rejected(v2_store):
    """Empty content is rejected."""
    from memory.write_gate import write_memory, WriteRejected
    with pytest.raises(WriteRejected, match="content"):
        write_memory(
            store=v2_store, content="", writer="user",
            provenance="user_stated", source_ref="t:1", confidence=0.5,
        )


# ----- Filesystem grounding demotion -----

def test_fs_grounding_demotes_when_code_ref_missing(v2_store):
    """A user-stated memory with a missing file ref gets demoted to candidate."""
    from memory.write_gate import write_memory
    rec_id = write_memory(
        store=v2_store,
        content="See nonexistent_phantom_file_99999.py for the bug fix",
        writer="user",
        provenance="user_stated",
        source_ref="t:1",
        confidence=0.9,
    )
    rec = next(r for r in v2_store.scan_v2() if r["id"] == rec_id)
    assert rec["tier"] == "candidate"  # demoted from verified
    assert rec["confidence"] < 0.9     # halved


def test_fs_grounding_passes_for_real_file(v2_store, tmp_path):
    """Real file ref preserves verified tier."""
    real_file = tmp_path / "real_test.py"
    real_file.write_text("# real")
    from memory.write_gate import write_memory
    rec_id = write_memory(
        store=v2_store,
        content=f"See {real_file} for the bug fix",
        writer="user",
        provenance="user_stated",
        source_ref="t:1",
        confidence=0.9,
    )
    rec = next(r for r in v2_store.scan_v2() if r["id"] == rec_id)
    assert rec["tier"] == "verified"
    assert rec["confidence"] == pytest.approx(0.9)


# ----- Dedup on write -----

def test_dedup_on_identical_content_returns_same_id(v2_store):
    """Identical content writes return the existing record id and bump seen_count."""
    from memory.write_gate import write_memory
    a = write_memory(
        store=v2_store, content="LanceDB is the chosen vector store",
        writer="user", provenance="user_stated",
        source_ref="t:1", confidence=1.0,
    )
    b = write_memory(
        store=v2_store, content="LanceDB is the chosen vector store",
        writer="user", provenance="user_stated",
        source_ref="t:2", confidence=1.0,
    )
    assert a == b
    rec = next(r for r in v2_store.scan_v2() if r["id"] == a)
    assert rec["seen_count"] >= 2


def test_dedup_does_not_apply_to_different_content(v2_store):
    """Distinct content creates distinct records."""
    from memory.write_gate import write_memory
    a = write_memory(
        store=v2_store, content="Atlas is GPU rental",
        writer="user", provenance="user_stated",
        source_ref="t:1", confidence=1.0,
    )
    b = write_memory(
        store=v2_store, content="Hermes is the AI co-pilot",
        writer="user", provenance="user_stated",
        source_ref="t:2", confidence=1.0,
    )
    assert a != b


# ----- Audit log -----

def test_subconscious_dedup_does_not_boost_verified_seen_count(v2_store):
    """When subconscious_haiku content matches a verified row, seen_count
    must NOT be bumped — that would let hallucinations falsely accelerate
    promotion of unrelated verified memories.
    """
    from memory.write_gate import write_memory
    # User writes a verified claim
    a = write_memory(
        store=v2_store,
        content="LanceDB is the chosen vector store",
        writer="user",
        provenance="user_stated",
        source_ref="t:user",
        confidence=1.0,
    )
    initial_seen = next(r for r in v2_store.scan_v2() if r["id"] == a)["seen_count"]

    # Subconscious dedups against the same content 3 times
    for i in range(3):
        result = write_memory(
            store=v2_store,
            content="LanceDB is the chosen vector store",
            writer="subconscious_haiku",
            provenance="llm_inferred",
            source_ref=f"t:sub_{i}",
            confidence=0.9,
        )
        assert result == a  # dedups to same id

    # seen_count must NOT have changed (subconscious dedup doesn't boost verified)
    final_seen = next(r for r in v2_store.scan_v2() if r["id"] == a)["seen_count"]
    assert final_seen == initial_seen, (
        f"Subconscious dedup boosted verified seen_count from {initial_seen} to {final_seen}"
    )


def test_user_dedup_does_boost_seen_count(v2_store):
    """User-writer dedup is legitimate and should bump seen_count normally."""
    from memory.write_gate import write_memory
    a = write_memory(
        store=v2_store, content="user dedup target",
        writer="user", provenance="user_stated",
        source_ref="t:1", confidence=1.0,
    )
    write_memory(
        store=v2_store, content="user dedup target",
        writer="user", provenance="user_stated",
        source_ref="t:2", confidence=1.0,
    )
    rec = next(r for r in v2_store.scan_v2() if r["id"] == a)
    assert rec["seen_count"] >= 2


def test_audit_log_records_create_op(v2_store):
    """Every write creates an audit log entry."""
    from memory.write_gate import write_memory
    rec_id = write_memory(
        store=v2_store, content="audit test",
        writer="user", provenance="user_stated",
        source_ref="t:1", confidence=1.0,
    )
    audits = v2_store.audit_scan()
    create_audits = [a for a in audits if a["op"] == "create" and a["memory_id"] == rec_id]
    assert len(create_audits) == 1
    assert create_audits[0]["who"] == "user"


def test_audit_log_records_dedup_bump(v2_store):
    """Dedup-bump produces a 'seen_bump' audit entry."""
    from memory.write_gate import write_memory
    a = write_memory(
        store=v2_store, content="dedup test target",
        writer="user", provenance="user_stated",
        source_ref="t:1", confidence=1.0,
    )
    write_memory(
        store=v2_store, content="dedup test target",
        writer="user", provenance="user_stated",
        source_ref="t:2", confidence=1.0,
    )
    audits = v2_store.audit_scan()
    bumps = [audit for audit in audits if audit["op"] == "seen_bump" and audit["memory_id"] == a]
    assert len(bumps) >= 1
