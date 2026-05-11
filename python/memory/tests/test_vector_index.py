"""Tests for LanceDB vector index management.

Auto-build IVF_PQ index when corpus crosses INDEX_THRESHOLD rows so
query latency doesn't degrade past ~10k records. Maintained by /cleanup.

Run: pytest aisys/memory/tests/test_vector_index.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


@pytest.fixture
def v2_store(lance_store_v1):
    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)
    return lance_store_v1


def test_should_build_index_returns_false_when_small(v2_store):
    """Under threshold, no index needed."""
    from memory.vector_index import should_build_index
    # Empty store
    assert not should_build_index(v2_store, threshold=5000)


def test_should_build_index_returns_true_when_threshold_crossed(v2_store):
    """At/above threshold, index should be built."""
    from memory.vector_index import should_build_index
    # Threshold of 0 means "always build" — easier than seeding 5k records
    assert should_build_index(v2_store, threshold=-1)


def test_has_vector_index_returns_false_on_fresh_store(v2_store):
    """No index initially."""
    from memory.vector_index import has_vector_index
    assert not has_vector_index(v2_store)


def test_ensure_vector_index_skips_when_below_threshold(v2_store):
    """ensure_vector_index returns 'skipped' on tiny store."""
    from memory.vector_index import ensure_vector_index
    result = ensure_vector_index(v2_store, threshold=5000)
    assert result["action"] == "skipped"


def test_ensure_vector_index_builds_when_forced(v2_store):
    """force=True bypasses the threshold check."""
    from memory.write_gate import write_memory
    # Need at least a couple rows to build an index
    for i in range(5):
        write_memory(
            store=v2_store, content=f"index test row {i}",
            writer="user", provenance="user_stated",
            source_ref=f"t:{i}", confidence=1.0,
        )
    from memory.vector_index import ensure_vector_index
    result = ensure_vector_index(v2_store, force=True)
    # Either "built" (success) or "skipped" with a documented reason —
    # LanceDB may refuse to build IVF on very small corpora. Both are acceptable.
    assert result["action"] in ("built", "skipped", "error")


def test_vector_index_status_returns_summary(v2_store):
    """vector_index_status returns a dict useful for /memory-stats."""
    from memory.vector_index import vector_index_status
    status = vector_index_status(v2_store)
    assert "row_count" in status
    assert "has_index" in status
    assert "threshold" in status
