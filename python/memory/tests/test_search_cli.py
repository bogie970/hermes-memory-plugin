"""UX 1+2 — search and forget CLI tests.

Search: surface L2 vector retrieval to the user as a slash command.
Forget: surgical tombstone of a specific memory (audit log preserved).

Run: pytest aisys/memory/tests/test_search_cli.py -v
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


@pytest.fixture
def seeded_store(v2_store):
    """Store with a few known memories so search has something to find."""
    from memory.write_gate import write_memory
    ids = {}
    seeds = [
        ("hermes is the AI co-pilot for Jacob's physics work", "physics"),
        ("LanceDB is the chosen vector store for hermes memory", "lancedb"),
        ("Atlas dashboard runs on streamlit at port 8501", "atlas"),
    ]
    for content, key in seeds:
        ids[key] = write_memory(
            store=v2_store, content=content,
            writer="user", provenance="user_stated",
            source_ref=f"seed:{key}", confidence=1.0,
        )
    return v2_store, ids


# ----- search() -----

def test_search_returns_matching_results(seeded_store):
    """Searching for content that matches a seed returns it as top hit."""
    store, ids = seeded_store
    from memory.search_cli import search
    results = search(store, query="LanceDB is the chosen vector store for hermes memory", k=3)
    assert len(results) >= 1
    # Top result should be the lancedb seed (identical content -> distance ~0)
    assert results[0]["id"] == ids["lancedb"]


def test_search_returns_empty_for_unrelated(seeded_store):
    """Unrelated query with high-confidence threshold returns nothing useful."""
    store, _ids = seeded_store
    from memory.search_cli import search
    results = search(store, query="totally unrelated quantum chromodynamics", k=3, min_confidence=0.95)
    assert results == []


def test_search_excludes_tombstoned(seeded_store):
    """Tombstoned memories are not returned even if they match."""
    store, ids = seeded_store
    # Tombstone the lancedb memory
    with store.lock:
        store.table.update(
            where=f"id = '{ids['lancedb']}'",
            values={"tier": "tombstoned"},
        )

    from memory.search_cli import search
    results = search(store, query="LanceDB is the chosen vector store", k=3)
    assert all(r["id"] != ids["lancedb"] for r in results)


def test_search_returns_fields_useful_for_user(seeded_store):
    """Each result carries id, content, tier, source_ref, distance.

    Uses identical-content query so FakeEmbedder yields a deterministic match.
    """
    store, _ids = seeded_store
    from memory.search_cli import search
    results = search(
        store,
        query="LanceDB is the chosen vector store for hermes memory",
        k=1,
    )
    assert len(results) == 1
    r = results[0]
    for field in ("id", "content", "tier", "source_ref", "distance"):
        assert field in r


# ----- forget() -----

def test_forget_tombstones_target(seeded_store):
    """forget(id) sets tier=tombstoned, audit log records it."""
    store, ids = seeded_store
    target = ids["physics"]

    from memory.search_cli import forget
    result = forget(store, memory_id=target, reason="test removal")
    assert result["tombstoned"] == 1
    rec = next(r for r in store.scan_v2() if r["id"] == target)
    assert rec["tier"] == "tombstoned"

    # Audit recorded
    audits = store.audit_scan()
    forget_audits = [a for a in audits if a["op"] == "user_forget" and a["memory_id"] == target]
    assert len(forget_audits) == 1


def test_forget_unknown_id_returns_zero(seeded_store):
    """Forgetting a nonexistent id is a graceful no-op."""
    store, _ids = seeded_store
    from memory.search_cli import forget
    result = forget(store, memory_id="does-not-exist-xyz", reason="test")
    assert result["tombstoned"] == 0


def test_forget_already_tombstoned_is_noop(seeded_store):
    """Re-forgetting an already-tombstoned record returns 0 (idempotent)."""
    store, ids = seeded_store
    target = ids["atlas"]
    from memory.search_cli import forget
    forget(store, memory_id=target, reason="first")
    result2 = forget(store, memory_id=target, reason="second")
    assert result2["tombstoned"] == 0


def test_forget_records_reason_in_audit(seeded_store):
    """Audit log captures the user-supplied reason."""
    store, ids = seeded_store
    from memory.search_cli import forget
    forget(store, memory_id=ids["lancedb"], reason="contradicts new architecture")

    audits = store.audit_scan()
    rec = next(a for a in audits
               if a["op"] == "user_forget" and a["memory_id"] == ids["lancedb"])
    assert "contradicts new architecture" in rec["why"]


# ----- format_search_output -----

def test_format_search_output_human_readable(seeded_store):
    """format_search_output produces readable text, not raw JSON."""
    store, _ids = seeded_store
    from memory.search_cli import search, format_search_output
    results = search(
        store,
        query="LanceDB is the chosen vector store for hermes memory",
        k=2,
    )
    text = format_search_output(results, query="LanceDB")
    assert "LanceDB" in text
    assert "tier=" in text


def test_format_search_output_handles_empty():
    from memory.search_cli import format_search_output
    text = format_search_output([], query="missing")
    assert "No results" in text
