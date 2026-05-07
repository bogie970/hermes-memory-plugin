"""DEEP-AUDIT 2 + 6 — weekly cleanup pass tests.

Covers garbage collection of unbounded growth: Lance versions, candidate
tier accumulation, tombstoned tier rot, marker files, session state files,
maintenance log rotation.

Run: pytest aisys/memory/tests/test_cleanup.py -v
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


@pytest.fixture
def v2_store(lance_store_v1):
    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)
    return lance_store_v1


# ----- Stale-candidate tombstoning -----

def test_cleanup_tombstones_stale_candidate(v2_store):
    """seen_count=1 candidate older than threshold -> tombstoned."""
    from memory.write_gate import write_memory
    rec_id = write_memory(
        store=v2_store,
        content="stale haiku claim",
        writer="subconscious_haiku",
        provenance="llm_inferred",
        source_ref="t:stale",
        confidence=0.4,
    )
    # Backdate created_at and last_seen_at to 31 days ago
    old_ts = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    with v2_store.lock:
        v2_store.table.update(
            where=f"id = '{rec_id}'",
            values={"created_at": old_ts, "last_seen_at": old_ts},
        )

    from memory.cleanup import tombstone_stale_candidates
    n = tombstone_stale_candidates(v2_store, age_days=30)
    assert n == 1
    rec = next(r for r in v2_store.scan_v2() if r["id"] == rec_id)
    assert rec["tier"] == "tombstoned"


def test_cleanup_keeps_recent_candidate(v2_store):
    """Recent candidate not aged enough — stays candidate."""
    from memory.write_gate import write_memory
    rec_id = write_memory(
        store=v2_store, content="fresh claim",
        writer="subconscious_haiku", provenance="llm_inferred",
        source_ref="t:fresh", confidence=0.4,
    )
    from memory.cleanup import tombstone_stale_candidates
    n = tombstone_stale_candidates(v2_store, age_days=30)
    assert n == 0
    rec = next(r for r in v2_store.scan_v2() if r["id"] == rec_id)
    assert rec["tier"] == "candidate"


def test_cleanup_keeps_recurring_candidate(v2_store):
    """Old candidate with seen_count>1 NOT tombstoned (legitimate recurrence)."""
    from memory.write_gate import write_memory
    rec_id = write_memory(
        store=v2_store, content="recurring claim",
        writer="subconscious_haiku", provenance="llm_inferred",
        source_ref="t:rec", confidence=0.5,
    )
    old_ts = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    with v2_store.lock:
        v2_store.table.update(
            where=f"id = '{rec_id}'",
            values={"created_at": old_ts, "last_seen_at": old_ts, "seen_count": 3},
        )

    from memory.cleanup import tombstone_stale_candidates
    n = tombstone_stale_candidates(v2_store, age_days=30)
    assert n == 0


# ----- Hard-delete old tombstoned -----

def test_cleanup_purges_old_tombstoned(v2_store):
    """Tombstoned older than 90 days hard-deleted."""
    from memory.write_gate import write_memory
    rec_id = write_memory(
        store=v2_store, content="zombie",
        writer="user", provenance="user_stated",
        source_ref="t:z", confidence=1.0,
    )
    old_ts = (datetime.now(timezone.utc) - timedelta(days=91)).isoformat()
    with v2_store.lock:
        v2_store.table.update(
            where=f"id = '{rec_id}'",
            values={"tier": "tombstoned", "created_at": old_ts},
        )

    from memory.cleanup import purge_old_tombstoned
    n = purge_old_tombstoned(v2_store, age_days=90)
    assert n == 1
    rows = v2_store.scan_v2()
    assert not any(r["id"] == rec_id for r in rows)


# ----- Marker file sweep -----

def test_cleanup_sweeps_old_consumed_markers(tmp_path):
    """Consumed markers older than 7 days deleted."""
    marker_dir = tmp_path / "l1_markers"
    marker_dir.mkdir()
    old = marker_dir / "l1_evicted_aaaa.consumed-12345.md"
    old.write_text("old marker")
    new = marker_dir / "l1_evicted_bbbb.consumed-67890.md"
    new.write_text("new marker")

    # Backdate the old one
    old_time = time.time() - (8 * 86400)
    import os
    os.utime(old, (old_time, old_time))

    from memory.cleanup import sweep_consumed_markers
    n = sweep_consumed_markers(marker_dir, age_days=7)
    assert n == 1
    assert not old.exists()
    assert new.exists()


def test_cleanup_keeps_active_markers(tmp_path):
    """Active (.md not .consumed-*.md) markers always preserved."""
    marker_dir = tmp_path / "l1_markers"
    marker_dir.mkdir()
    active = marker_dir / "l1_evicted_xxxx.md"
    active.write_text("active marker")
    old_time = time.time() - (30 * 86400)
    import os
    os.utime(active, (old_time, old_time))

    from memory.cleanup import sweep_consumed_markers
    n = sweep_consumed_markers(marker_dir, age_days=7)
    assert n == 0
    assert active.exists()


# ----- Session state file sweep -----

def test_cleanup_sweeps_old_session_files(tmp_path):
    """session-*.json files older than threshold deleted."""
    session_dir = tmp_path / ".letta" / "claude"
    session_dir.mkdir(parents=True)
    old = session_dir / "session-aaaa.json"
    old.write_text("{}")
    new = session_dir / "session-bbbb.json"
    new.write_text("{}")

    import os
    old_ts = time.time() - (31 * 86400)
    os.utime(old, (old_ts, old_ts))

    from memory.cleanup import sweep_session_files
    n = sweep_session_files(session_dir, age_days=30)
    assert n == 1
    assert not old.exists()
    assert new.exists()


# ----- Lance version compaction -----

def test_cleanup_runs_compact_files(v2_store):
    """compact_files() called on the table; no exception."""
    # Generate a few row updates to create versions
    for i in range(5):
        from memory.write_gate import write_memory
        write_memory(
            store=v2_store, content=f"compact test {i}",
            writer="user", provenance="user_stated",
            source_ref=f"t:{i}", confidence=1.0,
        )

    from memory.cleanup import compact_lance
    info = compact_lance(v2_store)
    assert "memories" in info  # reports per-table


def test_cleanup_runs_cleanup_old_versions(v2_store):
    """cleanup_old_versions() doesn't blow up."""
    from memory.cleanup import cleanup_old_lance_versions
    info = cleanup_old_lance_versions(v2_store, older_than_days=1)
    # We just want it not to throw; deletion count varies
    assert info is not None


# ----- Maintenance log rotation -----

def test_cleanup_rotates_large_maintenance_log(tmp_path, monkeypatch):
    """maintenance_log/runs.jsonl rotated when over size threshold."""
    monkeypatch.setenv("HERMES_DATA_DIR", str(tmp_path))
    log_path = tmp_path / "maintenance" / "runs.jsonl"
    log_path.parent.mkdir(parents=True)
    # Write past the 1KB threshold we'll test against
    with open(log_path, "w") as f:
        for _ in range(100):
            f.write(json.dumps({"job": "x", "ts": "2026-01-01"}) + "\n")
    initial_size = log_path.stat().st_size
    assert initial_size > 1000  # sanity: confirm we wrote enough

    from memory.cleanup import rotate_maintenance_log
    rotated = rotate_maintenance_log(max_bytes=1000)  # 1KB threshold
    assert rotated
    assert log_path.exists()
    assert log_path.stat().st_size < initial_size  # truncated
    assert (log_path.parent / "runs.jsonl.1").exists()


def test_cleanup_skips_small_maintenance_log(tmp_path, monkeypatch):
    """Small log not rotated."""
    monkeypatch.setenv("HERMES_DATA_DIR", str(tmp_path))
    log_path = tmp_path / "maintenance" / "runs.jsonl"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(json.dumps({"job": "x"}) + "\n")

    from memory.cleanup import rotate_maintenance_log
    assert not rotate_maintenance_log(max_bytes=100_000)


# ----- Full cleanup pass -----

def test_run_weekly_cleanup_returns_summary(v2_store):
    """Top-level run_weekly_cleanup returns a summary dict and records to log."""
    from memory.cleanup import run_weekly_cleanup
    result = run_weekly_cleanup(v2_store)
    assert isinstance(result, dict)
    assert "tombstoned_stale_candidates" in result
    assert "purged_old_tombstoned" in result
    assert "compact_info" in result
