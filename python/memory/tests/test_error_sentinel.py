"""Tests for the hook error sentinel."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_DATA_DIR", str(tmp_path))
    return tmp_path


def test_record_error_creates_sentinel(isolated):
    from memory.error_sentinel import record_error, _sentinel_path
    record_error(source="l1_watch", error="boom", context="cwd=foo")
    assert _sentinel_path().exists()


def test_read_recent_returns_newest_first(isolated):
    from memory.error_sentinel import record_error, read_recent
    for i in range(3):
        record_error(source=f"src_{i}", error=f"err_{i}")
    recent = read_recent(limit=2)
    assert len(recent) == 2
    assert recent[0]["source"] == "src_2"
    assert recent[1]["source"] == "src_1"


def test_read_recent_empty_when_no_sentinel(isolated):
    from memory.error_sentinel import read_recent
    assert read_recent() == []


def test_clear_removes_sentinel(isolated):
    from memory.error_sentinel import record_error, clear, _sentinel_path
    record_error(source="x", error="y")
    assert _sentinel_path().exists()
    clear()
    assert not _sentinel_path().exists()


def test_record_caps_entry_count(isolated):
    """Cap at 50 entries — old ones drop off the front."""
    from memory.error_sentinel import record_error, read_recent
    for i in range(60):
        record_error(source=f"src_{i}", error=f"err_{i}")
    all_recent = read_recent(limit=100)
    assert len(all_recent) == 50
    # Newest is src_59
    assert all_recent[0]["source"] == "src_59"
    # Oldest still around is src_10 (60 - 50 = 10 dropped)
    assert all_recent[-1]["source"] == "src_10"


def test_error_truncated_to_500_chars(isolated):
    from memory.error_sentinel import record_error, read_recent
    huge = "x" * 1000
    record_error(source="t", error=huge)
    recent = read_recent()
    assert len(recent[0]["error"]) == 500
