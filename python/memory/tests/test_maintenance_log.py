"""Tests for maintenance log + status formatting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


@pytest.fixture
def isolated_log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_DATA_DIR", str(tmp_path))
    return tmp_path


def test_log_run_appends_jsonl(isolated_log_dir):
    from memory.maintenance_log import log_run, _log_path
    log_run(job="sonnet_daily", duration_s=1.5,
            summary={"processed": 3, "promoted": 2})
    path = _log_path()
    assert path.exists()
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["job"] == "sonnet_daily"
    assert entry["duration_s"] == 1.5
    assert entry["summary"]["processed"] == 3
    assert entry["error"] is None


def test_log_run_records_error(isolated_log_dir):
    from memory.maintenance_log import log_run, read_recent
    log_run(job="sonnet_daily", duration_s=0.1,
            summary={}, error="Sonnet API timeout")
    runs = read_recent()
    assert len(runs) == 1
    assert runs[0]["error"] == "Sonnet API timeout"


def test_read_recent_returns_newest_first(isolated_log_dir):
    from memory.maintenance_log import log_run, read_recent
    for i in range(5):
        log_run(job=f"job_{i}", duration_s=float(i), summary={"i": i})
    runs = read_recent(limit=3)
    assert len(runs) == 3
    # Newest first: job_4, job_3, job_2
    assert runs[0]["job"] == "job_4"
    assert runs[1]["job"] == "job_3"
    assert runs[2]["job"] == "job_2"


def test_read_recent_returns_empty_when_no_log(isolated_log_dir):
    from memory.maintenance_log import read_recent
    assert read_recent() == []


def test_format_status_handles_empty_list():
    from memory.maintenance_log import format_status
    assert "No maintenance runs" in format_status([])


def test_format_status_includes_run_info():
    from memory.maintenance_log import format_status
    runs = [{
        "ts": "2026-05-06T14:00:00Z",
        "job": "sonnet_daily",
        "duration_s": 2.34,
        "summary": {"promoted": 5, "rejected": 1},
        "error": None,
    }]
    out = format_status(runs)
    assert "sonnet_daily" in out
    assert "OK" in out
    assert "promoted" in out


def test_format_status_marks_errors():
    from memory.maintenance_log import format_status
    runs = [{
        "ts": "2026-05-06T14:00:00Z",
        "job": "sonnet_daily",
        "duration_s": 0.1,
        "summary": {},
        "error": "API key missing",
    }]
    out = format_status(runs)
    assert "ERR" in out
    assert "API key missing" in out
