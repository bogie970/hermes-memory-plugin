"""Maintenance run log — append-only JSONL of every Sonnet/Opus pass.

Produces a queryable record so /maintenance-status can show recent runs.
Lives at: <data_dir>/maintenance/runs.jsonl
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))


def _log_path() -> pathlib.Path:
    """Resolve <data_dir>/maintenance/runs.jsonl, ensuring the dir exists."""
    data_dir = pathlib.Path(
        os.environ.get("HERMES_DATA_DIR")
        or pathlib.Path.home() / ".hermes"
    )
    log_dir = data_dir / "maintenance"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "runs.jsonl"


def log_run(
    *,
    job: str,                       # "sonnet_daily" | "opus_weekly" | "l1_evict_manual"
    duration_s: float,
    summary: dict[str, Any],
    error: str | None = None,
) -> None:
    """Append one maintenance run record."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "job": job,
        "duration_s": round(duration_s, 3),
        "summary": summary,
        "error": error,
    }
    with open(_log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_recent(limit: int = 10) -> list[dict]:
    """Return the most recent N runs, newest first."""
    path = _log_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict] = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(out) >= limit:
            break
    return out


def format_status(runs: list[dict]) -> str:
    """Human-readable status report for /maintenance-status."""
    if not runs:
        return "No maintenance runs recorded yet."

    lines = ["# Hermes Maintenance — recent runs\n"]
    for r in runs:
        ts = r.get("ts", "?")
        job = r.get("job", "?")
        dur = r.get("duration_s", 0)
        err = r.get("error")
        summary = r.get("summary", {})
        marker = "ERR " if err else "OK  "
        lines.append(f"- {marker} {ts} {job} ({dur}s)")
        if err:
            lines.append(f"    error: {err[:200]}")
        for k, v in summary.items():
            lines.append(f"    {k}: {v}")
    return "\n".join(lines)
