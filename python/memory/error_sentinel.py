"""Hook error sentinel.

All hooks exit 0 by design (failure must never break a user's session).
That's the right policy for liveness — but it means silent failures
are invisible. This sentinel records the most recent N errors so
/memory-stats can surface "something's broken" to the user.

Lives at: <data_dir>/maintenance/last_error.jsonl (capped to last 50)
"""

from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime, timezone

_MAX_ENTRIES = 50


def _sentinel_path() -> pathlib.Path:
    data_dir = pathlib.Path(
        os.environ.get("HERMES_DATA_DIR")
        or pathlib.Path.home() / ".hermes"
    )
    log_dir = data_dir / "maintenance"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "last_error.jsonl"


def record_error(*, source: str, error: str, context: str = "") -> None:
    """Append an error record. Keeps only the last _MAX_ENTRIES.

    source: hook name or component (e.g. "l1_watch", "promotion_cli")
    error: short error string (avoid full traceback — keep <500 chars)
    context: optional additional info (cwd, session_id, etc.)
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "error": error[:500],
        "context": context[:200],
    }
    path = _sentinel_path()
    # Append, then truncate-from-front if over cap
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        # Truncate-rotate
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > _MAX_ENTRIES:
            path.write_text("\n".join(lines[-_MAX_ENTRIES:]) + "\n",
                             encoding="utf-8")
    except OSError:
        # If we can't even write the sentinel, there's nothing we can do
        pass


def read_recent(limit: int = 5) -> list[dict]:
    """Return the most recent N error entries, newest first."""
    path = _sentinel_path()
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
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


def clear() -> None:
    """Wipe the sentinel (e.g., user acknowledges the error)."""
    path = _sentinel_path()
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
