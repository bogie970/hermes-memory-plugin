"""Weekly cleanup pass — covers all unbounded-growth concerns.

Run via /cleanup slash command or as part of the weekly Opus audit.
Idempotent and conservative: never destroys verified memory; never
deletes audit history newer than 180 days; preserves user-stated
patterns and procedures.

Operations:
  1. compact LanceDB versions (cleanup_old_versions + compact_files)
  2. tombstone stale low-trust candidates (seen=1, age>30d)
  3. hard-delete old tombstoned (age>90d) — audit log is the receipt
  4. hard-delete l1_evict candidates with access_count=0, age>14d
  5. sweep .consumed-*.md markers older than 7 days
  6. sweep session-*.json files older than 30 days
  7. rotate maintenance_log/runs.jsonl when >10MB
  8. record a "cleanup_weekly" entry in maintenance_log
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from memory.maintenance_log import _log_path, log_run
from memory.store import AUDIT_TABLE_NAME, MemoryStore


# ---- Tunables ----

CANDIDATE_STALE_DAYS = 30
TOMBSTONE_PURGE_DAYS = 90
L1_EVICT_PURGE_DAYS = 14
MARKER_SWEEP_DAYS = 7
SESSION_FILE_SWEEP_DAYS = 30
LANCE_VERSION_RETAIN_DAYS = 7
LOG_ROTATE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB


def _iso_days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _sanitize(value: str) -> str:
    return str(value).replace("'", "''")


# ---- LanceDB compaction ----

def cleanup_old_lance_versions(store: MemoryStore, older_than_days: int = LANCE_VERSION_RETAIN_DAYS) -> dict[str, Any]:
    """Drop manifest versions older than the cutoff.

    Returns per-table info (best-effort; LanceDB API surface varies).
    """
    info: dict[str, Any] = {}
    cutoff = timedelta(days=older_than_days)

    for table_name in store.list_tables():
        try:
            t = store.db.open_table(table_name)
            if hasattr(t, "cleanup_old_versions"):
                result = t.cleanup_old_versions(older_than=cutoff)
                info[table_name] = str(result) if result else "ok"
            else:
                info[table_name] = "skipped (no cleanup_old_versions API)"
        except Exception as e:
            info[table_name] = f"error: {e!r}"
    return info


def compact_lance(store: MemoryStore) -> dict[str, Any]:
    """Compact all LanceDB tables in this database."""
    info: dict[str, Any] = {}
    for table_name in store.list_tables():
        try:
            t = store.db.open_table(table_name)
            if hasattr(t, "compact_files"):
                with store.lock:
                    result = t.compact_files()
                info[table_name] = str(result) if result else "ok"
            else:
                info[table_name] = "skipped (no compact_files API)"
        except Exception as e:
            info[table_name] = f"error: {e!r}"
    return info


# ---- Tier maintenance ----

def tombstone_stale_candidates(store: MemoryStore, age_days: int = CANDIDATE_STALE_DAYS) -> int:
    """Move untrusted, never-recurring candidates to tombstoned.

    Criteria: tier='candidate' AND seen_count <= 1 AND created_at < cutoff.
    Does NOT tombstone candidates with seen_count > 1 (those are recurring and
    deserve more time to be promoted by Sonnet).
    """
    rows = store.scan_v2_lean()
    cutoff_iso = _iso_days_ago(age_days)
    targets: list[str] = []

    for row in rows:
        if row.get("tier") != "candidate":
            continue
        if int(row.get("seen_count") or 1) > 1:
            continue
        created_at = row.get("created_at") or ""
        if created_at < cutoff_iso:
            targets.append(row["id"])

    now_iso = datetime.now(timezone.utc).isoformat()
    with store.lock:
        for memory_id in targets:
            store.table.update(
                where=f"id = '{_sanitize(memory_id)}'",
                values={"tier": "tombstoned"},
            )
            # Audit
            from memory.write_gate import _audit
            _audit(
                store, memory_id=memory_id, op="cleanup_tombstone",
                who="cleanup_weekly",
                why=f"stale candidate (age>{age_days}d, seen<=1)",
                before="candidate", after="tombstoned",
            )
    return len(targets)


def purge_old_tombstoned(store: MemoryStore, age_days: int = TOMBSTONE_PURGE_DAYS) -> int:
    """Hard-delete tombstoned rows older than threshold.

    The audit log retains the create+tombstone ops as the historical record.
    """
    rows = store.scan_v2_lean()
    cutoff_iso = _iso_days_ago(age_days)
    targets: list[str] = []

    for row in rows:
        if row.get("tier") != "tombstoned":
            continue
        created_at = row.get("created_at") or ""
        if created_at < cutoff_iso:
            targets.append(row["id"])

    with store.lock:
        for memory_id in targets:
            store.table.delete(f"id = '{_sanitize(memory_id)}'")
    return len(targets)


def purge_unused_l1_evict_candidates(store: MemoryStore, age_days: int = L1_EVICT_PURGE_DAYS) -> int:
    """Hard-delete l1_evict candidates that never got accessed.

    L1-eviction chunks accumulate fast (5-20 per evict event). If they were
    never retrieved (access_count=0) AND haven't been promoted (still
    candidate) after the cutoff, they're noise.
    """
    rows = store.scan_v2_lean()
    cutoff_iso = _iso_days_ago(age_days)
    targets: list[str] = []

    for row in rows:
        if row.get("tier") != "candidate":
            continue
        if "l1_evict" not in (row.get("category") or ""):
            continue
        if int(row.get("access_count") or 0) > 0:
            continue
        created_at = row.get("created_at") or ""
        if created_at < cutoff_iso:
            targets.append(row["id"])

    with store.lock:
        for memory_id in targets:
            store.table.delete(f"id = '{_sanitize(memory_id)}'")
    return len(targets)


# ---- Filesystem sweeps ----

def sweep_consumed_markers(marker_dir: pathlib.Path, age_days: int = MARKER_SWEEP_DAYS) -> int:
    """Delete .consumed-*.md marker files older than threshold."""
    if not marker_dir.exists():
        return 0
    cutoff = time.time() - (age_days * 86400)
    n = 0
    for entry in marker_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if not name.startswith("l1_evicted_") or ".consumed-" not in name:
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink()
                n += 1
        except OSError:
            continue
    return n


def sweep_session_files(session_dir: pathlib.Path, age_days: int = SESSION_FILE_SWEEP_DAYS) -> int:
    """Delete session-*.json files older than threshold."""
    if not session_dir.exists():
        return 0
    cutoff = time.time() - (age_days * 86400)
    n = 0
    for entry in session_dir.iterdir():
        if not entry.is_file():
            continue
        if not (entry.name.startswith("session-") and entry.name.endswith(".json")):
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink()
                n += 1
        except OSError:
            continue
    return n


def rotate_maintenance_log(max_bytes: int = LOG_ROTATE_MAX_BYTES) -> bool:
    """Rotate runs.jsonl if larger than max_bytes. Returns True if rotated."""
    log = _log_path()
    if not log.exists():
        return False
    if log.stat().st_size <= max_bytes:
        return False

    # Move current to .1, truncate
    rotated = log.parent / f"{log.name}.1"
    try:
        if rotated.exists():
            rotated.unlink()
        log.rename(rotated)
        log.touch()
    except OSError:
        return False
    return True


# ---- Top-level orchestrator ----

def _all_marker_dirs() -> list[pathlib.Path]:
    """Find all l1_markers/ dirs under ~/.claude/projects/*/."""
    projects = pathlib.Path.home() / ".claude" / "projects"
    if not projects.exists():
        return []
    return [p / "l1_markers" for p in projects.iterdir()
            if p.is_dir() and (p / "l1_markers").exists()]


def _all_session_dirs() -> list[pathlib.Path]:
    """Find all .letta/claude/ dirs under known locations.

    The session-*.json files live under <cwd>/.letta/claude/. Scan cwd
    plus HERMES_DATA_DIR's parent.
    """
    candidates = [pathlib.Path.cwd() / ".letta" / "claude"]
    home = os.environ.get("HERMES_DATA_DIR") or str(pathlib.Path.home() / ".hermes")
    candidates.append(pathlib.Path(home).parent / ".letta" / "claude")
    return [p for p in candidates if p.exists()]


@dataclass
class CleanupResult:
    tombstoned_stale_candidates: int = 0
    purged_old_tombstoned: int = 0
    purged_unused_l1_evict: int = 0
    swept_markers: int = 0
    swept_session_files: int = 0
    rotated_maintenance_log: bool = False
    compact_info: dict[str, Any] | None = None
    cleanup_versions_info: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tombstoned_stale_candidates": self.tombstoned_stale_candidates,
            "purged_old_tombstoned": self.purged_old_tombstoned,
            "purged_unused_l1_evict": self.purged_unused_l1_evict,
            "swept_markers": self.swept_markers,
            "swept_session_files": self.swept_session_files,
            "rotated_maintenance_log": self.rotated_maintenance_log,
            "compact_info": self.compact_info,
            "cleanup_versions_info": self.cleanup_versions_info,
            "error": self.error,
        }


def run_weekly_cleanup(store: MemoryStore) -> dict[str, Any]:
    """Run the full weekly cleanup pass. Returns a summary dict.

    Conservative ordering: tier maintenance first (so compaction picks up the
    deletions in the same pass), then filesystem sweeps, then log rotation,
    then Lance version cleanup.
    """
    t0 = time.monotonic()
    result = CleanupResult()

    try:
        # Tier maintenance (writes to store)
        result.tombstoned_stale_candidates = tombstone_stale_candidates(store)
        result.purged_old_tombstoned = purge_old_tombstoned(store)
        result.purged_unused_l1_evict = purge_unused_l1_evict_candidates(store)

        # Filesystem sweeps
        for marker_dir in _all_marker_dirs():
            result.swept_markers += sweep_consumed_markers(marker_dir)
        for session_dir in _all_session_dirs():
            result.swept_session_files += sweep_session_files(session_dir)

        # Log rotation
        result.rotated_maintenance_log = rotate_maintenance_log()

        # Lance cleanup (last so tier deletions get compacted away)
        result.cleanup_versions_info = cleanup_old_lance_versions(store)
        result.compact_info = compact_lance(store)

    except Exception as e:
        result.error = str(e)[:300]

    duration_s = time.monotonic() - t0
    summary = result.to_dict()
    log_run(
        job="cleanup_weekly",
        duration_s=duration_s,
        summary=summary,
        error=result.error,
    )
    return summary


# ---- CLI entrypoint ----

def main() -> int:
    """python -m memory.cleanup — run the weekly cleanup pass."""
    import json
    from memory.embeddings import EmbeddingService

    store = MemoryStore(embedder=EmbeddingService())
    summary = run_weekly_cleanup(store)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if not summary.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
