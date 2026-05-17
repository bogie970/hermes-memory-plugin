"""Thin bridge: spawned by TS hooks, imports and runs the Python worker from hermes.

The TS hook sets HERMES_ROOT env var before spawning this script.
This script adds HERMES_ROOT to sys.path and delegates to the actual runner.

Concurrency cap (2026-05-17): on startup, count live local_worker.py
processes. If > MAX_CONCURRENT_WORKERS, exit silently. Prevents the
runaway-spawn pattern that produced 20+ stuck workers when the memory
store lock was contended. Stop hook fires every turn — missing one is
cheap; piling up is dangerous.

Usage: python local_worker.py <payload.json>
"""

import os
import sys

MAX_CONCURRENT_WORKERS = 3


def _count_live_local_workers() -> int:
    """Number of currently-running local_worker.py processes (incl. self).
    Returns 1 (self) on any error — fails open, doesn't block normal operation.
    """
    try:
        import psutil  # type: ignore
        n = 0
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = p.info.get("cmdline") or []
                if any("local_worker.py" in (a or "") for a in cmdline):
                    n += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return max(n, 1)
    except ImportError:
        pass

    # psutil unavailable — try platform fallback
    if os.name == "nt":
        try:
            import subprocess
            out = subprocess.run(
                ["wmic", "process", "where",
                 "CommandLine like '%local_worker.py%'", "get", "ProcessId"],
                capture_output=True, text=True, timeout=5,
            )
            return max(out.stdout.count("\n") - 1, 1)
        except Exception:
            return 1
    else:
        try:
            import subprocess
            out = subprocess.run(
                ["pgrep", "-fc", "local_worker.py"],
                capture_output=True, text=True, timeout=3,
            )
            return max(int(out.stdout.strip() or 1), 1)
        except Exception:
            return 1


# Concurrency cap check — must run before heavy imports
_live = _count_live_local_workers()
if _live > MAX_CONCURRENT_WORKERS:
    # Already enough workers in flight. Next Stop hook will get through.
    sys.exit(0)


hermes_root = os.environ.get("HERMES_ROOT")
if not hermes_root:
    print("ERROR: HERMES_ROOT environment variable not set", file=sys.stderr)
    sys.exit(1)

# Insert hermes/aisys FIRST so `memory.*` resolves to the canonical hermes copy.
# Otherwise PYTHONPATH (which points to claude-subconscious/python/) wins and
# we end up importing a stale duplicate of the memory package.
sys.path.insert(0, os.path.join(hermes_root, "aisys"))
sys.path.insert(1, hermes_root)

from aisys.subconscious.runner import main  # noqa: E402

main(sys.argv[1] if len(sys.argv) > 1 else None)
