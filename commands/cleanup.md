---
description: Run the weekly Hermes Memory cleanup pass — compact LanceDB, GC stale candidates, sweep old marker/session files
---

Runs the full weekly cleanup pass on the Hermes memory store. This is the
maintenance job that prevents unbounded growth:

1. Tombstone stale candidates (seen_count=1, age >30 days)
2. Hard-delete tombstoned older than 90 days
3. Hard-delete unused l1_evict candidates (access_count=0, age >14 days)
4. Sweep `.consumed-*.md` markers older than 7 days
5. Sweep `session-*.json` files older than 30 days
6. Rotate `runs.jsonl` if over 10 MB
7. Run LanceDB version cleanup + table compaction

Logs to `~/.hermes/maintenance/runs.jsonl` as `cleanup_weekly`.
Idempotent — safe to run repeatedly.

!python -m memory.cleanup
