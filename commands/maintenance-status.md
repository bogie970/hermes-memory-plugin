---
description: Show recent Hermes memory maintenance runs (Sonnet/Opus jobs, L1 evictions, errors)
---

Show the last 10 maintenance runs for the Hermes memory system, including
Sonnet daily promotions, Opus weekly audits, and L1 eviction triggers,
plus a tier breakdown of the current LanceDB store.

!python -m memory.status_cli --stats --limit 10
