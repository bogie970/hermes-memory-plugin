---
description: Show the current Hermes memory store stats — record counts by tier, audit log size
---

Report the current state of the Hermes memory store:
- Total record count
- Per-tier breakdown (candidate / probationary / verified / tombstoned)
- Audit log size

!python -m aisys.memory.status_cli --stats --limit 3
