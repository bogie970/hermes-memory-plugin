---
description: Manually trigger a full Hermes maintenance cycle — dry-run first, then promotion
---

Run a full memory maintenance cycle on demand. Useful when you want
the cleanup to happen NOW instead of waiting for the scheduled job.

This first reports what's eligible for review, then runs the Sonnet
promotion pass. Both phases are logged.

# Step 1: dry-run (count eligible candidates)
!python -m aisys.memory.promotion_cli --dry-run

# Step 2: real promotion pass
!python -m aisys.memory.promotion_cli --model sonnet

# Step 3: show updated status
!python -m aisys.memory.status_cli --stats --limit 5
