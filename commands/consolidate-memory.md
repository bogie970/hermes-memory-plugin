---
description: Run the Sonnet daily consolidation pass — promotes candidate memories that survive review
---

Trigger the Sonnet-driven daily promotion pass on the Hermes memory store.

This iterates every candidate-tier memory with `seen_count >= 2` OR
`confidence >= 0.8` and asks Sonnet for a `{promote, reject, hold}`
verdict. Promotions move tier candidate -> probationary. Rejections
stay candidate. Errors are logged but never block the pipeline.

Each run is logged to `~/.hermes/maintenance/runs.jsonl` and visible
via `/maintenance-status`.

!python -m aisys.memory.promotion_cli --model sonnet
