---
description: Report current context window usage by source — static files + transcript breakdown
---

Show how much of the context window is currently consumed.

Run the measurement script, then add estimates for sources the script can't see (system prompt, tool schemas, retrieved-memory injections):

!python -m memory.context_budget_cli --window 200000

After the script output, append a brief note with your own estimates for:
- System prompt + safety rules (~8-12k baseline)
- Tool schemas (varies; ~5-15k for active tool set)
- Retrieved memory injections from UserPromptSubmit hook (visible in recent turns)
- Any skills loaded this turn

If running on a 1M context model, re-run with `--window 1000000`.
