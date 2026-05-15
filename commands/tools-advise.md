---
description: Run the tool lifecycle advisor — emit a whisper recommending /clear if budget is heavy and a cold tool is taking space
---

Evaluate the current loaded-tool budget. If the soft cap is breached AND there's a cold tool (score < 0.1, unused 10+ turns), or the hard cap is breached, emit an advisory whisper. Defaults to dry_run mode (logs to ~/.hermes/runtime/tool_advisor_dryrun.jsonl, no user-facing whisper).

To flip into whisper mode (actually emit), edit `~/.hermes/runtime/tool_flags.json` and set `"advisor_mode": "whisper"`.

!python -m aisys.tools.advisor_cli
