"""
/context-budget — measure current context window consumption.

Hybrid: script measures dynamic sources (transcript + tool results),
caller (assistant) layers in static sources it can see (system prompt, CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


CHARS_PER_TOKEN = 3.5  # rough English heuristic


def find_latest_jsonl(project_dir: Path) -> Path | None:
    candidates = sorted(
        project_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def extract_text(content) -> str:
    """Flatten message content (string or list of blocks) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                out.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                # input as JSON string
                out.append(json.dumps(block.get("input", {})))
            elif block.get("type") == "tool_result":
                c = block.get("content", "")
                if isinstance(c, list):
                    for sub in c:
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            out.append(sub.get("text", ""))
                else:
                    out.append(str(c))
            elif block.get("type") == "thinking":
                out.append(block.get("thinking", ""))
        return "\n".join(out)
    return str(content)


def tokens(s: str) -> int:
    return int(len(s) / CHARS_PER_TOKEN)


def find_last_compact_line(jsonl_path: Path) -> int:
    """Return line index of the most recent compact_boundary event, or 0 if none."""
    last = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "system" and d.get("subtype") == "compact_boundary":
                last = i
    return last


def measure_transcript(jsonl_path: Path, start_line: int = 0) -> dict:
    """Return token counts grouped by category, from start_line onward."""
    buckets = {
        "compact_summary": 0,
        "user_messages": 0,
        "assistant_text": 0,
        "assistant_tool_use": 0,
        "tool_results": 0,
        "system_reminders": 0,
        "attachments": 0,
    }
    counts = {k: 0 for k in buckets}

    with jsonl_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = d.get("type")
            msg = d.get("message", {}) or {}
            role = msg.get("role")
            content = msg.get("content", d.get("content", ""))

            if etype == "user":
                text = extract_text(content)
                if d.get("isCompactSummary"):
                    buckets["compact_summary"] += tokens(text)
                    counts["compact_summary"] += 1
                # tool_result blocks come in user messages
                elif isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content
                ):
                    buckets["tool_results"] += tokens(text)
                    counts["tool_results"] += 1
                else:
                    buckets["user_messages"] += tokens(text)
                    counts["user_messages"] += 1
            elif etype == "assistant":
                text = extract_text(content)
                if isinstance(content, list):
                    text_parts = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
                    tool_parts = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
                    text_str = extract_text(text_parts)
                    tool_str = extract_text(tool_parts)
                    buckets["assistant_text"] += tokens(text_str)
                    buckets["assistant_tool_use"] += tokens(tool_str)
                    if text_parts:
                        counts["assistant_text"] += 1
                    if tool_parts:
                        counts["assistant_tool_use"] += 1
                else:
                    buckets["assistant_text"] += tokens(text)
                    counts["assistant_text"] += 1
            elif etype == "attachment":
                text = json.dumps(d)
                buckets["attachments"] += tokens(text)
                counts["attachments"] += 1
            elif etype == "system":
                text = extract_text(content) if content else json.dumps(d)
                buckets["system_reminders"] += tokens(text)
                counts["system_reminders"] += 1

    return {"tokens": buckets, "counts": counts}


def measure_static_sources() -> dict:
    """Measure files the assistant always has loaded."""
    home = Path(os.path.expanduser("~"))
    hermes = Path("C:/Users/jbogi/claude-nodes/hermes")

    sources = {
        "CLAUDE.md (project)": hermes / "CLAUDE.md",
        "MEMORY.md (auto-memory)": home / ".claude/projects/C--Users-jbogi-claude-nodes-hermes/memory/MEMORY.md",
        "Pattern blocks (.letta)": hermes / ".letta/claude/local_blocks.json",
    }
    out = {}
    for label, path in sources.items():
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            out[label] = {"tokens": tokens(text), "bytes": len(text)}
        else:
            out[label] = {"tokens": 0, "bytes": 0, "missing": True}
    return out


def fmt_row(label: str, tok: int, total: int, width: int = 38) -> str:
    pct = (tok / total * 100) if total else 0
    bar_width = 20
    fill = int(pct / 100 * bar_width)
    bar = "#" * fill + "." * (bar_width - fill)
    return f"  {label:<{width}} {tok:>7,} tok  [{bar}] {pct:5.1f}%"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=200_000,
                    help="Context window size in tokens (default 200k; use 1000000 for 1M models)")
    ap.add_argument("--project-dir", default=None)
    args = ap.parse_args(argv)

    project_dir = Path(args.project_dir) if args.project_dir else Path(
        os.path.expanduser("~/.claude/projects/C--Users-jbogi-claude-nodes-hermes")
    )

    jsonl = find_latest_jsonl(project_dir)
    if not jsonl:
        print(f"No .jsonl found in {project_dir}", file=sys.stderr)
        return 1

    compact_line = find_last_compact_line(jsonl)
    # Start one line after the boundary event (summary itself is line+1)
    start = compact_line + 1 if compact_line else 0

    print(f"Session: {jsonl.name}")
    print(f"Window:  {args.window:,} tokens")
    if compact_line:
        print(f"Last compaction at line {compact_line}; measuring from line {start}")
    else:
        print("No compaction detected; measuring full transcript")
    print()

    transcript = measure_transcript(jsonl, start_line=start)
    static = measure_static_sources()

    static_total = sum(v["tokens"] for v in static.values())
    transcript_total = sum(transcript["tokens"].values())
    grand_total = static_total + transcript_total

    print("STATIC (every turn):")
    for label, info in static.items():
        missing = " (MISSING)" if info.get("missing") else ""
        print(fmt_row(label + missing, info["tokens"], args.window))
    print(fmt_row("  subtotal", static_total, args.window))
    print()

    print("TRANSCRIPT (post-compaction, currently in window):")
    order = ["compact_summary", "user_messages", "assistant_text", "assistant_tool_use",
             "tool_results", "system_reminders", "attachments"]
    for k in order:
        tok = transcript["tokens"][k]
        cnt = transcript["counts"][k]
        label = f"{k} ({cnt})"
        print(fmt_row(label, tok, args.window))
    print(fmt_row("  subtotal", transcript_total, args.window))
    print()

    pct = (grand_total / args.window * 100) if args.window else 0
    print(f"TOTAL (measured): {grand_total:,} tok / {args.window:,} = {pct:.1f}%")
    print()
    print("Note: does NOT include system prompt, tool schemas, retrieved-memory")
    print("injections, or skill loads. Assistant adds those from what it sees.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
