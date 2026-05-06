"""Transcript parser for Claude Code JSONL transcripts.

Reads the JSONL file, extracts user/assistant exchanges, truncates to
fit within the model's context window, and formats as XML for the
subconscious LLM.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .config import TRANSCRIPT_MAX_CHARS

log = logging.getLogger(__name__)

# Patterns that look like prompt injection attempts
_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions"),
    re.compile(r"(?i)you\s+are\s+now\s+(DAN|jailbroken|unrestricted)"),
    re.compile(r"(?i)SYSTEM:\s*override"),
]

# Patterns for secrets that should never be stored
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),         # OpenAI
    re.compile(r"ghp_[a-zA-Z0-9]{36,}"),         # GitHub PAT
    re.compile(r"AKIA[A-Z0-9]{16}"),             # AWS
    re.compile(r"xox[bpras]-[a-zA-Z0-9\-]+"),   # Slack
]


def _escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "... [truncated]"


def _sanitize(text: str) -> str:
    """Strip potential prompt injection patterns and redact secrets."""
    for pat in _INJECTION_PATTERNS:
        text = pat.sub("[REDACTED-INJECTION]", text)
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED-SECRET]", text)
    return text


def parse_transcript(
    transcript_path: str,
    start_index: int = -1,
) -> list[dict[str, str]]:
    """Parse a Claude Code JSONL transcript into simplified messages.

    Args:
        transcript_path: Path to the .jsonl transcript file.
        start_index: Only process messages after this index (-1 = all).

    Returns:
        List of {role: "user"|"assistant"|"system", text: str} dicts.
    """
    path = Path(transcript_path)
    if not path.exists():
        log.warning("Transcript not found: %s", transcript_path)
        return []

    raw_messages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_messages.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    formatted = []
    for i, msg in enumerate(raw_messages):
        if i <= start_index:
            continue

        msg_type = msg.get("type", "")

        if msg_type == "summary":
            summary = msg.get("summary", "")
            if summary:
                formatted.append({"role": "system", "text": f"[Session Summary]: {summary}"})
            continue

        if msg_type in ("file-history-snapshot", "system"):
            continue

        content = msg.get("message", {}).get("content") or msg.get("content")

        if msg_type == "user":
            text = _extract_text(content)
            if text:
                formatted.append({"role": "user", "text": text})

        elif msg_type == "assistant":
            text = _extract_text(content)
            tools = _extract_tool_uses(content)
            if tools:
                for tool in tools:
                    formatted.append({
                        "role": "assistant",
                        "text": f"[Tool: {tool['name']}] {_truncate(tool['summary'], 100)}",
                    })
            if text:
                formatted.append({"role": "assistant", "text": text})

    return formatted


def _extract_text(content: str | list | None) -> str:
    """Extract plain text from message content (string or content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(block["text"])
        return "\n".join(parts) if parts else ""
    return ""


def _extract_tool_uses(content: str | list | None) -> list[dict[str, str]]:
    """Extract tool use summaries from content blocks."""
    if not isinstance(content, list):
        return []
    tools = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = block.get("name", "unknown")
            inp = block.get("input", {})
            summary = ""
            if name in ("Read", "Edit", "Write") and isinstance(inp, dict):
                summary = inp.get("file_path", "")
            elif name == "Bash" and isinstance(inp, dict):
                summary = inp.get("command", "")
            elif name in ("Glob", "Grep") and isinstance(inp, dict):
                summary = inp.get("pattern", "")
            else:
                summary = json.dumps(inp)[:100] if inp else ""
            tools.append({"name": name, "summary": summary})
    return tools


def format_as_xml(
    messages: list[dict[str, str]],
    max_chars: int = TRANSCRIPT_MAX_CHARS,
    session_id: str = "",
) -> str:
    """Format messages as XML for the subconscious LLM prompt.

    Truncates from the oldest messages first to stay within max_chars.
    Sanitizes content to prevent prompt injection.
    """
    if not messages:
        return "<transcript>(empty)</transcript>"

    # Truncate from the front (keep most recent messages)
    total = 0
    start = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        msg_len = len(messages[i]["text"]) + 50  # overhead for XML tags
        if total + msg_len > max_chars:
            break
        total += msg_len
        start = i

    kept = messages[start:]
    if not kept:
        kept = messages[-1:]  # always keep at least the last message

    parts = []
    if start > 0:
        parts.append(f"<!-- {start} earlier messages truncated -->")

    for msg in kept:
        role = msg["role"]
        if role == "assistant":
            role = "claude_code"
        text = _escape_xml(_sanitize(msg["text"]))
        parts.append(f'<message role="{role}">\n{text}\n</message>')

    header = f'<transcript session_id="{session_id}">' if session_id else "<transcript>"
    return f"{header}\n" + "\n".join(parts) + "\n</transcript>"
