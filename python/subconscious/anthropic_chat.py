"""Anthropic API chat wrapper — drop-in replacement for ollama_chat.

Uses the same interface (chat, extract_tool_calls, make_tool_result_message)
so loop.py can swap providers with a single import change.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


class AnthropicError(Exception):
    """Raised when the Anthropic API returns an error."""


# Alias so loop.py can catch the same exception name pattern
OllamaError = AnthropicError


_client = None

def _get_client():
    global _client
    if _client is None:
        try:
            import anthropic
        except ImportError:
            raise AnthropicError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise AnthropicError("ANTHROPIC_API_KEY not set")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _convert_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Ollama/OpenAI tool format to Anthropic format."""
    converted = []
    for tool in tools:
        func = tool.get("function", {})
        converted.append({
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })
    return converted


def _convert_messages_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert Ollama-style messages to Anthropic format.

    Returns (system_prompt, messages_list).
    Groups consecutive tool results into a single user message.
    """
    system = ""
    anthropic_msgs: list[dict[str, Any]] = []

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "system":
            system = msg.get("content", "")
            i += 1

        elif role == "user":
            anthropic_msgs.append({
                "role": "user",
                "content": msg.get("content", ""),
            })
            i += 1

        elif role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            text = msg.get("content", "")
            if text:
                content_blocks.append({"type": "text", "text": text})
            for tc in msg.get("_tool_use_blocks", []):
                content_blocks.append(tc)
            if content_blocks:
                anthropic_msgs.append({"role": "assistant", "content": content_blocks})
            i += 1

        elif role == "tool":
            # Collect consecutive tool results into one user message
            tool_results: list[dict[str, Any]] = []
            while i < len(messages) and messages[i].get("role") == "tool":
                m = messages[i]
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": m.get("_tool_use_id", "unknown"),
                    "content": m.get("content", ""),
                })
                i += 1
            anthropic_msgs.append({"role": "user", "content": tool_results})

        else:
            i += 1

    return system, anthropic_msgs


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str = "",
    host: str = "",
    timeout: int = 120,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Send a chat request to Anthropic. Returns an Ollama-compatible message dict."""
    if not model:
        model = MODEL
    client = _get_client()

    system_prompt, anthropic_msgs = _convert_messages_to_anthropic(messages)

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "messages": anthropic_msgs,
        "temperature": temperature,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if tools:
        kwargs["tools"] = _convert_tools_to_anthropic(tools)

    try:
        response = client.messages.create(**kwargs)
    except Exception as e:
        raise AnthropicError(f"Anthropic API error: {e}")

    text_parts: list[str] = []
    tool_use_blocks: list[dict[str, Any]] = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_use_blocks.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })

    result: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts),
        "_tool_use_blocks": tool_use_blocks,
    }

    if tool_use_blocks:
        result["tool_calls"] = [
            {
                "function": {
                    "name": b["name"],
                    "arguments": b["input"],
                },
                "_id": b["id"],
            }
            for b in tool_use_blocks
        ]

    log.debug(
        "Anthropic response: content_len=%d tool_calls=%d stop=%s",
        len(result["content"]),
        len(tool_use_blocks),
        response.stop_reason,
    )
    return result


def extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from response. Includes _id for tool result mapping."""
    raw = message.get("tool_calls", [])
    if not raw:
        return []

    calls = []
    for tc in raw:
        func = tc.get("function", {})
        name = func.get("name")
        args = func.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name:
            calls.append({
                "name": name,
                "arguments": args,
                "_id": tc.get("_id", name),
            })
    return calls


def make_tool_result_message(tool_name: str, content: str, tool_use_id: str = "") -> dict[str, Any]:
    """Create a tool result message with Anthropic tool_use_id tracking."""
    return {
        "role": "tool",
        "name": tool_name,
        "content": content,
        "_tool_use_id": tool_use_id or tool_name,
    }
