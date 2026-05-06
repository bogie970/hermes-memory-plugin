"""Ollama /api/chat wrapper with tool calling support.

Uses non-streaming mode for tool calls (Ollama streaming has known bugs
with tool_calls being dropped or returned as a single block).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from .config import OLLAMA_HOST, OLLAMA_MODEL

log = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised when Ollama API returns an error or is unreachable."""


def ping(host: str = OLLAMA_HOST, timeout: float = 3) -> bool:
    """Check if Ollama is running."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    timeout: int = 120,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Send a chat request to Ollama. Returns the response message dict.

    Args:
        messages: List of message dicts with role/content/tool_calls/tool_name.
        tools: Optional list of tool definitions in Ollama format.
        model: Model name (e.g. "qwen2.5:7b").
        host: Ollama base URL.
        timeout: Request timeout in seconds.
        temperature: Sampling temperature.

    Returns:
        The message dict from the response: {role, content, tool_calls?}

    Raises:
        OllamaError: On connection or API errors.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if tools:
        payload["tools"] = tools

    url = f"{host.rstrip('/')}/api/chat"

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.ConnectionError:
        raise OllamaError(f"Cannot connect to Ollama at {host}")
    except requests.Timeout:
        raise OllamaError(f"Ollama request timed out after {timeout}s")

    if resp.status_code != 200:
        raise OllamaError(f"Ollama API error {resp.status_code}: {resp.text[:500]}")

    try:
        data = resp.json()
    except (ValueError, requests.exceptions.JSONDecodeError):
        raise OllamaError(f"Ollama returned invalid JSON: {resp.text[:200]}")
    msg = data.get("message", {})
    log.debug(
        "Ollama response: role=%s content_len=%d tool_calls=%s",
        msg.get("role"),
        len(msg.get("content", "")),
        len(msg.get("tool_calls", [])) if msg.get("tool_calls") else 0,
    )
    return msg


def extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from an Ollama response message.

    Returns list of dicts: [{name: str, arguments: dict}, ...]
    Normalizes the Ollama format (function.name, function.arguments)
    to a flat {name, arguments} for easier consumption.
    """
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
                log.warning("Failed to parse tool args as JSON: %s", args[:200])
                args = {}
        if name:
            calls.append({"name": name, "arguments": args})
        else:
            log.warning("Tool call missing function name: %s", tc)
    return calls


def make_tool_result_message(tool_name: str, content: str, tool_use_id: str = "") -> dict[str, Any]:
    """Create a tool result message to send back to Ollama."""
    return {
        "role": "tool",
        "name": tool_name,
        "content": content,
    }
