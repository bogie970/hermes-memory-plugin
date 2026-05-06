"""Subconscious configuration constants."""

from __future__ import annotations

import os

__all__ = [
    "OLLAMA_HOST",
    "OLLAMA_MODEL",
    "MAX_LOOP_ITERATIONS",
    "LOOP_TIMEOUT_SECONDS",
    "TRANSCRIPT_MAX_CHARS",
    "BLOCK_LABELS",
    "BLOCK_CHAR_LIMITS",
    "TOTAL_CHAR_LIMIT",
]

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

MAX_LOOP_ITERATIONS = 8
LOOP_TIMEOUT_SECONDS = 120
TRANSCRIPT_MAX_CHARS = 6000

BLOCK_LABELS = [
    "user_preferences",
    "project_context",
    "session_patterns",
    "pending_items",
    "guidance",
]

BLOCK_CHAR_LIMITS: dict[str, int] = {
    "user_preferences": 3000,
    "project_context": 3000,
    "session_patterns": 3000,
    "pending_items": 3000,
    "guidance": 3000,
}

TOTAL_CHAR_LIMIT = 15000
