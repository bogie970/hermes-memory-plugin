"""BlockStore — thread-safe read/write for local memory blocks JSON file.

Writes use FileLock + atomic rename (NTFS rename is atomic).
Reads are lock-free for hook performance.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from filelock import FileLock, Timeout as FileLockTimeout
except ImportError:
    raise ImportError(
        "filelock is required for BlockStore. Install it: pip install filelock"
    )

import re

from .config import BLOCK_CHAR_LIMITS, BLOCK_LABELS, TOTAL_CHAR_LIMIT

log = logging.getLogger(__name__)

# Patterns that should never appear in block content
_DANGEROUS_PATTERNS = [
    re.compile(r"</?system\b", re.IGNORECASE),
    re.compile(r"</?instruction\b", re.IGNORECASE),
    re.compile(r"</?(system-reminder|user-prompt)\b", re.IGNORECASE),
    re.compile(r"</?tool_use\b", re.IGNORECASE),
    re.compile(r"</?tool_result\b", re.IGNORECASE),
    re.compile(r"</?function_calls?\b", re.IGNORECASE),
    re.compile(r"</?invoke\b", re.IGNORECASE),
    re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{36,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
]


def _sanitize_content(text: str) -> str:
    """Strip dangerous patterns from block content before writing."""
    for pat in _DANGEROUS_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


class BlockStore:
    """Manages local_blocks.json with atomic writes and lock-free reads."""

    def __init__(self, blocks_path: str | Path) -> None:
        self.path = Path(blocks_path)
        self._lock_path = self.path.with_suffix(".json.lock")
        self._lock = FileLock(str(self._lock_path), timeout=5)
        self._cleanup_stale_temps()

    def read_all(self) -> dict[str, dict[str, Any]]:
        """Lock-free read of all blocks. Returns {label: block_dict}."""
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return raw.get("blocks", {})
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to read blocks: %s", e)
            return {}

    def read_block(self, label: str) -> str:
        """Read a single block's value."""
        blocks = self.read_all()
        block = blocks.get(label)
        return (block.get("value") or "") if block else ""

    def _write_all(self, blocks: dict[str, dict[str, Any]]) -> None:
        """Atomic write: write to temp file, then rename."""
        data = {"version": 1, "blocks": blocks}
        self.path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=".blocks_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, str(self.path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _with_lock(self, fn: Any) -> Any:
        try:
            with self._lock:
                return fn()
        except FileLockTimeout:
            log.error("Could not acquire block lock within 5s — skipping write")
            return False

    def replace(self, label: str, old_str: str, new_str: str) -> bool:
        """Replace exact substring within a block. Returns True if found."""
        def _do() -> bool:
            blocks = self.read_all()
            block = blocks.get(label)
            if not block:
                log.warning("Block %r not found", label)
                return False
            value = block.get("value") or ""
            if old_str not in value:
                log.warning("old_str not found in block %r", label)
                return False
            new_value = _sanitize_content(value.replace(old_str, new_str, 1))
            limit = BLOCK_CHAR_LIMITS.get(label, 3000)
            if len(new_value) > limit:
                log.warning("Block %r would exceed limit (%d > %d)", label, len(new_value), limit)
                return False
            block["value"] = new_value
            block["updated_at"] = _now()
            self._write_all(blocks)
            return True
        return self._with_lock(_do)

    def insert(self, label: str, new_str: str, insert_line: int = -1) -> bool:
        """Insert text at a line position (-1 = append)."""
        def _do() -> bool:
            blocks = self.read_all()
            block = blocks.get(label)
            if not block:
                log.warning("Block %r not found", label)
                return False
            value = block.get("value") or ""
            lines = value.split("\n") if value else []
            if insert_line < 0 or insert_line >= len(lines):
                lines.append(new_str)
            else:
                lines.insert(insert_line, new_str)
            new_value = _sanitize_content("\n".join(lines))
            limit = BLOCK_CHAR_LIMITS.get(label, 3000)
            if len(new_value) > limit:
                log.warning("Block %r would exceed limit (%d > %d)", label, len(new_value), limit)
                return False
            block["value"] = new_value
            block["updated_at"] = _now()
            self._write_all(blocks)
            return True
        return self._with_lock(_do)

    def rethink(self, label: str, new_memory: str) -> bool:
        """Full rewrite of a block's content."""
        def _do() -> bool:
            blocks = self.read_all()
            block = blocks.get(label)
            if not block:
                log.warning("Block %r not found", label)
                return False
            sanitized = _sanitize_content(new_memory)
            limit = BLOCK_CHAR_LIMITS.get(label, 3000)
            if len(sanitized) > limit:
                log.warning("Block %r would exceed limit (%d > %d)", label, len(sanitized), limit)
                return False
            block["value"] = sanitized
            block["updated_at"] = _now()
            self._write_all(blocks)
            return True
        return self._with_lock(_do)

    def total_chars(self) -> int:
        blocks = self.read_all()
        return sum(len(b.get("value") or "") for b in blocks.values())

    def summary(self) -> dict[str, int]:
        """Returns {label: char_count} for all blocks."""
        blocks = self.read_all()
        return {label: len(b.get("value") or "") for label, b in blocks.items()}


    def _cleanup_stale_temps(self) -> None:
        """Remove .blocks_*.tmp files older than 60 seconds."""
        try:
            if not self.path.parent.exists():
                return
            now = time.time()
            for f in self.path.parent.glob(".blocks_*.tmp"):
                try:
                    if now - f.stat().st_mtime > 60:
                        f.unlink()
                except OSError:
                    pass
        except OSError:
            pass


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
