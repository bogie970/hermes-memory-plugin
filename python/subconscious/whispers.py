"""Whisper messages — ephemeral one-shot messages from worker to Claude Code.

Whispers are short, time-sensitive observations that appear once in Claude's
context on the next user prompt, then are consumed (deleted). They use the
same atomic write pattern as BlockStore.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path

from filelock import FileLock, Timeout as FileLockTimeout

log = logging.getLogger(__name__)

WHISPERS_FILENAME = "whispers.json"
MAX_WHISPER_LEN = 500
MAX_WHISPERS = 5


def _whispers_path(letta_dir: Path) -> Path:
    return letta_dir / WHISPERS_FILENAME


def _lock_path(letta_dir: Path) -> Path:
    return letta_dir / "whispers.json.lock"


def write_whisper(letta_dir: Path, text: str) -> None:
    """Append a whisper to the queue. Atomic write with file lock."""
    from .blocks import _sanitize_content
    text = _sanitize_content(text)

    wp = _whispers_path(letta_dir)
    lp = _lock_path(letta_dir)

    def _do() -> None:
        existing: list[dict] = []
        if wp.exists():
            try:
                existing = json.loads(wp.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except (json.JSONDecodeError, OSError):
                existing = []

        ts = int(time.time())
        h = hashlib.sha256(f"{ts}:{text}".encode()).hexdigest()[:4]

        existing.append({
            "id": f"w_{ts}_{h}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "text": text[:MAX_WHISPER_LEN],
            "priority": "normal",
        })

        if len(existing) > MAX_WHISPERS:
            existing = existing[-MAX_WHISPERS:]

        wp.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(wp.parent), prefix=".whispers_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            os.replace(tmp, str(wp))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        log.info("Wrote whisper (id=w_%d_%s, %d chars)", ts, h, len(text))

    try:
        with FileLock(str(lp), timeout=5):
            _do()
    except FileLockTimeout:
        log.error("Could not acquire whisper lock within 5s — dropping whisper")


def read_and_consume(letta_dir: Path) -> list[dict]:
    """Read all pending whispers and delete the file. Once-only delivery."""
    wp = _whispers_path(letta_dir)
    if not wp.exists():
        return []

    lp = _lock_path(letta_dir)

    def _do() -> list[dict]:
        if not wp.exists():
            return []
        try:
            raw = json.loads(wp.read_text(encoding="utf-8"))
            if not isinstance(raw, list) or not raw:
                return []
            try:
                wp.unlink()
            except OSError:
                # Unlink failed — write empty to prevent duplicate delivery
                try:
                    wp.write_text("[]", encoding="utf-8")
                except OSError:
                    pass
            log.info("Consumed %d whisper(s)", len(raw))
            return raw
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to read whispers: %s", e)
            return []

    try:
        with FileLock(str(lp), timeout=5):
            return _do()
    except FileLockTimeout:
        log.error("Could not acquire whisper lock for consume — skipping")
        return []
