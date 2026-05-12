"""CLI wrapper for memory.l1_manager.evict — invoked by TS hooks.

Usage:
    python -m aisys.memory.l1_manager_cli \
        --transcript /path/to/session.jsonl \
        --marker-dir /path/to/markers \
        --session-id abc123 \
        [--evict-fraction 0.5] \
        [--pin-recent 20] \
        [--timeout 60]

Always exits 0 on expected skip paths (transcript missing, below threshold)
so the calling hook never surfaces a false-error to the user.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def load_transcript_jsonl(path: str) -> list[dict]:
    """Load Claude Code's session JSONL into a list of message dicts.

    Skips malformed lines silently (transcript may have corrupt entries
    from interrupted writes).
    """
    msgs: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Normalize: extract role + content for l1_manager consumption
                msg = obj.get("message") or {}
                role = msg.get("role") or obj.get("type") or "system"
                content = msg.get("content")
                if isinstance(content, list):
                    # Content blocks — extract text portions
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            t = block.get("text") or block.get("content") or ""
                            if t:
                                text_parts.append(str(t)[:1000])
                        elif isinstance(block, str):
                            text_parts.append(block[:1000])
                    content = "\n".join(text_parts)
                elif content is None:
                    content = obj.get("summary") or ""
                normalized = {
                    "role": role,
                    "content": str(content),
                }
                # Carry tool_use_id if present (l1 manager needs this for safe-cut)
                tool_use_id = obj.get("toolUseID") or msg.get("tool_use_id")
                if tool_use_id:
                    normalized["tool_use_id"] = tool_use_id
                msgs.append(normalized)
    except FileNotFoundError:
        return []
    except OSError:
        return []
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser(description="L1 eviction CLI wrapper")
    ap.add_argument("--transcript", required=True, help="Path to session JSONL")
    ap.add_argument("--marker-dir", required=True, help="Where to write marker file")
    ap.add_argument("--session-id", required=True, help="Session id for tagging")
    ap.add_argument("--evict-fraction", type=float, default=0.5)
    ap.add_argument("--pin-recent", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=60, help="Soft hint, unused for now")
    args = ap.parse_args()

    # Lazy import — keep consistent with all other modules in this package
    from memory import l1_manager
    from memory.embeddings import EmbeddingService
    from memory.store import MemoryStore

    # Acquire eviction mutex per marker dir (one in-flight eviction per
    # project at a time). Prevents the race where two Stop hooks in quick
    # succession both spawn detached evictors.
    marker_dir = pathlib.Path(args.marker_dir)
    marker_dir.mkdir(parents=True, exist_ok=True)
    lock_path = marker_dir / ".l1_evict.lock"
    try:
        from filelock import FileLock, Timeout
    except ImportError:
        # filelock not available — fall through, accept the race
        FileLock = None  # type: ignore
        Timeout = Exception  # type: ignore

    if FileLock is not None:
        try:
            with FileLock(str(lock_path), timeout=1):
                return _run_eviction(args, marker_dir, l1_manager,
                                       EmbeddingService, MemoryStore)
        except Timeout:
            print(json.dumps({"skipped": "another_eviction_in_progress"}))
            return 0
    else:
        return _run_eviction(args, marker_dir, l1_manager,
                              EmbeddingService, MemoryStore)


def _run_eviction(args, marker_dir, l1_manager, EmbeddingService, MemoryStore) -> int:
    transcript = load_transcript_jsonl(args.transcript)
    if not transcript:
        print(json.dumps({"skipped": "transcript_empty_or_missing"}))
        return 0

    # Production store path (lives in hermes/data/memory_store)
    store = MemoryStore(embedder=EmbeddingService())

    try:
        result = l1_manager.evict(
            transcript=transcript,
            store=store,
            evict_fraction=args.evict_fraction,
            pin_recent=args.pin_recent,
            marker_dir=marker_dir,
            session_id=args.session_id,
        )
    except Exception as e:
        # Record to error sentinel so /memory-stats can surface it.
        # TS hook still exits 0 — we never break the user's session.
        try:
            from memory.error_sentinel import record_error
            record_error(
                source="l1_manager_cli",
                error=str(e)[:500],
                context=f"session={args.session_id}",
            )
        except Exception:
            pass
        print(json.dumps({"error": str(e)[:300]}), file=sys.stderr)
        return 0  # never bubble up — TS hook stays clean

    print(json.dumps({
        "skipped": result.skipped,
        "block_id": result.block_id,
        "cut_index": result.cut_index,
        "chunks_stored": result.chunks_stored,
        "fallback_used": result.fallback_used,
        "marker_path": result.marker_path,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
