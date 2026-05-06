"""L1 manager — eviction orchestrator.

When the live transcript approaches the compaction threshold, this module:
  1. Picks a safe cut index (oldest fraction, not splitting tool_use pairs)
  2. Sends the older half to Haiku for semantic segmentation
  3. Vectorizes each chunk into L2 (LanceDB) via the write gate
  4. Writes a placeholder marker file describing what was evicted

Idempotent via block_id (sha256 of evicted content).

Failure modes handled:
  - Haiku returns invalid JSON -> fallback to non-LLM segmentation
  - Already-evicted block (same block_id seen) -> skipped no-op
  - Transcript too short to evict -> skipped no-op
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from memory.schema import MemoryRecord, MemoryType
from memory.store import MemoryStore
from memory.write_gate import write_memory


# ---- Result dataclass ----

@dataclass
class EvictResult:
    chunks_stored: int = 0
    block_id: str = ""
    cut_index: int = 0
    skipped: str | None = None
    fallback_used: bool = False
    summary: str = ""
    topics: list[str] = field(default_factory=list)
    marker_path: str | None = None


# ---- Cut-point algorithm ----

MIN_TURNS_TO_EVICT = 10  # don't evict from very short transcripts


def find_safe_cut(
    transcript: list[dict],
    target_fraction: float,
    pin_recent: int,
) -> int:
    """Return an index `i` such that messages[0:i] should be evicted.

    Constraints:
      - At least `pin_recent` messages must remain after cut (untouched)
      - Cut must NOT land between a tool_use and its tool_result
      - Cut must land on an assistant boundary when possible
      - Returns 0 if eviction would not yield meaningful work
    """
    n = len(transcript)
    if n < MIN_TURNS_TO_EVICT:
        return 0
    if pin_recent >= n:
        return 0

    # Naive target: drop oldest target_fraction
    raw_cut = int(n * target_fraction)
    # Cap at n - pin_recent so we leave headroom
    raw_cut = min(raw_cut, n - pin_recent)
    if raw_cut <= 0:
        return 0

    # Tool-pair safety: if cut falls between tool_use and tool_result, shift
    # forward (consume both) until safe.
    while raw_cut < n:
        msg_at_cut = transcript[raw_cut] if raw_cut < n else None
        prev = transcript[raw_cut - 1] if raw_cut - 1 >= 0 else None

        # Bad split: prev is a tool_use, current is its tool_result
        if prev and msg_at_cut:
            prev_tu_id = prev.get("tool_use_id")
            cur_role = msg_at_cut.get("role")
            cur_tu_id = msg_at_cut.get("tool_use_id")
            if (prev_tu_id and cur_role == "tool" and cur_tu_id == prev_tu_id):
                raw_cut += 1
                continue
        break

    # Ensure we still leave pin_recent
    if raw_cut > n - pin_recent:
        return 0
    return raw_cut


# ---- Block id (idempotency key) ----

def compute_block_id(messages: Iterable[dict]) -> str:
    """Stable hash of the evicted message slice."""
    h = hashlib.sha256()
    for m in messages:
        h.update(json.dumps(m, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


# ---- Haiku segmentation ----

EVICT_PROMPT_TEMPLATE = """You are processing a Claude Code conversation transcript fragment that
is being moved out of active context (L1) into vector storage (L2).

Your job: produce semantic chunks suitable for later retrieval, plus a
short overview that will replace the raw messages in L1.

Return ONLY a JSON object with this shape:
{{
  "summary": "one paragraph, ~150 tokens, what was discussed",
  "topics": ["topic1", "topic2", ...],
  "decisions": ["decision1", ...],
  "files": ["path/to/file.py", ...],
  "open": ["unresolved_thing_1", ...],
  "chunks": [
    {{"text": "chunk content", "topic_tag": "tag", "importance": 0.0-1.0}},
    ...
  ]
}}

Do NOT include any text outside the JSON. Do not wrap in markdown.

Transcript fragment:
{transcript}
"""


def _default_chat_fn(messages, tools=None, model="haiku", timeout=60, **kw):
    """Find and call the production claude_cli_chat module.

    Tries multiple known locations; returns None if not available.
    The plugin ships claude_cli_chat in claude-subconscious/python/subconscious/.
    """
    import importlib
    for module_path in ("subconscious.claude_cli_chat",
                        "claude_subconscious.python.subconscious.claude_cli_chat"):
        try:
            mod = importlib.import_module(module_path)
            return mod.chat(messages=messages, tools=tools, model=model,
                             timeout=timeout, **kw)
        except (ImportError, ModuleNotFoundError):
            continue
    return None


def _call_haiku_segmenter(
    transcript_chunk: list[dict],
    chat_fn=None,
) -> dict | None:
    """Send the slice to Haiku, parse JSON. Returns None on failure.

    chat_fn: callable(messages, tools, model, timeout, **kw) -> response dict.
    Defaults to the production claude_cli_chat lookup; tests inject a fake.
    """
    if chat_fn is None:
        chat_fn = _default_chat_fn

    transcript_text = "\n".join(
        f"{m.get('role', '?')}: {m.get('content', '')}"[:2000]
        for m in transcript_chunk
    )
    prompt = EVICT_PROMPT_TEMPLATE.format(transcript=transcript_text[:6000])

    try:
        response = chat_fn(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            model="haiku",
            timeout=60,
        )
        if response is None:
            return None
        content = response.get("content", "")
        # Extract JSON (may be wrapped in markdown despite instructions)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return None


# ---- Fallback non-LLM segmentation ----

def _fallback_chunks(transcript_chunk: list[dict], chunk_size: int = 4) -> dict:
    """No-LLM segmentation: bundle every N messages into a chunk.

    Used when Haiku is unavailable or returns malformed JSON.
    """
    chunks = []
    for i in range(0, len(transcript_chunk), chunk_size):
        slice_msgs = transcript_chunk[i : i + chunk_size]
        text = "\n".join(
            f"{m.get('role', '?')}: {m.get('content', '')}"[:1000]
            for m in slice_msgs
        )
        chunks.append({"text": text, "topic_tag": "unsegmented", "importance": 0.4})

    return {
        "summary": "(unsegmented - Haiku unavailable)",
        "topics": [],
        "decisions": [],
        "files": [],
        "open": [],
        "chunks": chunks,
    }


# ---- Marker file ----

def _format_marker(
    block_id: str,
    cut_index: int,
    total: int,
    parsed: dict,
    chunk_ids: list[str],
) -> str:
    topics = ", ".join(parsed.get("topics", []) or ["(none)"])
    decisions_lines = "\n".join(f"- {d}" for d in parsed.get("decisions", []))
    files_line = ", ".join(parsed.get("files", []) or [])
    open_lines = "\n".join(f"- {o}" for o in parsed.get("open", []))

    summary = parsed.get("summary", "(no summary)")
    timestamp = datetime.now(timezone.utc).isoformat()

    return f"""<l1_evicted block_id="{block_id}" turns="{cut_index}" of="{total}" timestamp="{timestamp}">
Topics: {topics}
{summary}

Decisions:
{decisions_lines if decisions_lines else "- (none)"}

Files: {files_line or "(none)"}

Open:
{open_lines if open_lines else "- (none)"}

L2 chunk ids: {", ".join(chunk_ids)}

Retrieve via: memory_recall(query, scope="l1_evict")
</l1_evicted>
"""


# ---- Main entry point ----

def evict(
    *,
    transcript: list[dict],
    store: MemoryStore,
    evict_fraction: float = 0.5,
    pin_recent: int = 20,
    marker_dir: Path | None = None,
    session_id: str = "session-unknown",
    chat_fn=None,
) -> EvictResult:
    """Run the full eviction pipeline. Idempotent.

    Args:
        transcript: list of message dicts (role, content, optional tool_use_id)
        store: MemoryStore instance (must be v2 schema)
        evict_fraction: fraction of oldest content to evict (default 0.5)
        pin_recent: minimum number of messages to leave in L1 after cut
        marker_dir: directory to write the placeholder marker file (or None)
        session_id: tag chunks with this session for cross-session attribution

    Returns:
        EvictResult with chunks_stored, block_id, cut_index, skipped reason.
    """
    # 1. Find cut point
    cut_index = find_safe_cut(transcript, evict_fraction, pin_recent)
    if cut_index <= 0:
        return EvictResult(skipped="below_threshold")

    evicted = transcript[:cut_index]
    block_id = compute_block_id(evicted)

    # 2. Idempotency: if any L2 row already references this block_id, skip
    rows = store.scan_v2()
    for r in rows:
        if block_id in (r.get("source_ref") or ""):
            return EvictResult(skipped="already_evicted", block_id=block_id,
                               cut_index=cut_index)

    # 3. Try Haiku segmentation, fall back if it fails
    parsed = _call_haiku_segmenter(evicted, chat_fn=chat_fn)
    fallback_used = False
    if parsed is None or "chunks" not in parsed:
        parsed = _fallback_chunks(evicted)
        fallback_used = True

    # 4. Write each chunk to L2 via write_gate
    chunk_ids: list[str] = []
    for i, chunk in enumerate(parsed.get("chunks", [])):
        text = chunk.get("text", "")
        if not text.strip():
            continue
        importance = float(chunk.get("importance", 0.5))
        topic = chunk.get("topic_tag", "unsegmented")
        try:
            chunk_id = write_memory(
                store=store,
                content=text,
                writer="subconscious_haiku",
                provenance="llm_inferred",
                source_ref=f"l1_evict:{block_id}:{session_id}:{i}",
                confidence=importance,
                tags=[topic, "l1_evict"],
                memory_type=MemoryType.EPISODIC,
                category="l1_evict",
            )
            chunk_ids.append(chunk_id)
        except Exception:
            continue

    # 5. Write marker file
    marker_path: Path | None = None
    if marker_dir is not None:
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker_path = marker_dir / f"l1_evicted_{block_id}.md"
        marker_text = _format_marker(
            block_id=block_id,
            cut_index=cut_index,
            total=len(transcript),
            parsed=parsed,
            chunk_ids=chunk_ids,
        )
        marker_path.write_text(marker_text, encoding="utf-8")

    return EvictResult(
        chunks_stored=len(chunk_ids),
        block_id=block_id,
        cut_index=cut_index,
        skipped=None,
        fallback_used=fallback_used,
        summary=parsed.get("summary", ""),
        topics=list(parsed.get("topics", [])),
        marker_path=str(marker_path) if marker_path else None,
    )
