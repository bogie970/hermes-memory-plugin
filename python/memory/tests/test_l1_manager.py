"""Phase D — L1 manager tests.

Eviction algorithm: take oldest fraction of transcript, send to Haiku for
segmentation, vectorize chunks into L2, write placeholder marker.

Run: pytest aisys/memory/tests/test_l1_manager.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


@pytest.fixture
def v2_store(lance_store_v1):
    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)
    return lance_store_v1


@pytest.fixture
def synthetic_transcript():
    """50 messages, alternating user/assistant, with one tool_use pair."""
    msgs = []
    for i in range(50):
        if i == 20:
            msgs.append({"role": "assistant", "content": "calling tool",
                         "tool_use_id": "t_20"})
            msgs.append({"role": "tool", "tool_use_id": "t_20", "content": "result"})
        elif i % 2 == 0:
            msgs.append({"role": "user", "content": f"user message {i}"})
        else:
            msgs.append({"role": "assistant", "content": f"assistant reply {i}"})
    return msgs


@pytest.fixture
def fake_haiku_segmenter():
    """Mock Haiku that returns a fixed segmentation. Injected via chat_fn arg."""

    def _fake_chat(messages, tools=None, model="haiku", **kw):
        result = {
            "summary": "Discussed Ollama timeout debugging",
            "topics": ["ollama", "timeout", "subconscious"],
            "decisions": ["dropped Ollama in favor of Haiku via claude-cli"],
            "files": ["aisys/subconscious/loop.py"],
            "open": [],
            "chunks": [
                {"text": "user reported 30s Ollama cold-start timeout",
                 "topic_tag": "timeout", "importance": 0.6},
                {"text": "decided to switch to Haiku via claude --print",
                 "topic_tag": "decision", "importance": 0.8},
            ],
        }
        return {"role": "assistant", "content": json.dumps(result)}

    return _fake_chat


# ----- find_safe_cut: the eviction boundary -----

def test_find_safe_cut_respects_pin_recent(synthetic_transcript):
    """Last N turns (pin_recent) are exempt from eviction."""
    from memory.l1_manager import find_safe_cut
    cut = find_safe_cut(synthetic_transcript, target_fraction=0.5, pin_recent=20)
    # Cut index must leave at least pin_recent messages after it
    assert cut <= len(synthetic_transcript) - 20


def test_find_safe_cut_does_not_split_tool_pair(synthetic_transcript):
    """Tool_use and tool_result must stay together — cut never lands between them."""
    from memory.l1_manager import find_safe_cut
    # The synthetic transcript has tool_use at idx 20, tool result at idx 21
    # If the natural cut would be at 21, we must shift to 22 or to 20
    cut = find_safe_cut(synthetic_transcript, target_fraction=0.42, pin_recent=10)
    # Verify: if cut == 21, that's a bug (splits the tool pair)
    assert cut != 21


def test_find_safe_cut_returns_zero_for_short_transcript():
    """If transcript is short enough to skip eviction, returns 0."""
    from memory.l1_manager import find_safe_cut
    short = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hi"}]
    cut = find_safe_cut(short, target_fraction=0.5, pin_recent=20)
    # pin_recent (20) > total (2), nothing to cut
    assert cut == 0


# ----- evict: full pipeline -----

def test_evict_returns_skipped_when_below_threshold(v2_store, synthetic_transcript):
    """If transcript is already small, evict() reports skipped."""
    from memory.l1_manager import evict
    short = synthetic_transcript[:5]
    result = evict(transcript=short, store=v2_store, evict_fraction=0.5,
                   pin_recent=20, marker_dir=None)
    assert result.skipped


def test_evict_writes_l2_chunks(v2_store, synthetic_transcript, fake_haiku_segmenter, tmp_path):
    """Evicted chunks land in L2 (LanceDB) with source=l1_evict tag."""
    from memory.l1_manager import evict
    result = evict(transcript=synthetic_transcript, store=v2_store,
                   evict_fraction=0.5, pin_recent=10, marker_dir=tmp_path,
                   chat_fn=fake_haiku_segmenter)
    assert result.chunks_stored >= 1
    # Find the l1_evict-tagged rows
    rows = v2_store.scan_v2()
    evict_rows = [r for r in rows if "l1_evict" in (r.get("category") or "")]
    assert len(evict_rows) == result.chunks_stored


def test_evict_writes_marker_file(v2_store, synthetic_transcript, fake_haiku_segmenter, tmp_path):
    """Marker file written to marker_dir after eviction."""
    from memory.l1_manager import evict
    evict(transcript=synthetic_transcript, store=v2_store, evict_fraction=0.5,
          pin_recent=10, marker_dir=tmp_path, chat_fn=fake_haiku_segmenter)
    markers = list(tmp_path.glob("l1_evicted_*.md"))
    assert len(markers) == 1
    text = markers[0].read_text()
    assert "<l1_evicted" in text
    assert "Retrieve via" in text


def test_evict_idempotent_via_block_id(v2_store, synthetic_transcript, fake_haiku_segmenter, tmp_path):
    """Re-running evict() with the same transcript is a no-op."""
    from memory.l1_manager import evict
    first = evict(transcript=synthetic_transcript, store=v2_store,
                  evict_fraction=0.5, pin_recent=10, marker_dir=tmp_path,
                  chat_fn=fake_haiku_segmenter)
    second = evict(transcript=synthetic_transcript, store=v2_store,
                   evict_fraction=0.5, pin_recent=10, marker_dir=tmp_path,
                   chat_fn=fake_haiku_segmenter)
    assert second.skipped == "already_evicted"
    # Chunk count unchanged
    rows = v2_store.scan_v2()
    evict_rows = [r for r in rows if "l1_evict" in (r.get("category") or "")]
    assert len(evict_rows) == first.chunks_stored


def test_evict_chunks_carry_session_metadata(v2_store, synthetic_transcript, fake_haiku_segmenter, tmp_path):
    """Each L2 chunk has session_id and source_ref linking back to evict event."""
    from memory.l1_manager import evict
    result = evict(transcript=synthetic_transcript, store=v2_store,
                   evict_fraction=0.5, pin_recent=10, marker_dir=tmp_path,
                   session_id="test-session-001", chat_fn=fake_haiku_segmenter)
    rows = v2_store.scan_v2()
    evict_rows = [r for r in rows if "l1_evict" in (r.get("category") or "")]
    assert all("test-session-001" in (r.get("source_ref") or "") for r in evict_rows)


def test_evict_handles_haiku_invalid_json(v2_store, synthetic_transcript, tmp_path):
    """If Haiku returns malformed JSON, fall back to non-LLM segmentation."""

    def _bad_chat(messages, tools=None, model="haiku", **kw):
        return {"role": "assistant", "content": "not valid json {[}"}

    from memory.l1_manager import evict
    result = evict(transcript=synthetic_transcript, store=v2_store,
                   evict_fraction=0.5, pin_recent=10, marker_dir=tmp_path,
                   chat_fn=_bad_chat)
    # Fallback path — chunks created from raw messages, summary="(unsegmented)"
    assert result.chunks_stored >= 1
    assert result.fallback_used
