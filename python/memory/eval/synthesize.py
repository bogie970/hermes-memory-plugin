"""Bootstrap fixtures by asking an LLM to generate retrieval queries
from existing memories.

For each seed memory, generates 3 fixtures: direct query, paraphrase, negative.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from memory.eval.harness import Fixture
from memory.store import MemoryStore


SYNTH_PROMPT = """You are bootstrapping a memory retrieval test set.

Given this memory, return a JSON list of 3 objects:
- one "direct" query a user would ask to retrieve this memory
- one "paraphrase" — same intent, different words
- one "negative" — completely unrelated query

Return ONLY a JSON list. No prose, no markdown.
Format:
[
  {{"query": "...", "kind": "direct"}},
  {{"query": "...", "kind": "paraphrase"}},
  {{"query": "...", "kind": "negative"}}
]

Memory:
{content}
"""


def synthesize(
    store: MemoryStore,
    n_seeds: int = 50,
    chat_fn=None,
    model: str = "sonnet",
) -> list[Fixture]:
    """Generate fixtures from up to n_seeds memories in the store.

    Args:
        store: source of seed memories
        n_seeds: max seeds to draw (real LLM calls)
        chat_fn: callable(messages, model, **kw) -> response dict
        model: which model to use (defaults to sonnet for daily eval bootstrap)

    Returns:
        list of Fixture objects (3x n_seeds when fully populated)
    """
    if chat_fn is None:
        from memory.l1_manager import _default_chat_fn
        chat_fn = _default_chat_fn

    rows = store.scan_v2()
    seeds = [r for r in rows if (r.get("importance") or 0) >= 0.5][:n_seeds]

    fixtures: list[Fixture] = []
    for seed in seeds:
        content = seed.get("content", "")
        seed_id = seed.get("id", "")
        if not content:
            continue

        try:
            response = chat_fn(
                messages=[{"role": "user", "content": SYNTH_PROMPT.format(content=content)}],
                model=model,
                timeout=30,
            )
        except Exception:
            continue
        if response is None:
            continue

        text = response.get("content", "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            generated = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue

        for entry in generated:
            kind = entry.get("kind", "direct")
            query = entry.get("query", "").strip()
            if not query:
                continue
            category = "negative" if kind == "negative" else (
                "paraphrase" if kind == "paraphrase" else "single_hop"
            )
            expected = [] if category == "negative" else [seed_id]
            fixtures.append(Fixture(
                id=f"synth_{uuid.uuid4().hex[:8]}",
                query=query,
                expected_memory_ids=expected,
                k=5,
                category=category,
            ))

    return fixtures
