"""LLM-enriched note construction for memories (A-MEM pattern).

Before storing a memory, an LLM generates:
  - keywords: key concepts for search
  - context: a contextual description relating this to broader knowledge
  - generated_tags: semantic tags beyond what the caller provided

The embedding is then computed over concat(content, keywords, context, tags)
rather than raw content alone, producing richer vector representations.
"""

from __future__ import annotations

import json
import logging

from memory.llm_caller import get_llm_caller, get_llm_caller_for_agent, LLMCaller

log = logging.getLogger(__name__)

ENRICHMENT_PROMPT = """\
You are a memory indexing system. Given the following memory content, produce a JSON object with exactly three fields:

1. "keywords": a list of 3-7 key concepts/terms (short phrases, not sentences)
2. "context": one sentence describing how this memory relates to broader knowledge or why it matters
3. "generated_tags": a list of 3-5 lowercase single-word or hyphenated tags for retrieval

Memory content:
---
{content}
---
Category: {category}

Respond with ONLY valid JSON, no markdown fences, no explanation."""


def enrich_memory(content: str, category: str, llm: LLMCaller | None = None) -> dict:
    """Call LLM to generate keywords, context, and tags for a memory.

    Returns dict with keys: keywords (list[str]), context (str), generated_tags (list[str]).
    Returns empty enrichment if LLM is unavailable or fails.
    """
    if llm is None:
        llm = get_llm_caller()
    if llm is None:
        return _empty_enrichment()

    prompt = ENRICHMENT_PROMPT.format(content=content[:1500], category=category)

    try:
        raw = llm(prompt)
    except Exception as e:
        log.warning(f"Enrichment LLM call failed: {e}")
        return _empty_enrichment()

    if not raw:
        return _empty_enrichment()

    return _parse_enrichment(raw)


def build_enriched_text(content: str, enrichment: dict) -> str:
    """Build the concatenated text used for embedding.

    Following A-MEM: embed over concat(content, keywords, context, tags)
    for richer vector representations.
    """
    parts = [content]

    keywords = enrichment.get("keywords", [])
    if keywords:
        parts.append("Keywords: " + ", ".join(keywords))

    context = enrichment.get("context", "")
    if context:
        parts.append("Context: " + context)

    tags = enrichment.get("generated_tags", [])
    if tags:
        parts.append("Tags: " + ", ".join(tags))

    return " ".join(parts)


def _parse_enrichment(raw: str) -> dict:
    """Parse LLM output into enrichment dict, with fallbacks."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning(f"Enrichment parse failed, raw: {raw[:200]}")
        return _empty_enrichment()

    return {
        "keywords": _ensure_str_list(data.get("keywords", [])),
        "context": str(data.get("context", "")),
        "generated_tags": _ensure_str_list(data.get("generated_tags", [])),
    }


def _ensure_str_list(val) -> list[str]:
    if isinstance(val, list):
        return [str(v) for v in val]
    return []


def _empty_enrichment() -> dict:
    return {"keywords": [], "context": "", "generated_tags": []}
