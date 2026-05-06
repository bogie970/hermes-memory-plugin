"""Tests for triple-scored retrieval, hybrid search, routing, and self-RAG."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from memory.schema import MemoryRecord, MemoryType, ScoredMemory
from memory.embeddings import EmbeddingService
from memory.store import MemoryStore
from memory.retrieval import TripleScoredRetriever, HybridRetriever
from memory.router import QueryRouter


@pytest.fixture
def populated_store(tmp_path):
    """Store with diverse records for retrieval testing."""
    embedder = EmbeddingService()
    store = MemoryStore(db_path=str(tmp_path / "test_retrieval"), embedder=embedder)

    now = datetime.now(timezone.utc)

    records = [
        MemoryRecord(
            content="RTX 3070 spot price is $1.75/day on Clore marketplace",
            memory_type=MemoryType.SEMANTIC,
            category="pricing",
            source="hermes",
            importance=0.8,
            created_at=now - timedelta(hours=2),
            last_accessed=now - timedelta(hours=1),
            tags=["pricing"],
        ),
        MemoryRecord(
            content="Windows Update caused 9 hours of downtime on April 16",
            memory_type=MemoryType.EPISODIC,
            category="incident",
            source="hermes",
            importance=0.9,
            created_at=now - timedelta(days=7),
            last_accessed=now - timedelta(days=5),
            tags=["incident"],
        ),
        MemoryRecord(
            content="To restart collector: kill tmux session, run python collector.py",
            memory_type=MemoryType.PROCEDURAL,
            category="workflow",
            source="hermes",
            importance=0.6,
            created_at=now - timedelta(days=30),
            last_accessed=now - timedelta(days=20),
            tags=["procedure"],
        ),
        MemoryRecord(
            content="Vast.ai RTX 3070 rates are $0.04-0.08/hr with 0% host fee",
            memory_type=MemoryType.SEMANTIC,
            category="expansion",
            source="hermes",
            importance=0.7,
            created_at=now - timedelta(hours=12),
            last_accessed=now - timedelta(hours=6),
            tags=["vast", "expansion"],
        ),
        MemoryRecord(
            content="RTX 3090 is the best expansion GPU at $720 used with 24GB VRAM",
            memory_type=MemoryType.SEMANTIC,
            category="expansion",
            source="hermes",
            importance=0.75,
            created_at=now - timedelta(days=3),
            last_accessed=now - timedelta(days=2),
            tags=["expansion", "3090"],
        ),
    ]
    store.insert_many(records)
    return store


class TestTripleScoredRetriever:
    def test_returns_scored_results(self, populated_store):
        retriever = TripleScoredRetriever(populated_store)
        results = retriever.retrieve("What is the RTX 3070 price?", k=3)
        assert len(results) > 0
        assert all(isinstance(r, ScoredMemory) for r in results)
        assert all(r.combined_score > 0 for r in results)

    def test_relevance_beats_recency(self, populated_store):
        retriever = TripleScoredRetriever(populated_store)
        results = retriever.retrieve("RTX 3070 spot price Clore", k=3)
        assert results[0].record.category == "pricing"

    def test_importance_contributes(self, populated_store):
        retriever = TripleScoredRetriever(populated_store)
        results = retriever.retrieve("downtime and Windows Update problems", k=3)
        # Incident memory has highest importance (0.9) and matches keywords
        scores = [(r.record.category, r.combined_score) for r in results]
        incident_results = [s for s in scores if s[0] == "incident"]
        assert len(incident_results) > 0

    def test_category_filter(self, populated_store):
        retriever = TripleScoredRetriever(populated_store)
        results = retriever.retrieve("GPU", k=5, category_filter="expansion")
        for r in results:
            assert r.record.category == "expansion"

    def test_recency_scoring(self, populated_store):
        retriever = TripleScoredRetriever(populated_store)
        results = retriever.retrieve("RTX 3070 information", k=5)
        # All should have recency scores between 0 and 1
        for r in results:
            assert 0.0 <= r.recency <= 1.0


class TestHybridRetriever:
    def test_hybrid_returns_results(self, populated_store):
        retriever = HybridRetriever(populated_store)
        results = retriever.retrieve("RTX 3070 price", k=3)
        assert len(results) > 0

    def test_bm25_exact_match(self, populated_store):
        """BM25 should boost results that contain exact keywords."""
        retriever = HybridRetriever(populated_store)
        results = retriever.retrieve("Windows Update downtime", k=3)
        categories = [r.record.category for r in results]
        assert "incident" in categories

    def test_refresh_index(self, populated_store):
        retriever = HybridRetriever(populated_store)
        retriever.refresh_index()
        results = retriever.retrieve("collector restart", k=2)
        assert len(results) > 0


class TestQueryRouter:
    def test_pricing_intent(self):
        router = QueryRouter()
        plan = router.route("What should I set the spot price to?")
        assert plan.intent == "pricing"
        assert "pricing" in plan.category_filters

    def test_troubleshooting_intent(self):
        router = QueryRouter()
        plan = router.route("The server is offline, how do I fix it?")
        assert plan.intent == "troubleshooting"

    def test_expansion_intent(self):
        router = QueryRouter()
        plan = router.route("Should I buy a 3090 for expansion?")
        assert plan.intent == "expansion"

    def test_procedural_intent(self):
        router = QueryRouter()
        plan = router.route("How to restart the collector service?")
        assert plan.intent == "procedural"

    def test_general_fallback(self):
        router = QueryRouter()
        plan = router.route("Tell me about the project")
        assert plan.intent == "general"

    def test_platform_intent(self):
        router = QueryRouter()
        plan = router.route("Compare Vast.ai and Clore fees")
        assert plan.intent == "platform"




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
