"""Tests for the memory store — roundtrip, search, and schema validation."""

from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from memory.schema import MemoryRecord, MemoryType, ScoredMemory
from memory.embeddings import EmbeddingService
from memory.store import MemoryStore


@pytest.fixture
def tmp_store(tmp_path):
    """Create a MemoryStore with a temporary LanceDB directory."""
    embedder = EmbeddingService()
    store = MemoryStore(db_path=str(tmp_path / "test_memory"), embedder=embedder)
    return store


@pytest.fixture
def sample_records():
    """Create a set of diverse test records."""
    return [
        MemoryRecord(
            content="RTX 3070 market price is P50=$2.40/day on Clore.ai as of April 2026",
            memory_type=MemoryType.SEMANTIC,
            category="pricing",
            source="hermes",
            importance=0.7,
            tags=["pricing", "3070"],
        ),
        MemoryRecord(
            content="Windows Update caused 9 hours of downtime on 2026-04-16",
            memory_type=MemoryType.EPISODIC,
            category="incident",
            source="hermes",
            importance=0.8,
            tags=["incident", "downtime"],
        ),
        MemoryRecord(
            content="To restart the collector: kill the tmux session, then run python collector.py in a new tmux",
            memory_type=MemoryType.PROCEDURAL,
            category="workflow",
            source="hermes",
            importance=0.6,
            tags=["procedure", "collector"],
        ),
        MemoryRecord(
            content="Vast.ai offers $0.04-0.08/hr for RTX 3070 with 0% host fee",
            memory_type=MemoryType.SEMANTIC,
            category="expansion",
            source="hermes",
            importance=0.65,
            tags=["vast", "expansion"],
        ),
        MemoryRecord(
            content="CLORE token dropped 40% in 3 months from $0.0027 to $0.0016",
            memory_type=MemoryType.SEMANTIC,
            category="market",
            source="hermes",
            importance=0.5,
            tags=["token", "market"],
        ),
    ]


class TestMemoryStore:
    def test_insert_and_count(self, tmp_store, sample_records):
        ids = tmp_store.insert_many(sample_records)
        assert len(ids) == 5
        assert tmp_store.count() == 5

    def test_insert_single(self, tmp_store):
        record = MemoryRecord(
            content="Test memory record",
            memory_type=MemoryType.SEMANTIC,
            category="rule",
            source="test",
        )
        rid = tmp_store.insert(record)
        assert rid == record.id
        assert tmp_store.count() == 1

    def test_search_relevance(self, tmp_store, sample_records):
        tmp_store.insert_many(sample_records)

        results = tmp_store.search("What is the market price for RTX 3070?", k=3)
        assert len(results) > 0
        assert isinstance(results[0], ScoredMemory)
        # The pricing record should rank highest
        assert results[0].record.category == "pricing"
        assert results[0].relevance > 0

    def test_search_with_category_filter(self, tmp_store, sample_records):
        tmp_store.insert_many(sample_records)

        results = tmp_store.search("GPU rental", k=5, category_filter="incident")
        for r in results:
            assert r.record.category == "incident"

    def test_search_with_namespace_filter(self, tmp_store, sample_records):
        tmp_store.insert_many(sample_records)

        # Add one from a different source
        other = MemoryRecord(
            content="Physics experiment data",
            memory_type=MemoryType.SEMANTIC,
            category="rule",
            source="atlas",
        )
        tmp_store.insert(other)

        results = tmp_store.search("data", k=10, namespaces=["hermes"])
        for r in results:
            assert r.record.source == "hermes"

    def test_get_by_id(self, tmp_store, sample_records):
        tmp_store.insert_many(sample_records)
        target = sample_records[2]
        found = tmp_store.get_by_id(target.id)
        assert found is not None
        assert found.content == target.content
        assert found.memory_type == MemoryType.PROCEDURAL

    def test_get_by_id_not_found(self, tmp_store):
        result = tmp_store.get_by_id("nonexistent-id")
        assert result is None

    def test_archive(self, tmp_store, sample_records):
        tmp_store.insert_many(sample_records)
        target_id = sample_records[0].id

        tmp_store.archive(target_id)

        # Should not appear in normal search
        assert tmp_store.count(include_archived=False) == 4
        assert tmp_store.count(include_archived=True) == 5

    def test_delete(self, tmp_store, sample_records):
        tmp_store.insert_many(sample_records)
        target_id = sample_records[0].id

        tmp_store.delete(target_id)
        assert tmp_store.count() == 4
        assert tmp_store.get_by_id(target_id) is None

    def test_get_recent(self, tmp_store, sample_records):
        tmp_store.insert_many(sample_records)
        recent = tmp_store.get_recent(n=2)
        assert len(recent) == 2

    def test_update_importance(self, tmp_store, sample_records):
        tmp_store.insert_many(sample_records)
        target_id = sample_records[0].id

        tmp_store.update_importance(target_id, 0.99)
        updated = tmp_store.get_by_id(target_id)
        assert updated is not None
        assert abs(updated.importance - 0.99) < 0.01

    def test_empty_search(self, tmp_store):
        results = tmp_store.search("anything", k=5)
        assert results == []

    def test_memory_type_filter(self, tmp_store, sample_records):
        tmp_store.insert_many(sample_records)

        results = tmp_store.search(
            "how to restart",
            k=5,
            memory_type_filter="procedural",
        )
        for r in results:
            assert r.record.memory_type == MemoryType.PROCEDURAL


class TestMemoryRecord:
    def test_lance_roundtrip(self):
        record = MemoryRecord(
            content="Test content",
            memory_type=MemoryType.EPISODIC,
            category="session",
            source="test",
            importance=0.75,
            tags=["tag1", "tag2"],
            links=["link1"],
            metadata={"key": "value"},
        )
        fake_vector = [0.1] * 384
        lance_dict = record.to_lance_dict(fake_vector)

        reconstructed = MemoryRecord.from_lance_row(lance_dict)
        assert reconstructed.id == record.id
        assert reconstructed.content == record.content
        assert reconstructed.memory_type == record.memory_type
        assert reconstructed.category == record.category
        assert reconstructed.importance == record.importance
        assert reconstructed.tags == record.tags
        assert reconstructed.links == record.links
        assert reconstructed.metadata == record.metadata

    def test_empty_tags_links(self):
        record = MemoryRecord(
            content="Minimal record",
            memory_type=MemoryType.SEMANTIC,
            category="rule",
            source="test",
        )
        fake_vector = [0.0] * 384
        lance_dict = record.to_lance_dict(fake_vector)
        reconstructed = MemoryRecord.from_lance_row(lance_dict)
        assert reconstructed.tags == []
        assert reconstructed.links == []


class TestEmbeddingService:
    def test_embed_one(self):
        svc = EmbeddingService()
        vec = svc.embed_one("Hello world")
        assert len(vec) == 384
        assert isinstance(vec[0], float)

    def test_embed_batch(self):
        svc = EmbeddingService()
        vecs = svc.embed(["Hello", "World", "Test"])
        assert len(vecs) == 3
        assert all(len(v) == 384 for v in vecs)

    def test_dimension_property(self):
        svc = EmbeddingService()
        assert svc.dimension == 384


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
