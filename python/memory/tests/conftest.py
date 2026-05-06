"""Shared pytest fixtures for memory tests.

Goals:
- Fast: use deterministic fake embeddings instead of real model
- Isolated: every test gets a tmp_path-scoped LanceDB store
- Clean: no test pollutes user's actual ~/.hermes data
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


class FakeEmbedder:
    """Deterministic fake embedder — hash content -> stable vector.

    Used in fast tests to avoid loading the real 500MB model.
    Cosine similarity behavior approximates real embeddings for
    same/different content, but is NOT semantically meaningful.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed_one(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand 32 bytes to dim floats by repeating + normalizing
        floats = []
        while len(floats) < self.dim:
            for i in range(0, len(h), 4):
                if len(floats) >= self.dim:
                    break
                chunk = h[i : i + 4]
                if len(chunk) < 4:
                    chunk = chunk + b"\x00" * (4 - len(chunk))
                val = int.from_bytes(chunk, "big") / (2**32)
                floats.append(val * 2 - 1)  # range -1 to 1
            h = hashlib.sha256(h).digest()  # extend with hash chain
        # L2 normalize
        norm = sum(f * f for f in floats) ** 0.5
        return [f / norm for f in floats] if norm > 0 else floats

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


@pytest.fixture
def fake_embedder():
    """768-dim deterministic fake embedder (matches gte-modernbert-base dim)."""
    return FakeEmbedder(dim=768)


@pytest.fixture
def tmp_lance_path(tmp_path, monkeypatch):
    """Per-test isolated LanceDB path. Ensures no test touches ~/.hermes/."""
    lance_dir = tmp_path / "lance"
    lance_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_DATA_DIR", str(tmp_path))
    return lance_dir


@pytest.fixture
def lance_store_v1(tmp_lance_path, fake_embedder):
    """A MemoryStore at v1 schema (no Phase B fields yet).

    Used to test migration FROM v1 TO v2.
    """
    from memory.store import MemoryStore
    store = MemoryStore(db_path=str(tmp_lance_path), embedder=fake_embedder)
    return store


@pytest.fixture
def sample_v1_records():
    """Diverse v1 MemoryRecord set — representative of what's in production."""
    from memory.schema import MemoryRecord, MemoryType

    return [
        MemoryRecord(
            content="LanceDB is the chosen vector store for hermes",
            memory_type=MemoryType.SEMANTIC,
            category="architecture",
            source="hermes",
            importance=0.8,
            tags=["lancedb", "decision"],
        ),
        MemoryRecord(
            content="Atlas dashboard runs on streamlit at port 8501",
            memory_type=MemoryType.SEMANTIC,
            category="system",
            source="atlas",
            importance=0.6,
            tags=["dashboard", "atlas"],
        ),
        MemoryRecord(
            content="To restart the collector: pkill -f collector && python collector.py",
            memory_type=MemoryType.PROCEDURAL,
            category="workflow",
            source="hermes",
            importance=0.7,
            tags=["procedure", "restart"],
        ),
    ]


@pytest.fixture
def frozen_now(monkeypatch):
    """Deterministic timestamps."""
    fixed = datetime(2026, 5, 6, 14, 0, 0, tzinfo=timezone.utc)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed if tz is None else fixed.astimezone(tz)

    monkeypatch.setattr("memory.schema.datetime", FixedDatetime)
    return fixed
