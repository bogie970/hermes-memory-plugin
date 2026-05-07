"""Embedding service for the memory system.

Uses sentence-transformers with CPU-only inference to avoid
conflicting with GPU rentals. Model is lazy-loaded on first use.
"""

from __future__ import annotations

import threading

from memory.config import EMBEDDING_MODEL, EMBEDDING_DEVICE


# Pinned commit on Alibaba-NLP/gte-modernbert-base to prevent silent
# model swap if the upstream HF repo changes (or is compromised).
# Update this SHA when intentionally upgrading; re-embedding pipeline
# in migrate_embeddings.py handles the schema migration.
EMBEDDING_REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"


# Module-level singleton — share one warm model across all in-process
# EmbeddingService instances. Eliminates double-load cost when multiple
# subsystems within the same Python process construct EmbeddingService
# (e.g. MemoryStore + write_gate + cleanup all in one CLI invocation).
# Cross-process sharing requires the daemon (see docs/memory/DAEMON_DESIGN.md).
_SHARED_MODEL = None
_SHARED_MODEL_KEY: tuple[str, str, str] | None = None
_SHARED_LOCK = threading.Lock()


class EmbeddingService:
    """Lazy-loaded sentence transformer for CPU-only embedding."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
        revision: str = EMBEDDING_REVISION,
    ):
        self._model = None
        self._model_name = model_name
        self._device = device
        self._revision = revision
        self._lock = threading.Lock()

    def _load(self):
        global _SHARED_MODEL, _SHARED_MODEL_KEY
        if self._model is not None:
            return
        # Try to reuse a process-wide warm model first
        key = (self._model_name, self._device, self._revision)
        with _SHARED_LOCK:
            if _SHARED_MODEL is not None and _SHARED_MODEL_KEY == key:
                self._model = _SHARED_MODEL
                return
            with self._lock:
                if self._model is not None:
                    return
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                    revision=self._revision,
                )
                _SHARED_MODEL = self._model
                _SHARED_MODEL_KEY = key

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns normalized vectors."""
        self._load()
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_embedding_dimension()
