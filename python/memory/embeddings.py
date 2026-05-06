"""Embedding service for the memory system.

Uses sentence-transformers with CPU-only inference to avoid
conflicting with GPU rentals. Model is lazy-loaded on first use.
"""

from __future__ import annotations

import threading

from memory.config import EMBEDDING_MODEL, EMBEDDING_DEVICE


class EmbeddingService:
    """Lazy-loaded sentence transformer for CPU-only embedding."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
    ):
        self._model = None
        self._model_name = model_name
        self._device = device
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self._model_name, device=self._device)

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
