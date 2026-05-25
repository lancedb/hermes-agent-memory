"""Sentence Transformers embedding helper."""
from __future__ import annotations

import logging
import threading
from typing import List

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """Lazy wrapper around SentenceTransformers with normalized outputs."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._dim: int | None = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("loading sentence-transformers model %s", self.model_name)
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            dim = getattr(self.model, "get_sentence_embedding_dimension")()
            if dim is None:
                vector = self.embed_one("dimension probe")
                dim = len(vector)
            self._dim = int(dim)
        return self._dim

    def warm(self) -> int:
        """Load the model and return its embedding dimension."""
        return self.dim

    def embed_one(self, text: str) -> List[float]:
        vectors = self.embed([text])
        return vectors[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [row.astype("float32").tolist() for row in encoded]
