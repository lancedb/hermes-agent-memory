"""OpenAI embedding helper (text-embedding-3-small by default).

Embeddings are produced via the OpenAI embeddings API. OpenAI embeddings are
L2-normalized to unit length, so no extra normalization is applied. The client
reads credentials from the standard OPENAI_API_KEY / OPENAI_BASE_URL
environment variables.
"""
from __future__ import annotations

import logging
import threading
from typing import List

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"

# Native output dimensions for known OpenAI embedding models, so dim()/warm()
# don't need a network round-trip just to size the Lance vector schema.
_KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Inputs per embeddings request. The API allows far more, but chunking keeps
# request sizes bounded for large ingests (e.g. a long conversation haystack).
_MAX_BATCH = 128


class OpenAIEmbedder:
    """Lazy wrapper around the OpenAI embeddings API."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, dimensions: int | None = None) -> None:
        self.model_name = model_name or DEFAULT_MODEL
        self._dimensions = dimensions
        # Known offline; falls back to a one-shot probe for unknown models.
        self._dim: int | None = dimensions or _KNOWN_DIMS.get(self.model_name)
        self._client = None
        self._lock = threading.Lock()

    @property
    def client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from openai import OpenAI

                    logger.info("creating OpenAI client for embeddings (%s)", self.model_name)
                    # The SDK retries 408/409/429/5xx with exponential backoff;
                    # bump from the default 2 so a transient 500 during a long
                    # ingest doesn't abort the whole run.
                    self._client = OpenAI(max_retries=6, timeout=60.0)
        return self._client

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_one("dimension probe"))
        return self._dim

    def warm(self) -> int:
        """Return the embedding dimension. Offline for known models."""
        return self.dim

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), _MAX_BATCH):
            # OpenAI rejects empty strings; substitute a single space so row
            # ordering is preserved 1:1 with the response.
            batch = [t if t else " " for t in texts[start : start + _MAX_BATCH]]
            kwargs: dict = {"model": self.model_name, "input": batch}
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions
            response = self.client.embeddings.create(**kwargs)
            for item in response.data:
                out.append([float(x) for x in item.embedding])
        return out
