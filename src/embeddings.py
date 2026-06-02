"""OpenAI-compatible embedding helper (text-embedding-3-small by default).

Embeddings are produced via any OpenAI-compatible embeddings API. By default
this is OpenAI itself, but ``base_url`` and ``api_key_env`` let you point at any
endpoint that speaks the same request shape — Nous Portal, Together, vLLM,
Ollama / LM Studio in OpenAI-compatible mode, or a self-hosted server — without
new code. OpenAI's own embeddings are L2-normalized to unit length, so no extra
normalization is applied (other providers are assumed to do the same).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

# Native output dimensions for known OpenAI embedding models, so dim()/warm()
# don't need a network round-trip just to size the Lance vector schema. Unknown
# models (including any non-OpenAI endpoint) fall back to a one-shot probe.
_KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Inputs per embeddings request. Providers cap this differently — OpenAI allows
# up to 2048, but Google/Gemini caps at 100 and Cohere at 96 — so it's
# configurable (``max_batch``) and the default is the safe common denominator
# that works across OpenAI and Gemini. Chunking also keeps request sizes bounded
# for large ingests (e.g. a long conversation haystack).
_MAX_BATCH = 100


class OpenAICompatibleEmbedder:
    """Lazy wrapper around an OpenAI-compatible embeddings API.

    The endpoint is selected by config, not hardcoded:

    - ``base_url``    — override the API base (``None`` = OpenAI's default).
    - ``api_key_env`` — the environment variable holding the API key.
    - ``dimensions``  — optional output dimensions (matryoshka models).
    - ``max_batch``   — max inputs per request (provider-dependent cap).

    Defaults reproduce the original OpenAI behavior exactly, so existing setups
    that configure only a model keep working.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        base_url: str | None = None,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        dimensions: int | None = None,
        max_batch: int | None = None,
    ) -> None:
        self.model_name = model_name or DEFAULT_MODEL
        self.base_url = base_url or None
        self.api_key_env = api_key_env or DEFAULT_API_KEY_ENV
        self._dimensions = dimensions
        self.max_batch = int(max_batch) if max_batch else _MAX_BATCH
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

                    logger.info(
                        "creating OpenAI-compatible client for embeddings "
                        "(model=%s, base_url=%s, api_key_env=%s)",
                        self.model_name,
                        self.base_url or "default",
                        self.api_key_env,
                    )
                    # The SDK retries 408/409/429/5xx with exponential backoff;
                    # bump from the default 2 so a transient 500 during a long
                    # ingest doesn't abort the whole run.
                    kwargs: Dict[str, Any] = {"max_retries": 6, "timeout": 60.0}
                    # Pass the key explicitly when the configured env var is set;
                    # otherwise let the SDK fall back to its own OPENAI_API_KEY
                    # lookup (preserves the original default behavior).
                    api_key = os.environ.get(self.api_key_env)
                    if api_key:
                        kwargs["api_key"] = api_key
                    if self.base_url:
                        kwargs["base_url"] = self.base_url
                    self._client = OpenAI(**kwargs)
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
        for start in range(0, len(texts), self.max_batch):
            # OpenAI rejects empty strings; substitute a single space so row
            # ordering is preserved 1:1 with the response.
            batch = [t if t else " " for t in texts[start : start + self.max_batch]]
            kwargs: dict = {"model": self.model_name, "input": batch}
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions
            response = self.client.embeddings.create(**kwargs)
            for item in response.data:
                out.append([float(x) for x in item.embedding])
        return out


# Backwards-compatible alias: the class was historically named OpenAIEmbedder.
OpenAIEmbedder = OpenAICompatibleEmbedder


def embedder_from_config(embedding_cfg: Dict[str, Any] | None) -> OpenAICompatibleEmbedder:
    """Build an embedder from a ``plugins.lancedb.embedding`` config block.

    Single place that maps config keys to constructor args, shared by the
    provider, setup warmup, and the benchmark so they can't drift.
    """
    cfg = embedding_cfg or {}
    return OpenAICompatibleEmbedder(
        cfg.get("model", DEFAULT_MODEL),
        base_url=cfg.get("base_url") or None,
        api_key_env=cfg.get("api_key_env") or DEFAULT_API_KEY_ENV,
        dimensions=cfg.get("dimensions"),
        max_batch=cfg.get("max_batch"),
    )
