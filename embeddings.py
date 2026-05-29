"""Embedding helpers for LanceDB memory."""
from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, List

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    """Lazy OpenAI embedding wrapper with normalized float vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
        *,
        api_key: str = "",
        api_base: str = "",
        dimension: int | None = None,
    ) -> None:
        self.model_name = model_name or DEFAULT_OPENAI_EMBEDDING_MODEL
        self.api_key = _expand_optional_env(api_key)
        self.api_base = _expand_optional_env(api_base)
        self._dim = int(dimension or OPENAI_EMBEDDING_DIMS.get(self.model_name, 1536))
        self._client = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    load_default_env_files()
                    from openai import OpenAI

                    kwargs = {}
                    if self.api_key:
                        kwargs["api_key"] = self.api_key
                    if self.api_base:
                        kwargs["base_url"] = self.api_base
                    logger.info("loading OpenAI embedding client for %s", self.model_name)
                    self._client = OpenAI(**kwargs)
        return self._client

    def warm(self) -> int:
        """Return the configured embedding dimension without making a network call."""
        return self.dim

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
        }
        if self.model_name.startswith("text-embedding-3-"):
            kwargs["dimensions"] = self.dim
        response = self.client.embeddings.create(**kwargs)
        rows = sorted(response.data, key=lambda item: item.index)
        return [_normalize_vector(list(row.embedding)) for row in rows]


def create_embedder(config: dict[str, Any] | None = None) -> OpenAIEmbedder:
    cfg = dict(config or {})
    provider = str(cfg.get("provider") or "openai")
    if provider == "sentence-transformers":
        logger.warning(
            "Migrating legacy sentence-transformers embedding config to OpenAI %s",
            DEFAULT_OPENAI_EMBEDDING_MODEL,
        )
        cfg = {
            "provider": "openai",
            "model": DEFAULT_OPENAI_EMBEDDING_MODEL,
            "dimension": OPENAI_EMBEDDING_DIMS[DEFAULT_OPENAI_EMBEDDING_MODEL],
        }
        provider = "openai"
    if provider != "openai":
        raise ValueError(f"Unsupported embedding provider for LanceDB memory: {provider}")
    return OpenAIEmbedder(
        str(cfg.get("model") or DEFAULT_OPENAI_EMBEDDING_MODEL),
        api_key=str(cfg.get("api_key") or ""),
        api_base=str(cfg.get("api_base") or ""),
        dimension=cfg.get("dimension"),
    )


def load_default_env_files() -> None:
    """Load common Hermes/repo .env files without overriding process env."""
    paths = [
        Path.cwd() / ".env",
        Path.home() / ".hermes" / ".env",
    ]
    for path in paths:
        load_env_file(path)


def load_env_file(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        key, value = _parse_env_line(line)
        if key and key not in os.environ:
            os.environ[key] = value


def _parse_env_line(line: str) -> tuple[str, str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return "", ""
    if stripped.startswith("export "):
        stripped = stripped.removeprefix("export ").strip()
    if "=" not in stripped:
        return "", ""
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return "", ""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, os.path.expandvars(value)


def _expand_optional_env(value: str) -> str:
    expanded = os.path.expandvars(value or "")
    if expanded.startswith("${") and expanded.endswith("}"):
        return ""
    return expanded


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    if norm <= 0:
        return [float(v) for v in vector]
    return [float(v) / norm for v in vector]
