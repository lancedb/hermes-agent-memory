from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_embeddings_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "lancedb_embeddings_under_test", root / "src" / "embeddings.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


emb_mod = _load_embeddings_module()


class _FakeEmbeddingsAPI:
    def __init__(self):
        self.batches: list[list[str]] = []

    def create(self, *, model, input, **kwargs):
        self.batches.append(list(input))

        class _Item:
            def __init__(self, embedding):
                self.embedding = embedding

        class _Resp:
            pass

        resp = _Resp()
        # Deterministic 2-dim vectors; order preserved 1:1 with input.
        resp.data = [_Item([float(len(t)), 0.5]) for t in input]
        return resp


class _FakeClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddingsAPI()


def _embedder():
    e = emb_mod.OpenAIEmbedder("text-embedding-3-small")
    e._client = _FakeClient()  # inject so no network / OPENAI_API_KEY needed
    return e


def test_known_model_dim_is_offline():
    e = emb_mod.OpenAIEmbedder("text-embedding-3-small")
    # No client injected: dim must resolve without any API call.
    assert e.dim == 1536
    assert e.warm() == 1536


def test_embed_preserves_order_and_handles_empty_input():
    e = _embedder()
    assert e.embed([]) == []
    vectors = e.embed(["alice", "bob", ""])
    assert len(vectors) == 3
    # Empty string is substituted with a single space before the API call.
    assert e._client.embeddings.batches[0][2] == " "
    assert vectors[0] == [5.0, 0.5]  # len("alice") == 5


def test_embed_chunks_large_inputs():
    e = _embedder()
    texts = [f"t{i}" for i in range(emb_mod._MAX_BATCH * 2 + 5)]
    vectors = e.embed(texts)
    assert len(vectors) == len(texts)
    # 2 full batches + 1 remainder = 3 API calls.
    assert len(e._client.embeddings.batches) == 3


def test_embed_one_returns_single_vector():
    e = _embedder()
    assert e.embed_one("hello") == [5.0, 0.5]


def test_default_max_batch_is_provider_safe():
    # The default must not exceed the smallest common provider cap we claim to
    # support out of the box (Gemini = 100). Larger batches 400 on Gemini.
    assert emb_mod._MAX_BATCH <= 100
    assert emb_mod.OpenAICompatibleEmbedder("m").max_batch == emb_mod._MAX_BATCH


def test_max_batch_is_configurable_and_chunks_accordingly():
    e = emb_mod.OpenAICompatibleEmbedder("m", max_batch=10)
    e._client = _FakeClient()
    texts = [f"t{i}" for i in range(25)]  # 10 + 10 + 5 -> 3 requests
    vectors = e.embed(texts)
    assert len(vectors) == 25
    assert len(e._client.embeddings.batches) == 3
    assert [len(b) for b in e._client.embeddings.batches] == [10, 10, 5]


def test_embedder_from_config_threads_max_batch():
    e = emb_mod.embedder_from_config({"model": "m", "max_batch": 96})
    assert e.max_batch == 96
    # Omitted -> falls back to the provider-safe default.
    assert emb_mod.embedder_from_config({"model": "m"}).max_batch == emb_mod._MAX_BATCH


def test_backward_compatible_alias():
    # The class was historically OpenAIEmbedder; the alias must still resolve to
    # the new implementation so existing imports keep working.
    assert emb_mod.OpenAIEmbedder is emb_mod.OpenAICompatibleEmbedder


def test_embedder_from_config_wires_endpoint_key_and_dims():
    e = emb_mod.embedder_from_config(
        {
            "model": "nomic-embed-text",
            "base_url": "http://localhost:11434/v1",
            "api_key_env": "MY_EMBED_KEY",
            "dimensions": 768,
        }
    )
    assert e.model_name == "nomic-embed-text"
    assert e.base_url == "http://localhost:11434/v1"
    assert e.api_key_env == "MY_EMBED_KEY"
    assert e._dimensions == 768


def test_embedder_from_config_defaults_match_openai():
    # An empty/partial block must reproduce the original OpenAI defaults so
    # existing setups that configure only a model are unaffected.
    e = emb_mod.embedder_from_config({})
    assert e.model_name == emb_mod.DEFAULT_MODEL
    assert e.base_url is None
    assert e.api_key_env == "OPENAI_API_KEY"


def test_client_uses_configured_base_url_and_key(monkeypatch):
    monkeypatch.setenv("MY_EMBED_KEY", "test-key-123")
    e = emb_mod.OpenAICompatibleEmbedder(
        "nomic-embed-text",
        base_url="http://localhost:11434/v1",
        api_key_env="MY_EMBED_KEY",
    )
    client = e.client  # constructs the real OpenAI-compatible client
    assert client.api_key == "test-key-123"
    assert str(client.base_url).rstrip("/") == "http://localhost:11434/v1"
