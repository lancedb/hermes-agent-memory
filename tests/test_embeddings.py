from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_embeddings_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("lancedb_embeddings_under_test", root / "embeddings.py")
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
