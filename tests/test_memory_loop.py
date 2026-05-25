from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_plugin():
    root = Path(__file__).resolve().parents[1]
    hermes_agent = root.parent / "hermes-agent"
    if hermes_agent.exists():
        sys.path.insert(0, str(hermes_agent))
    module_name = "lancedb_memory_test_plugin"
    spec = importlib.util.spec_from_file_location(
        module_name,
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeEmbedder:
    dim = 4

    def embed(self, texts):
        rows = []
        for text in texts:
            seed = sum(ord(ch) for ch in text)
            rows.append([float((seed + i) % 17) / 17.0 for i in range(self.dim)])
        return rows

    def embed_one(self, text):
        return self.embed([text])[0]


def test_remember_recall_read_and_forget(tmp_path):
    plugin = _load_plugin()
    provider = plugin.LanceDBMemoryProvider()
    provider._embedder = FakeEmbedder()
    provider.initialize(
        "session-1",
        hermes_home=str(tmp_path),
        platform="cli",
        agent_context="primary",
        agent_identity="test",
        agent_workspace="workspace-a",
    )

    remembered = json.loads(
        provider.handle_tool_call(
            "lancedb_remember",
            {
                "content": "The user prefers pytest over unittest.",
                "category": "preference",
                "tags": ["testing"],
            },
        )
    )
    assert remembered["status"] == "stored"

    recalled = json.loads(
        provider.handle_tool_call("lancedb_recall", {"query": "testing framework preference"})
    )
    assert recalled["total"] >= 1
    memory_id = recalled["results"][0]["id"]

    read = json.loads(
        provider.handle_tool_call(
            "lancedb_read",
            {"id": memory_id, "include_provenance": True},
        )
    )
    assert read["memory"]["content"] == "The user prefers pytest over unittest."

    preview = json.loads(
        provider.handle_tool_call(
            "lancedb_forget",
            {"action": "preview", "query": "pytest preference"},
        )
    )
    assert preview["candidates"]

    deleted = json.loads(
        provider.handle_tool_call("lancedb_forget", {"action": "delete", "id": memory_id})
    )
    assert deleted["deleted"]["id"] == memory_id

    provider.shutdown()


def test_auto_compaction_runs_after_threshold(tmp_path):
    plugin = _load_plugin()
    store_mod = importlib.import_module(f"{plugin.__name__}.store")
    store = store_mod.LanceDBStore(
        tmp_path,
        FakeEmbedder(),
        optimize_every_commits=5,
        cleanup_older_than_days=7,
    )
    store.open()
    sentinel = store._optimize_state_path
    assert not sentinel.exists()

    for i in range(6):
        store.add_row(
            {
                "kind": "fact",
                "content": f"row {i}",
                "agent_workspace": "ws",
            }
        )

    # Optimize runs in a daemon thread; wait briefly for it to land.
    deadline = __import__("time").time() + 5.0
    while __import__("time").time() < deadline and not sentinel.exists():
        __import__("time").sleep(0.05)

    assert sentinel.exists(), "auto-compaction sentinel file not written"
    assert int(sentinel.read_text()) >= 5


def test_reranker_cached_and_fetches_rerank_top_n(tmp_path, monkeypatch):
    """Cross-encoder reranker should be constructed once and see rerank_top_n candidates."""
    import lancedb.rerankers as lr

    construction_count = {"n": 0}
    rerank_seen = {"sizes": []}

    class StubReranker:
        def __init__(self, *, model_name, column):
            construction_count["n"] += 1
            self.model_name = model_name
            self.column = column

        def rerank_hybrid(self, query, vector_results, fts_results):
            import pyarrow as pa

            try:
                combined = pa.concat_tables([vector_results, fts_results])
            except Exception:
                combined = vector_results if len(vector_results) >= len(fts_results) else fts_results
            rerank_seen["sizes"].append(len(combined))
            return combined

        def rerank_vector(self, query, vector_results):
            rerank_seen["sizes"].append(len(vector_results))
            return vector_results

        def rerank_fts(self, query, fts_results):
            rerank_seen["sizes"].append(len(fts_results))
            return fts_results

    monkeypatch.setattr(lr, "CrossEncoderReranker", StubReranker)

    plugin = _load_plugin()
    provider = plugin.LanceDBMemoryProvider()
    provider._embedder = FakeEmbedder()
    provider.initialize(
        "session-rerank",
        hermes_home=str(tmp_path),
        platform="cli",
        agent_context="primary",
        agent_identity="test",
        agent_workspace="workspace-rr",
    )
    # Override config after initialize() — initialize() reloads from disk and
    # would otherwise stomp on these test values.
    provider._config["retrieval"].setdefault("reranker", {})
    provider._config["retrieval"]["reranker"]["type"] = "cross-encoder"
    provider._config["retrieval"]["reranker"]["rerank_top_n"] = 12

    # Seed enough rows that rerank_top_n < total rows.
    for i in range(20):
        provider.handle_tool_call(
            "lancedb_remember",
            {"content": f"fact about pytest item {i}", "category": "preference"},
        )

    # Multiple recall calls should reuse the cached reranker — not construct each time.
    for _ in range(3):
        provider.recall("pytest", mode="vector", kind="fact", limit=3)

    assert construction_count["n"] == 1, (
        f"expected reranker constructed once, got {construction_count['n']}"
    )
    # rerank_top_n=12 with limit=3 means the reranker should see 12 candidates.
    assert rerank_seen["sizes"], "stub reranker was never called"
    assert all(size >= 3 for size in rerank_seen["sizes"]), rerank_seen["sizes"]
    # At least one rerank pass should have received more than `limit` candidates.
    assert max(rerank_seen["sizes"]) > 3, rerank_seen["sizes"]

    provider.shutdown()
