from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def _load_benchmark_module():
    root = Path(__file__).resolve().parents[1]
    module_name = "longmemeval_benchmark_run"
    spec = importlib.util.spec_from_file_location(
        module_name,
        root / "benchmarks" / "longmemeval" / "run.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


lme = _load_benchmark_module()


class FakeEmbedder:
    dim = 4

    def embed(self, texts):
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text):
        lower = text.lower()
        return [
            float(lower.count("paris")),
            float(lower.count("rome")),
            float(lower.count("alice")),
            float(len(lower) % 11),
        ]


def _raw_case():
    return {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "Where did Alice say the team offsite is?",
        "answer": "Paris",
        "question_date": "2024-01-03",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2024-01-01", "2024-01-02"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Alice said the team offsite is in Paris.", "has_answer": True},
                {"role": "assistant", "content": "I will remember the offsite location."},
            ],
            [
                {"role": "user", "content": "We discussed Rome for a different trip."},
                {"role": "assistant", "content": "Rome was for vacation planning."},
            ],
        ],
        "answer_session_ids": ["s1"],
    }


def test_parse_case_strips_has_answer_and_keeps_internal_turn_ids():
    case = lme.parse_case(_raw_case())

    assert case.answer_turn_ids
    assert "has_answer" not in case.haystack_sessions[0][0]
    assert case.haystack_sessions[0][0] == {
        "role": "user",
        "content": "Alice said the team offsite is in Paris.",
    }


def test_expand_variants_maps_expected_matrix():
    variants = lme.expand_variants(
        "hermes-builtin-memory,hermes-holographic,openviking,markdown-lexical,lancedb-vector,lancedb-hybrid-cross-encoder"
    )

    assert [(v.name, v.backend, v.mode, v.reranker_type) for v in variants] == [
        ("hermes-builtin-memory", "hermes-builtin", "", "rrf"),
        ("hermes-holographic", "holographic", "", "rrf"),
        ("openviking", "openviking", "", "rrf"),
        ("markdown-lexical", "markdown-lexical", "", "rrf"),
        ("lancedb-vector", "lancedb", "vector", "rrf"),
        ("lancedb-hybrid-cross-encoder", "lancedb", "hybrid", "cross-encoder"),
    ]


def test_markdown_ingestion_does_not_leak_labels_or_reference_answer(tmp_path):
    case = lme.parse_case(_raw_case())
    index = lme.MarkdownMemoryIndex(case, tmp_path)
    index.ingest()

    combined = "\n".join(path.read_text() for path in (tmp_path / "markdown-memory").glob("*.md"))
    assert "has_answer" not in combined
    assert "Reference answer" not in combined
    assert "Paris" in combined


def test_hermes_builtin_memory_ingestion_is_no_index_and_bounded(tmp_path):
    case = lme.parse_case(_raw_case())
    index = lme.HermesBuiltinMemoryIndex(case, tmp_path)
    index.ingest()

    memory_text = (tmp_path / "hermes-builtin-memory" / "memories" / "MEMORY.md").read_text()
    rows = index.retrieve("Alice Paris offsite", limit=1)

    assert "has_answer" not in memory_text
    assert len(memory_text) <= lme.BUILTIN_MEMORY_CHAR_LIMIT
    assert len(rows) > 1
    assert any(row.session_id == "s1" for row in rows)


def test_retrieval_metrics_marks_session_and_turn_hits():
    case = lme.parse_case(_raw_case())
    snippets = [
        lme.RetrievedSnippet(
            id=case.answer_turn_ids[0],
            session_id="s1",
            turn_index=0,
            role="user",
            date="2024-01-01",
            text="Alice said the team offsite is in Paris.",
        )
    ]

    metrics = lme.retrieval_metrics(snippets, case.answer_session_ids, case.answer_turn_ids)

    assert metrics["session_hit"] is True
    assert metrics["turn_hit"] is True


def test_judge_label_parser_accepts_yes_no_styles():
    assert lme.parse_judge_label("Yes.") == "yes"
    assert lme.parse_judge_label("NO - it does not match") == "no"
    assert lme.parse_judge_label("correct") == "yes"
    assert lme.parse_judge_label("unclear") == "unknown"


def test_holographic_fts_query_sanitizes_natural_questions():
    assert lme.holographic_fts_query("What degree did I graduate with?") == "degree* OR graduate*"
    assert lme.holographic_fts_query("Where do I take yoga classes?") == "take* OR yoga* OR classes*"


def test_run_benchmark_writes_jsonl_and_summary_with_mocked_llm(tmp_path):
    case = lme.parse_case(_raw_case())
    calls = []

    def fake_llm(**kwargs):
        calls.append(kwargs)
        if kwargs["task"] == "longmemeval_answer":
            assert "has_answer" not in json.dumps(kwargs["messages"])
            return {"text": "Paris", "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
        return {"text": "yes", "usage": {"prompt_tokens": 8, "completion_tokens": 1}}

    args = Namespace(
        output_dir=tmp_path,
        keep_temp=False,
        top_k=2,
        answer_provider="",
        answer_model="cheap-answer",
        judge_provider="",
        judge_model="judge-model",
        answer_max_tokens=64,
        judge_max_tokens=8,
        batch_size=4,
        temperature=0.0,
        quiet=True,
    )

    result = lme.run_benchmark(
        [case],
        lme.expand_variants("hermes-builtin-memory"),
        args,
        llm_call=fake_llm,
    )

    rows = [
        json.loads(line)
        for line in Path(result["jsonl_path"]).read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))

    assert len(calls) == 2
    assert len(rows) == 1
    assert rows[0]["correct"] is True
    assert rows[0]["retrieval"]["session_hit"] is True
    assert summary["cases"] == 1
    assert summary["by_variant"]["hermes-builtin-memory"]["accuracy"] == 1.0


def test_run_benchmark_batches_async_llm_calls(tmp_path):
    cases = []
    for i in range(4):
        raw = _raw_case()
        raw["question_id"] = f"q{i}"
        cases.append(lme.parse_case(raw))
    active = {"count": 0, "max": 0}

    async def fake_llm(**kwargs):
        active["count"] += 1
        active["max"] = max(active["max"], active["count"])
        await asyncio.sleep(0.01)
        active["count"] -= 1
        if kwargs["task"] == "longmemeval_answer":
            return {"text": "Paris"}
        return {"text": "yes"}

    args = Namespace(
        output_dir=tmp_path,
        keep_temp=False,
        top_k=2,
        answer_provider="",
        answer_model="cheap-answer",
        judge_provider="",
        judge_model="judge-model",
        answer_max_tokens=64,
        judge_max_tokens=8,
        batch_size=4,
        temperature=0.0,
        quiet=True,
    )

    lme.run_benchmark(cases, lme.expand_variants("hermes-builtin-memory"), args, llm_call=fake_llm)

    assert active["max"] > 1


def test_single_variant_progress_prefix_omits_variant_counter():
    case = lme.parse_case(_raw_case())
    variant = lme.expand_variants("lancedb-hybrid-rrf")[0]

    prefix = lme.format_progress_prefix(
        completed=1,
        total=10,
        case_index=1,
        case_total=10,
        case=case,
        variant_index=1,
        variant_total=1,
        variant=variant,
    )

    assert "variant 1/1" not in prefix
    assert prefix == "[1/10] case 1/10 q1"


def test_lancedb_turn_ingestion_is_queryable_with_session_ids(tmp_path):
    case = lme.parse_case(_raw_case())
    variant = lme.expand_variants("lancedb-vector")[0]
    index = lme.LanceDBCaseIndex(case, tmp_path, variant, embedder=FakeEmbedder())
    try:
        index.ingest()
        rows = index.retrieve("Alice Paris offsite", limit=2)
    finally:
        index.close()

    assert rows
    assert any(row.session_id == "s1" for row in rows)


def test_holographic_turn_ingestion_is_queryable_with_session_ids(tmp_path):
    case = lme.parse_case(_raw_case())
    index = lme.HolographicCaseIndex(case, tmp_path)
    try:
        index.ingest()
        rows = index.retrieve("Alice Paris offsite", limit=2)
        assert index.provider._store.hrr_dim == lme.HOLOGRAPHIC_BENCHMARK_HRR_DIM
    finally:
        index.close()

    assert rows
    assert any(row.session_id == "s1" for row in rows)


class _FakeResponse:
    def __init__(self, data):
        self._data = data
        self.status_code = 200

    def json(self):
        return self._data


class _FakeHttpx:
    """Minimal httpx stand-in routing OpenVikingCaseIndex._long_request calls."""

    def __init__(self, store):
        self._store = store

    def request(self, method, url, *, json=None, params=None, headers=None, timeout=None):
        if method == "DELETE" and url.endswith("/api/v1/fs"):
            prefix = params["uri"]
            for uri in [u for u in self._store if u.startswith(prefix)]:
                del self._store[uri]
            return _FakeResponse({"result": {"uri": prefix, "estimated_deleted_count": 0}})
        if method == "POST" and url.endswith("/api/v1/content/reindex"):
            # vectors_only reindex is what makes leaf chunks searchable; no-op for the fake.
            return _FakeResponse({"result": {"status": "completed", "mode": json.get("mode")}})
        if method == "POST" and url.endswith("/api/v1/system/wait"):
            return _FakeResponse({"result": {}})
        raise AssertionError(f"{method} {url}")


def _make_openviking_fakes(store):
    class FakeClient:
        _httpx = _FakeHttpx(store)

        def _url(self, path):
            return f"http://test{path}"

        def _headers(self):
            return {}

        def _parse_response(self, resp):
            return resp.json()

        def post(self, path, payload):
            if path == "/api/v1/content/write":
                assert payload["wait"] is False
                store[payload["uri"]] = payload["content"]
                return {"result": {"written_bytes": len(payload["content"])}}
            if path == "/api/v1/search/find":
                assert "level" not in payload
                query_terms = lme.tokenize(payload["query"])
                rows = []
                for uri, content in store.items():
                    if not uri.startswith(payload["target_uri"]):
                        continue
                    score = lme.lexical_score(query_terms, content)
                    rows.append({"uri": uri, "score": score, "abstract": content[:120]})
                rows.sort(key=lambda row: row["score"], reverse=True)
                return {"result": {"memories": rows[: payload["limit"]], "total": len(rows)}}
            raise AssertionError(path)

        def get(self, path, **kwargs):
            if path == "/api/v1/content/read":
                return {"result": {"content": store.get(kwargs["params"]["uri"], "")}}
            raise AssertionError(path)

    class FakeProvider:
        def __init__(self):
            self._client = None
            self._user = "bench-user"

        def initialize(self, session_id):
            self._client = FakeClient()

        def shutdown(self):
            pass

    class FakePlugin:
        OpenVikingMemoryProvider = FakeProvider

    return FakePlugin


def test_openviking_turn_ingestion_is_queryable_with_session_ids(tmp_path, monkeypatch):
    case = lme.parse_case(_raw_case())
    store = {}

    monkeypatch.setattr(lme, "load_openviking_plugin", lambda: _make_openviking_fakes(store))
    index = lme.OpenVikingCaseIndex(case, tmp_path)
    try:
        index.ingest()
        rows = index.retrieve("Alice Paris offsite", limit=2)
    finally:
        index.close()

    assert index.scope == "viking://user/bench-user/memories/longmemeval-tpd4-q1/"
    assert rows
    assert any(row.session_id == "s1" for row in rows)


def test_extract_openviking_hits_accepts_nested_response_shapes():
    payload = {
        "result": {
            "nodes": [
                {"uri": "viking://user/default/a.md", "score": 0.2},
                {"uri": "viking://user/default/b.md", "score": 0.9},
            ],
            "relations": [{"uri": "not-viking"}],
        }
    }

    hits = lme.extract_openviking_hits(payload)

    assert [hit["uri"] for hit in hits] == [
        "viking://user/default/b.md",
        "viking://user/default/a.md",
    ]


def test_openviking_reingestion_is_idempotent(tmp_path, monkeypatch):
    """Re-running ingest wipes the per-case scope first, so it does not error or
    accumulate duplicate chunk documents."""
    case = lme.parse_case(_raw_case())
    store = {}

    monkeypatch.setattr(lme, "load_openviking_plugin", lambda: _make_openviking_fakes(store))
    first = lme.OpenVikingCaseIndex(case, tmp_path)
    second = lme.OpenVikingCaseIndex(case, tmp_path)
    try:
        first.ingest()
        chunk_count_after_first = len(store)
        assert chunk_count_after_first > 0
        second.ingest()
    finally:
        first.close()
        second.close()

    assert len(store) == chunk_count_after_first
    assert all(uri.startswith(first.scope) for uri in store)
    # _raw_case has 4 turns; at the default 4 turns/doc that packs into one chunk.
    assert len(store) == 1
    assert store[next(iter(store))].count(lme.ENTRY_DELIMITER) == 3
