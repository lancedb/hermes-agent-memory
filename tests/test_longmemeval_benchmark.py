from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from argparse import Namespace
from collections import Counter
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
        "lancedb-vector,lancedb-hybrid-rrf,lancedb-hybrid-cross-encoder"
    )

    assert [(v.name, v.backend, v.mode, v.reranker_type) for v in variants] == [
        ("lancedb-vector", "lancedb", "vector", "rrf"),
        ("lancedb-hybrid-rrf", "lancedb", "hybrid", "rrf"),
        ("lancedb-hybrid-cross-encoder", "lancedb", "hybrid", "cross-encoder"),
    ]


def test_expand_variants_all_and_default_match_default_sweep():
    expected = list(lme.VARIANT_NAMES)
    assert [v.name for v in lme.expand_variants("all")] == expected
    assert [v.name for v in lme.expand_variants("")] == expected
    # The default sweep is the session-search baseline plus the LanceDB modes;
    # full-context is recognized but excluded from the default sweep.
    assert "hermes-session-search" in expected
    assert "full-context" not in expected


def test_expand_variants_maps_baseline_backends():
    by_name = {v.name: v for v in lme.expand_variants("hermes-session-search,full-context")}
    assert by_name["hermes-session-search"].backend == "hermes-session-search"
    assert by_name["full-context"].backend == "full-context"


def test_expand_variants_linear_is_vector_biased_hybrid():
    v = lme.expand_variants("lancedb-hybrid-linear")[0]
    assert v.backend == "lancedb"
    assert v.mode == "hybrid"
    assert v.reranker_type == "linear"
    assert v.reranker_weight == 0.85  # 0.85 vector / 0.15 FTS


def test_build_fts_query_is_disjunctive_and_deduped():
    query = lme.build_fts_query("Where did Alice say the OFFSITE is, Alice?")
    terms = query.split(" OR ")
    assert "alice" in terms
    assert "offsite" in terms
    assert terms.count("alice") == 1  # deduped despite appearing twice
    assert " OR " in query  # disjunctive, not implicit-AND


def test_parse_variant_limits_parses_and_validates():
    limits = lme.parse_variant_limits(["full-context=1", "lancedb-vector=5"])
    assert limits == {"full-context": 1, "lancedb-vector": 5}

    import pytest

    with pytest.raises(ValueError):
        lme.parse_variant_limits(["full-context"])  # missing =N
    with pytest.raises(ValueError):
        lme.parse_variant_limits(["full-context=x"])  # non-int
    with pytest.raises(ValueError):
        lme.parse_variant_limits(["ghost=1"], {"full-context"})  # unknown variant


def test_full_context_index_returns_every_turn():
    case = lme.parse_case(_raw_case())
    variant = lme.expand_variants("full-context")[0]
    index = lme.build_case_index(case, Path("/tmp"), variant, lme.BenchmarkResources())
    index.ingest()
    snippets = index.retrieve(case.question, limit=1)  # limit ignored
    total_turns = sum(len(s) for s in case.haystack_sessions)
    assert len(snippets) == total_turns
    assert any(s.session_id == "s1" for s in snippets)


def test_expand_variants_rejects_unknown_variant():
    import pytest

    with pytest.raises(ValueError):
        lme.expand_variants("openviking")


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
    assert metrics["turn_recall"] == 1.0  # the single gold turn was retrieved
    assert metrics["reciprocal_rank"] == 1.0  # gold turn at rank 1


def test_retrieval_metrics_mrr_and_partial_recall():
    # Two gold turns; one retrieved at rank 2, the other not retrieved.
    gold = ["turn_a", "turn_b"]

    def snip(turn_id):
        return lme.RetrievedSnippet(
            id=turn_id, session_id="s1", turn_index=0, role="user", date="", text="x"
        )

    snippets = [snip("turn_x"), snip("turn_a"), snip("turn_y")]
    metrics = lme.retrieval_metrics(snippets, ["s1"], gold)

    assert metrics["turn_hit"] is True
    assert metrics["turn_recall"] == 0.5  # 1 of 2 gold turns retrieved
    assert metrics["reciprocal_rank"] == 0.5  # first gold turn at rank 2
    assert metrics["gold_turn_count"] == 2


def test_retrieval_metrics_none_when_no_gold_turns():
    snippets = [
        lme.RetrievedSnippet(id="t1", session_id="s1", turn_index=0, role="user", date="", text="x")
    ]
    metrics = lme.retrieval_metrics(snippets, [], [])
    assert metrics["turn_hit"] is None
    assert metrics["turn_recall"] is None
    assert metrics["reciprocal_rank"] is None


def test_summary_includes_mrr_recall_and_per_type():
    k = 5

    def rec(variant, qtype, correct, rr, recall):
        return {
            "variant": variant,
            "question_type": qtype,
            "correct": correct,
            "overflow": False,
            "retrieval": {
                "turn_hit": recall > 0,
                "session_hit": True,
                "turn_recall": recall,
                "session_recall": 1.0,
                "reciprocal_rank": rr,
            },
            "latency_s": {"retrieval_s": 0.1, "answer_s": 0.2, "judge_s": 0.1, "total_s": 0.4},
            "usage": {"answer": {"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}, "judge": {}},
        }

    records = [
        rec("lancedb-vector", "single-session-user", True, 1.0, 1.0),
        rec("lancedb-vector", "multi-session", False, 0.5, 0.5),
    ]
    args = Namespace(top_k=k, answer_provider="", answer_model="m", judge_provider="", judge_model="j")
    summary = lme.build_summary(
        records,
        cases=[],
        variants=lme.expand_variants("lancedb-vector"),
        args=args,
        started_at="a",
        finished_at="b",
    )
    v = summary["by_variant"]["lancedb-vector"]
    assert v[f"mrr@{k}"] == 0.75  # mean(1.0, 0.5)
    assert v[f"turn_recall@{k}"] == 0.75
    # Both cases retrieved >=1 gold turn, so hit-rate is 1.0 even though graded
    # recall (0.75) and MRR (0.75) differ — exactly why the graded metrics matter.
    assert v[f"retrieval_turn_hit@{k}"] == 1.0
    bt = v["by_question_type"]
    assert bt["single-session-user"]["accuracy"] == 1.0
    assert bt["multi-session"]["accuracy"] == 0.0
    assert bt["multi-session"][f"mrr@{k}"] == 0.5

    md = lme.format_summary_markdown(summary)
    assert "Accuracy by question type" in md
    assert f"MRR@{k}" in md
    assert f"Recall@{k}" in md


def test_load_cases_stratified_per_type(tmp_path):
    data = []
    for qtype, n in [("single-session-user", 4), ("multi-session", 4), ("temporal-reasoning", 2)]:
        for i in range(n):
            data.append(
                {
                    "question_id": f"{qtype}-{i}",
                    "question_type": qtype,
                    "question": "q",
                    "answer": "a",
                    "haystack_session_ids": ["s1"],
                    "haystack_dates": ["d"],
                    "haystack_sessions": [[{"role": "user", "content": "c"}]],
                    "answer_session_ids": ["s1"],
                }
            )
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    cases = lme.load_cases(path, per_type=2)
    counts = Counter(c.question_type for c in cases)
    assert counts == Counter(
        {"single-session-user": 2, "multi-session": 2, "temporal-reasoning": 2}
    )


def test_caching_embedder_dedupes_across_calls_and_resets():
    class CountingEmbedder:
        dim = 2

        def __init__(self):
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return [[float(len(t)), 0.0] for t in texts]

        def embed_one(self, text):
            return self.embed([text])[0]

        def warm(self):
            return self.dim

    inner = CountingEmbedder()
    cache = lme.CachingEmbedder(inner)

    first = cache.embed(["x", "y", "x"])  # x deduped within the call (ingest path)
    cache.embed(["x", "z"])  # x served from cache; only z is new
    assert first[0] == first[2]
    assert inner.calls == [["x", "y"], ["z"]]  # x never re-embedded
    assert cache.api_texts == 3  # x, y, z each sent exactly once
    assert cache.dim == 2

    cache.reset()
    cache.embed(["x"])  # after a per-case reset, x is embedded again
    assert inner.calls == [["x", "y"], ["z"], ["x"]]


def test_caching_embedder_does_not_cache_queries():
    # embed_one is the query path: never cached, so each variant pays its own
    # query embedding and query latency stays comparable across variants.
    class CountingEmbedder:
        dim = 2

        def __init__(self):
            self.one_calls = []

        def embed(self, texts):
            return [[float(len(t)), 0.0] for t in texts]

        def embed_one(self, text):
            self.one_calls.append(text)
            return [float(len(text)), 0.0]

        def warm(self):
            return self.dim

    inner = CountingEmbedder()
    cache = lme.CachingEmbedder(inner)
    cache.embed_one("same query")
    cache.embed_one("same query")  # NOT served from cache
    assert inner.one_calls == ["same query", "same query"]  # embedded both times


def test_judge_label_parser_accepts_yes_no_styles():
    assert lme.parse_judge_label("Yes.") == "yes"
    assert lme.parse_judge_label("NO - it does not match") == "no"
    assert lme.parse_judge_label("correct") == "yes"
    assert lme.parse_judge_label("unclear") == "unknown"


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
        lme.expand_variants("lancedb-vector"),
        args,
        llm_call=fake_llm,
        embedder=FakeEmbedder(),
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
    assert summary["by_variant"]["lancedb-vector"]["accuracy"] == 1.0


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

    lme.run_benchmark(
        cases,
        lme.expand_variants("lancedb-vector"),
        args,
        llm_call=fake_llm,
        embedder=FakeEmbedder(),
    )

    assert active["max"] > 1


def test_run_benchmark_applies_per_variant_limits(tmp_path):
    cases = []
    for i in range(2):
        raw = _raw_case()
        raw["question_id"] = f"q{i}"
        cases.append(lme.parse_case(raw))

    def fake_llm(**kwargs):
        if kwargs["task"] == "longmemeval_answer":
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
        variant_limit=["full-context=1"],
    )

    result = lme.run_benchmark(
        cases,
        lme.expand_variants("lancedb-vector,full-context"),
        args,
        llm_call=fake_llm,
        embedder=FakeEmbedder(),
    )

    rows = [
        json.loads(line)
        for line in Path(result["jsonl_path"]).read_text(encoding="utf-8").splitlines()
    ]
    counts = Counter(r["variant"] for r in rows)
    assert counts["lancedb-vector"] == 2  # runs the full set
    assert counts["full-context"] == 1  # capped

    summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))
    assert "tokens_per_question" in summary["by_variant"]["full-context"]
    assert "latency_s_p95" in summary["by_variant"]["lancedb-vector"]


def test_full_context_overflow_is_recorded_without_llm_calls(tmp_path):
    case = lme.parse_case(_raw_case())
    calls = []

    def fake_llm(**kwargs):
        calls.append(kwargs)
        return {"text": "Paris", "usage": {"prompt_tokens": 10, "completion_tokens": 1}}

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
        variant_limit=[],
        full_context_token_budget=1,  # force overflow on the tiny synthetic case
    )

    result = lme.run_benchmark(
        [case],
        lme.expand_variants("full-context"),
        args,
        llm_call=fake_llm,
        embedder=FakeEmbedder(),
    )

    rows = [
        json.loads(line)
        for line in Path(result["jsonl_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert calls == []  # no LLM calls spent on an overflow case
    assert rows[0]["overflow"] is True
    assert rows[0]["correct"] is False
    assert rows[0]["prompt_tokens_est"] > 1
    summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))
    assert summary["by_variant"]["full-context"]["overflow_cases"] == 1


def test_hermes_session_search_index_ingests_and_retrieves(tmp_path):
    import pytest

    lme.ensure_hermes_state_importable()
    try:
        from hermes_state import SessionDB  # noqa: F401
    except Exception:
        pytest.skip("hermes-agent (hermes_state.SessionDB) not importable in this environment")

    case = lme.parse_case(_raw_case())
    variant = lme.expand_variants("hermes-session-search")[0]
    index = lme.build_case_index(case, tmp_path, variant, lme.BenchmarkResources())
    try:
        index.ingest()
        snippets = index.retrieve("Alice offsite Paris", limit=5)
    finally:
        index.close()

    assert snippets
    assert any(s.session_id == "s1" for s in snippets)
    # Gold turn id must be recoverable so the turn-hit metric works.
    assert any(s.id == case.answer_turn_ids[0] for s in snippets)


def test_run_benchmark_skips_case_on_ingest_error_without_aborting(tmp_path):
    # A transient embedding failure during ingest must skip that case/variant
    # and let the run finish, not crash the whole benchmark.
    case = lme.parse_case(_raw_case())

    class RaisingEmbedder:
        dim = 4

        def warm(self):
            return self.dim

        def embed(self, texts):
            raise RuntimeError("simulated API 500")

        def embed_one(self, text):
            raise RuntimeError("simulated API 500")

    def fake_llm(**kwargs):
        return {"text": "Paris"} if kwargs["task"] == "longmemeval_answer" else {"text": "yes"}

    args = Namespace(
        output_dir=tmp_path,
        keep_temp=False,
        top_k=2,
        answer_provider="",
        answer_model="m",
        judge_provider="",
        judge_model="j",
        answer_max_tokens=64,
        judge_max_tokens=8,
        batch_size=4,
        temperature=0.0,
        quiet=True,
        variant_limit=[],
        full_context_token_budget=0,
    )

    result = lme.run_benchmark(
        [case],
        lme.expand_variants("lancedb-vector"),
        args,
        llm_call=fake_llm,
        embedder=RaisingEmbedder(),
    )

    summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))
    assert summary["skipped_runs"]  # the failed run was recorded, not raised
    assert summary["by_variant"]["lancedb-vector"]["cases"] == 0


def test_complete_prepared_batch_skips_failed_answer_call(tmp_path):
    # A failing answer call for one item must not abort the batch.
    cases = []
    for i in range(2):
        raw = _raw_case()
        raw["question_id"] = f"q{i}"
        raw["question"] = "boom" if i == 0 else "ok"
        cases.append(lme.parse_case(raw))

    def fake_llm(**kwargs):
        # Fail the answer call for the "boom" question only.
        if kwargs["task"] == "longmemeval_answer" and "boom" in json.dumps(kwargs["messages"]):
            raise RuntimeError("simulated answer 500")
        return {"text": "Paris"} if kwargs["task"] == "longmemeval_answer" else {"text": "yes"}

    args = Namespace(
        output_dir=tmp_path,
        keep_temp=False,
        top_k=2,
        answer_provider="",
        answer_model="m",
        judge_provider="",
        judge_model="j",
        answer_max_tokens=64,
        judge_max_tokens=8,
        batch_size=4,
        temperature=0.0,
        quiet=True,
        variant_limit=[],
        full_context_token_budget=0,
    )

    result = lme.run_benchmark(
        cases,
        lme.expand_variants("lancedb-vector"),
        args,
        llm_call=fake_llm,
        embedder=FakeEmbedder(),
    )
    rows = [
        json.loads(line)
        for line in Path(result["jsonl_path"]).read_text(encoding="utf-8").splitlines()
    ]
    # One case answered fine; the failing one was skipped (not crashed).
    assert len(rows) == 1
    assert rows[0]["question_id"] == "q1"


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


def test_lancedb_hybrid_does_not_silently_fall_back_to_vector(tmp_path, caplog):
    import logging

    case = lme.parse_case(_raw_case())
    variant = lme.expand_variants("lancedb-hybrid-rrf")[0]
    index = lme.LanceDBCaseIndex(case, tmp_path, variant, embedder=FakeEmbedder())
    with caplog.at_level(logging.WARNING):
        try:
            index.ingest()
            rows = index.retrieve("Alice Paris offsite", limit=2)
        finally:
            index.close()

    assert rows
    # Regression: hybrid must fuse vector + FTS (RRF), not error on the
    # _relevance_score projection and quietly degrade to pure vector.
    assert "recall failed" not in caplog.text
    # RRF relevance scores are small (~0.01-0.03); an L2 _distance fallback
    # would be on a very different scale.
    assert rows[0].score is not None and rows[0].score < 1.0


def test_lancedb_hybrid_linear_fuses_and_produces_relevance_scores(tmp_path, caplog):
    import logging

    case = lme.parse_case(_raw_case())
    variant = lme.expand_variants("lancedb-hybrid-linear")[0]
    index = lme.LanceDBCaseIndex(case, tmp_path, variant, embedder=FakeEmbedder())
    with caplog.at_level(logging.WARNING):
        try:
            index.ingest()
            rows = index.retrieve("Alice Paris offsite", limit=2)
        finally:
            index.close()

    assert rows
    assert "recall failed" not in caplog.text  # didn't fall back to vector
    assert "linear reranker unavailable" not in caplog.text  # linear reranker constructed
    assert rows[0].score is not None  # a fused relevance score was returned


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
