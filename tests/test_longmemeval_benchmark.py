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
        "lancedb-vector,lancedb-hybrid-rrf,lancedb-hybrid-cross-encoder"
    )

    assert [(v.name, v.backend, v.mode, v.reranker_type) for v in variants] == [
        ("lancedb-vector", "lancedb", "vector", "rrf"),
        ("lancedb-hybrid-rrf", "lancedb", "hybrid", "rrf"),
        ("lancedb-hybrid-cross-encoder", "lancedb", "hybrid", "cross-encoder"),
    ]


def test_expand_variants_all_and_default_are_lancedb_only():
    expected = list(lme.VARIANT_NAMES)
    assert [v.name for v in lme.expand_variants("all")] == expected
    assert [v.name for v in lme.expand_variants("")] == expected
    assert all(v.backend == "lancedb" for v in lme.expand_variants("all"))


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
