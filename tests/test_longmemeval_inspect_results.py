from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_inspect_module():
    root = Path(__file__).resolve().parents[1]
    module_name = "longmemeval_inspect_results"
    spec = importlib.util.spec_from_file_location(
        module_name,
        root / "benchmarks" / "longmemeval" / "inspect_results.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


inspect = _load_inspect_module()


def test_inspect_results_filters_and_formats_expected_vs_actual(tmp_path):
    path = tmp_path / "cases.jsonl"
    rows = [
        {
            "question_id": "q1",
            "variant": "lancedb-hybrid-rrf",
            "correct": False,
            "judge_label": "no",
            "question_type": "single-session",
            "question": "Where was the offsite?",
            "reference_answer": "Paris",
            "model_answer": "Rome",
            "judge_output": "no",
            "answer_session_ids": ["s1"],
            "answer_turn_ids": ["t1"],
            "retrieval": {"session_hit": True, "turn_hit": True},
            "latency_s": {"total_s": 1.25},
            "retrieved": [
                {
                    "id": "t1",
                    "session_id": "s1",
                    "turn_index": 0,
                    "role": "user",
                    "score": 0.9,
                    "text": "Alice said the offsite was in Paris.",
                }
            ],
        },
        {
            "question_id": "q2",
            "variant": "lancedb-hybrid-rrf",
            "correct": True,
            "question_type": "single-session",
            "retrieval": {"session_hit": False, "turn_hit": False},
            "latency_s": {"retrieval_s": 2.0, "total_s": 3.0},
            "retrieved": [],
        },
        {
            "question_id": "q3",
            "variant": "openviking",
            "correct": True,
            "question_type": "single-session",
            "retrieval": {"session_hit": True, "turn_hit": False},
            "latency_s": {"retrieval_s": 4.0, "total_s": 6.0},
            "retrieved": [],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    args = inspect.parse_args([str(path), "--status", "incorrect", "--limit", "5"])
    loaded = inspect.load_rows(path)
    filtered = inspect.filter_rows(loaded, args)
    markdown = inspect.format_markdown(
        inspect.sample_rows(filtered, limit=5, seed=0),
        rows=loaded,
        filtered=filtered,
        args=args,
    )

    assert len(filtered) == 1
    assert "## Variant Summary" in markdown
    assert "| lancedb-hybrid-rrf | 2 | 0.500 | 0.500 | 0.500 | 1.00s | 2.12s |" in markdown
    assert "| openviking | 1 | 1.000 | 1.000 | 0.000 | 4.00s | 6.00s |" in markdown
    assert "## Sampled Records" in markdown
    assert "**Expected**" in markdown
    assert "Paris" in markdown
    assert "**Actual**" in markdown
    assert "Rome" in markdown
    assert "answer-session" in markdown
    assert "q2" not in markdown
