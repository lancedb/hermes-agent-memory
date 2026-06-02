"""Run LongMemEval against the LanceDB memory plugin variants."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import importlib.util
import json
from json import JSONDecodeError
import re
import shutil
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable


DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
    "longmemeval_s_cleaned.json"
)
# Default sweep — the comparison set that is safe to run at scale: Hermes's
# built-in session search (FTS5 transcript retrieval) plus the LanceDB plugin
# retrieval modes. `full-context` is also a recognized variant (see
# expand_variants) but is deliberately excluded from the default sweep because
# it feeds the entire haystack to the answer model and is expensive — opt into
# it explicitly, ideally capped via `--variant-limit full-context=N`.
VARIANT_NAMES = (
    "hermes-session-search",
    "lancedb-vector",
    "lancedb-hybrid-rrf",
    "lancedb-hybrid-linear",
    "lancedb-hybrid-cross-encoder",
)
# Vector weight for the linear-combination hybrid variant (0.85 vector / 0.15
# FTS) — biased toward vector, since equal-weight RRF underperforms vector here.
LINEAR_VECTOR_WEIGHT = 0.85
# Recognized but not part of the default sweep.
EXTRA_VARIANT_NAMES = ("full-context",)


@dataclass(frozen=True)
class LongMemEvalCase:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_session_ids: list[str]
    haystack_dates: list[str]
    haystack_sessions: list[list[dict[str, str]]]
    answer_session_ids: list[str]
    answer_turn_ids: list[str]


@dataclass(frozen=True)
class Variant:
    name: str
    backend: str
    mode: str = ""
    reranker_type: str = "rrf"
    reranker_weight: float = 0.7  # linear fusion only: vector weight (0-1)


@dataclass(frozen=True)
class RetrievedSnippet:
    id: str
    session_id: str
    turn_index: int
    role: str
    date: str
    text: str
    score: float | None = None


@dataclass
class LlmResult:
    text: str
    latency_s: float
    usage: dict[str, int]
    provider: str
    model: str


@dataclass
class BenchmarkResources:
    embedder: Any = None
    rerankers: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedCaseVariant:
    case: LongMemEvalCase
    variant: Variant
    progress_prefix: str
    timings: dict[str, float]
    snippets: list[RetrievedSnippet]
    metrics: dict[str, Any]
    answer_messages: list[dict[str, str]]
    overflow: bool = False
    prompt_tokens_est: int = 0


_PLUGIN_MODULE: ModuleType | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("benchmarks/data/longmemeval_s_cleaned.json"),
        help="Path to longmemeval_s_cleaned.json.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the cleaned LongMemEval-S dataset if --dataset-path is missing.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Overwrite --dataset-path when downloading.",
    )
    parser.add_argument("--limit", type=int, default=25, help="Maximum cases to run (global ceiling).")
    parser.add_argument(
        "--per-type",
        type=int,
        default=0,
        help=(
            "Stratified sampling: take up to N cases of EACH question_type "
            "(deterministic, dataset order). Overrides --limit. The dataset has 6 "
            "types, so --per-type 3 ≈ 18 cases. Good for a balanced smoke test."
        ),
    )
    parser.add_argument(
        "--variant-limit",
        action="append",
        default=[],
        metavar="NAME=N",
        help=(
            "Per-variant case cap, e.g. --variant-limit full-context=1. Caps that "
            "variant below the global --limit (cannot exceed the loaded case count). "
            "Repeatable."
        ),
    )
    parser.add_argument("--offset", type=int, default=0, help="Skip this many matching cases first.")
    parser.add_argument(
        "--question-types",
        default="",
        help="Comma-separated LongMemEval question_type filter.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of snippets to retrieve.")
    parser.add_argument(
        "--variants",
        default=",".join(VARIANT_NAMES),
        help=(
            "Comma-separated variants, or 'all' for the default sweep. Choices: "
            + ", ".join(VARIANT_NAMES + EXTRA_VARIANT_NAMES)
            + "."
        ),
    )
    parser.add_argument("--answer-provider", default="", help="Explicit Hermes provider for answers.")
    parser.add_argument("--answer-model", default="", help="Explicit answer model.")
    parser.add_argument("--judge-provider", default="", help="Explicit Hermes provider for judging.")
    parser.add_argument("--judge-model", default="", help="Explicit judge model.")
    # Defaults sized for reasoning models: their hidden reasoning tokens count
    # against this budget, so a tight cap (the old 256/16) can leave no room for
    # the actual answer/verdict — a 16-token reasoning judge emits nothing and
    # every case scores "unknown". Non-reasoning models stop when done, so the
    # headroom costs nothing there.
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=1024,
        help="Max output tokens for the answer model (default 1024). Reasoning "
        "models spend reasoning tokens from this budget — keep it generous.",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=512,
        help="Max output tokens for the judge (default 512). Must be high enough "
        "for a reasoning judge to finish reasoning AND emit yes/no; too low (e.g. "
        "16) yields empty verdicts scored as 'unknown'.",
    )
    parser.add_argument(
        "--full-context-token-budget",
        type=int,
        default=0,
        help=(
            "If > 0, full-context cases whose estimated answer-prompt tokens "
            "(~chars/4) exceed this are recorded as overflow instead of calling "
            "the model. Set to your answer model's context window (with headroom). "
            "0 disables the guard (assumes a large-context model)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of case/variant answer and judge LLM calls to run concurrently.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/runs/dev"))
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep isolated Hermes homes under the output directory for inspection.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_benchmark_env()
    if args.download and (args.force_download or not args.dataset_path.exists()):
        log_progress(args, f"Downloading LongMemEval-S to {args.dataset_path} ...")
        download_dataset(args.dataset_path)
        log_progress(args, f"Downloaded {args.dataset_path} ({args.dataset_path.stat().st_size:,} bytes)")
    cases = load_cases(
        args.dataset_path,
        limit=args.limit,
        offset=args.offset,
        question_types=_split_csv(args.question_types),
        per_type=getattr(args, "per_type", 0) or None,
    )
    variants = expand_variants(args.variants)
    log_progress(
        args,
        f"Loaded {len(cases)} case(s); running {len(variants)} variant(s): "
        + ", ".join(v.name for v in variants),
    )
    output = run_benchmark(cases, variants, args)
    print(f"Wrote {output['jsonl_path']}")
    print(f"Wrote {output['summary_json_path']}")
    print(f"Wrote {output['summary_md_path']}")
    return 0


def download_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_URL) as response, path.open("wb") as out:
        shutil.copyfileobj(response, out)


def log_progress(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "quiet", False):
        return
    print(message, file=sys.stderr, flush=True)


def load_benchmark_env() -> None:
    """Best-effort load of .env files so OPENAI_API_KEY etc. are available.

    Existing environment variables win (we never overwrite). Self-contained so
    the harness has no python-dotenv dependency.
    """
    import os

    root = Path(__file__).resolve().parents[2]
    for path in (root / ".env", Path.home() / ".hermes" / ".env"):
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            continue


def load_cases(
    path: Path,
    *,
    limit: int | None = None,
    offset: int = 0,
    question_types: set[str] | None = None,
    per_type: int | None = None,
) -> list[LongMemEvalCase]:
    if limit == 0:
        return []
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Pass --download or place longmemeval_s_cleaned.json there."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        size = path.stat().st_size
        raise ValueError(
            f"Dataset file is not valid JSON: {path} ({size:,} bytes). "
            "This usually means the download was interrupted or saved as the wrong file. "
            "Remove it and re-run with --download, or use --download --force-download."
        ) from exc
    if not isinstance(raw, list):
        raise ValueError("LongMemEval dataset must be a JSON list of cases")

    out: list[LongMemEvalCase] = []
    skipped = 0
    per_type_counts: Counter[str] = Counter()
    for item in raw:
        case = parse_case(item)
        if question_types and case.question_type not in question_types:
            continue
        if skipped < max(0, offset):
            skipped += 1
            continue
        if per_type is not None and per_type > 0:
            # Stratified: cap per question_type; --limit does not apply.
            if per_type_counts[case.question_type] >= per_type:
                continue
            per_type_counts[case.question_type] += 1
            out.append(case)
            continue
        out.append(case)
        if limit is not None and limit > 0 and len(out) >= limit:
            break
    return out


def parse_case(item: dict[str, Any]) -> LongMemEvalCase:
    sessions: list[list[dict[str, str]]] = []
    answer_turn_ids: list[str] = []
    session_ids = [str(v) for v in item.get("haystack_session_ids") or []]
    raw_sessions = item.get("haystack_sessions") or []
    for session_index, raw_session in enumerate(raw_sessions):
        session_id = session_ids[session_index] if session_index < len(session_ids) else str(session_index)
        clean_session: list[dict[str, str]] = []
        for turn_index, raw_turn in enumerate(raw_session or []):
            role = str(raw_turn.get("role") or "").strip()
            content = str(raw_turn.get("content") or "")
            if raw_turn.get("has_answer") is True:
                answer_turn_ids.append(make_benchmark_turn_id(session_id, turn_index, role, content))
            clean_session.append({"role": role, "content": content})
        sessions.append(clean_session)

    return LongMemEvalCase(
        question_id=str(item.get("question_id") or ""),
        question_type=str(item.get("question_type") or ""),
        question=str(item.get("question") or ""),
        answer=str(item.get("answer") or ""),
        question_date=str(item.get("question_date") or ""),
        haystack_session_ids=session_ids,
        haystack_dates=[str(v) for v in item.get("haystack_dates") or []],
        haystack_sessions=sessions,
        answer_session_ids=[str(v) for v in item.get("answer_session_ids") or []],
        answer_turn_ids=answer_turn_ids,
    )


def expand_variants(value: str | Iterable[str]) -> list[Variant]:
    names = list(value) if not isinstance(value, str) else _split_csv(value)
    if not names:
        names = list(VARIANT_NAMES)
    # Expand the "all" token inline (the default sweep) so it can be combined
    # with extra variants, e.g. "all,full-context". Preserve order, dedupe.
    expanded: list[str] = []
    for name in names:
        for resolved in (VARIANT_NAMES if name == "all" else (name,)):
            if resolved not in expanded:
                expanded.append(resolved)
    names = expanded
    variants: list[Variant] = []
    for name in names:
        if name == "hermes-session-search":
            variants.append(Variant(name=name, backend="hermes-session-search"))
        elif name == "full-context":
            variants.append(Variant(name=name, backend="full-context"))
        elif name == "lancedb-vector":
            variants.append(Variant(name=name, backend="lancedb", mode="vector"))
        elif name == "lancedb-hybrid-rrf":
            variants.append(Variant(name=name, backend="lancedb", mode="hybrid", reranker_type="rrf"))
        elif name == "lancedb-hybrid-linear":
            variants.append(
                Variant(
                    name=name,
                    backend="lancedb",
                    mode="hybrid",
                    reranker_type="linear",
                    reranker_weight=LINEAR_VECTOR_WEIGHT,
                )
            )
        elif name == "lancedb-hybrid-cross-encoder":
            variants.append(
                Variant(
                    name=name,
                    backend="lancedb",
                    mode="hybrid",
                    reranker_type="cross-encoder",
                )
            )
        else:
            raise ValueError(f"Unknown LongMemEval variant: {name}")
    return variants


def run_benchmark(
    cases: list[LongMemEvalCase],
    variants: list[Variant],
    args: argparse.Namespace,
    *,
    llm_call: Callable[..., Any] | None = None,
    embedder: Any = None,
) -> dict[str, str]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "cases.jsonl"
    summary_json_path = args.output_dir / "summary.json"
    summary_md_path = args.output_dir / "summary.md"
    started_at = datetime.now(timezone.utc).isoformat()
    batch_size = max(1, int(getattr(args, "batch_size", 4) or 4))
    variant_limits = parse_variant_limits(getattr(args, "variant_limit", None), {v.name for v in variants})
    for name, cap in sorted(variant_limits.items()):
        if cap < len(cases):
            log_progress(args, f"Variant {name} capped to {cap} case(s) (global limit {len(cases)}).")
    work_items = [
        (case_index, case, variant_index, variant)
        for case_index, case in enumerate(cases, start=1)
        for variant_index, variant in enumerate(variants, start=1)
        if case_index <= variant_limits.get(variant.name, len(cases))
    ]
    resources = (
        build_benchmark_resources(variants, args, embedder=embedder)
        if work_items
        else BenchmarkResources(embedder=embedder)
    )

    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    total = len(work_items)
    current_case_index: int | None = None
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for batch_start in range(0, total, batch_size):
            batch = work_items[batch_start : batch_start + batch_size]
            prepared: list[PreparedCaseVariant] = []
            for offset, (case_index, case, variant_index, variant) in enumerate(batch, start=1):
                # Work items are case-major (all variants of a case are
                # consecutive), so resetting the embedding cache when the case
                # changes keeps the 3x within-case savings while bounding memory.
                if case_index != current_case_index:
                    reset = getattr(resources.embedder, "reset", None)
                    if callable(reset):
                        reset()
                    current_case_index = case_index
                completed = batch_start + offset
                progress_prefix = format_progress_prefix(
                    completed=completed,
                    total=total,
                    case_index=case_index,
                    case_total=len(cases),
                    case=case,
                    variant_index=variant_index,
                    variant_total=len(variants),
                    variant=variant,
                )
                log_progress(args, f"{progress_prefix}: starting")
                # Isolate per-(case,variant) failures (e.g. a transient API 500
                # during ingest) so one bad item is skipped and logged rather
                # than aborting the whole run and losing every other result.
                try:
                    prepared.append(
                        prepare_case_variant(
                            case,
                            variant,
                            args,
                            resources=resources,
                            progress_prefix=progress_prefix,
                        )
                    )
                except Exception as exc:
                    skipped.append(progress_prefix)
                    log_progress(args, f"{progress_prefix}: SKIPPED after error: {exc!r}")
            batch_records = asyncio.run(complete_prepared_batch(prepared, args, llm_call=llm_call))
            for record in batch_records:
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                handle.flush()

    if skipped:
        log_progress(
            args,
            f"WARNING: {len(skipped)}/{total} case/variant runs were skipped after errors: "
            + "; ".join(skipped),
        )

    summary = build_summary(
        records,
        cases=cases,
        variants=variants,
        args=args,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    summary["skipped_runs"] = skipped
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_md_path.write_text(format_summary_markdown(summary), encoding="utf-8")
    return {
        "jsonl_path": str(jsonl_path),
        "summary_json_path": str(summary_json_path),
        "summary_md_path": str(summary_md_path),
    }


def run_case_variant(
    case: LongMemEvalCase,
    variant: Variant,
    args: argparse.Namespace,
    *,
    llm_call: Callable[..., Any] | None = None,
    embedder: Any = None,
    progress_prefix: str = "",
) -> dict[str, Any]:
    resources = build_benchmark_resources([variant], args, embedder=embedder)
    prepared = prepare_case_variant(
        case,
        variant,
        args,
        resources=resources,
        progress_prefix=progress_prefix,
    )
    return asyncio.run(complete_prepared_batch([prepared], args, llm_call=llm_call))[0]


def prepare_case_variant(
    case: LongMemEvalCase,
    variant: Variant,
    args: argparse.Namespace,
    *,
    resources: BenchmarkResources,
    progress_prefix: str = "",
) -> PreparedCaseVariant:
    case_dir = args.output_dir / "stores" / variant.name / safe_name(case.question_id)
    if case_dir.exists() and not args.keep_temp:
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    # Time ingestion and retrieval separately. The two stages have very
    # different cost profiles and we report them apart:
    #   - ingest_s: building the store (embedding the whole haystack, indexing).
    #     One-time and amortized in production, so it is NOT in the headline
    #     latency. Across the three LanceDB variants of a case the haystack
    #     embedding is shared via the CachingEmbedder, so only the first
    #     variant pays it — fine, because ingest_s isn't the metric we compare.
    #   - query_s: the actual retrieval-stage latency a user feels per recall —
    #     embed the query + ANN/BM25 search + (for cross-encoder) the rerank
    #     pass. Each variant embeds its OWN query here (CachingEmbedder caches
    #     only the ingest/batch path, never embed_one), so query_s is fair and
    #     comparable across variants. The judge call is deliberately excluded:
    #     it's answer validation, downstream of and outside the retrieval stage.
    timings: dict[str, float] = {}
    log_progress(args, f"{progress_prefix}: ingesting and retrieving top-{args.top_k}")
    retriever = build_case_index(case, case_dir, variant, resources)
    try:
        ingest_start = time.perf_counter()
        retriever.ingest()
        timings["ingest_s"] = time.perf_counter() - ingest_start
        query_start = time.perf_counter()
        snippets = retriever.retrieve(case.question, limit=args.top_k)
        timings["query_s"] = time.perf_counter() - query_start
    finally:
        retriever.close()
    # retrieval_s kept for continuity = ingest + query; total_s (user-facing
    # latency) uses query_s only — ingest is a one-time, amortized cost.
    timings["retrieval_s"] = timings["ingest_s"] + timings["query_s"]
    log_progress(
        args,
        f"{progress_prefix}: query {timings['query_s']:.3f}s, {len(snippets)} snippet(s)",
    )
    metrics = retrieval_metrics(snippets, case.answer_session_ids, case.answer_turn_ids)

    if not args.keep_temp:
        shutil.rmtree(case_dir, ignore_errors=True)

    # Full-context feeds the whole haystack verbatim, so it must not truncate
    # individual turns the way the retrieval display safeguard does.
    max_snippet_chars = None if variant.backend == "full-context" else 1200
    answer_messages = build_answer_messages(case, snippets, max_snippet_chars=max_snippet_chars)

    prompt_tokens_est = estimate_message_tokens(answer_messages)
    budget = int(getattr(args, "full_context_token_budget", 0) or 0)
    overflow = (
        variant.backend == "full-context" and budget > 0 and prompt_tokens_est > budget
    )
    if overflow:
        log_progress(
            args,
            f"{progress_prefix}: OVERFLOW ~{prompt_tokens_est:,} est tokens > budget {budget:,}; "
            "recording without calling the model",
        )

    return PreparedCaseVariant(
        case=case,
        variant=variant,
        progress_prefix=progress_prefix,
        timings=timings,
        snippets=snippets,
        metrics=metrics,
        answer_messages=answer_messages,
        overflow=overflow,
        prompt_tokens_est=prompt_tokens_est,
    )


async def complete_prepared_batch(
    prepared: list[PreparedCaseVariant],
    args: argparse.Namespace,
    *,
    llm_call: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    if not prepared:
        return []

    # Overflow items (full-context prompts beyond the budget) are recorded
    # without spending any LLM calls.
    live = [item for item in prepared if not item.overflow]

    # return_exceptions=True: a transient API failure on one item is skipped
    # (logged) instead of aborting the whole batch and losing the run.
    for item in live:
        log_progress(args, f"{item.progress_prefix}: calling answer model")
    raw_answers = await asyncio.gather(
        *[
            call_model_async(
                task="longmemeval_answer",
                messages=item.answer_messages,
                provider=args.answer_provider,
                model=args.answer_model,
                max_tokens=args.answer_max_tokens,
                temperature=args.temperature,
                llm_call=llm_call,
            )
            for item in live
        ],
        return_exceptions=True,
    )
    answered: list[tuple[PreparedCaseVariant, LlmResult]] = []
    for item, answer in zip(live, raw_answers):
        if isinstance(answer, Exception):
            log_progress(args, f"{item.progress_prefix}: SKIPPED (answer call failed): {answer!r}")
            continue
        log_progress(args, f"{item.progress_prefix}: answer model returned in {answer.latency_s:.2f}s")
        answered.append((item, answer))

    for item, _ in answered:
        log_progress(args, f"{item.progress_prefix}: calling judge model")
    raw_judges = await asyncio.gather(
        *[
            call_model_async(
                task="longmemeval_judge",
                messages=build_judge_messages(item.case.question, item.case.answer, answer.text),
                provider=args.judge_provider,
                model=args.judge_model,
                max_tokens=args.judge_max_tokens,
                temperature=0.0,
                llm_call=llm_call,
            )
            for item, answer in answered
        ],
        return_exceptions=True,
    )

    answer_by_item: dict[int, LlmResult] = {}
    judge_by_item: dict[int, LlmResult] = {}
    for (item, answer), judge in zip(answered, raw_judges):
        if isinstance(judge, Exception):
            log_progress(args, f"{item.progress_prefix}: SKIPPED (judge call failed): {judge!r}")
            continue
        answer_by_item[id(item)] = answer
        judge_by_item[id(item)] = judge

    records = []
    for item in prepared:
        if item.overflow:
            answer = LlmResult(text="", latency_s=0.0, usage={}, provider=args.answer_provider or "", model=args.answer_model or "")
            judge = LlmResult(text="", latency_s=0.0, usage={}, provider=args.judge_provider or "", model=args.judge_model or "")
            label = "overflow"
        elif id(item) in answer_by_item:
            answer = answer_by_item[id(item)]
            judge = judge_by_item[id(item)]
            label = parse_judge_label(judge.text)
            log_progress(args, f"{item.progress_prefix}: judge returned {label} in {judge.latency_s:.2f}s")
        else:
            # answer or judge call failed for this item — already logged; skip.
            continue
        record = build_record(item, answer, judge, label, args)
        records.append(record)
        log_progress(
            args,
            f"{item.progress_prefix}: done "
            f"correct={record['correct']} overflow={record['overflow']} "
            f"total={record['latency_s']['total_s']:.2f}s",
        )
    return records


def build_record(
    prepared: PreparedCaseVariant,
    answer: LlmResult,
    judge: LlmResult,
    label: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    case = prepared.case
    return {
        "question_id": case.question_id,
        "question_type": case.question_type,
        "variant": prepared.variant.name,
        "top_k": args.top_k,
        "question": case.question,
        "question_date": case.question_date,
        "retrieved": [snippet_to_dict(s) for s in prepared.snippets],
        "reference_answer": case.answer,
        "model_answer": answer.text,
        "judge_output": judge.text,
        "judge_label": label,
        "correct": label == "yes",
        "overflow": prepared.overflow,
        "prompt_tokens_est": prepared.prompt_tokens_est,
        "answer_session_ids": case.answer_session_ids,
        "answer_turn_ids": case.answer_turn_ids,
        "retrieval": prepared.metrics,
        "latency_s": {
            **prepared.timings,
            "answer_s": answer.latency_s,
            # judge is benchmark-only grading overhead — recorded for
            # transparency but excluded from total_s (the user-facing latency).
            "judge_s": judge.latency_s,
            # total_s = query + answer: what a user waits for per recall.
            # Ingestion is one-time/amortized, so it is NOT in total_s.
            "total_s": prepared.timings.get("query_s", prepared.timings.get("retrieval_s", 0.0))
            + answer.latency_s,
        },
        "usage": {"answer": answer.usage, "judge": judge.usage},
        "providers": {
            "answer_provider": answer.provider,
            "answer_model": answer.model,
            "judge_provider": judge.provider,
            "judge_model": judge.model,
        },
    }


def format_progress_prefix(
    *,
    completed: int,
    total: int,
    case_index: int,
    case_total: int,
    case: LongMemEvalCase,
    variant_index: int,
    variant_total: int,
    variant: Variant,
) -> str:
    prefix = f"[{completed}/{total}] case {case_index}/{case_total} {case.question_id or '<no-id>'}"
    if variant_total > 1:
        prefix += f" | variant {variant_index}/{variant_total} {variant.name}"
    return prefix


class CachingEmbedder:
    """Caches only the INGEST embeddings (the haystack), not query embeddings.

    The LanceDB variants ingest the identical haystack within a case, so
    without caching we'd re-embed ~500 turns once per variant — that's the real
    cost saving, via the batch ``embed()`` path (store.add_rows). The query,
    embedded via
    ``embed_one()`` (retrieval.recall), is deliberately NOT cached: it's one
    tiny string, caching it saves nothing, and a cache hit would zero out the
    query-embedding cost for every variant after the first — making per-variant
    query latency meaningless (vector pays the embed, hybrid/cross-encoder read
    a free cached vector). Each variant embeds its own query so ``query_s``
    reflects real retrieval latency. The cache is reset per case to bound
    memory. ``api_texts`` counts texts actually sent to the API.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self._cache: dict[str, list[float]] = {}
        self.api_texts = 0
        self.cache_hits = 0

    @property
    def dim(self) -> int:
        return self.inner.dim

    def warm(self) -> int:
        return self.inner.warm()

    def reset(self) -> None:
        self._cache.clear()

    def embed_one(self, text: str) -> list[float]:
        # Query path — never cached, so each variant's query_s includes a real
        # query embedding (see class docstring).
        return self.inner.embed_one(text)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        uncached = [t for t in dict.fromkeys(texts) if t not in self._cache]
        self.cache_hits += len(texts) - len(uncached)
        if uncached:
            self.api_texts += len(uncached)
            for text, vector in zip(uncached, self.inner.embed(uncached)):
                self._cache[text] = vector
        return [self._cache[t] for t in texts]


def build_benchmark_resources(
    variants: list[Variant],
    args: argparse.Namespace,
    *,
    embedder: Any = None,
) -> BenchmarkResources:
    resources = BenchmarkResources(embedder=embedder)
    if not any(variant.backend == "lancedb" for variant in variants):
        return resources

    plugin = load_lancedb_plugin()
    config_mod = importlib.import_module(f"{plugin.__name__}.src.config")
    # Use the plugin's SHIPPED defaults (default_config.yaml via DEFAULTS), NOT
    # load_config() — which would merge the user's personal ~/.hermes/config.yaml
    # and make benchmark results depend on their machine. The benchmark measures
    # the plugin as shipped, so it must be reproducible from the repo alone.
    cfg = config_mod.DEFAULTS
    if resources.embedder is None:
        embeddings_mod = importlib.import_module(f"{plugin.__name__}.src.embeddings")
        embedding_cfg = cfg.get("embedding", {}) or {}
        model_name = embedding_cfg.get("model", "text-embedding-3-small")
        log_progress(args, f"Using embedding model: {model_name}")
        resources.embedder = embeddings_mod.OpenAIEmbedder(model_name)
        resources.embedder.warm()

    # Cache embeddings so the LanceDB variants (vector / hybrid-rrf /
    # cross-encoder) don't re-embed the same haystack + query for each case.
    # Reset per case (see run_benchmark) to bound memory.
    if not isinstance(resources.embedder, CachingEmbedder):
        resources.embedder = CachingEmbedder(resources.embedder)

    if any(variant.reranker_type == "cross-encoder" for variant in variants):
        reranker_cfg = cfg.get("retrieval", {}).get("reranker", {}) or {}
        model_name = reranker_cfg.get("model") or "cross-encoder/ettin-reranker-17m-v1"
        log_progress(args, f"Loading cross-encoder reranker once: {model_name}")
        from lancedb.rerankers import CrossEncoderReranker

        resources.rerankers["cross-encoder"] = CrossEncoderReranker(
            model_name=model_name,
            column="content",
        )
    return resources


class LanceDBCaseIndex:
    def __init__(
        self,
        case: LongMemEvalCase,
        root: Path,
        variant: Variant,
        *,
        embedder: Any = None,
        reranker: Any = None,
    ) -> None:
        self.case = case
        self.root = root / "hermes-home"
        self.variant = variant
        self.embedder = embedder
        self.reranker = reranker
        self.provider = None
        # turn id -> session date. The date is provenance metadata, so it is
        # NOT embedded into the content column (that would poison the vector and
        # the cross-encoder); we re-attach it for the answer prompt on retrieve.
        self._date_by_id: dict[str, str] = {}

    def ingest(self) -> None:
        plugin = load_lancedb_plugin()
        provider = plugin.LanceDBMemoryProvider()
        if self.embedder is not None:
            provider._embedder = self.embedder
        if self.reranker is not None:
            provider._reranker = self.reranker
        provider.initialize(
            f"longmemeval-{self.case.question_id}",
            hermes_home=str(self.root),
            platform="benchmark",
            agent_context="primary",
            agent_identity="longmemeval",
            agent_workspace=f"longmemeval-{self.case.question_id}",
            user_id="longmemeval",
        )
        provider._config.setdefault("retrieval", {})["mode"] = self.variant.mode
        provider._config["retrieval"]["top_k"] = 50
        provider._config["retrieval"]["search_kinds"] = ["turn"]
        provider._config["retrieval"].setdefault("reranker", {})["type"] = self.variant.reranker_type
        provider._config["retrieval"]["reranker"]["weight"] = self.variant.reranker_weight
        rows = []
        for session_index, session in enumerate(self.case.haystack_sessions):
            session_id = _session_id(self.case, session_index)
            date = _session_date(self.case, session_index)
            for turn_index, turn in enumerate(session):
                role = turn.get("role") or ""
                content = turn.get("content") or ""
                turn_id = make_benchmark_turn_id(session_id, turn_index, role, content)
                self._date_by_id[turn_id] = date
                rows.append(
                    {
                        "id": turn_id,
                        "kind": "turn",
                        # Raw turn text only — provenance (date/session/turn/role)
                        # lives in the metadata columns below and is re-attached
                        # in the answer prompt. Keeping it out of `content` means
                        # the vector embedding and the cross-encoder rerank score
                        # the actual message, not boilerplate labels.
                        "content": content,
                        "abstract": "",
                        "category": "",
                        "tags": ["longmemeval"],
                        "provenance_turn_ids": [],
                        "session_id": session_id,
                        "turn_index": turn_index,
                        "role": role,
                        "user_id": "longmemeval",
                        "agent_identity": "longmemeval",
                        "agent_workspace": f"longmemeval-{self.case.question_id}",
                        "platform": "benchmark",
                        "source": "longmemeval_turn",
                    }
                )
        provider.store.add_rows(rows)
        self.provider = provider

    def retrieve(self, query: str, *, limit: int) -> list[RetrievedSnippet]:
        rows = self.provider.recall(query, mode=self.variant.mode, kind="turn", limit=limit)
        snippets = []
        for row in rows:
            row_id = str(row.get("id") or "")
            snippets.append(
                RetrievedSnippet(
                    id=row_id,
                    session_id=str(row.get("session_id") or ""),
                    turn_index=int(row.get("turn_index") or 0),
                    role=str(row.get("role") or ""),
                    date=self._date_by_id.get(row_id, ""),
                    text=str(row.get("content") or ""),  # raw content
                    score=first_score(row),
                )
            )
        return snippets

    def close(self) -> None:
        if self.provider is not None:
            self.provider.shutdown()


def build_case_index(
    case: LongMemEvalCase,
    case_dir: Path,
    variant: Variant,
    resources: BenchmarkResources,
):
    if variant.backend == "lancedb":
        return LanceDBCaseIndex(
            case,
            case_dir,
            variant,
            embedder=resources.embedder,
            reranker=resources.rerankers.get(variant.reranker_type),
        )
    if variant.backend == "hermes-session-search":
        return HermesSessionSearchCaseIndex(case, case_dir, variant)
    if variant.backend == "full-context":
        return FullContextCaseIndex(case, case_dir, variant)
    raise ValueError(f"Unknown variant backend: {variant.backend}")


class FullContextCaseIndex:
    """Accuracy ceiling: no retrieval — feed every haystack turn to the answer
    model. ``retrieve`` ignores ``limit`` and returns the whole transcript in
    chronological order."""

    def __init__(self, case: LongMemEvalCase, root: Path, variant: Variant) -> None:
        self.case = case
        self.variant = variant

    def ingest(self) -> None:
        return None

    def retrieve(self, query: str, *, limit: int) -> list[RetrievedSnippet]:
        return list(_all_turn_snippets(self.case))

    def close(self) -> None:
        return None


class HermesSessionSearchCaseIndex:
    """Canonical baseline: Hermes's built-in session store (SQLite FTS5 over
    verbatim messages). Ingests the haystack via the real ``HermesState`` API
    into an isolated per-case ``state.db``, then retrieves with
    ``search_messages``. Zero LLM cost. This is transcript retrieval, not a
    long-term memory store."""

    def __init__(self, case: LongMemEvalCase, root: Path, variant: Variant) -> None:
        self.case = case
        self.root = root / "hermes-home"
        self.variant = variant
        self.state = None
        # Hermes message row id -> the original turn. search_messages returns a
        # highlighted snippet, not the raw content, so we reconstruct the full
        # turn (for the answer prompt) and its benchmark id (for hit metrics)
        # from our own ingest-time record, keyed by the row id.
        self._turn_by_msg_id: dict[int, dict[str, Any]] = {}

    def ingest(self) -> None:
        ensure_hermes_state_importable()
        from hermes_state import SessionDB

        db_path = self.root / "state.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = SessionDB(db_path=db_path)
        for session_index, session in enumerate(self.case.haystack_sessions):
            session_id = _session_id(self.case, session_index)
            date = _session_date(self.case, session_index)
            self.state.create_session(session_id, source="benchmark")
            for turn_index, turn in enumerate(session):
                role = turn.get("role") or ""
                content = turn.get("content") or ""
                # Store raw content so FTS5 indexes the real words (not our
                # display labels); we reformat for the answer prompt on read.
                msg_id = self.state.append_message(
                    session_id=session_id,
                    role=role,
                    content=content,
                )
                self._turn_by_msg_id[int(msg_id)] = {
                    "session_id": session_id,
                    "turn_index": turn_index,
                    "role": role,
                    "content": content,
                    "date": date,
                }

    def retrieve(self, query: str, *, limit: int) -> list[RetrievedSnippet]:
        fts_query = build_fts_query(query)
        if not fts_query or self.state is None:
            return []
        rows = self.state.search_messages(fts_query, limit=limit)
        snippets = []
        for row in rows:
            turn = self._turn_by_msg_id.get(int(_row_value(row, "id") or 0))
            if turn is None:
                continue
            snippets.append(
                RetrievedSnippet(
                    id=make_benchmark_turn_id(
                        turn["session_id"], turn["turn_index"], turn["role"], turn["content"]
                    ),
                    session_id=turn["session_id"],
                    turn_index=turn["turn_index"],
                    role=turn["role"],
                    date=turn["date"],
                    text=turn["content"],  # raw content; provenance is labelled by the answer prompt
                    score=None,
                )
            )
        return snippets

    def close(self) -> None:
        if self.state is not None:
            close = getattr(self.state, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def _all_turn_snippets(case: LongMemEvalCase) -> Iterable[RetrievedSnippet]:
    for session_index, session in enumerate(case.haystack_sessions):
        session_id = _session_id(case, session_index)
        date = _session_date(case, session_index)
        for turn_index, turn in enumerate(session):
            role = turn.get("role") or ""
            content = turn.get("content") or ""
            yield RetrievedSnippet(
                id=make_benchmark_turn_id(session_id, turn_index, role, content),
                session_id=session_id,
                turn_index=turn_index,
                role=role,
                date=date,
                text=content,  # raw content; provenance is labelled by the answer prompt
                score=None,
            )


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    """Rough token estimate (~chars/4) for the answer prompt — used only to
    decide full-context overflow against a configured budget."""
    chars = sum(len(m.get("content") or "") for m in messages)
    return chars // 4


def build_fts_query(question: str) -> str:
    """Turn a natural-language question into a disjunctive FTS5 query.

    Hermes's ``_sanitize_fts5_query`` leaves space-separated terms as an implicit
    AND, so passing a raw question would require one message to contain *every*
    word — near-zero recall. The standard lexical-IR formulation is a
    bag-of-words OR query, letting BM25 rank by term relevance. We do this
    uniformly for every question (no per-question hand-tuning) so the baseline
    is fair and reproducible.
    """
    tokens: list[str] = []
    for tok in re.findall(r"\w+", (question or "").lower()):
        if tok in {"and", "or", "not"}:  # bare FTS5 operators if lowercased
            continue
        if tok not in tokens:
            tokens.append(tok)
    return " OR ".join(tokens)


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    """Read a column from a search_messages row (dict or sqlite3.Row)."""
    try:
        if isinstance(row, dict):
            return row.get(key, default)
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def ensure_hermes_state_importable() -> None:
    root = Path(__file__).resolve().parents[2]
    ensure_hermes_agent_path(root.parent / "hermes-agent")


def load_lancedb_plugin() -> ModuleType:
    global _PLUGIN_MODULE
    if _PLUGIN_MODULE is not None:
        return _PLUGIN_MODULE
    root = Path(__file__).resolve().parents[2]
    ensure_hermes_agent_path(root.parent / "hermes-agent")
    module_name = "hermes_agent_memory_benchmark_plugin"
    spec = importlib.util.spec_from_file_location(
        module_name,
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load LanceDB plugin from {root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _PLUGIN_MODULE = module
    return module


def ensure_hermes_agent_path(hermes_agent: Path) -> None:
    if not hermes_agent.exists():
        return
    hermes_path = str(hermes_agent)
    while hermes_path in sys.path:
        sys.path.remove(hermes_path)
    sys.path.insert(0, hermes_path)
    shadowed_tools = sys.modules.get("tools")
    if shadowed_tools is not None and not hasattr(shadowed_tools, "__path__"):
        del sys.modules["tools"]


async def call_model_async(
    *,
    task: str,
    messages: list[dict[str, str]],
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
    llm_call: Callable[..., Any] | None,
) -> LlmResult:
    start = time.perf_counter()
    if llm_call is None:
        from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning

        response = await async_call_llm(
            task=task,
            provider=provider or None,
            model=model or None,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = extract_content_or_reasoning(response)
    elif inspect.iscoroutinefunction(llm_call):
        response = await llm_call(
            task=task,
            provider=provider or None,
            model=model or None,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = extract_response_text(response)
    else:
        response = await asyncio.to_thread(
            llm_call,
            task=task,
            provider=provider or None,
            model=model or None,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if inspect.isawaitable(response):
            response = await response
        text = extract_response_text(response)
    latency = time.perf_counter() - start
    return LlmResult(
        text=text.strip(),
        latency_s=latency,
        usage=extract_usage(response),
        provider=provider or "",
        model=model or "",
    )


def call_model(
    *,
    task: str,
    messages: list[dict[str, str]],
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
    llm_call: Callable[..., Any] | None,
) -> LlmResult:
    start = time.perf_counter()
    if llm_call is None:
        from agent.auxiliary_client import call_llm, extract_content_or_reasoning

        response = call_llm(
            task=task,
            provider=provider or None,
            model=model or None,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = extract_content_or_reasoning(response)
    else:
        response = llm_call(
            task=task,
            provider=provider or None,
            model=model or None,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = extract_response_text(response)
    latency = time.perf_counter() - start
    return LlmResult(
        text=text.strip(),
        latency_s=latency,
        usage=extract_usage(response),
        provider=provider or "",
        model=model or "",
    )


def build_answer_messages(
    case: LongMemEvalCase,
    snippets: list[RetrievedSnippet],
    *,
    max_snippet_chars: int | None = 1200,
) -> list[dict[str, str]]:
    context = format_retrieved_context(snippets, max_snippet_chars=max_snippet_chars)
    user = (
        f"Question date: {case.question_date or 'unknown'}\n"
        f"Question: {case.question}\n\n"
        "Retrieved memory snippets:\n"
        f"{context}\n\n"
        "Answer the question using only the retrieved memory snippets. "
        "If the snippets do not contain enough information, say you do not know."
    )
    return [
        {
            "role": "system",
            "content": (
                "You answer LongMemEval questions from retrieved conversation memory. "
                "Be concise and do not mention benchmark labels."
            ),
        },
        {"role": "user", "content": user},
    ]


def build_judge_messages(question: str, reference: str, hypothesis: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a strict LongMemEval QA judge. Reply with exactly 'yes' if the "
                "hypothesis answers the question equivalently to the reference answer, "
                "otherwise reply with exactly 'no'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Reference answer: {reference}\n"
                f"Hypothesis: {hypothesis}\n\n"
                "Is the hypothesis correct? Reply yes or no."
            ),
        },
    ]


def parse_judge_label(text: str) -> str:
    normalized = (text or "").strip().lower()
    normalized = re.sub(r"^[^a-z]+", "", normalized)
    first = re.split(r"[^a-z]+", normalized, maxsplit=1)[0] if normalized else ""
    if first in {"yes", "y", "correct", "true"}:
        return "yes"
    if first in {"no", "n", "incorrect", "false"}:
        return "no"
    if re.search(r"\byes\b", normalized) and not re.search(r"\bno\b", normalized):
        return "yes"
    if re.search(r"\bno\b", normalized) and not re.search(r"\byes\b", normalized):
        return "no"
    return "unknown"


def retrieval_metrics(
    snippets: list[RetrievedSnippet],
    answer_session_ids: list[str],
    answer_turn_ids: list[str],
) -> dict[str, Any]:
    """Per-case retrieval diagnostics over the ranked snippet list.

    ``snippets`` are in retrieval-rank order, so the first gold turn's position
    gives the reciprocal rank (aggregated to MRR). ~60% of LongMemEval cases
    have multiple gold turns, so ``turn_recall`` (fraction of gold turns
    retrieved) is reported alongside the boolean ``turn_hit`` (>=1 gold turn).
    Cases with no gold labels yield None and are excluded from aggregates.
    """
    answer_sessions = set(answer_session_ids)
    answer_turns = set(answer_turn_ids)
    retrieved_sessions = [s.session_id for s in snippets]
    retrieved_turns = [s.id for s in snippets]

    reciprocal_rank: float | None = None
    if answer_turns:
        reciprocal_rank = 0.0
        for rank, turn_id in enumerate(retrieved_turns, start=1):
            if turn_id in answer_turns:
                reciprocal_rank = 1.0 / rank
                break

    return {
        "retrieved_session_ids": retrieved_sessions,
        "retrieved_turn_ids": retrieved_turns,
        "gold_turn_count": len(answer_turns),
        "session_hit": bool(answer_sessions.intersection(retrieved_sessions))
        if answer_sessions
        else None,
        "turn_hit": bool(answer_turns.intersection(retrieved_turns)) if answer_turns else None,
        "session_recall": _rate(len(answer_sessions & set(retrieved_sessions)), len(answer_sessions))
        if answer_sessions
        else None,
        "turn_recall": _rate(len(answer_turns & set(retrieved_turns)), len(answer_turns))
        if answer_turns
        else None,
        "reciprocal_rank": reciprocal_rank,
    }


def _retrieval_metric_bundle(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    """Accuracy + retrieval metrics over a set of records. Reused for the
    overall per-variant rollup and the per-question-type breakdown."""

    def hit_rate(key: str) -> float:
        vals = [r["retrieval"][key] for r in rows if r.get("retrieval", {}).get(key) is not None]
        return _rate(sum(1 for v in vals if v), len(vals))

    def mean(key: str) -> float:
        vals = [r["retrieval"][key] for r in rows if r.get("retrieval", {}).get(key) is not None]
        return _rate(sum(float(v) for v in vals), len(vals))

    return {
        "cases": len(rows),
        "overflow_cases": sum(1 for r in rows if r.get("overflow")),
        "accuracy": _rate(sum(1 for r in rows if r.get("correct")), len(rows)),
        f"turn_recall@{top_k}": mean("turn_recall"),
        f"session_recall@{top_k}": mean("session_recall"),
        f"mrr@{top_k}": mean("reciprocal_rank"),
        f"retrieval_turn_hit@{top_k}": hit_rate("turn_hit"),
        f"retrieval_session_hit@{top_k}": hit_rate("session_hit"),
    }


def build_summary(
    records: list[dict[str, Any]],
    *,
    cases: list[LongMemEvalCase],
    variants: list[Variant],
    args: argparse.Namespace,
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    by_variant: dict[str, dict[str, Any]] = {}
    for variant in variants:
        rows = [r for r in records if r["variant"] == variant.name]
        bundle = _retrieval_metric_bundle(rows, args.top_k)
        bundle.update(
            {
                "latency_s_avg": average_latency(rows),
                "latency_s_p50": latency_percentiles(rows, 50.0),
                "latency_s_p95": latency_percentiles(rows, 95.0),
                "usage_total": sum_usage(rows),
                "tokens_per_question": tokens_per_question(rows),
                "by_question_type": {
                    qt: _retrieval_metric_bundle(
                        [r for r in rows if r["question_type"] == qt], args.top_k
                    )
                    for qt in sorted({r["question_type"] for r in rows})
                },
            }
        )
        by_variant[variant.name] = bundle
    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "cases": len(cases),
        "top_k": args.top_k,
        "variants": [v.name for v in variants],
        "answer_provider": args.answer_provider,
        "answer_model": args.answer_model,
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "by_variant": by_variant,
    }


def format_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LongMemEval Benchmark Summary",
        "",
        f"- Cases: {summary['cases']}",
        f"- Top-k: {summary['top_k']}",
        f"- Started: {summary['started_at']}",
        f"- Finished: {summary['finished_at']}",
        "",
        f"| Variant | Cases | Accuracy | Recall@{summary['top_k']} | MRR@{summary['top_k']} "
        f"| Query p50 | Query p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    k = summary["top_k"]
    recall_key = f"turn_recall@{k}"
    mrr_key = f"mrr@{k}"
    for name, stats in summary["by_variant"].items():
        p50 = stats.get("latency_s_p50", {})
        p95 = stats.get("latency_s_p95", {})
        lines.append(
            "| {name} | {cases} | {acc:.3f} | {recall:.3f} | {mrr:.3f} "
            "| {q50:.3f}s | {q95:.3f}s |".format(
                name=name,
                cases=stats["cases"],
                acc=stats["accuracy"],
                recall=stats.get(recall_key, 0.0),
                mrr=stats.get(mrr_key, 0.0),
                q50=p50.get("query_s", 0.0),
                q95=p95.get("query_s", 0.0),
            )
        )

    # Per-question-type accuracy matrix (rows = type, cols = variant).
    variant_names = list(summary["by_variant"].keys())
    all_types = sorted(
        {qt for stats in summary["by_variant"].values() for qt in stats.get("by_question_type", {})}
    )
    if all_types:
        lines += [
            "",
            "## Accuracy by question type",
            "",
            "| Question type | " + " | ".join(variant_names) + " |",
            "|---" + "|---:" * len(variant_names) + "|",
        ]
        for qt in all_types:
            cells = []
            for name in variant_names:
                bucket = summary["by_variant"][name].get("by_question_type", {}).get(qt)
                if bucket:
                    cells.append(f"{bucket['accuracy']:.3f} (n={bucket['cases']})")
                else:
                    cells.append("—")
            lines.append(f"| {qt} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def format_retrieved_context(
    snippets: list[RetrievedSnippet],
    *,
    max_snippet_chars: int | None = 1200,
) -> str:
    if not snippets:
        return "(none)"
    lines = []
    for i, snippet in enumerate(snippets, start=1):
        text = " ".join(snippet.text.split())
        if max_snippet_chars is not None and len(text) > max_snippet_chars:
            text = text[: max_snippet_chars - 3] + "..."
        label = f"[{i}] session={snippet.session_id} turn={snippet.turn_index} role={snippet.role}"
        if snippet.date:
            label += f" date={snippet.date}"
        lines.append(f"{label}\n{text}")
    return "\n\n".join(lines)


_LATENCY_KEYS = ("ingest_s", "query_s", "retrieval_s", "answer_s", "judge_s", "total_s")


def average_latency(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        key: _rate(sum(float(r["latency_s"].get(key, 0.0)) for r in rows), len(rows))
        for key in _LATENCY_KEYS
    }


def latency_percentiles(rows: list[dict[str, Any]], q: float) -> dict[str, float]:
    return {
        key: _percentile([float(r["latency_s"].get(key, 0.0)) for r in rows], q)
        for key in _LATENCY_KEYS
    }


def tokens_per_question(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average answer-phase tokens per question — the efficiency axis.

    The answer phase is where the retrieved/stuffed context lands, so its input
    tokens are what separates full-context from retrieval. Judge tokens are
    excluded (they score the answer, not the memory).
    """
    keys = ("input_tokens", "output_tokens", "total_tokens")
    totals: dict[str, float] = {key: 0.0 for key in keys}
    for row in rows:
        answer_usage = (row.get("usage", {}) or {}).get("answer") or {}
        for key in keys:
            totals[key] += float(answer_usage.get(key, 0) or 0)
    return {key: _rate(totals[key], len(rows)) for key in keys}


def sum_usage(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for row in rows:
        for phase in ("answer", "judge"):
            for key, value in (row.get("usage", {}).get(phase) or {}).items():
                totals[f"{phase}_{key}"] += int(value or 0)
    return dict(totals)


def extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if "text" in response:
            return str(response["text"])
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            return str(message.get("content") or "")
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        return str(getattr(message, "content", "") or "")
    return str(response or "")


def extract_usage(response: Any) -> dict[str, int]:
    raw = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if raw is None:
        return {}

    def get_int(name: str) -> int:
        value = raw.get(name) if isinstance(raw, dict) else getattr(raw, name, None)
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    usage = {
        "input_tokens": get_int("prompt_tokens") or get_int("input_tokens"),
        "output_tokens": get_int("completion_tokens") or get_int("output_tokens"),
        "total_tokens": get_int("total_tokens"),
        "cache_read_tokens": get_int("cache_read_input_tokens") or get_int("cache_read_tokens"),
        "cache_write_tokens": get_int("cache_creation_input_tokens") or get_int("cache_write_tokens"),
    }
    if not usage["total_tokens"]:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def snippet_to_dict(snippet: RetrievedSnippet) -> dict[str, Any]:
    return {
        "id": snippet.id,
        "session_id": snippet.session_id,
        "turn_index": snippet.turn_index,
        "role": snippet.role,
        "date": snippet.date,
        "text": snippet.text,
        "score": snippet.score,
    }


def first_score(row: dict[str, Any]) -> float | None:
    for key in ("_relevance_score", "_distance", "_score"):
        if key in row:
            score = safe_float(row[key])
            if score is not None:
                return score
            else:
                return None
    return None


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_tool_json(raw: str, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw or "{}")
    except JSONDecodeError as exc:
        raise RuntimeError(f"{context} returned non-JSON output: {raw!r}") from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"{context} failed: {payload['error']}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} returned unexpected JSON: {payload!r}")
    return payload


def make_benchmark_turn_id(session_id: str, turn_index: int, role: str, content: str) -> str:
    digest = hashlib.sha256(
        f"{session_id}\0{turn_index}\0{role}\0{content}".encode("utf-8")
    ).hexdigest()[:24]
    return f"turn_lme_{digest}"


def safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "case")
    return clean[:120] or "case"


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def parse_variant_limits(
    raw: Iterable[str] | None,
    known_variants: set[str] | None = None,
) -> dict[str, int]:
    """Parse repeated ``NAME=N`` --variant-limit values into a {name: cap} map."""
    limits: dict[str, int] = {}
    for entry in raw or []:
        if "=" not in entry:
            raise ValueError(f"--variant-limit must be NAME=N, got: {entry!r}")
        name, _, value = entry.partition("=")
        name = name.strip()
        try:
            cap = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"--variant-limit cap must be an integer, got: {entry!r}") from exc
        if cap < 0:
            raise ValueError(f"--variant-limit cap must be >= 0, got: {entry!r}")
        if known_variants is not None and name not in known_variants:
            raise ValueError(
                f"--variant-limit names unknown variant {name!r}; "
                f"running variants: {', '.join(sorted(known_variants))}"
            )
        limits[name] = cap
    return limits


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]). Empty -> 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def _session_id(case: LongMemEvalCase, session_index: int) -> str:
    if session_index < len(case.haystack_session_ids):
        return case.haystack_session_ids[session_index]
    return str(session_index)


def _session_date(case: LongMemEvalCase, session_index: int) -> str:
    if session_index < len(case.haystack_dates):
        return case.haystack_dates[session_index]
    return ""


def _rate(numerator: float, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
