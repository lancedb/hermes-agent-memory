"""Run LongMemEval against Hermes and LanceDB memory variants."""
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
VARIANT_NAMES = (
    "hermes-builtin-memory",
    "hermes-holographic",
    "openviking",
    "lancedb-vector",
    "lancedb-hybrid-rrf",
    "lancedb-hybrid-cross-encoder",
)
BUILTIN_MEMORY_CHAR_LIMIT = 2200
ENTRY_DELIMITER = "\n§\n"
HOLOGRAPHIC_BENCHMARK_HRR_DIM = 4096
OPENVIKING_INDEX_PREFIX = "longmemeval"


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


_PLUGIN_MODULE: ModuleType | None = None
_OPENVIKING_PLUGIN_MODULE: ModuleType | None = None


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
    parser.add_argument("--limit", type=int, default=25, help="Maximum cases to run.")
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
        help="Comma-separated variants. Use 'all' for the full matrix.",
    )
    parser.add_argument("--answer-provider", default="", help="Explicit Hermes provider for answers.")
    parser.add_argument("--answer-model", default="", help="Explicit answer model.")
    parser.add_argument("--judge-provider", default="", help="Explicit Hermes provider for judging.")
    parser.add_argument("--judge-model", default="", help="Explicit judge model.")
    parser.add_argument("--answer-max-tokens", type=int, default=256)
    parser.add_argument("--judge-max-tokens", type=int, default=16)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of case/variant answer and judge LLM calls to run concurrently.",
    )
    parser.add_argument(
        "--openviking-index-wait-s",
        type=float,
        default=900.0,
        help="Max seconds to wait for OpenViking vectors_only reindex + queue drain during ingestion.",
    )
    parser.add_argument(
        "--openviking-turns-per-doc",
        type=int,
        default=4,
        help="Number of benchmark turns to pack into each OpenViking leaf document.",
    )
    parser.add_argument(
        "--openviking-index-prefix",
        default=OPENVIKING_INDEX_PREFIX,
        help="OpenViking memory namespace prefix for deterministic benchmark indexes.",
    )
    parser.add_argument(
        "--openviking-use-prebuilt-index",
        action="store_true",
        help="Skip OpenViking ingestion and search a prebuilt index from build_openviking_index.py.",
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
    root = Path(__file__).resolve().parents[2]
    env_paths = [root / ".env", Path.home() / ".hermes" / ".env"]
    try:
        from embeddings import load_env_file
    except ImportError:
        return
    for path in env_paths:
        load_env_file(path)


def load_cases(
    path: Path,
    *,
    limit: int | None = None,
    offset: int = 0,
    question_types: set[str] | None = None,
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
    for item in raw:
        case = parse_case(item)
        if question_types and case.question_type not in question_types:
            continue
        if skipped < max(0, offset):
            skipped += 1
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
    if not names or names == ["all"]:
        names = list(VARIANT_NAMES)
    variants: list[Variant] = []
    for name in names:
        if name == "hermes-builtin-memory":
            variants.append(Variant(name=name, backend="hermes-builtin"))
        elif name == "hermes-holographic":
            variants.append(Variant(name=name, backend="holographic"))
        elif name == "openviking":
            variants.append(Variant(name=name, backend="openviking"))
        elif name in {"markdown-lexical", "builtin-markdown"}:
            variants.append(Variant(name=name, backend="markdown-lexical"))
        elif name == "lancedb-vector":
            variants.append(Variant(name=name, backend="lancedb", mode="vector"))
        elif name == "lancedb-hybrid-rrf":
            variants.append(Variant(name=name, backend="lancedb", mode="hybrid", reranker_type="rrf"))
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
    work_items = [
        (case_index, case, variant_index, variant)
        for case_index, case in enumerate(cases, start=1)
        for variant_index, variant in enumerate(variants, start=1)
    ]
    resources = (
        build_benchmark_resources(variants, args, embedder=embedder)
        if work_items
        else BenchmarkResources(embedder=embedder)
    )

    records: list[dict[str, Any]] = []
    total = len(work_items)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for batch_start in range(0, total, batch_size):
            batch = work_items[batch_start : batch_start + batch_size]
            prepared: list[PreparedCaseVariant] = []
            for offset, (case_index, case, variant_index, variant) in enumerate(batch, start=1):
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
                prepared.append(
                    prepare_case_variant(
                        case,
                        variant,
                        args,
                        resources=resources,
                        progress_prefix=progress_prefix,
                    )
                )
            batch_records = asyncio.run(complete_prepared_batch(prepared, args, llm_call=llm_call))
            for record in batch_records:
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                handle.flush()

    summary = build_summary(
        records,
        cases=cases,
        variants=variants,
        args=args,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
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

    timings: dict[str, float] = {}
    retrieval_start = time.perf_counter()
    log_progress(args, f"{progress_prefix}: ingesting and retrieving top-{args.top_k}")
    if variant.backend == "hermes-builtin":
        retriever = HermesBuiltinMemoryIndex(case, case_dir)
        retriever.ingest()
        snippets = retriever.retrieve(case.question, limit=args.top_k)
    elif variant.backend == "holographic":
        retriever = HolographicCaseIndex(case, case_dir)
        try:
            retriever.ingest()
            snippets = retriever.retrieve(case.question, limit=args.top_k)
        finally:
            retriever.close()
    elif variant.backend == "openviking":
        openviking_turns_per_doc = max(1, int(getattr(args, "openviking_turns_per_doc", 4) or 4))
        retriever = OpenVikingCaseIndex(
            case,
            case_dir,
            turns_per_doc=openviking_turns_per_doc,
            index_wait_s=float(getattr(args, "openviking_index_wait_s", 900.0) or 900.0),
            scope_name=openviking_scope_name(
                case,
                str(getattr(args, "openviking_index_prefix", OPENVIKING_INDEX_PREFIX) or ""),
                openviking_turns_per_doc,
            ),
        )
        try:
            if getattr(args, "openviking_use_prebuilt_index", False):
                log_progress(args, f"{progress_prefix}: using prebuilt OpenViking index")
                retriever.connect()
            else:
                retriever.ingest()
            snippets = retriever.retrieve(case.question, limit=args.top_k)
        finally:
            retriever.close()
    elif variant.backend == "markdown-lexical":
        retriever = MarkdownMemoryIndex(case, case_dir)
        retriever.ingest()
        snippets = retriever.retrieve(case.question, limit=args.top_k)
    else:
        retriever = LanceDBCaseIndex(
            case,
            case_dir,
            variant,
            embedder=resources.embedder,
            reranker=resources.rerankers.get(variant.reranker_type),
        )
        try:
            retriever.ingest()
            snippets = retriever.retrieve(case.question, limit=args.top_k)
        finally:
            retriever.close()
    timings["retrieval_s"] = time.perf_counter() - retrieval_start
    log_progress(
        args,
        f"{progress_prefix}: retrieved {len(snippets)} snippet(s) in {timings['retrieval_s']:.2f}s",
    )
    metrics = retrieval_metrics(snippets, case.answer_session_ids, case.answer_turn_ids)

    if not args.keep_temp:
        shutil.rmtree(case_dir, ignore_errors=True)

    return PreparedCaseVariant(
        case=case,
        variant=variant,
        progress_prefix=progress_prefix,
        timings=timings,
        snippets=snippets,
        metrics=metrics,
        answer_messages=build_answer_messages(case, snippets),
    )


async def complete_prepared_batch(
    prepared: list[PreparedCaseVariant],
    args: argparse.Namespace,
    *,
    llm_call: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    if not prepared:
        return []

    for item in prepared:
        log_progress(args, f"{item.progress_prefix}: calling answer model")
    answers = await asyncio.gather(
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
            for item in prepared
        ]
    )
    for item, answer in zip(prepared, answers):
        log_progress(args, f"{item.progress_prefix}: answer model returned in {answer.latency_s:.2f}s")

    for item in prepared:
        log_progress(args, f"{item.progress_prefix}: calling judge model")
    judges = await asyncio.gather(
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
            for item, answer in zip(prepared, answers)
        ]
    )

    records = []
    for item, answer, judge in zip(prepared, answers, judges):
        label = parse_judge_label(judge.text)
        log_progress(args, f"{item.progress_prefix}: judge returned {label} in {judge.latency_s:.2f}s")
        record = build_record(item, answer, judge, label, args)
        records.append(record)
        log_progress(
            args,
            f"{item.progress_prefix}: done "
            f"correct={record['correct']} "
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
        "answer_session_ids": case.answer_session_ids,
        "answer_turn_ids": case.answer_turn_ids,
        "retrieval": prepared.metrics,
        "latency_s": {
            **prepared.timings,
            "answer_s": answer.latency_s,
            "judge_s": judge.latency_s,
            "total_s": prepared.timings["retrieval_s"] + answer.latency_s + judge.latency_s,
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
    config_mod = importlib.import_module(f"{plugin.__name__}.config")
    cfg = config_mod.load_config()
    if resources.embedder is None:
        embeddings_mod = importlib.import_module(f"{plugin.__name__}.embeddings")
        embedding_cfg = cfg.get("embedding", {}) or {}
        model_name = embedding_cfg.get("model", "text-embedding-3-small")
        log_progress(args, f"Loading embedding model once: {model_name}")
        resources.embedder = embeddings_mod.create_embedder(embedding_cfg)
        resources.embedder.warm()

    if any(variant.reranker_type == "cross-encoder" for variant in variants):
        reranker_cfg = cfg.get("retrieval", {}).get("reranker", {}) or {}
        model_name = reranker_cfg.get("model", "cross-encoder/ettin-reranker-32m-v1")
        log_progress(args, f"Loading cross-encoder reranker once: {model_name}")
        from lancedb.rerankers import CrossEncoderReranker

        resources.rerankers["cross-encoder"] = CrossEncoderReranker(
            model_name=model_name,
            column="content",
        )
    return resources


class MarkdownMemoryIndex:
    def __init__(self, case: LongMemEvalCase, root: Path) -> None:
        self.case = case
        self.root = root / "markdown-memory"
        self.docs: list[RetrievedSnippet] = []

    def ingest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for session_index, session in enumerate(self.case.haystack_sessions):
            session_id = _session_id(self.case, session_index)
            date = _session_date(self.case, session_index)
            lines = [f"# Session {session_id}", "", f"Date: {date}", ""]
            for turn_index, turn in enumerate(session):
                role = turn.get("role") or ""
                content = turn.get("content") or ""
                turn_id = make_benchmark_turn_id(session_id, turn_index, role, content)
                lines.extend([f"## {turn_index}. {role}", "", content, ""])
                self.docs.append(
                    RetrievedSnippet(
                        id=turn_id,
                        session_id=session_id,
                        turn_index=turn_index,
                        role=role,
                        date=date,
                        text=content,
                    )
                )
            (self.root / f"{safe_name(session_id)}.md").write_text(
                "\n".join(lines), encoding="utf-8"
            )

    def retrieve(self, query: str, *, limit: int) -> list[RetrievedSnippet]:
        query_terms = tokenize(query)
        scored = []
        for doc in self.docs:
            score = lexical_score(query_terms, doc.text)
            scored.append((score, doc))
        scored.sort(key=lambda item: (-item[0], item[1].session_id, item[1].turn_index))
        return [
            RetrievedSnippet(**{**snippet_to_dict(doc), "score": float(score)})
            for score, doc in scored[: max(1, limit)]
        ]


class HermesBuiltinMemoryIndex:
    """No-index baseline matching Hermes's file-backed prompt memory shape."""

    def __init__(self, case: LongMemEvalCase, root: Path) -> None:
        self.case = case
        self.root = root / "hermes-builtin-memory"
        self.docs: list[RetrievedSnippet] = []

    def ingest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        memory_dir = self.root / "memories"
        memory_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for session_index, session in enumerate(self.case.haystack_sessions):
            session_id = _session_id(self.case, session_index)
            date = _session_date(self.case, session_index)
            for turn_index, turn in enumerate(session):
                role = turn.get("role") or ""
                content = turn.get("content") or ""
                entry = format_turn_content(date, session_id, turn_index, role, content)
                next_entries = entries + [entry]
                if len(ENTRY_DELIMITER.join(next_entries)) > BUILTIN_MEMORY_CHAR_LIMIT:
                    (memory_dir / "MEMORY.md").write_text(
                        ENTRY_DELIMITER.join(entries),
                        encoding="utf-8",
                    )
                    return
                entries.append(entry)
                self.docs.append(
                    RetrievedSnippet(
                        id=make_benchmark_turn_id(session_id, turn_index, role, content),
                        session_id=session_id,
                        turn_index=turn_index,
                        role=role,
                        date=date,
                        text=entry,
                    )
                )
        (memory_dir / "MEMORY.md").write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")

    def retrieve(self, query: str, *, limit: int) -> list[RetrievedSnippet]:
        del query, limit
        return list(self.docs)


class HolographicCaseIndex:
    def __init__(self, case: LongMemEvalCase, root: Path) -> None:
        self.case = case
        self.root = root / "holographic-memory"
        self.provider = None

    def ingest(self) -> None:
        plugin = load_holographic_plugin()
        provider = plugin.HolographicMemoryProvider(
            config={
                "db_path": str(self.root / "memory_store.db"),
                "auto_extract": False,
                "default_trust": 0.5,
                "hrr_dim": HOLOGRAPHIC_BENCHMARK_HRR_DIM,
                "hrr_weight": 0.3,
                "min_trust_threshold": 0.0,
            }
        )
        self.root.mkdir(parents=True, exist_ok=True)
        provider.initialize(f"longmemeval-{self.case.question_id}")
        for session_index, session in enumerate(self.case.haystack_sessions):
            session_id = _session_id(self.case, session_index)
            date = _session_date(self.case, session_index)
            for turn_index, turn in enumerate(session):
                role = turn.get("role") or ""
                content = turn.get("content") or ""
                provider._store.add_fact(
                    format_turn_content(date, session_id, turn_index, role, content),
                    category="turn",
                    tags="longmemeval",
                )
        self.provider = provider

    def retrieve(self, query: str, *, limit: int) -> list[RetrievedSnippet]:
        fts_query = holographic_fts_query(query) or query
        rows = self.provider._retriever.search(
            fts_query,
            category="turn",
            min_trust=0.0,
            limit=limit,
        )
        snippets = []
        for row in rows:
            content = str(row.get("content") or "")
            parsed = parse_turn_content(content)
            snippets.append(
                RetrievedSnippet(
                    id=make_benchmark_turn_id(
                        parsed["session_id"],
                        parsed["turn_index"],
                        parsed["role"],
                        parsed["content"],
                    ),
                    session_id=parsed["session_id"],
                    turn_index=parsed["turn_index"],
                    role=parsed["role"],
                    date=parsed["date"],
                    text=content,
                    score=float(row.get("score") or 0.0),
                )
            )
        return snippets

    def close(self) -> None:
        if self.provider is not None:
            self.provider.shutdown()


class OpenVikingCaseIndex:
    """Mode A (raw-turn leaf retrieval) OpenViking adapter for LongMemEval.

    OpenViking's session/extract memory pipeline is lossy by design and drops the
    specific facts LongMemEval asks about, so it cannot answer the benchmark. This
    adapter instead stores the raw conversation turns as leaf documents in an
    isolated per-case scope and retrieves them directly, which is apples-to-apples
    with the flat LanceDB variants.

    The critical step is ``content/reindex`` with ``mode=vectors_only``: plain
    ``content/write`` only embeds the rolled-up directory abstract, so a scoped
    ``search/find`` returns just that summary and never the answer turn. Reindexing
    forces per-leaf vector embeddings (no LLM cost), after which a scoped
    ``search/find`` returns the leaf chunks. Isolation is by ``target_uri`` scope;
    no global store wipe is required.
    """

    def __init__(
        self,
        case: LongMemEvalCase,
        root: Path,
        *,
        wait_every_write: bool = False,
        turns_per_doc: int = 4,
        scope_name: str = "",
        index_wait_s: float = 900.0,
    ) -> None:
        self.case = case
        self.root = root / "openviking-memory"
        self.provider = None
        self.scope = ""
        # Retained for call-site compatibility; ingestion no longer waits per write.
        self.wait_every_write = wait_every_write
        self.turns_per_doc = max(1, turns_per_doc)
        self.index_wait_s = max(60.0, index_wait_s)
        self.scope_name = scope_name or openviking_scope_name(
            case,
            OPENVIKING_INDEX_PREFIX,
            self.turns_per_doc,
        )

    def connect(self) -> Any:
        plugin = load_openviking_plugin()
        provider = plugin.OpenVikingMemoryProvider()
        provider.initialize(f"longmemeval-{self.case.question_id}")
        if getattr(provider, "_client", None) is None:
            raise RuntimeError(
                "OpenViking benchmark requires a reachable OpenViking server. "
                "Start openviking-server, set OPENVIKING_ENDPOINT if it is not "
                "http://127.0.0.1:1933, then rerun or omit --variants openviking."
            )

        user = str(getattr(provider, "_user", "") or "default")
        self.root.mkdir(parents=True, exist_ok=True)
        self.scope = f"viking://user/{user}/memories/{self.scope_name}/"
        self.provider = provider
        return provider

    def _long_request(self, method: str, path: str, *, json_body: dict | None = None,
                      params: dict | None = None, timeout: float) -> Any:
        """Issue a request that may exceed the plugin client's 30s default timeout.

        The plugin's ``_VikingClient`` hardcodes a 30s timeout and forwards no
        override, but reindex and system/wait can run longer, so call httpx
        directly while reusing the client's URL/header/parse helpers.
        """
        client = self.provider._client
        resp = client._httpx.request(
            method,
            client._url(path),
            json=json_body,
            params=params,
            headers=client._headers(),
            timeout=timeout,
        )
        return client._parse_response(resp)

    def _delete_scope(self) -> None:
        """Remove any prior contents of this case's scope for idempotent re-runs."""
        try:
            self._long_request(
                "DELETE",
                "/api/v1/fs",
                params={"uri": self.scope, "recursive": "true"},
                timeout=120.0,
            )
        except Exception:
            pass

    def ingest(self) -> None:
        provider = self.connect()
        self._delete_scope()

        turns = []
        for session_index, session in enumerate(self.case.haystack_sessions):
            session_id = _session_id(self.case, session_index)
            date = _session_date(self.case, session_index)
            for turn_index, turn in enumerate(session):
                role = turn.get("role") or ""
                content = turn.get("content") or ""
                turn_id = make_benchmark_turn_id(session_id, turn_index, role, content)
                formatted = format_turn_content(date, session_id, turn_index, role, content)
                turns.append((formatted, turn_id))

        docs = []
        for start_index in range(0, len(turns), self.turns_per_doc):
            chunk = turns[start_index : start_index + self.turns_per_doc]
            doc_index = (start_index // self.turns_per_doc) + 1
            uri = f"{self.scope}chunk_{doc_index:04d}.md"
            content = ENTRY_DELIMITER.join(formatted for formatted, _ in chunk)
            docs.append((uri, content, chunk[0][1]))

        # Write all docs without waiting (fast); content is stored synchronously.
        for uri, content, turn_id in docs:
            self._write_doc(provider, uri, content, turn_id)

        # Force per-leaf vector embeddings so scoped search returns leaf chunks
        # rather than only the directory abstract. reindex(wait=True) blocks until
        # the vectors are rebuilt and committed, which is all leaf retrieval needs.
        #
        # We deliberately do NOT call /system/wait here: that also blocks on the
        # Semantic queue, which generates directory abstracts/overviews via a VLM
        # (gpt-5.4-mini) — work this benchmark never reads. Waiting on it roughly
        # tripled per-case ingest time for no retrieval benefit. Those summaries
        # still run in the background per write; they are simply not awaited.
        self._long_request(
            "POST",
            "/api/v1/content/reindex",
            json_body={"uri": self.scope, "mode": "vectors_only", "wait": True},
            timeout=self.index_wait_s,
        )

    def _write_doc(self, provider: Any, uri: str, content: str, turn_id: str) -> Any:
        payload = {"uri": uri, "content": content, "mode": "create", "wait": False}
        try:
            return provider._client.post("/api/v1/content/write", payload)
        except Exception as exc:
            if not is_openviking_already_exists(exc):
                raise RuntimeError(f"OpenViking failed to store benchmark chunk {turn_id}: {exc}") from exc

        payload["mode"] = "replace"
        try:
            return provider._client.post("/api/v1/content/write", payload)
        except Exception as exc:
            raise RuntimeError(
                f"OpenViking failed to overwrite existing benchmark chunk {turn_id}: {exc}"
            ) from exc

    def retrieve(self, query: str, *, limit: int) -> list[RetrievedSnippet]:
        payload = self.provider._client.post(
            "/api/v1/search/find",
            {
                "query": query,
                "target_uri": self.scope,
                "limit": max(1, limit),
            },
        )
        hits = extract_openviking_hits(payload)
        snippets: list[RetrievedSnippet] = []
        seen_docs: set[str] = set()
        for rank, item in enumerate(hits, start=1):
            uri = str(item.get("uri") or "")
            # Search hits may address sub-chunks (chunk_0007.md#chunk_0002); read
            # the parent leaf document once to recover its Session:/Turn: lines.
            doc_uri = uri.split("#", 1)[0]
            if doc_uri and doc_uri in seen_docs:
                continue
            if doc_uri:
                seen_docs.add(doc_uri)
            content = self._read_content(doc_uri)
            if not content:
                content = str(
                    item.get("content")
                    or item.get("text")
                    or item.get("abstract")
                    or item.get("summary")
                    or ""
                )
            snippets.extend(
                openviking_snippets_from_content(
                    content,
                    query=query,
                    source_id=doc_uri or f"openviking-{rank}",
                    base_score=safe_float(item.get("score")),
                )
            )
            if len(snippets) >= limit:
                break
        return snippets[: max(1, limit)]

    def _read_content(self, uri: str) -> str:
        if not uri:
            return ""
        uri = uri.split("#", 1)[0]
        try:
            payload = self.provider._client.get(
                "/api/v1/content/read",
                params={"uri": uri, "raw": True},
            )
        except Exception:
            return ""
        result = payload.get("result", payload) if isinstance(payload, dict) else payload
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return str(result.get("content") or result.get("text") or "")
        return ""

    def close(self) -> None:
        if self.provider is not None:
            self.provider.shutdown()


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
        rows = []
        for session_index, session in enumerate(self.case.haystack_sessions):
            session_id = _session_id(self.case, session_index)
            date = _session_date(self.case, session_index)
            for turn_index, turn in enumerate(session):
                role = turn.get("role") or ""
                content = turn.get("content") or ""
                turn_id = make_benchmark_turn_id(session_id, turn_index, role, content)
                rows.append(
                    {
                        "id": turn_id,
                        "kind": "turn",
                        "content": format_turn_content(date, session_id, turn_index, role, content),
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
            snippets.append(
                RetrievedSnippet(
                    id=str(row.get("id") or ""),
                    session_id=str(row.get("session_id") or ""),
                    turn_index=int(row.get("turn_index") or 0),
                    role=str(row.get("role") or ""),
                    date=extract_turn_date(str(row.get("content") or "")),
                    text=str(row.get("content") or ""),
                    score=first_score(row),
                )
            )
        return snippets

    def close(self) -> None:
        if self.provider is not None:
            self.provider.shutdown()


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


def load_holographic_plugin() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    hermes_agent = root.parent / "hermes-agent"
    ensure_hermes_agent_path(hermes_agent)
    module_name = "hermes_holographic_benchmark_plugin"
    plugin_root = hermes_agent / "plugins" / "memory" / "holographic"
    spec = importlib.util.spec_from_file_location(
        module_name,
        plugin_root / "__init__.py",
        submodule_search_locations=[str(plugin_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Holographic plugin from {plugin_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_openviking_plugin() -> ModuleType:
    global _OPENVIKING_PLUGIN_MODULE
    if _OPENVIKING_PLUGIN_MODULE is not None:
        return _OPENVIKING_PLUGIN_MODULE
    root = Path(__file__).resolve().parents[2]
    hermes_agent = root.parent / "hermes-agent"
    ensure_hermes_agent_path(hermes_agent)
    module_name = "hermes_openviking_benchmark_plugin"
    plugin_root = hermes_agent / "plugins" / "memory" / "openviking"
    spec = importlib.util.spec_from_file_location(
        module_name,
        plugin_root / "__init__.py",
        submodule_search_locations=[str(plugin_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load OpenViking plugin from {plugin_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _OPENVIKING_PLUGIN_MODULE = module
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
) -> list[dict[str, str]]:
    context = format_retrieved_context(snippets)
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
    answer_sessions = set(answer_session_ids)
    answer_turns = set(answer_turn_ids)
    retrieved_sessions = [s.session_id for s in snippets]
    retrieved_turns = [s.id for s in snippets]
    return {
        "retrieved_session_ids": retrieved_sessions,
        "retrieved_turn_ids": retrieved_turns,
        "session_hit": bool(answer_sessions.intersection(retrieved_sessions))
        if answer_sessions
        else None,
        "turn_hit": bool(answer_turns.intersection(retrieved_turns)) if answer_turns else None,
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
        correct = sum(1 for r in rows if r.get("correct"))
        session_hits = [
            r["retrieval"]["session_hit"]
            for r in rows
            if r.get("retrieval", {}).get("session_hit") is not None
        ]
        turn_hits = [
            r["retrieval"]["turn_hit"]
            for r in rows
            if r.get("retrieval", {}).get("turn_hit") is not None
        ]
        by_type: dict[str, dict[str, Any]] = {}
        for question_type in sorted({r["question_type"] for r in rows}):
            subset = [r for r in rows if r["question_type"] == question_type]
            by_type[question_type] = {
                "cases": len(subset),
                "accuracy": _rate(sum(1 for r in subset if r.get("correct")), len(subset)),
            }
        by_variant[variant.name] = {
            "cases": len(rows),
            "accuracy": _rate(correct, len(rows)),
            f"retrieval_session_hit@{args.top_k}": _rate(sum(1 for v in session_hits if v), len(session_hits)),
            f"retrieval_turn_hit@{args.top_k}": _rate(sum(1 for v in turn_hits if v), len(turn_hits)),
            "latency_s_avg": average_latency(rows),
            "usage_total": sum_usage(rows),
            "accuracy_by_question_type": by_type,
        }
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
        "| Variant | Cases | Accuracy | Session Hit | Turn Hit | Total Latency Avg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    session_key = f"retrieval_session_hit@{summary['top_k']}"
    turn_key = f"retrieval_turn_hit@{summary['top_k']}"
    for name, stats in summary["by_variant"].items():
        lines.append(
            "| {name} | {cases} | {acc:.3f} | {sess:.3f} | {turn:.3f} | {lat:.3f}s |".format(
                name=name,
                cases=stats["cases"],
                acc=stats["accuracy"],
                sess=stats[session_key],
                turn=stats[turn_key],
                lat=stats["latency_s_avg"]["total_s"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def format_retrieved_context(snippets: list[RetrievedSnippet]) -> str:
    if not snippets:
        return "(none)"
    lines = []
    for i, snippet in enumerate(snippets, start=1):
        text = " ".join(snippet.text.split())
        if len(text) > 1200:
            text = text[:1197] + "..."
        label = f"[{i}] session={snippet.session_id} turn={snippet.turn_index} role={snippet.role}"
        if snippet.date:
            label += f" date={snippet.date}"
        lines.append(f"{label}\n{text}")
    return "\n\n".join(lines)


def average_latency(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("retrieval_s", "answer_s", "judge_s", "total_s")
    return {key: _rate(sum(float(r["latency_s"].get(key, 0.0)) for r in rows), len(rows)) for key in keys}


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


def extract_openviking_hits(payload: Any) -> list[dict[str, Any]]:
    """Extract URI-bearing search hits from multiple OpenViking response shapes."""
    root = payload.get("result", payload) if isinstance(payload, dict) else payload
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            uri = value.get("uri")
            if isinstance(uri, str) and uri.startswith("viking://") and uri not in seen:
                seen.add(uri)
                hits.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(root)

    def sort_key(item: dict[str, Any]) -> float:
        score = safe_float(item.get("score"))
        return score if score is not None else float("-inf")

    if any(safe_float(item.get("score")) is not None for item in hits):
        hits.sort(key=sort_key, reverse=True)
    return hits


def openviking_scope_name(case: LongMemEvalCase, prefix: str, turns_per_doc: int) -> str:
    return safe_name(f"{prefix or OPENVIKING_INDEX_PREFIX}-tpd{max(1, turns_per_doc)}-{case.question_id}")


def openviking_snippets_from_content(
    content: str,
    *,
    query: str,
    source_id: str,
    base_score: float | None,
) -> list[RetrievedSnippet]:
    parts = [part.strip() for part in (content or "").split(ENTRY_DELIMITER) if part.strip()]
    if not parts and content:
        parts = [content]
    query_terms = tokenize(query)
    ranked: list[tuple[float, RetrievedSnippet]] = []
    for index, part in enumerate(parts, start=1):
        parsed = parse_turn_content(part)
        turn_text = parsed["content"] or part
        local_score = lexical_score(query_terms, part)
        score = (base_score if base_score is not None else 0.0) + (local_score / 1000.0)
        if parsed["session_id"]:
            snippet_id = make_benchmark_turn_id(
                parsed["session_id"],
                parsed["turn_index"],
                parsed["role"],
                turn_text,
            )
        else:
            snippet_id = f"{source_id}#{index}"
        ranked.append(
            (
                local_score,
                RetrievedSnippet(
                    id=snippet_id,
                    session_id=parsed["session_id"],
                    turn_index=parsed["turn_index"],
                    role=parsed["role"],
                    date=parsed["date"],
                    text=part,
                    score=score,
                ),
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [snippet for _, snippet in ranked]


def is_openviking_already_exists(exc: Exception) -> bool:
    message = str(exc)
    return "ALREADY_EXISTS" in message or "File already exists" in message


def format_turn_content(date: str, session_id: str, turn_index: int, role: str, content: str) -> str:
    return (
        f"Date: {date}\n"
        f"Session: {session_id}\n"
        f"Turn: {turn_index}\n"
        f"Role: {role}\n"
        f"Content: {content}"
    )


def parse_turn_content(content: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "date": "",
        "session_id": "",
        "turn_index": 0,
        "role": "",
        "content": "",
    }
    current = None
    body_lines = []
    for line in (content or "").splitlines():
        if line.startswith("Date: "):
            fields["date"] = line.removeprefix("Date: ").strip()
        elif line.startswith("Session: "):
            fields["session_id"] = line.removeprefix("Session: ").strip()
        elif line.startswith("Turn: "):
            try:
                fields["turn_index"] = int(line.removeprefix("Turn: ").strip())
            except ValueError:
                fields["turn_index"] = 0
        elif line.startswith("Role: "):
            fields["role"] = line.removeprefix("Role: ").strip()
        elif line.startswith("Content: "):
            current = "content"
            body_lines.append(line.removeprefix("Content: "))
        elif current == "content":
            body_lines.append(line)
    fields["content"] = "\n".join(body_lines).strip()
    return fields


def extract_turn_date(content: str) -> str:
    match = re.search(r"^Date:\s*(.*)$", content or "", flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def make_benchmark_turn_id(session_id: str, turn_index: int, role: str, content: str) -> str:
    digest = hashlib.sha256(
        f"{session_id}\0{turn_index}\0{role}\0{content}".encode("utf-8")
    ).hexdigest()[:24]
    return f"turn_lme_{digest}"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


HOLOGRAPHIC_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
    "with",
}


def holographic_fts_query(query: str) -> str:
    """Build a permissive FTS5 query for the holographic plugin.

    The plugin passes queries directly to SQLite FTS5 MATCH. Raw natural
    questions can be invalid FTS syntax and usually over-constrain matching.
    """
    terms = []
    seen = set()
    for token in tokenize(query):
        if len(token) < 3 or token in HOLOGRAPHIC_QUERY_STOPWORDS:
            continue
        term = f"{token}*"
        if term not in seen:
            terms.append(term)
            seen.add(term)
    return " OR ".join(terms)


def lexical_score(query_terms: list[str], text: str) -> float:
    if not query_terms or not text:
        return 0.0
    counts = Counter(tokenize(text))
    score = 0.0
    for term in query_terms:
        score += counts.get(term, 0)
    unique_overlap = len(set(query_terms).intersection(counts))
    return score + (unique_overlap * 2.0)


def safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "case")
    return clean[:120] or "case"


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


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
