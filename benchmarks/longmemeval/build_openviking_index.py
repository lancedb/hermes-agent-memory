"""Build a deterministic OpenViking index for LongMemEval benchmark cases."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import run as benchmark


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("benchmarks/data/longmemeval_s_cleaned.json"),
        help="Path to longmemeval_s_cleaned.json.",
    )
    parser.add_argument("--limit", type=int, default=25, help="Maximum cases to index.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many matching cases first.")
    parser.add_argument(
        "--question-types",
        default="",
        help="Comma-separated LongMemEval question_type filter.",
    )
    parser.add_argument(
        "--turns-per-doc",
        type=int,
        default=4,
        help="Number of benchmark turns to pack into each OpenViking leaf document.",
    )
    parser.add_argument(
        "--index-prefix",
        default=benchmark.OPENVIKING_INDEX_PREFIX,
        help="OpenViking memory namespace prefix. Must match run.py --openviking-index-prefix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/runs/openviking-index"),
        help="Local directory for the build manifest and temporary benchmark state.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    benchmark.load_benchmark_env()
    cases = benchmark.load_cases(
        args.dataset_path,
        limit=args.limit,
        offset=args.offset,
        question_types=set(benchmark._split_csv(args.question_types)) or None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    turns_per_doc = max(1, int(args.turns_per_doc or 4))
    manifest: dict[str, Any] = {
        "dataset_path": str(args.dataset_path),
        "limit": args.limit,
        "offset": args.offset,
        "question_types": benchmark._split_csv(args.question_types),
        "index_prefix": args.index_prefix,
        "turns_per_doc": turns_per_doc,
        "cases": [],
    }

    log(args, f"Indexing {len(cases)} LongMemEval case(s) into OpenViking")
    for case_index, case in enumerate(cases, start=1):
        scope_name = benchmark.openviking_scope_name(case, args.index_prefix, turns_per_doc)
        total_turns = sum(len(session) for session in case.haystack_sessions)
        chunk_count = (total_turns + turns_per_doc - 1) // turns_per_doc
        prefix = f"[{case_index}/{len(cases)}] {case.question_id}"
        log(args, f"{prefix}: writing {total_turns} turn(s) as {chunk_count} indexed chunk document(s)")
        start = time.perf_counter()
        index = benchmark.OpenVikingCaseIndex(
            case,
            args.output_dir / "stores" / benchmark.safe_name(case.question_id),
            turns_per_doc=turns_per_doc,
            scope_name=scope_name,
        )
        try:
            index.ingest()
            scope = index.scope
        finally:
            index.close()
        elapsed = time.perf_counter() - start
        log(args, f"{prefix}: indexed in {elapsed:.2f}s at {scope}")
        manifest["cases"].append(
            {
                "question_id": case.question_id,
                "question_type": case.question_type,
                "scope": scope,
                "scope_name": scope_name,
                "turns": total_turns,
                "chunks": chunk_count,
                "elapsed_s": elapsed,
            }
        )

    manifest_path = args.output_dir / "openviking-index-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


def log(args: argparse.Namespace, message: str) -> None:
    if args.quiet:
        return
    print(message, file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
