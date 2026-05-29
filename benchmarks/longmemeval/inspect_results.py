"""Inspect LongMemEval benchmark JSONL artifacts as readable Markdown."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases_jsonl",
        type=Path,
        help="Path to a benchmark cases.jsonl file.",
    )
    parser.add_argument(
        "--status",
        choices=["all", "correct", "incorrect"],
        default="incorrect",
        help="Which judged answers to include.",
    )
    parser.add_argument("--variant", default="", help="Only include one benchmark variant.")
    parser.add_argument("--question-type", default="", help="Only include one question_type.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum sampled records to print.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument(
        "--top-snippets",
        type=int,
        default=3,
        help="Number of retrieved snippets to show per record.",
    )
    parser.add_argument(
        "--max-answer-chars",
        type=int,
        default=900,
        help="Maximum characters for expected/actual/judge text.",
    )
    parser.add_argument(
        "--max-snippet-chars",
        type=int,
        default=700,
        help="Maximum characters per retrieved snippet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write Markdown to this path instead of stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.cases_jsonl)
    filtered = filter_rows(rows, args)
    sampled = sample_rows(filtered, limit=args.limit, seed=args.seed)
    markdown = format_markdown(sampled, rows=rows, filtered=filtered, args=args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        sys.stdout.write(markdown)
    return 0


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"cases JSONL not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if args.status == "correct" and row.get("correct") is not True:
            continue
        if args.status == "incorrect" and row.get("correct") is not False:
            continue
        if args.variant and row.get("variant") != args.variant:
            continue
        if args.question_type and row.get("question_type") != args.question_type:
            continue
        out.append(row)
    return out


def sample_rows(rows: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, limit)


def format_markdown(
    sampled: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]],
    filtered: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# LongMemEval Result Sample",
        "",
        f"- Source: `{args.cases_jsonl}`",
        f"- Total records: {len(rows)}",
        f"- Matching records: {len(filtered)}",
        f"- Sampled records: {len(sampled)}",
        f"- Status filter: `{args.status}`",
    ]
    if args.variant:
        lines.append(f"- Variant filter: `{args.variant}`")
    if args.question_type:
        lines.append(f"- Question type filter: `{args.question_type}`")
    lines.append("")

    lines.extend(format_variant_table(rows))
    lines.append("")

    if not sampled:
        lines.append("_No records matched the filters._")
        lines.append("")
        return "\n".join(lines)

    lines.extend(["## Sampled Records", ""])
    for index, row in enumerate(sampled, start=1):
        lines.extend(format_record(index, row, args))
    return "\n".join(lines)


def format_variant_table(rows: list[dict[str, Any]]) -> list[str]:
    variant_names = sorted({str(row.get("variant") or "") for row in rows if row.get("variant")})
    lines = [
        "## Variant Summary",
        "",
        "| Variant | Cases | Accuracy | Session Hit | Turn Hit | Retrieval Avg | Total Avg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if not variant_names:
        lines.append("| _(none)_ | 0 | 0.000 | 0.000 | 0.000 | 0.00s | 0.00s |")
        return lines

    for variant in variant_names:
        variant_rows = [row for row in rows if row.get("variant") == variant]
        lines.append(
            "| {variant} | {cases} | {accuracy:.3f} | {session_hit:.3f} | "
            "{turn_hit:.3f} | {retrieval_s:.2f}s | {total_s:.2f}s |".format(
                variant=escape_table_cell(variant),
                cases=len(variant_rows),
                accuracy=rate([row.get("correct") is True for row in variant_rows]),
                session_hit=rate_optional(
                    (row.get("retrieval") or {}).get("session_hit") for row in variant_rows
                ),
                turn_hit=rate_optional(
                    (row.get("retrieval") or {}).get("turn_hit") for row in variant_rows
                ),
                retrieval_s=average_float(
                    (row.get("latency_s") or {}).get("retrieval_s") for row in variant_rows
                ),
                total_s=average_float(
                    (row.get("latency_s") or {}).get("total_s") for row in variant_rows
                ),
            )
        )
    return lines


def format_record(index: int, row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    retrieval = row.get("retrieval") or {}
    latency = row.get("latency_s") or {}
    lines = [
        f"## {index}. {row.get('question_id') or '<no-id>'} | {row.get('variant')}",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Correct | `{row.get('correct')}` |",
        f"| Judge label | `{row.get('judge_label')}` |",
        f"| Question type | `{row.get('question_type') or ''}` |",
        f"| Session hit | `{retrieval.get('session_hit')}` |",
        f"| Turn hit | `{retrieval.get('turn_hit')}` |",
        f"| Total latency | `{float(latency.get('total_s') or 0.0):.2f}s` |",
        "",
        "**Question**",
        "",
        block(row.get("question") or "", args.max_answer_chars),
        "",
        "**Expected**",
        "",
        block(row.get("reference_answer") or "", args.max_answer_chars),
        "",
        "**Actual**",
        "",
        block(row.get("model_answer") or "", args.max_answer_chars),
        "",
        "**Judge Output**",
        "",
        block(row.get("judge_output") or "", args.max_answer_chars),
        "",
        "**Retrieved Snippets**",
        "",
    ]
    snippets = row.get("retrieved") or []
    if not snippets:
        lines.extend(["_No snippets recorded._", ""])
        return lines
    answer_sessions = set(row.get("answer_session_ids") or [])
    answer_turns = set(row.get("answer_turn_ids") or [])
    for rank, snippet in enumerate(snippets[: max(0, args.top_snippets)], start=1):
        session_id = str(snippet.get("session_id") or "")
        turn_id = str(snippet.get("id") or "")
        markers = []
        if session_id in answer_sessions:
            markers.append("answer-session")
        if turn_id in answer_turns:
            markers.append("answer-turn")
        marker_text = f" ({', '.join(markers)})" if markers else ""
        lines.extend(
            [
                f"### Snippet {rank}{marker_text}",
                "",
                f"- Session: `{session_id}`",
                f"- Turn: `{snippet.get('turn_index')}`",
                f"- Role: `{snippet.get('role') or ''}`",
                f"- Score: `{snippet.get('score')}`",
                "",
                block(snippet.get("text") or "", args.max_snippet_chars),
                "",
            ]
        )
    return lines


def block(text: str, max_chars: int) -> str:
    return "```text\n" + truncate(text, max_chars) + "\n```"


def truncate(text: str, max_chars: int) -> str:
    clean = str(text or "").strip()
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def rate_optional(values: Any) -> float:
    observed = [value for value in values if value is not None]
    if not observed:
        return 0.0
    return sum(1 for value in observed if value is True) / len(observed)


def average_float(values: Any) -> float:
    observed = []
    for value in values:
        try:
            observed.append(float(value or 0.0))
        except (TypeError, ValueError):
            continue
    if not observed:
        return 0.0
    return sum(observed) / len(observed)


def escape_table_cell(value: str) -> str:
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
