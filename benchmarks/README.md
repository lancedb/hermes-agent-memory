# Benchmarks

This directory contains benchmark harnesses for the LanceDB Hermes memory plugin.

## LongMemEval

`benchmarks/longmemeval/run.py` runs the [LongMemEval-S](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
long-conversation QA benchmark. For each case it builds an **isolated memory
store per variant**, ingests every conversation turn of the haystack, retrieves
the top-k turns for the question, has an answer model answer from those turns,
and has a judge model grade the answer against the reference.

It compares what a Hermes user actually has for long-term recall against the
LanceDB plugin's retrieval modes:

| Variant | What it is |
|---|---|
| `hermes-session-search` | **Baseline.** Hermes's built-in session store — SQLite FTS5 (BM25) over verbatim transcript messages, via the real `SessionDB` API. Zero-LLM retrieval. |
| `lancedb-vector` | LanceDB `mode=vector`: ANN over OpenAI `text-embedding-3-small`. |
| `lancedb-hybrid-rrf` | LanceDB `mode=hybrid`: vector + BM25 fused with Reciprocal Rank Fusion (equal-weight). |
| `lancedb-hybrid-linear` | Hybrid with a weighted linear combination of the vector + FTS scores, biased toward vector (0.85 vector / 0.15 FTS). Included to show the cost of RRF's equal-weight fusion. |
| `lancedb-hybrid-cross-encoder` | Hybrid + a cross-encoder reranking pass (local sentence-transformers model). |

`--variants all` (and the default) runs **those five**. Name a subset to run
fewer, e.g. `--variants lancedb-hybrid-rrf`.

A fifth variant, **`full-context`**, is recognized but deliberately *excluded*
from `all`: it feeds the entire ~110k-token haystack to the answer model (no
retrieval) as an accuracy ceiling. It's expensive and overflows a 128k context
window on most LongMemEval-S cases, so run it separately and capped (see below).

### Results (illustrative)

A 60-case stratified run (`--per-type 10`, top-k 5), answered by `gpt-5.4` and
judged by `gpt-5.4-mini`, on the plugin's shipped defaults (OpenAI
`text-embedding-3-small`; 17M cross-encoder). The LanceDB variants show 59 cases
— one case hit a transient API error and was skipped, and the run finished
anyway.

| Variant | Accuracy | Recall@5 | MRR@5 | Query p50 | Query p95 |
|---|---:|---:|---:|---:|---:|
| hermes-session-search | 0.533 | 0.659 | 0.639 | 0.002s | 0.004s |
| lancedb-vector | **0.661** | **0.795** | 0.682 | 0.207s | 1.046s |
| lancedb-hybrid-rrf | 0.610 | 0.650 | 0.635 | 0.235s | 0.934s |
| lancedb-hybrid-linear | 0.610 | 0.718 | 0.676 | 0.246s | 1.457s |
| lancedb-hybrid-cross-encoder | **0.678** | 0.754 | **0.689** | 0.702s | 3.062s |

Takeaways:

- **LanceDB beats Hermes's built-in baseline.** Pure vector (0.661 acc / 0.795
  recall) clearly tops `hermes-session-search` (0.533 / 0.659): semantic recall
  finds the right turns where lexical FTS misses paraphrases — and it's still
  ~0.2s per query.
- **RRF hybrid underperforms pure vector** (0.610 vs 0.661): equal-weight fusion
  lets noisy lexical matches displace good vector hits. This is why the shipped
  default is `vector`.
- **Vector-biased linear fusion recovers most of the recall** (0.718 vs RRF's
  0.650) — a better hybrid if you want lexical signal without RRF's penalty.
- **The cross-encoder leads on quality** (0.678 acc / 0.754 recall) at higher
  latency (~0.7s p50), and is opt-in (needs `sentence-transformers`).

Accuracy varies sharply by question type (single-session is easy; multi-session
and preference are hard for every method) — see the per-type matrix in the run's
`summary.md`:

| Question type | session-search | vector | hybrid-rrf | hybrid-linear | cross-encoder |
|---|---:|---:|---:|---:|---:|
| knowledge-update | 0.70 | 0.78 | 0.56 | 0.67 | 1.00 |
| multi-session | 0.10 | 0.50 | 0.30 | 0.40 | 0.40 |
| single-session-assistant | 0.60 | 0.90 | 0.90 | 0.90 | 0.80 |
| single-session-preference | 0.20 | 0.30 | 0.40 | 0.10 | 0.10 |
| single-session-user | 0.80 | 0.90 | 0.90 | 1.00 | 0.90 |
| temporal-reasoning | 0.80 | 0.60 | 0.60 | 0.60 | 0.90 |

> Illustrative only: n=60, and absolute accuracy tracks the answer model (here
> `gpt-5.4`). The point is the *relative* ordering of retrieval methods, which is
> stable. Reproduce with the [stratified run](#stratified-smoke-test-across-all-scenarios-recommended-start) below.

### What the harness does

- Stores LongMemEval turns as `kind=turn` rows; it does **not** run the plugin's
  fact extraction. This measures the retrieval substrate (verbatim recall), not
  the full memory-provider lifecycle.
- Strips the dataset's `has_answer` / answer labels before any model-facing
  prompt is built.
- For `hermes-session-search`, the natural-language question is turned into a
  disjunctive bag-of-words FTS5 query (`alice OR offsite OR ...`). Hermes's query
  sanitizer treats space-separated terms as implicit AND, which would force one
  message to contain every word; the bag-of-words form is the standard lexical-IR
  formulation and is applied uniformly (no per-question tuning).
- **Embeddings are cached per case.** The LanceDB variants embed the identical
  haystack, so the cache means it's embedded once per case rather than once per
  variant. (Query embeddings are *not* cached, so each variant's query latency
  is measured fairly.) The cache resets between cases to bound memory.
- **Reproducible config.** The harness reads the plugin's *shipped* defaults
  (`default_config.yaml` via `config.DEFAULTS`), **not** your personal
  `~/.hermes/config.yaml`. So results reflect the plugin as shipped and don't
  vary with your machine setup. To benchmark a different embedding/reranker
  model, edit `default_config.yaml` (the repo's single config source).

## Requirements

Install the runtime + dev tooling from this checkout:

```sh
uv sync --extra dev
```

That covers the LanceDB variants, which use the same runtime deps as the plugin:
`lancedb`, `openai`, `pyyaml` (all already in the project dependencies).

A few extra requirements depending on what you run:

- **`OPENAI_API_KEY`** — embeddings use OpenAI `text-embedding-3-small`. The
  runner auto-loads the key from the repo `.env` or `~/.hermes/.env` if it isn't
  already exported.
- **A sibling Hermes checkout at `../hermes-agent`** — required for both
  `hermes-session-search` (which drives Hermes's `SessionDB`) and the answer/judge
  LLM calls (`agent.auxiliary_client`). If Hermes lives elsewhere, run from a
  workspace where `../hermes-agent` exists or add it to `PYTHONPATH`. Real
  answer/judge calls need Hermes provider credentials (`hermes setup`).
- **`sentence-transformers`** (pulls in **`torch`, ~2 GB**) — needed *only* for
  `lancedb-hybrid-cross-encoder` (the cross-encoder reranker). It is not a
  project dependency. Add it just for the run with `uv run --with
  sentence-transformers ...`. If it's missing, that variant silently falls back
  to unranked hybrid.

The embedding cache and (when selected) the cross-encoder reranker are loaded
once at startup and reused across cases.

## Dataset

Download the cleaned LongMemEval-S dataset (500 cases):

```sh
uv run benchmarks/longmemeval/run.py \
  --download \
  --dataset-path benchmarks/data/longmemeval_s_cleaned.json \
  --limit 0
```

The dataset is large, so `benchmarks/data/` is git-ignored.

## Running

### Stratified smoke test across all scenarios (recommended start)

LongMemEval-S has six question types. `--per-type N` samples N cases of **each**
type, so you cover every scenario with a small, balanced run. `--per-type 3` ≈ 18
cases:

```sh
uv run --with sentence-transformers benchmarks/longmemeval/run.py \
  --per-type 3 \
  --variants all \
  --top-k 5 \
  --answer-model gpt-4o-mini \
  --judge-model gpt-4o-mini \
  --output-dir benchmarks/runs/strat-3
```

This runs the five comparison variants across all six types. Drop
`--with sentence-transformers` if you omit the cross-encoder variant. Scale up by
raising `--per-type` to `5`, `10`, or `30` (the smallest type, single-session-
preference, has 30 cases, so that is the per-type ceiling). `--per-type` overrides
`--limit`.

### Full-context ceiling (run separately)

`full-context` is the accuracy ceiling. Run it on its own, capped, with the
overflow guard set to your model's window so oversized cases are recorded rather
than crashing:

```sh
uv run benchmarks/longmemeval/run.py \
  --per-type 1 \
  --variants full-context \
  --top-k 5 \
  --answer-model gpt-4o-mini \
  --full-context-token-budget 120000 \
  --output-dir benchmarks/runs/ceiling
```

Cases whose estimated answer prompt exceeds `--full-context-token-budget`
(~chars/4) are marked `overflow` and counted incorrect without an LLM call. Set
the budget to `0` to disable the guard (assumes a large-context model).

### Key flags

| Flag | Default | Purpose |
|---|---|---|
| `--per-type N` | `0` (off) | Stratified: up to N cases per question type. Overrides `--limit`. |
| `--limit N` | `25` | Global case ceiling (when not using `--per-type`). |
| `--variant-limit NAME=N` | — | Cap one variant below the global count, e.g. `--variant-limit full-context=1`. Repeatable. |
| `--top-k N` | `10` | Snippets retrieved per question. |
| `--full-context-token-budget N` | `0` | Overflow guard for `full-context` (0 = off). |
| `--answer-model` / `--judge-model` | Hermes default | Answer / judge model. `--answer-provider` / `--judge-provider` for explicit routing. |
| `--batch-size N` | `4` | Concurrency for answer/judge calls. Lower for rate limits. |
| `--question-types a,b` | — | Restrict to specific question types. |

Provider/model routing uses Hermes `agent.auxiliary_client`, so configured
providers, endpoints, and model aliases apply. The runner prepares retrieval
synchronously, then sends answer and judge calls asynchronously in batches.

## Outputs & metrics

Each run writes to `--output-dir`:

- `cases.jsonl` — one record per (case, variant): retrieved snippets, answer,
  reference answer, judge verdict, retrieval diagnostics, per-phase latency,
  token usage, and `overflow`.
- `summary.json` — full aggregate, overall and **per question type**.
- `summary.md` — human-readable table + an accuracy-by-question-type matrix.

Per variant (and per question type):

- **Accuracy** — judge yes/no on the final answer. The headline QA metric.
- **Recall@k** (`turn_recall@k`) — fraction of gold turns retrieved. ~60% of
  LongMemEval-S cases have multiple gold turns (median 2), so graded recall is
  more informative than a boolean hit.
- **MRR@k** (`mrr@k`) — reciprocal rank of the first gold turn. Sensitive to
  reranking even when accuracy ties (it sees rank 3 → 1 moves that hit rate
  can't).
- **Hit rates** — `retrieval_turn_hit@k` (≥1 gold turn retrieved) and
  `retrieval_session_hit@k` (≥1 turn from a gold session).
- **Latency** — reported split:
  - `query_s` — per-recall latency (the number a user feels).
  - `ingest_s` — one-time store-build cost (amortized in production; with
    embedding caching it's paid once per case across the LanceDB variants).
  - `total_s = query_s + answer_s` — **excludes** the judge (benchmark-only
    grading) and ingest (one-time). Reported as avg / p50 / p95.
- **Tokens/question** and **overflow_cases**.

> Note: `full-context` returns the whole transcript unranked, so its Recall@k /
> MRR are not meaningful — read only its accuracy and token cost.

## Inspect results

Sample human-readable expected vs actual answers from a `cases.jsonl`:

```sh
uv run benchmarks/longmemeval/inspect_results.py \
  benchmarks/runs/strat-3/cases.jsonl \
  --status incorrect \
  --limit 10 \
  --seed 7 \
  --output benchmarks/runs/strat-3/incorrect-sample.md
```

Useful filters:

```sh
# Correct answers for one variant
uv run benchmarks/longmemeval/inspect_results.py \
  benchmarks/runs/strat-3/cases.jsonl \
  --status correct --variant lancedb-hybrid-rrf --limit 10

# Short incorrect sample to stdout with only the top retrieved snippet
uv run benchmarks/longmemeval/inspect_results.py \
  benchmarks/runs/strat-3/cases.jsonl \
  --status incorrect --top-snippets 1 --limit 5
```

Each sampled record shows question id, variant, correctness, judge label,
retrieval hit flags, latency, question, expected/actual answers, judge output,
and the top retrieved snippets (those from a gold session/turn are marked).

## Zero-token preflight

Validate the harness end-to-end with mocked LLM calls (no credentials, no spend):

```sh
uv run --extra dev python -m pytest tests/test_longmemeval_benchmark.py
```

This covers dataset parsing, label stripping, variant expansion, stratified
sampling, retrieval metrics (Recall@k / MRR), per-question-type rollups, the
embedding cache, full-context overflow handling, judge parsing, and JSONL /
summary writing.
</content>
