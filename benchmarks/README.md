# Benchmarks

This directory contains benchmark harnesses for the LanceDB Hermes memory plugin.

## LongMemEval

`benchmarks/longmemeval/run.py` runs LongMemEval-S question answering over an
isolated memory store per case and variant. It compares:

- `hermes-builtin-memory`: Hermes-style `MEMORY.md` prompt memory with no index
- `hermes-holographic`: Hermes bundled SQLite/FTS5/HRR holographic memory provider
- `openviking`: Hermes bundled OpenViking memory provider
- `markdown-lexical`: markdown/file memory corpus with simple lexical retrieval
- `lancedb-vector`: LanceDB `mode=vector`
- `lancedb-hybrid-rrf`: LanceDB `mode=hybrid` with LanceDB's default fusion
- `lancedb-hybrid-cross-encoder`: LanceDB `mode=hybrid` with cross-encoder reranking

The harness stores LongMemEval turns as `kind=turn` rows and does not run fact
extraction. It strips `has_answer` before any model-facing prompt or benchmark
memory corpus is built.

`hermes-builtin-memory` intentionally does not retrieve top-k. It writes a
bounded `MEMORY.md`-style snapshot using Hermes's default 2,200 character memory
budget, then feeds the included entries as no-index prompt memory. This is the
closest harness approximation of Hermes's built-in memory behavior, and should
not be interpreted as a search backend.

`hermes-holographic` uses `hrr_dim=4096` in the benchmark harness. Hermes's
provider default is lower, but LongMemEval cases can ingest hundreds of turns
into one category, so the benchmark uses a larger HRR space to reduce
signal-to-noise warnings during retrieval.

`openviking` stores each benchmark case as raw conversation turns in a
deterministic, isolated OpenViking memory scope and searches only that scope
(leaf-retrieval / "Mode A"). This is apples-to-apples with the flat LanceDB
variants: it compares OpenViking's vector retrieval over the same raw turns,
not its memory-extraction product.

OpenViking's designed session-commit/extract pipeline ("Mode B") is **not** used
here: that pipeline is lossy by construction (it curates a profile / preferences
/ entities summary and discards most specific statements), so the verbatim facts
LongMemEval asks about are frequently never stored and cannot be retrieved. The
raw-turn path keeps every turn.

The ingestion sequence is: write all turn-chunk documents with `wait=False`
(fast), then call `POST /api/v1/content/reindex {mode: vectors_only}` to force
per-leaf vector embeddings, then `POST /api/v1/system/wait` to drain the index
queues. The reindex step is essential: without it, plain `content/write` only
embeds the rolled-up directory abstract, so a scoped `search/find` returns just
that summary and never the answer turn (this was the original adapter's silent
zero-recall bug). Retrieval is a scoped `search/find` with no `level` filter,
which returns the leaf chunk documents.

Isolation is per-case via the scope `target_uri`; no global store wipe is
required (re-running a case wipes only that case's own scope first, for
idempotency).

## Install Dependencies

Install the repo dependencies plus the dev tools used by the benchmark tests:

```sh
uv sync --extra dev
```

The real LanceDB benchmark variants use the same runtime dependencies as the
plugin itself:

- `lancedb`
- `openai`
- `pyyaml`

Those are already listed in the project dependencies, so `uv sync --extra dev`
is enough for local benchmark runs from this checkout.

The OpenViking variant uses the Hermes OpenViking plugin from the sibling
Hermes checkout and requires a reachable OpenViking server. Start it from this
repo with your `.env` loaded:

```sh
uv pip install openviking
set -a
source .env
set +a
uv run openviking-server --host 127.0.0.1 --port 1933
```

If your server requires auth, also set `OPENVIKING_API_KEY`. If OpenViking is
not running, omit `openviking` from `--variants`. To restart the server, stop
the running process with `Ctrl-C` and run the same command again.

### Disable VLM summarization (strongly recommended)

OpenViking's leaf-retrieval path here does not use the directory abstracts /
overviews that OpenViking generates with a VLM (`gpt-5.4-mini`) on every write.
Leaving the VLM configured roughly triples per-case ingest time (≈100s vs ≈35s)
and spends summarization tokens for nothing. Remove the `vlm` block from your
OpenViking config (`~/.openviking/ov.conf`) and restart the server:

```jsonc
{
  "storage": { "workspace": "/Users/<you>/.openviking/data" },
  "embedding": {
    "dense": {
      "provider": "openai",
      "model": "text-embedding-3-small",
      "api_key": "${OPENAI_API_KEY}",
      "api_base": "https://api.openai.com/v1",
      "dimension": 1536
    }
  }
  // no "vlm" block — semantic summaries no-op with "VLM not available"
}
```

The server logs `VLM not available, using empty summary` and skips the LLM call;
writes, vector reindex, and scoped leaf retrieval are unaffected. The
`auto_generate_l0` / `auto_generate_l1` config flags look like they should do
this but are unwired in OpenViking 0.3.x, so removing the `vlm` block is the
working lever.

The LanceDB benchmark variants use OpenAI `text-embedding-3-small` by default,
matching the recommended OpenViking embedding config. The runner loads
`OPENAI_API_KEY` from the repo `.env` or `~/.hermes/.env` when it is not already
exported.

Cross-encoder reranking may pull additional model/runtime packages on first use.
If you only want the cheap smoke path, start with `hermes-builtin-memory`,
`hermes-holographic`, `lancedb-vector`, or `lancedb-hybrid-rrf` before running
`lancedb-hybrid-cross-encoder`.

For LanceDB variants, the embedding model is loaded once at benchmark startup
and reused across cases. The cross-encoder reranker is also loaded once when the
`lancedb-hybrid-cross-encoder` variant is selected.

The benchmark imports Hermes provider routing from a sibling Hermes checkout:

```text
../hermes-agent
```

That matches the repo test configuration. If Hermes is elsewhere, run from a
workspace where `../hermes-agent` exists or add that checkout to `PYTHONPATH`.

Real answer and judge calls require Hermes provider credentials, the same as a
normal Hermes session. Configure them with `hermes setup` or the environment
variables used by your selected provider.

## Dataset

Download the cleaned LongMemEval-S dataset:

```sh
uv run python benchmarks/longmemeval/run.py \
  --download \
  --dataset-path benchmarks/data/longmemeval_s_cleaned.json \
  --limit 0
```

The dataset is large, so `benchmarks/data/` is ignored by git.

## Cheap Smoke Test

Run one real case against the Hermes baselines and one LanceDB variant:

```sh
uv run python benchmarks/longmemeval/run.py \
  --dataset-path benchmarks/data/longmemeval_s_cleaned.json \
  --limit 1 \
  --top-k 5 \
  --variants hermes-builtin-memory,hermes-holographic,lancedb-hybrid-rrf \
  --batch-size 4 \
  --answer-max-tokens 128 \
  --judge-max-tokens 16 \
  --output-dir benchmarks/runs/smoke-1
```

This makes exactly two LLM calls per variant: one answer call and one judge call.
LanceDB retrieval runs locally, with embeddings generated by OpenAI
`text-embedding-3-small`.

For the minimum LanceDB-only check:

```sh
uv run python benchmarks/longmemeval/run.py \
  --dataset-path benchmarks/data/longmemeval_s_cleaned.json \
  --limit 1 \
  --top-k 5 \
  --variants lancedb-hybrid-rrf \
  --batch-size 4 \
  --answer-max-tokens 128 \
  --judge-max-tokens 16 \
  --output-dir benchmarks/runs/smoke-lancedb
```

## OpenViking Run

OpenViking ingestion (write turn-chunks + `vectors_only` reindex) runs inline
per case and takes roughly a minute or two per case, so no separate prebuild
step is required:

```sh
uv run python benchmarks/longmemeval/run.py \
  --dataset-path benchmarks/data/longmemeval_s_cleaned.json \
  --limit 10 \
  --top-k 5 \
  --variants openviking \
  --openviking-turns-per-doc 4 \
  --batch-size 4 \
  --answer-max-tokens 128 \
  --judge-max-tokens 16 \
  --output-dir benchmarks/runs/smoke-openviking-10
```

`--openviking-turns-per-doc` controls leaf-document granularity (default 4;
smaller is sharper but writes more documents). `--openviking-index-wait-s`
(default 900) caps how long ingestion waits for the reindex + queue drain.

If you want to separate indexing from retrieval, prebuild the per-case scopes
first and then run with `--openviking-use-prebuilt-index` (the prefix and
turns-per-doc must match between the two steps):

```sh
uv run python benchmarks/longmemeval/build_openviking_index.py \
  --dataset-path benchmarks/data/longmemeval_s_cleaned.json \
  --limit 10 \
  --turns-per-doc 4 \
  --output-dir benchmarks/runs/openviking-index-10
```

The script writes deterministic per-case scopes such as
`viking://user/default/memories/longmemeval-tpd4-e47becba/` and an
`openviking-index-manifest.json` under the output directory.

OpenViking's `session_hit` / `turn_hit` recall numbers are directly comparable
to the LanceDB variants here because both index the same raw turns. Note that
OpenViking pays an embedding cost per leaf during reindex (same
`text-embedding-3-small` model as LanceDB) but no per-document summarization LLM
cost on this path.

## Full Matrix

```sh
uv run python benchmarks/longmemeval/run.py \
  --dataset-path benchmarks/data/longmemeval_s_cleaned.json \
  --limit 25 \
  --top-k 10 \
  --variants hermes-builtin-memory,hermes-holographic,openviking,lancedb-vector,lancedb-hybrid-rrf,lancedb-hybrid-cross-encoder \
  --openviking-use-prebuilt-index \
  --batch-size 4 \
  --answer-model <model> \
  --judge-model <model> \
  --output-dir benchmarks/runs/dev
```

Provider and model routing uses Hermes `agent.auxiliary_client.call_llm`, so the
same configured providers, endpoints, and model aliases apply here. Use
`--answer-provider`, `--answer-model`, `--judge-provider`, and `--judge-model`
when you want explicit overrides.

The runner prepares retrieval synchronously, then sends answer calls and judge
calls asynchronously in batches. `--batch-size 4` is the default; lower it if
your provider rate-limits concurrent requests, or raise it for faster runs when
your provider can handle more concurrency.

## Outputs

Each run writes:

- `cases.jsonl`: one record per case and variant, including retrieved snippets,
  answer, reference answer, judge verdict, retrieval hit diagnostics, latencies,
  and token usage when providers report it.
- `summary.json`: aggregate QA accuracy, accuracy by question type, retrieval
  hit rates, average latency, model/provider names, and token usage.
- `summary.md`: compact human-readable summary table.

The main retrieval diagnostic is `retrieval_session_hit@K`: whether any
retrieved snippet came from one of LongMemEval's `answer_session_ids`. When
turn-level `has_answer` labels are present, `retrieval_turn_hit@K` reports
whether the exact labeled turn was retrieved.

## Inspect Results

Use `inspect_results.py` to sample human-readable expected vs actual answers
from a `cases.jsonl` file:

```sh
uv run python benchmarks/longmemeval/inspect_results.py \
  benchmarks/runs/dev/cases.jsonl \
  --status incorrect \
  --limit 10 \
  --seed 7 \
  --output benchmarks/runs/dev/incorrect-sample.md
```

Useful filters:

```sh
# Sample correct answers for one variant
uv run python benchmarks/longmemeval/inspect_results.py \
  benchmarks/runs/dev/cases.jsonl \
  --status correct \
  --variant lancedb-hybrid-rrf \
  --limit 10

# Print a short incorrect sample to stdout with only the top retrieved snippet
uv run python benchmarks/longmemeval/inspect_results.py \
  benchmarks/runs/dev/cases.jsonl \
  --status incorrect \
  --top-snippets 1 \
  --limit 5
```

Each sampled record includes question id, variant, correctness, judge label,
retrieval session/turn-hit flags, latency, question, expected answer, actual
answer, judge output, and the top retrieved snippets. Snippets from an answer
session or labeled answer turn are marked inline.

## Zero-Token Preflight

Run the synthetic test harness with mocked LLM calls:

```sh
uv run pytest tests/test_longmemeval_benchmark.py
```

This validates dataset parsing, label stripping, variant expansion, markdown
ingestion, retrieval metrics, judge parsing, JSONL output, and summary writing
without API credentials.
