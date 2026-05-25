# hermes-agent-memory

LanceDB-backed memory provider plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Embeds a workspace-scoped LanceDB table at `~/.hermes/lancedb/memories.lance` and exposes four tools to the agent: `lancedb_recall`, `lancedb_remember`, `lancedb_read`, `lancedb_forget`. Recall is hybrid (vector ANN + BM25, fused via RRF) with an optional Sentence Transformers cross-encoder reranker. Durable facts are extracted from sessions at pre-compress and session end. Everything runs in Hermes's Python process — no external service, no server.

## Features

- **Hybrid recall**: vector + BM25 fused with RRF; per-call switchable to pure vector or pure FTS.
- **Rerankers (optional)**: `cross-encoder/ettin-reranker-32m-v1` by default; configurable model and candidate-pool size.
- *Workspace isolation*: every row carries an `agent_workspace` tag and recall pre-filters by it.
- **Fact-first retrieval**: recall surfaces extracted facts; raw conversation turns are stored as provenance and used only as fallback.
- **Mid-session extraction**: facts are pulled out via an auxiliary LLM on `on_pre_compress` and `on_session_end`, so insights survive context compression.
- **Transparent forget**: preview candidates, then delete by exact ID.
- **Auto-compaction**: periodic `table.optimize(cleanup_older_than=...)` runs in the background to bound fragment and version-file growth from single-row writes.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) installed locally
- An LLM API key (OpenAI, OpenRouter, Anthropic, …)

Runtime dependencies installed into Hermes's venv: `lancedb >= 0.13`, `sentence-transformers >= 3.0`, `pyyaml`.

---

## Installation: users

Use this section if you want LanceDB memory in your own Hermes setup. If you plan to edit the plugin's source, jump to [Installation: developers](#installation--developers).

### 1. Install Hermes Agent

```sh
# macOS / Linux / WSL2
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# Windows (PowerShell)
iex (irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1)
```

The installer handles `uv`, Python 3.11, Node.js, ripgrep, ffmpeg, and (on Windows) MinGit. It clones Hermes into `~/.hermes/hermes-agent/` and symlinks the binary to `~/.local/bin/hermes`. After it finishes:

```sh
hermes doctor --fix     # repairs symlinks, dirs, etc.
hermes setup            # interactive: .env, API key, model picker
hermes doctor           # final sanity check
```

> If you have AWS credentials in your shell environment, `hermes doctor` may log a Bedrock `AccessDeniedException`. This is Hermes's provider auto-detection and is ignorable if you're using OpenAI / Anthropic / OpenRouter.

### 2. Install the plugin

Once published:

```sh
hermes plugins install lancedb/hermes-agent-memory
```

Until then, or if you want a local copy:

```sh
git clone https://github.com/lancedb/hermes-agent-memory ~/code/hermes-agent-memory
ln -sf ~/code/hermes-agent-memory ~/.hermes/plugins/lancedb
```

### 3. Install runtime dependencies into Hermes's venv

Hermes loads plugins inside its own Python interpreter. Install `lancedb` and `sentence-transformers` *there* — not into a separate venv.

```sh
# If Hermes is at a source checkout in ~/code/hermes-agent
uv pip install --python ~/code/hermes-agent/venv/bin/python3 lancedb sentence-transformers

# If you used the one-line installer
uv pip install --python ~/.hermes/hermes-agent/venv/bin/python3 lancedb sentence-transformers
```

This step is deliberately manual. `hermes memory setup` does not install these packages: `sentence-transformers` can exceed the setup-time install budget.

### 4. Activate the provider

```sh
hermes memory setup
# pick "lancedb"
```

This writes `memory.provider: lancedb` into `~/.hermes/config.yaml`, writes the plugin defaults under `plugins.lancedb`, and warms `BAAI/bge-small-en-v1.5` (~133 MB) into `~/.cache/huggingface/` so the first chat doesn't hang on a model download.

### 5. Verify

```sh
hermes plugins list           # should list "lancedb"
hermes memory status
hermes chat -q "Hello"        # agent.log should contain `lancedb provider initialized`
```

---

## Installation: developers

Use this section if you're working on the plugin's source.

### 1. Clone and create the dev venv

```sh
git clone https://github.com/lancedb/hermes-agent-memory ~/code/hermes-agent-memory
cd ~/code/hermes-agent-memory
uv sync --extra dev
```

`pyproject.toml` sets `[tool.uv] package = false` — `uv sync` only manages a venv for tests, lint, and ad-hoc imports. The plugin itself is loaded by Hermes from its directory, not pip-installed.

### 2. Symlink into Hermes's plugins directory

```sh
ln -sf ~/code/hermes-agent-memory ~/.hermes/plugins/lancedb
```

Edits to source files are picked up on the next Hermes session — no reinstall.

### 3. Install runtime deps into Hermes's venv

The dev venv only runs pytest / ruff. For end-to-end testing inside Hermes itself you still need the runtime deps installed against Hermes's Python:

```sh
uv pip install --python ~/code/hermes-agent/venv/bin/python3 lancedb sentence-transformers
```

### 4. Tests and lint

```sh
uv run pytest -v
uv run ruff check .
```

Add dev-only dependencies via:

```sh
uv add --dev pytest-mock
```

---

## Tools exposed to the agent

| Tool | Purpose |
|---|---|
| `lancedb_recall` | Hybrid (default) / vector / FTS recall over workspace memory. Returns IDs, snippets, scores, provenance turn IDs. |
| `lancedb_remember` | Store a durable fact (`preference`, `entity`, `event`, `case`, `pattern`, `general`). Deduplicated by content hash. |
| `lancedb_read` | Fetch one memory by ID, optionally with the full provenance turns it was extracted from. |
| `lancedb_forget` | Two-step: `action: preview` to list candidates by description, then `action: delete` with the exact ID. |

The provider's system-prompt block instructs the model when to use each tool: `lancedb_remember` only when the user explicitly asks to remember, `lancedb_forget preview` before any delete, etc.

---

## How recall works

1. The tool call enters `LanceDBMemoryProvider.recall()` with `mode`, `query`, `kind`, optional `category`, and `limit`.
2. A `WHERE` filter is built on workspace + user + kind + category, quoted via `quote_sql`, and passed as a prefilter.
3. The base retriever depends on `mode`:
   - `hybrid`: vector ANN + BM25, fused by LanceDB's built-in RRF.
   - `vector`: ANN over the `vector` column (normalized sentence-transformers embeddings).
   - `fts`: BM25 over the `content` column.
4. If `reranker.type` is `cross-encoder`, the candidate pool is expanded to `rerank_top_n`, the cross-encoder reorders the pool, and the top `top_k` are sliced in Python. The reranker instance is cached on the provider and warmed at `initialize()` so the first query doesn't pay the model-load cost.
5. The per-mode score column (`_distance`, `_score`, or `_relevance_score`) is explicitly projected to silence LanceDB's auto-projection deprecation warning and to keep score metadata in tool responses.

If hybrid fails (e.g. the FTS index hasn't been built yet), `recall()` falls back to pure vector with reranking disabled.

---

## Configuration reference

Defaults are local and keyless. Override under `plugins.lancedb` in `~/.hermes/config.yaml`:

```yaml
plugins:
  lancedb:
    retrieval:
      mode: hybrid              # hybrid | vector | fts
      top_k: 10
      search_kinds: [fact]      # which row kinds recall returns; "turn" rows are provenance/fallback
      reranker:
        type: rrf               # rrf | cross-encoder
                                #   rrf          : Reciprocal Rank Fusion. The built-in
                                #                   fusion strategy for hybrid mode.
                                #                   No-op for vector/fts (native distance
                                #                   / BM25 order applies).
                                #   cross-encoder: replace RRF / native ordering with a
                                #                   sentence-transformers cross-encoder.
        model: cross-encoder/ettin-reranker-32m-v1
        rerank_top_n: 50        # cross-encoder only: pull this many candidates from the
                                # base retriever, rerank, then slice to top_k. Larger =
                                # better recall, slower latency.
    extraction:
      enabled: true             # set false to disable LLM extraction at session boundaries
      min_turns: 3              # skip extraction for very short sessions
    embedding:
      provider: sentence-transformers
      model: BAAI/bge-small-en-v1.5
    maintenance:
      enabled: true             # background optimize() of the Lance table
      optimize_every_commits: 50
                                # trigger when table.version - last_optimized >= N
      cleanup_older_than_days: 7
                                # passed to table.optimize(cleanup_older_than=...): old
                                # version files are garbage-collected on each run
```

### Knob-by-knob

| Section | Key | Default | Notes |
|---|---|---|---|
| `retrieval` | `mode` | `hybrid` | Per-call override available via the `mode` parameter on `lancedb_recall`. |
| | `top_k` | `10` | Hard cap inside the retrieval layer is 50. |
| | `search_kinds` | `[fact]` | Recall surfaces facts; turn rows are stored as provenance and used as fallback when no facts match. |
| `retrieval.reranker` | `type` | `rrf` | `rrf` is a no-op for `mode: vector` / `mode: fts`: there's only one ranked list to return. |
| | `model` | `cross-encoder/ettin-reranker-32m-v1` | Any HuggingFace cross-encoder ID; lazy-loaded on first use. |
| | `rerank_top_n` | `50` | Enforced as `max(rerank_top_n, top_k)` so you never fetch fewer than you return. |
| `extraction` | `enabled` | `true` | Set `false` to skip the auxiliary LLM call. |
| | `min_turns` | `3` | Skip extraction when the user has spoken fewer than N turns. |
| `embedding` | `provider` | `sentence-transformers` | Only sentence-transformers is wired today. |
| | `model` | `BAAI/bge-small-en-v1.5` | Embedding dim must match the existing table: recreate the table if you change models against an existing store. |
| `maintenance` | `enabled` | `true` | Set `false` to disable auto-compaction. |
| | `optimize_every_commits` | `50` | Each `add` / `delete` advances `table.version`; auto-compaction fires when delta ≥ this value. |
| | `cleanup_older_than_days` | `7` | Passed as `timedelta(days=...)` to `table.optimize()`. Set `0` or negative to skip cleanup (compaction only). |

### Auxiliary LLM for extraction

`extraction` uses Hermes's auxiliary client. Point it at a cheaper model independent of your main chat model:

```yaml
auxiliary:
  lancedb_extraction:
    provider: openrouter
    model: google/gemini-3-flash
```

Hermes handles provider routing, fallback, and credit exhaustion.

---

## Storage layout

| Path | Contents |
|---|---|
| `~/.hermes/lancedb/memories.lance/` | LanceDB dataset directory (fragments, manifest, indexes). |
| `~/.hermes/lancedb/.last_optimize_version` | Sentinel file: `table.version` at the most recent successful `optimize()`. Used to decide when the next auto-compaction fires. |
| `~/.cache/huggingface/` | Sentence Transformers and cross-encoder model cache (managed by HuggingFace). |

The dataset is a single table named `memories` containing both fact and turn rows; the `kind` column distinguishes them. To poke at it directly:

```sh
uv run --project ~/.hermes/hermes-agent python -c "
import lancedb
db = lancedb.connect('~/.hermes/lancedb')
df = db.open_table('memories').to_pandas()
print(df[['kind', 'category', 'content']].head())
"
```

---

## Auto-compaction

Every `add` / `delete` on the table is a Lance commit. Without intervention, single-row writes (which dominate agent workloads) accumulate tiny fragments and version files indefinitely.

The plugin tracks `table.version` against the sentinel file at `~/.hermes/lancedb/.last_optimize_version` and runs `table.optimize(cleanup_older_than=timedelta(days=N))` in a daemon thread when the delta crosses `optimize_every_commits`. A non-blocking lock guarantees only one optimize runs at a time: re-triggers while one is in flight are skipped, and writers are never blocked.

If `maintenance.enabled: false`, none of this runs and the dataset will grow without bound.

---

## Troubleshooting

**`hermes plugins list` doesn't show `lancedb`.** Check the symlink: `ls -l ~/.hermes/plugins/lancedb` should resolve to this repo (or wherever you installed it).

**`lancedb_*` tools missing from the agent.** Confirm `memory.provider: lancedb` in `~/.hermes/config.yaml` and that `agent.log` contains `lancedb provider initialized` on session start.

**First recall hangs for 1–2 seconds.** First-time model load. After the embedding model (and, if enabled, the cross-encoder) are cached in `~/.cache/huggingface/`, subsequent runs are fast. With `reranker.type: cross-encoder`, the reranker is preloaded during `initialize()` to avoid this on the first user query.

**Table fragments / `.lance` directory growing.** Check `maintenance.enabled: true` and that `~/.hermes/lancedb/.last_optimize_version` is advancing across sessions. `agent.log` will show `lancedb optimize starting` when a compaction fires.

**Changed `embedding.model` and recall returns nothing.** The new model's dim doesn't match the existing column. Delete `~/.hermes/lancedb/memories.lance/` to recreate the table on the next session.

---

## License

Apache 2.0
