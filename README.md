# hermes-agent-memory

LanceDB-backed memory provider plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Embeds a workspace-scoped LanceDB table at `~/.hermes/lancedb/memories.lance` and exposes four tools to the agent: `lancedb_recall`, `lancedb_remember`, `lancedb_read`, `lancedb_forget`. Recall defaults to pure vector ANN over OpenAI embeddings, with an optional hybrid mode (vector + BM25, fused via RRF / linear / cross-encoder) per call or via config. Durable facts are extracted from sessions at pre-compress and session end. The memory store runs entirely in Hermes's Python process — no external memory service, no server (embeddings call your configured embeddings API).

## Features

- **Vector recall by default**: ANN over OpenAI embeddings — lightest, no reranker. Switch to hybrid (vector + BM25) per call or via config.
- **Hybrid fusion (configurable)**: default is RRF; `reranker.type: linear` does a weighted vector/FTS combination (`weight` biases toward vector); `reranker.type: cross-encoder` adds a reranking pass (default model `cross-encoder/ettin-reranker-17m-v1`, configurable). Only the cross-encoder needs `sentence-transformers`.
- *Workspace isolation*: every row carries an `agent_workspace` tag and recall pre-filters by it.
- **Fact-first retrieval**: recall surfaces extracted facts; raw conversation turns are stored as provenance and used only as fallback.
- **Mid-session extraction**: facts are pulled out via an auxiliary LLM on `on_pre_compress` and `on_session_end`, so insights survive context compression.
- **Transparent forget**: preview candidates, then delete by exact ID.
- **Auto-compaction**: periodic `table.optimize(cleanup_older_than=...)` runs in the background to bound fragment and version-file growth from single-row writes.

## Repo layout

This repo's primary purpose is the **LanceDB memory plugin**. The benchmark is auxiliary — it exists only to show the plugin is fast, cheap, and accurate. Hermes loads a plugin from its directory root (the repo-root `__init__.py` + `plugin.yaml`); the implementation lives in the `src/` subpackage, which the entry point re-exports. If you only want the plugin, everything you need is under `src/` — you never have to touch the benchmark.

| Path | What it is |
|---|---|
| `__init__.py` | Thin entry point — Hermes loads this; it re-exports the provider from `src/` and defines `register()`. |
| `plugin.yaml` | Hermes plugin manifest (name, hooks). |
| `src/` | **The plugin** — `provider.py`, `store.py`, `retrieval.py`, `config.py`, `embeddings.py`, `extraction.py`, `tools.py`, and `default_config.yaml` (the single source of defaults, copied into `~/.hermes/config.yaml`). |
| `benchmarks/` | **Benchmark only** (LongMemEval harness). Never imported by the plugin. |
| `tests/` | Test suite. |

The plugin and the benchmark are cleanly separated: the benchmark borrows the plugin via its loader but the plugin never imports anything under `benchmarks/`.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) installed locally
- An LLM API key (OpenAI, OpenRouter, Anthropic, …)

Runtime dependencies installed into Hermes's venv: `lancedb >= 0.33`, `openai`, `pyyaml`. Embeddings go through an OpenAI-compatible client — by default OpenAI (`text-embedding-3-small`, so an `OPENAI_API_KEY`), but you can point it at any OpenAI-compatible endpoint via config (see [Configuration reference](#configuration-reference)). The default install needs **no** local ML stack. Only if you opt into the cross-encoder reranker (`reranker.type: cross-encoder`) do you also need `sentence-transformers` — which pulls in **`torch` (~2 GB)**.

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

> [!NOTE]
> If you have AWS credentials in your shell environment, `hermes doctor` may log a Bedrock `AccessDeniedException`. This is Hermes's provider auto-detection and is ignorable if you're using OpenAI / Anthropic / OpenRouter.

### 2. Install the plugin

```sh
hermes plugins install lancedb/hermes-agent-memory
```

This shallow-clones `https://github.com/lancedb/hermes-agent-memory.git` into `~/.hermes/plugins/lancedb/` and renders `after-install.md` in a Rich panel telling you what's next. To pull updates later, re-run the same command.

### 3. Install runtime dependencies into Hermes's venv

Hermes loads plugins inside its own Python interpreter. Install `lancedb` and `openai` *there* — not into a separate venv.

```sh
# If Hermes is at a source checkout in /path/to/your/hermes-agent
uv pip install --python /path/to/your/hermes-agent/venv/bin/python3 lancedb openai pyyaml

# If you used the one-line installer
uv pip install --python ~/.hermes/hermes-agent/venv/bin/python3 lancedb openai pyyaml
```

Embeddings call the OpenAI API, so set `OPENAI_API_KEY` in your environment (or `~/.hermes/.env`). **Only if you enable the cross-encoder reranker** (`reranker.type: cross-encoder`) do you also need `sentence-transformers` — install it the same way (`uv pip install --python … sentence-transformers`). Note it pulls in **`torch` (~2 GB)** and can exceed the setup-time install budget of 120s; the default plugin needs neither.

### 4. Activate the provider

```sh
hermes memory setup
# pick "lancedb"
```

This writes `memory.provider: lancedb` into `~/.hermes/config.yaml` and writes the plugin defaults under `plugins.lancedb`. Embeddings use OpenAI `text-embedding-3-small` (1536-dim) via the API — there's no local model to download, but `OPENAI_API_KEY` must be set.

```sh
# ✓ LanceDB memory configured (embedding dim: 1536)
#  Start a new session to activate.
```

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
git clone https://github.com/lancedb/hermes-agent-memory /path/to/your/hermes-agent-memory
cd /path/to/your/hermes-agent-memory
uv sync --extra dev
```

`pyproject.toml` sets `[tool.uv] package = false` − `uv sync` only manages a venv for tests, lint, and ad-hoc imports. The plugin itself is loaded by Hermes from its directory, not pip-installed.

### 2. Symlink into Hermes's plugins directory

```sh
ln -sf /path/to/your/hermes-agent-memory ~/.hermes/plugins/lancedb
```

Edits to source files are picked up on the next Hermes session: no reinstall.

### 3. Install runtime deps into Hermes's venv

The dev venv only runs pytest / ruff. For end-to-end testing inside Hermes itself you still need the runtime deps installed against Hermes's Python:

```sh
uv pip install --python /path/to/your/hermes-agent/venv/bin/python3 lancedb openai pyyaml
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
| `lancedb_recall` | Vector (default) / hybrid recall over workspace memory. Returns IDs, snippets, scores, provenance turn IDs. |
| `lancedb_remember` | Store a durable fact (`preference`, `entity`, `event`, `case`, `pattern`, `general`). Deduplicated by content hash. |
| `lancedb_read` | Fetch one memory by ID, optionally with the full provenance turns it was extracted from. |
| `lancedb_forget` | Two-step: `action: preview` to list candidates by description, then `action: delete` with the exact ID. |

The provider's system-prompt block instructs the model when to use each tool: `lancedb_remember` only when the user explicitly asks to remember, `lancedb_forget preview` before any delete, etc.

---

## How recall works

`lancedb_recall` searches workspace memory and returns the top matches. You control two things:

| You choose | Options | Set in | Scope |
|---|---|---|---|
| **Search mode** | `vector` (default) · `hybrid` | `lancedb_recall`'s `mode` argument; default from key `plugins.lancedb.retrieval.mode` in `~/.hermes/config.yaml` | per call |
| **Hybrid fusion** | `rrf` · `linear` · `cross-encoder` | key `plugins.lancedb.retrieval.reranker.type` in `~/.hermes/config.yaml` | global |

Fusion only applies to `hybrid` mode and is config-only — the agent picks the *mode* per call, but the *fusion* is a global setting. To switch RRF → vector-biased `linear`, set `reranker.type: linear` (and `reranker.weight`) in `~/.hermes/config.yaml`.

> A pure-lexical `fts` mode (BM25 only, no embeddings) also exists as a valid `mode` value, but it's a niche escape hatch and **not recommended** — semantic recall lives in `vector`/`hybrid`, which is what these docs and the benchmark cover.

### Under the hood

1. Build a `WHERE` prefilter on workspace + user + kind + category.
2. Run the retriever for the chosen **mode**:
   - `vector` — ANN over `text-embedding-3-small` embeddings *(score: `_distance`)*.
   - `hybrid` — run a vector leg and a BM25 full-text leg, then fuse *(score: `_relevance_score`)*.
3. For `hybrid`, fuse by `reranker.type`:
   - `rrf` — Reciprocal Rank Fusion (rank-based, equal-weight legs).
   - `linear` — weighted vector + FTS scores; `reranker.weight` is the vector weight (0–1).
   - `cross-encoder` — rerank an oversampled pool (`rerank_top_n`) with a sentence-transformers model, then slice to `top_k` (cached, warmed at `initialize()`).
4. Return the top `top_k` rows.

Two details: `vector` projects its score column, but `hybrid` fetches unprojected and drops the `vector` column in Python (naming `_relevance_score` in `select()` errors — it pushes down to the FTS leg). And if hybrid fails (e.g. the full-text leg's index isn't ready), recall logs a warning and falls back to pure vector.

---

## Configuration reference

**You don't have to configure anything** — once the provider is activated (`hermes memory setup`, which sets `memory.provider: lancedb`), the plugin runs on its shipped defaults from [`default_config.yaml`](src/default_config.yaml). `~/.hermes/config.yaml` is purely for *overrides*: keys you set there win, keys you omit fall back to the defaults. To customize, **copy the blocks from `default_config.yaml` into your `~/.hermes/config.yaml`** and edit only what you want to change.

By default embeddings call the OpenAI API (`OPENAI_API_KEY` required); everything else is local. Don't edit `default_config.yaml` to change your own setup — a plugin update overwrites it; edit `~/.hermes/config.yaml`.

To use a different embeddings backend, point the OpenAI-compatible client at any endpoint that speaks the same shape (no code change needed). For example, a hosted non-OpenAI model via OpenRouter:

```yaml
# ~/.hermes/config.yaml
plugins:
  lancedb:
    embedding:
      model: google/gemini-embedding-001
      base_url: https://openrouter.ai/api/v1
      api_key_env: OPENROUTER_API_KEY
```

…or fully local embeddings via Ollama:

```yaml
# ~/.hermes/config.yaml
plugins:
  lancedb:
    embedding:
      model: nomic-embed-text
      base_url: http://localhost:11434/v1
      api_key_env: OLLAMA_API_KEY      # any value works for local Ollama
```

Changing the embedding model (or its dimension) against an existing store requires recreating the table — the plugin fails loudly on a dim mismatch rather than silently returning nothing.

### Knob-by-knob

| Section | Key | Default | Notes |
|---|---|---|---|
| `retrieval` | `mode` | `vector` | `vector` (default) or `hybrid`. Per-call override via the `mode` parameter on `lancedb_recall`. (A lexical-only `fts` value also works but is a niche, unrecommended escape hatch.) |
| | `top_k` | `10` | Hard cap inside the retrieval layer is 50. |
| | `search_kinds` | `[fact]` | Recall surfaces facts; turn rows are stored as provenance and used as fallback when no facts match. |
| `retrieval.reranker` | `type` | `rrf` | Hybrid fusion: `rrf` \| `linear` \| `cross-encoder`. No-op for `mode: vector` / `mode: fts` (one ranked list). |
| | `weight` | `0.7` | `linear` only: vector weight (0–1) for the weighted vector/FTS combination; higher leans on vector. |
| | `model` | `cross-encoder/ettin-reranker-17m-v1` | `cross-encoder` only. Any HuggingFace cross-encoder ID; lazy-loaded on first use. |
| | `rerank_top_n` | `50` | `cross-encoder` only. Enforced as `max(rerank_top_n, top_k)` so you never fetch fewer than you return. |
| `extraction` | `enabled` | `true` | Set `false` to skip the auxiliary LLM call. |
| | `min_turns` | `3` | Skip extraction when the user has spoken fewer than N turns. |
| `embedding` | `provider` | `openai` | Label; currently always selects the OpenAI-compatible client. The actual endpoint is controlled by `base_url` / `api_key_env` below. |
| | `model` | `text-embedding-3-small` | 1536-dim for the default. Embedding dim must match the existing table: recreate the table if you change models (or dim) against an existing store — the plugin now fails loudly on a mismatch instead of silently returning nothing. |
| | `base_url` | `null` | `null` = OpenAI's default endpoint. Set to any OpenAI-compatible embeddings endpoint — OpenRouter (`https://openrouter.ai/api/v1`), Nous, Together, vLLM, Ollama / LM Studio in OpenAI-compatible mode (e.g. `http://localhost:11434/v1`), or a self-hosted server. |
| | `api_key_env` | `OPENAI_API_KEY` | Name of the environment variable holding the API key. Point it at a different var to keep your embedding key separate from `OPENAI_API_KEY`. |
| | `dimensions` | `null` | Optional output dimensions for matryoshka models (`text-embedding-3-*`). `null` = the model's native dimension. |
| | `max_batch` | `100` | Max inputs per embeddings request. Providers cap this differently (Gemini 100, Cohere 96, OpenAI up to 2048); the default is the safe common denominator. Lower it for a stricter provider, raise it to cut request count on OpenAI. |
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
| `~/.cache/huggingface/` | Cross-encoder reranker model cache (managed by HuggingFace). Only present if `reranker.type: cross-encoder` is enabled — embeddings use the OpenAI API and cache nothing locally. |

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

**Recall fails with an auth error.** Embeddings call the OpenAI API — make sure `OPENAI_API_KEY` is set in the environment (or `~/.hermes/.env`). With `reranker.type: cross-encoder`, the sentence-transformers reranker model is downloaded to `~/.cache/huggingface/` on first use and preloaded during `initialize()` so the first user query doesn't pay the model-load cost.

**Table fragments / `.lance` directory growing.** Check `maintenance.enabled: true` and that `~/.hermes/lancedb/.last_optimize_version` is advancing across sessions. `agent.log` will show `lancedb optimize starting` when a compaction fires.

**Changed `embedding.model` and recall returns nothing.** The new model's dim doesn't match the existing column. Delete `~/.hermes/lancedb/memories.lance/` to recreate the table on the next session.

---

## License

Apache 2.0
