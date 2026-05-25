# hermes-agent-memory

LanceDB-backed memory provider plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Embedded hybrid recall (vector + BM25 + RRF), workspace-contained durable facts,
transparent forget flows, mid-session fact extraction, and optional
Sentence Transformers cross-encoder reranking.

---

## What is Hermes Agent (one-paragraph version)

Hermes is an open-source agent framework from Nous Research. You chat with it in a
terminal TUI, and an LLM (OpenAI, Anthropic, OpenRouter, local — your choice) responds
between tool calls — file ops, web fetch, shell, vision, **memory**, etc. Sessions
persist locally in `~/.hermes/`, can be resumed, and can also run via messaging
bridges (Telegram, Discord) or on a cron schedule. Plugins extend specific surfaces;
this one plugs into the memory surface, replacing the default in-prompt memory
with an embedded LanceDB store.

## What this plugin gives you

- **Hybrid retrieval** — vector + BM25 fused via LanceDB's built-in RRF.
  Switchable per-config or per-tool-call to pure vector or pure FTS mode for
  workloads where one beats the other.
- **Fact-first memory** — raw turns are stored as provenance, but recall prefers
  durable facts with short abstracts so noisy chat history does not crowd out
  user preferences and project decisions.
- **Mid-session extraction** — facts get extracted on `on_pre_compress`
  (not just session end), so insights are recallable before context compression
  discards them.
- **Transparent memory management** — provider tools let the agent recall,
  remember, read provenance, and preview/delete memories without exposing raw
  database operations to users.
- **Embedded, no server** — the table lives at `~/.hermes/lancedb/memories.lance`.
  Open it in any LanceDB client to debug retrieval directly.

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — fast Python package manager
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) cloned somewhere
- An LLM API key (OpenAI, OpenRouter, Anthropic, etc.)

---

## Quickstart — full path from zero to chatting

If you've never run Hermes before, do every step in order. If Hermes is already
working on your machine, skip to **Step 3**.

### Step 1 — Install Hermes Agent

One line on macOS / Linux / WSL2:

```sh
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

(Windows users: `iex (irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1)`)

The installer handles `uv`, Python 3.11, Node.js, ripgrep, ffmpeg, and on
Windows MinGit. It git-clones Hermes for you into `~/.hermes/hermes-agent/`,
creates the venv at `~/.hermes/hermes-agent/venv/`, and symlinks the
binary to `~/.local/bin/hermes`. You don't have to manage any of that
directly.

If the installer reports gaps afterwards, the two commands that fix almost
everything are:

```sh
hermes doctor --fix     # creates missing symlinks, dirs, etc.
hermes setup            # interactive: .env, API key, model picker
```

### Step 2 — Configure your LLM

`hermes setup` (from Step 1) walks you through this interactively. It will:

1. Create `~/.hermes/.env` and prompt for your API key (OpenAI, Anthropic,
   OpenRouter, etc.). Pasted keys stay in `.env`, never in shell history.
2. Pick a model — for OpenAI, `gpt-4o-mini` is cheap+fast, `gpt-4o` for
   stronger reasoning.
3. Write the choice into `~/.hermes/config.yaml`.

Verify everything is wired:

```sh
hermes doctor
```

`hermes doctor` is the canary — it validates env vars, config, and deps,
and prints red lines for anything missing. Fix any issues before continuing.

> **Note on AWS Bedrock complaints.** If you have AWS credentials in your
> environment (`AWS_ACCESS_KEY_ID` etc.) from other work, Hermes will try
> to auto-detect Bedrock as a possible provider and may report
> `AccessDeniedException`. This is ignorable if you're using OpenAI — it's
> just Hermes failing gracefully through its provider auto-detection.

First chat to confirm the LLM responds:

```sh
hermes chat -q "Hello, what tools do you have access to?"
```

### Step 3 — Install this plugin

#### Option A — once it's published (the user-facing flow)

```sh
hermes plugins install lancedb/hermes-agent-memory
```

This git-clones into `~/.hermes/plugins/lancedb/` and renders `after-install.md`
in a Rich panel telling you what's next.

#### Option B — local dev (clone and symlink)

```sh
git clone https://github.com/lancedb/hermes-agent-memory ~/code/hermes-memory-lancedb
ln -sf ~/code/hermes-memory-lancedb ~/.hermes/plugins/lancedb
```

The symlink lets you edit the plugin in place and have Hermes pick up changes
on the next session.

### Step 4 — Install the memory runtime deps

Hermes runs plugins inside Hermes's own Python environment. Install LanceDB and
Sentence Transformers there first. For a source checkout at `~/code/hermes-agent`,
use:

```sh
uv pip install --python ~/code/hermes-agent/venv/bin/python3 lancedb sentence-transformers
```

If you installed Hermes with the one-line installer, the venv is usually under
`~/.hermes`:

```sh
uv pip install --python ~/.hermes/hermes-agent/venv/bin/python3 lancedb sentence-transformers
```

This manual step avoids Hermes setup's short dependency-install timeout while the
heavier `sentence-transformers` stack resolves and installs. This plugin does
not ask `hermes memory setup` to install those packages automatically.

### Step 5 — Run memory setup

Now activate the provider:

```sh
hermes memory setup
# choose "lancedb"
```

Setup writes `memory.provider: lancedb`, writes `plugins.lancedb` defaults, and
warms the default embedding model once so download failures happen during setup,
not during the first chat.

Confirm Hermes sees the active provider:

```sh
hermes plugins list
# Should list "lancedb" in the output.

hermes memory status
```

### Step 6 — Start chatting

Open a chat:

```sh
hermes chat -q "Remember that I prefer pytest over unittest"
```

Open a new session and check recall:

```sh
hermes chat -q "What testing framework do I prefer?"
```

---

## Configuration

Defaults are local and keyless. To tweak, add a `plugins.lancedb` block to
`~/.hermes/config.yaml`. The most commonly tuned knobs:

```yaml
plugins:
  lancedb:
    retrieval:
      mode: hybrid              # hybrid | vector | fts
      rerank: none              # none | cross-encoder
      top_k: 10
      search_kinds: [fact]      # fact by default; turn rows are provenance/fallback
    extraction:
      enabled: true             # set false to skip LLM extraction at session boundaries
      min_turns: 3              # skip extraction for very short sessions
    embedding:
      provider: sentence-transformers
      model: BAAI/bge-small-en-v1.5
    reranker:
      model: cross-encoder/ettin-reranker-32m-v1
```

To use a cheaper LLM for fact extraction (not your main chat model):

```yaml
auxiliary:
  lancedb_extraction:
    provider: openrouter
    model: google/gemini-3-flash
```

Hermes's auxiliary client handles provider routing, fallback, and credit
exhaustion automatically.

See [`after-install.md`](after-install.md) for the setup flow.

---

## Verifying the plugin works

| Command | What you should see |
|---|---|
| `hermes plugins list` | `lancedb` in the output |
| `hermes doctor` | No red lines about memory |
| `hermes chat -q "test"` | Session opens, no crashes, agent.log shows `lancedb provider initialized` |
| `tail -f ~/.hermes/logs/agent.log` | Live view of provider activity |

Quick way to peek at what got stored (after a few chats):

```sh
uv run --project ~/.hermes/hermes-agent python -c "
import lancedb
db = lancedb.connect('~/.hermes/lancedb')
print(db.open_table('memories').to_pandas()[['kind', 'category', 'content']])
"
```

---

## Development setup (for contributing to this plugin)

This plugin has its own `pyproject.toml` with `[tool.uv] package = false` —
the dev venv is just for tests, lint, and isolated imports. The plugin code
itself is loaded by Hermes from the symlink, not pip-installed.

```sh
cd /path/to/hermes-memory-lancedb
uv sync --extra dev
```

Run the test suite, linter, and ad-hoc imports against the dev venv:

```sh
uv run pytest -v
uv run ruff check .
uv run python -c "import config; print(config.load_config())"
```

Runtime dependencies must be installed into Hermes's Python environment, not
only this repo's dev venv:

```sh
uv pip install --python /path/to/hermes/python lancedb sentence-transformers
```

For local development, add dev-only dependencies here:

```sh
uv add --dev pytest-mock
```

---

## Status

v0.1 in active development. See
[`scratch/brainstorming/plan-v0.1.html`](scratch/brainstorming/plan-v0.1.html)
for the full build plan.

| Phase | Description | Status |
|---|---|---|
| 1 | Skeleton + registration | ✅ done |
| 2 | Setup wizard installs deps + activates provider | ✅ done |
| 3 | LanceDB write path for turns and explicit facts | ✅ done |
| 4 | Recall path — hybrid / vector / fts over workspace facts | ✅ done |
| 5 | `lancedb_remember`, `lancedb_read`, transparent `lancedb_forget` | ✅ done |
| 6 | LLM extraction with provenance turn IDs + short abstracts | ✅ done |
| 7 | Cross-encoder reranker (`cross-encoder/ettin-reranker-32m-v1`, opt-in) | ✅ wired, opt-in |
| 8 | Tests + working agent-loop smoke test | ✅ local smoke |
| 9 | Publish + announce | pending |

---

## v0.1 design priorities

- Install plugin, run one setup command, start chatting.
- Keep memory workspace-contained by default.
- Recall durable facts first; use raw turns only as provenance or fallback.
- Make forget operations transparent: preview candidates, then delete by ID.
- Keep embeddings local through Sentence Transformers.

---

## License

Apache 2.0
