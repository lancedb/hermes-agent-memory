# LanceDB memory plugin — next steps

Install runtime dependencies into Hermes's Python environment. For a source
checkout at `~/code/hermes-agent`, use:

```
uv pip install --python ~/code/hermes-agent/venv/bin/python3 lancedb openai pyyaml
```

If you installed Hermes with the one-line installer, the venv is usually under
`~/.hermes`:

```
uv pip install --python ~/.hermes/hermes-agent/venv/bin/python3 lancedb openai pyyaml
```

Then run the setup wizard:

```
hermes memory setup
# pick "lancedb" from the list
```

Setup writes `memory.provider: lancedb`. Embeddings use OpenAI
`text-embedding-3-small` (1536-dim) via the API — there's no local model to
download, but you must set `OPENAI_API_KEY` (in your environment or
`~/.hermes/.env`).

## Optional

All settings live in `~/.hermes/config.yaml`. The plugin ships a
`default_config.yaml` with every option documented — copy its contents into
`~/.hermes/config.yaml` and edit. A few common tweaks:

Use a cheaper LLM for fact extraction:

```yaml
# ~/.hermes/config.yaml
auxiliary:
  lancedb_extraction:
    provider: openrouter
    model: google/gemini-3-flash
```

Use a different OpenAI embedding model:

```yaml
# ~/.hermes/config.yaml
plugins:
  lancedb:
    embedding:
      model: text-embedding-3-large   # changing dim requires recreating the table
```

Or point at any OpenAI-compatible endpoint — e.g. fully local embeddings via
Ollama (no code change needed):

```yaml
# ~/.hermes/config.yaml
plugins:
  lancedb:
    embedding:
      model: nomic-embed-text
      base_url: http://localhost:11434/v1
      api_key_env: OLLAMA_API_KEY     # any value works for local Ollama
```

Enable the cross-encoder reranker (replaces RRF in hybrid mode; this is the one
feature that needs `sentence-transformers` installed, which pulls in `torch`
(~2 GB) — `uv pip install --python … sentence-transformers`. Nothing else in the
plugin needs it):

```yaml
plugins:
  lancedb:
    retrieval:
      reranker:
        type: cross-encoder
        model: cross-encoder/ettin-reranker-17m-v1
        rerank_top_n: 50
```
