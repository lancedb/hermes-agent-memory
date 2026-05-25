# LanceDB memory plugin — next steps

Install runtime dependencies into Hermes's Python environment. For a source
checkout at `~/code/hermes-agent`, use:

```
uv pip install --python ~/code/hermes-agent/venv/bin/python3 lancedb sentence-transformers
```

If you installed Hermes with the one-line installer, the venv is usually under
`~/.hermes`:

```
uv pip install --python ~/.hermes/hermes-agent/venv/bin/python3 lancedb sentence-transformers
```

Then run the setup wizard:

```
hermes memory setup
# pick "lancedb" from the list
```

Setup writes `memory.provider: lancedb` and warms the default embedding model
once. The model `BAAI/bge-small-en-v1.5` (~133 MB) is cached in
`~/.cache/huggingface`.

## Optional

Use a cheaper LLM for fact extraction:

```yaml
# ~/.hermes/config.yaml
auxiliary:
  lancedb_extraction:
    provider: openrouter
    model: google/gemini-3-flash
```

Use a different local Sentence Transformers embedding model:

```yaml
# ~/.hermes/config.yaml
plugins:
  lancedb:
    embedding:
      provider: sentence-transformers
      model: BAAI/bge-small-en-v1.5
```

Enable the cross-encoder reranker (replaces RRF in hybrid mode):

```yaml
plugins:
  lancedb:
    retrieval:
      reranker:
        type: cross-encoder
        model: cross-encoder/ettin-reranker-32m-v1
        rerank_top_n: 50
```
