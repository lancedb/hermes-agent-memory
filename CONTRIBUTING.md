# Contributing

Thanks for your interest in improving the LanceDB memory plugin for Hermes Agent! Issues and PRs are welcome.

## Development setup

This repo isn't pip-installed — Hermes loads it from its directory. For local development:

```sh
git clone https://github.com/lancedb/hermes-agent-memory
cd hermes-agent-memory
uv sync --extra dev
```

The test suite imports a few Hermes modules (e.g. `agent.memory_provider`, `tools.registry`), so it expects a **sibling Hermes checkout** at `../hermes-agent`:

```sh
git clone https://github.com/NousResearch/hermes-agent ../hermes-agent
```

To wire the plugin into a live Hermes for end-to-end testing, see the **Installation: developers** section of the [README](README.md) (symlink approach).

## Before you open a PR

Run lint and tests locally — they must pass, and CI runs the same checks:

```sh
uv run ruff check .
uv run pytest
```

Guidelines:

- Match the existing style: ruff, 100-char lines, type hints, `from __future__ import annotations`.
- Add or update tests for any behavior change. **Tests must run offline** — mock the embeddings/LLM client instead of calling a real API (see `tests/test_embeddings.py` for the pattern).
- Keep the plugin and the benchmark separate: code under `src/` must never import from `benchmarks/`.

## Versioning & releases

We follow [SemVer](https://semver.org). On a release:

1. Bump the version in **both** `pyproject.toml` and `plugin.yaml` (CI fails if they drift).
2. Add a `CHANGELOG.md` entry.
3. Tag the commit (`vX.Y.Z`) and cut a GitHub Release.

Because Hermes installs the plugin by cloning `main`, keep `main` releasable at all times.

## Reporting issues

Use the issue templates. For bugs, include your Hermes version, plugin version, OS/Python, the retrieval mode in use, and relevant `agent.log` lines plus your `hermes memory status` output.
