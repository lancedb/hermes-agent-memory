# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

- Initial release: a LanceDB-backed Hermes memory provider exposing four tools —
  `lancedb_remember`, `lancedb_recall`, `lancedb_read`, and `lancedb_forget`.
- Vector recall by default, with optional hybrid (vector + BM25) retrieval and
  RRF / vector-biased linear / cross-encoder fusion.
- Mid-session fact extraction on pre-compress and session end, with provenance
  back to the source messages.
- Workspace-scoped storage, content-hash dedupe, and background auto-compaction.
- OpenAI-compatible embeddings client (point it at any compatible endpoint).
- LongMemEval benchmark harness under `benchmarks/`.
