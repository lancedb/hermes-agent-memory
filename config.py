"""Plugin config — loaded from $HERMES_HOME/config.yaml under plugins.lancedb.

Returns the defaults merged with any user overrides. Falls back to defaults
gracefully when running outside Hermes (e.g. unit tests where the hermes_*
modules aren't on the import path).
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "embedding": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimension": 1536,
    },
    "retrieval": {
        "mode": "hybrid",            # "hybrid" | "vector" | "fts"
        "top_k": 10,
        "search_kinds": ["fact"],
        "reranker": {
            # "rrf"           — Reciprocal Rank Fusion. The fusion strategy for
            #                   hybrid mode (no-op for mode=vector/fts, which
            #                   return vector-distance / BM25 order natively).
            # "cross-encoder" — replace RRF / native ordering with a
            #                   LanceDB reranker.
            "type": "rrf",
            "model": "cross-encoder/ettin-reranker-32m-v1",
            # Cross-encoder only: pull this many candidates from the base
            # retriever, let the cross-encoder reorder them, then slice to
            # top_k. Larger = better recall, slower latency. 50 is a sensible
            # default for top_k in the 5-10 range.
            "rerank_top_n": 50,
        },
    },
    "extraction": {
        "enabled": True,
        "min_turns": 3,
    },
    "maintenance": {
        # Auto-compaction: Lance commits a new version on every add/delete, and
        # most agent writes are single-row. Without periodic optimize() the
        # dataset accumulates tiny fragments and old version files indefinitely.
        "enabled": True,
        "optimize_every_commits": 50,
        "cleanup_older_than_days": 7,
    },
}


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `overlay` over `base`. Returns a new dict."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> Dict[str, Any]:
    """Return defaults merged with user overrides from config.yaml."""
    try:
        from hermes_constants import get_hermes_home
        from hermes_cli.config import cfg_get
        import yaml
    except ImportError:
        return copy.deepcopy(DEFAULTS)

    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return copy.deepcopy(DEFAULTS)

    try:
        with open(config_path, encoding="utf-8-sig") as f:
            raw = yaml.safe_load(f) or {}
        user_cfg = cfg_get(raw, "plugins", "lancedb", default={}) or {}
        return _deep_merge(DEFAULTS, user_cfg)
    except Exception as exc:
        logger.warning("Failed to load lancedb plugin config (%s); using defaults", exc)
        return copy.deepcopy(DEFAULTS)


def save_plugin_config(values: Dict[str, Any], hermes_home: str) -> None:
    """Write LanceDB plugin config under plugins.lancedb in config.yaml."""
    config_path = Path(hermes_home) / "config.yaml"
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to save lancedb config") from exc

    existing: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8-sig") as f:
            existing = yaml.safe_load(f) or {}
    if not isinstance(existing, dict):
        existing = {}

    merged_values = _deep_merge(DEFAULTS, values or {})
    existing.setdefault("plugins", {})
    existing["plugins"]["lancedb"] = merged_values

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(existing, f, default_flow_style=False, sort_keys=False)
