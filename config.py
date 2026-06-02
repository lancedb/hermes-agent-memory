"""Plugin config.

The plugin's default settings live in ``default_config.yaml`` (the single
source of truth). This module loads them and merges any user overrides from
``$HERMES_HOME/config.yaml`` under ``plugins.lancedb``.

Users configure the plugin by editing ``~/.hermes/config.yaml`` — copy-paste
``default_config.yaml`` to get started. Do not edit this module or
``default_config.yaml`` to change your own setup; a plugin update overwrites
both.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


def _load_defaults() -> Dict[str, Any]:
    """Load the plugin's defaults from default_config.yaml (single source).

    Returns the ``plugins.lancedb`` block. Raised errors are intentional — a
    missing or malformed defaults file is a packaging bug, not a runtime
    fallback condition.
    """
    raw = yaml.safe_load(_DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return (raw.get("plugins", {}) or {}).get("lancedb", {}) or {}


DEFAULTS: Dict[str, Any] = _load_defaults()


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
    """Return defaults merged with user overrides from ~/.hermes/config.yaml.

    Falls back to the defaults when running outside Hermes (e.g. unit tests
    where the hermes_* modules aren't importable) or when no config.yaml exists.
    """
    try:
        from hermes_constants import get_hermes_home
        from hermes_cli.config import cfg_get
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
