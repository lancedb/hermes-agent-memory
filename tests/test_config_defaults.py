from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


def _load_config_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("lancedb_config_under_test", root / "config.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
cfg = _load_config_module()


def test_defaults_are_loaded_from_default_config_yaml():
    # DEFAULTS must be sourced from default_config.yaml (single source of truth),
    # not a hardcoded dict — so the two can never drift.
    raw = yaml.safe_load((ROOT / "default_config.yaml").read_text(encoding="utf-8"))
    expected = raw["plugins"]["lancedb"]
    assert cfg.DEFAULTS == expected


def test_defaults_have_expected_shape():
    d = cfg.DEFAULTS
    assert d["embedding"]["provider"] == "openai"
    assert d["embedding"]["model"] == "text-embedding-3-small"
    assert d["retrieval"]["reranker"]["type"] == "rrf"
    assert "weight" in d["retrieval"]["reranker"]
    assert d["retrieval"]["reranker"]["model"].startswith("cross-encoder/")
