"""LanceDB memory plugin implementation package.

All plugin source lives here so the repo root stays legible (manifest + thin
entry point + this package). Hermes loads the plugin from the repo-root
``__init__.py``, which re-exports :class:`LanceDBMemoryProvider` from this
subpackage. Nothing under ``benchmarks/`` is imported from here.
"""
from __future__ import annotations

from .provider import LanceDBMemoryProvider

__all__ = ["LanceDBMemoryProvider"]
