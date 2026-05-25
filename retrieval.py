"""Recall queries over the LanceDB memory table."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from .store import RESULT_COLUMNS, build_filter

logger = logging.getLogger(__name__)


def _limit(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, 50))


def _apply_reranker(builder, *, enabled: bool, model: str, instance: Any = None):
    if not enabled:
        return builder
    if instance is not None:
        try:
            return builder.rerank(instance)
        except Exception as exc:
            logger.warning("lancedb reranker.rerank() failed (%s); falling back unranked", exc)
            return builder
    try:
        from lancedb.rerankers import CrossEncoderReranker

        return builder.rerank(CrossEncoderReranker(model_name=model, column="content"))
    except Exception as exc:
        logger.warning("lancedb cross-encoder reranker unavailable: %s", exc)
        return builder


def recall(
    store,
    query: str,
    *,
    mode: str = "hybrid",
    kind: str = "fact",
    category: str = "",
    workspace: str = "",
    user_id: str = "",
    limit: int = 10,
    reranker_type: str = "rrf",
    reranker_model: str = "cross-encoder/ettin-reranker-32m-v1",
    reranker: Any = None,
    rerank_top_n: int = 50,
) -> List[Dict[str, Any]]:
    """Return ranked memory rows."""
    query = (query or "").strip()
    if not query:
        return []
    mode = mode if mode in {"hybrid", "vector", "fts"} else "hybrid"
    kind = kind if kind in {"fact", "turn", "any"} else "fact"
    limit = _limit(limit, 10)

    where = build_filter(workspace=workspace, user_id=user_id, kind=kind, category=category)
    table = store.table

    if mode == "vector":
        vector = store.embedder.embed_one(query)
        builder = table.search(vector, query_type="vector", vector_column_name="vector")
        score_col = "_distance"
    elif mode == "fts":
        builder = table.search(query, query_type="fts", fts_columns="content")
        score_col = "_score"
    else:
        vector = store.embedder.embed_one(query)
        builder = (
            table.search(query_type="hybrid", vector_column_name="vector", fts_columns="content")
            .vector(vector)
            .text(query)
        )
        score_col = "_relevance_score"

    if where:
        builder = builder.where(where, prefilter=True)

    cross_encoder_active = reranker_type == "cross-encoder"
    builder = _apply_reranker(
        builder,
        enabled=cross_encoder_active,
        model=reranker_model,
        instance=reranker,
    )
    # Explicitly project the per-mode score column. Lance still auto-includes
    # it today but emits a Rust-level deprecation warning when select() omits
    # it; future versions will drop the column entirely.
    projection = RESULT_COLUMNS + [score_col]
    # Cross-encoder: pull rerank_top_n candidates (must be >= top_k) and let
    # the reranker reorder them; slice back to limit after .to_list().
    fetch_limit = max(rerank_top_n, limit) if cross_encoder_active else limit
    try:
        rows = builder.select(projection).limit(fetch_limit).to_list()
        return rows[:limit]
    except Exception as exc:
        logger.debug("lancedb recall failed (%s): %s", mode, exc)
        if mode == "hybrid":
            return recall(
                store,
                query,
                mode="vector",
                kind=kind,
                category=category,
                workspace=workspace,
                user_id=user_id,
                limit=limit,
                reranker_type="rrf",
                reranker_model=reranker_model,
            )
        return []


def format_prefetch(rows: list[dict[str, Any]], *, max_items: int = 5) -> str:
    if not rows:
        return ""
    lines = ["## LanceDB Memory"]
    for row in rows[:max_items]:
        content = row.get("abstract") or row.get("content") or ""
        content = " ".join(str(content).split())
        if len(content) > 500:
            content = content[:497] + "..."
        category = row.get("category") or row.get("kind") or "memory"
        lines.append(f"- ({category}, id={row.get('id')}) {content}")
    return "\n".join(lines)
