"""Recall queries over the LanceDB memory table."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from .config import DEFAULTS
from .store import RESULT_COLUMNS, build_filter

logger = logging.getLogger(__name__)

# The default reranker model is defined once, in config.py DEFAULTS.
DEFAULT_RERANKER_MODEL = DEFAULTS["retrieval"]["reranker"]["model"]

# Score columns Lance may attach to a search result (per mode / fusion).
_SCORE_COLUMNS = ("_relevance_score", "_distance", "_score")


def _limit(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, 50))


def _apply_cross_encoder(builder, *, model: str, instance: Any = None):
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


def _apply_linear(builder, *, weight: float):
    """Weighted linear combination of the vector + FTS scores. `weight` is the
    vector weight (0-1); higher favors vector. Falls back to default RRF on
    error."""
    try:
        from lancedb.rerankers import LinearCombinationReranker

        return builder.rerank(LinearCombinationReranker(weight=weight))
    except Exception as exc:
        logger.warning("lancedb linear reranker unavailable (%s); using default RRF", exc)
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
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    reranker_weight: float = 0.7,
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

    # Hybrid fusion strategy. rrf -> leave the builder alone (LanceDB applies
    # its default RRF). linear -> weighted score combination biased by
    # reranker_weight. cross-encoder -> a rerank pass over an oversampled pool.
    cross_encoder_active = reranker_type == "cross-encoder"
    if cross_encoder_active:
        builder = _apply_cross_encoder(builder, model=reranker_model, instance=reranker)
    elif reranker_type == "linear" and mode == "hybrid":
        builder = _apply_linear(builder, weight=reranker_weight)
    # Cross-encoder: pull rerank_top_n candidates (must be >= top_k) and let
    # the reranker reorder them; slice back to limit after .to_list().
    fetch_limit = max(rerank_top_n, limit) if cross_encoder_active else limit
    try:
        if mode == "hybrid":
            # Hybrid can't name a score column in select(): _relevance_score is
            # produced only AFTER the vector + FTS legs fuse, and naming any
            # score (_relevance_score / _distance / _score) pushes the projection
            # down to the FTS leg and raises a schema error (which used to fall
            # back to vector silently). Selecting base columns alone instead
            # trips Lance's score auto-projection *deprecation warning*. So we
            # fetch unprojected (nothing "omitted" -> no warning) and trim in
            # Python to the same shape the other modes return — dropping the
            # heavy `vector` column while keeping RESULT_COLUMNS + the score.
            keep = set(RESULT_COLUMNS) | set(_SCORE_COLUMNS)
            rows = builder.limit(fetch_limit).to_list()
            rows = [{k: v for k, v in row.items() if k in keep} for row in rows]
        else:
            # vector/fts are single-leg: naming the score column is safe and
            # avoids the auto-projection deprecation warning.
            rows = builder.select(RESULT_COLUMNS + [score_col]).limit(fetch_limit).to_list()
        return rows[:limit]
    except Exception as exc:
        # Warn (not debug): a hybrid->vector fallback silently degrades the
        # default retrieval mode, so it must be visible.
        logger.warning("lancedb recall failed (mode=%s); falling back: %s", mode, exc)
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
