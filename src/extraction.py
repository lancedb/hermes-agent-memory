"""LLM fact extraction for LanceDB memory."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """Extract durable workspace memories from this conversation.
Return JSON with a facts array. Each fact has:
  content    - one concise durable fact
  abstract   - optional one-sentence summary
  category   - preference, entity, event, case, pattern, or general
  tags       - short search tags
  evidence   - message indexes that support the fact

Skip trivial, temporary, repeated, or purely session-local details."""

VALID_CATEGORIES = {"preference", "entity", "event", "case", "pattern", "general"}


def format_messages_with_indexes(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, message in enumerate(messages):
        role = message.get("role", "")
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"[{idx}] {role}: {content}")
    return "\n\n".join(lines)


def extract(messages: List[Dict[str, Any]], context: Dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Extract durable facts using Hermes's auxiliary LLM client."""
    if not messages:
        return []
    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:
        logger.debug("auxiliary client unavailable for lancedb extraction: %s", exc)
        return []

    try:
        response = call_llm(
            task="lancedb_extraction",
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": format_messages_with_indexes(messages)},
            ],
            response_format={"type": "json_object"},
            timeout=30,
        )
    except Exception as exc:
        logger.debug("lancedb extraction call failed: %s", exc)
        return []

    text = getattr(response, "content", response)
    if not isinstance(text, str):
        text = str(text)
    try:
        payload = json.loads(text)
    except Exception as exc:
        logger.debug("lancedb extraction returned non-json: %s", exc)
        return []

    facts = payload.get("facts", [])
    if not isinstance(facts, list):
        return []
    cleaned = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        content = str(fact.get("content") or "").strip()
        if not content:
            continue
        category = str(fact.get("category") or "general").strip()
        if category not in VALID_CATEGORIES:
            category = "general"
        evidence = fact.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = []
        tags = fact.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        cleaned.append(
            {
                "content": content,
                "abstract": str(fact.get("abstract") or "").strip(),
                "category": category,
                "tags": [str(tag).strip() for tag in tags if str(tag).strip()],
                "evidence": [int(i) for i in evidence if isinstance(i, int) or str(i).isdigit()],
            }
        )
    return cleaned
