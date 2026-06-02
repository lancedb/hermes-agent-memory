"""Tool schemas and dispatch for LanceDB memory."""
from __future__ import annotations

import json
from typing import Any, Dict

try:
    from tools.registry import tool_error
except Exception:  # pragma: no cover - outside Hermes
    def tool_error(message: str) -> str:
        return json.dumps({"error": message})


LANCEDB_RECALL = {
    "name": "lancedb_recall",
    "description": (
        "Recall durable workspace memory from LanceDB. Uses hybrid vector+FTS by default. "
        "Returns memory IDs, snippets, scores, and provenance turn IDs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search memory for."},
            "mode": {"type": "string", "enum": ["hybrid", "vector", "fts"]},
            "kind": {"type": "string", "enum": ["fact", "turn", "any"]},
            "category": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    },
}

LANCEDB_REMEMBER = {
    "name": "lancedb_remember",
    "description": "Store a durable fact the user would expect Hermes to remember.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "abstract": {"type": "string"},
            "category": {
                "type": "string",
                "enum": ["preference", "entity", "event", "case", "pattern", "general"],
            },
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["content"],
    },
}

LANCEDB_READ = {
    "name": "lancedb_read",
    "description": "Read one memory or its provenance turns by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "include_provenance": {"type": "boolean"},
        },
        "required": ["id"],
    },
}

LANCEDB_FORGET = {
    "name": "lancedb_forget",
    "description": (
        "Preview or delete a memory. Preview candidates first when the user asks "
        "to forget something by description. Delete requires an exact ID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["preview", "delete"]},
            "query": {"type": "string", "description": "Required for preview."},
            "id": {"type": "string", "description": "Required for delete."},
            "limit": {"type": "integer"},
        },
        "required": ["action"],
    },
}

TOOL_SCHEMAS = [LANCEDB_RECALL, LANCEDB_REMEMBER, LANCEDB_READ, LANCEDB_FORGET]


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _clean_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    seen = set()
    for item in value:
        tag = str(item).strip()
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


class LanceDBToolDispatcher:
    def __init__(self, provider) -> None:
        self.provider = provider

    def handle(self, tool_name: str, args: Dict[str, Any]) -> str:
        if tool_name == "lancedb_recall":
            return self._recall(args)
        if tool_name == "lancedb_remember":
            return self._remember(args)
        if tool_name == "lancedb_read":
            return self._read(args)
        if tool_name == "lancedb_forget":
            return self._forget(args)
        return tool_error(f"Unknown LanceDB memory tool: {tool_name}")

    def _recall(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        rows = self.provider.recall(
            query,
            mode=args.get("mode") or "",
            kind=args.get("kind") or "fact",
            category=args.get("category") or "",
            limit=args.get("limit") or self.provider.config["retrieval"]["top_k"],
        )
        return _json({"results": [_format_result(row) for row in rows], "total": len(rows)})

    def _remember(self, args: Dict[str, Any]) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        category = str(args.get("category") or "general").strip() or "general"
        row = self.provider.build_fact_row(
            content=content,
            abstract=str(args.get("abstract") or "").strip(),
            category=category,
            tags=_clean_tags(args.get("tags")),
            provenance_turn_ids=[],
            source="remember",
        )
        existing = self.provider.store.find_by_hash(
            row["content_hash"],
            workspace=row.get("agent_workspace", ""),
            kind="fact",
        )
        if existing:
            return _json({"status": "exists", "id": existing.get("id"), "content": existing.get("content")})
        self.provider.store.add_row(row)
        return _json({"status": "stored", "id": row["id"], "content": content})

    def _read(self, args: Dict[str, Any]) -> str:
        memory_id = str(args.get("id") or "").strip()
        if not memory_id:
            return tool_error("id is required")
        row = self.provider.store.get_by_id(memory_id)
        if not row:
            return tool_error(f"memory not found: {memory_id}")
        payload = {"memory": _format_result(row, include_full=True)}
        if args.get("include_provenance"):
            payload["provenance"] = [
                _format_result(item, include_full=True)
                for item in self.provider.store.get_by_ids(row.get("provenance_turn_ids") or [])
            ]
        return _json(payload)

    def _forget(self, args: Dict[str, Any]) -> str:
        action = str(args.get("action") or "").strip()
        if action == "preview":
            query = str(args.get("query") or "").strip()
            if not query:
                return tool_error("query is required for preview")
            rows = self.provider.recall(
                query,
                mode="hybrid",
                kind="fact",
                limit=args.get("limit") or 5,
            )
            return _json({
                "action": "preview",
                "candidates": [_format_result(row) for row in rows],
                "instruction": "Ask the user to confirm the exact ID before delete if there is any ambiguity.",
            })
        if action == "delete":
            memory_id = str(args.get("id") or "").strip()
            if not memory_id:
                return tool_error("id is required for delete")
            row = self.provider.store.get_by_id(memory_id)
            if not row:
                return tool_error(f"memory not found: {memory_id}")
            self.provider.store.delete_by_id(memory_id)
            return _json({"action": "delete", "deleted": _format_result(row, include_full=True)})
        return tool_error("action must be preview or delete")


def _format_result(row: Dict[str, Any], *, include_full: bool = False) -> Dict[str, Any]:
    content = row.get("content") or ""
    snippet = " ".join(str(row.get("abstract") or content).split())
    if len(snippet) > 700 and not include_full:
        snippet = snippet[:697] + "..."
    payload = {
        "id": row.get("id"),
        "kind": row.get("kind"),
        "category": row.get("category") or "",
        "snippet": snippet,
        "tags": row.get("tags") or [],
        "provenance_turn_ids": row.get("provenance_turn_ids") or [],
        "created_at": row.get("created_at"),
    }
    for score_key in ("_relevance_score", "_distance", "_score"):
        if score_key in row:
            payload[score_key] = row[score_key]
    if include_full:
        payload["content"] = content
        payload["session_id"] = row.get("session_id") or ""
        payload["role"] = row.get("role") or ""
        payload["source"] = row.get("source") or ""
    return payload
