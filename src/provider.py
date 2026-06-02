"""LanceDB-backed Hermes MemoryProvider implementation."""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

from .config import DEFAULTS, load_config, save_plugin_config
from .embeddings import OpenAICompatibleEmbedder, embedder_from_config
from .extraction import extract
from .retrieval import format_prefetch, recall as recall_memories
from .store import (
    LanceDBStore,
    content_hash,
    make_id,
    stable_turn_id,
    utc_now,
)
from .tools import TOOL_SCHEMAS, LanceDBToolDispatcher

logger = logging.getLogger(__name__)


class LanceDBMemoryProvider(MemoryProvider):
    """Hybrid vector + FTS memory backed by an embedded LanceDB table."""

    def __init__(self) -> None:
        self._config: Dict[str, Any] = load_config()
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._platform: str = ""
        self._agent_context: str = "primary"
        self._agent_identity: str = ""
        self._agent_workspace: str = ""
        self._user_id: str = ""
        self._message_index: int = 0
        self._initialized: bool = False
        self._embedder: OpenAICompatibleEmbedder | None = None
        self._reranker: Any = None
        self._store: LanceDBStore | None = None
        self._tool_dispatcher = LanceDBToolDispatcher(self)
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None

    @property
    def name(self) -> str:
        return "lancedb"

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def store(self) -> LanceDBStore:
        if self._store is None:
            maint = self._config.get("maintenance", {}) or {}
            self._store = LanceDBStore(
                self._resolve_hermes_home(),
                self._get_embedder(),
                optimize_every_commits=int(maint.get("optimize_every_commits", 50)),
                cleanup_older_than_days=int(maint.get("cleanup_older_than_days", 7)),
                maintenance_enabled=bool(maint.get("enabled", True)),
            )
        return self._store

    def is_available(self) -> bool:
        """Verify hard dependencies are importable.

        Embeddings use the OpenAI API; the cross-encoder reranker is optional
        and lazily imported only when enabled, so it is not required here. Per
        the ABC contract, this must not make network calls.
        """
        try:
            import lancedb  # noqa: F401
            import openai  # noqa: F401
        except ImportError as exc:
            logger.debug("lancedb provider not available: %s", exc)
            return False
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = load_config()
        self._session_id = session_id
        self._hermes_home = str(kwargs.get("hermes_home") or self._resolve_hermes_home())
        self._platform = str(kwargs.get("platform") or "")
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._agent_identity = str(kwargs.get("agent_identity") or "")
        self._agent_workspace = str(kwargs.get("agent_workspace") or "")
        self._user_id = str(kwargs.get("user_id") or "")
        self._message_index = 0
        self.store.start_worker()
        # Eagerly load the cross-encoder when reranking is enabled so the first
        # recall call doesn't pay a 1-2s model load.
        reranker_cfg = self._config.get("retrieval", {}).get("reranker", {}) or {}
        if reranker_cfg.get("type") == "cross-encoder":
            self._get_reranker()
        self._initialized = True
        logger.info(
            "lancedb provider initialized (session=%s, platform=%s, agent_identity=%s)",
            session_id,
            self._platform or "?",
            self._agent_identity or "?",
        )

    def system_prompt_block(self) -> str:
        return (
            "# LanceDB Memory\n"
            "Active. Recall durable workspace memory with lancedb_recall. "
            "Use lancedb_remember when the user explicitly asks you to remember "
            "something important. Use lancedb_read for full content/provenance. "
            "For forgetting, preview candidates with lancedb_forget before deleting "
            "one exact ID."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not query or not self._initialized:
            return

        def _run() -> None:
            try:
                rows = self.recall(
                    query,
                    mode=self._config["retrieval"].get("mode", "hybrid"),
                    kind="fact",
                    limit=min(int(self._config["retrieval"].get("top_k", 10)), 5),
                )
                if not rows:
                    rows = self.recall(query, mode="hybrid", kind="turn", limit=3)
                formatted = format_prefetch(rows)
                if formatted:
                    with self._prefetch_lock:
                        self._prefetch_result = formatted
            except Exception as exc:
                logger.debug("lancedb prefetch failed: %s", exc)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="lancedb-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._should_write():
            return
        sid = session_id or self._session_id
        user_row = self._build_turn_row("user", user_content, sid)
        assistant_row = self._build_turn_row("assistant", assistant_content, sid)
        self.store.enqueue(user_row)
        self.store.enqueue(assistant_row)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        return self._tool_dispatcher.handle(tool_name, args or {})

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        self._session_id = new_session_id
        if reset:
            self._message_index = 0
        if kwargs.get("agent_workspace") is not None:
            self._agent_workspace = str(kwargs.get("agent_workspace") or "")
        if kwargs.get("agent_identity") is not None:
            self._agent_identity = str(kwargs.get("agent_identity") or "")
        if kwargs.get("user_id") is not None:
            self._user_id = str(kwargs.get("user_id") or "")

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        if action != "add" or not content or not self._should_write():
            return
        category = "preference" if target == "user" else "general"
        row = self.build_fact_row(
            content=content,
            abstract="",
            category=category,
            tags=[],
            provenance_turn_ids=[],
            source="memory_write_mirror",
        )
        if self.store.find_by_hash(row["content_hash"], workspace=self._agent_workspace, kind="fact"):
            return
        self.store.enqueue(row)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        inserted = self._extract_and_store(messages, source="pre_compress")
        if not inserted:
            return ""
        lines = ["LanceDB extracted durable facts before compression:"]
        for row in inserted[:8]:
            lines.append(f"- {row['content']}")
        return "\n".join(lines)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self._extract_and_store(messages, source="session_end")

    def shutdown(self) -> None:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
        if self._store is not None:
            self._store.shutdown()
        self._initialized = False
        logger.info("lancedb provider shutdown")

    def get_config_schema(self) -> List[Dict[str, Any]]:
        reranker_cfg = self._config.get("retrieval", {}).get("reranker", {}) or {}
        return [
            {
                "key": "retrieval_mode",
                "description": "Default recall mode",
                "default": self._config["retrieval"]["mode"],
                "choices": ["hybrid", "vector", "fts"],
            },
            {
                "key": "reranker_type",
                "description": "Reranker (rrf = hybrid fusion default; no-op for vector/fts modes)",
                "default": reranker_cfg.get("type", "rrf"),
                "choices": ["rrf", "cross-encoder"],
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        cfg = dict(self._config)
        if values.get("retrieval_mode"):
            cfg.setdefault("retrieval", {})["mode"] = values["retrieval_mode"]
        if values.get("reranker_type"):
            cfg.setdefault("retrieval", {}).setdefault("reranker", {})["type"] = values["reranker_type"]
        save_plugin_config(cfg, hermes_home)

    def post_setup(self, hermes_home: str, config: Dict[str, Any]) -> None:
        import sys

        from hermes_cli.config import save_config

        if not self.is_available():
            print("\n  ⚠ LanceDB memory dependencies are not importable.")
            print(
                "  Run manually: "
                f"uv pip install --python {sys.executable} lancedb openai"
            )
            print("  Then re-run: hermes memory setup\n")
            return

        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        config["memory"]["provider"] = "lancedb"
        save_config(config)
        save_plugin_config(DEFAULTS, hermes_home)

        embedding_cfg = DEFAULTS.get("embedding", {})
        api_key_env = embedding_cfg.get("api_key_env") or "OPENAI_API_KEY"
        try:
            dim = embedder_from_config(embedding_cfg).warm()
            print(f"\n  ✓ LanceDB memory configured (embedding dim: {dim})")
        except Exception as exc:
            print("\n  ✓ LanceDB memory configured")
            print(f"  ⚠ Embedding model warmup failed: {exc}")
            print(f"  Re-run setup or start a chat once {api_key_env} is set.")
        print("  Start a new session to activate.\n")

    def recall(
        self,
        query: str,
        *,
        mode: str = "",
        kind: str = "fact",
        category: str = "",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        retrieval_cfg = self._config.get("retrieval", {})
        reranker_cfg = retrieval_cfg.get("reranker", {}) or {}
        reranker_type = reranker_cfg.get("type", "rrf")
        reranker = self._get_reranker() if reranker_type == "cross-encoder" else None
        return recall_memories(
            self.store,
            query,
            mode=mode or retrieval_cfg.get("mode", "hybrid"),
            kind=kind,
            category=category,
            workspace=self._agent_workspace,
            user_id=self._user_id,
            limit=limit or retrieval_cfg.get("top_k", 10),
            reranker_type=reranker_type,
            reranker_model=reranker_cfg.get("model", DEFAULTS["retrieval"]["reranker"]["model"]),
            reranker_weight=float(reranker_cfg.get("weight", DEFAULTS["retrieval"]["reranker"]["weight"])),
            reranker=reranker,
            rerank_top_n=int(reranker_cfg.get("rerank_top_n", 50)),
        )

    def build_fact_row(
        self,
        *,
        content: str,
        abstract: str = "",
        category: str = "general",
        tags: list[str] | None = None,
        provenance_turn_ids: list[str] | None = None,
        source: str = "remember",
    ) -> Dict[str, Any]:
        return {
            "id": make_id("fact"),
            "kind": "fact",
            "content": content,
            "abstract": abstract,
            "category": category or "general",
            "tags": tags or [],
            "provenance_turn_ids": provenance_turn_ids or [],
            "session_id": self._session_id,
            "turn_index": 0,
            "role": "",
            "user_id": self._user_id,
            "agent_identity": self._agent_identity,
            "agent_workspace": self._agent_workspace,
            "platform": self._platform,
            "source": source,
            "created_at": utc_now(),
            "content_hash": content_hash(content, workspace=self._agent_workspace, kind="fact"),
        }

    def _build_turn_row(self, role: str, content: str, session_id: str) -> Dict[str, Any]:
        message_index = self._message_index
        self._message_index += 1
        return {
            "id": stable_turn_id(session_id, message_index, role, content),
            "kind": "turn",
            "content": content,
            "abstract": "",
            "category": "",
            "tags": [],
            "provenance_turn_ids": [],
            "session_id": session_id,
            "turn_index": message_index,
            "role": role,
            "user_id": self._user_id,
            "agent_identity": self._agent_identity,
            "agent_workspace": self._agent_workspace,
            "platform": self._platform,
            "source": "sync_turn",
            "created_at": utc_now(),
            "content_hash": content_hash(content, workspace=self._agent_workspace, kind="turn"),
        }

    def _extract_and_store(self, messages: List[Dict[str, Any]], *, source: str) -> list[dict[str, Any]]:
        if not self._should_write() or not self._config.get("extraction", {}).get("enabled", True):
            return []
        min_turns = int(self._config.get("extraction", {}).get("min_turns", 3))
        user_turns = sum(1 for msg in messages if msg.get("role") == "user")
        if user_turns < min_turns:
            return []
        facts = extract(messages, self._context())
        inserted = []
        for fact in facts:
            provenance_ids = self._evidence_to_turn_ids(messages, fact.get("evidence") or [])
            row = self.build_fact_row(
                content=fact["content"],
                abstract=fact.get("abstract", ""),
                category=fact.get("category", "general"),
                tags=fact.get("tags", []),
                provenance_turn_ids=provenance_ids,
                source=source,
            )
            if self.store.find_by_hash(row["content_hash"], workspace=self._agent_workspace, kind="fact"):
                continue
            self.store.add_row(row)
            inserted.append(row)
        return inserted

    def _evidence_to_turn_ids(self, messages: List[Dict[str, Any]], evidence: list[int]) -> list[str]:
        ids = []
        for idx in evidence:
            if idx < 0 or idx >= len(messages):
                continue
            msg = messages[idx]
            role = str(msg.get("role") or "")
            content = str(msg.get("content") or "")
            if role not in {"user", "assistant"} or not content:
                continue
            ids.append(stable_turn_id(self._session_id, idx, role, content))
        return ids

    def _context(self) -> Dict[str, Any]:
        return {
            "session_id": self._session_id,
            "platform": self._platform,
            "agent_identity": self._agent_identity,
            "agent_workspace": self._agent_workspace,
            "user_id": self._user_id,
        }

    def _should_write(self) -> bool:
        return self._agent_context not in {"cron", "subagent", "flush"}

    def _get_embedder(self) -> OpenAICompatibleEmbedder:
        if self._embedder is None:
            self._embedder = embedder_from_config(self._config.get("embedding", {}))
        return self._embedder

    def _get_reranker(self) -> Any:
        if self._reranker is not None:
            return self._reranker
        reranker_cfg = self._config.get("retrieval", {}).get("reranker", {}) or {}
        model_name = reranker_cfg.get("model") or DEFAULTS["retrieval"]["reranker"]["model"]
        try:
            from lancedb.rerankers import CrossEncoderReranker
        except ImportError as exc:
            logger.warning("lancedb cross-encoder reranker unavailable: %s", exc)
            return None
        try:
            logger.info("loading cross-encoder reranker %s", model_name)
            self._reranker = CrossEncoderReranker(model_name=model_name, column="content")
        except Exception as exc:
            logger.warning("failed to construct cross-encoder reranker (%s): %s", model_name, exc)
            self._reranker = None
        return self._reranker

    def _resolve_hermes_home(self) -> str:
        if self._hermes_home:
            return self._hermes_home
        try:
            from hermes_constants import get_hermes_home

            return str(get_hermes_home())
        except Exception:
            from pathlib import Path

            return str(Path.home() / ".hermes")
