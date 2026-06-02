"""LanceDB storage layer for Hermes memory."""
from __future__ import annotations

import hashlib
import logging
import queue
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .embeddings import OpenAIEmbedder

logger = logging.getLogger(__name__)

TABLE_NAME = "memories"
SCHEMA_VERSION = 1

RESULT_COLUMNS = [
    "id",
    "kind",
    "content",
    "abstract",
    "category",
    "tags",
    "provenance_turn_ids",
    "session_id",
    "turn_index",
    "role",
    "user_id",
    "agent_identity",
    "agent_workspace",
    "platform",
    "source",
    "created_at",
    "schema_version",
    "content_hash",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def content_hash(content: str, *, workspace: str = "", kind: str = "") -> str:
    normalized = " ".join((content or "").strip().lower().split())
    payload = f"{kind}\0{workspace}\0{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def stable_turn_id(session_id: str, message_index: int, role: str, content: str) -> str:
    # Use role+content rather than the local message counter so extraction
    # evidence can resolve provenance even when Hermes includes system/tool
    # messages that sync_turn never receives.
    digest = hashlib.sha256(
        f"{session_id}\0{role}\0{content}".encode("utf-8")
    ).hexdigest()[:24]
    return f"turn_{digest}"


def quote_sql(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def build_filter(
    *,
    workspace: str = "",
    user_id: str = "",
    kind: str = "fact",
    category: str = "",
) -> str:
    clauses: list[str] = []
    if kind and kind != "any":
        clauses.append(f"kind = {quote_sql(kind)}")
    if workspace:
        clauses.append(f"agent_workspace = {quote_sql(workspace)}")
    if user_id:
        clauses.append(f"(user_id = {quote_sql(user_id)} OR user_id = '')")
    if category:
        clauses.append(f"category = {quote_sql(category)}")
    return " AND ".join(clauses)


class LanceDBStore:
    """Small synchronous store plus optional background writer queue."""

    def __init__(
        self,
        hermes_home: str | Path,
        embedder: OpenAIEmbedder,
        *,
        optimize_every_commits: int = 50,
        cleanup_older_than_days: int = 7,
        maintenance_enabled: bool = True,
    ) -> None:
        self.hermes_home = Path(hermes_home).expanduser()
        self.db_path = self.hermes_home / "lancedb"
        self.embedder = embedder
        self._db = None
        self._table = None
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=256)
        self._worker: threading.Thread | None = None
        self._closed = threading.Event()
        self._maintenance_enabled = bool(maintenance_enabled)
        self._optimize_every = max(0, int(optimize_every_commits))
        self._cleanup_older_than = (
            timedelta(days=int(cleanup_older_than_days))
            if cleanup_older_than_days and int(cleanup_older_than_days) > 0
            else None
        )
        self._optimize_lock = threading.Lock()
        self._optimize_state_path = self.db_path / ".last_optimize_version"

    @property
    def table(self):
        if self._table is None:
            self.open()
        return self._table

    def open(self) -> None:
        if self._table is not None:
            return
        import lancedb

        self.db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.db_path))
        try:
            self._table = self._db.open_table(TABLE_NAME)
        except Exception:
            self._table = self._db.create_table(TABLE_NAME, schema=self._schema())
        self._ensure_fts_index()

    def _schema(self):
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("kind", pa.string(), nullable=False),
                pa.field("content", pa.string(), nullable=False),
                pa.field("vector", pa.list_(pa.float32(), self.embedder.dim), nullable=False),
                pa.field("abstract", pa.string()),
                pa.field("category", pa.string()),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("provenance_turn_ids", pa.list_(pa.string())),
                pa.field("session_id", pa.string()),
                pa.field("turn_index", pa.int64()),
                pa.field("role", pa.string()),
                pa.field("user_id", pa.string()),
                pa.field("agent_identity", pa.string()),
                pa.field("agent_workspace", pa.string()),
                pa.field("platform", pa.string()),
                pa.field("source", pa.string()),
                pa.field("created_at", pa.timestamp("us", tz="UTC")),
                pa.field("schema_version", pa.int64()),
                pa.field("content_hash", pa.string()),
            ]
        )

    def _ensure_fts_index(self) -> None:
        try:
            self._table.create_fts_index("content")
        except Exception as exc:
            logger.debug("create_fts_index skipped or failed: %s", exc)

    def start_worker(self) -> None:
        self.open()
        if self._worker and self._worker.is_alive():
            return
        self._closed.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="lancedb-writer")
        self._worker.start()

    def enqueue(self, row: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            logger.warning("lancedb writer queue full; dropping memory row")

    def _worker_loop(self) -> None:
        batch: list[dict[str, Any]] = []
        while not self._closed.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                item = None
            if item is None:
                if batch:
                    self.add_rows(batch)
                    batch = []
                if self._closed.is_set():
                    return
                continue
            batch.append(item)
            self._queue.task_done()
            if len(batch) >= 16:
                self.add_rows(batch)
                batch = []

    def shutdown(self, timeout: float = 5.0) -> None:
        if self._worker and self._worker.is_alive():
            self._closed.set()
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._worker.join(timeout=timeout)
        leftovers = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item:
                leftovers.append(item)
        if leftovers:
            self.add_rows(leftovers)

    def add_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.open()
        prepared = self._prepare_rows(rows)
        if prepared:
            self.table.add(prepared)
            self._maybe_optimize()

    def add_row(self, row: dict[str, Any]) -> None:
        self.add_rows([row])

    def _read_last_optimize_version(self) -> int:
        try:
            return int(self._optimize_state_path.read_text().strip())
        except (FileNotFoundError, ValueError, OSError):
            return 0

    def _write_last_optimize_version(self, version: int) -> None:
        try:
            self._optimize_state_path.write_text(str(version))
        except OSError as exc:
            logger.debug("failed to persist last optimize version: %s", exc)

    def _maybe_optimize(self) -> None:
        if not self._maintenance_enabled or self._optimize_every <= 0 or self._table is None:
            return
        try:
            current = int(self._table.version)
        except Exception as exc:
            logger.debug("could not read table.version for compaction check: %s", exc)
            return
        last = self._read_last_optimize_version()
        if current - last < self._optimize_every:
            return
        # Non-blocking lock: if another optimize is in flight, skip this trigger
        # — the next write that crosses the threshold will pick it up.
        if not self._optimize_lock.acquire(blocking=False):
            return
        thread = threading.Thread(
            target=self._run_optimize,
            args=(current,),
            daemon=True,
            name="lancedb-optimize",
        )
        thread.start()

    def _run_optimize(self, version: int) -> None:
        try:
            logger.info(
                "lancedb optimize starting (version=%s, cleanup_older_than=%s)",
                version,
                self._cleanup_older_than,
            )
            if self._cleanup_older_than is not None:
                self._table.optimize(cleanup_older_than=self._cleanup_older_than)
            else:
                self._table.optimize()
            self._write_last_optimize_version(version)
            logger.info("lancedb optimize finished (version=%s)", version)
        except Exception as exc:
            logger.warning("lancedb optimize failed: %s", exc)
        finally:
            self._optimize_lock.release()

    def _prepare_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        texts = [str(row.get("content") or "") for row in rows]
        vectors = self.embedder.embed(texts)
        prepared = []
        for row, vector in zip(rows, vectors):
            item = dict(row)
            item.setdefault("id", make_id(str(item.get("kind") or "mem")))
            item.setdefault("abstract", "")
            item.setdefault("category", "")
            item.setdefault("tags", [])
            item.setdefault("provenance_turn_ids", [])
            item.setdefault("session_id", "")
            item.setdefault("turn_index", 0)
            item.setdefault("role", "")
            item.setdefault("user_id", "")
            item.setdefault("agent_identity", "")
            item.setdefault("agent_workspace", "")
            item.setdefault("platform", "")
            item.setdefault("source", "")
            item.setdefault("created_at", utc_now())
            item.setdefault("schema_version", SCHEMA_VERSION)
            item.setdefault(
                "content_hash",
                content_hash(
                    str(item.get("content") or ""),
                    workspace=str(item.get("agent_workspace") or ""),
                    kind=str(item.get("kind") or ""),
                ),
            )
            item["vector"] = vector
            prepared.append(item)
        return prepared

    def get_by_id(self, memory_id: str) -> Optional[dict[str, Any]]:
        if not memory_id:
            return None
        rows = (
            self.table.search()
            .where(f"id = {quote_sql(memory_id)}")
            .select(RESULT_COLUMNS)
            .limit(1)
            .to_list()
        )
        return rows[0] if rows else None

    def get_by_ids(self, ids: Iterable[str]) -> list[dict[str, Any]]:
        clean = [str(v) for v in ids if v]
        if not clean:
            return []
        values = ", ".join(quote_sql(v) for v in clean)
        return (
            self.table.search()
            .where(f"id IN ({values})")
            .select(RESULT_COLUMNS)
            .limit(len(clean))
            .to_list()
        )

    def find_by_hash(self, digest: str, *, workspace: str = "", kind: str = "fact") -> Optional[dict[str, Any]]:
        clauses = [f"content_hash = {quote_sql(digest)}"]
        if workspace:
            clauses.append(f"agent_workspace = {quote_sql(workspace)}")
        if kind:
            clauses.append(f"kind = {quote_sql(kind)}")
        rows = (
            self.table.search()
            .where(" AND ".join(clauses))
            .select(RESULT_COLUMNS)
            .limit(1)
            .to_list()
        )
        return rows[0] if rows else None

    def delete_by_id(self, memory_id: str) -> None:
        self.table.delete(f"id = {quote_sql(memory_id)}")
        self._maybe_optimize()
