"""SQLite 存储层：默认零依赖、单文件、可嵌入；同时实现 MemoryStore 协议。

表结构刻意保持简单（memories / edges / blocks / audit），方便迁移到 Postgres、
Redis 或图数据库：上层只依赖 Store 协议的方法，不依赖 SQL。
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
import time
from typing import Any, Iterable, Sequence

from .types import MemoryBlock, MemoryItem, MemoryKind, MemoryStatus, Scope


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _pack_vec(vec: Sequence[float] | None) -> bytes | None:
    """把向量打包成 float32 BLOB：约省一半空间，反序列化比 JSON 快一个量级。"""
    if vec is None:
        return None
    return struct.pack(f"<{len(vec)}f", *[float(v) for v in vec])


def _unpack_vec(blob: Any) -> list[float] | None:
    """解包 float32 BLOB；兼容旧版 JSON 字符串格式（str 也走 json 路径）。"""
    if blob is None:
        return None
    if isinstance(blob, (bytes, bytearray, memoryview)):
        n = len(blob) // 4
        if n * 4 != len(blob):
            return _loads(blob if isinstance(blob, str) else blob.decode("utf-8", "replace"), None)
        return list(struct.unpack(f"<{n}f", bytes(blob)))
    return _loads(blob, None)


class SQLiteStore:
    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    # ----------------------------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    session_id TEXT,
                    data TEXT,
                    status TEXT NOT NULL,
                    created_at REAL,
                    updated_at REAL,
                    valid_from REAL,
                    valid_to REAL,
                    last_access REAL,
                    access_count INTEGER DEFAULT 0,
                    importance REAL DEFAULT 0.5,
                    confidence REAL DEFAULT 0.8,
                    embedding BLOB,
                    source_ids TEXT,
                    meta TEXT,
                    content_hash TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_mem_scope
                    ON memories(tenant_id, user_id, agent_id, session_id);
                CREATE INDEX IF NOT EXISTS idx_mem_kind_status
                    ON memories(kind, status);
                CREATE INDEX IF NOT EXISTS idx_mem_hash
                    ON memories(content_hash);
                CREATE TABLE IF NOT EXISTS edges (
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    created_at REAL,
                    meta TEXT,
                    PRIMARY KEY(src, dst, type)
                );
                CREATE INDEX IF NOT EXISTS idx_edge_src ON edges(src);
                CREATE INDEX IF NOT EXISTS idx_edge_dst ON edges(dst);
                CREATE TABLE IF NOT EXISTS blocks (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    value TEXT,
                    "limit" INTEGER DEFAULT 2000,
                    updated_at REAL,
                    PRIMARY KEY(tenant_id, user_id, agent_id, label)
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    op TEXT,
                    scope TEXT,
                    detail TEXT
                );
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                """
            )

    # ----------------------------------------------------------------------------------
    def save(self, item: MemoryItem) -> None:
        scope = item.scope
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO memories (
                    id, kind, text, tenant_id, user_id, agent_id, session_id, data, status,
                    created_at, updated_at, valid_from, valid_to, last_access, access_count,
                    importance, confidence, embedding, source_ids, meta, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    text=excluded.text, data=excluded.data, status=excluded.status,
                    updated_at=excluded.updated_at, valid_to=excluded.valid_to,
                    last_access=excluded.last_access, access_count=excluded.access_count,
                    importance=excluded.importance, confidence=excluded.confidence,
                    embedding=excluded.embedding, source_ids=excluded.source_ids,
                    meta=excluded.meta, content_hash=excluded.content_hash
                """,
                (
                    item.id, item.kind.value, item.text,
                    scope.tenant_id, scope.user_id, scope.agent_id, scope.session_id,
                    _dumps(item.data), item.status.value,
                    item.created_at, item.updated_at, item.valid_from, item.valid_to,
                    item.last_access, item.access_count, item.importance, item.confidence,
                    _pack_vec(item.embedding),
                    _dumps(item.source_ids), _dumps(item.meta),
                    item.meta.get("content_hash"),
                ),
            )
            self._bump_revision_locked()

    def get(self, memory_id: str) -> MemoryItem | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return self._row_to_item(row) if row else None

    def exists(self, memory_id: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM memories WHERE id=?", (memory_id,)).fetchone()
        return row is not None

    def find(
        self,
        scope: Scope,
        kinds: Sequence[str] | None = None,
        statuses: Sequence[MemoryStatus] | None = None,
        session_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int | None = None,
    ) -> list[MemoryItem]:
        clauses = ["tenant_id=?", "user_id=?", "agent_id=?"]
        params: list[Any] = [scope.tenant_id, scope.user_id, scope.agent_id]
        if session_id:
            clauses.append("session_id=?")
            params.append(session_id)
        if kinds:
            clauses.append(f"kind IN ({','.join('?' for _ in kinds)})")
            params.extend([k.value if isinstance(k, MemoryKind) else str(k) for k in kinds])
        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            params.extend([s.value if isinstance(s, MemoryStatus) else str(s) for s in statuses])
        if since is not None:
            clauses.append("(valid_to IS NULL OR valid_to >= ?)")
            params.append(since)
        if until is not None:
            clauses.append("valid_from <= ?")
            params.append(until)
        sql = "SELECT * FROM memories WHERE " + " AND ".join(clauses)
        sql += " ORDER BY importance DESC, created_at DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_item(row) for row in rows]

    def find_by_hash(self, scope: Scope, content_hash: str, kind: str) -> MemoryItem | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM memories
                   WHERE tenant_id=? AND user_id=? AND agent_id=? AND kind=?
                     AND content_hash=? AND status!=?
                   LIMIT 1""",
                (scope.tenant_id, scope.user_id, scope.agent_id, kind, content_hash,
                 MemoryStatus.DELETED.value),
            ).fetchone()
        return self._row_to_item(row) if row else None

    def ids_in_scope(self, scope: Scope) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM memories WHERE tenant_id=? AND user_id=? AND agent_id=?",
                (scope.tenant_id, scope.user_id, scope.agent_id),
            ).fetchall()
        return {row["id"] for row in rows}

    def update_status(self, memory_id: str, status: MemoryStatus, valid_to: float | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE memories SET status=?, valid_to=?, updated_at=? WHERE id=?",
                (status.value, valid_to, time.time(), memory_id),
            )
            self._bump_revision_locked()

    def bump_access(self, memory_ids: Iterable[str], now: float) -> None:
        ids = list(memory_ids)
        if not ids:
            return
        with self._lock, self._conn:
            # 访问计数只影响 prior 通道的实时打分（每次查询按需读最新值），
            # 不影响 BM25 倒排 / 图邻接缓存的内容，因此不 bump revision，
            # 否则每次检索都会让索引缓存失效。
            self._conn.executemany(
                "UPDATE memories SET last_access=?, access_count=access_count+1 WHERE id=?",
                [(now, mid) for mid in ids],
            )

    def delete(self, memory_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            self._conn.execute("DELETE FROM edges WHERE src=? OR dst=?", (memory_id, memory_id))
            self._bump_revision_locked()

    # ----------------------------------------------------------------------------------
    def put_edge(
        self,
        src: str,
        dst: str,
        edge_type: str,
        weight: float = 1.0,
        meta: dict[str, Any] | None = None,
        created_at: float | None = None,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO edges(src, dst, type, weight, created_at, meta)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(src, dst, type) DO UPDATE SET
                     weight=excluded.weight, meta=excluded.meta""",
                (src, dst, edge_type, weight, created_at or time.time(), _dumps(meta or {})),
            )
            self._bump_revision_locked()

    def edges(
        self,
        src: str | None = None,
        dst: str | None = None,
        types: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if src:
            clauses.append("src=?")
            params.append(src)
        if dst:
            clauses.append("dst=?")
            params.append(dst)
        if types:
            clauses.append(f"type IN ({','.join('?' for _ in types)})")
            params.extend(types)
        sql = "SELECT * FROM edges"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "src": row["src"],
                "dst": row["dst"],
                "type": row["type"],
                "weight": row["weight"],
                "created_at": row["created_at"],
                "meta": _loads(row["meta"], {}),
            }
            for row in rows
        ]

    def delete_edges(self, src: str, dst: str | None = None, types: Sequence[str] | None = None) -> None:
        clauses = ["src=?"]
        params: list[Any] = [src]
        if dst:
            clauses.append("dst=?")
            params.append(dst)
        if types:
            clauses.append(f"type IN ({','.join('?' for _ in types)})")
            params.extend(types)
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM edges WHERE " + " AND ".join(clauses), params)
            self._bump_revision_locked()

    # ----------------------------------------------------------------------------------
    def set_block(self, scope: Scope, label: str, value: str, limit: int = 2000) -> MemoryBlock:
        value = value[:limit]
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO blocks(tenant_id, user_id, agent_id, label, value, "limit", updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(tenant_id, user_id, agent_id, label) DO UPDATE SET
                     value=excluded.value, "limit"=excluded."limit", updated_at=excluded.updated_at""",
                (scope.tenant_id, scope.user_id, scope.agent_id, label, value, limit, now),
            )
            self._bump_revision_locked()
        return MemoryBlock(label=label, value=value, limit=limit, updated_at=now, scope=scope)

    def get_block(self, scope: Scope, label: str) -> MemoryBlock | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM blocks WHERE tenant_id=? AND user_id=? AND agent_id=? AND label=?""",
                (scope.tenant_id, scope.user_id, scope.agent_id, label),
            ).fetchone()
        if not row:
            return None
        return MemoryBlock(
            label=row["label"], value=row["value"] or "", limit=row["limit"],
            updated_at=row["updated_at"] or 0.0, scope=scope,
        )

    def list_blocks(self, scope: Scope) -> list[MemoryBlock]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM blocks WHERE tenant_id=? AND user_id=? AND agent_id=?
                   ORDER BY label""",
                (scope.tenant_id, scope.user_id, scope.agent_id),
            ).fetchall()
        return [
            MemoryBlock(label=r["label"], value=r["value"] or "", limit=r["limit"],
                        updated_at=r["updated_at"] or 0.0, scope=scope)
            for r in rows
        ]

    # ----------------------------------------------------------------------------------
    def audit(self, op: str, scope: Scope, detail: dict[str, Any] | None = None) -> None:
        with self._lock, self._conn:
            # 审计只追加日志，不改变检索可见数据，因此不 bump revision
            self._conn.execute(
                "INSERT INTO audit(ts, op, scope, detail) VALUES (?, ?, ?, ?)",
                (time.time(), op, _dumps(scope.to_dict()), _dumps(detail or {})),
            )

    def list_audit(
        self,
        scope: Scope | None = None,
        op: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """查询审计日志（治理能力：谁在何时对哪些记忆做了什么）。"""
        clauses = []
        params: list[Any] = []
        if scope is not None:
            clauses.append("scope=?")
            params.append(_dumps(scope.to_dict()))
        if op:
            clauses.append("op=?")
            params.append(op)
        sql = "SELECT * FROM audit"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "id": row["id"],
                "ts": row["ts"],
                "op": row["op"],
                "scope": _loads(row["scope"], {}),
                "detail": _loads(row["detail"], {}),
            }
            for row in rows
        ]

    # ----------------------------------------------------------------------------------
    # 数据版本号：检索层用它做一致性缓存（任何写操作使缓存失效）
    # ----------------------------------------------------------------------------------
    def revision(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key='data_revision'"
            ).fetchone()
        return int(row["value"]) if row else 0

    def _bump_revision_locked(self) -> None:
        """必须在已持有 self._lock 与事务内调用。"""
        self._conn.execute(
            """INSERT INTO meta(key, value) VALUES ('data_revision', '1')
               ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"""
        )

    def stats(self, scope: Scope) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT kind, status, COUNT(*) AS n FROM memories
                   WHERE tenant_id=? AND user_id=? AND agent_id=?
                   GROUP BY kind, status""",
                (scope.tenant_id, scope.user_id, scope.agent_id),
            ).fetchall()
            edge_count = self._conn.execute(
                """SELECT COUNT(*) AS n FROM edges e
                   WHERE EXISTS (SELECT 1 FROM memories m WHERE m.id=e.src
                                 AND m.tenant_id=? AND m.user_id=? AND m.agent_id=?)""",
                (scope.tenant_id, scope.user_id, scope.agent_id),
            ).fetchone()["n"]
        counts: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for row in rows:
            counts[row["kind"]] = counts.get(row["kind"], 0) + row["n"]
            by_status[row["status"]] = by_status.get(row["status"], 0) + row["n"]
        return {"by_kind": counts, "by_status": by_status, "edges": edge_count}

    def export_scope(self, scope: Scope) -> dict[str, Any]:
        items = self.find(scope)
        ids = {item.id for item in items}
        blocks = self.list_blocks(scope)
        edges = [e for e in self.edges() if e["src"] in ids and e["dst"] in ids]
        return {
            "version": 1,
            "scope": scope.to_dict(),
            "memories": [self._item_to_dict(item) for item in items],
            "edges": edges,
            "blocks": [
                {"label": b.label, "value": b.value, "limit": b.limit, "updated_at": b.updated_at}
                for b in blocks
            ],
        }

    def import_data(self, payload: dict[str, Any]) -> int:
        count = 0
        for row in payload.get("memories", []):
            item = self._dict_to_item(row)
            self.save(item)
            count += 1
        for edge in payload.get("edges", []):
            self.put_edge(
                edge["src"], edge["dst"], edge["type"],
                float(edge.get("weight", 1.0)), edge.get("meta"), edge.get("created_at"),
            )
        scope = Scope.from_dict(payload.get("scope") or {})
        for block in payload.get("blocks", []):
            self.set_block(scope, block["label"], block.get("value", ""), int(block.get("limit", 2000)))
        return count

    # ----------------------------------------------------------------------------------
    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            kind=MemoryKind(row["kind"]),
            text=row["text"],
            scope=Scope(
                tenant_id=row["tenant_id"], user_id=row["user_id"],
                agent_id=row["agent_id"], session_id=row["session_id"],
            ),
            data=_loads(row["data"], {}),
            status=MemoryStatus(row["status"]),
            created_at=row["created_at"] or 0.0,
            updated_at=row["updated_at"] or 0.0,
            valid_from=row["valid_from"] or 0.0,
            valid_to=row["valid_to"],
            last_access=row["last_access"] or 0.0,
            access_count=row["access_count"] or 0,
            importance=row["importance"] or 0.5,
            confidence=row["confidence"] or 0.8,
            embedding=_unpack_vec(row["embedding"]),
            source_ids=_loads(row["source_ids"], []),
            meta=_loads(row["meta"], {}),
        )

    def _item_to_dict(self, item: MemoryItem) -> dict[str, Any]:
        return {
            "id": item.id,
            "kind": item.kind.value,
            "text": item.text,
            "scope": item.scope.to_dict(),
            "data": item.data,
            "status": item.status.value,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "valid_from": item.valid_from,
            "valid_to": item.valid_to,
            "last_access": item.last_access,
            "access_count": item.access_count,
            "importance": item.importance,
            "confidence": item.confidence,
            "embedding": item.embedding,
            "source_ids": item.source_ids,
            "meta": item.meta,
        }

    def _dict_to_item(self, row: dict[str, Any]) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            kind=MemoryKind(row.get("kind", "fact")),
            text=row.get("text", ""),
            scope=Scope.from_dict(row.get("scope")),
            data=dict(row.get("data") or {}),
            status=MemoryStatus(row.get("status", "active")),
            created_at=float(row.get("created_at") or 0.0),
            updated_at=float(row.get("updated_at") or 0.0),
            valid_from=float(row.get("valid_from") or 0.0),
            valid_to=row.get("valid_to"),
            last_access=float(row.get("last_access") or 0.0),
            access_count=int(row.get("access_count") or 0),
            importance=float(row.get("importance") or 0.5),
            confidence=float(row.get("confidence") or 0.8),
            embedding=row.get("embedding"),
            source_ids=list(row.get("source_ids") or []),
            meta=dict(row.get("meta") or {}),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
