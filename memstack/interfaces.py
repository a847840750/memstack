"""可移植性契约：记忆系统只依赖这些 Protocol，不依赖任何具体框架。

接入方可以替换任意一层：
  - Embedder：本地 hash / OpenAI / BGE / 自研
  - Extractor：规则 / LLM / 领域模型
  - Store：SQLite / Postgres / Redis / 向量库
  - Reranker：启发式 / cross-encoder
  - Clock：真实时间 / 测试模拟时间
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from .types import MemoryBlock, MemoryItem, MemoryStatus, Scope


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


@runtime_checkable
class Extractor(Protocol):
    def extract(self, text: str, event_time: float, context: dict[str, Any] | None = None) -> list[ExtractedMemory]:
        ...


@runtime_checkable
class Summarizer(Protocol):
    def summarize(self, texts: Sequence[str], max_chars: int = 600) -> str:
        ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[MemoryItem]) -> list[float]:
        ...


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float:
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """存储层完整接口。默认实现是 SQLiteStore。

    这是“可移植性”的核心契约：换 Postgres/Redis/图库时，只需实现这些方法，
    写入、检索、巩固、工具层都不需要改。
    """

    def save(self, item: MemoryItem) -> None:
        ...

    def get(self, memory_id: str) -> MemoryItem | None:
        ...

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
        ...

    def find_by_hash(self, scope: Scope, content_hash: str, kind: str) -> MemoryItem | None:
        ...

    def ids_in_scope(self, scope: Scope) -> set[str]:
        ...

    def update_status(self, memory_id: str, status: MemoryStatus, valid_to: float | None = None) -> None:
        ...

    def bump_access(self, memory_ids: Sequence[str], now: float) -> None:
        ...

    def delete(self, memory_id: str) -> None:
        ...

    def put_edge(
        self,
        src: str,
        dst: str,
        edge_type: str,
        weight: float = 1.0,
        meta: dict[str, Any] | None = None,
        created_at: float | None = None,
    ) -> None:
        ...

    def edges(self, src: str | None = None, dst: str | None = None, types: Sequence[str] | None = None) -> list[dict[str, Any]]:
        ...

    def delete_edges(self, src: str, dst: str | None = None, types: Sequence[str] | None = None) -> None:
        ...

    def set_block(self, scope: Scope, label: str, value: str, limit: int = 2000) -> MemoryBlock:
        ...

    def get_block(self, scope: Scope, label: str) -> MemoryBlock | None:
        ...

    def list_blocks(self, scope: Scope) -> list[MemoryBlock]:
        ...

    def audit(self, op: str, scope: Scope, detail: dict[str, Any] | None = None) -> None:
        ...

    def list_audit(self, scope: Scope | None = None, op: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        ...

    def revision(self) -> int:
        """数据版本号：任何写操作加一；检索层用它做一致性缓存。"""
        ...

    def stats(self, scope: Scope) -> dict[str, Any]:
        ...

    def export_scope(self, scope: Scope) -> dict[str, Any]:
        ...

    def import_data(self, payload: dict[str, Any]) -> int:
        ...

    def close(self) -> None:
        ...
