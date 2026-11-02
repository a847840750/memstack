"""MemStack 核心数据类型。

设计原则：所有跨层传递的数据都用 dataclass / Enum 表达，存储层只负责序列化，
这样上层（写入、检索、适配器）不依赖任何具体数据库或框架。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# --------------------------------------------------------------------------------------
# 枚举
# --------------------------------------------------------------------------------------
class MemoryKind(str, Enum):
    """记忆类型。对应认知科学中的情景/语义/程序记忆，外加摘要与实体索引。"""

    EPISODE = "episode"        # 原始交互窗口（可验证证据）
    FACT = "fact"              # 原子事实命题（semantic）
    PREFERENCE = "preference"  # 用户偏好（semantic 的子类，需要时效与冲突处理）
    PROCEDURE = "procedure"    # 可复用流程（procedural）
    SUMMARY = "summary"        # 会话/主题摘要（导航层）
    ENTITY = "entity"          # 实体索引节点（只用于图扩展，不直接进上下文）
    REFLECTION = "reflection"  # 离线归纳出的高层结论


class MemoryStatus(str, Enum):
    """记忆生命周期状态。删除默认是软删除，保留可审计历史。"""

    ACTIVE = "active"
    SUPERSEDED = "superseded"  # 被更新的事实取代，但保留用于历史查询
    ARCHIVED = "archived"      # 低价值/过期，默认不参与检索
    DELETED = "deleted"        # 软删除，仅审计可见


class EdgeType(str, Enum):
    """记忆图边类型。"""

    PROVENANCE = "provenance"    # 派生记忆 -> 原始证据
    MENTIONS = "mentions"        # 记忆 -> 实体
    SIMILAR = "similar"          # 语义近邻
    SUPERSEDES = "supersedes"    # 新事实 -> 被取代的旧事实
    DERIVED_FROM = "derived_from"  # 摘要/反思 -> 来源记忆
    NEXT = "next"                # 同一会话内的时序链


class WriteAction(str, Enum):
    ADD = "ADD"
    MERGE = "MERGE"
    SUPERSEDE = "SUPERSEDE"
    NOOP = "NOOP"


# --------------------------------------------------------------------------------------
# 作用域
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Scope:
    """多租户/多用户/多 Agent 隔离。session_id 为空表示跨会话。"""

    tenant_id: str = "default"
    user_id: str = "default"
    agent_id: str = "default"
    session_id: str | None = None

    def as_filter(self, include_session: bool = True) -> dict[str, str]:
        data = {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
        }
        if include_session and self.session_id:
            data["session_id"] = self.session_id
        return data

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Scope":
        data = data or {}
        return cls(
            tenant_id=str(data.get("tenant_id", "default")),
            user_id=str(data.get("user_id", "default")),
            agent_id=str(data.get("agent_id", "default")),
            session_id=data.get("session_id"),
        )


# --------------------------------------------------------------------------------------
# 记忆条目
# --------------------------------------------------------------------------------------
@dataclass
class MemoryItem:
    """统一的记忆节点。不同 kind 共享同一结构，差异放在 data 里。"""

    id: str
    kind: MemoryKind
    text: str
    scope: Scope
    data: dict[str, Any] = field(default_factory=dict)
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: float = 0.0
    updated_at: float = 0.0
    valid_from: float = 0.0
    valid_to: float | None = None
    last_access: float = 0.0
    access_count: int = 0
    importance: float = 0.5
    confidence: float = 0.8
    embedding: list[float] | None = None
    source_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def slot(self) -> tuple[str, str, str] | None:
        """返回 (subject, predicate, object)，没有结构化槽位则返回 None。"""
        subject = self.data.get("subject")
        predicate = self.data.get("predicate")
        obj = self.data.get("object")
        if subject and predicate and obj is not None:
            return str(subject), str(predicate), str(obj)
        return None

    @property
    def entities(self) -> list[str]:
        return list(self.data.get("entities") or [])

    @property
    def is_current(self) -> bool:
        return self.status == MemoryStatus.ACTIVE and self.valid_to is None

    def touch(self, now: float) -> None:
        self.last_access = now
        self.access_count += 1


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------------------
# 查询与结果
# --------------------------------------------------------------------------------------
@dataclass
class RecallQuery:
    text: str
    scope: Scope
    top_k: int = 8
    token_budget: int = 800
    kinds: list[MemoryKind] | None = None
    include_history: bool | None = None  # None = 自动根据 query 判断
    time_range: tuple[float, float] | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    min_score: float = 0.0


@dataclass
class RecallHit:
    memory: MemoryItem
    score: float
    channel_scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.memory.id,
            "kind": self.memory.kind.value,
            "text": self.memory.text,
            "status": self.memory.status.value,
            "score": round(self.score, 4),
            "channels": {k: round(v, 4) for k, v in self.channel_scores.items()},
            "reason": self.reason,
            "valid_from": self.memory.valid_from,
            "valid_to": self.memory.valid_to,
            "source_ids": self.memory.source_ids,
        }


@dataclass
class ContextBundle:
    """最终注入 agent 的上下文包，附带可解释性与是否继续检索的信号。"""

    text: str
    items: list[RecallHit]
    query: str
    token_estimate: int
    confidence: float
    coverage: float
    gaps: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    should_continue: bool = False
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.text,
            "items": [h.to_dict() for h in self.items],
            "query": self.query,
            "token_estimate": self.token_estimate,
            "confidence": round(self.confidence, 4),
            "coverage": round(self.coverage, 4),
            "gaps": self.gaps,
            "conflicts": self.conflicts,
            "should_continue": self.should_continue,
            "trace": self.trace,
        }


@dataclass
class ExtractedMemory:
    """抽取器的输出，尚未分配 id / 作用域。"""

    kind: MemoryKind
    text: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    entities: list[str] = field(default_factory=list)
    importance: float = 0.5
    confidence: float = 0.8
    event_time: float | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RememberDecision:
    action: WriteAction
    memory_id: str
    kind: str
    text: str
    target_id: str | None = None
    reason: str = ""


@dataclass
class RememberResult:
    episode_ids: list[str] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)
    decisions: list[RememberDecision] = field(default_factory=list)
    extracted: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_ids": self.episode_ids,
            "memory_ids": self.memory_ids,
            "extracted": self.extracted,
            "decisions": [
                {
                    "action": d.action.value,
                    "memory_id": d.memory_id,
                    "kind": d.kind,
                    "text": d.text,
                    "target_id": d.target_id,
                    "reason": d.reason,
                }
                for d in self.decisions
            ],
        }


@dataclass
class MemoryBlock:
    """Letta 风格的常驻上下文块（core memory）。"""

    label: str
    value: str
    limit: int = 2000
    updated_at: float = 0.0
    scope: Scope = field(default_factory=Scope)


# --------------------------------------------------------------------------------------
# 时间工具
# --------------------------------------------------------------------------------------
def utc_now() -> float:
    return time.time()


def iso(ts: float | None) -> str:
    if not ts:
        return "-"
    return time.strftime("%Y-%m-%d", time.gmtime(ts))
