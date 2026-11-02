"""多 Agent 支持：agent 注册、团队共享空间、跨 agent 授权、事件总线。

设计原则（保持零第三方依赖与可移植性）：

1. 记忆本体仍然全部存于 MemoryStack 的 memories 表，用 Scope 表达归属：
   - 私有记忆：Scope(tenant, user_id="agents", agent_id=<agent_id>)
   - 团队共享记忆：Scope(tenant, user_id="team:<team_id>", agent_id="shared")
   这样与现有检索、巩固、遗忘管线完全兼容，不复制存储。
2. 元数据（agent 档案、团队、成员、授权、事件）存于独立 SQLite 表
   （hub 表），默认与 stack 同一个数据库文件，可单独指定。
3. 权限模型：
   - 私有记忆默认只有本人可读；通过 grant 显式授权其他 agent/team 读或写。
   - open 团队：任何注册 agent 可读，仅成员可写。
   - closed 团队：仅成员可读可写。
4. 事件总线是一等公民：post_event 写入 events 表，任何有读权限的 agent
   可以 pull_events(team, since_id) 增量拉取，再 remember_as 入自己的记忆。

示例：

    stack = MemoryStack()
    hub = MultiAgentHub(stack)
    hub.register_agent("planner", role="planner")
    hub.register_agent("executor", role="executor")
    hub.create_team("trip", members=["planner", "executor"], policy="closed")

    hub.remember_as("planner", [{"role": "user", "content": "预算上限 5000 元"}],
                    team="trip", share=True)
    bundle = hub.recall_as("executor", "这次旅行预算多少？", teams=["trip"])
    print(bundle.text)

    hub.post_event("trip", "planner", "plan_ready", "行程已确认，等待付款")
    events = hub.pull_events("executor", "trip")
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .core import MemoryStack
from .types import ContextBundle, Scope


# --------------------------------------------------------------------------------------
# 数据模型
# --------------------------------------------------------------------------------------
@dataclass
class AgentProfile:
    name: str
    role: str = "member"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class AccessDenied(PermissionError):
    """没有读/写权限时抛出；MCP 与 HTTP 层可映射为工具错误。"""


# --------------------------------------------------------------------------------------
# Hub
# --------------------------------------------------------------------------------------
class MultiAgentHub:
    """挂在 MemoryStack 之上的多 agent 协作层。"""

    def __init__(
        self,
        stack: MemoryStack,
        tenant_id: str = "default",
        hub_db: str | None = None,
    ) -> None:
        self.stack = stack
        self.tenant_id = tenant_id
        if hub_db is None:
            store_path = getattr(stack.store, "path", ":memory:")
            hub_db = store_path if store_path != ":memory:" else ":memory:"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(hub_db, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ----------------------------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS hub_agents (
                    name TEXT PRIMARY KEY,
                    role TEXT NOT NULL DEFAULT 'member',
                    metadata TEXT,
                    created_at REAL
                );
                CREATE TABLE IF NOT EXISTS hub_teams (
                    team_id TEXT PRIMARY KEY,
                    name TEXT,
                    policy TEXT NOT NULL DEFAULT 'closed',  -- open | closed
                    metadata TEXT,
                    created_at REAL
                );
                CREATE TABLE IF NOT EXISTS hub_memberships (
                    team_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'member',    -- owner | member
                    joined_at REAL,
                    PRIMARY KEY(team_id, agent_id)
                );
                CREATE TABLE IF NOT EXISTS hub_grants (
                    owner_agent TEXT NOT NULL,   -- 被授权人是谁的记忆
                    principal TEXT NOT NULL,     -- 谁获得权限（agent 或 team:xxx）
                    access TEXT NOT NULL,        -- read | write
                    granted_by TEXT,
                    created_at REAL,
                    PRIMARY KEY(owner_agent, principal, access)
                );
                CREATE TABLE IF NOT EXISTS hub_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    type TEXT,
                    text TEXT NOT NULL,
                    metadata TEXT,
                    ts REAL
                );
                CREATE INDEX IF NOT EXISTS idx_hub_events_team
                    ON hub_events(team_id, id);
                """
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----------------------------------------------------------------------------------
    # 注册与团队
    # ----------------------------------------------------------------------------------
    def register_agent(self, name: str, role: str = "member", metadata: dict[str, Any] | None = None) -> AgentProfile:
        profile = AgentProfile(name=name, role=role, metadata=metadata or {}, created_at=time.time())
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO hub_agents(name, role, metadata, created_at) VALUES(?,?,?,?)"
                " ON CONFLICT(name) DO UPDATE SET role=excluded.role, metadata=excluded.metadata",
                (name, role, json.dumps(profile.metadata, ensure_ascii=False), profile.created_at),
            )
        return profile

    def get_agent(self, name: str) -> AgentProfile | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM hub_agents WHERE name=?", (name,)
            ).fetchone()
        if not row:
            return None
        return AgentProfile(
            name=row["name"], role=row["role"],
            metadata=json.loads(row["metadata"] or "{}"), created_at=row["created_at"],
        )

    def create_team(
        self,
        team_id: str,
        members: Sequence[str] | None = None,
        policy: str = "closed",
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if policy not in {"open", "closed"}:
            raise ValueError("policy 必须是 open 或 closed")
        members = list(members or [])
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO hub_teams(team_id, name, policy, metadata, created_at) VALUES(?,?,?,?,?)"
                " ON CONFLICT(team_id) DO UPDATE SET name=excluded.name, policy=excluded.policy",
                (team_id, name or team_id, policy, json.dumps(metadata or {}, ensure_ascii=False), now),
            )
            for member in members:
                self._conn.execute(
                    "INSERT OR IGNORE INTO hub_memberships(team_id, agent_id, role, joined_at) VALUES(?,?,?,?)",
                    (team_id, member, "owner" if member == members[0] else "member", now),
                )
        return {"team_id": team_id, "policy": policy, "members": members}

    def add_member(self, team_id: str, agent_id: str, role: str = "member") -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO hub_memberships(team_id, agent_id, role, joined_at) VALUES(?,?,?,?)"
                " ON CONFLICT(team_id, agent_id) DO UPDATE SET role=excluded.role",
                (team_id, agent_id, role, time.time()),
            )

    def team_members(self, team_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT agent_id FROM hub_memberships WHERE team_id=? ORDER BY role, joined_at",
                (team_id,),
            ).fetchall()
        return [row["agent_id"] for row in rows]

    def list_agents(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM hub_agents ORDER BY name").fetchall()
        return [
            {"name": r["name"], "role": r["role"], "metadata": json.loads(r["metadata"] or "{}")}
            for r in rows
        ]

    def list_teams(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT t.team_id, t.name, t.policy, t.metadata, t.created_at FROM hub_teams t"
                " ORDER BY t.created_at"
            ).fetchall()
            memberships: dict[str, list[dict[str, str]]] = {}
            member_rows = self._conn.execute(
                "SELECT team_id, agent_id, role FROM hub_memberships"
            ).fetchall()
        for row in member_rows:
            memberships.setdefault(row["team_id"], []).append(
                {"agent_id": row["agent_id"], "role": row["role"]}
            )
        return [
            {
                "team_id": r["team_id"], "name": r["name"], "policy": r["policy"],
                "metadata": json.loads(r["metadata"] or "{}"),
                "members": memberships.get(r["team_id"], []),
                "visible": agent_id is None or r["policy"] == "open"
                or any(m["agent_id"] == agent_id for m in memberships.get(r["team_id"], [])),
            }
            for r in rows
        ]

    # ----------------------------------------------------------------------------------
    # 作用域与权限
    # ----------------------------------------------------------------------------------
    @staticmethod
    def _private_scope(agent_id: str, tenant_id: str) -> Scope:
        return Scope(tenant_id=tenant_id, user_id="agents", agent_id=agent_id)

    def _shared_scope(self, team_id: str) -> Scope:
        return Scope(tenant_id=self.tenant_id, user_id=f"team:{team_id}", agent_id="shared")

    def _require_agent(self, agent_id: str) -> None:
        if not self.get_agent(agent_id):
            raise ValueError(f"未知 agent: {agent_id}；请先 register_agent")

    def _is_member(self, team_id: str, agent_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM hub_memberships WHERE team_id=? AND agent_id=?",
                (team_id, agent_id),
            ).fetchone()
        return row is not None

    def _team_policy(self, team_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT policy FROM hub_teams WHERE team_id=?", (team_id,)
            ).fetchone()
        return row["policy"] if row else None

    def _check_read(self, team_id: str, agent_id: str) -> None:
        policy = self._team_policy(team_id)
        if policy is None:
            raise ValueError(f"未知团队: {team_id}")
        if policy == "closed" and not self._is_member(team_id, agent_id):
            raise AccessDenied(f"agent '{agent_id}' 不是 closed 团队 '{team_id}' 的成员，不可读共享记忆")

    def _check_write(self, team_id: str, agent_id: str) -> None:
        policy = self._team_policy(team_id)
        if policy is None:
            raise ValueError(f"未知团队: {team_id}")
        if not self._is_member(team_id, agent_id):
            if policy == "closed":
                raise AccessDenied(f"agent '{agent_id}' 不是团队 '{team_id}' 的成员，不可写共享记忆")

    # ----------------------------------------------------------------------------------
    # 授权
    # ----------------------------------------------------------------------------------
    def grant(self, owner_agent: str, principal: str, access: str) -> None:
        """把 owner_agent 的私有记忆授权给 principal（agent 或 team:xxx）读/写。"""
        if access not in {"read", "write"}:
            raise ValueError("access 必须是 read 或 write")
        self._require_agent(owner_agent)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO hub_grants(owner_agent, principal, access, granted_by, created_at)"
                " VALUES(?,?,?,?,?) ON CONFLICT(owner_agent, principal, access) DO NOTHING",
                (owner_agent, principal, access, owner_agent, time.time()),
            )

    def revoke(self, owner_agent: str, principal: str, access: str | None = None) -> None:
        with self._lock, self._conn:
            if access:
                self._conn.execute(
                    "DELETE FROM hub_grants WHERE owner_agent=? AND principal=? AND access=?",
                    (owner_agent, principal, access),
                )
            else:
                self._conn.execute(
                    "DELETE FROM hub_grants WHERE owner_agent=? AND principal=?",
                    (owner_agent, principal),
                )

    def _granted_owners(self, agent_id: str, access: str = "read") -> list[str]:
        """谁授权了 agent_id 读/写其私有记忆。"""
        with self._lock:
            team_rows = self._conn.execute(
                "SELECT team_id FROM hub_memberships WHERE agent_id=?", (agent_id,)
            ).fetchall()
            principals = [agent_id] + [f"team:{r['team_id']}" for r in team_rows]
            placeholders = ",".join("?" for _ in principals)
            rows = self._conn.execute(
                f"SELECT DISTINCT owner_agent FROM hub_grants WHERE access=? AND principal IN ({placeholders})",
                [access] + principals,
            ).fetchall()
        return [row["owner_agent"] for row in rows]

    def list_access(self, agent_id: str) -> dict[str, Any]:
        """agent 对他人私有记忆的访问权，以及对自己记忆的授权。"""
        return {
            "can_read": self._granted_owners(agent_id, "read"),
            "can_write": self._granted_owners(agent_id, "write"),
            "granted_to": self._outgoing_grants(agent_id),
        }

    def _outgoing_grants(self, owner_agent: str) -> list[dict[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT principal, access FROM hub_grants WHERE owner_agent=?",
                (owner_agent,),
            ).fetchall()
        return [{"principal": r["principal"], "access": r["access"]} for r in rows]

    # ----------------------------------------------------------------------------------
    # 写入与召回
    # ----------------------------------------------------------------------------------
    def remember_as(
        self,
        agent_id: str,
        messages: Any,
        team_id: str | None = None,
        share: bool = False,
        event_time: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """以 agent 身份写入。share=True 时同步写一份到团队共享空间。"""
        self._require_agent(agent_id)
        private = self.stack.remember(
            messages, self._private_scope(agent_id, self.tenant_id),
            event_time=event_time, metadata=metadata,
        )
        result: dict[str, Any] = {"private": private.to_dict(), "shared": None}
        if team_id and (share or team_id):
            self._check_write(team_id, agent_id)
            shared = self.stack.remember(
                messages, self._shared_scope(team_id),
                event_time=event_time,
                metadata={
                    **(metadata or {}),
                    "sender": agent_id,
                    "origin": "agent-share",
                },
            )
            result["shared"] = shared.to_dict()
        return result

    def recall_as(
        self,
        agent_id: str,
        query: str,
        teams: Sequence[str] | None = None,
        token_budget: int = 800,
        top_k: int = 8,
        include_grants: bool = True,
        **kwargs: Any,
    ) -> ContextBundle:
        """以 agent 身份召回：私有记忆 + 有权限的团队共享记忆 + 被授权的他人记忆。"""
        self._require_agent(agent_id)
        private = self.stack.recall(
            query, self._private_scope(agent_id, self.tenant_id),
            token_budget=token_budget, top_k=top_k, **kwargs,
        )
        bundles: list[tuple[str, ContextBundle]] = [("agent-private", private)]
        for team_id in teams or []:
            self._check_read(team_id, agent_id)
            shared = self.stack.recall(
                query, self._shared_scope(team_id),
                token_budget=token_budget, top_k=top_k, **kwargs,
            )
            bundles.append((f"team:{team_id}", shared))
        if include_grants:
            for owner in self._granted_owners(agent_id, "read"):
                granted = self.stack.recall(
                    query, self._private_scope(owner, self.tenant_id),
                    token_budget=token_budget, top_k=top_k, **kwargs,
                )
                bundles.append((f"granted:{owner}", granted))
        return self._merge_bundles(query, bundles, token_budget)

    @staticmethod
    def _merge_bundles(
        query: str,
        bundles: Sequence[tuple[str, ContextBundle]],
        token_budget: int,
    ) -> ContextBundle:
        """把多个 ContextBundle 拼成带来源段落的单一上下文。"""
        budget = max(token_budget, 200)
        cap = max(120, budget // len(bundles))
        sections: list[str] = []
        items = []
        total = 0
        for label, bundle in bundles:
            text = bundle.text if len(bundle.text) <= cap else bundle.text[:cap]
            sections.append(f"## [{label}]\n{text}")
            total += bundle.token_estimate // len(bundles)
            for hit in bundle.items:
                hit.reason = f"[{label}] {hit.reason}" if hit.reason else f"[{label}]"
                items.append(hit)
        items.sort(key=lambda h: h.score, reverse=True)
        items = items[: len(bundles) * 8]
        merged_text = "\n\n".join(sections)[: budget * 4]
        best = bundles[0][1]
        return ContextBundle(
            text=merged_text,
            items=items,
            query=query,
            token_estimate=min(total, budget),
            confidence=max(b.confidence for _, b in bundles),
            coverage=sum(b.coverage for _, b in bundles) / max(len(bundles), 1),
            gaps=[g for _, b in bundles for g in b.gaps],
            conflicts=[c for _, b in bundles for c in b.conflicts],
            should_continue=any(b.should_continue for _, b in bundles),
            trace={"merge": {"parts": [label for label, _ in bundles]},
                   **{f"part.{label}": b.trace for label, b in bundles}},
        )

    # ----------------------------------------------------------------------------------
    # 事件总线
    # ----------------------------------------------------------------------------------
    def post_event(
        self,
        team_id: str,
        sender: str,
        event_type: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        as_memory: bool = False,
    ) -> int:
        """向团队发布事件；as_memory=True 时同步写入共享记忆空间。"""
        self._check_write(team_id, sender)
        ts = time.time()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT INTO hub_events(team_id, sender, type, text, metadata, ts) VALUES(?,?,?,?,?,?)",
                (team_id, sender, event_type, text, json.dumps(metadata or {}, ensure_ascii=False), ts),
            )
            event_id = cursor.lastrowid
        if as_memory:
            self.remember_as(
                sender,
                [{"role": "user", "content": f"[{event_type}] {text}"}],
                team_id=team_id, share=True, metadata={**(metadata or {}), "sender": sender},
            )
        return int(event_id)

    def pull_events(
        self,
        agent_id: str,
        team_id: str,
        since_id: int = 0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """增量拉取团队事件；任何有读权限的 agent 可拉取。"""
        self._check_read(team_id, agent_id)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM hub_events WHERE team_id=? AND id>? ORDER BY id LIMIT ?",
                (team_id, since_id, limit),
            ).fetchall()
        return [
            {
                "id": row["id"], "team_id": row["team_id"], "sender": row["sender"],
                "type": row["type"], "text": row["text"],
                "metadata": json.loads(row["metadata"] or "{}"), "ts": row["ts"],
            }
            for row in rows
        ]

    # ----------------------------------------------------------------------------------
    # 其它
    # ----------------------------------------------------------------------------------
    def team_profile(self, team_id: str, agent_id: str) -> dict[str, Any]:
        self._check_read(team_id, agent_id)
        scope = self._shared_scope(team_id)
        return self.stack.stats(scope)

    def forget_shared(self, team_id: str, agent_id: str, ids: Sequence[str] | None = None, query: str | None = None) -> list[str]:
        """删除团队共享记忆；返回实际删除的记忆 id 列表。"""
        self._check_write(team_id, agent_id)
        return self.stack.forget(memory_ids=list(ids) if ids else None, query=query, scope=self._shared_scope(team_id))

    def stats(self) -> dict[str, Any]:
        with self._lock:
            agents = self._conn.execute("SELECT COUNT(*) AS n FROM hub_agents").fetchone()["n"]
            teams = self._conn.execute("SELECT COUNT(*) AS n FROM hub_teams").fetchone()["n"]
            events = self._conn.execute("SELECT COUNT(*) AS n FROM hub_events").fetchone()["n"]
        return {"agents": agents, "teams": teams, "events": events, "tenant_id": self.tenant_id}