"""Agent 接入层：OpenAI function-calling 工具 + 轻量中间件。

任何 agent 只要能调用函数（OpenAI/Anthropic/MCP/自研），就可以通过这里的
schema 与 dispatch 接入 MemStack，不需要理解内部存储与检索算法。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .types import ContextBundle, Scope


def openai_tool_specs() -> list[dict[str, Any]]:
    scope_schema = {
        "type": "object",
        "description": "记忆作用域；多租户/多用户隔离。",
        "properties": {
            "tenant_id": {"type": "string"},
            "user_id": {"type": "string"},
            "agent_id": {"type": "string"},
            "session_id": {"type": "string"},
        },
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": "检索与当前任务相关的长期记忆，返回带来源与置信度的上下文片段。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "要检索的问题或关键词"},
                        "top_k": {"type": "integer", "default": 8},
                        "token_budget": {"type": "integer", "default": 800},
                        "include_history": {
                            "type": "boolean",
                            "description": "是否包含已失效/历史事实；省略时由系统根据 query 自动判断",
                        },
                        "scope": scope_schema,
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_add",
                "description": "把一段对话、事件或文档写入长期记忆；系统会自动抽取事实、去重与处理冲突。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "object"}},
                            ],
                            "description": "字符串或 [{role, content}] 消息列表",
                        },
                        "event_time": {
                            "type": "string",
                            "description": "事件发生时间（ISO8601），省略则为当前时间",
                        },
                        "metadata": {"type": "object"},
                        "scope": scope_schema,
                    },
                    "required": ["messages"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_forget",
                "description": "按 id 或查询软删除记忆；默认保留审计记录，hard=true 才物理删除。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "ids": {"type": "array", "items": {"type": "string"}},
                        "hard": {"type": "boolean", "default": False},
                        "scope": scope_schema,
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_profile",
                "description": "读取用户画像块与记忆库统计，适合在会话开始时加载。",
                "parameters": {
                    "type": "object",
                    "properties": {"scope": scope_schema},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_consolidate",
                "description": "离线巩固：去重、冲突消解、生成会话摘要、刷新画像。建议在会话结束后调用。",
                "parameters": {
                    "type": "object",
                    "properties": {"scope": scope_schema},
                },
            },
        },
    ]


def _parse_scope(arguments: dict[str, Any], scope: Scope | None) -> Scope:
    base = (scope or Scope()).to_dict()
    override = arguments.get("scope") or {}
    if not isinstance(override, dict):
        override = {}
    merged = {**base, **{k: v for k, v in override.items() if v is not None}}
    return Scope.from_dict(merged)


def _parse_event_time(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def dispatch_tool(
    stack: Any,
    name: str,
    arguments: dict[str, Any] | str,
    scope: Scope | None = None,
) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return {"ok": False, "error": "arguments 不是合法 JSON"}
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "arguments 必须是对象"}

    active_scope = _parse_scope(arguments, scope)
    if name == "memory_search":
        bundle = stack.recall(
            query=str(arguments.get("query", "")),
            scope=active_scope,
            top_k=int(arguments.get("top_k") or 8),
            token_budget=int(arguments.get("token_budget") or 800),
            include_history=arguments.get("include_history"),
        )
        return {"ok": True, "result": bundle.to_dict()}

    if name == "memory_add":
        messages = arguments.get("messages")
        if messages is None:
            return {"ok": False, "error": "缺少 messages"}
        result = stack.remember(
            messages=messages,
            scope=active_scope,
            event_time=_parse_event_time(arguments.get("event_time")),
            metadata=dict(arguments.get("metadata") or {}),
        )
        return {"ok": True, "result": result.to_dict()}

    if name == "memory_forget":
        ids = stack.forget(
            memory_ids=arguments.get("ids"),
            query=arguments.get("query"),
            scope=active_scope,
            hard=bool(arguments.get("hard", False)),
        )
        return {"ok": True, "result": {"forgotten": ids}}

    if name == "memory_profile":
        blocks = [
            {"label": b.label, "value": b.value, "limit": b.limit, "updated_at": b.updated_at}
            for b in stack.store.list_blocks(active_scope)
        ]
        return {"ok": True, "result": {"blocks": blocks, "stats": stack.stats(active_scope)}}

    if name == "memory_consolidate":
        return {"ok": True, "result": stack.consolidate(active_scope)}

    return {"ok": False, "error": f"未知工具: {name}"}


class MemoryMiddleware:
    """最小 agent 中间件：before 注入记忆，after 写回交互。"""

    def __init__(self, stack: Any, scope: Scope | None = None, token_budget: int = 800) -> None:
        self.stack = stack
        self.scope = scope or Scope()
        self.token_budget = token_budget

    def before(self, query: str) -> tuple[str, ContextBundle]:
        bundle = self.stack.recall(query, self.scope, token_budget=self.token_budget)
        system = (
            "你有一个外部长期记忆库。下面是按相关度筛选出的记忆，"
            "请优先使用带 slot/evidence 的记忆，并在不确定时明确说明。\n\n"
            f"{bundle.text}\n\n"
            "如果记忆不足，调用 memory_search 继续检索；不要编造未出现的事实。"
        )
        return system, bundle

    def after(self, messages: Any, metadata: dict[str, Any] | None = None) -> Any:
        return self.stack.remember(messages, self.scope, metadata=metadata)
