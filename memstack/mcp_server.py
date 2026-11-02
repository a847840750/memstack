"""零依赖 MCP server：stdio + JSON-RPC 2.0，可被任意 MCP 客户端挂载。

实现说明（不依赖任何第三方包）：
  - 传输：标准输入/输出上的**换行分隔 JSON**（MCP stdio 最常见形态）；
    同时兼容 `Content-Length` 头格式，遇到即解析。
  - 协议：MCP 2025-03-26 的核心子集——initialize / ping / tools/list / tools/call，
    外加 notifications/initialized 的静默应答。
  - 工具：把 MemoryStack 的 OpenAI 工具面（memory_add/search/forget/profile/consolidate）
    自动转换成 MCP 工具，并追加多 agent 工具（agent_register/team_create/agent_remember/
    agent_recall/event_post/event_pull）。

用法：
    python -m memstack.mcp_server --db ./mem.db          # 交互式 stdio
    python -m memstack.mcp_server list-tools              # 打印工具清单便于调试

客户端配置示例（Claude Desktop / Cursor mcp.json）：
    {"mcpServers": {"memstack": {"command": "python", "args": ["-m", "memstack.mcp_server", "--db", "mem.db"]}}}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from typing import Any, Sequence

from .core import MemoryStack
from .multiagent import AccessDenied, MultiAgentHub
from .tools import dispatch_tool, openai_tool_specs

PROTOCOL_VERSION = "2025-03-26"
SERVER_NAME = "memstack"
SERVER_VERSION = "0.3.0"


# --------------------------------------------------------------------------------------
# 工具面
# --------------------------------------------------------------------------------------
def _mcp_tools(stack: MemoryStack, hub: MultiAgentHub | None) -> list[dict[str, Any]]:
    """把 OpenAI 函数定义转成 MCP tool 定义，并追加多 agent 工具。"""
    tools: list[dict[str, Any]] = []
    for spec in openai_tool_specs():
        fn = spec.get("function", {})
        tools.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "inputSchema": fn.get("parameters", {"type": "object"}),
            }
        )
    tools.append(
        {
            "name": "agent_register",
            "description": "注册一个 agent（多 agent 协作的前提），返回档案；重复注册会更新角色与元数据。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string", "default": "member"},
                    "metadata": {"type": "object"},
                },
                "required": ["name"],
            },
        }
    )
    tools.append(
        {
            "name": "team_create",
            "description": "创建团队共享记忆空间。closed 团队仅成员可读；open 团队任何人可读。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "team_id": {"type": "string"},
                    "members": {"type": "array", "items": {"type": "string"}},
                    "policy": {"type": "string", "enum": ["open", "closed"], "default": "closed"},
                },
                "required": ["team_id", "members"],
            },
        }
    )
    tools.append(
        {
            "name": "agent_remember",
            "description": "以 agent 身份写入记忆；team_id 存在时同步写入团队共享空间。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "messages": {"oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "object"}},
                    ]},
                    "team_id": {"type": "string"},
                    "share": {"type": "boolean", "default": True},
                    "event_time": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["agent_id", "messages"],
            },
        }
    )
    tools.append(
        {
            "name": "agent_recall",
            "description": "以 agent 身份检索：私有记忆 + 有权限的团队共享记忆 + 被授权(grant)的他人记忆，返回带来源段落的上下文。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "query": {"type": "string"},
                    "teams": {"type": "array", "items": {"type": "string"}},
                    "token_budget": {"type": "integer", "default": 800},
                    "top_k": {"type": "integer", "default": 8},
                },
                "required": ["agent_id", "query"],
            },
        }
    )
    tools.append(
        {
            "name": "agent_grant",
            "description": "把某个 agent 的私有记忆授权给另一个 agent（或 team:id）读取。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "owner_agent": {"type": "string"},
                    "principal": {"type": "string"},
                    "access": {"type": "string", "enum": ["read", "write"], "default": "read"},
                },
                "required": ["owner_agent", "principal"],
            },
        }
    )
    tools.append(
        {
            "name": "event_post",
            "description": "向团队发布事件（agent 间消息）；as_memory=true 时同步存入共享记忆。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "team_id": {"type": "string"},
                    "sender": {"type": "string"},
                    "event_type": {"type": "string"},
                    "text": {"type": "string"},
                    "as_memory": {"type": "boolean", "default": False},
                },
                "required": ["team_id", "sender", "event_type", "text"],
            },
        }
    )
    tools.append(
        {
            "name": "event_pull",
            "description": "增量拉取团队事件流（since_id 之后的事件）。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "team_id": {"type": "string"},
                    "since_id": {"type": "integer", "default": 0},
                    "limit": {"type": "integer", "default": 50},
                },
                "required": ["agent_id", "team_id"],
            },
        }
    )
    return tools


def _call_tool(
    name: str,
    arguments: dict[str, Any],
    stack: MemoryStack,
    hub: MultiAgentHub | None,
) -> Any:
    """执行一个工具调用；工具错误（如越权）以结构化错误返回而非抛异常。"""
    if hub is None and name.startswith(("agent_", "team_", "event_")):
        return {"error": "本 server 未启用多 agent 支持（--hub 未开启）"}
    try:
        if hub is not None and name in {
            "agent_register", "team_create", "agent_remember", "agent_recall",
            "agent_grant", "event_post", "event_pull",
        }:
            return _MultiAgentTools.call(hub, name, arguments)
        return dispatch_tool(stack, name, arguments)
    except (AccessDenied, ValueError, KeyError) as exc:
        return {"error": str(exc)}


# --------------------------------------------------------------------------------------
# JSON-RPC 循环
# --------------------------------------------------------------------------------------
class _MultiAgentTools:
    """多 agent 工具的 MCP 适配：把 JSON 参数映射到 MultiAgentHub 的方法。"""

    @staticmethod
    def call(hub: MultiAgentHub, name: str, arguments: dict[str, Any]) -> Any:
        if name == "agent_register":
            profile = hub.register_agent(
                str(arguments["name"]),
                role=str(arguments.get("role", "member")),
                metadata=arguments.get("metadata"),
            )
            return profile.to_dict()
        if name == "team_create":
            return hub.create_team(
                str(arguments["team_id"]),
                members=[str(m) for m in (arguments.get("members") or [])],
                policy=str(arguments.get("policy", "closed")),
            )
        if name == "agent_remember":
            return hub.remember_as(
                str(arguments["agent_id"]),
                arguments["messages"],
                team_id=arguments.get("team_id"),
                share=bool(arguments.get("share", True)),
                event_time=arguments.get("event_time"),
                metadata=arguments.get("metadata"),
            )
        if name == "agent_recall":
            bundle = hub.recall_as(
                str(arguments["agent_id"]),
                str(arguments["query"]),
                teams=arguments.get("teams"),
                token_budget=int(arguments.get("token_budget", 800)),
                top_k=int(arguments.get("top_k", 8)),
            )
            return bundle.to_dict()
        if name == "agent_grant":
            hub.grant(
                str(arguments["owner_agent"]),
                str(arguments["principal"]),
                str(arguments.get("access", "read")),
            )
            return {"ok": True}
        if name == "event_post":
            event_id = hub.post_event(
                str(arguments["team_id"]),
                str(arguments["sender"]),
                str(arguments["event_type"]),
                str(arguments["text"]),
                as_memory=bool(arguments.get("as_memory", False)),
            )
            return {"event_id": event_id}
        if name == "event_pull":
            return hub.pull_events(
                str(arguments["agent_id"]),
                str(arguments["team_id"]),
                since_id=int(arguments.get("since_id", 0)),
                limit=int(arguments.get("limit", 50)),
            )
        return {"error": f"未知多 agent 工具: {name}"}


def _dispatch(
    stack: MemoryStack,
    hub: MultiAgentHub | None,
    message: dict[str, Any],
) -> dict[str, Any] | None:
    method = message.get("method", "")
    msg_id = message.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": _mcp_tools(stack, hub)},
        }
    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        result = _call_tool(name, arguments, stack, hub)
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                "isError": isinstance(result, dict) and "error" in result,
            },
        }
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def _write(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def serve_stdio(stack: MemoryStack, hub: MultiAgentHub | None = None) -> None:
    """运行 stdio JSON-RPC 循环，直到 stdin 关闭。"""
    buffer = ""
    content_length: int | None = None
    for raw_line in sys.stdin:
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue
        if content_length is None:
            match = re.match(r"^Content-Length:\s*(\d+)$", line, re.I)
            if match:
                content_length = int(match.group(1))
                continue
        if content_length is not None:
            buffer = line
            if len(buffer.encode("utf-8")) >= content_length:
                line, content_length = buffer, None
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _write({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
            continue
        try:
            response = _dispatch(stack, hub, message)
        except Exception as exc:  # noqa: BLE001 - MCP 循环不能因单个工具崩溃
            response = {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"},
            }
        if response is not None:
            _write(response)


# --------------------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MemStack MCP server (stdio JSON-RPC)")
    parser.add_argument("--db", default=":memory:", help="SQLite 数据库路径；默认内存库")
    parser.add_argument("--hub", action="store_true", help="启用多 agent 支持（agent/team/event 工具）")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("list-tools", nargs="?", help="打印工具清单后退出")
    args = parser.parse_args()
    stack = MemoryStack()
    hub = MultiAgentHub(stack, tenant_id=args.tenant) if args.hub else None
    if vars(args)["list-tools"] == "list-tools":
        print(json.dumps({"tools": _mcp_tools(stack, hub)}, ensure_ascii=False, indent=2))
        return
    try:
        serve_stdio(stack, hub)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()