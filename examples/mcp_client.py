"""用标准库驱动 MemStack MCP server 的测试客户端（零 mcp SDK 依赖）。

先启动 server（端口无关，走 stdio）：
    python -m memstack.mcp_client  --demo            # 内置演示：直接驱动进程内 server
或手动：
    python -m memstack.mcp_server --hub | python -m memstack.mcp_client --stdin-mode
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class McpStdioClient:
    """极简 MCP stdio 客户端：把 JSON-RPC 请求发送给子进程并读回响应。"""

    def __init__(self, command: list[str]) -> None:
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        self.next_id = 0
        self._request(
            "initialize", {"protocolVersion": "2025-03-26", "capabilities": {}}
        )
        self._notify("notifications/initialized", {})

    def _request(self, method: str, params: dict) -> dict:
        self.next_id += 1
        message = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params}
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        return json.loads(line)

    def _notify(self, method: str, params: dict) -> None:
        self.next_id += 1
        message = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params}
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def list_tools(self) -> list[dict]:
        response = self._request("tools/list", {})
        return response["result"]["tools"]

    def call_tool(self, name: str, arguments: dict) -> dict:
        response = self._request("tools/call", {"name": name, "arguments": arguments})
        result = response["result"]
        text = result["content"][0]["text"] if result.get("content") else ""
        return {"isError": result.get("isError", False), "text": text}

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.terminate()
        self.proc.wait(timeout=10)


def run_demo() -> None:
    command = [sys.executable, "-m", "memstack.mcp_server", "--hub"]
    client = McpStdioClient(command)
    try:
        tools = client.list_tools()
        print(f"server 暴露 {len(tools)} 个工具:", ", ".join(t["name"] for t in tools[:6]), "...")

        print(client.call_tool("agent_register", {"name": "alice", "role": "researcher"}))
        print(client.call_tool("team_create", {"team_id": "lab", "members": ["alice"]}))
        print(client.call_tool(
            "agent_remember",
            {"agent_id": "alice", "team_id": "lab",
             "messages": [{"role": "user", "content": "实验环境用 Python 3.12。"}]},
        ))
        recall = client.call_tool(
            "agent_recall", {"agent_id": "alice", "team_id": "lab", "query": "实验环境 Python 版本"}
        )
        body = json.loads(recall["text"])
        print("agent_recall 命中:", "3.12" in body["context"], "| 置信度:", body["confidence"])
    finally:
        client.close()


if __name__ == "__main__":
    run_demo()
    print("MCP 客户端演示完成")