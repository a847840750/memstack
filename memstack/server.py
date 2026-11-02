"""零依赖 HTTP 网关：让任何语言/框架的 agent 通过 HTTP 接入记忆。

运行：
    python -m memstack.server --db ./memstack.db --port 8765

接口：
    GET  /health
    POST /v1/remember      {messages, scope, event_time?, metadata?}
    POST /v1/search        {query, scope?, top_k?, token_budget?, include_history?}
    POST /v1/forget        {ids?|query?, scope?, hard?}
    POST /v1/consolidate   {scope?}
    GET  /v1/profile?tenant_id=..&user_id=..&agent_id=..
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .core import MemoryStack
from .storage import SQLiteStore
from .types import Scope


def make_handler(stack: MemoryStack, token: str | None = None) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "MemStack/0.1"

        def _auth(self) -> bool:
            if not token:
                return True
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {token}"

        def _send(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if not self._auth():
                self._send(401, {"ok": False, "error": "unauthorized"})
                return
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._send(200, {"ok": True, "version": "0.1.0"})
                return
            if parsed.path == "/v1/profile":
                query = parse_qs(parsed.query)
                scope = Scope(
                    tenant_id=(query.get("tenant_id") or ["default"])[0],
                    user_id=(query.get("user_id") or ["default"])[0],
                    agent_id=(query.get("agent_id") or ["default"])[0],
                    session_id=(query.get("session_id") or [None])[0],
                )
                blocks = [
                    {"label": b.label, "value": b.value, "limit": b.limit}
                    for b in stack.store.list_blocks(scope)
                ]
                self._send(200, {"ok": True, "blocks": blocks, "stats": stack.stats(scope)})
                return
            self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._auth():
                self._send(401, {"ok": False, "error": "unauthorized"})
                return
            body = self._body()
            scope = Scope.from_dict(body.get("scope") or {})
            try:
                if self.path == "/v1/remember":
                    result = stack.remember(
                        body.get("messages", ""), scope,
                        event_time=body.get("event_time"),
                        metadata=body.get("metadata"),
                    )
                    self._send(200, {"ok": True, "result": result.to_dict()})
                elif self.path == "/v1/search":
                    bundle = stack.recall(
                        str(body.get("query", "")), scope,
                        top_k=int(body.get("top_k") or 8),
                        token_budget=int(body.get("token_budget") or 800),
                        include_history=body.get("include_history"),
                    )
                    self._send(200, {"ok": True, "result": bundle.to_dict()})
                elif self.path == "/v1/forget":
                    ids = stack.forget(
                        memory_ids=body.get("ids"), query=body.get("query"),
                        scope=scope, hard=bool(body.get("hard", False)),
                    )
                    self._send(200, {"ok": True, "forgotten": ids})
                elif self.path == "/v1/consolidate":
                    self._send(200, {"ok": True, "result": stack.consolidate(scope)})
                else:
                    self._send(404, {"ok": False, "error": "not found"})
            except Exception as exc:  # 网关层兜底，避免线程崩溃
                self._send(500, {"ok": False, "error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def serve(db: str = ":memory:", host: str = "127.0.0.1", port: int = 8765, token: str | None = None) -> None:
    stack = MemoryStack(store=SQLiteStore(db))
    server = ThreadingHTTPServer((host, port), make_handler(stack, token))
    print(f"MemStack HTTP server: http://{host}:{port} (db={db})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        stack.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="MemStack HTTP gateway")
    parser.add_argument("--db", default=":memory:")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    serve(args.db, args.host, args.port, args.token)


if __name__ == "__main__":
    main()
