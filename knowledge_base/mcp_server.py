"""Small Streamable HTTP MCP endpoint for the local knowledge base."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .store import build_state, list_files, search_documents

MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8080"))
MCP_PATH = "/mcp"
SESSIONS: dict[str, float] = {}
SESSIONS_LOCK = threading.Lock()


def _result(request_id: Any, value: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": value}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tools() -> list[dict[str, Any]]:
    return [
        {"name": "search_knowledge", "description": "Search the local Chroma knowledge base.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "n_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3}}, "required": ["query"]}},
        {"name": "list_knowledge", "description": "Return collection and document counts.", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "list_knowledge_files", "description": "List uploaded source files.", "inputSchema": {"type": "object", "properties": {}}},
    ]


def _call_tool(name: str, args: dict[str, Any]) -> Any:
    if name == "list_knowledge":
        state = build_state()
        return {"collection": state["collection"], "count": state["document_count"], "file_count": state["file_count"]}
    if name == "list_knowledge_files":
        return list_files()
    if name == "search_knowledge":
        query = args.get("query")
        count = args.get("n_results", 3)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10:
            raise ValueError("n_results must be between 1 and 10")
        return search_documents(query, count)
    raise ValueError(f"Unknown tool: {name}")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LocalKnowledgeBaseMCP/1.0"

    def _send(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK, session: str | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if session:
            self.send_header("Mcp-Session-Id", session)
        self.end_headers()
        self.wfile.write(body)

    def _empty(self, status: HTTPStatus, session: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        if session:
            self.send_header("Mcp-Session-Id", session)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == MCP_PATH:
            self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
            self.send_header("Allow", "POST, DELETE")
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._empty(HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        session = self.headers.get("Mcp-Session-Id")
        with SESSIONS_LOCK:
            SESSIONS.pop(session, None)
        self._empty(HTTPStatus.NO_CONTENT)

    def do_POST(self) -> None:
        if self.path != MCP_PATH:
            self._empty(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            self._send(_error(None, -32700, "Parse error"), HTTPStatus.BAD_REQUEST)
            return
        if not isinstance(body, dict):
            self._send(_error(None, -32600, "Invalid request"), HTTPStatus.BAD_REQUEST)
            return

        request_id, method, params = body.get("id"), body.get("method"), body.get("params") or {}
        session = self.headers.get("Mcp-Session-Id")
        if method == "initialize":
            session = uuid.uuid4().hex
            with SESSIONS_LOCK:
                SESSIONS[session] = time.monotonic()
        elif not session or session not in SESSIONS:
            self._send(_error(request_id, -32000, "Missing or expired MCP session"), HTTPStatus.BAD_REQUEST)
            return
        else:
            with SESSIONS_LOCK:
                SESSIONS[session] = time.monotonic()

        if method == "notifications/initialized":
            self._empty(HTTPStatus.ACCEPTED, session)
        elif method == "initialize":
            self._send(_result(request_id, {"protocolVersion": params.get("protocolVersion", "2025-03-26"), "capabilities": {"tools": {}}, "serverInfo": {"name": "local-knowledge-base", "version": "1.0.0"}}), session=session)
        elif method == "ping":
            self._send(_result(request_id, {}), session=session)
        elif method == "tools/list":
            self._send(_result(request_id, {"tools": _tools()}), session=session)
        elif method == "tools/call":
            try:
                value = _call_tool(params.get("name"), params.get("arguments") or {})
                payload = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False, default=str)}], "isError": False}
            except Exception as exc:
                payload = {"content": [{"type": "text", "text": str(exc)}], "isError": True}
            self._send(_result(request_id, payload), session=session)
        else:
            self._send(_error(request_id, -32601, f"Method not found: {method}"), HTTPStatus.NOT_FOUND, session)


def main() -> None:
    server = ThreadingHTTPServer((MCP_HOST, MCP_PORT), Handler)
    print(f"MCP server: http://{MCP_HOST}:{MCP_PORT}{MCP_PATH}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
