"""Mock downstream MCP server (Streamable-HTTP style, JSON responses).

Exposes a few normal tools and two admin_ tools. It records every tools/call it
receives so tests can prove the gateway never let an unauthorised call through.
If the request's Accept header asks only for text/event-stream it answers as SSE,
to exercise the gateway's streaming pass-through.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

TOOLS = [
    {"name": "get_weather", "description": "Weather for a city", "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}}},
    {"name": "list_files", "description": "List files", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "admin_reset_key", "description": "Rotate an API key", "inputSchema": {"type": "object", "properties": {"key_id": {"type": "string"}}}},
    {"name": "admin_delete_user", "description": "Delete a user", "inputSchema": {"type": "object", "properties": {"user_id": {"type": "string"}}}},
]


def create_mock_app() -> FastAPI:
    app = FastAPI(title="Mock MCP server")
    app.state.calls: list[dict[str, Any]] = []
    app.state.headers_seen: list[dict[str, str]] = []

    @app.post("/mcp")
    async def mcp(request: Request) -> Response:
        app.state.headers_seen.append(dict(request.headers))
        msg = await request.json()
        method, rid = msg.get("method"), msg.get("id")

        if rid is None:  # notification
            return Response(status_code=202)
        if method == "initialize":
            result: Any = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-downstream", "version": "0.1"},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            app.state.calls.append(msg["params"])
            name = msg["params"]["name"]
            result = {"content": [{"type": "text", "text": f"executed {name}"}], "isError": False}
        elif method == "ping":
            result = {}
        else:
            return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}})

        payload = {"jsonrpc": "2.0", "id": rid, "result": result}
        accept = request.headers.get("accept", "")
        if "text/event-stream" in accept and "application/json" not in accept:
            async def sse():
                yield f"event: message\ndata: {json.dumps(payload)}\n\n".encode()
            return StreamingResponse(sse(), media_type="text/event-stream", headers={"mcp-session-id": "sess-123"})
        return JSONResponse(payload, headers={"mcp-session-id": "sess-123"})

    return app


app = create_mock_app()

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("MOCK_MCP_PORT", "9100")))
