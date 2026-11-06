"""Task 2 - MCP Security Gateway (HTTP / JSON-RPC reverse proxy).

    AI agent --(Bearer JWT)--> [ this gateway ] --(no client token)--> downstream MCP server

Request pipeline for POST /mcp
------------------------------
1. Authenticate   Bearer <JWT> -> Principal(sub, role). Failure: HTTP 401 + JSON-RPC error.
2. Parse          Body must be one JSON-RPC 2.0 object. Bad JSON -> -32700, bad shape -> -32600.
                  JSON arrays (batches) are refused: MCP 2025-06-18 removed batching, and
                  accepting them would let an admin_ call hide inside a batch.
3. Authorise      tools/call whose params.name starts with "admin_" requires role=admin.
                  Otherwise -> JSON-RPC error -32001 "Unauthorized Tool Call", returned by the
                  gateway itself. The downstream server is never contacted.
4. Forward        Everything else (tools/list, initialize, ping, allowed tools/call, ...) is
                  proxied with the ORIGINAL body bytes, and the response is streamed back
                  unchanged, so JSON and SSE (text/event-stream) responses both work.

Security notes
--------------
* The client's bearer token is NOT forwarded downstream ("token passthrough" is
  explicitly forbidden by the MCP authorization spec). Only an allow-list of MCP headers
  is forwarded, plus X-Gateway-* identity headers the downstream may trust.
* Tool-name check is normalised (strip + casefold) so "ADMIN_reset_key" or
  " admin_reset_key" cannot slip past a naive case-sensitive prefix test.
* Downstream failures map to a generic JSON-RPC error; internal hostnames, stack traces
  and upstream bodies are logged, never returned.
* Every authorisation decision is audit-logged (sub, role, method, tool, decision).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from auth import AuthError, Principal, extract_bearer, verify_token

log = logging.getLogger("mcp_gateway")
audit = logging.getLogger("mcp_gateway.audit")

# JSON-RPC 2.0 codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
# Implementation-defined server errors (-32000..-32099)
UNAUTHENTICATED = -32000
UNAUTHORIZED_TOOL_CALL = -32001
DOWNSTREAM_UNAVAILABLE = -32002

FORWARD_REQUEST_HEADERS = ("content-type", "accept", "mcp-session-id", "mcp-protocol-version", "last-event-id")
FORWARD_RESPONSE_HEADERS = ("content-type", "mcp-session-id", "cache-control")
MAX_BODY_BYTES = 1_000_000


@dataclass
class Settings:
    downstream_url: str = field(default_factory=lambda: os.getenv("DOWNSTREAM_MCP_URL", "http://127.0.0.1:9100/mcp"))
    jwt_secret: str = field(default_factory=lambda: os.getenv("GATEWAY_JWT_SECRET", "dev-only-change-me"))
    admin_prefix: str = "admin_"
    timeout_seconds: float = 30.0


def rpc_error(req_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _valid_id(obj: Any) -> Any:
    """Echo the request id only if it is a legal JSON-RPC id, else null."""
    if isinstance(obj, dict):
        rid = obj.get("id")
        if isinstance(rid, (str, int)) and not isinstance(rid, bool):
            return rid
    return None


def is_admin_tool(name: str, prefix: str) -> bool:
    return name.strip().casefold().startswith(prefix.casefold())


def create_app(settings: Settings | None = None, downstream_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or Settings()
    client = httpx.AsyncClient(transport=downstream_transport, timeout=settings.timeout_seconds)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="MCP Security Gateway", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    def authenticate(request: Request) -> Principal:
        return verify_token(extract_bearer(request.headers.get("authorization")), settings.jwt_secret)

    def unauthenticated(message: str, req_id: Any = None) -> JSONResponse:
        return JSONResponse(
            rpc_error(req_id, UNAUTHENTICATED, message),
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="mcp-gateway"'},
        )

    @app.post("/mcp")
    async def mcp_post(request: Request) -> Response:
        request_id = str(uuid.uuid4())
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return JSONResponse(rpc_error(None, INVALID_REQUEST, "Request too large"), status_code=413)

        # 1. parse (before auth only so we can echo the id in a 401; nothing is executed)
        try:
            msg = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            msg = None
            parse_failed = True
        else:
            parse_failed = False

        # 2. authenticate
        try:
            principal = authenticate(request)
        except AuthError as exc:
            audit.info("deny reason=unauthenticated detail=%s rid=%s", exc, request_id)
            return unauthenticated(str(exc), _valid_id(msg))

        if parse_failed:
            return JSONResponse(rpc_error(None, PARSE_ERROR, "Parse error"), status_code=400)
        if isinstance(msg, list):
            return JSONResponse(
                rpc_error(None, INVALID_REQUEST, "Batch requests are not supported (MCP 2025-06-18)"),
                status_code=400,
            )
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            # Responses from the client (e.g. to sampling requests) have no "method"; let them through.
            if isinstance(msg, dict) and msg.get("jsonrpc") == "2.0" and ("result" in msg or "error" in msg):
                return await forward(request, body, principal, request_id)
            return JSONResponse(rpc_error(_valid_id(msg), INVALID_REQUEST, "Invalid Request"), status_code=400)

        method: str = msg["method"]
        req_id = _valid_id(msg)

        # 3. authorise
        if method == "tools/call":
            params = msg.get("params")
            name = params.get("name") if isinstance(params, dict) else None
            if not isinstance(name, str) or not name.strip():
                return JSONResponse(rpc_error(req_id, INVALID_PARAMS, "params.name must be a non-empty string"))
            if is_admin_tool(name, settings.admin_prefix) and not principal.is_admin:
                audit.warning(
                    "deny reason=role sub=%s role=%s method=%s tool=%s rid=%s",
                    principal.subject, principal.role, method, name, request_id,
                )
                return JSONResponse(
                    rpc_error(
                        req_id,
                        UNAUTHORIZED_TOOL_CALL,
                        "Unauthorized Tool Call",
                        {"tool": name, "required_role": "admin"},
                    ),
                    status_code=200,  # JSON-RPC-level failure on a well-formed HTTP exchange
                )
            audit.info(
                "allow sub=%s role=%s method=%s tool=%s rid=%s", principal.subject, principal.role, method, name, request_id
            )
        else:
            audit.info("allow sub=%s role=%s method=%s rid=%s", principal.subject, principal.role, method, request_id)

        # 4. forward
        return await forward(request, body, principal, request_id, req_id)

    @app.get("/mcp")
    async def mcp_get(request: Request) -> Response:
        """Server-to-client SSE stream (Streamable HTTP transport). Auth required, then proxied."""
        try:
            principal = authenticate(request)
        except AuthError as exc:
            return unauthenticated(str(exc))
        return await forward(request, b"", principal, str(uuid.uuid4()), method="GET")

    @app.delete("/mcp")
    async def mcp_delete(request: Request) -> Response:
        """Session termination. Auth required, then proxied."""
        try:
            principal = authenticate(request)
        except AuthError as exc:
            return unauthenticated(str(exc))
        return await forward(request, b"", principal, str(uuid.uuid4()), method="DELETE")

    async def forward(
        request: Request,
        body: bytes,
        principal: Principal,
        request_id: str,
        req_id: Any = None,
        method: str = "POST",
    ) -> Response:
        headers = {k: v for k in FORWARD_REQUEST_HEADERS if (v := request.headers.get(k)) is not None}
        headers.update({"x-gateway-subject": principal.subject, "x-gateway-role": principal.role, "x-request-id": request_id})
        upstream_req = client.build_request(method, settings.downstream_url, content=body, headers=headers)
        try:
            upstream = await client.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            log.error("downstream request failed rid=%s err=%r", request_id, exc)
            return JSONResponse(
                rpc_error(req_id, DOWNSTREAM_UNAVAILABLE, "Downstream MCP server unavailable", {"request_id": request_id}),
                status_code=502,
            )

        resp_headers = {k: v for k in FORWARD_RESPONSE_HEADERS if (v := upstream.headers.get(k)) is not None}
        resp_headers["x-request-id"] = request_id
        # Stream the body through untouched (works for application/json and text/event-stream).
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=resp_headers,
            background=BackgroundTask(upstream.aclose),
        )

    return app


app = create_app()  # `uvicorn gateway:app`

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.getenv("GATEWAY_PORT", "8080")))
