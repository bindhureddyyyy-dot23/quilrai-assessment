"""Task 2 tests - gateway and mock downstream wired together in-process via ASGI transports."""

from __future__ import annotations

import json
import time

import httpx
import pytest

from auth import AuthError, mint_token, verify_token
from gateway import Settings, create_app
from mock_mcp_server import create_mock_app

SECRET = "test-secret"
DOWNSTREAM = "http://downstream/mcp"


@pytest.fixture
def downstream():
    return create_mock_app()


@pytest.fixture
async def client(downstream):
    gw = create_app(Settings(downstream_url=DOWNSTREAM, jwt_secret=SECRET), httpx.ASGITransport(app=downstream))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gw), base_url="http://gw") as c:
        yield c


def bearer(role: str, **kw) -> dict[str, str]:
    return {"Authorization": f"Bearer {mint_token(SECRET, role=role, sub=f'{role}-user', **kw)}"}


def call(name, rid=1, arguments=None):
    return {"jsonrpc": "2.0", "id": rid, "method": "tools/call", "params": {"name": name, "arguments": arguments or {}}}


# --- forwarding -------------------------------------------------------------
async def test_tools_list_forwarded_transparently(client, downstream):
    r = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 7, "method": "tools/list"}, headers=bearer("viewer"))
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 7
    assert "admin_reset_key" in {t["name"] for t in body["result"]["tools"]}  # unchanged list
    assert r.headers["mcp-session-id"] == "sess-123"  # MCP headers passed back


async def test_viewer_can_call_normal_tool(client, downstream):
    r = await client.post("/mcp", json=call("get_weather", arguments={"city": "Austin"}), headers=bearer("viewer"))
    assert r.json()["result"]["content"][0]["text"] == "executed get_weather"
    assert downstream.state.calls == [{"name": "get_weather", "arguments": {"city": "Austin"}}]


async def test_admin_can_call_admin_tool(client, downstream):
    r = await client.post("/mcp", json=call("admin_reset_key", arguments={"key_id": "k1"}), headers=bearer("admin"))
    assert r.json()["result"]["isError"] is False
    assert downstream.state.calls[0]["name"] == "admin_reset_key"


async def test_sse_response_streamed_through(client):
    r = await client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
        headers={**bearer("viewer"), "Accept": "text/event-stream"},
    )
    assert r.headers["content-type"].startswith("text/event-stream")
    data_line = next(line for line in r.text.splitlines() if line.startswith("data: "))
    assert json.loads(data_line[6:])["id"] == 3


async def test_client_token_not_forwarded_downstream(client, downstream):
    await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers=bearer("viewer"))
    seen = downstream.state.headers_seen[-1]
    assert "authorization" not in seen
    assert seen["x-gateway-role"] == "viewer" and seen["x-gateway-subject"] == "viewer-user"


async def test_notification_forwarded(client):
    r = await client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=bearer("viewer"))
    assert r.status_code == 202


# --- authorisation ------------------------------------------------------------
@pytest.mark.parametrize("tool", ["admin_reset_key", "admin_delete_user", "ADMIN_reset_key", "  admin_reset_key", "Admin_x"])
async def test_viewer_blocked_from_admin_tools(client, downstream, tool):
    r = await client.post("/mcp", json=call(tool, rid="abc"), headers=bearer("viewer"))
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "jsonrpc": "2.0",
        "id": "abc",
        "error": {"code": -32001, "message": "Unauthorized Tool Call", "data": {"tool": tool, "required_role": "admin"}},
    }
    assert downstream.state.calls == []  # downstream never reached


async def test_tool_named_like_admin_but_not_prefixed_is_allowed(client, downstream):
    r = await client.post("/mcp", json=call("list_files"), headers=bearer("viewer"))
    assert "result" in r.json()


@pytest.mark.parametrize("params", [None, {}, {"name": 123}, {"name": ""}, "admin_reset_key"])
async def test_malformed_tools_call_params(client, downstream, params):
    msg = {"jsonrpc": "2.0", "id": 5, "method": "tools/call"}
    if params is not None:
        msg["params"] = params
    r = await client.post("/mcp", json=msg, headers=bearer("viewer"))
    assert r.json()["error"]["code"] == -32602
    assert downstream.state.calls == []


async def test_batch_cannot_smuggle_admin_call(client, downstream):
    batch = [call("get_weather", 1), call("admin_reset_key", 2)]
    r = await client.post("/mcp", json=batch, headers=bearer("viewer"))
    assert r.status_code == 400 and r.json()["error"]["code"] == -32600
    assert downstream.state.calls == []


# --- authentication ----------------------------------------------------------
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer not.a.jwt"},
        {"Authorization": f"Bearer {mint_token('wrong-secret', role='admin')}"},
        {"Authorization": f"Bearer {mint_token(SECRET, role='superuser')}"},
        {"Authorization": f"Bearer {mint_token(SECRET, role='admin', ttl_seconds=-3600)}"},
    ],
)
async def test_unauthenticated_requests_rejected(client, downstream, headers):
    r = await client.post("/mcp", json=call("get_weather", rid=42), headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Bearer")
    assert r.json()["error"]["code"] == -32000 and r.json()["id"] == 42
    assert downstream.state.calls == []


def test_alg_none_rejected():
    import base64

    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    forged = f"{b64({'alg': 'none'})}.{b64({'role': 'admin', 'exp': time.time() + 60})}."
    with pytest.raises(AuthError):
        verify_token(forged, SECRET)


def test_tampered_role_rejected():
    token = mint_token(SECRET, role="viewer")
    h, p, s = token.split(".")
    import base64

    payload = json.loads(base64.urlsafe_b64decode(p + "=="))
    payload["role"] = "admin"
    p2 = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    with pytest.raises(AuthError, match="signature"):
        verify_token(f"{h}.{p2}.{s}", SECRET)


# --- wire-format errors ------------------------------------------------------
async def test_parse_error(client):
    r = await client.post("/mcp", content=b"{not json", headers={**bearer("viewer"), "content-type": "application/json"})
    assert r.status_code == 400 and r.json() == {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}


@pytest.mark.parametrize("msg", [{"id": 1, "method": "tools/list"}, {"jsonrpc": "2.0", "id": 1}, "hello", 42])
async def test_invalid_request(client, msg):
    r = await client.post("/mcp", json=msg, headers=bearer("viewer"))
    assert r.status_code == 400 and r.json()["error"]["code"] == -32600


async def test_downstream_down_returns_sanitised_error():
    gw = create_app(Settings(downstream_url="http://127.0.0.1:1/mcp", jwt_secret=SECRET, timeout_seconds=1))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gw), base_url="http://gw") as c:
        r = await c.post("/mcp", json={"jsonrpc": "2.0", "id": 9, "method": "tools/list"}, headers=bearer("viewer"))
    assert r.status_code == 502
    body = r.json()
    assert body["id"] == 9 and body["error"]["code"] == -32002
    assert "127.0.0.1" not in r.text  # no internal topology leaked
