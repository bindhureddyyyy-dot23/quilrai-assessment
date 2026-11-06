"""Task 1 tests.

Two layers:
1. Raw wire tests - spawn the server as a subprocess, speak newline-delimited JSON-RPC
   over its stdin, and assert that EVERY line on stdout is a valid JSON-RPC 2.0 frame.
2. SDK client tests - drive the server through the official `mcp` ClientSession.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

HERE = Path(__file__).parent
SERVER = str(HERE / "server.py")

INIT = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "1"}},
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


class RawStdioClient:
    """Minimal line-oriented JSON-RPC client for wire-level assertions."""

    def __init__(self, argv: list[str]):
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=HERE
        )
        self.stdout_lines: list[str] = []

    def send(self, msg) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def recv(self) -> dict:
        assert self.proc.stdout
        line = self.proc.stdout.readline()
        assert line, "server closed stdout unexpectedly"
        self.stdout_lines.append(line)
        return json.loads(line)

    def request(self, msg) -> dict:
        self.send(msg)
        return self.recv()

    def close(self) -> str:
        assert self.proc.stdin
        self.proc.stdin.close()
        try:
            _, err = self.proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            _, err = self.proc.communicate()
        return err


def call(id_: int, name: str, arguments: dict | None) -> dict:
    params: dict = {"name": name}
    if arguments is not None:
        params["arguments"] = arguments
    return {"jsonrpc": "2.0", "id": id_, "method": "tools/call", "params": params}


@pytest.fixture
def raw():
    client = RawStdioClient([sys.executable, SERVER])
    resp = client.request(INIT)
    assert resp["result"]["serverInfo"]["name"] == "customer-support-mcp"
    client.send(INITIALIZED)
    yield client
    client.close()


# ---------------------------------------------------------------------------
# STDIO isolation
# ---------------------------------------------------------------------------
def test_every_stdout_line_is_jsonrpc(raw: RawStdioClient):
    raw.request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    raw.request(call(2, "get_customer_record", {"customer_id": "CUST-10001"}))
    raw.request(call(3, "get_customer_record", {"customer_id": "bad"}))
    raw.request(call(4, "trigger_refund", {"customer_id": "CUST-10001", "amount": 5, "reason": "late delivery x"}))
    err = raw.close()

    for line in raw.stdout_lines:
        frame = json.loads(line)  # raises if anything non-JSON leaked
        assert frame["jsonrpc"] == "2.0"
        assert ("result" in frame) ^ ("error" in frame)
    # logs went to stderr instead
    assert "customer-mcp" in err


def test_stray_print_is_redirected_to_stderr():
    """Even a careless print() after import must not reach the protocol stream."""
    bootstrap = (
        "import sys; sys.path.insert(0, '.'); import anyio, server; "
        "print('STRAY DEBUG OUTPUT'); anyio.run(server.main)"
    )
    client = RawStdioClient([sys.executable, "-c", bootstrap])
    resp = client.request(INIT)
    assert resp["id"] == 0 and "result" in resp
    err = client.close()
    assert "STRAY DEBUG OUTPUT" in err
    assert all("STRAY" not in line for line in client.stdout_lines)


# ---------------------------------------------------------------------------
# Protocol compliance and validation
# ---------------------------------------------------------------------------
def test_tools_list_exposes_strict_schemas(raw: RawStdioClient):
    tools = {t["name"]: t for t in raw.request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]}
    assert set(tools) == {"get_customer_record", "trigger_refund"}
    refund = tools["trigger_refund"]["inputSchema"]
    assert refund["additionalProperties"] is False
    assert refund["properties"]["reason"]["minLength"] == 10
    assert refund["properties"]["amount"]["exclusiveMinimum"] == 0
    assert tools["get_customer_record"]["inputSchema"]["properties"]["customer_id"]["pattern"] == r"^CUST-[0-9]{5}$"


def test_valid_lookup(raw: RawStdioClient):
    resp = raw.request(call(1, "get_customer_record", {"customer_id": "CUST-10001"}))
    assert resp["result"]["isError"] is False
    assert resp["result"]["structuredContent"]["name"] == "Ada Lovelace"


def test_unknown_customer_is_tool_error_not_protocol_error(raw: RawStdioClient):
    resp = raw.request(call(1, "get_customer_record", {"customer_id": "CUST-99999"}))
    assert "error" not in resp
    assert resp["result"]["isError"] is True


@pytest.mark.parametrize(
    "customer_id",
    ["CUST-1234", "CUST-123456", "cust-12345", "CUST_12345", "CUST-1234A", " CUST-12345", "CUST-12345\n", "", "CUST-١٢٣٤٥"],
)
def test_bad_customer_id_formats(raw: RawStdioClient, customer_id: str):
    resp = raw.request(call(1, "get_customer_record", {"customer_id": customer_id}))
    assert resp["error"]["code"] == -32602
    assert resp["error"]["data"]["errors"][0]["field"] == "customer_id"


@pytest.mark.parametrize(
    "arguments, bad_field",
    [
        ({"customer_id": 12345}, "customer_id"),  # wrong type, no coercion
        ({}, "customer_id"),  # missing
        ({"customer_id": "CUST-10001", "extra": 1}, "extra"),  # unknown key
    ],
)
def test_get_customer_record_malformed(raw: RawStdioClient, arguments: dict, bad_field: str):
    resp = raw.request(call(1, "get_customer_record", arguments))
    assert resp["error"]["code"] == -32602
    assert bad_field in {e["field"] for e in resp["error"]["data"]["errors"]}


GOOD_REFUND = {"customer_id": "CUST-10001", "amount": 25.5, "reason": "Item arrived damaged"}


@pytest.mark.parametrize(
    "override, bad_field",
    [
        ({"amount": 0}, "amount"),
        ({"amount": -10.0}, "amount"),
        ({"amount": "25.50"}, "amount"),  # string, strict mode refuses coercion
        ({"amount": True}, "amount"),  # bool is not a number here
        ({"amount": 10.001}, "amount"),  # more than 2 decimals
        ({"amount": 1e9}, "amount"),  # above max
        ({"reason": "too short"}, "reason"),  # 9 chars
        ({"reason": "          x"}, "reason"),  # whitespace padding does not count
        ({"reason": None}, "reason"),
        ({"customer_id": "CUST-1"}, "customer_id"),
    ],
)
def test_trigger_refund_malformed(raw: RawStdioClient, override: dict, bad_field: str):
    resp = raw.request(call(1, "trigger_refund", {**GOOD_REFUND, **override}))
    assert resp["error"]["code"] == -32602, resp
    assert bad_field in {e["field"] for e in resp["error"]["data"]["errors"]}


def test_validation_errors_do_not_echo_input(raw: RawStdioClient):
    secret = "CUST-SECRET-<script>"
    resp = raw.request(call(1, "get_customer_record", {"customer_id": secret}))
    assert secret not in json.dumps(resp)


def test_valid_refund_and_integer_amount(raw: RawStdioClient):
    resp = raw.request(call(1, "trigger_refund", {**GOOD_REFUND, "amount": 25}))
    assert resp["result"]["isError"] is False
    assert resp["result"]["structuredContent"]["amount"] == 25.0


def test_refund_for_suspended_customer_is_tool_error(raw: RawStdioClient):
    resp = raw.request(call(1, "trigger_refund", {**GOOD_REFUND, "customer_id": "CUST-10003"}))
    assert resp["result"]["isError"] is True


def test_unknown_tool(raw: RawStdioClient):
    resp = raw.request(call(1, "drop_database", {}))
    assert resp["error"]["code"] == -32602


def test_missing_arguments_object(raw: RawStdioClient):
    resp = raw.request(call(1, "trigger_refund", None))
    assert resp["error"]["code"] == -32602


def test_unknown_method(raw: RawStdioClient):
    resp = raw.request({"jsonrpc": "2.0", "id": 9, "method": "tools/explode"})
    assert resp["error"]["code"] in (-32601, -32602)


def test_non_json_line_gets_parse_error_and_server_survives(raw: RawStdioClient):
    assert raw.proc.stdin
    raw.proc.stdin.write("this is not json\n")
    raw.proc.stdin.flush()
    err = raw.recv()
    assert err == {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
    resp = raw.request(call(7, "get_customer_record", {"customer_id": "CUST-10002"}))
    assert resp["id"] == 7 and resp["result"]["isError"] is False


@pytest.mark.parametrize(
    "payload, expected_id",
    [
        ({"id": 11, "method": "tools/list"}, 11),  # missing "jsonrpc"
        ({"jsonrpc": "1.0", "id": 12, "method": "tools/list"}, 12),  # wrong version
        ([1, 2, 3], None),  # not an object
        ("just a string", None),
    ],
)
def test_invalid_jsonrpc_envelope(raw: RawStdioClient, payload, expected_id):
    resp = raw.request(payload)
    assert resp["error"]["code"] == -32600
    assert resp["id"] == expected_id


# ---------------------------------------------------------------------------
# Official SDK client end-to-end
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_sdk_client_roundtrip():
    params = StdioServerParameters(command=sys.executable, args=[SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == {"get_customer_record", "trigger_refund"}

            ok = await session.call_tool("get_customer_record", {"customer_id": "CUST-10002"})
            assert not ok.isError

            with pytest.raises(McpError) as exc:
                await session.call_tool("trigger_refund", {"customer_id": "CUST-10001", "amount": -1, "reason": "x"})
            assert exc.value.error.code == -32602


@pytest.fixture
def anyio_backend():
    return "asyncio"
