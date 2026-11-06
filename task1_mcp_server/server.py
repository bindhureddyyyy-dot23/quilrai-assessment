"""Task 1 - Customer Support MCP server over stdio.

Tools
-----
* get_customer_record(customer_id)              customer_id must match ^CUST-[0-9]{5}$
* trigger_refund(customer_id, amount, reason)   amount > 0, reason >= 10 chars

Error mapping (JSON-RPC 2.0 / MCP spec)
---------------------------------------
* Unknown tool name                 -> JSON-RPC error -32602 (Invalid params)
* Arguments fail Pydantic schema    -> JSON-RPC error -32602 with field-level details in `data`
* Unexpected server fault           -> JSON-RPC error -32603 (Internal error), no stack trace leaked
* Valid input, business rule fails  -> CallToolResult(isError=True)   (e.g. customer not found)
  The MCP spec separates *protocol* errors (bad request) from *tool execution* errors
  (the tool ran but could not complete) so the LLM can see and react to the latter.

STDIO isolation
---------------
stdout carries JSON-RPC frames only. At import time we keep a private handle to the
real stdout for the transport and re-point `sys.stdout` at stderr, so any stray
`print()` - ours or from a third-party library - lands on stderr instead of
corrupting the protocol stream. All logging is configured to stderr.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from decimal import Decimal
from io import TextIOWrapper
from typing import Any

# ---------------------------------------------------------------------------
# STDIO isolation - must run before anything else can write to stdout.
# ---------------------------------------------------------------------------
_PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr

logging.basicConfig(
    stream=sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [customer-mcp] %(message)s",
)
log = logging.getLogger("customer_mcp")

import anyio  # noqa: E402
import mcp.types as types  # noqa: E402
from mcp.server.lowlevel import NotificationOptions, Server  # noqa: E402
from mcp.server.models import InitializationOptions  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.shared.exceptions import McpError  # noqa: E402
from pydantic import (  # noqa: E402
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

SERVER_NAME = "customer-support-mcp"
SERVER_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Input schemas (single source of truth: the JSON Schema advertised in
# tools/list is generated from these models, so docs and enforcement never drift)
# ---------------------------------------------------------------------------
# [0-9] not \d: \d also matches non-ASCII digits such as Arabic-Indic "١٢٣٤٥".
CUSTOMER_ID_PATTERN = r"^CUST-[0-9]{5}$"
MAX_REFUND = 10_000.00


class _StrictArgs(BaseModel):
    # extra="forbid": unknown keys are rejected, not silently ignored.
    # strict=True: no type coercion ("12.5" is NOT accepted as a float, 123 is NOT a str).
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=False)


class GetCustomerRecordArgs(_StrictArgs):
    customer_id: str = Field(
        ...,
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer identifier formatted as CUST-XXXXX (5 digits), e.g. CUST-10001",
        examples=["CUST-10001"],
    )


class TriggerRefundArgs(_StrictArgs):
    customer_id: str = Field(..., pattern=CUSTOMER_ID_PATTERN, description="Customer identifier, CUST-XXXXX")
    amount: float = Field(
        ...,
        gt=0,
        le=MAX_REFUND,
        allow_inf_nan=False,
        description=f"Refund amount, positive, at most {MAX_REFUND:.2f}, max 2 decimal places",
    )
    reason: str = Field(..., min_length=10, max_length=500, description="Why the refund is issued (>= 10 chars)")

    @field_validator("amount", mode="before")
    @classmethod
    def _reject_bool_and_allow_ints(cls, v: Any) -> Any:
        # JSON has one number type; accept 25 as 25.0, but bool is a subclass of int in Python.
        if isinstance(v, bool):
            raise ValueError("amount must be a number, not a boolean")
        if isinstance(v, int):
            return float(v)
        return v

    @field_validator("amount")
    @classmethod
    def _max_two_decimals(cls, v: float) -> float:
        if Decimal(str(v)).as_tuple().exponent < -2:
            raise ValueError("amount must have at most 2 decimal places")
        return v

    @field_validator("reason")
    @classmethod
    def _reason_not_blank(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("reason must contain at least 10 non-whitespace characters")
        return v


# ---------------------------------------------------------------------------
# Mock data layer
# ---------------------------------------------------------------------------
CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-10001": {"name": "Ada Lovelace", "tier": "gold", "balance": 1250.00, "status": "active"},
    "CUST-10002": {"name": "Alan Turing", "tier": "silver", "balance": 89.50, "status": "active"},
    "CUST-10003": {"name": "Grace Hopper", "tier": "bronze", "balance": 0.00, "status": "suspended"},
}
_refund_seq = 0


def get_customer_record(args: GetCustomerRecordArgs) -> types.CallToolResult:
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        return _tool_error(f"Customer {args.customer_id} not found")
    payload = {"customer_id": args.customer_id, **record}
    return _tool_ok(payload)


def trigger_refund(args: TriggerRefundArgs) -> types.CallToolResult:
    global _refund_seq
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        return _tool_error(f"Customer {args.customer_id} not found")
    if record["status"] != "active":
        return _tool_error(f"Refunds are not allowed for customers with status '{record['status']}'")
    _refund_seq += 1
    refund_id = f"RF-{_refund_seq:06d}"
    log.info("refund issued id=%s customer=%s amount=%.2f", refund_id, args.customer_id, args.amount)
    return _tool_ok(
        {
            "refund_id": refund_id,
            "customer_id": args.customer_id,
            "amount": round(args.amount, 2),
            "reason": args.reason,
            "status": "queued",
        }
    )


def _tool_ok(payload: dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=False,
    )


def _tool_error(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=message)], isError=True)


# name -> (args model, handler, description)
TOOLS = {
    "get_customer_record": (
        GetCustomerRecordArgs,
        get_customer_record,
        "Fetch a customer's profile, tier, balance and account status.",
    ),
    "trigger_refund": (
        TriggerRefundArgs,
        trigger_refund,
        "Queue a refund for an active customer. Amount must be positive; reason >= 10 characters.",
    ),
}


def _format_validation_errors(exc: ValidationError) -> list[dict[str, Any]]:
    """Client-safe error details: field path + message. The raw input value is never echoed."""
    return [
        {"field": ".".join(str(p) for p in err["loc"]) or "<root>", "type": err["type"], "message": err["msg"]}
        for err in exc.errors(include_url=False, include_input=False)
    ]


# ---------------------------------------------------------------------------
# MCP wiring
# ---------------------------------------------------------------------------
server: Server = Server(SERVER_NAME, version=SERVER_VERSION)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(name=name, description=desc, inputSchema=model.model_json_schema())
        for name, (model, _handler, desc) in TOOLS.items()
    ]


async def handle_call_tool(req: types.CallToolRequest) -> types.ServerResult:
    """Custom tools/call handler.

    The SDK's @server.call_tool() decorator converts *every* failure (including schema
    violations) into a successful response with isError=True. The assessment asks for
    standard JSON-RPC error codes on invalid input, so we register our own handler and
    raise McpError, which the session layer serialises as a JSON-RPC `error` object.
    """
    name = req.params.name
    arguments = req.params.arguments or {}

    entry = TOOLS.get(name)
    if entry is None:
        raise McpError(
            types.ErrorData(
                code=types.INVALID_PARAMS,
                message=f"Unknown tool: {name}",
                data={"available_tools": sorted(TOOLS)},
            )
        )
    model, handler, _ = entry

    try:
        args = model.model_validate(arguments)
    except ValidationError as exc:
        details = _format_validation_errors(exc)
        log.warning("invalid params for %s: %s", name, details)
        raise McpError(
            types.ErrorData(
                code=types.INVALID_PARAMS,
                message=f"Invalid params for tool '{name}'",
                data={"errors": details},
            )
        ) from None

    try:
        result = handler(args)
    except Exception:  # noqa: BLE001 - last-resort guard, details go to stderr only
        log.exception("tool %s crashed", name)
        raise McpError(types.ErrorData(code=types.INTERNAL_ERROR, message="Internal error")) from None
    return types.ServerResult(result)


server.request_handlers[types.CallToolRequest] = handle_call_tool


# ---------------------------------------------------------------------------
# Transport hardening: JSON-RPC parse / invalid-request errors
# ---------------------------------------------------------------------------
class _SerializedWriter:
    """stdout wrapper that writes each frame + flush under one lock.

    Both the SDK's writer task and our parse-error replies go through it, so two frames
    can never interleave on the wire.
    """

    def __init__(self, inner: anyio.AsyncFile[str]):
        self._inner = inner
        self._lock = anyio.Lock()
        self._pending = ""

    async def write(self, data: str) -> None:
        # The SDK calls write(frame + "\n") then flush(); buffer until flush to keep it atomic.
        self._pending += data

    async def flush(self) -> None:
        async with self._lock:
            data, self._pending = self._pending, ""
            if data:
                await self._inner.write(data)
            await self._inner.flush()

    async def send_frame(self, frame: dict[str, Any]) -> None:
        async with self._lock:
            await self._inner.write(json.dumps(frame, separators=(",", ":")) + "\n")
            await self._inner.flush()


class _ValidatingStdin:
    """stdin wrapper that answers malformed input with spec-compliant JSON-RPC errors.

    Out of the box the SDK reacts to a non-JSON line with a log notification, not with the
    -32700 Parse error that JSON-RPC 2.0 section 5.1 requires. Here, unparseable lines get
    -32700 (id: null) and well-formed JSON that is not a JSON-RPC message gets -32600.
    Valid lines pass through untouched.
    """

    def __init__(self, inner: anyio.AsyncFile[str], out: _SerializedWriter):
        self._inner = inner
        self._out = out

    def __aiter__(self):
        return self._lines()

    async def _lines(self):
        async for line in self._inner:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                log.warning("rejecting non-JSON input line (-32700)")
                await self._out.send_frame(_rpc_error(None, types.PARSE_ERROR, "Parse error"))
                continue
            try:
                types.JSONRPCMessage.model_validate(obj)
            except ValidationError:
                req_id = obj.get("id") if isinstance(obj, dict) and isinstance(obj.get("id"), (str, int)) else None
                log.warning("rejecting invalid JSON-RPC message (-32600)")
                await self._out.send_frame(_rpc_error(req_id, types.INVALID_REQUEST, "Invalid Request"))
                continue
            yield line


def _rpc_error(req_id: str | int | None, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


async def main() -> None:
    raw_out = anyio.wrap_file(TextIOWrapper(_PROTOCOL_STDOUT.buffer, encoding="utf-8"))
    protocol_out = _SerializedWriter(raw_out)
    raw_in = anyio.wrap_file(TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace"))
    protocol_in = _ValidatingStdin(raw_in, protocol_out)
    log.info("starting %s v%s on stdio", SERVER_NAME, SERVER_VERSION)
    async with stdio_server(stdin=protocol_in, stdout=protocol_out) as (read_stream, write_stream):  # type: ignore[arg-type]
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=SERVER_NAME,
                server_version=SERVER_VERSION,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(), experimental_capabilities={}
                ),
            ),
        )


if __name__ == "__main__":
    anyio.run(main)
