# QuilrAI – AI Field Developer / Solutions Engineer Assessment

**Candidate:** Bindhu Reddy Minukuri · **Language:** Python 3.10+ · **Tests:** 169 passing

| # | Task | Folder | Core idea |
|---|------|--------|-----------|
| 1 | MCP server, strict validation, stdio | [`task1_mcp_server`](task1_mcp_server) | Pydantic strict models → JSON-RPC `-32602`; stdout physically isolated |
| 2 | MCP security gateway (tool filtering + auth) | [`task2_mcp_gateway`](task2_mcp_gateway) | JWT role → `admin_*` tools gated, `-32001` returned without touching downstream |
| 3 | LLM gateway streaming PII guardrail | [`task3_pii_stream`](task3_pii_stream) | Bounded hold-back buffer: redacts PII split across SSE chunks with ~1-word latency |
| 4 | Rate limiter + model fallback router | [`task4_router`](task4_router) | SQLite sliding-window log (atomic `BEGIN IMMEDIATE`), 3000 ms deadline failover |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
./run_tests.sh                     # or: bash run_tests.sh
```

Each task is a self-contained service with its own mocks, so nothing external (API keys, Docker) is needed.

---

## Task 1 – Custom MCP server (`task1_mcp_server/server.py`)

Tools: `get_customer_record(customer_id)` and `trigger_refund(customer_id, amount, reason)`.

**Validation** – Pydantic v2 models with `strict=True` and `extra="forbid"`; the JSON Schema shown in `tools/list` is generated from the same models, so the advertised contract and the enforced one can't drift.

* `customer_id` must match `^CUST-[0-9]{5}$`. `[0-9]` instead of `\d` is deliberate: the test suite caught that `\d` accepts Arabic-Indic digits (`CUST-١٢٣٤٥`).
* `amount` > 0, ≤ 10,000, max 2 decimals, no NaN/Inf, no `"25.50"` strings, no booleans (Python `True` is an `int`).
* `reason` ≥ 10 *non-whitespace-padded* characters, ≤ 500.
* Error details list the failing field and rule but never echo the raw input value back.

**Error mapping**

| Situation | Response |
|---|---|
| Schema violation / unknown tool | JSON-RPC error `-32602 Invalid params` (+ field-level `data`) |
| Non-JSON line on stdin | `-32700 Parse error`, `id: null` |
| Valid JSON, not a JSON-RPC message | `-32600 Invalid Request` |
| Unexpected crash | `-32603 Internal error` (trace to stderr only) |
| Valid input, business rule fails (unknown / suspended customer) | `result.isError = true` – a *tool execution* error, per the MCP spec, so the LLM can read and react to it |

The SDK's `@server.call_tool()` decorator converts *every* failure into `isError: true`, so I register a custom `tools/call` handler that raises `McpError`. The SDK also answers a garbage line with a log notification instead of `-32700`, so a thin stdin wrapper adds spec-compliant parse errors.

**STDIO isolation** – at import, the real stdout handle is kept privately for the transport and `sys.stdout` is pointed at stderr. A stray `print()` anywhere (ours or a dependency's) lands on stderr. All frames are written through one lock so they can't interleave. Tests assert every stdout line parses as JSON-RPC, including a test that deliberately calls `print()`.

```bash
cd task1_mcp_server
python server.py                                # speaks MCP on stdio
npx @modelcontextprotocol/inspector python server.py   # optional: interactive inspector
```

## Task 2 – MCP security gateway (`task2_mcp_gateway/gateway.py`)

```
agent --Bearer JWT--> gateway --(no client token)--> downstream MCP server
```

1. **Authenticate** – HS256 JWT with a `role` claim (`admin`/`viewer`). Algorithm pinned (rejects `alg: none`), constant-time signature check, `exp`/`nbf` enforced, unknown roles rejected (not silently downgraded). Failure → HTTP 401 + `WWW-Authenticate`.
2. **Parse** – bad JSON `-32700`; wrong shape `-32600`. **Batches are refused**: MCP 2025-06-18 removed batching, and allowing them would let an `admin_` call hide inside an array.
3. **Authorize** – `tools/call` whose `params.name` starts with `admin_` requires `admin`; otherwise the gateway itself returns `{"code": -32001, "message": "Unauthorized Tool Call"}` and the downstream server is **never contacted** (tests assert zero downstream calls). The prefix check is normalized (`strip().casefold()`), so `ADMIN_reset_key` or `" admin_reset_key"` can't bypass it.
4. **Forward** – `tools/list` and everything else is proxied with the original bytes; the response is *streamed* back untouched, so JSON and SSE (Streamable HTTP) both work. Only an allow-list of MCP headers (`Mcp-Session-Id`, `Mcp-Protocol-Version`, …) is forwarded.

Security choices: the client's token is **not** passed downstream (token passthrough is prohibited by the MCP auth spec); identity is sent as `X-Gateway-Subject/Role` instead. Downstream outages return a generic `-32002` with no hostnames. Every allow/deny decision is audit-logged.

```bash
cd task2_mcp_gateway
python mock_mcp_server.py &        # downstream on :9100
python gateway.py &                # gateway on :8080
python -c "from auth import mint_token; print(mint_token('dev-only-change-me', role='viewer'))"
# then:
curl -s localhost:8080/mcp -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{}}}'
# -> {"jsonrpc":"2.0","id":1,"error":{"code":-32001,"message":"Unauthorized Tool Call",...}}
```

## Task 3 – Streaming PII guardrail (`task3_pii_stream/`)

**The problem:** providers stream text in arbitrary fragments, so `jane.d` | `oe@exam` | `ple.com` defeats a per-chunk regex, while buffering the whole response destroys TTFT.

**The approach (`redactor.py`)** – like a censor reading a ticker tape: everything up to the last *safe cut point* is final and released immediately; only the trailing fragment that could still grow into PII is held back. Because every pattern is built from "token" characters (letters, digits, `._%+-@`) plus single spaces between digit groups, the held tail is just the last partial word (extended backwards over `4111 1111 …`-style digit groups). The scan is backwards over the tail only, so released text is never rescanned.

* **Latency:** ~one word behind upstream. Prose words are released as soon as the following space arrives.
* **Memory:** O(256 chars) per stream regardless of response length (an email can't legally exceed 254 chars). A pathological unbroken run is force-flushed at the cap.
* **Correctness invariant**, property-tested over thousands of random chunkings: `"".join(feed(c) for c in chunks) + flush() == redact(full_text)`.
* **Patterns:** email; SSN (excludes never-issued `000`/`666`/`9xx` areas, `00` group, `0000` serial); cards in real layouts (contiguous, 4-4-4-4, 4-4-4-4-3, Amex 4-6-5, Diners 4-6-4) with **Luhn check** so order numbers aren't redacted. Tests caught that a naive "digits with optional separators" card regex greedily glues a card to an adjacent SSN, fails Luhn, and leaks *both* – the layouts above plus a `(?!-?\d)` guard prevent it.

**The proxy (`gateway.py`)** is OpenAI-compatible: it consumes upstream SSE line by line, rewrites `delta.content` per choice index (separate redactor state for `n > 1`), passes every other field through, flushes the held tail before `finish_reason`/`[DONE]`, and emits an SSE `error` event instead of silently truncating if upstream dies mid-stream. The end-to-end test runs real sockets and asserts the first content arrives in under 25% of total stream time.

```bash
cd task3_pii_stream
python mock_llm.py &               # fake provider on :9200, streams 1-6 char chunks
python gateway.py &                # gateway on :8081
curl -N localhost:8081/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"m","stream":true,"messages":[{"role":"user","content":"canned"}]}'
# Real provider: UPSTREAM_BASE_URL=https://api.openai.com/v1 UPSTREAM_API_KEY=sk-... python gateway.py
```

## Task 4 – Rate limiter + fallback router (`task4_router/`)

**Rate limiter (`rate_limiter.py`)** – *sliding-window log* in an **on-disk SQLite** file: one row per admitted request `(tenant, ts, tokens)`; admit if `sum(tokens in last 60 s) + requested ≤ limit` (default 50,000/min per tenant API key). No fixed-window boundary burst; exact `Retry-After` (time until enough old rows expire).

* **Reserve → reconcile → release:** the estimate (prompt chars/4 + `max_tokens`) is booked up-front so concurrent in-flight requests can't jointly overshoot; it's corrected to the provider's reported `usage` afterwards; if all providers fail the reservation is refunded.
* **Concurrency:** `BEGIN IMMEDIATE` takes the write lock *before* reading the sum, making check-then-insert atomic across coroutines, threads **and separate worker processes** sharing the DB (tested: 120 concurrent 1k-token requests on a 50k limit → exactly 50 admitted). WAL mode; blocking calls run in a worker thread.
* **Eviction:** expired rows deleted inline on each tenant's reserve, plus a background sweeper for idle tenants – the table never holds more than one window.
* API keys are stored as SHA-256 hashes, never in plain text.

**Router (`router.py`)** – primary → secondary. Fails over on **429**, **no response within 3000 ms**, network errors, 5xx, 401/403 (our credential problem) and malformed bodies. Other 4xx are *not* retried – the same request would fail again. The deadline wraps the whole call with `asyncio.wait_for`, because httpx timeouts are per-phase (a provider trickling a byte a second never trips them); cancellation also guarantees a late primary response can't race the fallback. Optional per-provider model override (e.g. cheaper backup model).

**Error sanitization (`app.py`)** – every failure path (gateway errors, validation, 404/405, unhandled exceptions) returns one shape:

```json
{"error": {"type": "rate_limit_error", "code": "rate_limit_exceeded", "message": "...", "request_id": "..."}}
```

Upstream bodies, stack traces and hostnames go to server logs keyed by `request_id`, never to the client; tests inject a provider that returns a Python traceback and assert none of it reaches the response. Responses carry `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens`, `retry-after`, `x-gateway-provider`, `x-gateway-fallback`.

```bash
cd task4_router
python mock_providers.py &         # primary :9301, secondary :9302
python app.py &                    # gateway :8082, DB at ./data/gateway.db
curl -s -X POST localhost:9301/__behavior -H "Content-Type: application/json" -d '{"mode":"slow"}'
curl -si localhost:8082/v1/chat/completions -H "Authorization: Bearer sk-tenant-a" \
  -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"hi"}]}'
# -> served by secondary after exactly 3.0 s, x-gateway-fallback: true
```
Modes for `/__behavior`: `ok`, `429`, `slow`, `500`, `400`.

---

## What I'd add for production

* **Task 2:** filter `tools/list` results per role (hide `admin_*` from viewers), JWKS/OIDC instead of a shared HS256 secret, per-tool rate limits.
* **Task 3:** redact `tool_calls[].function.arguments` deltas and the *inbound* prompt too; pluggable detectors (phone, IBAN, API keys, NER model) behind the same hold-back interface.
* **Task 4:** circuit breaker so a failing primary is skipped for a cool-down instead of costing 3 s per request; Redis/Postgres backend behind the same limiter interface for multi-host deployments; streaming support with mid-stream usage reconciliation.
