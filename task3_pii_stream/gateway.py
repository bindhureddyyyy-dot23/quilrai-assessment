"""Task 3 - LLM Gateway with a real-time PII-redaction guardrail on the response stream.

    client --POST /v1/chat/completions--> gateway --> LLM provider (OpenAI-compatible SSE)
           <------ redacted SSE stream ----------  <--

Design
------
* The upstream SSE stream is consumed line-by-line (`httpx.aiter_lines`) and every event is
  re-emitted as soon as it is processed - no response-level buffering.
* Each choice index gets its own StreamingRedactor; only the trailing partial token that could
  still turn into PII is held (bounded by MAX_HOLDBACK chars), so TTFT is ~ upstream TTFT plus
  at most one word, and memory per stream is O(1) w.r.t. response length.
* When a choice finishes (`finish_reason`) or the stream ends (`[DONE]`), the held tail is
  flushed through the redactor before the terminal event, so nothing is lost or leaked.
* All non-content fields (id, model, role, finish_reason, usage, ...) pass through unchanged,
  so any OpenAI-compatible client keeps working.
* Non-streaming requests are redacted in one pass.
* Upstream failures return a sanitised error; if a failure happens mid-stream, a final SSE
  `error` event is sent instead of silently truncating the stream.

Config (env): UPSTREAM_BASE_URL (default: bundled mock), UPSTREAM_API_KEY, UPSTREAM_TIMEOUT.
Point UPSTREAM_BASE_URL at https://api.openai.com/v1 to run against a real provider.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from redactor import RedactionStats, StreamingRedactor, redact

log = logging.getLogger("pii_gateway")


@dataclass
class Settings:
    upstream_base_url: str = field(default_factory=lambda: os.getenv("UPSTREAM_BASE_URL", "http://127.0.0.1:9200/v1"))
    upstream_api_key: str = field(default_factory=lambda: os.getenv("UPSTREAM_API_KEY", "mock-key"))
    timeout_seconds: float = field(default_factory=lambda: float(os.getenv("UPSTREAM_TIMEOUT", "60")))


def gateway_error(status: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"type": "gateway_error", "code": code, "message": message, "request_id": request_id}},
        status_code=status,
        headers={"x-request-id": request_id},
    )


def _sse(data: str) -> bytes:
    return f"data: {data}\n\n".encode()


class StreamRedactionSession:
    """Holds per-choice redactor state for a single streamed response."""

    def __init__(self) -> None:
        self.redactors: dict[int, StreamingRedactor] = {}
        self.stats = RedactionStats()
        self.last_chunk_meta: dict[str, Any] = {}

    def _redactor(self, index: int) -> StreamingRedactor:
        if index not in self.redactors:
            self.redactors[index] = StreamingRedactor(stats=self.stats)
        return self.redactors[index]

    def process_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Rewrite delta.content in place. Returns None if the event carries nothing to send."""
        self.last_chunk_meta = {k: event[k] for k in ("id", "object", "created", "model") if k in event}
        choices = event.get("choices") or []
        keep_event = not choices  # e.g. a usage-only chunk
        for choice in choices:
            idx = choice.get("index", 0)
            delta = choice.get("delta") or {}
            red = self._redactor(idx)
            content = delta.get("content")
            out = red.feed(content) if isinstance(content, str) else ""
            if choice.get("finish_reason") is not None:
                out += red.flush()
            if isinstance(content, str) or out:
                delta["content"] = out
                choice["delta"] = delta
            # Drop chunks whose only payload was content we are still holding back.
            other_fields = {k for k, v in delta.items() if k != "content" and v is not None}
            if out or other_fields or choice.get("finish_reason") is not None:
                keep_event = True
        return event if keep_event else None

    def flush_all(self) -> list[dict[str, Any]]:
        """Tails for choices that never sent finish_reason (truncated upstream)."""
        events = []
        for idx, red in self.redactors.items():
            tail = red.flush()
            if tail:
                events.append(
                    {**self.last_chunk_meta, "choices": [{"index": idx, "delta": {"content": tail}, "finish_reason": None}]}
                )
        return events


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or Settings()
    client = httpx.AsyncClient(
        transport=transport,
        base_url=settings.upstream_base_url,
        timeout=httpx.Timeout(settings.timeout_seconds, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="LLM Gateway - PII streaming guardrail", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return gateway_error(400, "invalid_json", "Request body must be valid JSON", request_id)
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return gateway_error(400, "invalid_request", "'messages' must be a list", request_id)

        headers = {"authorization": f"Bearer {settings.upstream_api_key}", "x-request-id": request_id}
        stream = bool(body.get("stream"))

        if not stream:
            try:
                resp = await client.post("/chat/completions", json=body, headers=headers)
            except httpx.HTTPError as exc:
                log.error("upstream error rid=%s err=%r", request_id, exc)
                return gateway_error(502, "upstream_unavailable", "Upstream model provider unavailable", request_id)
            if resp.status_code != 200:
                return _upstream_status_error(resp.status_code, request_id)
            data = resp.json()
            stats = RedactionStats()
            for choice in data.get("choices", []):
                msg = choice.get("message") or {}
                if isinstance(msg.get("content"), str):
                    msg["content"] = redact(msg["content"], stats)
            return JSONResponse(data, headers={"x-request-id": request_id, "x-pii-redactions": str(stats.total)})

        # --- streaming path -------------------------------------------------
        upstream_req = client.build_request("POST", "/chat/completions", json=body, headers=headers)
        try:
            upstream = await client.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            log.error("upstream connect error rid=%s err=%r", request_id, exc)
            return gateway_error(502, "upstream_unavailable", "Upstream model provider unavailable", request_id)
        if upstream.status_code != 200:
            await upstream.aclose()
            return _upstream_status_error(upstream.status_code, request_id)

        return StreamingResponse(
            _redacted_stream(upstream, request_id),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no", "x-request-id": request_id},
        )

    return app


def _upstream_status_error(status: int, request_id: str) -> JSONResponse:
    if status == 429:
        return gateway_error(429, "rate_limited", "Upstream rate limit reached, retry later", request_id)
    if 400 <= status < 500:
        return gateway_error(status, "upstream_rejected", "Request rejected by model provider", request_id)
    return gateway_error(502, "upstream_error", "Upstream model provider error", request_id)


async def _redacted_stream(upstream: httpx.Response, request_id: str) -> AsyncIterator[bytes]:
    session = StreamRedactionSession()
    started = time.perf_counter()
    first_byte_at: float | None = None
    try:
        async for line in upstream.aiter_lines():
            # OpenAI-style SSE: one `data:` line per event, blank line separators, `: comment` keep-alives.
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                for ev in session.flush_all():
                    yield _sse(json.dumps(ev))
                yield _sse("[DONE]")
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                log.warning("dropping malformed upstream SSE line rid=%s", request_id)
                continue
            out = session.process_event(event)
            if out is not None:
                if first_byte_at is None:
                    first_byte_at = time.perf_counter()
                yield _sse(json.dumps(out, separators=(",", ":")))
        else:
            # upstream closed without [DONE]; still flush the held tail
            for ev in session.flush_all():
                yield _sse(json.dumps(ev))
    except httpx.HTTPError as exc:
        log.error("upstream stream broke rid=%s err=%r", request_id, exc)
        for ev in session.flush_all():
            yield _sse(json.dumps(ev))
        err = {"error": {"type": "gateway_error", "code": "upstream_stream_interrupted",
                         "message": "Upstream stream interrupted", "request_id": request_id}}
        yield f"event: error\ndata: {json.dumps(err)}\n\n".encode()
    finally:
        await upstream.aclose()
        peak = max((r.peak_buffer for r in session.redactors.values()), default=0)
        log.info(
            "stream done rid=%s redactions=%d (email=%d ssn=%d card=%d) gateway_ttfb_ms=%s peak_buffer_chars=%d",
            request_id, session.stats.total, session.stats.email, session.stats.ssn, session.stats.card,
            f"{(first_byte_at - started) * 1000:.1f}" if first_byte_at else "n/a", peak,
        )


app = create_app()  # `uvicorn gateway:app`

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("GATEWAY_PORT", "8081")))
