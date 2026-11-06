"""Task 4 - LLM Gateway: per-tenant token rate limiting + primary/secondary model failover.

POST /v1/chat/completions   (OpenAI-compatible, non-streaming)
    Authorization: Bearer <tenant API key>

Flow:  authenticate tenant -> validate body -> estimate tokens -> reserve in SQLite window
       -> route (primary, fail over on 429 / 3000 ms timeout / 5xx) -> reconcile actual usage
       -> respond.   On total failure the reservation is released (tenant is not charged).

Every error, from any layer, has the same shape and never contains upstream details:
    {"error": {"type": "...", "code": "...", "message": "...", "request_id": "..."}}

Env: GATEWAY_DB_PATH (default ./data/gateway.db), PRIMARY_URL, PRIMARY_API_KEY,
     PRIMARY_TIMEOUT_MS (default 3000), SECONDARY_URL, SECONDARY_API_KEY, SECONDARY_TIMEOUT_MS,
     SECONDARY_MODEL, DEFAULT_TPM (default 50000), DEMO_TENANT_KEYS="key:tenant[:tpm],..."
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from rate_limiter import SQLiteTokenRateLimiter
from router import GatewayError, ModelRouter, Provider

log = logging.getLogger("llm_gateway")

DEFAULT_MAX_TOKENS = 512


# --------------------------------------------------------------------------- config
@dataclass
class Settings:
    db_path: str = field(default_factory=lambda: os.getenv("GATEWAY_DB_PATH", str(Path(__file__).parent / "data" / "gateway.db")))
    window_seconds: float = 60.0
    default_tpm: int = field(default_factory=lambda: int(os.getenv("DEFAULT_TPM", "50000")))
    primary: Provider = field(
        default_factory=lambda: Provider(
            name="primary",
            base_url=os.getenv("PRIMARY_URL", "http://127.0.0.1:9301/v1"),
            api_key=os.getenv("PRIMARY_API_KEY", ""),
            timeout_seconds=int(os.getenv("PRIMARY_TIMEOUT_MS", "3000")) / 1000,
        )
    )
    secondary: Provider = field(
        default_factory=lambda: Provider(
            name="secondary",
            base_url=os.getenv("SECONDARY_URL", "http://127.0.0.1:9302/v1"),
            api_key=os.getenv("SECONDARY_API_KEY", ""),
            timeout_seconds=int(os.getenv("SECONDARY_TIMEOUT_MS", "10000")) / 1000,
            model=os.getenv("SECONDARY_MODEL") or None,
        )
    )
    # "key:tenant[:tpm]" pairs seeded at startup for the demo
    demo_tenants: str = field(default_factory=lambda: os.getenv("DEMO_TENANT_KEYS", "sk-tenant-a:tenant-a,sk-tenant-b:tenant-b"))
    sweep_interval_seconds: float = 30.0


# --------------------------------------------------------------------------- request model
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[Any] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # pass through temperature, top_p, tools, ...
    model: str = "default"
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_tokens: int | None = Field(None, ge=1, le=32_768)
    stream: bool = False


def estimate_prompt_tokens(req: ChatCompletionRequest) -> int:
    """Cheap, provider-agnostic estimate (~4 chars/token + per-message framing overhead).

    It only has to be good enough for admission control: the reservation is corrected
    to the provider's reported usage once the response arrives.
    """
    chars = 0
    for m in req.messages:
        if isinstance(m.content, str):
            chars += len(m.content)
        elif isinstance(m.content, list):
            chars += sum(len(str(part.get("text", ""))) if isinstance(part, dict) else len(str(part)) for part in m.content)
    return math.ceil(chars / 4) + 4 * len(req.messages) + 3


def actual_tokens(body: dict[str, Any]) -> int | None:
    usage = body.get("usage")
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if isinstance(total, int):
            return total
        p, c = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if isinstance(p, int) and isinstance(c, int):
            return p + c
    return None


# --------------------------------------------------------------------------- errors
ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error",
               413: "invalid_request_error", 422: "invalid_request_error", 429: "rate_limit_error"}


def error_response(status: int, code: str, message: str, request_id: str, headers: dict[str, str] | None = None) -> JSONResponse:
    err_type = ERROR_TYPES.get(status, "api_error" if status < 500 else "gateway_error")
    return JSONResponse(
        {"error": {"type": err_type, "code": code, "message": message, "request_id": request_id}},
        status_code=status,
        headers={"x-request-id": request_id, **(headers or {})},
    )


def _rid(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    if rid is None:
        rid = request.state.request_id = str(uuid.uuid4())
    return rid


# --------------------------------------------------------------------------- app
def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    limiter: SQLiteTokenRateLimiter | None = None,
) -> FastAPI:
    settings = settings or Settings()
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    limiter = limiter or SQLiteTokenRateLimiter(settings.db_path, window_seconds=settings.window_seconds)
    router = ModelRouter(settings.primary, settings.secondary, transport=transport)

    for entry in filter(None, (e.strip() for e in settings.demo_tenants.split(","))):
        key, tenant_id, *tpm = entry.split(":")
        limiter.upsert_tenant_sync(key, tenant_id, int(tpm[0]) if tpm else settings.default_tpm)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async def sweeper():
            while True:
                await asyncio.sleep(settings.sweep_interval_seconds)
                try:
                    removed = await limiter.sweep()
                    if removed:
                        log.debug("evicted %d expired rate-limit rows", removed)
                except Exception:  # noqa: BLE001
                    log.exception("rate-limit sweep failed")

        task = asyncio.create_task(sweeper())
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await router.aclose()

    app = FastAPI(title="LLM Gateway - rate limiting & failover", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.limiter = limiter
    app.state.router = router

    # ---- uniform, sanitised error handling for *every* failure path ----
    @app.exception_handler(GatewayError)
    async def _gateway_error(request: Request, exc: GatewayError):
        headers = {"retry-after": str(math.ceil(exc.retry_after))} if exc.retry_after is not None else None
        return error_response(exc.status, exc.code, exc.message, _rid(request), headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        fields = sorted({".".join(str(p) for p in e["loc"] if p != "body") for e in exc.errors()})
        return error_response(400, "invalid_request", f"Invalid request fields: {', '.join(fields) or 'body'}", _rid(request))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return error_response(exc.status_code, code, "Request could not be served", _rid(request))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error rid=%s", _rid(request))  # full trace stays in server logs
        return error_response(500, "internal_error", "An internal error occurred", _rid(request))

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: ChatCompletionRequest):
        rid = _rid(request)

        # 1. tenant auth
        auth = request.headers.get("authorization", "")
        api_key = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
        tenant = await limiter.get_tenant(api_key) if api_key else None
        if tenant is None:
            raise GatewayError(401, "invalid_api_key", "Missing or invalid API key")
        if body.stream:
            raise GatewayError(400, "stream_not_supported", "Streaming is not supported on this endpoint")

        # 2. token-aware admission control
        max_tokens = body.max_tokens or DEFAULT_MAX_TOKENS
        estimate = estimate_prompt_tokens(body) + max_tokens
        decision = await limiter.reserve(tenant, estimate)
        rl_headers = {"x-ratelimit-limit-tokens": str(decision.limit), "x-ratelimit-remaining-tokens": str(decision.remaining)}
        if not decision.allowed:
            if decision.reason == "request_exceeds_limit":
                return error_response(
                    413, "request_too_large",
                    f"Request needs ~{estimate} tokens, above the tenant limit of {decision.limit}/min", rid, rl_headers,
                )
            return error_response(
                429, "rate_limit_exceeded",
                f"Token rate limit of {decision.limit}/min exceeded for this API key", rid,
                {**rl_headers, "retry-after": str(max(1, math.ceil(decision.retry_after_seconds or 1)))},
            )

        # 3. route with failover
        payload = body.model_dump(exclude_none=True)
        payload["max_tokens"] = max_tokens
        try:
            result = await router.complete(payload, rid)
        except BaseException:
            # nothing was served (or the client went away) -> refund the reservation
            await asyncio.shield(limiter.release(decision.reservation_id))
            raise

        # 4. charge what was actually used
        used = actual_tokens(result.body)
        if used is not None:
            await limiter.reconcile(decision.reservation_id, used)
            remaining = decision.remaining + estimate - used
            rl_headers["x-ratelimit-remaining-tokens"] = str(max(0, remaining))

        log.info(
            "rid=%s tenant=%s provider=%s fallback=%s attempts=%s est=%d used=%s",
            rid, tenant.tenant_id, result.provider, result.fell_back,
            [(a.provider, a.outcome, round(a.latency_ms)) for a in result.attempts], estimate, used,
        )
        return JSONResponse(
            result.body,
            headers={**rl_headers, "x-request-id": rid, "x-gateway-provider": result.provider,
                     "x-gateway-fallback": str(result.fell_back).lower()},
        )

    return app


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.getenv("GATEWAY_PORT", "8082")))
