"""Model router with automatic failover: primary -> secondary.

Failover policy
---------------
                                 primary result
  200 OK ...................................... return it
  429 Too Many Requests ....................... fail over
  no response within 3000 ms (total deadline) . fail over
  network error / 5xx / unparseable body ...... fail over (provider fault, not the caller's)
  other 4xx (bad request, auth, ...) .......... do NOT fail over: the same request would be
                                                rejected again; return a sanitised 4xx

Timeout semantics
-----------------
httpx timeouts are per phase (connect / each read), so a provider that trickles a byte
every second never trips them. The deadline is therefore enforced around the *whole*
call with asyncio.wait_for, which also cancels the in-flight request so a late primary
response can never race with (or be returned after) the fallback.

Error sanitisation
------------------
Upstream bodies, exception messages, hostnames and stack traces are logged server-side
with the request id and never placed in a client response. Callers only ever see
GatewayError(status, code, message).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger("llm_router")


class GatewayError(Exception):
    """A client-safe error. `message` must never contain upstream or internal details."""

    def __init__(self, status: int, code: str, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str = ""
    timeout_seconds: float = 3.0
    model: str | None = None  # override the requested model (e.g. a cheaper backup model)


@dataclass
class Attempt:
    provider: str
    outcome: str  # ok | rate_limited | timeout | network_error | upstream_5xx | bad_response | client_error
    status: int | None
    latency_ms: float


@dataclass
class RouteResult:
    provider: str
    body: dict[str, Any]
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def fell_back(self) -> bool:
        return len(self.attempts) > 1


class _Failover(Exception):
    def __init__(self, attempt: Attempt, retry_after: float | None = None):
        self.attempt = attempt
        self.retry_after = retry_after


class ModelRouter:
    def __init__(
        self,
        primary: Provider,
        secondary: Provider,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.providers = [primary, secondary]
        self._client = httpx.AsyncClient(
            transport=transport,
            # Generous per-phase limits; the real deadline is wait_for() per provider.
            timeout=httpx.Timeout(60.0, connect=5.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, payload: dict[str, Any], request_id: str) -> RouteResult:
        attempts: list[Attempt] = []
        retry_afters: list[float] = []
        for provider in self.providers:
            try:
                body, attempt = await self._attempt(provider, payload, request_id)
            except _Failover as fo:
                attempts.append(fo.attempt)
                if fo.retry_after is not None:
                    retry_afters.append(fo.retry_after)
                log.warning(
                    "provider=%s outcome=%s status=%s latency_ms=%.0f rid=%s -> trying next provider",
                    fo.attempt.provider, fo.attempt.outcome, fo.attempt.status, fo.attempt.latency_ms, request_id,
                )
                continue
            attempts.append(attempt)
            return RouteResult(provider=provider.name, body=body, attempts=attempts)

        # Every provider failed.
        if attempts and all(a.outcome == "rate_limited" for a in attempts):
            raise GatewayError(
                429, "upstream_rate_limited", "All model providers are rate limiting requests. Retry later.",
                retry_after=min(retry_afters) if retry_afters else 30.0,
            )
        raise GatewayError(503, "all_providers_unavailable", "No model provider is currently available. Retry later.",
                           retry_after=5.0)

    async def _attempt(self, provider: Provider, payload: dict[str, Any], request_id: str) -> tuple[dict[str, Any], Attempt]:
        body = dict(payload)
        if provider.model:
            body["model"] = provider.model
        headers = {"x-request-id": request_id}
        if provider.api_key:
            headers["authorization"] = f"Bearer {provider.api_key}"

        started = time.perf_counter()

        def attempt(outcome: str, status: int | None = None) -> Attempt:
            return Attempt(provider.name, outcome, status, (time.perf_counter() - started) * 1000)

        try:
            resp = await asyncio.wait_for(
                self._client.post(f"{provider.base_url.rstrip('/')}/chat/completions", json=body, headers=headers),
                timeout=provider.timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise _Failover(attempt("timeout")) from None
        except httpx.HTTPError as exc:
            log.debug("provider=%s transport error rid=%s: %r", provider.name, request_id, exc)
            raise _Failover(attempt("network_error")) from None

        if resp.status_code == 429:
            raise _Failover(attempt("rate_limited", 429), _parse_retry_after(resp.headers.get("retry-after")))
        if resp.status_code in (401, 403):
            # Our credentials for this provider are bad - a provider/config fault, not the caller's.
            log.error("provider=%s auth failure status=%s rid=%s", provider.name, resp.status_code, request_id)
            raise _Failover(attempt("provider_auth_error", resp.status_code))
        if resp.status_code >= 500:
            log.debug("provider=%s 5xx body (not returned to client) rid=%s: %.500s", provider.name, request_id, resp.text)
            raise _Failover(attempt("upstream_5xx", resp.status_code))
        if resp.status_code != 200:
            log.info("provider=%s rejected request status=%s rid=%s body=%.500s", provider.name, resp.status_code, request_id, resp.text)
            status = resp.status_code if resp.status_code in (400, 404, 413, 422) else 502
            raise GatewayError(status, "upstream_rejected_request", "The model provider rejected the request.")
        try:
            data = resp.json()
            if not isinstance(data, dict) or "choices" not in data:
                raise ValueError("missing choices")
        except ValueError:
            raise _Failover(attempt("bad_response", resp.status_code)) from None
        return data, attempt("ok", 200)


def _parse_retry_after(value: str | None) -> float | None:
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None
