"""Task 4 tests: SQLite sliding-window limiter, failover router, and sanitised gateway errors."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time

import httpx
import pytest

from app import Settings, create_app, estimate_prompt_tokens, ChatCompletionRequest
from mock_providers import make_provider
from rate_limiter import SQLiteTokenRateLimiter, Tenant, hash_key
from router import Provider


# --------------------------------------------------------------------------- helpers
class FakeClock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class HostRouter(httpx.AsyncBaseTransport):
    """Sends http://primary/... to one ASGI app and http://secondary/... to another."""

    def __init__(self, apps: dict[str, object]):
        self.transports = {host: httpx.ASGITransport(app=a) for host, a in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self.transports[request.url.host].handle_async_request(request)


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "gateway.db")


@pytest.fixture
def providers():
    return make_provider("primary", slow_seconds=5.0), make_provider("secondary")


def build(db, providers, *, primary_timeout=0.3, tenants="sk-a:tenant-a,sk-small:tenant-small:1200"):
    primary, secondary = providers
    settings = Settings(
        db_path=db,
        primary=Provider("primary", "http://primary/v1", timeout_seconds=primary_timeout),
        secondary=Provider("secondary", "http://secondary/v1", timeout_seconds=2.0, model="backup-model"),
        demo_tenants=tenants,
    )
    app = create_app(settings, transport=HostRouter({"primary": primary, "secondary": secondary}))
    return app, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")


def req(content="hello", max_tokens=100, **extra):
    return {"model": "gpt-x", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, **extra}


AUTH = {"Authorization": "Bearer sk-a"}

LEAK_MARKERS = ["Traceback", "/srv/internal", "db-password", "gpu-17", ".internal", "primary/v1", "secondary/v1", "KeyError"]


def assert_standard_error(r: httpx.Response, status: int, code: str):
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"type", "code", "message", "request_id"}
    assert body["error"]["code"] == code
    assert body["error"]["request_id"] == r.headers["x-request-id"]
    for marker in LEAK_MARKERS:
        assert marker not in r.text, f"leaked {marker!r}: {r.text}"


# =========================================================================== limiter
async def test_admits_until_limit_then_rejects(db):
    lim = SQLiteTokenRateLimiter(db, clock=FakeClock())
    t = Tenant("t1", 50_000)
    for _ in range(5):
        assert (await lim.reserve(t, 10_000)).allowed
    d = await lim.reserve(t, 1)
    assert not d.allowed and d.reason == "rate_limited" and d.remaining == 0


async def test_window_slides_rather_than_resets(db):
    clock = FakeClock()
    lim = SQLiteTokenRateLimiter(db, clock=clock)
    t = Tenant("t1", 1000)
    assert (await lim.reserve(t, 600)).allowed  # t=0
    clock.t += 30
    assert (await lim.reserve(t, 400)).allowed  # t=30, window full
    assert not (await lim.reserve(t, 1)).allowed
    clock.t += 30.5  # t=60.5: the 600 has expired, the 400 has not
    assert (await lim.reserve(t, 600)).allowed
    assert not (await lim.reserve(t, 1)).allowed


async def test_retry_after_is_exact(db):
    clock = FakeClock()
    lim = SQLiteTokenRateLimiter(db, clock=clock)
    t = Tenant("t1", 1000)
    await lim.reserve(t, 300)  # t=0
    clock.t += 10
    await lim.reserve(t, 700)  # t=10
    clock.t += 5  # t=15
    d = await lim.reserve(t, 200)  # needs the t=0 row (300) to expire -> at t=60
    assert d.retry_after_seconds == pytest.approx(45.0)
    d = await lim.reserve(t, 900)  # needs both rows gone -> at t=70
    assert d.retry_after_seconds == pytest.approx(55.0)


async def test_eviction_deletes_expired_rows(db):
    clock = FakeClock()
    lim = SQLiteTokenRateLimiter(db, clock=clock)
    for i in range(10):
        await lim.reserve(Tenant("busy", 50_000), 100)
        await lim.reserve(Tenant(f"idle-{i}", 50_000), 100)
    clock.t += 61
    await lim.reserve(Tenant("busy", 50_000), 1)  # evicts busy's own expired rows inline
    assert await lim.sweep() == 10  # sweeper evicts the idle tenants' rows
    (rows,) = sqlite3.connect(db).execute("SELECT COUNT(*) FROM token_events").fetchone()
    assert rows == 1


async def test_reconcile_and_release(db):
    lim = SQLiteTokenRateLimiter(db, clock=FakeClock())
    t = Tenant("t1", 1000)
    d = await lim.reserve(t, 900)
    await lim.reconcile(d.reservation_id, 120)
    assert (await lim.usage("t1"))[0] == 120
    d2 = await lim.reserve(t, 800)
    assert d2.allowed
    await lim.release(d2.reservation_id)
    assert (await lim.usage("t1"))[0] == 120


async def test_request_larger_than_limit(db):
    lim = SQLiteTokenRateLimiter(db, clock=FakeClock())
    d = await lim.reserve(Tenant("t1", 1000), 1001)
    assert not d.allowed and d.reason == "request_exceeds_limit" and d.retry_after_seconds is None


async def test_concurrent_reservations_never_overshoot(db):
    lim = SQLiteTokenRateLimiter(db)
    t = Tenant("t1", 50_000)
    results = await asyncio.gather(*(lim.reserve(t, 1_000) for _ in range(120)))
    assert sum(r.allowed for r in results) == 50
    assert (await lim.usage("t1"))[0] == 50_000


def test_multiple_processes_share_one_window(db):
    """Two independent limiter instances (as two gateway workers would have) on the same file."""
    a, b = SQLiteTokenRateLimiter(db), SQLiteTokenRateLimiter(db)
    t = Tenant("t1", 10_000)
    allowed = []
    lock = threading.Lock()

    def worker(lim):
        for _ in range(20):
            ok = lim._reserve_sync(t, 500).allowed
            with lock:
                allowed.append(ok)

    threads = [threading.Thread(target=worker, args=(lim,)) for lim in (a, b, a, b)]
    [th.start() for th in threads]
    [th.join() for th in threads]
    assert sum(allowed) == 20


def test_api_keys_are_hashed_at_rest(db):
    lim = SQLiteTokenRateLimiter(db)
    lim.upsert_tenant_sync("sk-super-secret", "t1", 1000)
    dump = "\n".join(sqlite3.connect(db).iterdump())
    assert "sk-super-secret" not in dump and hash_key("sk-super-secret") in dump


def test_token_estimate():
    r = ChatCompletionRequest(messages=[{"role": "user", "content": "x" * 400}])
    assert estimate_prompt_tokens(r) == 100 + 4 + 3


# =========================================================================== routing
async def test_primary_serves_when_healthy(db, providers):
    app, c = build(db, providers)
    r = await c.post("/v1/chat/completions", json=req(), headers=AUTH)
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "primary" and r.headers["x-gateway-fallback"] == "false"
    assert "primary provider" in r.json()["choices"][0]["message"]["content"]


async def test_failover_on_429(db, providers):
    providers[0].state.mode = "429"
    app, c = build(db, providers)
    r = await c.post("/v1/chat/completions", json=req(), headers=AUTH)
    assert r.status_code == 200 and r.headers["x-gateway-provider"] == "secondary"
    assert r.json()["model"] == "backup-model"  # secondary model override applied


async def test_failover_on_timeout_cancels_primary(db, providers):
    providers[0].state.mode = "slow"
    app, c = build(db, providers, primary_timeout=0.3)
    t0 = time.perf_counter()
    r = await c.post("/v1/chat/completions", json=req(), headers=AUTH)
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200 and r.headers["x-gateway-provider"] == "secondary"
    assert 0.3 <= elapsed < 1.5  # did not wait for the 5 s primary


async def test_default_deadline_is_3000ms(db, providers):
    providers[0].state.mode = "slow"
    app, c = build(db, providers, primary_timeout=Settings().primary.timeout_seconds)
    assert Settings().primary.timeout_seconds == 3.0
    t0 = time.perf_counter()
    r = await c.post("/v1/chat/completions", json=req(), headers=AUTH)
    elapsed = time.perf_counter() - t0
    assert r.headers["x-gateway-provider"] == "secondary"
    assert 3.0 <= elapsed < 4.0


async def test_failover_on_5xx_without_leaking(db, providers):
    providers[0].state.mode = "500"
    app, c = build(db, providers)
    r = await c.post("/v1/chat/completions", json=req(), headers=AUTH)
    assert r.headers["x-gateway-provider"] == "secondary"
    assert "Traceback" not in r.text


async def test_both_down_is_sanitised_503_and_refunds_tokens(db, providers):
    providers[0].state.mode = "500"
    providers[1].state.mode = "500"
    app, c = build(db, providers)
    r = await c.post("/v1/chat/completions", json=req(max_tokens=5000), headers=AUTH)
    assert_standard_error(r, 503, "all_providers_unavailable")
    assert "retry-after" in r.headers
    assert (await app.state.limiter.usage("tenant-a"))[0] == 0  # reservation released


async def test_both_rate_limited_returns_429_with_retry_after(db, providers):
    providers[0].state.mode = "429"
    providers[1].state.mode = "429"
    app, c = build(db, providers)
    r = await c.post("/v1/chat/completions", json=req(), headers=AUTH)
    assert_standard_error(r, 429, "upstream_rate_limited")
    assert r.headers["retry-after"] == "7"


async def test_client_error_is_not_retried_on_secondary(db, providers):
    providers[0].state.mode = "400"
    app, c = build(db, providers)
    r = await c.post("/v1/chat/completions", json=req(), headers=AUTH)
    assert_standard_error(r, 400, "upstream_rejected_request")
    assert providers[1].state.hits == 0


async def test_network_error_fails_over():
    settings = Settings(
        db_path=":memory:",
        primary=Provider("primary", "http://127.0.0.1:1/v1", timeout_seconds=1),
        secondary=Provider("secondary", "http://secondary/v1"),
    )
    secondary = make_provider("secondary")

    class T(httpx.AsyncBaseTransport):
        def __init__(self):
            self.real, self.asgi = httpx.AsyncHTTPTransport(), httpx.ASGITransport(app=secondary)

        async def handle_async_request(self, request):
            return await (self.asgi if request.url.host == "secondary" else self.real).handle_async_request(request)

    import tempfile, os
    settings.db_path = os.path.join(tempfile.mkdtemp(), "g.db")
    app = create_app(settings, transport=T())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        r = await c.post("/v1/chat/completions", json=req(), headers={"Authorization": "Bearer sk-tenant-a"})
    assert r.headers["x-gateway-provider"] == "secondary"


# =========================================================================== gateway-level limits & errors
async def test_usage_is_reconciled_to_provider_report(db, providers):
    app, c = build(db, providers)
    r = await c.post("/v1/chat/completions", json=req(content="x" * 40, max_tokens=4000), headers=AUTH)
    used = r.json()["usage"]["total_tokens"]
    assert (await app.state.limiter.usage("tenant-a"))[0] == used  # not the 4000+ estimate
    assert int(r.headers["x-ratelimit-remaining-tokens"]) == 50_000 - used


async def test_inflight_reservations_count_against_limit(db, providers):
    """Two concurrent ~1000-token requests on a 1200 TPM tenant: exactly one is admitted."""
    providers[0].state.mode = "slow"
    app, c = build(db, providers, primary_timeout=0.3)
    small = {"Authorization": "Bearer sk-small"}
    r1, r2 = await asyncio.gather(
        c.post("/v1/chat/completions", json=req(max_tokens=1000), headers=small),
        c.post("/v1/chat/completions", json=req(max_tokens=1000), headers=small),
    )
    statuses = sorted([r1.status_code, r2.status_code])
    assert statuses == [200, 429]
    limited = r1 if r1.status_code == 429 else r2
    assert_standard_error(limited, 429, "rate_limit_exceeded")
    assert int(limited.headers["retry-after"]) >= 1
    assert limited.headers["x-ratelimit-limit-tokens"] == "1200"


async def test_request_bigger_than_tenant_limit(db, providers):
    app, c = build(db, providers)
    r = await c.post("/v1/chat/completions", json=req(max_tokens=2000), headers={"Authorization": "Bearer sk-small"})
    assert_standard_error(r, 413, "request_too_large")


async def test_per_tenant_isolation(db, providers):
    app, c = build(db, providers, tenants="sk-a:tenant-a:1000,sk-b:tenant-b:1000")
    providers[0].state.mode = "500"
    providers[1].state.mode = "500"  # failures refund, so use direct reservations instead
    lim = app.state.limiter
    await lim.reserve(Tenant("tenant-a", 1000), 1000)
    providers[0].state.mode = "ok"
    ra = await c.post("/v1/chat/completions", json=req(max_tokens=10), headers={"Authorization": "Bearer sk-a"})
    rb = await c.post("/v1/chat/completions", json=req(max_tokens=10), headers={"Authorization": "Bearer sk-b"})
    assert ra.status_code == 429 and rb.status_code == 200


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": "Basic sk-a"}])
async def test_bad_api_key(db, providers, headers):
    app, c = build(db, providers)
    r = await c.post("/v1/chat/completions", json=req(), headers=headers)
    assert_standard_error(r, 401, "invalid_api_key")


@pytest.mark.parametrize(
    "payload",
    [{}, {"messages": []}, {"messages": [{"role": "wizard", "content": "x"}]}, req(max_tokens=0), req(max_tokens="lots")],
)
async def test_invalid_body_standard_error(db, providers, payload):
    app, c = build(db, providers)
    r = await c.post("/v1/chat/completions", json=payload, headers=AUTH)
    assert_standard_error(r, 400, "invalid_request")


async def test_unknown_route_standard_error(db, providers):
    app, c = build(db, providers)
    r = await c.get("/v1/secret-admin", headers=AUTH)
    assert_standard_error(r, 404, "not_found")


async def test_unexpected_exception_is_sanitised(db, providers):
    app, c = build(db, providers)

    async def boom(*a, **k):
        raise RuntimeError("password=hunter2 at /srv/internal/secrets.py")

    app.state.router.complete = boom
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://gw") as c2:
        r = await c2.post("/v1/chat/completions", json=req(), headers=AUTH)
    assert_standard_error(r, 500, "internal_error")
    assert "hunter2" not in r.text
