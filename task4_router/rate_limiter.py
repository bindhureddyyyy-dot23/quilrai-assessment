"""Token-aware sliding-window rate limiter persisted in an on-disk SQLite database.

Algorithm: sliding window *log*
-------------------------------
Every admitted request writes one row (tenant, ts, tokens). A request is admitted when

    sum(tokens of this tenant's rows with ts > now - window) + requested  <=  limit

Unlike a fixed window there is no boundary burst (2x the limit across a minute edge),
and unlike a token bucket the accounting is exact and auditable row by row.

Reserve -> reconcile -> (release)
---------------------------------
The real token count of a completion is unknown until the provider answers, so:
1. reserve(estimate)   prompt-token estimate + max_tokens is booked up-front, atomically
2. reconcile(actual)   the row is corrected to the provider's reported usage afterwards
3. release()           if every provider failed, the reservation is refunded
This keeps concurrent requests from jointly overshooting the limit while they are in flight.

Concurrency
-----------
`BEGIN IMMEDIATE` takes SQLite's write lock *before* reading the window sum, so
check-then-insert is atomic across coroutines, threads AND separate gateway processes
sharing the same DB file. WAL mode lets readers proceed while a writer holds the lock.
SQLite calls are blocking, so the async API runs them in a worker thread.

Eviction
--------
Expired rows for a tenant are deleted inside every reserve() for that tenant, and sweep()
(run periodically by the app) removes expired rows of idle tenants, so the table holds at
most one window of history.

API keys are stored as SHA-256 hashes; raw keys never touch the database.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    api_key_hash      TEXT PRIMARY KEY,
    tenant_id         TEXT NOT NULL UNIQUE,
    tokens_per_minute INTEGER NOT NULL CHECK (tokens_per_minute > 0)
);
CREATE TABLE IF NOT EXISTS token_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT    NOT NULL,
    ts        REAL    NOT NULL,
    tokens    INTEGER NOT NULL CHECK (tokens >= 0)
);
CREATE INDEX IF NOT EXISTS idx_events_tenant_ts ON token_events (tenant_id, ts);
"""


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


@dataclass(frozen=True)
class Tenant:
    tenant_id: str
    tokens_per_minute: int


@dataclass(frozen=True)
class Decision:
    allowed: bool
    limit: int
    remaining: int
    reservation_id: int | None = None
    retry_after_seconds: float | None = None  # None + not allowed => request can never fit
    reason: str | None = None


class SQLiteTokenRateLimiter:
    def __init__(self, db_path: str | Path, window_seconds: float = 60.0, clock: Callable[[], float] = time.time):
        self.db_path = str(db_path)
        self.window = window_seconds
        self.clock = clock
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None -> we issue BEGIN/COMMIT ourselves. busy timeout waits for the lock.
        conn = sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ----------------------------------------------------------------- tenants
    def upsert_tenant_sync(self, api_key: str, tenant_id: str, tokens_per_minute: int) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO tenants (api_key_hash, tenant_id, tokens_per_minute) VALUES (?, ?, ?) "
                "ON CONFLICT(api_key_hash) DO UPDATE SET tenant_id=excluded.tenant_id, "
                "tokens_per_minute=excluded.tokens_per_minute",
                (hash_key(api_key), tenant_id, tokens_per_minute),
            )
        finally:
            conn.close()

    def _get_tenant_sync(self, api_key: str) -> Tenant | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT tenant_id, tokens_per_minute FROM tenants WHERE api_key_hash = ?", (hash_key(api_key),)
            ).fetchone()
        finally:
            conn.close()
        return Tenant(*row) if row else None

    async def get_tenant(self, api_key: str) -> Tenant | None:
        return await asyncio.to_thread(self._get_tenant_sync, api_key)

    # ------------------------------------------------------------- core logic
    def _reserve_sync(self, tenant: Tenant, tokens: int) -> Decision:
        limit = tenant.tokens_per_minute
        if tokens > limit:
            return Decision(False, limit, limit, reason="request_exceeds_limit")

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = self.clock()
            cutoff = now - self.window
            conn.execute("DELETE FROM token_events WHERE tenant_id = ? AND ts <= ?", (tenant.tenant_id, cutoff))
            (used,) = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_events WHERE tenant_id = ? AND ts > ?",
                (tenant.tenant_id, cutoff),
            ).fetchone()

            if used + tokens > limit:
                retry_after = self._retry_after(conn, tenant.tenant_id, cutoff, used, tokens, limit, now)
                conn.execute("COMMIT")  # persists the eviction
                return Decision(False, limit, max(0, limit - used), retry_after_seconds=retry_after, reason="rate_limited")

            cur = conn.execute(
                "INSERT INTO token_events (tenant_id, ts, tokens) VALUES (?, ?, ?)", (tenant.tenant_id, now, tokens)
            )
            conn.execute("COMMIT")
            return Decision(True, limit, limit - used - tokens, reservation_id=cur.lastrowid)
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _retry_after(self, conn, tenant_id, cutoff, used, tokens, limit, now) -> float:
        """Earliest time at which enough old rows will have expired for `tokens` to fit."""
        need_to_free = used + tokens - limit
        freed = 0
        for ts, row_tokens in conn.execute(
            "SELECT ts, tokens FROM token_events WHERE tenant_id = ? AND ts > ? ORDER BY ts ASC", (tenant_id, cutoff)
        ):
            freed += row_tokens
            if freed >= need_to_free:
                return max(0.0, ts + self.window - now)
        return self.window  # unreachable when tokens <= limit; defensive

    def _reconcile_sync(self, reservation_id: int, actual_tokens: int) -> None:
        conn = self._connect()
        try:
            conn.execute("UPDATE token_events SET tokens = ? WHERE id = ?", (max(0, int(actual_tokens)), reservation_id))
        finally:
            conn.close()

    def _release_sync(self, reservation_id: int) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM token_events WHERE id = ?", (reservation_id,))
        finally:
            conn.close()

    def _sweep_sync(self) -> int:
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM token_events WHERE ts <= ?", (self.clock() - self.window,))
            return cur.rowcount
        finally:
            conn.close()

    def _usage_sync(self, tenant_id: str) -> tuple[int, int]:
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT COALESCE(SUM(tokens), 0), COUNT(*) FROM token_events WHERE tenant_id = ? AND ts > ?",
                (tenant_id, self.clock() - self.window),
            ).fetchone()
        finally:
            conn.close()

    # --------------------------------------------------------------- async API
    async def reserve(self, tenant: Tenant, tokens: int) -> Decision:
        return await asyncio.to_thread(self._reserve_sync, tenant, tokens)

    async def reconcile(self, reservation_id: int, actual_tokens: int) -> None:
        await asyncio.to_thread(self._reconcile_sync, reservation_id, actual_tokens)

    async def release(self, reservation_id: int) -> None:
        await asyncio.to_thread(self._release_sync, reservation_id)

    async def sweep(self) -> int:
        return await asyncio.to_thread(self._sweep_sync)

    async def usage(self, tenant_id: str) -> tuple[int, int]:
        """(tokens used in current window, live row count) - for tests and observability."""
        return await asyncio.to_thread(self._usage_sync, tenant_id)
