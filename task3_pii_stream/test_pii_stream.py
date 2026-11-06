"""Task 3 tests: redactor correctness, chunk-boundary property tests, and end-to-end streaming."""

from __future__ import annotations

import json
import random
import re
import socket
import threading
import time

import httpx
import pytest
import uvicorn

import mock_llm
from gateway import Settings, StreamRedactionSession, create_app
from redactor import MAX_HOLDBACK, REDACTED, StreamingRedactor, luhn_valid, redact, safe_cut

# ---------------------------------------------------------------------------
# Pattern correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "mail jane.doe@example.com now",
        "first.last+tag@sub.domain.co.uk",
        "SSN 123-45-6789",
        "SSN 123 45 6789",
        "visa 4111111111111111",
        "visa 4111 1111 1111 1111",
        "mc 5555-5555-5555-4444",
        "amex 3782 822463 10005",
    ],
)
def test_detects_pii(text):
    out = redact(text)
    assert REDACTED in out
    assert "@" not in out
    assert not re.search(r"\d{4}", out)


@pytest.mark.parametrize(
    "text",
    [
        "order 1234567890123 shipped",  # 13 digits, fails Luhn
        "invalid SSN 000-12-3456 and 666-12-3456 and 123-00-4567",
        "version 3.14.159 and year 2024",
        "price is $1,299.99",
        "user@localhost has no TLD",
        "mixed separators 123-45 6789",
        "card with mixed separators 4111-1111 1111-1111",
    ],
)
def test_ignores_non_pii(text):
    assert redact(text) == text


def test_adjacent_card_and_ssn_both_redacted():
    assert redact("4111 1111 1111 1111 123-45-6789 x") == f"{REDACTED} {REDACTED} x"


def test_luhn():
    assert luhn_valid("4111 1111 1111 1111")
    assert not luhn_valid("4111 1111 1111 1112")
    assert not luhn_valid("411111")


# ---------------------------------------------------------------------------
# Streaming invariant: any chunking == whole-text redaction
# ---------------------------------------------------------------------------
CORPUS = [
    "Contact jane.doe+work@example.co.uk or call. SSN 123-45-6789, card 4111 1111 1111 1111 exp 12/29.",
    "Order 1234567890123 is not a card. Amex 3782-822463-10005 and 5555555555554444 are. Mail bob@x.io.",
    "SSN 123 45 6789 then 078-05-1120; bad 000-12-3456. Email: a@b.cd, end.",
    "no pii here, just words and numbers 42 and 2024 and 3.14 ok",
    "ends with card 4111111111111111",
    "ends with email someone@example.org",
    "unicode ✓ café naïve — test@exämple.com and test@example.com 🚀",
    "",
    "4111111111111111",
]


def _stream(text: str, rnd: random.Random, max_chunk: int) -> str:
    r = StreamingRedactor()
    out, i = [], 0
    while i < len(text):
        n = rnd.randint(1, max_chunk)
        out.append(r.feed(text[i : i + n]))
        i += n
    out.append(r.flush())
    return "".join(out)


@pytest.mark.parametrize("text", CORPUS)
@pytest.mark.parametrize("max_chunk", [1, 3, 8, 40])
def test_any_chunking_matches_full_redaction(text, max_chunk):
    rnd = random.Random(hash((text, max_chunk)) & 0xFFFF)
    expected = redact(text)
    for _ in range(300):
        assert _stream(text, rnd, max_chunk) == expected


def test_random_generated_documents():
    rnd = random.Random(1234)
    parts = ["hello", "world", "4111 1111 1111 1111", "123-45-6789", "a.b@c.io", "42", ",", ".", "\n", "x" * 30, "1234 5678"]
    for _ in range(500):
        doc = " ".join(rnd.choice(parts) for _ in range(rnd.randint(1, 25)))
        assert _stream(doc, rnd, rnd.randint(1, 10)) == redact(doc)


def test_no_pii_fragment_ever_emitted():
    """Stronger than the invariant: no individual emitted piece may contain a PII digit run."""
    text = "card 4111 1111 1111 1111 and ssn 123-45-6789 ok"
    for size in range(1, 12):
        r = StreamingRedactor()
        pieces = [r.feed(text[i : i + size]) for i in range(0, len(text), size)] + [r.flush()]
        for p in pieces:
            assert "4111" not in p and "6789" not in p


def test_prose_is_released_promptly():
    r = StreamingRedactor()
    assert r.feed("Hello there ") == "Hello there "  # complete words released immediately
    assert r.feed("gene") == ""  # partial word held
    assert r.feed("ral kenobi") == "general "
    assert r.flush() == "kenobi"


def test_memory_is_bounded():
    r = StreamingRedactor()
    for _ in range(10_000):
        r.feed("word ")
        r.feed("y" * 7)
    assert r.peak_buffer <= MAX_HOLDBACK + 7
    huge = StreamingRedactor()
    huge.feed("z" * 100_000)  # one pathological token with no boundaries
    assert huge.pending <= MAX_HOLDBACK


def test_safe_cut_holds_digit_groups():
    buf = "pay 4111 1111 1111 "
    assert buf[safe_cut(buf):] == "4111 1111 1111 "


# ---------------------------------------------------------------------------
# SSE session rewriting
# ---------------------------------------------------------------------------
def test_session_rewrites_deltas_and_flushes_on_finish():
    s = StreamRedactionSession()
    base = {"id": "c1", "object": "chat.completion.chunk", "model": "m"}
    events = [{**base, "choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}]} for c in ["Email: jo", "hn@ex", "ample.com thanks"]]
    events.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    out = [s.process_event(e) for e in events]
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in out if e)
    assert text == "Email: [REDACTED] thanks"
    assert out[-1]["choices"][0]["finish_reason"] == "stop"


def test_session_handles_multiple_choices():
    s = StreamRedactionSession()
    ev = lambda i, c, f=None: {"choices": [{"index": i, "delta": {"content": c}, "finish_reason": f}]}  # noqa: E731
    got = {0: "", 1: ""}
    for e in [ev(0, "a@b."), ev(1, "123-45-"), ev(0, "io x", "stop"), ev(1, "6789 y", "stop")]:
        o = s.process_event(e)
        if o:
            c = o["choices"][0]
            got[c["index"]] += c["delta"].get("content", "")
    assert got == {0: "[REDACTED] x", 1: "[REDACTED] y"}


# ---------------------------------------------------------------------------
# End-to-end over real sockets (proves true incremental streaming)
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server(threading.Thread):
    def __init__(self, app, port):
        super().__init__(daemon=True)
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))

    def run(self):
        self.server.run()

    def __enter__(self):
        self.start()
        while not self.server.started:
            time.sleep(0.01)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.join(timeout=5)


@pytest.fixture(scope="module")
def live_gateway():
    import os

    os.environ["MOCK_CHUNK_DELAY_MS"] = "20"
    up_port, gw_port = _free_port(), _free_port()
    gw_app = create_app(Settings(upstream_base_url=f"http://127.0.0.1:{up_port}/v1", upstream_api_key="k"))
    with _Server(mock_llm.app, up_port), _Server(gw_app, gw_port):
        yield f"http://127.0.0.1:{gw_port}"


def _collect(base: str, prompt: str):
    first_content_at, chunks = None, []
    t0 = time.perf_counter()
    with httpx.stream(
        "POST", f"{base}/v1/chat/completions", json={"model": "m", "stream": True, "seed": 7, "messages": [{"role": "user", "content": prompt}]}, timeout=30
    ) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            c = ev["choices"][0]["delta"].get("content")
            if c:
                if first_content_at is None:
                    first_content_at = time.perf_counter() - t0
                chunks.append(c)
    return first_content_at, time.perf_counter() - t0, "".join(chunks)


def test_e2e_stream_is_redacted_and_incremental(live_gateway):
    first, total, text = _collect(live_gateway, "canned")
    assert text == redact(mock_llm.CANNED)
    assert "jane.doe" not in text and "123-45-6789" not in text and "4111" not in text
    assert "1234567890123" in text  # non-card digits preserved
    # ~50 upstream chunks x 20ms ~= 1s total; first content must arrive long before the end.
    assert first < total * 0.25, (first, total)


def test_e2e_non_streaming(live_gateway):
    r = httpx.post(f"{live_gateway}/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "SSN 123-45-6789 ok"}]})
    assert r.json()["choices"][0]["message"]["content"] == "SSN [REDACTED] ok"
    assert r.headers["x-pii-redactions"] == "1"


def test_e2e_bad_request(live_gateway):
    r = httpx.post(f"{live_gateway}/v1/chat/completions", content=b"nope", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_json"


def test_upstream_down_is_sanitised():
    import asyncio

    async def go():
        app = create_app(Settings(upstream_base_url="http://127.0.0.1:1/v1"))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
            return await c.post("/v1/chat/completions", json={"stream": True, "messages": []})

    r = asyncio.run(go())
    assert r.status_code == 502 and "127.0.0.1" not in r.text
