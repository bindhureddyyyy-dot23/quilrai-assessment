"""Streaming PII redactor with a bounded hold-back buffer.

The problem
-----------
LLM providers stream text in arbitrary fragments, so PII is routinely split across
chunks:  "...mail jane.d"  |  "oe@exam"  |  "ple.com now"
A per-chunk regex misses it; buffering the whole response kills TTFT and memory.

The approach (think of a newsroom censor reading a ticker tape)
---------------------------------------------------------------
Everything up to the last *safe cut point* is final: it is redacted and released
immediately. Only the trailing fragment that could still grow into PII is held back.

A cut point is safe when no PII match can straddle it. All patterns here are built from
"token" characters (letters, digits, . _ % + - @) plus single spaces *between digit
groups* (card / SSN numbers written as "4111 1111 ..."). So the held-back tail is:

    the trailing run of token characters, extended backwards over any
    "<1-6 digits><space>" groups that precede a digit-led run.

For ordinary prose that is just the last partial word, so added latency is about one
token. Memory is O(MAX_HOLDBACK), independent of response length: if a single token run
ever exceeds MAX_HOLDBACK characters it is force-flushed (an email address cannot
legally exceed 254 characters, so the default of 256 loses nothing real).

Invariant (verified by property tests): for any chunking of a text,
    "".join(feed(c) for c in chunks) + flush()  ==  redact(text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

REDACTED = "[REDACTED]"
MAX_HOLDBACK = 256

_EMAIL = r"(?P<email>[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,})"
# US SSN: AAA-GG-SSSS (dash or space), excluding never-issued area/group/serial values.
_SSN = r"(?P<ssn>\b(?!000|666|9\d\d)\d{3}(?P<sep>[- ])(?!00)\d{2}(?P=sep)(?!0000)\d{4}\b)"
# Payment cards, Luhn-checked in redact(). Accepted layouts (single consistent separator):
#   contiguous 13-19 digits | 4-4-4-4 | 4-4-4-4-3 | 4-6-5 (Amex) | 4-6-4 (Diners)
# A permissive "any digit with optional separators" pattern is a trap: it greedily glues a card
# to a following SSN ("4111 1111 1111 1111 123-45-6789"), fails Luhn, and leaks BOTH. The
# (?!-?\d) guard stops the 19-digit layout from eating the start of an adjacent number.
_CARD = (
    r"(?P<card>\b(?:\d{13,19}"
    r"|\d{4}(?P<csep>[ \-])(?:\d{4}(?P=csep)\d{4}(?P=csep)\d{4}(?:(?P=csep)\d{3})?|\d{6}(?P=csep)\d{4,5})"
    r")\b(?!-?\d))"
)

PII_PATTERN = re.compile(f"{_EMAIL}|{_SSN}|{_CARD}", re.ASCII)

_TOKEN_EXTRA = frozenset("._%+-@")
_DIGITS = frozenset("0123456789")


def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c in _DIGITS]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass
class RedactionStats:
    email: int = 0
    ssn: int = 0
    card: int = 0

    @property
    def total(self) -> int:
        return self.email + self.ssn + self.card


def redact(text: str, stats: RedactionStats | None = None) -> str:
    """Redact a *complete* piece of text."""

    def _sub(m: re.Match[str]) -> str:
        if m.group("card") is not None:
            if not luhn_valid(m.group("card")):
                return m.group(0)  # long digit run that is not a card (order id, tracking no, ...)
            kind = "card"
        elif m.group("ssn") is not None:
            kind = "ssn"
        else:
            kind = "email"
        if stats is not None:
            setattr(stats, kind, getattr(stats, kind) + 1)
        return REDACTED

    return PII_PATTERN.sub(_sub, text)


def _is_token_char(c: str) -> bool:
    return c.isalnum() or c in _TOKEN_EXTRA


def safe_cut(buf: str, max_holdback: int = MAX_HOLDBACK) -> int:
    """Index such that buf[:idx] can be finalised; buf[idx:] may still become PII.

    O(len of held tail) backwards scan - never rescans already-released text.
    """
    n = len(buf)
    lo = max(0, n - max_holdback)
    i = n
    # A trailing space right after a digit may be a separator inside "4111 1111 ...".
    if i - 2 >= lo and buf[i - 1] == " " and buf[i - 2] in _DIGITS:
        i -= 1
    # Trailing run of token characters.
    while i > lo and _is_token_char(buf[i - 1]):
        i -= 1
    # Extend backwards over "<1-6 digits><space>" groups that precede a digit-led run.
    while i - 1 > lo and buf[i - 1] == " " and i < n and buf[i] in _DIGITS:
        k = i - 1
        while k > lo and buf[k - 1] in _DIGITS:
            k -= 1
        group_len = (i - 1) - k
        if 1 <= group_len <= 6 and (k == 0 or not _is_token_char(buf[k - 1])):
            i = k
        else:
            break
    return i


@dataclass
class StreamingRedactor:
    """Incremental redactor. One instance per stream (per choice index)."""

    max_holdback: int = MAX_HOLDBACK
    stats: RedactionStats = field(default_factory=RedactionStats)
    _buf: str = ""
    peak_buffer: int = 0  # observability: proves memory stays bounded

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        self.peak_buffer = max(self.peak_buffer, len(self._buf))
        cut = safe_cut(self._buf, self.max_holdback)
        if cut == 0:
            return ""
        ready, self._buf = self._buf[:cut], self._buf[cut:]
        return redact(ready, self.stats)

    def flush(self) -> str:
        ready, self._buf = self._buf, ""
        return redact(ready, self.stats) if ready else ""

    @property
    def pending(self) -> int:
        return len(self._buf)
