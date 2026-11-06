"""Bearer token verification for the MCP gateway.

Tokens are compact HS256 JWTs carrying a `role` claim (admin | viewer). Verification is
implemented with the standard library (hmac + hashlib) to keep the gateway dependency-light:

* signature checked with hmac.compare_digest (constant time)
* header `alg` pinned to HS256 - rejects `alg: none` and algorithm-confusion tricks
* `exp` (required) and `nbf` (optional) enforced with a small clock-skew leeway
* `role` must be one of the known roles; anything else is rejected rather than
  silently downgraded, so a typo in an issuer never yields surprise access
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

KNOWN_ROLES = frozenset({"admin", "viewer"})
LEEWAY_SECONDS = 30


class AuthError(Exception):
    """Raised for any token problem. The message is safe to return to the client."""


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def extract_bearer(authorization: str | None) -> str:
    if not authorization:
        raise AuthError("Missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("Authorization header must be 'Bearer <token>'")
    return token.strip()


def verify_token(token: str, secret: str, *, now: float | None = None) -> Principal:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        payload = json.loads(_b64url_decode(payload_b64))
        signature = _b64url_decode(sig_b64)
    except (ValueError, json.JSONDecodeError):
        raise AuthError("Malformed token") from None

    if not isinstance(header, dict) or header.get("alg") != "HS256":
        raise AuthError("Unsupported token algorithm")

    expected = hmac.new(secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise AuthError("Invalid token signature")

    if not isinstance(payload, dict):
        raise AuthError("Malformed token")

    now = time.time() if now is None else now
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)) or now > exp + LEEWAY_SECONDS:
        raise AuthError("Token expired")
    nbf = payload.get("nbf")
    if isinstance(nbf, (int, float)) and now + LEEWAY_SECONDS < nbf:
        raise AuthError("Token not yet valid")

    role = payload.get("role")
    if role not in KNOWN_ROLES:
        raise AuthError("Token has no recognised role")
    return Principal(subject=str(payload.get("sub", "unknown")), role=role)


def mint_token(secret: str, *, role: str, sub: str = "demo-user", ttl_seconds: int = 3600, **extra: Any) -> str:
    """Issue a token. Used by tests and the demo script; a real deployment uses its IdP."""
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {"sub": sub, "role": role, "iat": now, "exp": now + ttl_seconds, **extra}
    h = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    p = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url_encode(sig)}"
