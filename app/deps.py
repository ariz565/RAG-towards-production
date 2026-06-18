"""FastAPI auth dependencies — resolve the calling Principal (→ tenant scope).

- `get_principal`          : required auth (401 if no/invalid token). Use for uploads.
- `get_principal_optional` : optional auth → falls back to the default tenant, so
  CLI/legacy flows keep working without a token.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataclasses import dataclass

from fastapi import Header, HTTPException, Cookie

from app.config import settings
from app.services.auth import verify_token


@dataclass
class Principal:
    user_id: str
    tenant_id: str
    email: str = ""
    anonymous: bool = False


def _extract_token(authorization: str | None, cookie_token: str | None) -> str | None:
    if cookie_token:
        return cookie_token
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return None


def _from_token(token: str | None) -> Principal | None:
    if not token:
        return None
    payload = verify_token(token)
    if not payload:
        return None
    return Principal(payload["sub"], payload["tenant"], payload.get("email", ""))


def get_principal(
    authorization: str | None = Header(default=None),
    vision_access_token: str | None = Cookie(default=None)
) -> Principal:
    principal = _from_token(_extract_token(authorization, vision_access_token))
    if principal is None:
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")
    return principal


def get_principal_optional(
    authorization: str | None = Header(default=None),
    vision_access_token: str | None = Cookie(default=None)
) -> Principal:
    return _from_token(_extract_token(authorization, vision_access_token)) or Principal(
        settings.default_tenant, settings.default_tenant, anonymous=True
    )
