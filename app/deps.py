"""FastAPI auth dependencies — resolve the calling Principal (→ tenant scope).

- `get_principal`          : required auth (401 if no/invalid token). Use for uploads.
- `get_principal_optional` : optional auth → falls back to the default tenant, so
  CLI/legacy flows keep working without a token.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException

from app.config import settings
from app.services.auth import verify_token


@dataclass
class Principal:
    user_id: str
    tenant_id: str
    email: str = ""
    anonymous: bool = False


def _from_header(authorization: str | None) -> Principal | None:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    payload = verify_token(authorization.split(" ", 1)[1].strip())
    if not payload:
        return None
    return Principal(payload["sub"], payload["tenant"], payload.get("email", ""))


def get_principal(authorization: str | None = Header(default=None)) -> Principal:
    principal = _from_header(authorization)
    if principal is None:
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")
    return principal


def get_principal_optional(authorization: str | None = Header(default=None)) -> Principal:
    return _from_header(authorization) or Principal(
        settings.default_tenant, settings.default_tenant, anonymous=True
    )
