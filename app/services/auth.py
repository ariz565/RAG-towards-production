"""Authentication — signup/login with salted hashing + signed bearer tokens.

Dependency-free (stdlib `hashlib`/`hmac`): PBKDF2 password hashing and an
HMAC-SHA256 signed token. Users persist in a JSON file via `UserStore` (swap the
store for a DB without touching callers). Each user gets their own `tenant_id`
by default → natural per-user isolation.

Production: replace the JSON store with a real DB and the signed token with
JWT/OAuth 2.1; keep `auth_secret` in a secret manager.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import settings

_PBKDF2_ROUNDS = 200_000


@dataclass
class User:
    user_id: str
    email: str
    tenant_id: str
    password_hash: str
    salt: str


def _hash_pw(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ROUNDS).hex()


class UserStore:
    def __init__(self, path: str | None = None):
        self.path = Path(path or settings.users_db_path)

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {}

    def _save(self, db: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(db, indent=2), encoding="utf-8")

    def get_by_email(self, email: str) -> dict | None:
        email = email.lower()
        return next((u for u in self._load().values() if u["email"] == email), None)

    def create(self, email: str, password: str) -> User:
        email = email.lower()
        if not email or "@" not in email:
            raise ValueError("invalid email")
        if len(password) < 8:
            raise ValueError("password must be at least 8 characters")
        db = self._load()
        if any(u["email"] == email for u in db.values()):
            raise ValueError("email already registered")
        salt = secrets.token_hex(16)
        uid = secrets.token_hex(8)
        record = {"user_id": uid, "email": email, "tenant_id": uid,
                  "password_hash": _hash_pw(password, salt), "salt": salt}
        db[uid] = record
        self._save(db)
        return User(**record)

    def verify(self, email: str, password: str) -> User | None:
        u = self.get_by_email(email)
        if not u:
            return None
        if hmac.compare_digest(u["password_hash"], _hash_pw(password, u["salt"])):
            return User(**u)
        return None


# ── Signed bearer tokens (HMAC-SHA256) ───────────────────────────────

def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


_ISSUER = "vision"


def create_token(user: User) -> str:
    now = int(time.time())
    payload = {
        "iss": _ISSUER, "typ": "access",
        "sub": user.user_id, "tenant": user.tenant_id, "email": user.email,
        "iat": now, "exp": now + settings.auth_token_ttl_seconds,
    }
    body = _b64(json.dumps(payload).encode())
    sig = _b64(hmac.new(settings.auth_secret.encode(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_token(token: str) -> dict | None:
    try:
        body, sig = token.split(".")
        # Constant-time signature check, then validate claims.
        expected = _b64(hmac.new(settings.auth_secret.encode(), body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_unb64(body))
        if payload.get("iss") != _ISSUER or payload.get("typ") != "access":
            return None
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


user_store = UserStore()
