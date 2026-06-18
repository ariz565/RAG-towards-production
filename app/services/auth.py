"""Authentication — signup/login with salted hashing + signed bearer tokens.

Supports both JSON (MVP) and PostgreSQL (production) backends. Stdlib only
for hashing/tokens (PBKDF2 + HMAC-SHA256). Each user gets their own `tenant_id`
by default → natural per-user isolation.

Postgres backend is optional: auto-detected if DATABASE_URL is set in config.
Falls back to JSON with graceful degradation if DB unavailable.

Production: use Postgres backend + JWT/OAuth 2.1 + secret manager for auth_secret.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.services.db import create_postgres_pool

logger = logging.getLogger(__name__)

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


# ═══════════════════════════════════════════════════════════════════════
# ABSTRACT BACKEND INTERFACE
# ═══════════════════════════════════════════════════════════════════════


class UserBackend(ABC):
    """Abstract user storage backend (JSON or Postgres)."""

    @abstractmethod
    def get_by_email(self, email: str) -> dict | None:
        """Fetch user record by email (case-insensitive)."""
        pass

    @abstractmethod
    def create(self, email: str, password: str) -> User:
        """Create a new user with email + password."""
        pass

    @abstractmethod
    def verify(self, email: str, password: str) -> User | None:
        """Verify email/password; return User if valid, else None."""
        pass

    async def connect(self) -> bool:
        """Initialize the backend connections during app startup."""
        return True

    async def close(self) -> None:
        """Cleanly close backend connections during app shutdown."""
        return None

    async def create_async(self, email: str, password: str) -> User:
        """Async-compatible create; defaults to sync backend."""
        return self.create(email, password)

    async def verify_async(self, email: str, password: str) -> User | None:
        """Async-compatible verify; defaults to sync backend."""
        return self.verify(email, password)


# ═══════════════════════════════════════════════════════════════════════
# JSON BACKEND (MVP)
# ═══════════════════════════════════════════════════════════════════════


class JSONUserStore(UserBackend):
    """User store backed by a local JSON file."""

    def __init__(self, path: str | None = None):
        self.path = Path(path or settings.users_db_path)

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {}

    def _save(self, db: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(db, indent=2), encoding="utf-8")

    async def connect(self) -> bool:
        return True

    async def close(self) -> None:
        return None

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


# ═══════════════════════════════════════════════════════════════════════
# POSTGRES BACKEND (Production)
# ═══════════════════════════════════════════════════════════════════════


class PostgresUserBackend(UserBackend):
    """User store backed by PostgreSQL (async)."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None
        self._fallback = JSONUserStore()  # Graceful degradation if DB unavailable

    async def connect(self) -> bool:
        return await self._ensure_connected()

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Postgres user pool closed")

    def _normalize_dsn(self, dsn: str) -> str:
        parsed = urllib.parse.urlparse(dsn)
        if parsed.scheme not in {"postgres", "postgresql"}:
            return dsn

        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        schema_values = query.pop("schema", [])
        if not schema_values:
            return dsn

        schema = schema_values[-1]
        search_option = f"-c search_path={schema}"
        if query.get("options"):
            query["options"] = [f"{existing} {search_option}".strip() for existing in query["options"]]
        else:
            query["options"] = [search_option]

        normalized_query = urllib.parse.urlencode(query, doseq=True)
        normalized = parsed._replace(query=normalized_query).geturl()
        logger.info(
            "Normalized Postgres DSN: converted schema=%s to options=search_path", schema,
        )
        return normalized

    async def _ensure_connected(self) -> bool:
        """Lazy initialization and connection check."""
        if self.pool:
            return True
        try:
            self.pool = await create_postgres_pool(self.db_url)
            await self._init_schema()
            logger.info("Connected to Postgres backend for users")
            return True
        except Exception as e:
            logger.warning(f"Postgres connection failed, falling back to JSON: {e}")
            return False

    async def _init_schema(self) -> None:
        """Create users table if missing."""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        tenant_id TEXT NOT NULL,
                        password_hash TEXT NOT NULL,
                        salt TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                    """
                )
        except Exception as e:
            logger.warning(f"Failed to create users table: {e}")

    def get_by_email(self, email: str) -> dict | None:
        """Sync wrapper for sync context (uses fallback)."""
        email = email.lower()
        if not self.pool:
            return self._fallback.get_by_email(email)
        # Fallback for sync calls
        return self._fallback.get_by_email(email)

    async def get_by_email_async(self, email: str) -> dict | None:
        """Async version: fetch from Postgres or fallback."""
        email = email.lower()
        if not await self._ensure_connected():
            return self._fallback.get_by_email(email)
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT user_id, email, tenant_id, password_hash, salt FROM users WHERE email = $1",
                    email
                )
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.warning(f"Postgres query failed, using fallback: {e}")
            return self._fallback.get_by_email(email)

    def create(self, email: str, password: str) -> User:
        """Sync wrapper: use fallback."""
        return self._fallback.create(email, password)

    async def create_async(self, email: str, password: str) -> User:
        """Async version: persist to Postgres."""
        email = email.lower()
        if not email or "@" not in email:
            raise ValueError("invalid email")
        if len(password) < 8:
            raise ValueError("password must be at least 8 characters")

        if not await self._ensure_connected():
            return self._fallback.create(email, password)

        salt = secrets.token_hex(16)
        uid = secrets.token_hex(8)
        password_hash = _hash_pw(password, salt)

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO users (user_id, email, tenant_id, password_hash, salt)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    uid, email, uid, password_hash, salt
                )
            logger.info(f"Created user {email} in Postgres")
            return User(user_id=uid, email=email, tenant_id=uid, 
                       password_hash=password_hash, salt=salt)
        except Exception as e:
            if "already registered" in str(e) or "unique" in str(e).lower():
                raise ValueError("email already registered")
            logger.warning(f"Postgres insert failed, using fallback: {e}")
            return self._fallback.create(email, password)

    def verify(self, email: str, password: str) -> User | None:
        """Sync wrapper: use fallback."""
        return self._fallback.verify(email, password)

    async def verify_async(self, email: str, password: str) -> User | None:
        """Async version: verify against Postgres or fallback."""
        u = await self.get_by_email_async(email)
        if not u:
            return None
        if hmac.compare_digest(u["password_hash"], _hash_pw(password, u["salt"])):
            return User(**u)
        return None


# ═══════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════


def get_user_backend() -> UserBackend:
    """Factory: select the configured users backend and fall back gracefully."""
    if settings.users_backend == "postgres" and settings.database_url:
        logger.info("Using Postgres backend for users")
        return PostgresUserBackend(settings.database_url)
    if settings.users_backend == "json":
        logger.info("Using JSON backend for users")
        return JSONUserStore()
    logger.warning(
        "Unknown users_backend '%s'; falling back to JSON backend",
        settings.users_backend,
    )
    return JSONUserStore()


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


user_store = get_user_backend()
