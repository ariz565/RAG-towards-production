"""Database helpers for Postgres-backed services.

Shared DSN normalization and `asyncpg` pool creation.
"""

from __future__ import annotations

import logging
import urllib.parse

from app.config import settings

logger = logging.getLogger(__name__)


def normalize_postgres_dsn(dsn: str) -> str:
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
        "Normalized Postgres DSN: converted schema=%s to options=search_path",
        schema,
    )
    return normalized


async def create_postgres_pool(dsn: str):
    try:
        import asyncpg
    except ImportError as e:
        raise RuntimeError("asyncpg is required for Postgres backends") from e

    normalized = normalize_postgres_dsn(dsn)
    return await asyncpg.create_pool(
        normalized,
        min_size=settings.pg_pool_min_size,
        max_size=settings.pg_pool_max_size,
        command_timeout=settings.pg_command_timeout_seconds,
        max_inactive_connection_lifetime=settings.pg_pool_max_inactive_connection_lifetime_seconds,
    )
