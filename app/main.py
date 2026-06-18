"""Vision — FastAPI Application.

Reasoning-grounded document Q&A powered by PageIndex + Hybrid RAG + LangGraph.
Multi-tenant, multi-strategy, multi-model document-intelligence platform.

Startup sequence:
1. Load settings from .env
2. Auto-load any existing PageIndex from data/index/
3. Auto-load any existing Hybrid index from data/bm25/
4. Ready to serve queries via either strategy
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers.ask import router
from app.routers.auth import router as auth_router
from app.routers.governance import router as governance_router
from app.services.audit import audit_store
from app.services.auth import user_store
from app.services.versions import version_store

# ── Branding banner (pure ASCII — safe on every console) ────────────

_BANNER = r"""
__     ___     _
\ \   / (_)___(_) ___  _ __
 \ \ / /| / __| |/ _ \| '_ \
  \ V / | \__ \ | (_) | | | |
   \_/  |_|___/_|\___/|_| |_|
"""


def _print_banner(app: FastAPI) -> None:
    """Print the product banner to stdout on startup."""
    # Windows consoles default to cp1252; force UTF-8 so output never errors.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    line = "=" * 62
    print(line)
    print(_BANNER)
    print("  VISION  --  reasoning-grounded answers over any document")
    print(f"  v{app.version}   |   model={settings.active_model.value}   "
          f"retrieval={settings.active_retrieval.value}   embed={settings.active_embedding.value}")
    print(f"  http://{settings.host}:{settings.port}   |   docs at /docs")
    print(line, flush=True)


# ── Logging ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — load indexes on startup if available."""
    _print_banner(app)

    # Ensure directories exist
    for dir_path in [settings.pdf_path, settings.index_path, settings.bm25_path, settings.chunks_path]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # ── Load all document bundles (multi-tenant registry) ──────────
    from app.services.registry import registry

    count = await registry.load_all()
    if count:
        for info in registry.list():
            logger.info(
                f"  • {info['tenant_id']}/{info['doc_id']} — "
                f"{info['total_pages']} pages, hybrid={info['hybrid']}, {len(info['topics'])} topics"
            )
    else:
        logger.info("No documents found. POST to /api/index or /api/index/hybrid to create one.")

    logger.info(f"Server ready at http://{settings.host}:{settings.port}")
    logger.info(f"Docs at http://{settings.host}:{settings.port}/docs")

    if await user_store.connect():
        logger.info("User backend initialized")
    else:
        logger.warning("User backend failed to initialize; falling back if available")

    if await version_store.connect():
        logger.info("Version store initialized")
    else:
        logger.warning("Version store backend failed to initialize; falling back if available")

    if await audit_store.connect():
        logger.info("Audit store initialized")
    else:
        logger.warning("Audit store backend failed to initialize; falling back if available")

    yield  # Application runs

    await audit_store.close()
    await version_store.close()
    await user_store.close()
    logger.info("Shutting down Vision")


# ── App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vision",
    description="Reasoning-grounded document Q&A — multi-tenant, multi-strategy agentic RAG platform.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(auth_router)
app.include_router(router)
app.include_router(governance_router)


# ── Entry Point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info",
    )
