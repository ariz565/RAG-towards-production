# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────
# Vision — multi-stage image built with uv.
# Stage 1 resolves + installs deps into a venv (cached by lockfile layers);
# stage 2 is a slim runtime that copies only the venv + app code.
# ─────────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS builder

# uv: fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Copy only dependency manifests first → this layer is cached unless deps change.
COPY pyproject.toml ./
COPY uv.lock* ./
RUN uv sync --no-dev --no-install-project

# Now copy source and finalize.
COPY . .
RUN uv sync --no-dev --no-install-project


FROM python:3.12-slim AS runtime

# Non-root runtime user.
RUN useradd --create-home --uid 10001 appuser

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Writable data dir (indexes, bm25, qdrant, audit, etc.).
RUN mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

# Lightweight liveness probe against the health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/api/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
