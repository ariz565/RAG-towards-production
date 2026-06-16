# Vision — developer task shortcuts.
# Uses `uv` (https://docs.astral.sh/uv). Install once:  pip install uv
#
# `.RECIPEPREFIX = >` lets recipe lines start with ">" instead of a TAB,
# which avoids the classic "missing separator" Makefile footgun.
.RECIPEPREFIX := >
.DEFAULT_GOAL := help

.PHONY: help install dev run test lint fmt typecheck check hooks docker-build docker-up docker-down clean

help:  ## Show this help
> @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create venv + install runtime + dev deps (uv)
> uv sync --group dev

dev:  ## Run the API with autoreload
> uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

run:  ## Run the API (production-style, no reload)
> uv run uvicorn app.main:app --host 0.0.0.0 --port 8001

test:  ## Run the test suite
> uv run pytest

lint:  ## Lint with ruff
> uv run ruff check .

fmt:  ## Auto-format + fix imports with ruff
> uv run ruff format . && uv run ruff check --fix .

typecheck:  ## Static type check with mypy
> uv run mypy app

check: lint typecheck test  ## Lint + types + tests (use this in CI)

hooks:  ## Install pre-commit git hooks
> uv run pre-commit install

docker-build:  ## Build the container image
> docker build -t vision:latest .

docker-up:  ## Start via docker compose
> docker compose up --build

docker-down:  ## Stop docker compose
> docker compose down

clean:  ## Remove caches and build artifacts
> rm -rf .ruff_cache .mypy_cache .pytest_cache **/__pycache__
