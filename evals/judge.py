"""DeepEval judge — reuses the project's multi-provider LLM service.

DeepEval's metrics need a judge model. Instead of locking the eval gate to one
vendor, we wrap the existing ``app.services.llm`` abstraction so the judge can be
OpenAI, Azure, Groq, Gemini, or local Ollama — controlled by EVAL_JUDGE_PROVIDER
(falls back to ACTIVE_MODEL).

NOTE: A strong hosted judge (e.g. gpt-4o-mini / an Azure deployment) gives far
more reliable scores than a small local model. Set EVAL_JUDGE_PROVIDER for CI.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging

from pydantic import BaseModel

from app.config import ModelProvider, settings
from app.services.llm import chat, chat_json

logger = logging.getLogger(__name__)


def _run_sync(coro):
    """Run an async coroutine from sync code, even if a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A loop is running (e.g. inside async test) — run in a fresh thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def _resolve_provider() -> ModelProvider | None:
    name = (settings.eval_judge_provider or "").strip()
    if not name:
        return None  # fall back to settings.active_model inside llm.py
    mp = {p.value: p for p in ModelProvider}
    return mp.get(name)


def _resolve_model() -> str | None:
    return (settings.eval_judge_model or "").strip() or None


class ProjectJudge:
    """A DeepEval-compatible judge backed by the project's LLM service.

    Implements the DeepEvalBaseLLM contract (generate / a_generate /
    get_model_name / load_model) including the optional ``schema`` argument used
    by structured metrics.
    """

    def __init__(self) -> None:
        self._provider = _resolve_provider()
        self._model = _resolve_model()

    # DeepEvalBaseLLM contract -------------------------------------------------

    def load_model(self):
        return self

    def get_model_name(self) -> str:
        prov = self._provider.value if self._provider else settings.active_model.value
        model = self._model or settings.active_model_name
        return f"project-judge ({prov}/{model})"

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is not None:
            data, _ = await chat_json(
                prompt
                + "\n\nRespond with ONLY a valid JSON object matching the required schema.",
                temperature=0,
                provider=self._provider,
                model_override=self._model,
            )
            try:
                return schema.model_validate(data)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Judge schema validation failed: {e}; raw={json.dumps(data)[:200]}")
                # Best effort: let pydantic coerce what it can.
                return schema.model_construct(**data) if isinstance(data, dict) else schema()
        resp = await chat(
            prompt, temperature=0, provider=self._provider, model_override=self._model
        )
        return resp.content

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return _run_sync(self.a_generate(prompt, schema))


def build_judge() -> ProjectJudge:
    """Factory used by the eval scripts/tests."""
    judge = ProjectJudge()
    logger.info(f"Eval judge: {judge.get_model_name()}")
    return judge
