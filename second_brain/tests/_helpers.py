"""Shared test helpers. Named with a leading underscore so pytest's `test_*.py`
collection pattern (see pyproject.toml) doesn't try to collect this as a test
module itself.
"""

from __future__ import annotations

from second_brain.adapters import OfflineLLMAdapter


class SpyLLMAdapter:
    """Wraps OfflineLLMAdapter for real dispatch behavior, while recording every
    (system, user) pair it was called with -- lets a test assert on exact prompt
    content (e.g. "did the lessons block actually reach this call") without
    needing a real LLM or mocking away the offline adapter's own logic."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._inner = OfflineLLMAdapter()

    @property
    def name(self) -> str:
        return "spy"

    async def complete_json(self, system: str, user: str) -> dict:
        self.calls.append((system, user))
        return await self._inner.complete_json(system, user)

    async def complete_text(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return await self._inner.complete_text(system, user)

    def calls_with_task(self, task: str) -> list[tuple[str, str]]:
        return [(s, u) for s, u in self.calls if s.strip().startswith(f"TASK: {task}")]
