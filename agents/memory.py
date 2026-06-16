"""Agent memory — the difference between a chatbot and an agent that remembers.

- **BufferMemory** — keep the last N turns verbatim (simple, exact, but grows).
- **SummaryMemory** — keep a rolling LLM summary so context stays bounded over
  long conversations (falls back to truncation offline).

(Production memory also adds *semantic* long-term memory — embed past turns into a
vector store and retrieve relevant ones — which is the embeddings+retrieval labs
applied to conversation history.)
"""

from __future__ import annotations

from agents.base import LLM, Memory


class BufferMemory(Memory):
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._turns: list[tuple[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self._turns.append((role, content))
        self._turns = self._turns[-self.max_turns:]

    def context(self) -> str:
        return "\n".join(f"{r}: {c}" for r, c in self._turns)


class SummaryMemory(Memory):
    def __init__(self, llm: LLM | None = None, max_chars: int = 800):
        self.llm = llm
        self.max_chars = max_chars
        self.summary = ""

    def add(self, role: str, content: str) -> None:
        turn = f"{role}: {content}"
        if self.llm:
            try:
                prompt = (f"Update the running summary to include the new turn, concisely.\n\n"
                          f"Summary so far:\n{self.summary}\n\nNew turn:\n{turn}\n\nUpdated summary:")
                self.summary = self.llm(prompt).strip()[: self.max_chars]
                return
            except Exception:
                pass
        # Fallback: append + truncate from the front.
        self.summary = (self.summary + "\n" + turn)[-self.max_chars:]

    def context(self) -> str:
        return self.summary
