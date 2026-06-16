"""Core types for the agents lab.

Standalone study module (not wired into the app). Models the pieces a real agent
needs — tools, a reasoning step, a transcript, budgets — behind small interfaces.

The "LLM" is a protocol so the agent loop is testable: inject a real model in
production, or use the dependency-free `RuleBasedLLM` to run the ReAct loop
offline (deterministic keyword→action planning).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Protocol


# ── Tools ────────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str = ""
    meta: dict = field(default_factory=dict)   # call_id, latency_ms, tool, blocked, etc.


@dataclass
class Tool:
    """A callable capability with the metadata + flags a real tool layer needs.

    `parameters` is a light JSON-schema-ish spec ({param: {type, enum, required,
    max_length}}) used for validation + the LLM/MCP tool spec. Flags drive policy:
    `dangerous`/`requires_approval` → HITL; `read_only`/`idempotent` → safe to
    retry; `rate_limit_per_min`/`timeout_s` → resource controls.
    """

    name: str
    description: str
    fn: Callable[[str], str]
    parameters: dict = field(default_factory=dict)
    read_only: bool = True
    idempotent: bool = True
    dangerous: bool = False
    requires_approval: bool = False
    timeout_s: float = 10.0
    rate_limit_per_min: int = 0          # 0 = unlimited
    version: str = "1.0"

    def run(self, arg: str) -> ToolResult:
        """Direct (unguarded) call. Prefer SecureToolRunner in agents."""
        try:
            return ToolResult(True, str(self.fn(arg)))
        except Exception as e:
            return ToolResult(False, "", f"{type(e).__name__}: {e}")

    def to_spec(self) -> dict:
        """LLM function-spec / MCP tools/list shape."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters or {"input": {"type": "string", "required": True}},
            "flags": {
                "read_only": self.read_only,
                "idempotent": self.idempotent,
                "dangerous": self.dangerous,
                "requires_approval": self.requires_approval,
            },
            "version": self.version,
        }


# ── Transcript ───────────────────────────────────────────────────────


@dataclass
class Step:
    thought: str = ""
    action: str = ""           # tool name or "final"
    action_input: str = ""
    observation: str = ""


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    tool_calls: int = 0
    stopped_reason: str = "final"   # final | budget | error
    tool_log: list[dict] = field(default_factory=list)   # structured per-call records


# ── LLM protocol (inject real model; offline default below) ──────────


class LLM(Protocol):
    def __call__(self, prompt: str) -> str:
        ...


class RuleBasedLLM:
    """Deterministic, dependency-free planner so the ReAct loop runs offline.

    Reads the ReAct prompt, picks a tool by keyword if a useful one hasn't run
    yet, otherwise emits a Final Answer from the latest Observation. Real agents
    swap this for an actual LLM — the loop code is unchanged.
    """

    def __init__(self, tool_names: list[str]):
        self.tool_names = tool_names

    def __call__(self, prompt: str) -> str:
        # Only inspect the live scratchpad (after the Question), not the format spec.
        tail = prompt.rsplit("Question:", 1)[-1]
        observations = re.findall(r"Observation:\s*(.*)", tail)
        # Once any tool has produced a result, answer from it (avoid over-calling).
        if observations:
            return f"Thought: I can answer now.\nFinal Answer: {observations[-1]}"

        question = _last(prompt, r"Question:\s*(.*)")
        # Arithmetic intent = an operator AND a digit (so "What GPA…" isn't math).
        has_math = (
            "calc" in self.tool_names
            and any(op in question for op in "+-*/")
            and any(c.isdigit() for c in question)
        )
        if has_math:
            expr = re.sub(r"[^0-9+\-*/(). ]", "", question).strip()
            return f"Thought: this needs arithmetic.\nAction: calc\nAction Input: {expr or '0'}"
        if "delete" in question.lower() and "delete_doc" in self.tool_names:
            m = re.search(r"\bd\d+\b", question)
            return (f"Thought: this is a delete request.\nAction: delete_doc\n"
                    f"Action Input: {m.group() if m else question}")
        if "search" in self.tool_names:
            return f"Thought: I should look this up.\nAction: search\nAction Input: {question}"
        return "Thought: I can answer directly.\nFinal Answer: I don't have a tool for this."


def _last(text: str, pattern: str) -> str:
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else ""


# ── Memory ───────────────────────────────────────────────────────────


class Memory(ABC):
    @abstractmethod
    def add(self, role: str, content: str) -> None: ...
    @abstractmethod
    def context(self) -> str: ...
