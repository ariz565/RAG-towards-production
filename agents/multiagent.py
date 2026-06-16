"""Multi-agent — a supervisor that routes to specialist agents (handoff).

The orchestrator pattern: a router picks the right specialist for a task and
delegates; specialists are themselves ReAct agents with their own tools. This is
how you scale beyond one over-stuffed prompt — bounded, debuggable, and each
agent stays focused.

Routing uses an injected LLM when available, else a dependency-free keyword
router over specialist descriptions (so it runs offline).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agents.base import LLM
from agents.react import ReActAgent


@dataclass
class Specialist:
    name: str
    description: str          # what this agent is good at (used for routing)
    agent: ReActAgent
    keywords: list[str] = field(default_factory=list)


@dataclass
class SupervisorResult:
    answer: str
    chosen: str
    trace: list[str] = field(default_factory=list)


class Supervisor:
    def __init__(self, specialists: list[Specialist], router_llm: LLM | None = None):
        if not specialists:
            raise ValueError("need at least one specialist")
        self.specialists = specialists
        self.router_llm = router_llm

    def run(self, task: str) -> SupervisorResult:
        chosen = self._route(task)
        trace = [f"router → {chosen.name}"]
        result = chosen.agent.run(task)
        trace.append(f"{chosen.name}: {result.stopped_reason} ({result.tool_calls} tool calls)")
        return SupervisorResult(answer=result.answer, chosen=chosen.name, trace=trace)

    def _route(self, task: str) -> Specialist:
        if self.router_llm:
            try:
                menu = "\n".join(f"{i}: {s.name} — {s.description}"
                                 for i, s in enumerate(self.specialists))
                ans = self.router_llm(
                    f"Pick the best specialist for the task. Reply with ONLY the index.\n\n"
                    f"{menu}\n\nTask: {task}"
                )
                idx = int(re.findall(r"\d+", ans)[0])
                if 0 <= idx < len(self.specialists):
                    return self.specialists[idx]
            except Exception:
                pass
        # Offline keyword router.
        tl = task.lower()
        best, best_score = self.specialists[0], -1
        for s in self.specialists:
            score = sum(1 for kw in s.keywords if kw in tl)
            if score > best_score:
                best, best_score = s, score
        return best
