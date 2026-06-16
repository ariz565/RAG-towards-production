"""ReAct agent — the Thought → Action → Observation loop.

The canonical agent pattern (Yao et al., 2022): the model reasons, picks a tool,
observes the result, and repeats until it can answer — bounded by a step budget
so it can't loop forever. Every step is traced.

Production hardening this demonstrates: a hard **step budget**, tool-error
capture (the agent sees the error and can recover), and a full transcript for
observability.
"""

from __future__ import annotations

import re

from agents.base import AgentResult, LLM, Memory, RuleBasedLLM, Step, Tool
from agents.runtime import ApprovalFn, SecureToolRunner
from agents.tools import default_toolset

_PROMPT = """Answer the question. Use a tool when it helps; otherwise answer directly.

Available tools:
{tools}

Use EXACTLY this format:
Thought: <your reasoning>
Action: <one tool name from the list>
Action Input: <input to the tool>
Observation: <tool result appears here>
... (Thought/Action/Action Input/Observation may repeat) ...
Thought: <final reasoning>
Final Answer: <the answer>

{memory}Question: {question}
{scratchpad}"""


class ReActAgent:
    def __init__(self, llm: LLM | None = None, tools: list[Tool] | None = None,
                 max_steps: int = 5, memory: Memory | None = None,
                 approval_fn: ApprovalFn | None = None, auto_approve_dangerous: bool = False):
        tools = tools or default_toolset()
        self.tools = {t.name: t for t in tools}
        self.llm = llm or RuleBasedLLM(list(self.tools))
        self.max_steps = max_steps
        self.memory = memory
        # All tool calls go through the guarded runner (validation/approval/logging).
        self.runner = SecureToolRunner(
            tools, approval_fn=approval_fn, auto_approve_dangerous=auto_approve_dangerous
        )

    def run(self, question: str) -> AgentResult:
        steps: list[Step] = []
        scratchpad = ""
        tool_calls = 0
        log_start = len(self.runner.calls)
        mem = f"Prior context:\n{self.memory.context()}\n\n" if self.memory else ""
        tool_desc = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())

        for _ in range(self.max_steps):
            prompt = _PROMPT.format(tools=tool_desc, memory=mem, question=question, scratchpad=scratchpad)
            out = self.llm(prompt)
            thought = _grab(out, r"Thought:\s*(.*)")

            if "Final Answer:" in out:
                answer = _grab(out, r"Final Answer:\s*(.*)")
                steps.append(Step(thought=thought, action="final", observation=answer))
                self._remember(question, answer)
                return self._result(answer, steps, tool_calls, "final", log_start)

            action = _grab(out, r"Action:\s*(\w+)")
            action_input = _grab(out, r"Action Input:\s*(.*)")
            # Guarded execution: validation, approval, rate-limit, timeout, logging.
            res = self.runner.call(action, action_input, session_id="react")
            observation = res.output if res.ok else f"ERROR: {res.error}"
            tool_calls += 1
            steps.append(Step(thought, action, action_input, observation))
            scratchpad += (f"Thought: {thought}\nAction: {action}\n"
                           f"Action Input: {action_input}\nObservation: {observation}\n")

        # Budget exhausted — return the best available observation.
        fallback = steps[-1].observation if steps else "No answer (step budget exhausted)."
        self._remember(question, fallback)
        return self._result(fallback, steps, tool_calls, "budget", log_start)

    def _result(self, answer, steps, tool_calls, reason, log_start) -> AgentResult:
        r = AgentResult(answer, steps, tool_calls, reason)
        r.tool_log = self.runner.log_dicts()[log_start:]
        return r

    def _remember(self, question: str, answer: str) -> None:
        if self.memory:
            self.memory.add("user", question)
            self.memory.add("assistant", answer)


def _grab(text: str, pattern: str) -> str:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""
