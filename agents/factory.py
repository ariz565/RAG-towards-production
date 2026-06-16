"""Factory — build a ReAct agent or a multi-agent supervisor by flag."""

from __future__ import annotations

from agents.base import LLM, Memory
from agents.multiagent import Specialist, Supervisor
from agents.react import ReActAgent
from agents.tools import default_toolset, make_calculator, make_corpus_search, make_delete_doc


def build_react_agent(
    llm: LLM | None = None,
    max_steps: int = 5,
    memory: Memory | None = None,
    risky: bool = False,
    auto_approve_dangerous: bool = False,
) -> ReActAgent:
    tools = default_toolset()
    if risky:
        tools.append(make_delete_doc())   # adds a dangerous tool to demo HITL approval
    return ReActAgent(
        llm=llm, tools=tools, max_steps=max_steps, memory=memory,
        auto_approve_dangerous=auto_approve_dangerous,
    )


def build_supervisor(llm: LLM | None = None, router_llm: LLM | None = None) -> Supervisor:
    """A math specialist + a knowledge specialist behind a router."""
    math_agent = ReActAgent(llm=llm, tools=[make_calculator()])
    kb_agent = ReActAgent(llm=llm, tools=[make_corpus_search()])
    specialists = [
        Specialist(
            name="math", description="arithmetic, GPAs, averages, calculations",
            agent=math_agent,
            keywords=["calc", "average", "sum", "plus", "times", "multiply", "+", "*", "gpa point"],
        ),
        Specialist(
            name="knowledge", description="facts about admissions, programs, deadlines, policies",
            agent=kb_agent,
            keywords=["what", "scholarship", "program", "deadline", "housing", "toefl",
                      "admission", "gpa", "requirement", "credit"],
        ),
    ]
    return Supervisor(specialists, router_llm=router_llm)


def get_agent(kind: str, **kwargs):
    if kind == "react":
        return build_react_agent(**kwargs)
    if kind == "supervisor":
        return build_supervisor()
    raise ValueError(f"unknown agent kind: {kind} (use 'react' or 'supervisor')")
