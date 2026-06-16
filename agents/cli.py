"""Flag-driven agents demo (offline via the dependency-free RuleBasedLLM).

Examples:
    python -m agents.cli --agent react --task "What is 3 * (4 + 1)?"
    python -m agents.cli --agent react --task "What GPA do I need for computer science?"
    python -m agents.cli --agent supervisor --task "How many credits does the CS degree need?"
"""

from __future__ import annotations

import argparse
import sys

from agents.factory import get_agent
from agents.multiagent import Supervisor


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Agents demo")
    p.add_argument("--agent", default="react", choices=["react", "supervisor"])
    p.add_argument("--task", default="What is the minimum GPA for computer science?")
    p.add_argument("--max-steps", type=int, default=5, dest="max_steps")
    p.add_argument("--risky", action="store_true", help="Include the dangerous delete_doc tool")
    p.add_argument("--approve", action="store_true", help="Auto-approve dangerous tools (HITL bypass)")
    p.add_argument("--show-log", action="store_true", help="Print the structured tool-call log")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.agent == "react":
        agent = get_agent("react", max_steps=args.max_steps, risky=args.risky,
                          auto_approve_dangerous=args.approve)
    else:
        agent = get_agent("supervisor")

    print(f"\n=== agent={args.agent} ===\ntask: {args.task}\n")

    if isinstance(agent, Supervisor):
        result = agent.run(args.task)
        print(f"route: {result.chosen}")
        for t in result.trace:
            print(f"  · {t}")
        print(f"\nanswer: {result.answer}")
        return 0

    result = agent.run(args.task)
    for i, s in enumerate(result.steps, 1):
        if s.action == "final":
            print(f"  [{i}] Thought: {s.thought}\n      Final: {s.observation}")
        else:
            print(f"  [{i}] Thought: {s.thought}\n      Action: {s.action}({s.action_input})"
                  f"\n      Obs: {s.observation[:90]}")
    print(f"\nanswer: {result.answer}")
    print(f"(stopped: {result.stopped_reason}, tool calls: {result.tool_calls})")

    if args.show_log:
        print("\n-- structured tool-call log --")
        for c in result.tool_log:
            flag = "BLOCKED" if c["blocked"] else ("ok" if c["ok"] else "err")
            extra = f" reason={c['reason']}" if c["reason"] else ""
            inj = f" injection={c['output_injection']}" if c["output_injection"] else ""
            print(f"  [{c['call_id']}] {c['tool']:<11} {flag:<8} {c['latency_ms']}ms{extra}{inj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
