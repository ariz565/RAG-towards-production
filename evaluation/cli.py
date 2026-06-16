"""Flag-driven RAG evaluation demo (RAGAS-style).

Offline by default (lexical/embedding fallbacks). Pass --use-llm to run the true
LLM-judged metrics (needs OPENAI_API_KEY).

Examples:
    python -m evaluation.cli                       # built-in faithful vs unfaithful demo
    python -m evaluation.cli --use-llm
    python -m evaluation.cli --question "..." --answer "..." --context "..." --ground-truth "..."
"""

from __future__ import annotations

import argparse
import sys

from evaluation.base import Sample
from evaluation.evaluate import evaluate
from evaluation.factory import default_suite

# A faithful sample and a hallucinated one — the contrast the metrics should catch.
_DEMO = [
    Sample(
        question="What is the minimum GPA for the computer science program?",
        answer="The Computer Science program requires a minimum GPA of 3.2.",
        contexts=["The Computer Science program requires a minimum GPA of 3.2 for admission."],
        ground_truth="A minimum GPA of 3.2 is required for Computer Science.",
    ),
    Sample(
        question="What is the minimum GPA for the computer science program?",
        answer="Computer Science requires a 3.8 GPA and a personal essay about robotics.",
        contexts=["The Computer Science program requires a minimum GPA of 3.2 for admission."],
        ground_truth="A minimum GPA of 3.2 is required for Computer Science.",
    ),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RAG evaluation (RAGAS-style)")
    p.add_argument("--question")
    p.add_argument("--answer")
    p.add_argument("--context", action="append", help="Retrieved context (repeatable)")
    p.add_argument("--ground-truth", dest="ground_truth")
    p.add_argument("--use-llm", action="store_true", help="Use LLM-judged metrics (needs OPENAI_API_KEY)")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    llm_fn = None
    if args.use_llm:
        from chunking.llm_util import default_llm_fn
        llm_fn = default_llm_fn

    if args.question and args.answer:
        samples = [Sample(args.question, args.answer, args.context or [], args.ground_truth)]
    else:
        samples = _DEMO
        print("(no --question/--answer given → running the built-in faithful-vs-hallucinated demo)")

    metrics = default_suite(llm_fn=llm_fn)
    report = evaluate(samples, metrics)

    names = [m.name for m in metrics]
    print(f"\nmode={'LLM-judged' if llm_fn else 'offline (lexical/embedding fallback)'}  samples={report['n']}\n")
    print("sample  " + "".join(n[:18].ljust(20) for n in names))
    print("-" * (8 + 20 * len(names)))
    for i, row in enumerate(report["per_sample"], 1):
        print(f"{i:<8}" + "".join(str(row[n]).ljust(20) for n in names))
    print("\naggregate: " + "  ".join(f"{n}={report['aggregate'][n]}" for n in names) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
