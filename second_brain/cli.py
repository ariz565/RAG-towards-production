"""CLI — `python -m second_brain [--backend offline|project] [--auto-approve] <command> ...`

Global flags come BEFORE the subcommand (standard argparse subparser behavior),
e.g.: `python -m second_brain --backend project ask "..."`.

Default backend is `offline` — zero setup, deterministic, no API key — so
ingest/ask/lint work immediately. Pass `--backend project` to route through
whatever this repo has configured as ACTIVE_MODEL in .env for real synthesis.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from second_brain import WORKSPACE_DIR
from second_brain.adapters import OfflineLLMAdapter, ProjectLLMAdapter
from second_brain.pipeline import SecondBrain

DEFAULT_DATA_DIR = WORKSPACE_DIR


def _build_brain(args: argparse.Namespace) -> SecondBrain:
    llm = ProjectLLMAdapter() if args.backend == "project" else OfflineLLMAdapter()
    return SecondBrain(llm, Path(args.dir), auto_approve=args.auto_approve)


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


async def _cmd_ingest(args: argparse.Namespace) -> None:
    brain = _build_brain(args)
    target = Path(args.target) if args.target else None
    _print_json(await brain.ingest(target))


async def _cmd_ask(args: argparse.Namespace) -> None:
    brain = _build_brain(args)
    result = await brain.ask(args.question)
    print("\n" + "-" * 60)
    print(result["answer"] or "(no answer text returned)")
    print("-" * 60)
    if result["nothing_to_file_reason"]:
        print(f"\n[write-back] declined: {result['nothing_to_file_reason']}")
    else:
        print(
            f"\n[write-back] {len(result['wiki_updates'])} page update(s), "
            f"{len(result['graph_links'])} link(s) proposed"
        )
    if result["retried_for_rule_6"]:
        print("[loop] house rule 6 (two outputs, always) was violated on the first pass -- retried once")
    if not args.auto_approve and (result["wiki_updates"] or result["graph_links"]):
        print("Run `python -m second_brain review list` to approve or reject.")


async def _cmd_lint(args: argparse.Namespace) -> None:
    brain = _build_brain(args)
    report = await brain.lint()
    print(report["narrative"])
    print("\nStats:", json.dumps(report["stats"], indent=2))
    print(f"\nFull report: {report['report_path']}")


def _cmd_review_list(args: argparse.Namespace) -> None:
    brain = _build_brain(args)
    pending = brain.review.list_pending()
    if not pending:
        print("Nothing pending review.")
        return
    for u in pending:
        print(f"[{u.id}] {u.kind} ({u.operation}) -- {u.rationale or '(no rationale given)'}")
        print(f"       {json.dumps(u.payload)[:200]}")


def _cmd_review_approve(args: argparse.Namespace) -> None:
    _print_json(_build_brain(args).apply_pending(args.id))


def _cmd_review_reject(args: argparse.Namespace) -> None:
    reason = " ".join(args.reason) if args.reason else "no reason given"
    _print_json(_build_brain(args).reject_pending(args.id, reason))


def _cmd_graph_html(args: argparse.Namespace) -> None:
    brain = _build_brain(args)
    out_path = Path(args.out) if args.out else None
    written = brain.export_graph_html(out_path)
    print(f"Wrote {written} -- open it directly in a browser (no server needed).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m second_brain", description=__doc__)
    parser.add_argument("--dir", default=str(DEFAULT_DATA_DIR), help="base data dir (raw/, wiki/, audit.log)")
    parser.add_argument("--backend", choices=["offline", "project"], default="offline")
    parser.add_argument(
        "--auto-approve", action="store_true",
        help="apply proposed wiki writes / graph links immediately instead of queuing them for review",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="process new files in raw/ (idempotent -- unchanged files are skipped)")
    p_ingest.add_argument("--target", help="ingest one specific file instead of the whole raw/ dir")
    p_ingest.set_defaults(func=_cmd_ingest, is_async=True)

    p_ask = sub.add_parser("ask", help="ask a question against the wiki")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=_cmd_ask, is_async=True)

    p_lint = sub.add_parser("lint", help="health-check the wiki: contradictions, orphans, broken links, stale sources")
    p_lint.set_defaults(func=_cmd_lint, is_async=True)

    p_review = sub.add_parser("review", help="manage the pending-update queue (the epistemic-integrity gate)")
    review_sub = p_review.add_subparsers(dest="review_command", required=True)

    p_review_list = review_sub.add_parser("list", help="show everything awaiting approval")
    p_review_list.set_defaults(func=_cmd_review_list, is_async=False)

    p_review_approve = review_sub.add_parser("approve", help="apply a pending update")
    p_review_approve.add_argument("id")
    p_review_approve.set_defaults(func=_cmd_review_approve, is_async=False)

    p_review_reject = review_sub.add_parser("reject", help="discard a pending update")
    p_review_reject.add_argument("id")
    p_review_reject.add_argument("reason", nargs="*")
    p_review_reject.set_defaults(func=_cmd_review_reject, is_async=False)

    p_graph = sub.add_parser("graph", help="graph exports")
    graph_sub = p_graph.add_subparsers(dest="graph_command", required=True)

    p_graph_html = graph_sub.add_parser(
        "html", help="write an interactive graph.html (vis.js via CDN, opens in any browser, no server)"
    )
    p_graph_html.add_argument("--out", help="output path (default: wiki/graph.html)")
    p_graph_html.set_defaults(func=_cmd_graph_html, is_async=False)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.is_async:
        asyncio.run(args.func(args))
    else:
        args.func(args)


if __name__ == "__main__":
    main()
