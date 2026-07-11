"""Layer 3 — the schema. Karpathy's CLAUDE.md/AGENTS.md, made concrete for a
non-agentic, library-call architecture.

In Claude Code / the Claude Agent SDK, a schema file like CLAUDE.md works because
the *runtime* auto-loads it into every turn. This module has no such runtime — it's
a plain async function calling an LLMPort — so the schema can't be "a file the model
happens to read." Instead SCHEMA_RULES is a plain string, embedded verbatim into
every system prompt built in prompts.py. The rule is present on every single call,
not contingent on the model deciding to go read a file.

`render_schema_doc()` produces a *human-readable copy* written into wiki/SCHEMA.md
once, purely for the person browsing the wiki in Obsidian/VS Code — it documents
the rules, it is never re-read by the pipeline itself.
"""

from __future__ import annotations

SCHEMA_RULES = """\
## House rules (non-negotiable)

1. SOURCE OF TRUTH: raw/ is immutable. Never invent facts that don't trace back to
   a raw/ source. If you don't have enough information, say `[TODO: needs a source]`
   rather than filling the gap with something plausible-sounding.
2. ONE PAGE PER CONCEPT/ENTITY. Before proposing a new page, check whether the idea
   already fits inside an existing one (the index you were given lists everything
   that exists).
3. DON'T SILENTLY OVERWRITE. If new content supersedes or contradicts something
   already in the wiki, say so explicitly in the page content and (for genuine
   contradictions between sources) create a `contradicts` graph link — don't just
   quietly replace the old claim.
4. TYPED RELATIONSHIPS, NOT VAGUE LINKS. Whenever two concepts relate as one of
   is-a / part-of / contradicts / supersedes / depends-on, propose it as a graph
   link with your confidence (0.0-1.0), not just prose. Confidence should reflect
   how directly the source supports the relationship — 1.0 only for something
   stated near-verbatim, lower for something you inferred.
5. CITE YOUR SOURCES. Every page must be traceable to the raw/ file(s) it came from.
6. TWO OUTPUTS, ALWAYS (ask only). Every answer must be paired with either a wiki
   update capturing genuinely new synthesis, or an explicit, honest statement of
   why nothing was worth filing. Silence on this point is not acceptable.
7. YOU PROPOSE, YOU DON'T COMMIT. Nothing you write is applied to the wiki directly
   — every page write and every graph link you propose goes into a review queue.
   A human (or an explicit --auto-approve run) decides whether it lands. This is
   deliberate: the same process that answers a question should not also be the
   sole judge of whether its own answer was correct.
"""


def render_schema_doc() -> str:
    """Human-readable copy for wiki/SCHEMA.md — not re-read by the pipeline."""
    return (
        "# Second Brain — Schema\n\n"
        "> This file documents the rules the pipeline enforces on every ingest/ask "
        "call (see `second_brain/schema.py`). It is written once for humans browsing "
        "the wiki; the pipeline itself embeds these rules directly into its prompts, "
        "it does not re-read this file.\n\n"
        f"{SCHEMA_RULES}\n"
        "## Layers\n\n"
        "1. **raw/** — immutable source documents.\n"
        "2. **wiki/** — LLM-*proposed*, human-*approved* markdown (this layer).\n"
        "3. **This schema** — the house rules, embedded in every prompt.\n"
    )
