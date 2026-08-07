# Second Brain — Karpathy's LLM Wiki, from scratch

A from-scratch, zero-new-dependency implementation of Andrej Karpathy's "LLM
Wiki" pattern — a standalone lab in the same spirit as `retrieval/`, `chunking/`,
`embeddings/`: not wired into the app, pure stdlib + this repo's own
`app.services.llm` (optional), runnable and testable offline.

**Why this exists:** RAG answers a question by re-digging through raw documents
every time — nothing it learns on question 1 helps with question 50. Karpathy's
fix: have an LLM compile raw sources into a small, structured, cross-linked
markdown wiki *once*, then keep it current. Querying means reading tidy notes,
not re-searching a haystack, and every query can *improve* the notes for next
time. See `code-n-concepts/andrej-karpathy.md` and `karpathy-second-brain.md` for
the full pattern this implements, and `116-loop-engineering-...md` for a
Claude-Agent-SDK version of the same idea — this module is a different
implementation of that same idea with no SDK, no tool-calling loop, and one
structural addition file 116 doesn't have (see "Epistemic integrity" below).

## The three layers

| Layer | What it is | Who owns it |
|---|---|---|
| `workspace/raw/` | Immutable source documents | You. Never modified by this module. |
| `workspace/wiki/` | Markdown pages + a typed-edge graph (`graph.db`) | Proposed by an LLM, applied by code, approved by you (see below) |
| `schema.py` | The house rules — cite sources, one page per concept, don't silently overwrite, typed relationships not vague links, two outputs always, propose-don't-commit | Embedded in every prompt, not "hoped for" |

## The adapter pattern — the actual point of "plugin adapter style"

Every operation (`ingest`/`ask`/`lint`) talks to an `LLMPort` (`ports.py`) — a
two-method `typing.Protocol`, not a class you inherit from. Nothing in
`pipeline.py`, `prompts.py`, or the CLI knows or cares which concrete backend is
answering. `adapters.py` ships two:

- **`OfflineLLMAdapter`** — deterministic, dependency-free. One page per raw
  file, content copied verbatim (no invented synthesis), naive keyword-overlap
  retrieval for `ask`, and it **always honestly declines to write back**
  (`nothing_to_file_reason: "offline adapter performs no synthesis"`) because a
  component that can't understand text shouldn't pretend it can. This is what
  runs by default — `python -m second_brain ingest` works with zero setup, zero
  API key, zero network, the same reason `retrieval/dense.py`'s `HashEmbedder`
  or `advanced_rag`'s `FakeLLM` exist elsewhere in this repo.
- **`ProjectLLMAdapter`** — lazily imports this repo's own `app.services.llm`
  (whatever `ACTIVE_MODEL` is configured in `.env`) for real synthesis quality.
  Lazy import means `second_brain/` has **zero import-time dependency on `app/`**
  — you only pay for it if you pick `--backend project`.

Swapping providers, or plugging this into a different codebase entirely, means
writing one new class with two `async def`s. Nothing else changes. That's the
whole promise of a port/adapter boundary, made concrete instead of asserted.

## Epistemic integrity — the one thing added on top of the source material

Both `karpathy-second-brain.md`'s community section and the graphify code review
converge on the same failure mode: an LLM that both writes the knowledge base
*and* grades its own writing, with no independent check, eventually produces a
convincing-looking but untrustworthy wiki. Concretely, `graphify`'s own
`save-result`/`reflect` loop has exactly this shape — the same agent that
answers a question self-labels its answer `useful`, and after two self-marks it
becomes a "preferred source" other sessions are told to trust. File 116's
`brain.py` closes a *different* gap (verifying the write-back actually happened,
via an independent audit log) but still runs `ingest` with
`permission_mode="acceptEdits"` — proposed edits land with no review step.

This module closes both gaps the same way: **`review.py`**. Every page write and
every graph link proposed by `ingest()`/`ask()` lands in `workspace/.review/pending/`
first. Nothing touches `wiki/*.md` or `graph.db` until it's explicitly applied —
either because you ran with `--auto-approve` (a real, named decision, not a
silent default) or because you ran `review approve <id>` yourself. The LLM
proposes; your own code is the only thing that ever writes to disk, and it only
does so on an explicit signal.

## The feedback loop — HITL, deliberately not RL

`review.py`'s pending/applied/rejected history is real feedback data — every
rejection is a labeled example of `(proposal, human verdict, reason)`, which is
literally the shape RLHF/DPO training consumes. `feedback.py` uses it, but only
halfway on purpose, because the other half is exactly what graphify's own
CHANGELOG names as unsafe to build without more machinery than this project has:
*"letting verdicts influence query traversal is deliberately deferred — it needs
propensity correction + exploration to avoid a self-reinforcing feedback loop."*

Three boundaries follow directly from that caution, all enforced in code, not
just described here:

- **Generation, never retrieval.** Recent rejections are embedded in
  `ingest_prompt` and `ask_answer_prompt` (what the LLM writes) — never in
  `ask_plan_prompt` (which pages get read). `test_feedback_wiring.py` asserts
  this directly: the plan call's captured prompt never contains a lesson, the
  answer call's does.
- **Rejections only, never approvals.** A rejection carries a clear, actionable
  signal ("don't do X, here's why not"). Treating "got approved" as "was good"
  is the subtler version of the same feedback-loop risk — the first plausible
  proposal for a topic is what gets approved, not necessarily the best one.
- **No scoring, no decay, no aggregation across runs.** Just the 5 most recent
  rejections, re-read fresh every call (`feedback.DEFAULT_LIMIT`). At
  personal-wiki scale there isn't enough data for a decayed/corroboration model
  (graphify's `reflect.py`) to be signal rather than noise — and that machinery
  is exactly what makes letting it touch retrieval risky in the first place.

This is HITL with a feedback loop, not RL — nothing here updates model weights
or a reward model. If you ever want real RLHF/DPO on top of this, you'd need a
self-hosted, fine-tunable model; `ProjectLLMAdapter` calls an API-based model
you don't have weight access to, so that's a different, much bigger project.

## The graph viewer

`graph.py` is a pure data layer (SQLite + query methods) — it renders nothing.
`graph_html.py` adds an actual interactive viewer: `python -m second_brain graph
html` writes one self-contained `graph.html` you open directly in a browser, no
server. It uses `vis-network` (vis.js) loaded from a CDN *inside the browser* —
not pip-installed — pinned with a real, fetched-and-verified Subresource
Integrity hash, the same technique confirmed in graphify's own `export.py`
during our review of that codebase. Click a node to see its relationships in a
side panel (with clickable links to traverse), search by label, toggle edge
visibility per relation type, drag/zoom/pan freely. Orphan nodes (no edges) get
a dashed red border — the same signal `lint()` surfaces as text, here you can
see it directly. This is a snapshot viewer, not a live one: it renders whatever
is currently in `graph.db`, regenerated on demand, not on every ingest.

## Quick start

```bash
cd RAG-towards-production   # this module has no external deps; runs from the repo root

# ingest the two seed docs already in workspace/raw/, staged for review
python -m second_brain ingest
python -m second_brain review list
python -m second_brain review approve <id>       # or: review reject <id> "reason"

# or skip the review step entirely for a fast offline demo
python -m second_brain --auto-approve ingest
python -m second_brain --auto-approve ask "how does RRF fuse BM25 and dense retrieval?"
python -m second_brain lint

# use this repo's real configured LLM instead of the offline stub
python -m second_brain --backend project --auto-approve ingest
python -m second_brain --backend project ask "how does chunking affect hybrid retrieval?"

# open an interactive view of the graph -- no server, just open the file
python -m second_brain graph html
```

Run the tests (offline, no API key — pytest-asyncio isn't assumed to be
installed, so tests use plain `def test_x(): asyncio.run(...)`):

```bash
python -m pytest second_brain/tests -q
```

Note: `second_brain/tests` isn't in this repo's root `testpaths` (`tests/`,
`evals/`), matching every other standalone lab — run it explicitly as above.

## What you get after `ingest` + `ask` + `lint`

```
workspace/wiki/
├── index.md              # fully regenerated every run from the actual pages +
│                          # graph — never hand-maintained, so it can't go stale
│                          # (karpathy-second-brain.md community insight #10)
├── SCHEMA.md              # human-readable copy of schema.py's rules
├── log.md                 # chronological, grep-able: `grep "^## \[" log.md`
├── graph.db               # typed edges: is-a / part-of / contradicts /
│                          # supersedes / depends-on, each with a confidence
├── .ingest_ledger.json    # content-hash ledger -- re-ingesting an unchanged
│                          # raw/ file is a no-op, not duplicate synthesis
├── .review/{pending,applied,rejected}/   # the epistemic-integrity queue
├── .lint-<date>.json      # structural findings + narrative from the last lint
├── graph.html             # generated on demand by `graph html`, not by ingest
└── concepts/*.md, entities/*.md   # the actual pages, each with source-hashed
                                    # frontmatter (stale_pages() flags drift)
```

## Layout

```
second_brain/
├── ports.py        # LLMPort — the one seam everything else depends on
├── adapters.py      # OfflineLLMAdapter (default), ProjectLLMAdapter
├── schema.py         # the house rules, embedded in every prompt
├── prompts.py         # pure (system, user) builders for ingest/ask_plan/ask_answer/lint
├── graph.py            # typed-edge SQLite graph, 5 relations, confidence-scored
├── graph_html.py        # self-contained interactive graph.html (vis.js via CDN)
├── store.py             # raw/+wiki/ file I/O: sandboxed writes, source-hash
│                         # staleness, wikilink validation, the chronological log
├── review.py             # propose -> pending -> approved/rejected queue
├── feedback.py            # recent rejections -> lessons embedded in future
│                          # ingest/ask prompts (generation only, never retrieval)
├── pipeline.py             # SecondBrain: ingest()/ask()/lint(), audit log,
│                           # the "two outputs, always" verification + retry
├── cli.py + __main__.py    # `python -m second_brain ...`
├── workspace/
│   ├── raw/                # seed docs (tracked in git)
│   └── wiki/               # generated (gitignored -- see workspace/.gitignore)
└── tests/                  # 53 tests, all offline, no pytest-asyncio required
```

## Known limits (the honest checklist, file-116-style)

- **Single-writer SQLite, single-writer JSON ledger/review-queue.** Fine for one
  person running `ingest`/`ask` sequentially; not safe for concurrent writers
  without adding file locking (same limit file 116 and the karpathy-second-brain
  community's "filesystem vs. database" discussion both call out explicitly).
- **No ontology beyond 5 relation types.** Deliberate (see `graph.py`'s
  docstring) — the alternative (an LLM inventing relation verbs freely, the way
  graphify does for code) doesn't fit a personal wiki you want to be able to
  reason about from memory. It's a real ceiling if your domain genuinely needs
  more relation types.
- **`OfflineLLMAdapter` proves the mechanics, not synthesis quality.** It
  deliberately never invents a graph link or a wiki edit — that's the honest
  thing for a component with no real understanding to do, but it means the
  offline demo alone won't show you what a *good* wiki looks like. Use
  `--backend project` for that.
- **The review queue is a safety gate, not a diff viewer.** `review list` prints
  the raw proposed JSON payload, not a rendered before/after diff of the page.
  Good enough to decide "does this look plausible," not a proper code-review UI.
- **No staleness auto-repair.** `lint()` flags `stale_pages()` (a cited raw/
  source changed or vanished) but doesn't re-ingest automatically — by design,
  since silently re-synthesizing without a human noticing is exactly the failure
  mode this module exists to avoid.
- **The graph viewer renders the whole graph, every time, with no clustering or
  node-count cap.** Fine at personal-wiki scale (tens to low hundreds of pages).
  graphify's `graph.html` falls back to an aggregated meta-graph past a node-count
  threshold — this one doesn't; a very large `graph.db` would just get slow/messy
  to look at, not crash, but that's the honest ceiling. It's also a snapshot, not
  a live view — re-run `graph html` after changes, it doesn't watch for them.

## References

- `code-n-concepts/andrej-karpathy.md` — Karpathy's original post: raw/ → LLM
  compiles a wiki → Obsidian as viewer → query-time write-back.
- `code-n-concepts/karpathy-second-brain.md` — the three-layer architecture, the
  Ingest/Query/Lint operations, and the 17 community-insight critiques this
  module implements directly rather than leaves as open problems: #5 (source
  provenance via content hashing → `stale_pages()`), #9 (typed ontology →
  `graph.py`'s 5 relations), #10 (index as a query, not a hand-maintained file →
  `_regenerate_index()`), and the "epistemic integrity" critique → `review.py`.
- `code-n-concepts/116-loop-engineering-...md` — a Claude-Agent-SDK
  implementation of the same pattern; this module's `graph.py` is a direct,
  independent rewrite of that file's `graph_store.py` design (same 5 relations),
  and its `review.py` is the fix for the one gap that file's own hardening
  checklist flags but doesn't close (`ingest`'s `acceptEdits` mode).
- This repo's own `retrieval/README.md`, `chunking/README.md` — the standalone-
  lab conventions (offline-by-default, ABC/Protocol interfaces, own README +
  tests) this module follows.
