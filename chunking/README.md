# Chunking Lab

Production-grade, interchangeable implementations of every common chunking
strategy — a **standalone reference** (not wired into the app). Pick one by flag.

> Why chunking is the first thing to get right in RAG: retrieval quality is
> capped by chunk quality. Too big → imprecise matches + wasted context + cost.
> Too small → matches lose the context needed to answer. The strategy is a
> **recall/precision/cost trade-off**, chosen per document type.

## Strategies

| Strategy | Tier | How it works | Best for | Trade-off |
|---|---|---|---|---|
| **fixed** | classical | Sliding window of N tokens, step `N − overlap` | Uniform/unstructured text; predictable budgets | Blind to meaning — cuts mid-idea |
| **recursive** | classical | Separator hierarchy (¶→line→sentence→word), merge to budget + overlap | **Sensible default** for most docs | Structure-agnostic (no headings/tables) |
| **sentence** | classical | Pack whole sentences to a token budget, overlap by N sentences | Prose/QA where sentence integrity matters | Topically-mixed sentences can share a chunk |
| **markdown** | classical | Split on header hierarchy, keep header path, size-bound sections | Structured docs/wikis/manuals | Only as good as the document structure |
| **parent_child** | classical | Small children to retrieve, large parents for context ("small-to-big") | Serious RAG where naive chunks are too small to answer / too big to match | Two granularities + parent lookup |
| **semantic** | embedding | Embed sentences, split where cosine **distance** spikes (topic shift) | Heterogeneous docs; precision-critical retrieval | Embedding cost + a tuning knob |
| **late** | embedding | Embed the **whole doc** (long-context), then mean-pool token spans → each chunk vector carries full-doc context | Anaphora/definitions where cross-chunk context loss hurts; docs ≤ model window | Needs a token-level long-context embedder |
| **contextual** | LLM | LLM prepends a doc-aware context to each base chunk before embedding (Anthropic) | High-value corpora; −49% retrieval failures (−67% + rerank) | One LLM call per chunk at index time |
| **proposition** | LLM | LLM decomposes text into atomic, self-contained facts (Dense X Retrieval) | Fact-dense specs/policies/FAQs; pinpoint retrieval | LLM cost; tiny chunks can lose narrative |
| **agentic** | LLM | LLM-as-editor groups propositions into coherent chunks | High-stakes, topically-uneven corpora (legal/clinical) | Most expensive — an LLM call per proposition |

## Quick start

```bash
cd agent-backend/admission-guide

python -m chunking.cli --strategy recursive --chunk-size 256 --overlap 32
python -m chunking.cli --strategy sentence  --max-tokens 200
python -m chunking.cli --strategy markdown  --file README.md
python -m chunking.cli --strategy parent_child --parent-size 1024 --child-size 256
python -m chunking.cli --strategy semantic  --percentile 90      # needs sentence-transformers
```

Programmatic:

```python
from chunking import get_chunker, stats

chunker = get_chunker("recursive", chunk_size=256, chunk_overlap=32)
chunks = chunker.chunk(my_text)
print(stats(chunks))
```

## Notes for the interview

- **Budgets are in tokens, not characters** — sizing uses `tiktoken` (`cl100k_base`)
  with a graceful word-heuristic fallback.
- **Overlap** preserves continuity across boundaries so an answer split across two
  chunks is still retrievable; cost is duplicated tokens. Typical: 10–20% of size.
- **Semantic chunking** (Greg Kamradt's method): embed each sentence (with a small
  neighbour *buffer* for stability), take cosine distance between neighbours, and
  break at the high-percentile spikes (topic shifts). The embedding fn is
  **dependency-injected** so it's testable and provider-agnostic.
- **Parent-child / small-to-big** is the highest-leverage upgrade for answer
  quality: embed + retrieve on small children, then feed the matched **parent** to
  the LLM (LangChain `ParentDocumentRetriever`, LlamaIndex auto-merging).
- **Evaluate, don't guess** — the right chunking is the one that wins on your
  retrieval evals (recall@k, context precision), not the one that sounds best.

### Decision cheat-sheet
- Default → **recursive** (512 tok, ~15% overlap, token-accurate).
- Answers need more context than the match → **parent_child**.
- Topic-heterogeneous corpus → **semantic** or **late** (late if it fits a long-context embedder).
- Cross-chunk context loss (pronouns/definitions) → **late** or **contextual**.
- Fact-dense, pinpoint retrieval → **proposition**.
- High-stakes, cost-tolerant, uneven density → **agentic**.
- Structured docs → **markdown** (+ layout-aware parsing upstream).

## References
- Jina — Late Chunking: https://jina.ai/news/late-chunking-in-long-context-embedding-models/ (paper: arXiv:2409.04701)
- Anthropic — Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- Chen et al. — Dense X Retrieval (propositions): https://arxiv.org/abs/2312.06648
- Best chunking strategies 2026 (Firecrawl): https://www.firecrawl.dev/blog/best-chunking-strategies-rag
- Weaviate / Pinecone chunking guides; Unstructured.io (layout-aware parsing).

## Layout

```
chunking/
├── base.py           # Chunk model, Chunker ABC, tiktoken counter, sentence splitter
├── fixed.py          # FixedSizeChunker (sliding window)
├── recursive.py      # RecursiveChunker (separator hierarchy + merge)
├── sentence.py       # SentenceChunker (sentence-window)
├── markdown.py       # MarkdownHeaderChunker (structure-aware)
├── parent_child.py   # ParentChildChunker (small-to-big)
├── semantic.py       # SemanticChunker (embedding breakpoints; injectable embed fn)
├── late.py           # LateChunker (Jina chunked pooling; injectable token embedder)
├── contextual.py     # ContextualChunker (Anthropic; decorator over any base chunker)
├── proposition.py    # PropositionChunker (atomic facts; injectable LLM fn)
├── agentic.py        # AgenticChunker (LLM-as-editor; injectable LLM fn)
├── llm_util.py       # shared injectable LLM caller (lazy OpenAI default)
├── factory.py        # ChunkingStrategy enum + get_chunker()
├── cli.py            # flag-driven demo
└── benchmark.py      # side-by-side strategy comparison
```

> **Not implemented here — the parsing layer.** *Layout-aware* chunking
> (multi-column PDFs, tables, reading order) is an **ingestion** concern handled
> by parsers like **Unstructured.io / LlamaParse / Document AI**, not a text
> splitter. ~80% of RAG failures trace to ingestion+chunking, and tables should
> be extracted and embedded **separately** from prose. Treat that as the step
> *before* these strategies.
