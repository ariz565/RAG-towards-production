"""CLI for one-time operations like indexing.

Commands:
  index          Build PageIndex tree (vectorless RAG)
  index-hybrid   Ingest PDF → chunks → BM25 + Qdrant hybrid index
  index-all      Build both PageIndex and Hybrid indexes
  config         Show current configuration
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from app.config import settings


# ── Helpers ─────────────────────────────────────────────────────────

def _find_pdf() -> Path:
    """Find the first PDF in data/pdf/ or exit."""
    pdf_dir = Path(settings.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"\n  No PDF files found in {pdf_dir.resolve()}/")
        print(f"  Place your admission guide PDF there and run again.\n")
        sys.exit(1)
    return pdf_files[0]


# ── Commands ────────────────────────────────────────────────────────

def index_command():
    """Build the PageIndex tree for vectorless RAG."""
    from app.services.indexer import DocumentIndex

    pdf_file = _find_pdf()
    print(f"\n  [PageIndex] Indexing: {pdf_file.name}")
    print(f"  This may take 5-10 minutes for large documents...\n")

    async def run():
        document_index = DocumentIndex()
        result = await document_index.build_index(
            pdf_path=pdf_file,
            index_dir=settings.index_path,
            model=settings.active_model_name,
            max_pages_per_node=settings.max_pages_per_node,
            max_tokens_per_node=settings.max_tokens_per_node,
            toc_check_pages=settings.toc_check_pages,
        )
        print(f"\n  Done!")
        print(f"  Document: {result['doc_name']}")
        print(f"  Pages: {result['total_pages']}")
        print(f"  Tree nodes: {result['tree_nodes']}")
        print(f"  Index saved to: {settings.index_dir}/\n")

    asyncio.run(run())


def index_hybrid_command():
    """Ingest PDF → chunks → BM25 + Qdrant hybrid index."""
    from app.services.hybrid_retrieval import HybridIndex
    from app.services.ingestion import ingest_pdf
    from app.services.registry import collection_for

    pdf_file = _find_pdf()
    print(f"\n  [Hybrid] Ingesting: {pdf_file.name}")
    print(f"  Strategy: {settings.active_retrieval.value}")
    print(f"  Embeddings: {settings.active_embedding.value}")
    print(f"  Chunk size: {settings.chunk_size} tokens, overlap: {settings.chunk_overlap}\n")

    async def run():
        hybrid_index = HybridIndex(collection_name=collection_for(pdf_file.stem))

        # Step 1: Ingest PDF → chunks
        print("  Step 1/3: Extracting text & chunking...")
        chunks = await ingest_pdf(pdf_file)
        print(f"          → {len(chunks)} chunks created")

        # Step 2: Build BM25 + vector indexes
        print("  Step 2/3: Building BM25 & vector indexes...")
        await hybrid_index.build(chunks, strategy=settings.active_retrieval)
        print(f"          → BM25 terms: {len(hybrid_index._bm25.doc_len) if hybrid_index._bm25 else 0}")
        print(f"          → Qdrant loaded: {hybrid_index._qdrant_client is not None}")

        # Step 3: Save to disk
        print("  Step 3/3: Saving indexes...")
        hybrid_index.save(pdf_file.stem)
        print(f"          → Saved to {settings.bm25_dir}/")

        print(f"\n  Done! {len(chunks)} chunks indexed for hybrid retrieval.\n")

    asyncio.run(run())


def index_all_command():
    """Build both PageIndex and Hybrid indexes."""
    print("\n  Building all indexes...\n")
    print("  ── Phase 1: PageIndex (tree) ──")
    index_command()
    print("  ── Phase 2: Hybrid (BM25 + Vector) ──")
    index_hybrid_command()
    print("  All indexes built successfully!\n")


def update_hybrid_command():
    """Incrementally re-index: embed only added/changed chunks, delete removed."""
    from app.services.hybrid_retrieval import HybridIndex
    from app.services.ingestion import ingest_pdf
    from app.services.registry import collection_for

    pdf_file = _find_pdf()
    print(f"\n  [Update] Incremental re-index: {pdf_file.name}\n")

    async def run():
        hybrid_index = HybridIndex(collection_name=collection_for(pdf_file.stem))
        if not hybrid_index.load(pdf_file.stem):
            print("  No existing hybrid index. Run 'index-hybrid' first.\n")
            sys.exit(1)
        chunks = await ingest_pdf(pdf_file)
        stats = await hybrid_index.update(chunks)
        hybrid_index.save(pdf_file.stem)
        print(f"  Done: {stats}\n")

    asyncio.run(run())


def profile_command():
    """Derive (and cache) the domain profile for the indexed document.

    Makes the guardrail / answer / redirect prompts adapt to whatever document
    is indexed instead of assuming a fixed domain.
    """
    from app.services.domain_profile import generate_profile, save_profile
    from app.services.indexer import DocumentIndex

    pdf_file = _find_pdf()
    index_file = settings.index_path / f"{pdf_file.stem}_index.json"
    if not index_file.exists():
        print(f"\n  No PageIndex found at {index_file}. Run 'index' first.\n")
        sys.exit(1)

    async def run():
        document_index = DocumentIndex()
        document_index.load_index(index_file, pdf_file)
        profile = await generate_profile(
            doc_name=document_index.doc_name or pdf_file.stem,
            sample_text=document_index.sample_text(),
        )
        if profile.is_profiled:
            save_profile(profile, pdf_file.stem)
        print(f"\n  Domain profile for: {profile.doc_name}")
        print(f"  ───────────────────────────────────────")
        print(f"  Description: {profile.description}")
        print(f"  Persona:     {profile.answer_persona}")
        print(f"  Topics:      {', '.join(profile.topics) or '(none — generic)'}")
        if profile.suggested_questions:
            print(f"  Suggested questions:")
            for q in profile.suggested_questions:
                print(f"    - {q}")
        print()

    asyncio.run(run())


def config_command():
    """Show current configuration."""
    print(f"""
  Vision Configuration
  ───────────────────────────────────────
  Active Model:      {settings.active_model.value} ({settings.active_model_name})
  Active Retrieval:  {settings.active_retrieval.value}
  Active Embedding:  {settings.active_embedding.value}
  ───────────────────────────────────────
  PDF Directory:     {settings.pdf_dir}
  Index Directory:   {settings.index_dir}
  BM25 Directory:    {settings.bm25_dir}
  Chunks Directory:  {settings.chunks_dir}
  ───────────────────────────────────────
  Chunk Size:        {settings.chunk_size} tokens
  Chunk Overlap:     {settings.chunk_overlap} tokens
  Hybrid Top-K:      {settings.retrieval_top_k}
  BM25 Top-K:        {settings.bm25_top_k}
  Vector Top-K:      {settings.vector_top_k}
  RRF K:             {settings.rrf_k}
  ───────────────────────────────────────
  Qdrant In-Memory:  {settings.qdrant_in_memory}
  Embedding Dim:     {settings.embedding_dim}
""")


# ── Main ────────────────────────────────────────────────────────────

COMMANDS = {
    "index": index_command,
    "index-hybrid": index_hybrid_command,
    "index-all": index_all_command,
    "update-hybrid": update_hybrid_command,
    "profile": profile_command,
    "config": config_command,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"\nUsage: python -m app.cli <command>")
        print(f"\nCommands:")
        print(f"  index          Build PageIndex tree (vectorless RAG)")
        print(f"  index-hybrid   Build BM25 + Qdrant hybrid index")
        print(f"  update-hybrid  Incremental re-index (embed only changed chunks)")
        print(f"  index-all      Build both indexes")
        print(f"  profile        Derive the document's domain profile")
        print(f"  config         Show current configuration\n")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()
