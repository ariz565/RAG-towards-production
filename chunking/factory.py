"""Strategy registry + factory — pick a chunker by flag.

    chunker = get_chunker("semantic", breakpoint_percentile=92)
    chunks  = chunker.chunk(text)
"""

from __future__ import annotations

from enum import Enum

from chunking.agentic import AgenticChunker
from chunking.base import Chunker
from chunking.contextual import ContextualChunker
from chunking.fixed import FixedSizeChunker
from chunking.late import LateChunker
from chunking.markdown import MarkdownHeaderChunker
from chunking.parent_child import ParentChildChunker
from chunking.proposition import PropositionChunker
from chunking.recursive import RecursiveChunker
from chunking.semantic import SemanticChunker
from chunking.sentence import SentenceChunker


class ChunkingStrategy(str, Enum):
    # Classical (deterministic, no external deps)
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    MARKDOWN = "markdown"
    PARENT_CHILD = "parent_child"
    # Embedding-driven
    SEMANTIC = "semantic"
    LATE = "late"
    # LLM-driven (index-time, higher cost)
    CONTEXTUAL = "contextual"
    PROPOSITION = "proposition"
    AGENTIC = "agentic"


_REGISTRY: dict[ChunkingStrategy, type[Chunker]] = {
    ChunkingStrategy.FIXED: FixedSizeChunker,
    ChunkingStrategy.RECURSIVE: RecursiveChunker,
    ChunkingStrategy.SENTENCE: SentenceChunker,
    ChunkingStrategy.MARKDOWN: MarkdownHeaderChunker,
    ChunkingStrategy.PARENT_CHILD: ParentChildChunker,
    ChunkingStrategy.SEMANTIC: SemanticChunker,
    ChunkingStrategy.LATE: LateChunker,
    ChunkingStrategy.CONTEXTUAL: ContextualChunker,
    ChunkingStrategy.PROPOSITION: PropositionChunker,
    ChunkingStrategy.AGENTIC: AgenticChunker,
}


def get_chunker(strategy: str | ChunkingStrategy, **kwargs) -> Chunker:
    """Construct a chunker for `strategy`, forwarding strategy-specific kwargs."""
    strategy = ChunkingStrategy(strategy)
    return _REGISTRY[strategy](**kwargs)
