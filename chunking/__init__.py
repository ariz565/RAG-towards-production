"""Chunking lab — production-grade implementations of every chunking strategy.

Standalone study/reference package (not wired into the app). Pick a strategy by
flag via `get_chunker`, or import a specific chunker directly.

Strategies: fixed · recursive · sentence · semantic · parent_child · markdown
"""

from chunking.agentic import AgenticChunker
from chunking.base import Chunk, Chunker, count_tokens, split_sentences, stats
from chunking.contextual import ContextualChunker
from chunking.factory import ChunkingStrategy, get_chunker
from chunking.fixed import FixedSizeChunker
from chunking.late import LateChunker
from chunking.markdown import MarkdownHeaderChunker
from chunking.parent_child import ParentChildChunker
from chunking.proposition import PropositionChunker
from chunking.recursive import RecursiveChunker
from chunking.semantic import SemanticChunker
from chunking.sentence import SentenceChunker

__all__ = [
    "Chunk",
    "Chunker",
    "ChunkingStrategy",
    "get_chunker",
    "count_tokens",
    "split_sentences",
    "stats",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SentenceChunker",
    "SemanticChunker",
    "ParentChildChunker",
    "MarkdownHeaderChunker",
    "LateChunker",
    "ContextualChunker",
    "PropositionChunker",
    "AgenticChunker",
]
