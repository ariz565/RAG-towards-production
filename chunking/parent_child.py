"""Parent-child / "small-to-big" chunking (the retrieval-quality power move).

Build LARGE parent chunks for context and SMALL child chunks for precise
retrieval. You embed + search the children, then return their parents to the LLM
— small chunks match the query sharply, big chunks give the model enough context
to answer. (a.k.a. LangChain's ParentDocumentRetriever / LlamaIndex auto-merging.)

When to use: when naive chunks are either too small to answer from or too big to
match precisely — i.e. most serious RAG systems.
Trade-off: two granularities to store + a parent-lookup at query time.
"""

from __future__ import annotations

from chunking.base import Chunk, Chunker
from chunking.recursive import RecursiveChunker


class ParentChildChunker(Chunker):
    def __init__(
        self,
        parent_chunk_size: int = 2048,
        child_chunk_size: int = 384,
        child_overlap: int = 64,
        parent_chunker: Chunker | None = None,
        child_chunker: Chunker | None = None,
    ):
        self.parent_chunker = parent_chunker or RecursiveChunker(
            chunk_size=parent_chunk_size, chunk_overlap=0
        )
        self.child_chunker = child_chunker or RecursiveChunker(
            chunk_size=child_chunk_size, chunk_overlap=child_overlap
        )

    def chunk_hierarchy(
        self, text: str, *, metadata: dict | None = None
    ) -> tuple[list[Chunk], list[Chunk]]:
        """Return (parents, children) with children linked to their parent."""
        parents = self.parent_chunker.chunk(text, metadata=metadata)
        children: list[Chunk] = []
        for p_index, parent in enumerate(parents):
            parent.id = f"parent-{p_index}"
            parent.metadata = {**parent.metadata, "strategy": "parent_child", "role": "parent"}
            for child in self.child_chunker.chunk(parent.text, metadata=metadata):
                child.index = len(children)
                child.id = f"child-{child.index}"
                child.parent_id = parent.id
                child.metadata = {
                    **child.metadata,
                    "strategy": "parent_child",
                    "role": "child",
                    "parent_id": parent.id,
                    "parent_text": parent.text,  # what you'd feed the LLM after a child hit
                }
                children.append(child)
        return parents, children

    def chunk(self, text: str, *, metadata: dict | None = None) -> list[Chunk]:
        # Interface returns the CHILD chunks — the units you embed and retrieve on.
        _, children = self.chunk_hierarchy(text, metadata=metadata)
        return children
