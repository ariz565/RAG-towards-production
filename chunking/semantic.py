"""Semantic chunking (embedding breakpoints — the "smart" strategy).

Idea (Greg Kamradt's method): embed each sentence (optionally with a small
neighbour buffer for stability), measure cosine *distance* between consecutive
sentences, and split where the distance spikes above a percentile threshold —
i.e. where the topic shifts. Chunks then align to meaning, not to a fixed size.

When to use: heterogeneous docs where topic boundaries matter more than uniform
size (research papers, mixed knowledge bases) and retrieval precision is critical.
Trade-off: cost + latency (an embedding per sentence at index time) and a tuning
knob (the percentile). The embedding fn is injected so this stays decoupled and
testable.
"""

from __future__ import annotations

from collections.abc import Callable

from chunking.base import Chunk, Chunker, split_sentences

EmbedFn = Callable[[list[str]], list[list[float]]]

_ST_MODEL = None


def default_embed_fn(texts: list[str]) -> list[list[float]]:
    """Local sentence-transformers embedder (lazy). Replace via injection in prod."""
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL.encode(texts, normalize_embeddings=True).tolist()


class SemanticChunker(Chunker):
    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        breakpoint_percentile: float = 90.0,
        buffer_size: int = 1,
        min_sentences: int = 1,
    ):
        self.embed_fn = embed_fn or default_embed_fn
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size            # neighbour context per sentence
        self.min_sentences = min_sentences

    def chunk(self, text: str, *, metadata: dict | None = None) -> list[Chunk]:
        import numpy as np  # lazy: only the semantic strategy needs numpy

        sentences = split_sentences(text)
        if len(sentences) <= self.min_sentences:
            return [Chunk(text=text.strip(), index=0,
                          metadata={**(metadata or {}), "strategy": "semantic"})] if text.strip() else []

        # Embed each sentence with a small neighbour window for stabler vectors.
        windows = [self._window(sentences, i) for i in range(len(sentences))]
        vectors = np.array(self.embed_fn(windows), dtype=float)

        distances = self._consecutive_distances(vectors)
        if distances.size == 0:
            return [Chunk(text=text.strip(), index=0,
                          metadata={**(metadata or {}), "strategy": "semantic"})]

        # Split at sentence boundaries whose semantic distance exceeds the percentile.
        threshold = float(np.percentile(distances, self.breakpoint_percentile))
        breakpoints = [i for i, d in enumerate(distances) if d > threshold]

        chunks: list[Chunk] = []
        start = 0
        for idx, bp in enumerate([*breakpoints, len(sentences) - 1]):
            group = sentences[start : bp + 1]
            if group:
                chunks.append(Chunk(
                    text=" ".join(group).strip(),
                    index=len(chunks),
                    metadata={**(metadata or {}), "strategy": "semantic", "sentences": len(group)},
                ))
            start = bp + 1
        return chunks

    def _window(self, sentences: list[str], i: int) -> str:
        lo = max(0, i - self.buffer_size)
        hi = min(len(sentences), i + self.buffer_size + 1)
        return " ".join(sentences[lo:hi])

    @staticmethod
    def _consecutive_distances(vectors):
        import numpy as np

        # cosine distance = 1 - cosine similarity (vectors assumed L2-normalized)
        a, b = vectors[:-1], vectors[1:]
        sims = np.sum(a * b, axis=1) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
        )
        return 1.0 - sims
