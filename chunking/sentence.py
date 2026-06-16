"""Sentence-window chunking.

Segment into sentences, then greedily pack whole sentences up to a token budget,
overlapping by a fixed number of sentences. Never cuts mid-sentence.

When to use: prose/QA where sentence integrity matters and you want clean,
readable chunks. A solid middle ground between fixed-size and semantic.
Trade-off: still uses a size budget, so topically-distinct sentences can share a
chunk.
"""

from __future__ import annotations

from chunking.base import Chunk, Chunker, count_tokens, split_sentences


class SentenceChunker(Chunker):
    def __init__(self, max_tokens: int = 256, sentence_overlap: int = 1):
        if sentence_overlap < 0:
            raise ValueError("sentence_overlap must be >= 0")
        self.max_tokens = max_tokens
        self.sentence_overlap = sentence_overlap

    def chunk(self, text: str, *, metadata: dict | None = None) -> list[Chunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            t = count_tokens(sent)
            if current and current_tokens + t > self.max_tokens:
                chunks.append(self._emit(current, len(chunks), metadata))
                # Carry the last N sentences forward as overlap.
                current = current[-self.sentence_overlap :] if self.sentence_overlap else []
                current_tokens = sum(count_tokens(s) for s in current)
            current.append(sent)
            current_tokens += t

        if current:
            chunks.append(self._emit(current, len(chunks), metadata))
        return chunks

    def _emit(self, sentences: list[str], index: int, metadata: dict | None) -> Chunk:
        return Chunk(
            text=" ".join(sentences).strip(),
            index=index,
            metadata={**(metadata or {}), "strategy": "sentence", "sentences": len(sentences)},
        )
