"""Late chunking (Jina "chunked pooling") — context-preserving chunk embeddings.

Conventional ("early") chunking splits text, THEN embeds each chunk in isolation
— so a chunk loses the surrounding context (pronouns, references, topic). Late
chunking flips the order: embed the WHOLE document with a long-context model to
get token embeddings (every token attends to every other), THEN split the token
sequence into chunks and mean-pool each span. Each chunk vector therefore carries
full-document context.

When to use: documents that fit a long-context embedder (≤8k tokens for
jina-v2) where cross-chunk context loss hurts retrieval (anaphora, definitions).
Trade-off: needs a token-level / long-context embedding model. The token
embedder is injected so this stays decoupled and testable.

Output: each Chunk carries its pooled vector in ``metadata["embedding"]``.
"""

from __future__ import annotations

from collections.abc import Callable

from chunking.base import Chunk, Chunker

# token_embed_fn(text) -> (offsets, token_vectors)
#   offsets:       list[(start_char, end_char)] per token
#   token_vectors: list[list[float]] aligned with offsets
TokenEmbedFn = Callable[[str], tuple[list[tuple[int, int]], list[list[float]]]]

_MODEL = None
_TOKENIZER = None


class LateChunker(Chunker):
    def __init__(
        self,
        chunk_tokens: int = 256,
        model_name: str = "jinaai/jina-embeddings-v2-base-en",
        token_embed_fn: TokenEmbedFn | None = None,
    ):
        self.chunk_tokens = chunk_tokens
        self.model_name = model_name
        self.token_embed_fn = token_embed_fn or self._default_token_embed

    def chunk(self, text: str, *, metadata: dict | None = None) -> list[Chunk]:
        import numpy as np

        text = text.strip()
        if not text:
            return []

        offsets, token_vectors = self.token_embed_fn(text)
        vectors = np.array(token_vectors, dtype=float)
        if vectors.size == 0:
            return []

        chunks: list[Chunk] = []
        # Split the encoded token sequence into fixed spans, then mean-pool each.
        for i, start in enumerate(range(0, len(offsets), self.chunk_tokens)):
            span = offsets[start : start + self.chunk_tokens]
            span_vecs = vectors[start : start + self.chunk_tokens]
            if not span or span_vecs.size == 0:
                break
            char_start = span[0][0]
            char_end = span[-1][1]
            pooled = span_vecs.mean(axis=0)
            chunks.append(Chunk(
                text=text[char_start:char_end].strip(),
                index=i,
                start_char=char_start,
                end_char=char_end,
                metadata={
                    **(metadata or {}),
                    "strategy": "late",
                    "embedding": pooled.tolist(),  # context-aware chunk vector
                },
            ))
        return chunks

    # Default token embedder via transformers (long-context model, offset mapping).
    def _default_token_embed(self, text):
        global _MODEL, _TOKENIZER
        import torch
        from transformers import AutoModel, AutoTokenizer

        if _MODEL is None:
            _TOKENIZER = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            _MODEL = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            _MODEL.eval()

        enc = _TOKENIZER(
            text, return_offsets_mapping=True, return_tensors="pt", truncation=True
        )
        offset_mapping = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            token_embeddings = _MODEL(**enc).last_hidden_state[0]  # [T, d]

        # Drop special tokens (offset (0,0)) so spans map to real characters.
        offsets, vectors = [], []
        for (s, e), vec in zip(offset_mapping, token_embeddings):
            if e > s:
                offsets.append((int(s), int(e)))
                vectors.append(vec.tolist())
        return offsets, vectors
