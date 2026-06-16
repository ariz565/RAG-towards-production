"""Multi-vector / late-interaction embeddings (ColBERT-style).

Instead of one vector per text, keep one vector **per token** and score with
**MaxSim**: for each query token, take its best match against any document token,
then sum. This "late interaction" captures fine-grained term matches a single
pooled vector loses — higher precision on long/nuanced docs.

Trade-off: storage + compute scale with token count. ColBERTv2 compresses token
vectors ~6–10× via residual quantization; production deployments run it on a
narrowed candidate set (retrieve cheap → rerank with MaxSim).

The per-token embedder is injectable; the default is a dependency-free hash so
this runs offline.
"""

from __future__ import annotations

import hashlib

from embeddings.base import Vector, cosine


def maxsim(query_tokens: list[Vector], doc_tokens: list[Vector]) -> float:
    """ColBERT MaxSim: sum over query tokens of the max similarity to any doc token."""
    if not query_tokens or not doc_tokens:
        return 0.0
    return sum(max(cosine(q, d) for d in doc_tokens) for q in query_tokens)


class ColBERTEmbedder:
    def __init__(self, token_embed_fn=None, dim: int = 128):
        self.dim = dim
        self.token_embed_fn = token_embed_fn or self._hash_token

    def embed(self, text: str) -> list[Vector]:
        return [self.token_embed_fn(tok) for tok in text.lower().split()]

    def score(self, query: str, document: str) -> float:
        return maxsim(self.embed(query), self.embed(document))

    # Dependency-free per-token vector (demo/testing only).
    def _hash_token(self, token: str) -> Vector:
        vec = [0.0] * self.dim
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % self.dim] = 1.0
        vec[(h >> 16) % self.dim] += 0.5
        return vec
