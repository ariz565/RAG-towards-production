"""Dense retrieval — semantic (bi-encoder) search.

Embeds the corpus once, embeds the query, ranks by cosine. Catches paraphrases
and synonyms that BM25 misses ("car" ~ "automobile"). Reuses the embeddings lab,
so it runs offline via the HashEmbedder and swaps to a real model trivially.
"""

from __future__ import annotations

from embeddings.base import Embedder, EmbedderConfig, cosine
from embeddings.dense import HashEmbedder

from retrieval.base import Doc, Retriever, RetrievedDoc


class DenseRetriever(Retriever):
    def __init__(self, embedder: Embedder | None = None):
        # Default = offline hash embedder (normalized so cosine == dot).
        self.embedder = embedder or HashEmbedder(EmbedderConfig(normalize=True))
        self._docs: list[Doc] = []
        self._vectors: list[list[float]] = []

    def index(self, docs: list[Doc]) -> None:
        self._docs = docs
        self._vectors = self.embedder.embed_documents([d.text for d in docs])

    def search(self, query: str, k: int = 10) -> list[RetrievedDoc]:
        qv = self.embedder.embed_query(query)
        scored = [
            (i, cosine(qv, v)) for i, v in enumerate(self._vectors)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievedDoc(self._docs[i].id, self._docs[i].text, s, rank + 1, "dense")
            for rank, (i, s) in enumerate(scored[:k])
        ]
