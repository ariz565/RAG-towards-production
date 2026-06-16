"""Content-hash embedding cache (decorator over any Embedder).

Embeddings are pure functions of (model, prefix, text) — so cache them. At index
time this saves re-embedding unchanged content; at query time it makes repeated
queries free. Keyed by a hash of the namespace + text.
"""

from __future__ import annotations

import hashlib

from embeddings.base import Embedder, EmbedderConfig, Vector


class CachingEmbedder(Embedder):
    def __init__(self, inner: Embedder, namespace: str = ""):
        super().__init__(inner.config)
        self.inner = inner
        self.namespace = namespace or type(inner).__name__
        self._cache: dict[str, Vector] = {}
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        h = hashlib.sha256(f"{self.namespace}|{self.config.prefix_preset}|{text}".encode())
        return h.hexdigest()

    def _embed(self, texts: list[str]) -> list[Vector]:
        out: list[Vector] = []
        for t in texts:
            key = self._key(t)
            cached = self._cache.get(key)
            if cached is not None:
                self.hits += 1
                out.append(cached)
            else:
                self.misses += 1
                vec = self.inner._embed([t])[0]
                self._cache[key] = vec
                out.append(vec)
        return out

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
                "size": len(self._cache)}
