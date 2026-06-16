"""Dense single-vector embedders (one vector per text).

Backends share the `Embedder` interface (prefix → embed → MRL → normalize):
- **HashEmbedder**  — deterministic feature-hashing, ZERO dependencies. Not for
  production semantics, but lets the whole lab (MRL, quantization, similarity,
  caching) run offline and be unit-tested.
- **SentenceTransformerEmbedder** — local OSS models (BGE, E5, Nomic, GTE…).
- **OpenAIEmbedder** — text-embedding-3-small/large (also supports server-side
  `dimensions` for MRL; here we truncate uniformly in post-processing).
- **OllamaEmbedder** — local models (nomic-embed-text, mxbai…).

Heavy SDKs are imported lazily, so `import embeddings` needs nothing installed.
"""

from __future__ import annotations

import hashlib

from embeddings.base import Embedder, EmbedderConfig, Vector


class HashEmbedder(Embedder):
    """Dependency-free feature-hashing embedder (deterministic, offline-friendly).

    Hashes each token into a fixed-dim bag-of-words vector with signed buckets.
    Shared vocabulary → higher cosine, which is enough to demo the pipeline.
    """

    def __init__(self, config: EmbedderConfig | None = None, hash_dim: int = 256):
        super().__init__(config)
        self.hash_dim = hash_dim

    def _embed(self, texts: list[str]) -> list[Vector]:
        return [self._hash_one(t) for t in texts]

    def _hash_one(self, text: str) -> Vector:
        vec = [0.0] * self.hash_dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self.hash_dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        return vec

    def _native_dim(self) -> int:
        return self.hash_dim


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, config: EmbedderConfig | None = None,
                 model_name: str = "BAAI/bge-small-en-v1.5"):
        super().__init__(config)
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts: list[str]) -> list[Vector]:
        # normalize=False here; our post-processing handles MRL-truncate + L2.
        return self._load().encode(texts, normalize_embeddings=False).tolist()


class OpenAIEmbedder(Embedder):
    def __init__(self, config: EmbedderConfig | None = None,
                 model_name: str = "text-embedding-3-small"):
        super().__init__(config)
        self.model_name = model_name
        self._client = None

    def _embed(self, texts: list[str]) -> list[Vector]:
        from openai import OpenAI
        if self._client is None:
            self._client = OpenAI()
        resp = self._client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in resp.data]


class OllamaEmbedder(Embedder):
    def __init__(self, config: EmbedderConfig | None = None,
                 model_name: str = "nomic-embed-text", host: str = "http://localhost:11434"):
        super().__init__(config)
        self.model_name = model_name
        self.host = host

    def _embed(self, texts: list[str]) -> list[Vector]:
        import ollama
        client = ollama.Client(host=self.host)
        out: list[Vector] = []
        for t in texts:
            resp = client.embed(model=self.model_name, input=t)
            out.append(resp["embeddings"][0])
        return out
