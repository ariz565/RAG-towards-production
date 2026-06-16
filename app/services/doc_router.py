"""Document router — pick the right document(s) for a query within a tenant.

Solves "ask across all my policies". When `/api/ask` has no `doc_id`, score the
query against each document's **domain profile** (description + topics + name):
- a **semantic** signal (cosine of query embedding vs a cached profile embedding),
- a **lexical** signal (term overlap — nails exact words like "maternity"),
blended into one score. Route to the best match; below a threshold, don't route
(the caller falls back / can ask for clarification).

Profile embeddings are cached per bundle, so routing costs one query embedding.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.config import settings
from app.services.embeddings import cosine_similarity, embed_query, embed_texts
from app.services.registry import registry

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def _lexical(query: str, text: str) -> float:
    q = set(_TOKEN.findall(query.lower()))
    if not q:
        return 0.0
    d = set(_TOKEN.findall(text.lower()))
    return len(q & d) / len(q)


@dataclass
class Route:
    doc_id: str
    score: float


class DocumentRouter:
    def __init__(self):
        self._vecs: dict[str, list[float]] = {}   # bundle.key → profile embedding

    @staticmethod
    def _profile_text(bundle) -> str:
        p = bundle.profile
        return " ".join([bundle.doc_id, p.description, " ".join(p.topics)]).strip()

    async def route(self, tenant_id: str, query: str, top_n: int = 1) -> list[Route]:
        bundles = registry.bundles_for(tenant_id)
        if not bundles:
            return []
        if len(bundles) == 1:
            return [Route(bundles[0].doc_id, 1.0)]

        # Embed any profiles we haven't cached yet.
        missing = [b for b in bundles if b.key not in self._vecs]
        if missing:
            vecs = await embed_texts([self._profile_text(b) for b in missing])
            for b, v in zip(missing, vecs):
                self._vecs[b.key] = v

        query_vec = await embed_query(query)
        routes: list[Route] = []
        for b in bundles:
            cos = cosine_similarity(query_vec, self._vecs[b.key])
            lex = _lexical(query, self._profile_text(b))
            routes.append(Route(b.doc_id, round(0.7 * cos + 0.3 * lex, 4)))

        routes.sort(key=lambda r: r.score, reverse=True)
        logger.info(f"Routing '{query[:40]}' → {[(r.doc_id, r.score) for r in routes[:top_n]]}")
        return routes[:top_n]


doc_router = DocumentRouter()
