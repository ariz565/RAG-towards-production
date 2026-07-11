"""The one seam the whole module depends on — a Port in the Ports & Adapters sense.

Every other file in second_brain/ (pipeline.py, prompts.py's callers) talks to an
`LLMPort`, never to a concrete provider. Swapping providers means writing a new
adapter in adapters.py — nothing else in the module changes. This is what "plugin
adapter style" means concretely: the core logic (ingest/ask/lint, the graph, the
review queue) has zero knowledge of *how* completions happen, only that something
answers `complete_json`/`complete_text`.

`typing.Protocol` (stdlib, PEP 544) gives us structural typing — an adapter doesn't
need to inherit from anything, it just needs the right methods. No ABC, no registry,
no new dependency.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMPort(Protocol):
    """What second_brain needs from *any* LLM backend."""

    async def complete_json(self, system: str, user: str) -> dict:
        """Return a parsed JSON object. Implementations must never raise on malformed
        model output — return {} (or a best-effort partial dict) and let the caller
        decide how to handle an empty/incomplete response."""
        ...

    async def complete_text(self, system: str, user: str) -> str:
        """Return plain text. Used for the optional narrative pass in lint()."""
        ...

    @property
    def name(self) -> str:
        """Human-readable backend id, stamped into audit.log entries."""
        ...
