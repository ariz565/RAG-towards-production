"""Domain Profile — makes the pipeline document-agnostic (per-document).

Instead of hardcoding "university admission guide" assumptions into the
guardrail, answer, and redirect prompts, we derive a lightweight profile of
each indexed document:

    - description:          one paragraph on what the document covers
    - topics:               in-scope subject areas (used by the guardrail)
    - persona:              how the answerer should introduce itself
    - suggested_questions:  good starter questions (used by the UI)

Phase C: profiles are per-document (owned by a DocumentBundle), not a global
singleton. The profile is generated ONCE via the centralized LLM (cheap, from a
sample of the document), cached to ``data/index/<stem>_profile.json``, and
loaded with each document bundle.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app import prompts
from app.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class DomainProfile:
    """A lightweight, document-derived description of an indexed corpus."""

    doc_name: str = ""
    description: str = ""
    topics: list[str] = field(default_factory=list)
    persona: str = "a knowledgeable assistant for this document"
    suggested_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    # ── Prompt fragments ────────────────────────────────────────────

    def guardrail_context(self) -> str:
        """Render the scope context injected into the guardrail prompt."""
        desc = (settings.domain_description_override or self.description or "").strip()
        topics = self.topics

        if not desc and not topics:
            return "The subject of this document is not yet profiled."

        lines = []
        if desc:
            lines.append(f"This document covers:\n{desc}")
        if topics:
            bullet = "\n".join(f"- {t}" for t in topics)
            lines.append(f"In-scope topics include:\n{bullet}")
        return "\n\n".join(lines)

    @property
    def answer_persona(self) -> str:
        return (settings.domain_persona_override or self.persona
                or "a knowledgeable assistant for this document").strip()

    @property
    def is_profiled(self) -> bool:
        return bool((settings.domain_description_override or self.description) or self.topics)


# ═══════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════


def _profile_path(doc_stem: str) -> Path:
    return settings.index_path / f"{doc_stem}_profile.json"


def default_profile(doc_name: str = "") -> DomainProfile:
    """A permissive generic profile (guardrail stays lenient until profiled)."""
    return DomainProfile(doc_name=doc_name)


def load_profile(doc_stem: str) -> DomainProfile | None:
    """Load a cached profile from disk, or None if absent/invalid."""
    path = _profile_path(doc_stem)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        profile = DomainProfile(**data)
        logger.info(f"Domain profile loaded: {profile.doc_name} ({len(profile.topics)} topics)")
        return profile
    except Exception as e:
        logger.warning(f"Failed to load domain profile for {doc_stem}: {e}")
        return None


def save_profile(profile: DomainProfile, doc_stem: str) -> None:
    path = _profile_path(doc_stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"Domain profile saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# DERIVATION
# ═══════════════════════════════════════════════════════════════════════


async def generate_profile(doc_name: str, sample_text: str) -> DomainProfile:
    """Derive a profile from a sample of the document via the active LLM."""
    from app.services.llm import chat_json

    sample = (sample_text or "")[:6000].strip()
    if not sample:
        logger.warning("No sample text to profile; using default profile.")
        return default_profile(doc_name)

    try:
        data, _ = await chat_json(prompts.domain_profile(sample), temperature=0)
        if not isinstance(data, dict) or "error" in data:
            raise ValueError(data.get("error", "profile generation failed") if isinstance(data, dict) else "bad json")
        profile = DomainProfile(
            doc_name=doc_name,
            description=str(data.get("description", "")).strip(),
            topics=[str(t).strip() for t in data.get("topics", []) if str(t).strip()],
            persona=str(data.get("persona", "")).strip()
            or "a knowledgeable assistant for this document",
            suggested_questions=[
                str(q).strip() for q in data.get("suggested_questions", []) if str(q).strip()
            ],
        )
        logger.info(
            f"Domain profile generated for '{doc_name}': "
            f"{len(profile.topics)} topics, {len(profile.suggested_questions)} questions"
        )
        return profile
    except Exception as e:
        logger.warning(f"Domain profile generation failed ({e}); using default profile.")
        return default_profile(doc_name)


async def ensure_profile(doc_name: str, doc_stem: str, sample_text: str) -> DomainProfile:
    """Load a cached profile, else generate (if enabled) and cache it."""
    cached = load_profile(doc_stem)
    if cached is not None:
        return cached
    if settings.auto_generate_profile:
        profile = await generate_profile(doc_name, sample_text)
        if profile.is_profiled:
            save_profile(profile, doc_stem)
        return profile
    return default_profile(doc_name)
