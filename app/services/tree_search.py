"""LLM Tree Search — reasoning-based navigation over a document tree (PageIndex).

Instead of vector similarity, an LLM reasons about which sections of a document
contain the answer — the way a human scans a table of contents. Two modes:

- **single-shot**: the whole compact tree fits the prompt budget → one call selects
  the relevant nodes.
- **hierarchical**: large trees are navigated level-by-level (beam descent) so we
  never overflow the context window and we exploit the document's structure.

Both modes are **domain-agnostic** (the active document's profile is injected as
context, never hardcoded) and **self-correcting** (returned node ids are validated
against the real tree, with one repair re-prompt if the model hallucinates ids).
"""

from __future__ import annotations

import logging

from app import prompts
from app.config import settings
from app.services.llm import chat_json

logger = logging.getLogger(__name__)

_SYSTEM = prompts.NAVIGATION_SYSTEM

# Safety cap on how many sibling nodes are rendered/expanded per level, so a very
# wide tree can never blow the prompt during hierarchical navigation.
_MAX_FRONTIER = 60


# ── Tree rendering helpers ───────────────────────────────────────────

def _flatten(nodes: list[dict], out: dict[str, dict]) -> dict[str, dict]:
    for n in nodes:
        nid = n.get("node_id")
        if nid:
            out[nid] = n
        _flatten(n.get("children", []), out)
    return out


def _node_line(node: dict, depth: int, *, children_hint: bool = False) -> str:
    indent = "  " * depth
    nid = node.get("node_id", "?")
    title = node.get("title", "Untitled")
    start = node.get("start_physical_index") or node.get("physical_index") or "?"
    end = node.get("end_physical_index") or start
    line = f"{indent}[{nid}] {title} (pp. {start}-{end})"
    summary = (node.get("summary") or "").strip()
    if summary:
        short = summary[:150] + "..." if len(summary) > 150 else summary
        line += f"\n{indent}    {short}"
    if children_hint and node.get("children"):
        line += f"\n{indent}    (contains {len(node['children'])} subsection(s) — can be expanded)"
    return line


def _render_full(nodes: list[dict], depth: int = 0) -> str:
    lines: list[str] = []
    for n in nodes:
        lines.append(_node_line(n, depth))
        if n.get("children"):
            lines.append(_render_full(n["children"], depth + 1))
    return "\n".join(lines)


def _render_level(nodes: list[dict]) -> str:
    return "\n".join(_node_line(n, 0, children_hint=True) for n in nodes[:_MAX_FRONTIER])


def _coerce_ids(value) -> list[str]:
    """Normalize an LLM 'ids' field that may be a list, a CSV string, or junk."""
    if isinstance(value, list):
        return [str(x).strip() for x in value if x not in (None, "") and str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _dedup(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


# ── Public API ───────────────────────────────────────────────────────

async def search_tree(
    query: str,
    *,
    tree: list[dict],
    valid_ids: set[str],
    domain_context: str = "",
) -> dict:
    """Navigate the document tree to find the node ids most likely to answer ``query``.

    Args:
        query: the user's question.
        tree: list of root node dicts (each may contain nested ``children``).
        valid_ids: every real node id in the tree (used for validation/repair).
        domain_context: short description of the document's domain (from its profile).

    Returns:
        ``{"thinking": str, "node_ids": [str], "confidence": float, "tokens_used": int}``
    """
    if not tree:
        return {"thinking": "No document indexed.", "node_ids": [], "confidence": 0.0, "tokens_used": 0}

    try:
        full = _render_full(tree)
        if len(full) <= settings.pageindex_tree_char_budget:
            return await _navigate_single_shot(query, full, valid_ids, domain_context)
        return await _navigate_hierarchical(query, tree, valid_ids, domain_context)
    except Exception as e:  # never let navigation crash the pipeline
        logger.error(f"Tree search failed: {e}")
        return {"thinking": f"Error during tree search: {e}", "node_ids": [], "confidence": 0.0, "tokens_used": 0}


async def rewrite_query(
    original_query: str,
    previous_thinking: str,
    *,
    domain_context: str = "",
) -> dict:
    """Rewrite a query for better tree navigation after a weak first attempt.

    Returns ``{"rewritten_query": str, "reasoning": str, "tokens_used": int}``.
    """
    try:
        result, tokens = await chat_json(
            prompts.navigation_rewrite(original_query, previous_thinking, domain_context),
            system=_SYSTEM, temperature=0.3,
        )
        return {
            "rewritten_query": result.get("rewritten_query", original_query) if isinstance(result, dict) else original_query,
            "reasoning": result.get("reasoning", "") if isinstance(result, dict) else "",
            "tokens_used": tokens,
        }
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        return {"rewritten_query": original_query, "reasoning": f"Rewrite failed: {e}", "tokens_used": 0}


# ── Navigation strategies ────────────────────────────────────────────

async def _navigate_single_shot(query: str, tree_text: str, valid_ids: set[str], domain_context: str) -> dict:
    prompt = prompts.navigation_select(
        query, tree_text, domain_context, max_nodes=settings.pageindex_max_select_nodes
    )
    result, tokens = await chat_json(prompt, system=_SYSTEM, temperature=0)
    thinking = result.get("thinking", "") if isinstance(result, dict) else ""
    confidence = float(result.get("confidence", 0.5)) if isinstance(result, dict) else 0.0
    ids = _coerce_ids(result.get("node_ids") or result.get("node_list")) if isinstance(result, dict) else []
    valid = [i for i in ids if i in valid_ids]

    if not valid and ids and settings.pageindex_repair_navigation and valid_ids:
        repaired, rtokens = await _repair(query, ids, valid_ids)
        tokens += rtokens
        valid = repaired

    valid = _dedup(valid)[: settings.pageindex_max_select_nodes]
    logger.info(f"Tree search (single-shot): {len(valid)} node(s), confidence {confidence:.2f}")
    return {"thinking": thinking, "node_ids": valid, "confidence": confidence if valid else 0.0, "tokens_used": tokens}


async def _navigate_hierarchical(query: str, tree: list[dict], valid_ids: set[str], domain_context: str) -> dict:
    """Descend the tree level-by-level, selecting and expanding nodes (bounded beam)."""
    flat = _flatten(tree, {})
    frontier: list[dict] = list(tree)
    selected: list[str] = []
    trace: list[str] = []
    total_tokens = 0

    for _ in range(settings.pageindex_max_depth):
        if not frontier:
            break
        result, tokens = await chat_json(
            prompts.navigation_drill(query, _render_level(frontier), domain_context),
            system=_SYSTEM, temperature=0,
        )
        total_tokens += tokens
        if isinstance(result, dict):
            if result.get("thinking"):
                trace.append(str(result["thinking"]))
            sel = [i for i in _coerce_ids(result.get("select")) if i in valid_ids]
            exp = [i for i in _coerce_ids(result.get("expand")) if i in flat]
        else:
            sel, exp = [], []

        selected.extend(sel)
        if len(selected) >= settings.pageindex_max_select_nodes or not exp:
            break

        # Next level = children of the nodes the model chose to expand (bounded).
        next_frontier: list[dict] = []
        for nid in exp:
            children = flat[nid].get("children", [])
            if children:
                next_frontier.extend(children)
            else:
                selected.append(nid)  # asked to expand a leaf → treat as a selection
        frontier = next_frontier[:_MAX_FRONTIER]

    selected = _dedup(selected)[: settings.pageindex_max_select_nodes]

    # Last resort: navigated but selected nothing → one repair against a truncated tree.
    if not selected and settings.pageindex_repair_navigation and valid_ids:
        repaired, rtokens = await _repair(query, [], valid_ids)
        total_tokens += rtokens
        selected = _dedup(repaired)[: settings.pageindex_max_select_nodes]

    logger.info(f"Tree search (hierarchical, depth≤{settings.pageindex_max_depth}): {len(selected)} node(s)")
    return {
        "thinking": " → ".join(t for t in trace if t),
        "node_ids": selected,
        "confidence": 0.7 if selected else 0.0,
        "tokens_used": total_tokens,
    }


async def _repair(query: str, bad_ids: list[str], valid_ids: set[str]) -> tuple[list[str], int]:
    """Re-prompt once, constraining the model to the real node ids."""
    try:
        result, tokens = await chat_json(
            prompts.navigation_repair(query, bad_ids, valid_ids), system=_SYSTEM, temperature=0
        )
        ids = _coerce_ids(result.get("node_ids")) if isinstance(result, dict) else []
        return [i for i in ids if i in valid_ids], tokens
    except Exception as e:
        logger.warning(f"Navigation repair failed: {e}")
        return [], 0
