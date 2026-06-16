"""
Vision — Streamlit Frontend
University Admission Guide Q&A — Multi-Strategy Agentic RAG Platform

Run:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests
import streamlit as st

# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════

API_BASE = "http://localhost:8001/api"

STRATEGY_OPTIONS = {
    "PageIndex (Tree)": "pageindex",
    "Hybrid (BM25 + Vector)": "hybrid",
    "BM25 Only": "bm25_only",
    "Vector Only": "vector_only",
}

MODEL_OPTIONS = {
    "Ollama (Local/Free)": "ollama",
    "OpenAI": "openai",
    "Azure OpenAI": "azure_openai",
    "OpenRouter": "openrouter",
    "Groq": "groq",
    "Google Gemini": "gemini",
}

# ═══════════════════════════════════════════════════════════════
#  Page Config
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Vision — Admission Guide Q&A",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════════════


def api_get(endpoint: str):
    """GET request to FastAPI backend."""
    try:
        r = requests.get(f"{API_BASE}/{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is the FastAPI server running on port 8001?")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_post(endpoint: str, payload: dict | None = None, timeout: int = 120):
    """POST request to FastAPI backend."""
    try:
        r = requests.post(f"{API_BASE}/{endpoint}", json=payload or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is the FastAPI server running on port 8001?")
        return None
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"API Error: {detail}")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def get_health():
    """Fetch backend health status."""
    return api_get("health")


def get_config():
    """Fetch current backend configuration."""
    return api_get("config")


# ═══════════════════════════════════════════════════════════════
#  Sidebar — Configuration Panel
# ═══════════════════════════════════════════════════════════════


def render_sidebar():
    """Render the sidebar with config switches and indexing controls."""
    with st.sidebar:
        st.title("⚙️ Configuration")

        # ── Health Check ────────────────────────────────────
        health = get_health()
        if health:
            status_emoji = "🟢" if health.get("indexed") or health.get("hybrid_index_loaded") else "🟡"
            st.markdown(f"{status_emoji} **Backend:** Connected")

            if health.get("doc_name"):
                st.caption(f"📄 {health['doc_name']} — {health.get('total_pages', 0)} pages")

            cols = st.columns(2)
            with cols[0]:
                st.metric("Model", health.get("active_model", "—"))
            with cols[1]:
                st.metric("Strategy", health.get("active_retrieval", "—"))
        else:
            st.markdown("🔴 **Backend:** Disconnected")
            st.caption("Start the server: `uvicorn app.main:app --port 8001`")

        st.divider()

        # ── Strategy Switch ─────────────────────────────────
        st.subheader("🔍 Retrieval Strategy")
        current_strategy = (
            health.get("active_retrieval", "hybrid") if health else "hybrid"
        )
        # Find the display label for the current strategy
        strategy_labels = list(STRATEGY_OPTIONS.keys())
        strategy_values = list(STRATEGY_OPTIONS.values())
        default_idx = strategy_values.index(current_strategy) if current_strategy in strategy_values else 1

        selected_strategy_label = st.radio(
            "Choose retrieval approach:",
            strategy_labels,
            index=default_idx,
            key="strategy_radio",
            help="PageIndex = LLM navigates a hierarchical document tree. "
                 "Hybrid = BM25 sparse + Qdrant vector search with RRF fusion.",
        )
        selected_strategy = STRATEGY_OPTIONS[selected_strategy_label]

        st.divider()

        # ── Model Switch ────────────────────────────────────
        st.subheader("🤖 LLM Provider")
        current_model = (
            health.get("active_model", "ollama") if health else "ollama"
        )
        model_labels = list(MODEL_OPTIONS.keys())
        model_values = list(MODEL_OPTIONS.values())
        default_model_idx = model_values.index(current_model) if current_model in model_values else 0

        selected_model_label = st.radio(
            "Choose LLM provider:",
            model_labels,
            index=default_model_idx,
            key="model_radio",
            help="Ollama = free/local. OpenAI, Azure OpenAI, OpenRouter, Groq, and Gemini require API keys in .env.",
        )
        selected_model = MODEL_OPTIONS[selected_model_label]

        # ── Apply Config Button ─────────────────────────────
        if st.button("💾 Apply Configuration", use_container_width=True, type="primary"):
            result = api_post("config", {
                "active_model": selected_model,
                "active_retrieval": selected_strategy,
            })
            if result:
                st.success("Configuration updated!")
                st.rerun()

        st.divider()

        # ── Indexing Controls ───────────────────────────────
        st.subheader("📚 Indexing")

        index_col1, index_col2 = st.columns(2)
        with index_col1:
            if st.button("🌳 PageIndex", use_container_width=True, help="Build tree index"):
                with st.spinner("Building PageIndex... This may take several minutes."):
                    # Find first PDF
                    result = _index_with_first_pdf("index")
                    if result and result.get("success"):
                        st.success(f"✅ {result.get('doc_name')} — {result.get('total_pages')} pages")
                        st.rerun()

        with index_col2:
            if st.button("🔗 Hybrid", use_container_width=True, help="Build BM25+Vector index"):
                with st.spinner("Building Hybrid index..."):
                    result = _index_with_first_pdf("index/hybrid")
                    if result and result.get("success"):
                        st.success(f"✅ {result.get('message', 'Done')}")
                        st.rerun()

        if st.button("🚀 Build All Indexes", use_container_width=True):
            with st.spinner("Building all indexes... This may take a while."):
                result = _index_with_first_pdf("index/all")
                if result and result.get("success"):
                    st.success(f"✅ Both indexes built!")
                    st.rerun()

        st.divider()

        # ── Info ────────────────────────────────────────────
        st.caption("Built with LangGraph + PageIndex + BM25 + Qdrant")
        st.caption("Place your PDF in `data/pdf/` and click an index button.")

    return selected_strategy, selected_model


def _index_with_first_pdf(endpoint: str):
    """Call indexing endpoint with the first available PDF name."""
    pdf_dir = Path("data/pdf")
    if pdf_dir.exists():
        pdfs = list(pdf_dir.glob("*.pdf"))
        if pdfs:
            return api_post(endpoint, {"filename": pdfs[0].name}, timeout=600)
    # Let the backend figure it out / return error
    return api_post(endpoint, {"filename": "guide.pdf"}, timeout=600)


# ═══════════════════════════════════════════════════════════════
#  Main Chat Interface
# ═══════════════════════════════════════════════════════════════


def render_chat(strategy: str, model: str):
    """Render the main chat interface."""
    st.title("🎓 Vision — Admission Guide Q&A")
    st.caption("Ask questions about university admissions. Powered by Agentic RAG.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show metadata for assistant messages
            if msg["role"] == "assistant" and msg.get("metadata"):
                _render_response_metadata(msg["metadata"])

    # Chat input
    if prompt := st.chat_input("Ask about admissions..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = time.time()
                response = api_post("ask", {
                    "query": prompt,
                    "strategy": strategy,
                    "model_provider": model,
                })
                elapsed = time.time() - start

            if response:
                answer = response.get("answer", "No answer received.")
                st.markdown(answer)

                metadata = {
                    "strategy_used": response.get("strategy_used", strategy),
                    "model_used": response.get("model_used", ""),
                    "confidence": response.get("confidence", 0),
                    "retrieval_attempts": response.get("retrieval_attempts", 1),
                    "rewritten_query": response.get("rewritten_query"),
                    "total_tokens": response.get("total_tokens", 0),
                    "total_duration_ms": response.get("total_duration_ms", elapsed * 1000),
                    "citations": response.get("citations", []),
                    "pipeline_steps": response.get("pipeline_steps", []),
                }
                _render_response_metadata(metadata)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": metadata,
                })
            else:
                st.error("Failed to get a response. Check the backend logs.")


def _render_response_metadata(meta: dict):
    """Render response metadata: citations, pipeline, stats."""
    # ── Quick Stats ─────────────────────────────────
    cols = st.columns(4)
    with cols[0]:
        confidence = meta.get("confidence", 0)
        conf_color = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"
        st.caption(f"{conf_color} Confidence: {confidence:.0%}")
    with cols[1]:
        st.caption(f"🔍 Strategy: {meta.get('strategy_used', '—')}")
    with cols[2]:
        st.caption(f"🤖 Model: {meta.get('model_used', '—')}")
    with cols[3]:
        duration = meta.get("total_duration_ms", 0)
        st.caption(f"⏱️ {duration / 1000:.1f}s | {meta.get('total_tokens', 0)} tokens")

    # ── Citations ───────────────────────────────────
    citations = meta.get("citations", [])
    if citations:
        with st.expander(f"📖 Citations ({len(citations)})", expanded=False):
            for i, cite in enumerate(citations, 1):
                pages = cite.get("page_numbers", [])
                section = cite.get("section_title", "")
                relevance = cite.get("relevance", "")
                st.markdown(
                    f"**{i}.** Pages {', '.join(map(str, pages))}"
                    + (f" — *{section}*" if section else "")
                )
                if relevance:
                    st.caption(f"   {relevance}")

    # ── Pipeline Steps ──────────────────────────────
    steps = meta.get("pipeline_steps", [])
    if steps:
        with st.expander("🔧 Pipeline Execution", expanded=False):
            for step in steps:
                name = step.get("node_name", "")
                status = step.get("status", "pending")
                thinking = step.get("thinking", "")
                duration_ms = step.get("duration_ms", 0)
                tokens = step.get("tokens_used", 0)

                status_icon = {
                    "completed": "✅",
                    "error": "❌",
                    "thinking": "🔄",
                    "pending": "⏳",
                }.get(status, "⏳")

                st.markdown(
                    f"{status_icon} **{name}** "
                    f"({duration_ms:.0f}ms, {tokens} tokens)"
                )
                if thinking:
                    st.caption(f"   💭 {thinking[:200]}{'...' if len(thinking) > 200 else ''}")

    # ── Rewritten Query ─────────────────────────────
    if meta.get("rewritten_query"):
        st.info(f"🔄 Query was rewritten: *{meta['rewritten_query']}*")


# ═══════════════════════════════════════════════════════════════
#  Tree Explorer Tab
# ═══════════════════════════════════════════════════════════════


def render_tree_explorer():
    """Render the PageIndex document tree explorer."""
    st.subheader("🌳 Document Tree Explorer")
    st.caption("Navigate the hierarchical structure built by PageIndex.")

    tree_data = api_get("tree")
    if not tree_data:
        st.info("No PageIndex tree loaded. Build one first via the sidebar.")
        return

    st.markdown(f"**{tree_data.get('doc_name', 'Document')}** — {tree_data.get('total_pages', 0)} pages")

    tree_nodes = tree_data.get("tree", [])
    if not tree_nodes:
        st.warning("Tree is empty.")
        return

    _render_tree_nodes(tree_nodes, level=0)


def _render_tree_nodes(nodes: list, level: int):
    """Recursively render tree nodes."""
    for node in nodes:
        title = node.get("title", "Untitled")
        summary = node.get("summary", "")
        page_start = node.get("page_start")
        page_end = node.get("page_end")
        children = node.get("children", [])
        node_id = node.get("node_id", "")

        indent = "│  " * level
        page_range = ""
        if page_start is not None:
            page_range = f" (p.{page_start}" + (f"–{page_end})" if page_end else ")")

        label = f"{indent}{'📁' if children else '📄'} **{title}**{page_range}"

        if children:
            with st.expander(label, expanded=(level < 1)):
                if summary:
                    st.caption(summary)
                _render_tree_nodes(children, level + 1)
        else:
            st.markdown(label)
            if summary:
                st.caption(f"{indent}   {summary[:150]}{'...' if len(summary) > 150 else ''}")


# ═══════════════════════════════════════════════════════════════
#  Page Viewer Tab
# ═══════════════════════════════════════════════════════════════


def render_page_viewer():
    """View raw page text from the indexed PDF."""
    st.subheader("📄 Page Viewer")

    health = get_health()
    total_pages = health.get("total_pages", 0) if health else 0

    if total_pages == 0:
        st.info("No document indexed. Build a PageIndex first.")
        return

    page_num = st.number_input(
        "Page number:",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
    )

    if st.button("Load Page"):
        page_data = api_get(f"page/{page_num}")
        if page_data:
            st.markdown(f"**Page {page_data.get('page_number', page_num)}** — {page_data.get('token_count', 0)} tokens")
            st.text_area("Page content:", value=page_data.get("text", ""), height=400, disabled=True)


# ═══════════════════════════════════════════════════════════════
#  Main App
# ═══════════════════════════════════════════════════════════════


def main():
    # Sidebar config
    strategy, model = render_sidebar()

    # Main tabs
    tab_chat, tab_tree, tab_pages, tab_config = st.tabs([
        "💬 Chat", "🌳 Tree Explorer", "📄 Page Viewer", "⚙️ System Config"
    ])

    with tab_chat:
        render_chat(strategy, model)

    with tab_tree:
        render_tree_explorer()

    with tab_pages:
        render_page_viewer()

    with tab_config:
        render_system_config()


def render_system_config():
    """Show full system configuration and status."""
    st.subheader("⚙️ System Configuration")

    config = get_config()
    if not config:
        st.error("Cannot fetch configuration from backend.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Active Settings")
        st.json({
            "model": config.get("active_model", "—"),
            "model_name": config.get("active_model_name", "—"),
            "retrieval": config.get("active_retrieval", "—"),
            "embedding": config.get("active_embedding", "—"),
        })

        st.markdown("### Available Providers")
        providers = config.get("available_providers", [])
        for p in providers:
            status = "✅" if p.get("available") else "❌"
            st.markdown(f"{status} **{p.get('provider', '?')}** — {p.get('model', '?')}")
            if not p.get("available"):
                st.caption(f"   {p.get('reason', 'Not configured')}")

    with col2:
        st.markdown("### Available Strategies")
        strategies = config.get("available_strategies", [])
        for s in strategies:
            st.markdown(f"🔍 **{s}**")

        st.markdown("### Health")
        health = get_health()
        if health:
            st.json(health)

    # ── Clear Chat ──────────────────────────────────
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
