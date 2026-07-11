"""Standalone interactive graph viewer.

`vis-network` (vis.js) is loaded from a CDN *inside the browser* via a <script>
tag with a verified Subresource Integrity hash -- nothing is pip-installed, so
this doesn't violate the module's zero-new-dependency rule (the hash below was
fetched and computed for real, the same way graphify's export.py pins its own
CDN script -- see that file's source for the technique this one is modeled on).

The generated file is fully self-contained: node/edge data is inlined as JSON in
a <script> block rather than fetched separately, so it opens correctly via
file:// with no server and no CORS problem. `</` is escaped to `<` + backslash +
`/` in the inlined JSON for the same reason graphify does it -- an LLM-controlled
label containing a literal "</script>" must not be able to terminate the block
early.

Known limit vs. graphify's graph.html: no clustering, no god-node ranking, no
node-count cap/aggregation -- this renders the whole graph as-is, which is fine
at the personal-wiki scale (tens to low hundreds of pages) this module targets.
If your wiki outgrows one readable view, that's the signal to add clustering,
not to keep stretching this file (see README's "known limits").
"""

from __future__ import annotations

import json
from pathlib import Path

from second_brain.graph import GraphStore
from second_brain.store import WikiStore

_VIS_JS_URL = "https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"
_VIS_JS_INTEGRITY = "sha384-Ux6phic9PEHJ38YtrijhkzyJ8yQlH8i/+buBR8s3mAZOJrP1gwyvAcIYl3GWtpX1"

_RELATION_COLORS = {
    "is-a": "#4C6EF5",
    "part-of": "#12B886",
    "contradicts": "#E03131",
    "supersedes": "#F08C00",
    "depends-on": "#7048E8",
}
_NODE_FILL = {"concept": "#E7F5FF", "entity": "#FFF9DB"}
_NODE_BORDER = {"concept": "#1971C2", "entity": "#F08C00"}


def node_records(graph: GraphStore, store: WikiStore) -> list[dict]:
    orphan_ids = set(graph.orphans())
    degree: dict[str, int] = {n.id: 0 for n in graph.all_nodes()}
    for edge in graph.all_edges():
        degree[edge.src] = degree.get(edge.src, 0) + 1
        degree[edge.dst] = degree.get(edge.dst, 0) + 1

    records = []
    for n in graph.all_nodes():
        meta = store.page_meta(f"{n.id}.md")
        is_orphan = n.id in orphan_ids
        records.append({
            "id": n.id,
            "label": n.title,
            "title": meta.tldr if meta else n.title,
            "group": n.node_type,
            "color": {
                "background": _NODE_FILL.get(n.node_type, "#E7F5FF"),
                "border": "#E03131" if is_orphan else _NODE_BORDER.get(n.node_type, "#1971C2"),
            },
            "borderWidth": 3 if is_orphan else 1,
            "shapeProperties": {"borderDashes": [4, 2] if is_orphan else False},
            "value": max(1, degree.get(n.id, 0)),
            "orphan": is_orphan,
        })
    return records


def edge_records(graph: GraphStore) -> list[dict]:
    records = []
    for i, e in enumerate(graph.all_edges()):
        tooltip = f"{e.relation} (confidence {e.confidence:.2f})"
        if e.rationale:
            tooltip += f" -- {e.rationale}"
        records.append({
            "id": f"e{i}",
            "from": e.src,
            "to": e.dst,
            "label": e.relation,
            "title": tooltip,
            "color": {"color": _RELATION_COLORS.get(e.relation, "#868E96")},
            "dashes": e.confidence < 0.6,
            "width": 1 + e.confidence * 2,
            "arrows": "to",
            "relation": e.relation,
        })
    return records


def render_graph_html(graph: GraphStore, store: WikiStore) -> str:
    nodes = node_records(graph, store)
    edges = edge_records(graph)
    relations = sorted({e["relation"] for e in edges}) or sorted(_RELATION_COLORS)
    legend = {r: _RELATION_COLORS.get(r, "#868E96") for r in relations}

    def _inline_json(obj) -> str:
        return json.dumps(obj).replace("</", "<\\/")

    html = _TEMPLATE
    html = html.replace("__VIS_JS_URL__", _VIS_JS_URL)
    html = html.replace("__VIS_JS_INTEGRITY__", _VIS_JS_INTEGRITY)
    html = html.replace("__NODES_JSON__", _inline_json(nodes))
    html = html.replace("__EDGES_JSON__", _inline_json(edges))
    html = html.replace("__LEGEND_JSON__", _inline_json(legend))
    html = html.replace("__NODE_COUNT__", str(len(nodes)))
    html = html.replace("__EDGE_COUNT__", str(len(edges)))
    return html


def write_graph_html(graph: GraphStore, store: WikiStore, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_graph_html(graph, store), encoding="utf-8")
    return out_path


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Second Brain -- Graph</title>
<script src="__VIS_JS_URL__" integrity="__VIS_JS_INTEGRITY__" crossorigin="anonymous"
        onerror="window.__visLoadFailed = true;"></script>
<style>
  :root { --border: #DEE2E6; --ink: #1A1A1A; --ink-soft: #495057; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; color: var(--ink); }
  #app { display: grid; grid-template-columns: 300px 1fr; grid-template-rows: 56px 1fr; height: 100vh; }
  header { grid-column: 1 / 3; display: flex; align-items: center; gap: 16px; padding: 0 16px;
           border-bottom: 1px solid var(--border); }
  header h1 { font-size: 15px; margin: 0; white-space: nowrap; }
  header .counts { font-size: 12px; color: var(--ink-soft); white-space: nowrap; }
  #search { flex: 1; max-width: 360px; padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; }
  #sidebar { border-right: 1px solid var(--border); overflow-y: auto; padding: 14px; }
  #network { position: relative; }
  #search-results { list-style: none; margin: 6px 0 16px; padding: 0; max-height: 140px; overflow-y: auto; }
  #search-results li { padding: 4px 8px; font-size: 13px; cursor: pointer; border-radius: 4px; }
  #search-results li:hover { background: #F1F3F5; }
  .section-title { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--ink-soft);
                   margin: 0 0 8px; }
  .legend-row { display: flex; align-items: center; gap: 6px; font-size: 13px; margin-bottom: 6px; cursor: pointer; }
  .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
  #info-panel { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--border); font-size: 13px; }
  #info-panel h3 { margin: 0 0 4px; font-size: 15px; }
  #info-panel .tldr { color: var(--ink-soft); margin: 0 0 6px; }
  #info-panel .meta { color: var(--ink-soft); font-size: 11px; margin: 0 0 10px; }
  #info-panel h4 { font-size: 11px; text-transform: uppercase; color: var(--ink-soft); margin: 10px 0 6px; }
  #info-panel ul { list-style: none; margin: 0; padding: 0; }
  #info-panel li { margin-bottom: 6px; }
  .rel-tag { color: white; font-size: 10px; padding: 1px 6px; border-radius: 8px; margin-right: 4px; }
  a.node-link { color: #1971C2; text-decoration: none; }
  a.node-link:hover { text-decoration: underline; }
  .empty-hint { font-size: 12px; color: var(--ink-soft); }
  #load-error { display: none; grid-column: 1 / 3; grid-row: 2; padding: 32px; overflow-y: auto; }
  #load-error .box { max-width: 640px; margin: 40px auto; padding: 24px 28px; border: 1px solid #FFC9C9;
                      background: #FFF5F5; border-radius: 8px; }
  #load-error h2 { margin: 0 0 10px; font-size: 16px; color: #C92A2A; }
  #load-error p { margin: 0 0 10px; font-size: 13px; line-height: 1.6; color: var(--ink); }
  #load-error code { background: #FFE3E3; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1>Second Brain -- Graph</h1>
    <span class="counts">__NODE_COUNT__ nodes &middot; __EDGE_COUNT__ edges</span>
    <input id="search" type="text" placeholder="Search pages...">
  </header>
  <div id="sidebar">
    <p class="section-title">Search results</p>
    <ul id="search-results"></ul>
    <p class="section-title">Relations (click to toggle)</p>
    <div id="legend"></div>
    <div id="info-panel">
      <p class="empty-hint">Click a node to see its relationships. Dashed borders mark orphans (no edges).</p>
    </div>
  </div>
  <div id="network"></div>
  <div id="load-error">
    <div class="box">
      <h2>The graph didn't load</h2>
      <p id="load-error-detail">Something went wrong.</p>
      <p><strong>Most likely cause:</strong> this page loads <code>vis-network</code> from a CDN
         (<code>unpkg.com</code>) at view time -- it isn't bundled into the file. If you're
         viewing this inside an editor's built-in preview/webview (VS Code's Simple Browser,
         a Markdown-preview-with-HTML panel, etc.), that preview's security policy often blocks
         loading external scripts entirely, silently.</p>
      <p><strong>Fix:</strong> open this exact file directly in a real browser tab (Chrome, Edge,
         Firefox) instead -- copy the file path and paste it into the address bar, or
         right-click the file and choose "Open with" your browser. If it still fails there,
         check that this machine can actually reach <code>unpkg.com</code> (corporate proxy/
         firewall block is the next most likely cause).</p>
    </div>
  </div>
</div>
<script>
  function showLoadError(detail) {
    document.getElementById("load-error-detail").textContent = detail;
    document.getElementById("load-error").style.display = "block";
    document.getElementById("network").style.display = "none";
  }

  if (window.__visLoadFailed || typeof vis === "undefined") {
    showLoadError(
      "The vis-network script never loaded (network request failed, was blocked, or the " +
      "page is running inside a restricted preview pane)."
    );
  } else {
    try {
      renderGraph();
    } catch (err) {
      showLoadError("A script error occurred while building the graph: " + err.message);
    }
  }

  function renderGraph() {
  const NODES = __NODES_JSON__;
  const EDGES = __EDGES_JSON__;
  const LEGEND = __LEGEND_JSON__;

  const nodesData = new vis.DataSet(NODES);
  const edgesData = new vis.DataSet(EDGES);

  const network = new vis.Network(
    document.getElementById("network"),
    { nodes: nodesData, edges: edgesData },
    {
      nodes: { shape: "dot", scaling: { min: 10, max: 40 }, font: { size: 14, color: "#1A1A1A" } },
      edges: { smooth: { type: "continuous" }, font: { size: 10, align: "middle", color: "#495057", strokeWidth: 0 } },
      physics: {
        solver: "barnesHut",
        barnesHut: { gravitationalConstant: -3000, springLength: 120, springConstant: 0.03 },
        stabilization: { iterations: 200 },
      },
      interaction: { hover: true, tooltipDelay: 100 },
    }
  );
  network.once("stabilizationIterationsDone", () => network.setOptions({ physics: false }));

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s == null ? "" : String(s);
    return div.innerHTML;
  }

  const infoPanel = document.getElementById("info-panel");

  function renderNodeInfo(nodeId) {
    const node = nodesData.get(nodeId);
    if (!node) return;
    const related = edgesData.get().filter((e) => e.from === nodeId || e.to === nodeId);
    let html = `<h3>${escapeHtml(node.label)}</h3><p class="tldr">${escapeHtml(node.title || "")}</p>`;
    html += `<p class="meta">type: ${escapeHtml(node.group)}${node.orphan ? " &middot; orphan (no relationships)" : ""}</p>`;
    if (related.length) {
      html += "<h4>Relationships</h4><ul>";
      for (const e of related) {
        const other = e.from === nodeId ? e.to : e.from;
        const dir = e.from === nodeId ? "&rarr;" : "&larr;";
        html += `<li><span class="rel-tag" style="background:${e.color.color}">${escapeHtml(e.label)}</span> ${dir} <a href="#" data-node="${escapeHtml(other)}" class="node-link">${escapeHtml(other)}</a></li>`;
      }
      html += "</ul>";
    }
    infoPanel.innerHTML = html;
    infoPanel.querySelectorAll(".node-link").forEach((a) => {
      a.addEventListener("click", (ev) => {
        ev.preventDefault();
        const id = a.getAttribute("data-node");
        network.focus(id, { scale: 1.2, animation: true });
        network.selectNodes([id]);
        renderNodeInfo(id);
      });
    });
  }

  network.on("click", (params) => {
    if (params.nodes.length) renderNodeInfo(params.nodes[0]);
  });

  const searchInput = document.getElementById("search");
  const searchResults = document.getElementById("search-results");
  searchInput.addEventListener("input", () => {
    const q = searchInput.value.trim().toLowerCase();
    searchResults.innerHTML = "";
    if (!q) return;
    const matches = nodesData.get().filter((n) => n.label.toLowerCase().includes(q)).slice(0, 20);
    for (const m of matches) {
      const li = document.createElement("li");
      li.textContent = m.label;
      li.addEventListener("click", () => {
        network.focus(m.id, { scale: 1.3, animation: true });
        network.selectNodes([m.id]);
        renderNodeInfo(m.id);
      });
      searchResults.appendChild(li);
    }
  });

  const legendDiv = document.getElementById("legend");
  for (const [relation, color] of Object.entries(LEGEND)) {
    const row = document.createElement("label");
    row.className = "legend-row";
    row.innerHTML = `<input type="checkbox" checked data-relation="${escapeHtml(relation)}"> <span class="swatch" style="background:${color}"></span> ${escapeHtml(relation)}`;
    legendDiv.appendChild(row);
  }
  legendDiv.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
    cb.addEventListener("change", () => {
      const hidden = new Set(
        Array.from(legendDiv.querySelectorAll('input[type="checkbox"]:not(:checked)')).map((c) => c.dataset.relation)
      );
      edgesData.update(edgesData.get().map((e) => ({ id: e.id, hidden: hidden.has(e.relation) })));
    });
  });
  } // end renderGraph()
</script>
</body>
</html>
"""
