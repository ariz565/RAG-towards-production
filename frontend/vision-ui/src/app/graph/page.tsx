"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as RPointerEvent, WheelEvent as RWheelEvent } from "react";
import { AppShell } from "@/components/layout/app-shell";
import { MOCK_SECOND_BRAIN_GRAPH } from "@/lib/mock-second-brain-graph";
// import { getSecondBrainGraph } from "@/lib/api"; // re-enable once the backend is reachable

// A real pannable/zoomable canvas with a continuously-running physics loop --
// not a one-shot static layout. Matches Obsidian's actual graph view behavior:
// drag empty space to pan, scroll to zoom, drag a node to reposition it (the
// simulation reacts live), nodes keep gently settling in the background.
//
// No external graph library, no CDN script, no npm install -- plain React +
// SVG + requestAnimationFrame. The previous vis-network-via-CDN version hung
// forever on this network (a blocked/dropped connection doesn't always fire a
// clean onerror), so this has zero network dependency by construction.

type GraphNode = {
  id: string;
  label: string;
  title: string;
  group: string;
  color: { background: string; border: string };
  value: number;
  orphan: boolean;
};
type GraphEdge = {
  id: string;
  from: string;
  to: string;
  label: string;
  title: string;
  color: { color: string };
  dashes: boolean;
  width: number;
  relation: string;
};
type GraphData = { nodes: GraphNode[]; edges: GraphEdge[]; node_count: number; edge_count: number };

const DATA: GraphData = MOCK_SECOND_BRAIN_GRAPH as GraphData;

const WIDTH = 900;
const HEIGHT = 580;

// Color = cluster membership (Obsidian colors by tag/folder; we have real
// semantic clusters, which is more meaningful than an arbitrary tag color).
const NODE_CLUSTER: Record<string, "retrieval" | "chunking" | "evaluation" | "orphan"> = {
  "concepts/hybrid-retrieval": "retrieval",
  "concepts/bm25": "retrieval",
  "concepts/dense-retrieval": "retrieval",
  "concepts/reciprocal-rank-fusion": "retrieval",
  "concepts/reranking": "retrieval",
  "concepts/chunking": "chunking",
  "concepts/fixed-size-chunking": "chunking",
  "concepts/recursive-chunking": "chunking",
  "concepts/chunk-overlap": "chunking",
  "concepts/rag-evaluation": "evaluation",
  "concepts/ndcg": "evaluation",
  "concepts/recall-at-k": "evaluation",
  "concepts/query-rewriting": "orphan",
};
const CLUSTER_COLOR: Record<string, string> = {
  retrieval: "#6FCF6F",
  chunking: "#56B8F0",
  evaluation: "#F2A65A",
  orphan: "#9A9A9A",
};
const CLUSTER_ANCHOR: Record<string, { x: number; y: number }> = {
  retrieval: { x: WIDTH * 0.34, y: HEIGHT * 0.4 },
  chunking: { x: WIDTH * 0.26, y: HEIGHT * 0.76 },
  evaluation: { x: WIDTH * 0.76, y: HEIGHT * 0.3 },
  orphan: { x: WIDTH * 0.82, y: HEIGHT * 0.8 },
};

function clusterOf(id: string) {
  return NODE_CLUSTER[id] ?? "retrieval";
}

type Point = { x: number; y: number; vx: number; vy: number };

function seedPositions(nodes: GraphNode[]): Map<string, Point> {
  const positions = new Map<string, Point>();
  nodes.forEach((n, i) => {
    const anchor = CLUSTER_ANCHOR[clusterOf(n.id)];
    const angle = ((i * 137.5) % 360) * (Math.PI / 180); // golden-angle spread, deterministic
    const radius = 30 + (i % 4) * 18;
    positions.set(n.id, { x: anchor.x + Math.cos(angle) * radius, y: anchor.y + Math.sin(angle) * radius, vx: 0, vy: 0 });
  });
  return positions;
}

const REPULSION = 11000;
const SPRING_LENGTH = 110;
const SPRING_STRENGTH = 0.02;
const CENTER_STRENGTH = 0.0007;
const DAMPING = 0.86;

// One simulation step, mutating `positions` in place. Called every animation
// frame -- this is what makes the layout visibly settle in real time instead
// of jumping straight to a precomputed final state. `pinnedId` is whatever
// node is currently being dragged: its position is driven by the pointer, not
// by physics, but it still repels/attracts everything else normally.
function stepPhysics(positions: Map<string, Point>, nodes: GraphNode[], edges: GraphEdge[], pinnedId: string | null) {
  for (let i = 0; i < nodes.length; i++) {
    const a = positions.get(nodes[i].id)!;
    for (let j = i + 1; j < nodes.length; j++) {
      const b = positions.get(nodes[j].id)!;
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const distSq = Math.max(dx * dx + dy * dy, 1);
      const dist = Math.sqrt(distSq);
      const force = REPULSION / distSq;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    }
  }
  for (const e of edges) {
    const a = positions.get(e.from);
    const b = positions.get(e.to);
    if (!a || !b) continue;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
    const displacement = dist - SPRING_LENGTH;
    const fx = (dx / dist) * displacement * SPRING_STRENGTH;
    const fy = (dy / dist) * displacement * SPRING_STRENGTH;
    a.vx += fx;
    a.vy += fy;
    b.vx -= fx;
    b.vy -= fy;
  }
  const cx = WIDTH / 2;
  const cy = HEIGHT / 2;
  for (const n of nodes) {
    const p = positions.get(n.id)!;
    if (n.id === pinnedId) {
      p.vx = 0;
      p.vy = 0;
      continue; // position is being set directly by the drag handler this frame
    }
    p.vx += (cx - p.x) * CENTER_STRENGTH;
    p.vy += (cy - p.y) * CENTER_STRENGTH;
    p.vx *= DAMPING;
    p.vy *= DAMPING;
    p.x += p.vx;
    p.y += p.vy;
  }
}

// Obsidian's own size curve is steep -- a handful of clear hubs, everything
// else a small uniform dot. sqrt keeps one hub (value 6) from dwarfing the
// rest the way a linear scale would.
function nodeRadius(n: GraphNode): number {
  return 4 + Math.sqrt(Math.max(n.value, 1)) * 4.2;
}

const CLICK_DRAG_THRESHOLD = 4; // px of movement before a pointerdown-then-up counts as a drag, not a click

export default function GraphPage() {
  const positionsRef = useRef<Map<string, Point>>(seedPositions(DATA.nodes));
  const nodesById = useMemo(() => new Map(DATA.nodes.map((n) => [n.id, n])), []);
  const [, setTick] = useState(0);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [view, setView] = useState({ x: 0, y: 0, scale: 1 });

  const draggingNodeRef = useRef<string | null>(null);
  const panRef = useRef<{ startX: number; startY: number; startViewX: number; startViewY: number; moved: number } | null>(null);

  // The continuous animation loop -- this is what makes it "real time" rather
  // than a static, precomputed layout.
  useEffect(() => {
    let raf = 0;
    function tick() {
      stepPhysics(positionsRef.current, DATA.nodes, DATA.edges, draggingNodeRef.current);
      setTick((t) => t + 1);
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const selected = selectedId ? nodesById.get(selectedId) ?? null : null;
  const selectedRelated = selectedId ? DATA.edges.filter((e) => e.from === selectedId || e.to === selectedId) : [];

  const activeId = hoverId ?? selectedId;
  const activeEdgeIds = activeId
    ? new Set(DATA.edges.filter((e) => e.from === activeId || e.to === activeId).map((e) => e.id))
    : null;
  const activeNeighborIds = activeId
    ? new Set(
        DATA.edges.filter((e) => e.from === activeId || e.to === activeId).flatMap((e) => [e.from, e.to]).concat(activeId)
      )
    : null;

  function handleBackgroundPointerDown(e: RPointerEvent<HTMLDivElement>) {
    panRef.current = { startX: e.clientX, startY: e.clientY, startViewX: view.x, startViewY: view.y, moved: 0 };
    e.currentTarget.setPointerCapture(e.pointerId);
  }
  function handleBackgroundPointerMove(e: RPointerEvent<HTMLDivElement>) {
    if (draggingNodeRef.current) {
      const p = positionsRef.current.get(draggingNodeRef.current);
      if (p) {
        p.x += e.movementX / view.scale;
        p.y += e.movementY / view.scale;
      }
      return;
    }
    if (panRef.current) {
      const dx = e.clientX - panRef.current.startX;
      const dy = e.clientY - panRef.current.startY;
      panRef.current.moved = Math.max(panRef.current.moved, Math.abs(dx), Math.abs(dy));
      setView((v) => ({ ...v, x: panRef.current!.startViewX + dx, y: panRef.current!.startViewY + dy }));
    }
  }
  function handleBackgroundPointerUp(e: RPointerEvent<HTMLDivElement>) {
    const wasRealPan = (panRef.current?.moved ?? 0) > CLICK_DRAG_THRESHOLD;
    const wasDraggingNode = draggingNodeRef.current !== null;
    panRef.current = null;
    draggingNodeRef.current = null;
    if (!wasRealPan && !wasDraggingNode) setSelectedId(null); // a real click on empty space
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  }
  function handleWheel(e: RWheelEvent<HTMLDivElement>) {
    e.preventDefault();
    setView((v) => ({ ...v, scale: Math.min(2.5, Math.max(0.3, v.scale - e.deltaY * 0.0012 * v.scale)) }));
  }
  function handleNodePointerDown(e: RPointerEvent<SVGGElement>, nodeId: string) {
    e.stopPropagation();
    draggingNodeRef.current = nodeId;
    setSelectedId(nodeId);
    // Capture on the node itself, not the background div (stopPropagation means
    // the background's own pointerdown never fires for this gesture). Pointer
    // capture events still bubble normally, so handleBackgroundPointerMove/Up
    // (attached higher up) keep firing even if the drag moves outside the
    // canvas bounds entirely -- without this, a fast drag off-canvas would
    // never deliver a pointerup and the node would stay "dragging" forever.
    e.currentTarget.setPointerCapture(e.pointerId);
  }

  return (
    <AppShell
      eyebrow="Chapter VIII · Second Brain"
      title={<>The knowledge <em className="italic text-cobalt-500">graph.</em></>}
      breadcrumb={["Workspace", "Graph"]}
    >
      <p className="max-w-2xl text-ink-soft text-lg mb-6 -mt-2">
        A live view of second_brain&apos;s typed-edge graph &mdash; is-a, part-of, contradicts,
        supersedes, depends-on. Read-only: generated by running the second_brain CLI, not by this app.
      </p>

      <div className="mb-6 px-4 py-3 rounded-md bg-amber-50 border border-amber-200 text-amber-900 text-sm font-mono flex items-center gap-2">
        <span className="h-1.5 w-1.5 rounded-full bg-amber-500 shrink-0" />
        Showing sample data &mdash; this is not your real graph. Wire up the backend endpoint
        (see the commented import above) once it&apos;s reachable.
      </div>

      {/* Full-bleed dark canvas, matching Obsidian's own graph view register --
          no competing sidebar column; node details float over the canvas instead. */}
      <div
        className="rounded-lg border border-[#2b2c30] relative touch-none select-none"
        style={{ background: "#1a1b1e", height: "min(720px, 70vh)", minHeight: 420, cursor: draggingNodeRef.current ? "grabbing" : "grab" }}
        onPointerDown={handleBackgroundPointerDown}
        onPointerMove={handleBackgroundPointerMove}
        onPointerUp={handleBackgroundPointerUp}
        onWheel={handleWheel}
      >
        <svg
          width={WIDTH}
          height={HEIGHT}
          style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`, transformOrigin: "0 0" }}
        >
          {DATA.edges.map((e) => {
            const a = positionsRef.current.get(e.from);
            const b = positionsRef.current.get(e.to);
            if (!a || !b) return null;
            const isActive = activeEdgeIds?.has(e.id) ?? false;
            const dimmed = activeEdgeIds !== null && !isActive;
            return (
              <line
                key={e.id}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke={isActive ? "rgba(255,255,255,0.55)" : "rgba(255,255,255,0.14)"}
                strokeWidth={isActive ? 1.6 : 1}
                opacity={dimmed ? 0.3 : 1}
              >
                <title>{e.title}</title>
              </line>
            );
          })}

          {DATA.nodes.map((n) => {
            const p = positionsRef.current.get(n.id);
            if (!p) return null;
            const r = nodeRadius(n);
            const isActive = activeId === n.id;
            const dimmed = activeNeighborIds !== null && !activeNeighborIds.has(n.id);
            const fill = CLUSTER_COLOR[clusterOf(n.id)];
            return (
              <g
                key={n.id}
                transform={`translate(${p.x}, ${p.y})`}
                opacity={dimmed ? 0.25 : 1}
                style={{ cursor: "pointer" }}
                onPointerDown={(e) => handleNodePointerDown(e, n.id)}
                onMouseEnter={() => setHoverId(n.id)}
                onMouseLeave={() => setHoverId(null)}
              >
                <circle
                  r={isActive ? r + 2 : r}
                  fill={fill}
                  style={{ filter: isActive ? `drop-shadow(0 0 6px ${fill}99)` : "none" }}
                />
                <text x={r + 5} y={3} fontSize={10.5} fontFamily="var(--font-sans)" fill={isActive ? "#F2F2F2" : "#A9A9AC"}>
                  {n.label.length > 26 ? n.label.slice(0, 25) + "…" : n.label}
                </text>
                <title>{n.title}</title>
              </g>
            );
          })}
        </svg>

        {selected && (
          <div
            className="absolute top-4 right-4 w-72 rounded-lg p-4 border border-[#2b2c30]"
            style={{ background: "rgba(26,27,30,0.96)", backdropFilter: "blur(4px)" }}
          >
            <h3 className="text-base font-semibold mb-1" style={{ color: "#F2F2F2" }}>
              {selected.label}
            </h3>
            <p className="text-xs text-[#a9a9ac] mb-1 leading-relaxed">{selected.title}</p>
            <p className="font-mono text-[10px] text-[#7a7a7d] uppercase tracking-widest mb-3">
              {selected.orphan ? "orphan · no relationships" : clusterOf(selected.id)}
            </p>
            {selectedRelated.length > 0 ? (
              <ul className="space-y-1.5">
                {selectedRelated.map((e) => {
                  const other = e.from === selected.id ? e.to : e.from;
                  const dir = e.from === selected.id ? "→" : "←";
                  const otherNode = nodesById.get(other);
                  return (
                    <li key={e.id} className="text-xs font-mono flex items-center gap-1.5 text-[#c7c7ca]">
                      <span className="px-1.5 py-0.5 rounded text-[#1a1b1e] font-semibold" style={{ background: e.color.color }}>
                        {e.label}
                      </span>
                      <span>{dir}</span>
                      <span>{otherNode?.label ?? other}</span>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="text-xs font-mono text-[#7a7a7d]">No relationships.</p>
            )}
          </div>
        )}

        <div
          className="absolute bottom-4 left-4 flex items-center gap-3 px-3 py-1.5 rounded-md border border-[#2b2c30] font-mono text-[10px] uppercase tracking-widest text-[#7a7a7d]"
          style={{ background: "rgba(26,27,30,0.9)" }}
        >
          {Object.entries(CLUSTER_COLOR).map(([name, color]) => (
            <span key={name} className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full shrink-0" style={{ background: color }} />
              {name}
            </span>
          ))}
        </div>
      </div>

      <p className="font-mono text-[10px] text-ink-soft uppercase tracking-widest mt-3">
        {DATA.node_count} nodes &middot; {DATA.edge_count} edges &middot; drag empty space to pan, scroll to zoom, drag a node to move it
      </p>
    </AppShell>
  );
}
