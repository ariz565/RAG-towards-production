import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { AppShell } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { mockDocs } from "@/lib/mock-data";
import { Upload, ArrowUpRight, CheckCircle2, Loader2, XCircle, MoreHorizontal, Search } from "lucide-react";

export const Route = createFileRoute("/documents")({
  head: () => ({ meta: [{ title: "Documents — Vision" }, { name: "description", content: "Upload and manage your indexed PDF documents." }] }),
  component: Documents,
});

function StatusDot({ s }: { s: string }) {
  const map: Record<string, { c: string; i: any; l: string }> = {
    indexed: { c: "text-emerald-700", i: CheckCircle2, l: "Indexed" },
    indexing: { c: "text-cobalt-500", i: Loader2, l: "Indexing" },
    failed: { c: "text-destructive", i: XCircle, l: "Failed" },
  };
  const { c, i: I, l } = map[s];
  return (
    <span className={`inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-widest ${c}`}>
      <I className={`h-3 w-3 ${s === "indexing" ? "animate-spin" : ""}`} /> {l}
    </span>
  );
}

function Documents() {
  const [dragOver, setDragOver] = useState(false);
  const [picked, setPicked] = useState<string | null>(null);

  return (
    <AppShell
      eyebrow="Chapter II · Library"
      title={<>The shelf, <em className="italic text-cobalt-500">curated.</em></>}
      breadcrumb={["Workspace", "Documents"]}
    >
      <p className="max-w-2xl text-ink-soft text-lg mb-10 -mt-2">A working library of {mockDocs.length} PDFs across {new Set(mockDocs.map(d=>d.family)).size} families. Drag one in to begin reading it.</p>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); setPicked(e.dataTransfer.files[0]?.name ?? null); }}
        className={`relative grain rounded-lg border-2 border-dashed transition-all p-14 text-center ${dragOver ? "border-cobalt-500 bg-cobalt-50/40" : "border-rule bg-card"}`}
      >
        <div className="mx-auto h-12 w-12 rounded-full bg-ink text-paper grid place-items-center"><Upload className="h-5 w-5" /></div>
        <h3 className="font-display text-3xl italic mt-5">Lay a PDF on the table.</h3>
        <p className="text-sm text-ink-soft mt-2">
          Drop it here or <label className="text-cobalt-500 cursor-pointer ink-link">browse from your computer
            <input type="file" accept="application/pdf" className="hidden" onChange={(e) => setPicked(e.target.files?.[0]?.name ?? null)} />
          </label>
        </p>
        <p className="font-mono text-[10px] uppercase tracking-widest text-ink-soft mt-3">PDF · up to 100MB · auto-indexed in ~14s</p>
      </div>

      {picked && (
        <div className="paper-card rounded-lg p-6 mt-6 animate-fade-in">
          <div className="flex items-center justify-between">
            <div>
              <div className="eyebrow">New document</div>
              <p className="font-display text-2xl italic mt-1">{picked}</p>
            </div>
            <span className="eyebrow text-cobalt-500">Ready to index</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div><Label className="eyebrow">Family</Label><Input placeholder="leave_policy" className="mt-2 bg-paper-tint border-rule" /></div>
            <div><Label className="eyebrow">Version</Label><Input placeholder="2026" className="mt-2 bg-paper-tint border-rule" /></div>
            <div><Label className="eyebrow">Effective date</Label><Input type="date" className="mt-2 bg-paper-tint border-rule" /></div>
          </div>
          <div className="flex justify-end gap-2 mt-6">
            <Button variant="ghost" onClick={() => setPicked(null)}>Cancel</Button>
            <Button className="bg-ink text-paper hover:bg-cobalt-700">Upload & index</Button>
          </div>
        </div>
      )}

      <div className="mt-16 flex items-end justify-between gap-6 pb-4 border-b border-rule">
        <div>
          <span className="eyebrow">Chapter III</span>
          <h2 className="font-display text-4xl italic mt-2 leading-none">Your library</h2>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="h-3.5 w-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-ink-soft" />
            <Input placeholder="Filter…" className="pl-8 h-9 w-56 bg-card border-rule text-sm" />
          </div>
          <select className="h-9 px-3 rounded-md border border-rule bg-card text-xs font-mono uppercase tracking-widest text-ink-soft">
            <option>All families</option>
            <option>leave_policy</option>
            <option>contracts</option>
            <option>workplace</option>
          </select>
        </div>
      </div>

      <div className="divide-y divide-rule">
        {mockDocs.map((d, i) => (
          <Link key={d.id} to="/inspector" className="group grid grid-cols-12 gap-6 items-baseline py-7 hover:bg-paper-tint/40 transition-colors px-2 -mx-2 rounded">
            <span className="col-span-1 font-mono text-[10px] text-ink-soft tracking-widest pt-2">№ {String(i + 1).padStart(2, "0")}</span>
            <div className="col-span-12 md:col-span-6">
              <h3 className="font-display text-2xl leading-tight text-ink group-hover:italic transition-all">{d.title}</h3>
              <p className="text-sm text-ink-soft mt-2 line-clamp-1 leading-relaxed">{d.preview}</p>
              <div className="mt-3 font-mono text-[10px] uppercase tracking-widest text-ink-soft flex items-center gap-3">
                <span>{d.family}</span><span>·</span><span>v{d.version}</span><span>·</span><span>eff. {d.effectiveDate}</span>
              </div>
            </div>
            <div className="col-span-4 md:col-span-2 text-center md:text-right">
              <div className="font-display text-3xl italic text-ink leading-none">{d.pages}</div>
              <div className="eyebrow mt-1">pages</div>
            </div>
            <div className="col-span-4 md:col-span-1 text-center md:text-right">
              <div className="font-display text-3xl italic text-cobalt-500 leading-none">{d.chunks}</div>
              <div className="eyebrow mt-1">chunks</div>
            </div>
            <div className="col-span-4 md:col-span-2 md:text-right flex md:justify-end items-center gap-3">
              <StatusDot s={d.status} />
              <ArrowUpRight className="h-4 w-4 text-ink-soft group-hover:text-ink group-hover:-translate-y-0.5 group-hover:translate-x-0.5 transition-transform" />
            </div>
          </Link>
        ))}
      </div>

      <div className="mt-12 flex items-center justify-center">
        <button className="font-mono text-[10px] uppercase tracking-widest text-ink-soft hover:text-ink ink-link inline-flex items-center gap-2">
          <MoreHorizontal className="h-3 w-3" /> Load older documents
        </button>
      </div>
    </AppShell>
  );
}
