import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { AppShell } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChevronRight, ChevronLeft, Search } from "lucide-react";

export const Route = createFileRoute("/inspector")({
  head: () => ({ meta: [{ title: "Inspector — Vision" }, { name: "description", content: "Inspect document structure, navigate by section, and view raw page content." }] }),
  component: Inspector,
});

interface Node { id: string; label: string; pages: string; children?: Node[]; }

const tree: Node[] = [
  { id: "1", label: "Cover & Table of Contents", pages: "1–2" },
  { id: "2", label: "§1. Purpose & Scope", pages: "3" },
  { id: "3", label: "§2. Definitions", pages: "3–4" },
  { id: "4", label: "§3. Entitlements", pages: "4–8", children: [
    { id: "4.1", label: "§3.1 Paid maternity leave", pages: "4–5" },
    { id: "4.2", label: "§3.2 Adoption leave", pages: "5–6" },
    { id: "4.3", label: "§3.3 Shared parental leave", pages: "6–7" },
    { id: "4.4", label: "§3.4 Extensions & unpaid leave", pages: "7–8" },
  ]},
  { id: "5", label: "§4. Application process", pages: "9–10" },
  { id: "6", label: "Appendix A — Eligibility", pages: "11" },
  { id: "7", label: "Appendix B — Pay calculation", pages: "12" },
];

const pageText = `§3.1 Paid maternity entitlement

Eligible employees are entitled to twenty-six (26) weeks of fully paid
maternity leave. The paid period commences on the day the employee elects
to begin her leave, which must fall within the eleven (11) weeks preceding
the expected date of delivery and the date of delivery itself.

3.1.1 Salary. During the paid period, the Company shall pay the employee
one hundred percent (100%) of her base salary, less applicable statutory
deductions. Variable compensation, including bonuses and commissions, shall
accrue in accordance with the relevant incentive plan.

3.1.2 Benefits. All health, dental, vision, and life insurance benefits
shall continue during the paid leave period at the same employer-paid
contribution rates that applied immediately prior to the leave.

3.1.3 Eligibility. To qualify for the paid entitlement set forth in this
Section 3.1, the employee must have completed twelve (12) months of
continuous service with the Company as of the leave commencement date.`;

function TreeItem({ node, depth = 0, active, onPick }: { node: Node; depth?: number; active: string; onPick: (id: string) => void }) {
  const [open, setOpen] = useState(true);
  const hasChildren = !!node.children?.length;
  const isActive = active === node.id;
  return (
    <div>
      <button
        onClick={() => { onPick(node.id); if (hasChildren) setOpen(!open); }}
        className={`group w-full text-left flex items-baseline gap-2 py-2 transition-colors ${isActive ? "text-ink" : "text-ink-soft hover:text-ink"}`}
        style={{ paddingLeft: depth * 16 }}
      >
        <span className="font-mono text-[10px] text-ink-soft/60 w-8 shrink-0">p.{node.pages}</span>
        <span className={`font-display leading-snug flex-1 ${depth === 0 ? "text-lg" : "text-base"} ${isActive ? "italic text-cobalt-500" : ""}`}>{node.label}</span>
        {hasChildren && <ChevronRight className={`h-3 w-3 shrink-0 mt-1 transition-transform ${open ? "rotate-90" : ""}`} />}
      </button>
      {hasChildren && open && (
        <div className="border-l border-rule ml-2 mt-1 mb-1">{node.children!.map((c) => <TreeItem key={c.id} node={c} depth={depth + 1} active={active} onPick={onPick} />)}</div>
      )}
    </div>
  );
}

function Inspector() {
  const [active, setActive] = useState("4.1");
  const [page, setPage] = useState(4);

  return (
    <AppShell
      eyebrow="Chapter V · Inspection"
      title={<>The document <em className="italic text-cobalt-500">x-ray.</em></>}
      breadcrumb={["Workspace", "Inspector", "Maternity Leave Policy 2026"]}
    >
      <div className="grid grid-cols-12 gap-8">
        <aside className="col-span-12 lg:col-span-4 lg:sticky lg:top-20 h-fit">
          <div className="paper-card rounded-lg p-6">
            <div className="eyebrow">Document</div>
            <h3 className="font-display text-2xl italic mt-1 leading-tight">Maternity Leave Policy 2026</h3>
            <p className="font-mono text-[10px] uppercase tracking-widest text-ink-soft mt-2">12 pages · 142 chunks · v2026</p>

            <div className="relative mt-5">
              <Search className="h-3 w-3 absolute left-3 top-1/2 -translate-y-1/2 text-ink-soft" />
              <Input placeholder="Filter sections…" className="pl-8 h-9 bg-paper-tint border-rule text-sm" />
            </div>

            <div className="eyebrow mt-6 mb-2">Table of contents</div>
            <div className="divide-y divide-rule">
              {tree.map((n) => <div key={n.id} className="py-1"><TreeItem node={n} active={active} onPick={setActive} /></div>)}
            </div>
          </div>
        </aside>

        <div className="col-span-12 lg:col-span-8">
          <div className="paper-card rounded-lg overflow-hidden">
            <div className="flex items-center justify-between px-7 py-4 hairline border-t-0 border-b">
              <div className="flex items-center gap-3">
                <Button variant="ghost" size="icon" onClick={() => setPage((p) => Math.max(1, p - 1))}><ChevronLeft className="h-4 w-4" /></Button>
                <div className="text-center">
                  <div className="eyebrow">Page</div>
                  <div className="font-display text-xl italic leading-none">{page} <span className="text-ink-soft text-sm">/ 12</span></div>
                </div>
                <Button variant="ghost" size="icon" onClick={() => setPage((p) => Math.min(12, p + 1))}><ChevronRight className="h-4 w-4" /></Button>
              </div>
              <div className="font-mono text-[10px] uppercase tracking-widest text-ink-soft text-right">
                <div>487 tokens</div>
                <div>hash 8c4f…a21</div>
              </div>
            </div>
            <div className="p-10 lg:p-14 bg-paper relative grain">
              <span className="absolute top-6 right-6 font-display text-7xl italic text-cobalt-500/15 leading-none select-none">{page}</span>
              <pre className="font-mono text-[13px] leading-7 whitespace-pre-wrap text-ink relative">{pageText}</pre>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
