import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { AppShell } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { mockDocs } from "@/lib/mock-data";
import { Sparkles, Copy, Download, Mail, Loader2 } from "lucide-react";

export const Route = createFileRoute("/summarize")({
  head: () => ({ meta: [{ title: "Summarize — Vision" }, { name: "description", content: "Generate concise, detailed, or executive summaries of any document." }] }),
  component: Summarize,
});

const styles = ["Concise", "Detailed", "Executive"] as const;
type Style = typeof styles[number];

const summary = `The 2026 Maternity Leave Policy provides **26 weeks of fully paid leave** to eligible employees with at least 12 months of continuous service, plus an optional **13 weeks unpaid extension**.\n\n### Key entitlements\n- 100% base salary for weeks 1–26\n- Optional unpaid extension (request 4 weeks in advance)\n- Health benefits continued throughout the entire leave period\n\n### Eligibility\nEmployees with 12+ months tenure as of leave start date qualify for the full entitlement. Part-time staff receive proportionally adjusted pay.\n\n### Effective\nJanuary 1, 2026. Supersedes the 2025 policy for any leave commencing on or after the effective date.`;

function Summarize() {
  const [doc, setDoc] = useState(mockDocs[0].id);
  const [style, setStyle] = useState<Style>("Executive");
  const [state, setState] = useState<"idle" | "loading" | "done">("done");

  const generate = () => { setState("loading"); setTimeout(() => setState("done"), 1200); };
  const docData = mockDocs.find(d => d.id === doc)!;

  return (
    <AppShell
      eyebrow="Chapter IV · Distillation"
      title={<>Long documents, <em className="italic text-cobalt-500">made short.</em></>}
      breadcrumb={["Workspace", "Summarize"]}
    >
      <div className="grid grid-cols-12 gap-8">
        <aside className="col-span-12 lg:col-span-4 space-y-6">
          <div className="paper-card rounded-lg p-6">
            <div className="eyebrow mb-3">Document</div>
            <select value={doc} onChange={(e) => setDoc(e.target.value)} className="w-full px-3 py-3 rounded-md border border-rule bg-paper-tint font-display text-lg italic">
              {mockDocs.map((d) => <option key={d.id} value={d.id}>{d.title}</option>)}
            </select>

            <div className="hairline mt-6 pt-6">
              <div className="eyebrow mb-3">Style</div>
              <div className="space-y-px bg-rule border border-rule rounded-md overflow-hidden">
                {styles.map((s, i) => (
                  <button
                    key={s}
                    onClick={() => setStyle(s)}
                    className={`w-full text-left px-4 py-3 flex items-baseline gap-3 transition-colors ${style === s ? "bg-ink text-paper" : "bg-card hover:bg-paper-tint"}`}
                  >
                    <span className={`font-mono text-[10px] tracking-widest ${style === s ? "text-cobalt-300" : "text-ink-soft"}`}>0{i + 1}</span>
                    <div>
                      <div className="font-display text-lg italic leading-none">{s}</div>
                      <div className={`text-xs mt-1 ${style === s ? "text-paper/70" : "text-ink-soft"}`}>
                        {s === "Concise" ? "~5 bullets, 30 seconds to read" : s === "Detailed" ? "Full section coverage, ~3 minutes" : "Board-ready brief with key dates"}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <Button onClick={generate} className="w-full mt-6 h-12 bg-ink text-paper hover:bg-cobalt-700 font-mono text-[10px] uppercase tracking-widest">
              <Sparkles className="h-3 w-3 mr-2" /> Generate summary
            </Button>
            <p className="font-mono text-[10px] uppercase tracking-widest text-ink-soft mt-4 leading-relaxed">
              Long documents run async — leave the page and we'll have it waiting.
            </p>
          </div>
        </aside>

        <div className="col-span-12 lg:col-span-8">
          {state === "loading" ? (
            <div className="paper-card rounded-lg p-10 space-y-3">
              <div className="h-8 w-2/3 shimmer-ink bg-paper-tint rounded" />
              <div className="h-4 w-full shimmer-ink bg-paper-tint rounded" />
              <div className="h-4 w-5/6 shimmer-ink bg-paper-tint rounded" />
              <div className="h-4 w-4/6 shimmer-ink bg-paper-tint rounded" />
              <div className="flex items-center gap-2 text-sm text-cobalt-500 mt-4 font-mono uppercase tracking-widest text-[10px]">
                <Loader2 className="h-3 w-3 animate-spin" /> Distilling · 30%
              </div>
            </div>
          ) : state === "done" ? (
            <article className="paper-card rounded-lg p-10 lg:p-14 animate-fade-in">
              <header className="hairline pb-6 mb-8 border-t-0 border-b">
                <div className="eyebrow">{style} summary · 5 min ago</div>
                <h2 className="font-display text-4xl italic mt-3 leading-tight text-ink">{docData.title}</h2>
                <div className="mt-4 grid grid-cols-3 gap-4">
                  {[
                    { l: "Original", v: `${docData.chunks}`, u: "chunks" },
                    { l: "Compressed", v: "8", u: "key points" },
                    { l: "Coverage", v: "96", u: "%" },
                  ].map(s => (
                    <div key={s.l}>
                      <div className="eyebrow">{s.l}</div>
                      <div className="font-display text-3xl italic text-cobalt-500 mt-1 leading-none">{s.v}<span className="text-base text-ink-soft ml-1">{s.u}</span></div>
                    </div>
                  ))}
                </div>
              </header>

              <div className="prose prose-base max-w-none prose-headings:font-display prose-headings:italic prose-headings:text-ink prose-strong:text-cobalt-700 prose-p:leading-relaxed first-letter:font-display first-letter:text-6xl first-letter:italic first-letter:text-cobalt-500 first-letter:float-left first-letter:mr-3 first-letter:leading-none first-letter:mt-1">
                <ReactMarkdown>{summary}</ReactMarkdown>
              </div>

              <div className="hairline pt-5 mt-10 flex flex-wrap gap-2 items-center justify-between">
                <span className="font-mono text-[10px] uppercase tracking-widest text-ink-soft">2,140 tokens · gpt-4o-mini</span>
                <div className="flex gap-1">
                  <Button variant="ghost" size="sm" className="font-mono text-[10px] uppercase tracking-widest"><Copy className="h-3 w-3 mr-1" /> Copy</Button>
                  <Button variant="ghost" size="sm" className="font-mono text-[10px] uppercase tracking-widest"><Download className="h-3 w-3 mr-1" /> Export</Button>
                  <Button variant="ghost" size="sm" className="font-mono text-[10px] uppercase tracking-widest"><Mail className="h-3 w-3 mr-1" /> Email</Button>
                </div>
              </div>
            </article>
          ) : (
            <div className="paper-card rounded-lg p-20 text-center border-dashed">
              <Sparkles className="h-8 w-8 text-cobalt-300 mx-auto" />
              <p className="font-display text-2xl italic mt-4">Pick a document and style.</p>
            </div>
          )}
        </div>
      </div>
    </AppShell>
  );
}
