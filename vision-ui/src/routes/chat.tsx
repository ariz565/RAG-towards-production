import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { AppShell } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { mockCitations, mockPipeline, mockQueries } from "@/lib/mock-data";
import { Send, ChevronDown, Copy, ThumbsUp, ThumbsDown, Settings2, Sparkles } from "lucide-react";

export const Route = createFileRoute("/chat")({
  head: () => ({ meta: [{ title: "Ask — Vision" }, { name: "description", content: "Ask your documents anything. Get grounded answers with citations." }] }),
  component: Chat,
});

const suggestions = [
  "How many weeks of paid maternity leave do eligible employees receive?",
  "Compare maternity and paternity entitlements in detail.",
  "What are the key dates in the Acme Corp MSA?",
  "How much is the remote work stipend, and how is it reimbursed?",
];

function Chat() {
  const [query, setQuery] = useState("");
  const [showAnswer, setShowAnswer] = useState(true);
  const [advanced, setAdvanced] = useState(false);
  const [pipelineOpen, setPipelineOpen] = useState(true);

  const answer = `Eligible employees receive **26 weeks of fully paid maternity leave**, with an option to extend by an additional **13 weeks unpaid**. Leave begins on a date selected by the employee within 11 weeks of the expected due date.\n\n- Paid portion: weeks 1–26 at 100% base salary\n- Unpaid extension: optional, must be requested 4 weeks in advance\n- Eligibility: 12+ months of continuous service`;

  return (
    <AppShell
      eyebrow="Chapter III · Inquiry"
      title={<>Ask, and your library <em className="italic text-cobalt-500">answers.</em></>}
      breadcrumb={["Workspace", "Ask"]}
    >
      <div className="grid grid-cols-12 gap-8">
        <div className="col-span-12 lg:col-span-5 space-y-6">
          <div className="paper-card rounded-lg p-6 grain relative">
            <div className="relative">
              <div className="eyebrow mb-3">Your question</div>
              <Textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What would you like to know?"
                rows={5}
                className="resize-none border-0 focus-visible:ring-0 px-0 text-xl font-display leading-snug placeholder:text-ink-soft/50 bg-transparent shadow-none"
              />
              <div className="hairline pt-4 mt-3 flex items-center justify-between">
                <button onClick={() => setAdvanced(!advanced)} className="font-mono text-[10px] uppercase tracking-widest text-ink-soft hover:text-ink inline-flex items-center gap-1.5">
                  <Settings2 className="h-3 w-3" /> Advanced
                </button>
                <Button onClick={() => setShowAnswer(true)} className="bg-ink text-paper hover:bg-cobalt-700 rounded-md font-mono text-[10px] uppercase tracking-widest">
                  Ask <Send className="h-3 w-3 ml-1" />
                </Button>
              </div>
            </div>
          </div>

          <div>
            <div className="eyebrow mb-3">Try asking</div>
            <div className="space-y-px bg-rule border border-rule rounded-md overflow-hidden">
              {suggestions.map((s, i) => (
                <button key={s} onClick={() => setQuery(s)} className="w-full text-left bg-card hover:bg-paper-tint px-4 py-3 flex items-start gap-3 transition-colors group">
                  <span className="font-mono text-[10px] text-ink-soft tracking-widest pt-1">0{i + 1}</span>
                  <span className="text-sm leading-snug group-hover:text-ink text-ink-soft">{s}</span>
                </button>
              ))}
            </div>
          </div>

          {advanced && (
            <div className="paper-card rounded-lg p-5 space-y-4 animate-fade-in">
              <div className="eyebrow">Parameters</div>
              {[
                { label: "Strategy", opts: ["hybrid (BM25 + vector)", "pageindex", "vector_only", "bm25_only"] },
                { label: "Model", opts: ["gpt-4o-mini", "gpt-4o", "claude-3.5-sonnet", "llama-3.1-70b"] },
                { label: "Pin to document", opts: ["Any document", "Maternity Leave Policy 2026", "Remote Work Handbook"] },
              ].map((f) => (
                <div key={f.label} className="grid grid-cols-[120px_1fr] items-center gap-3">
                  <span className="font-mono text-[10px] uppercase tracking-widest text-ink-soft">{f.label}</span>
                  <select className="px-3 py-2 rounded-md border border-rule bg-paper-tint text-sm">{f.opts.map(o => <option key={o}>{o}</option>)}</select>
                </div>
              ))}
              <div className="hairline pt-3 flex flex-col gap-2 text-sm">
                <label className="flex items-center gap-2"><input type="checkbox" defaultChecked className="accent-cobalt-500" /> Stream response</label>
                <label className="flex items-center gap-2"><input type="checkbox" defaultChecked className="accent-cobalt-500" /> Show pipeline trace</label>
              </div>
            </div>
          )}

          <div>
            <div className="eyebrow mb-3">In this session</div>
            <div className="divide-y divide-rule border-y border-rule">
              {mockQueries.map((q, i) => (
                <button key={q.id} className="w-full text-left py-3 flex items-start gap-3 hover:bg-paper-tint/40 px-2 -mx-2 rounded">
                  <span className="font-mono text-[10px] text-ink-soft tracking-widest pt-1">{String(i + 1).padStart(2, "0")}</span>
                  <div className="flex-1 min-w-0">
                    <p className="font-display text-base leading-snug truncate">{q.query}</p>
                    <p className="font-mono text-[10px] uppercase tracking-widest text-ink-soft mt-1">{q.ago} · {Math.round(q.confidence * 100)}%</p>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="col-span-12 lg:col-span-7 space-y-6">
          {!showAnswer ? (
            <div className="paper-card rounded-lg p-20 text-center border-dashed">
              <Sparkles className="h-8 w-8 text-cobalt-300 mx-auto" />
              <p className="font-display text-2xl italic mt-4">An answer will appear here.</p>
            </div>
          ) : (
            <>
              <article className="paper-card rounded-lg p-10 animate-fade-in relative">
                <header className="hairline pb-5 mb-7 border-t-0 border-b">
                  <div className="flex items-center justify-between flex-wrap gap-2">
                    <span className="eyebrow">Answer · gpt-4o-mini · 1.83s</span>
                    <div className="flex items-center gap-3 font-mono text-[10px] uppercase tracking-widest">
                      <span className="text-emerald-700">● 92% confident</span>
                      <span className="text-cobalt-500">● fully grounded</span>
                    </div>
                  </div>
                  <h3 className="font-display text-3xl italic mt-4 leading-tight">
                    On the duration of paid maternity leave.
                  </h3>
                  <p className="text-xs text-ink-soft mt-1 font-mono">Interpreted as a factual lookup against <span className="text-ink">Maternity Leave Policy 2026</span></p>
                </header>

                <div className="prose prose-base max-w-none prose-headings:font-display prose-strong:text-cobalt-700 prose-strong:font-semibold prose-p:text-ink prose-li:text-ink-soft prose-li:my-1 prose-p:leading-relaxed first-letter:font-display first-letter:text-5xl first-letter:italic first-letter:text-cobalt-500 first-letter:float-left first-letter:mr-2 first-letter:leading-none first-letter:mt-1">
                  <ReactMarkdown>{answer}</ReactMarkdown>
                </div>

                <div className="hairline pt-4 mt-8 flex items-center justify-between">
                  <span className="font-mono text-[10px] uppercase tracking-widest text-ink-soft">1,240 tokens · hybrid retrieval</span>
                  <div className="flex gap-1">
                    <Button variant="ghost" size="sm" className="font-mono text-[10px] uppercase tracking-widest text-ink-soft"><Copy className="h-3 w-3 mr-1" /> Copy</Button>
                    <Button variant="ghost" size="icon"><ThumbsUp className="h-3.5 w-3.5" /></Button>
                    <Button variant="ghost" size="icon"><ThumbsDown className="h-3.5 w-3.5" /></Button>
                  </div>
                </div>
              </article>

              <section className="paper-card rounded-lg p-7">
                <div className="flex items-baseline justify-between mb-5">
                  <div>
                    <span className="eyebrow">Sources</span>
                    <h4 className="font-display text-2xl italic mt-1">Cited in the answer</h4>
                  </div>
                  <span className="font-mono text-[10px] tracking-widest text-ink-soft">№ {mockCitations.length}</span>
                </div>
                <div className="divide-y divide-rule">
                  {mockCitations.map((c, i) => (
                    <div key={c.id} className="py-4 grid grid-cols-[28px_1fr_auto] gap-4 items-baseline">
                      <span className="font-mono text-[10px] text-cobalt-500 tracking-widest pt-1">[{i + 1}]</span>
                      <div>
                        <p className="font-display text-lg leading-snug">{c.section}</p>
                        <p className="text-xs text-ink-soft mt-1 leading-relaxed">{c.relevance}</p>
                      </div>
                      <div className="flex gap-1">
                        {c.pages.map((p) => (
                          <span key={p} className="px-2 py-1 rounded border border-rule bg-paper-tint text-[10px] font-mono">p.{p}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              <section className="paper-card rounded-lg overflow-hidden">
                <button onClick={() => setPipelineOpen(!pipelineOpen)} className="w-full px-7 py-5 flex items-center justify-between hover:bg-paper-tint">
                  <div className="text-left">
                    <div className="eyebrow">Trace</div>
                    <span className="font-display text-xl italic mt-1 block">Pipeline · 3 steps · 1.83s</span>
                  </div>
                  <ChevronDown className={`h-4 w-4 text-ink-soft transition-transform ${pipelineOpen ? "rotate-180" : ""}`} />
                </button>
                {pipelineOpen && (
                  <div className="px-7 pb-7 hairline pt-5">
                    {mockPipeline.map((s, i) => (
                      <div key={s.id} className="grid grid-cols-[40px_1fr] gap-4 py-4 border-b border-rule last:border-0">
                        <div className="flex flex-col items-center">
                          <div className="h-8 w-8 rounded-full bg-ink text-paper grid place-items-center font-mono text-xs">{i + 1}</div>
                          {i < mockPipeline.length - 1 && <div className="w-px flex-1 bg-rule mt-2 min-h-4" />}
                        </div>
                        <div>
                          <div className="flex items-baseline justify-between gap-3">
                            <p className="font-display text-lg leading-tight">{s.name}</p>
                            <span className="font-mono text-[10px] uppercase tracking-widest text-ink-soft shrink-0">{s.duration}ms{s.tokens ? ` · ${s.tokens} tok` : ""}</span>
                          </div>
                          <p className="text-sm text-ink-soft mt-1 leading-relaxed">{s.thinking}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            </>
          )}
        </div>
      </div>
    </AppShell>
  );
}
