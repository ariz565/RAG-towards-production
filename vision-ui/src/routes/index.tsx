import { createFileRoute, Link } from "@tanstack/react-router";
import { AppShell } from "@/components/app-shell";
import { mockDocs, mockQueries } from "@/lib/mock-data";
import { ArrowUpRight, ArrowRight } from "lucide-react";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Overview — Vision" },
      { name: "description", content: "Your document intelligence workspace at a glance." },
    ],
  }),
  component: Dashboard,
});

function Metric({ figure, label, sub, accent }: { figure: string; label: string; sub: string; accent?: boolean }) {
  return (
    <div className="py-7 first:pt-0">
      <div className="eyebrow">{label}</div>
      <div className={`mt-3 font-display text-[64px] leading-none tracking-tight ${accent ? "text-cobalt-500 italic" : "text-ink"}`}>{figure}</div>
      <div className="mt-2 text-sm text-ink-soft">{sub}</div>
    </div>
  );
}

function Dashboard() {
  const time = new Date().toLocaleString("en-US", { weekday: "long", month: "long", day: "numeric" });

  return (
    <AppShell
      eyebrow={`Edition № 142 · ${time}`}
      title={<>Good morning, Alex. <em className="italic text-cobalt-500">Your library is listening.</em></>}
    >
      <p className="max-w-2xl text-lg text-ink-soft leading-relaxed -mt-2 mb-12">
        Four documents indexed across three families. Hybrid retrieval is warm, gpt-4o-mini is on call, and your last 128 questions were answered with an average confidence of <span className="text-ink font-medium">87%</span>.
      </p>

      <div className="grid grid-cols-12 gap-8">
        <aside className="col-span-12 lg:col-span-4 divide-y divide-rule border-y border-rule">
          <Metric figure="04" label="Indexed documents" sub="3 ready · 1 indexing" />
          <Metric figure="128" label="Questions this week" accent sub="+24% vs last week" />
          <Metric figure="87%" label="Average confidence" sub="across 128 answers" />
        </aside>

        <section className="col-span-12 lg:col-span-8">
          <div className="flex items-baseline justify-between mb-6">
            <h2 className="font-display text-3xl italic">In your library</h2>
            <Link to="/documents" className="text-xs font-mono uppercase tracking-widest text-ink-soft hover:text-ink ink-link">View all →</Link>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-rule border border-rule">
            {mockDocs.slice(0, 4).map((d, i) => (
              <Link
                key={d.id}
                to="/chat"
                className="group block p-7 bg-card hover:bg-paper-tint transition-colors relative"
              >
                <div className="flex items-start justify-between">
                  <span className="font-mono text-[10px] tracking-widest text-ink-soft">№ {String(i + 1).padStart(2, "0")} · {d.family}</span>
                  <ArrowUpRight className="h-4 w-4 text-ink-soft group-hover:text-cobalt-500 group-hover:-translate-y-0.5 group-hover:translate-x-0.5 transition-transform" />
                </div>
                <h3 className="font-display text-2xl leading-tight mt-3 text-ink">{d.title}</h3>
                <p className="text-sm text-ink-soft mt-3 line-clamp-2 leading-relaxed">{d.preview}</p>
                <div className="mt-5 pt-4 border-t border-rule flex items-center justify-between font-mono text-[10px] tracking-wider text-ink-soft uppercase">
                  <span>{d.pages} pages · {d.chunks} chunks</span>
                  <span className={d.status === "indexed" ? "text-emerald-700" : "text-cobalt-500"}>● {d.status}</span>
                </div>
              </Link>
            ))}
          </div>
        </section>
      </div>

      <div className="mt-20 mb-10 flex items-center gap-6">
        <span className="eyebrow">Chapter II</span>
        <span className="flex-1 hairline" />
        <span className="font-mono text-[10px] text-ink-soft">Recent inquiries</span>
      </div>

      <section className="border-t border-b border-rule">
        {mockQueries.map((q, i) => (
          <Link
            key={q.id}
            to="/chat"
            className="group grid grid-cols-12 gap-4 items-baseline py-6 border-b border-rule last:border-0 hover:bg-paper-tint/50 transition-colors px-2"
          >
            <span className="col-span-1 font-mono text-[10px] text-ink-soft tracking-widest pt-1">{String(i + 1).padStart(2, "0")}</span>
            <div className="col-span-12 md:col-span-8">
              <h4 className="font-display text-xl leading-snug text-ink group-hover:italic transition-all">{q.query}</h4>
              <div className="mt-2 font-mono text-[10px] uppercase tracking-widest text-ink-soft flex items-center gap-3">
                <span>{q.doc}</span><span>·</span><span>{q.ago}</span>
              </div>
            </div>
            <div className="col-span-12 md:col-span-3 md:text-right flex md:justify-end items-center gap-3">
              <div className="flex flex-col items-end">
                <span className="font-display text-3xl italic text-cobalt-500 leading-none">{Math.round(q.confidence * 100)}<span className="text-base">%</span></span>
                <span className="eyebrow mt-1">confidence</span>
              </div>
              <ArrowRight className="h-4 w-4 text-ink-soft group-hover:text-ink group-hover:translate-x-1 transition-transform" />
            </div>
          </Link>
        ))}
      </section>

      <section className="mt-20 grid grid-cols-12 gap-6">
        <div className="col-span-12 lg:col-span-7 bg-ink text-paper p-10 lg:p-14 rounded-lg grain relative overflow-hidden">
          <span className="eyebrow text-paper/60 relative">Ask anything</span>
          <h3 className="font-display text-4xl lg:text-5xl mt-4 leading-tight max-w-md relative">
            <em className="italic text-cobalt-300">Speak</em> to your documents like they were colleagues.
          </h3>
          <Link to="/chat" className="mt-8 inline-flex items-center gap-3 text-paper font-mono text-xs uppercase tracking-widest border-b border-paper/40 pb-1 hover:border-cobalt-300 hover:text-cobalt-300 relative">
            Open the composer <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
        <div className="col-span-12 lg:col-span-5 paper-card rounded-lg p-10 flex flex-col">
          <span className="eyebrow">Add to your library</span>
          <h3 className="font-display text-3xl italic mt-3 leading-tight">A new PDF takes about 14 seconds to index.</h3>
          <Link to="/documents" className="mt-auto pt-8 inline-flex items-center gap-2 text-ink font-mono text-xs uppercase tracking-widest ink-link self-start">
            Upload document →
          </Link>
        </div>
      </section>
    </AppShell>
  );
}
