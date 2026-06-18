import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { AppShell } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { CheckCircle2, Circle } from "lucide-react";

export const Route = createFileRoute("/config")({
  head: () => ({ meta: [{ title: "Configuration — Vision" }, { name: "description", content: "Configure models, retrieval strategy, and embeddings for your workspace." }] }),
  component: Config,
});

function Section({ num, title, sub, children }: { num: string; title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section className="grid grid-cols-12 gap-6 py-12 border-b border-rule">
      <div className="col-span-12 lg:col-span-4">
        <div className="font-mono text-[10px] uppercase tracking-widest text-cobalt-500">§ {num}</div>
        <h3 className="font-display text-3xl italic mt-2 leading-tight">{title}</h3>
        {sub && <p className="text-sm text-ink-soft mt-3 leading-relaxed max-w-xs">{sub}</p>}
      </div>
      <div className="col-span-12 lg:col-span-8">{children}</div>
    </section>
  );
}

function Field({ label, value, options, onChange, hint }: { label: string; value: string; options: string[]; onChange: (v: string) => void; hint?: string }) {
  return (
    <div className="py-4 border-b border-rule last:border-0 grid grid-cols-12 items-baseline gap-4">
      <div className="col-span-12 md:col-span-5">
        <div className="font-display text-lg">{label}</div>
        {hint && <div className="text-xs text-ink-soft mt-1">{hint}</div>}
      </div>
      <div className="col-span-12 md:col-span-7">
        <select value={value} onChange={(e) => onChange(e.target.value)} className="w-full px-3 py-2.5 rounded-md border border-rule bg-paper-tint font-mono text-sm">
          {options.map((o) => <option key={o}>{o}</option>)}
        </select>
      </div>
    </div>
  );
}

function Config() {
  const [model, setModel] = useState("gpt-4o-mini");
  const [strategy, setStrategy] = useState("hybrid");
  const [embedding, setEmbedding] = useState("huggingface");

  return (
    <AppShell
      eyebrow="Chapter VI · Apparatus"
      title={<>The machinery, <em className="italic text-cobalt-500">on your terms.</em></>}
      breadcrumb={["Workspace", "Configuration"]}
    >
      <p className="text-lg text-ink-soft max-w-2xl -mt-2 mb-4 leading-relaxed">
        Every knob and dial Vision exposes. Changes apply immediately to new queries; historical answers remain untouched.
      </p>

      <Section num="01" title="Runtime switches" sub="Mutable at any time. New questions adopt the next tick.">
        <div className="paper-card rounded-lg p-6">
          <Field label="Active model" hint="The LLM that synthesizes every answer." value={model} onChange={setModel} options={["gpt-4o-mini", "gpt-4o", "claude-3.5-sonnet", "llama-3.1-70b"]} />
          <Field label="Retrieval strategy" hint="How relevant chunks are surfaced from your library." value={strategy} onChange={setStrategy} options={["hybrid", "pageindex", "vector_only", "bm25_only"]} />
          <Field label="Embedding provider" hint="Which vectors back your semantic search." value={embedding} onChange={setEmbedding} options={["huggingface", "openai", "ollama"]} />
          <div className="flex justify-end pt-5">
            <Button className="bg-ink text-paper hover:bg-cobalt-700 font-mono text-[10px] uppercase tracking-widest">Save changes</Button>
          </div>
        </div>
      </Section>

      <Section num="02" title="Credentials" sub="API keys are environment-scoped. Backend restart applies any change.">
        <div className="paper-card rounded-lg p-6 divide-y divide-rule">
          {[
            ["OpenAI", true, "sk_live_••••wxyz"],
            ["Anthropic", true, "sk_ant_••••abcd"],
            ["HuggingFace", true, "hf_••••mnop"],
            ["Cohere", false, ""],
          ].map(([n, ok, k]) => (
            <div key={n as string} className="py-4 flex items-center justify-between">
              <div>
                <div className="font-display text-lg">{n}</div>
                {ok ? <div className="font-mono text-[10px] text-ink-soft mt-1">{k as string}</div> : <div className="font-mono text-[10px] text-ink-soft mt-1">Add a key to enable</div>}
              </div>
              {ok ? (
                <span className="inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-widest text-emerald-700"><CheckCircle2 className="h-3 w-3" /> Configured</span>
              ) : (
                <Button variant="outline" size="sm" className="font-mono text-[10px] uppercase tracking-widest"><Circle className="h-3 w-3 mr-1" /> Add key</Button>
              )}
            </div>
          ))}
        </div>
      </Section>

      <Section num="03" title="Health & telemetry" sub="A glance at the backend powering this workspace.">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-px bg-rule border border-rule rounded-lg overflow-hidden">
          {[
            { l: "Status", v: "OK", a: true },
            { l: "API version", v: "v1.4.2" },
            { l: "Avg latency", v: "182ms" },
            { l: "Vector dim", v: "1024" },
            { l: "Documents", v: "04" },
            { l: "BM25 size", v: "8.4 MB" },
            { l: "Qdrant", v: "214 MB" },
            { l: "Uptime", v: "47d" },
          ].map((s) => (
            <div key={s.l} className="bg-card p-5">
              <div className="eyebrow">{s.l}</div>
              <div className={`font-display text-3xl italic mt-2 leading-none ${s.a ? "text-emerald-700" : "text-ink"}`}>{s.v}</div>
            </div>
          ))}
        </div>
      </Section>

      <Section num="04" title="Danger zone" sub="Irreversible operations. Type the tenant ID to confirm.">
        <div className="paper-card rounded-lg p-6 border-destructive/30 bg-destructive/5">
          <div className="font-display text-xl">Re-index entire library</div>
          <p className="text-sm text-ink-soft mt-1">Rebuild BM25 and vector indexes from scratch. Takes ~3 minutes per 100 pages.</p>
          <Button variant="outline" className="mt-4 border-destructive/40 text-destructive hover:bg-destructive/10 font-mono text-[10px] uppercase tracking-widest">Re-index all</Button>
        </div>
      </Section>
    </AppShell>
  );
}
