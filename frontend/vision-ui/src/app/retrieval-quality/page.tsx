"use client";

import { useEffect, useState } from "react";
import { AppShell } from "@/components/layout/app-shell";
import { getBenchmarkResults } from "@/lib/api";

type BenchmarkRow = {
  strategy: string;
  page_recall?: number;
  page_precision?: number;
  hit_rate?: number;
  mrr?: number;
  ndcg?: number;
  grounded_rate?: number;
  avg_confidence?: number;
  out_of_scope_accuracy?: number;
  avg_latency_ms?: number;
  total_tokens?: number;
  wall_seconds?: number;
  attribution?: Record<string, number>;
  error?: string;
};

const STAT_FIELDS: { key: keyof BenchmarkRow; label: string; pct: boolean }[] = [
  { key: "page_recall", label: "Page Recall", pct: true },
  { key: "page_precision", label: "Page Precision", pct: true },
  { key: "hit_rate", label: "Hit Rate", pct: true },
  { key: "mrr", label: "MRR", pct: true },
  { key: "ndcg", label: "nDCG", pct: true },
  { key: "grounded_rate", label: "Grounded Rate", pct: true },
  { key: "avg_confidence", label: "Avg Confidence", pct: true },
  { key: "out_of_scope_accuracy", label: "OOS Accuracy", pct: true },
  { key: "avg_latency_ms", label: "Avg Latency", pct: false },
];

export default function RetrievalQualityPage() {
  const [rows, setRows] = useState<BenchmarkRow[] | null>(null);
  const [available, setAvailable] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await getBenchmarkResults();
        setAvailable(!!res.available);
        setRows(res.rows || []);
      } catch (e: any) {
        setError(e.message || "Failed to reach the backend.");
      }
    }
    load();
  }, []);

  return (
    <AppShell
      eyebrow="Chapter IX · Retrieval Quality"
      title={<>Measure it, <em className="italic text-cobalt-500">don&apos;t guess it.</em></>}
      breadcrumb={["Workspace", "Retrieval Quality"]}
    >
      <p className="max-w-2xl text-ink-soft text-lg mb-6 -mt-2">
        Golden-set benchmark output from <code className="font-mono text-sm">evals/benchmark.py</code> —
        real recall/precision/grounding numbers per retrieval strategy, read
        straight from <code className="font-mono text-sm">evals/results/benchmark.json</code>. Nothing
        on this page is simulated: if the benchmark hasn&apos;t been run, that&apos;s shown honestly below.
      </p>

      {error && (
        <div className="mb-6 px-4 py-3 rounded-md bg-destructive/10 border border-destructive/30 text-destructive text-sm font-mono">
          {error}
        </div>
      )}

      {!error && rows !== null && !available && (
        <div className="mb-6 px-4 py-3 rounded-md bg-amber-50 border border-amber-200 text-amber-900 text-sm font-mono flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-amber-500 shrink-0" />
          No benchmark results yet. Index a document, then run{" "}
          <code>python -m evals.benchmark</code> to populate this page.
        </div>
      )}

      {available && rows && rows.length > 0 && (
        <div className="flex flex-col gap-6">
          {rows.map((row) => (
            <div key={row.strategy} className="paper-card rounded-lg p-6">
              <div className="font-display text-2xl italic mb-4">{row.strategy}</div>
              {row.error ? (
                <div className="font-mono text-sm text-destructive">{row.error}</div>
              ) : (
                <div className="grid grid-cols-2 lg:grid-cols-6 gap-px bg-rule border border-rule rounded-lg overflow-hidden">
                  {STAT_FIELDS.map((f) => {
                    const v = row[f.key];
                    const display =
                      v === undefined
                        ? "—"
                        : f.pct
                          ? `${(Number(v) * 100).toFixed(0)}%`
                          : `${Number(v).toFixed(0)}ms`;
                    return (
                      <div key={f.key} className="bg-card p-5">
                        <div className="eyebrow">{f.label}</div>
                        <div className="font-display text-2xl italic mt-2 leading-none">{display}</div>
                      </div>
                    );
                  })}
                </div>
              )}
              {row.attribution && (
                <div className="mt-4">
                  <div className="font-mono text-[10px] uppercase tracking-widest text-ink-soft mb-2">
                    Retrieval-vs-generation attribution
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(row.attribution)
                      .filter(([, count]) => Number(count) > 0)
                      .map(([verdict, count]) => (
                        <span
                          key={verdict}
                          className="font-mono text-[10px] px-2 py-1 rounded border border-rule bg-paper-tint"
                        >
                          {verdict}: {String(count)}
                        </span>
                      ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </AppShell>
  );
}
