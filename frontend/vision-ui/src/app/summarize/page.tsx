"use client";

import { useState, useEffect } from "react";
import { AppShell } from "@/components/layout/app-shell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { summarizeAsync, getJob, getDocuments } from "@/lib/api";
import { Loader2, FileText, CheckCircle2 } from "lucide-react";
import ReactMarkdown from "react-markdown";

export default function SummarizePage() {
  const [docs, setDocs] = useState<any[]>([]);
  const [docId, setDocId] = useState("");
  const [style, setStyle] = useState("concise");
  
  const [jobId, setJobId] = useState("");
  const [jobStatus, setJobStatus] = useState("");
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    // load docs
    getDocuments().then(res => {
      if (res.documents) setDocs(res.documents);
    }).catch(e => console.error(e));
  }, []);

  useEffect(() => {
    let interval: any;
    if (jobId && (jobStatus === "pending" || jobStatus === "running")) {
      interval = setInterval(async () => {
        try {
          const res = await getJob(jobId);
          setJobStatus(res.status);
          if (res.status === "done") {
            setSummary(res.result?.summary || "");
          } else if (res.status === "failed") {
            setError(res.error || "Job failed");
          }
        } catch (e) {
          console.error(e);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [jobId, jobStatus]);

  const handleStart = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!docId) return;
    setError("");
    setSummary("");
    setLoading(true);
    try {
      const res = await summarizeAsync({ doc_id: docId, style });
      setJobId(res.job_id);
      setJobStatus("pending");
    } catch (err: any) {
      setError(err.message || "Failed to start summarization");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AppShell eyebrow="Chapter IV" title="Summarize" breadcrumb={["Workspace", "Summarize"]}>
      <p className="max-w-2xl text-ink-soft text-lg mb-10 -mt-2">Map-reduce engine for extensive documents. Select a document and let the backend read every page to build a structured summary.</p>
      
      <div className="grid grid-cols-12 gap-8">
        <div className="col-span-12 lg:col-span-4">
          <div className="paper-card rounded-lg p-6">
            <h3 className="font-display text-2xl italic mb-6">Configuration</h3>
            <form onSubmit={handleStart} className="space-y-4">
              <div>
                <Label className="eyebrow">Document</Label>
                <select value={docId} onChange={e => setDocId(e.target.value)} required className="mt-2 w-full px-3 py-2.5 rounded-md border border-rule bg-paper-tint font-mono text-sm">
                  <option value="" disabled>Select a document...</option>
                  {docs.map(d => (
                    <option key={d.doc_id || d.id} value={d.doc_id || d.id}>{d.title || d.doc_id}</option>
                  ))}
                </select>
              </div>
              <div>
                <Label className="eyebrow">Style</Label>
                <select value={style} onChange={e => setStyle(e.target.value)} className="mt-2 w-full px-3 py-2.5 rounded-md border border-rule bg-paper-tint font-mono text-sm">
                  <option value="concise">Concise</option>
                  <option value="detailed">Detailed</option>
                  <option value="bullet points">Bullet Points</option>
                </select>
              </div>
              <div className="pt-4">
                <Button type="submit" disabled={loading || jobStatus === "pending" || jobStatus === "running"} className="w-full bg-ink text-paper hover:bg-cobalt-700 font-mono text-[10px] uppercase tracking-widest">
                  Start Map-Reduce
                </Button>
              </div>
            </form>
          </div>
        </div>

        <div className="col-span-12 lg:col-span-8">
          {error && (
            <div className="paper-card rounded-lg p-6 bg-destructive/10 border-destructive/20 text-destructive mb-6">
              <div className="font-mono text-xs uppercase tracking-widest mb-1">Error</div>
              <div>{error}</div>
            </div>
          )}

          {!jobId && !summary && !error && (
            <div className="paper-card rounded-lg p-20 text-center border-dashed">
              <FileText className="h-8 w-8 text-ink-soft/40 mx-auto" />
              <p className="font-display text-2xl italic mt-4 text-ink-soft">Select a document to begin.</p>
            </div>
          )}

          {jobId && (jobStatus === "pending" || jobStatus === "running") && (
            <div className="paper-card rounded-lg p-14 text-center animate-fade-in relative overflow-hidden">
              <Loader2 className="h-8 w-8 text-cobalt-500 mx-auto animate-spin" />
              <p className="font-display text-2xl italic mt-4 text-ink">Analyzing {docId}...</p>
              <p className="text-sm font-mono text-ink-soft uppercase tracking-widest mt-2">{jobStatus}</p>
            </div>
          )}

          {jobStatus === "done" && summary && (
            <div className="paper-card rounded-lg p-10 animate-fade-in">
               <div className="flex items-center gap-2 mb-6">
                 <CheckCircle2 className="h-4 w-4 text-emerald-700" />
                 <span className="font-mono text-[10px] uppercase tracking-widest text-emerald-700">Completed in {style} style</span>
               </div>
               <div className="prose prose-base max-w-none prose-headings:font-display prose-strong:text-cobalt-700 prose-p:text-ink">
                 <ReactMarkdown>{summary}</ReactMarkdown>
               </div>
            </div>
          )}
        </div>
      </div>
    </AppShell>
  );
}
