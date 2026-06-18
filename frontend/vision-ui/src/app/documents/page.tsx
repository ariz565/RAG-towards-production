"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { AppShell } from "@/components/layout/app-shell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { mockDocs, type MockDoc } from "@/lib/mock-data";
import { getDocuments, uploadDocument, indexDocument } from "@/lib/api";
import { Upload, ArrowUpRight, CheckCircle2, Loader2, XCircle, MoreHorizontal, Search, RefreshCw } from "lucide-react";

function StatusDot({ s }: { s: string }) {
  const map: Record<string, { c: string; i: any; l: string }> = {
    indexed: { c: "text-emerald-700", i: CheckCircle2, l: "Indexed" },
    indexing: { c: "text-cobalt-500", i: Loader2, l: "Indexing" },
    failed: { c: "text-destructive", i: XCircle, l: "Failed" },
  };
  const { c, i: I, l } = map[s] || map.indexing;
  return (
    <span className={`inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-widest ${c}`}>
      <I className={`h-3 w-3 ${s === "indexing" ? "animate-spin" : ""}`} /> {l}
    </span>
  );
}

export default function Documents() {
  const [dragOver, setDragOver] = useState(false);
  const [pickedFile, setPickedFile] = useState<File | null>(null);
  
  const [family, setFamily] = useState("");
  const [version, setVersion] = useState("");
  const [effectiveDate, setEffectiveDate] = useState("");
  
  const [docs, setDocs] = useState<any[]>(mockDocs);
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDocs();
  }, []);

  const fetchDocs = async () => {
    try {
      setLoadingDocs(true);
      const res = await getDocuments();
      if (res.documents && res.documents.length > 0) {
        setDocs(res.documents);
      } else {
        setDocs(mockDocs); // fallback if empty
      }
    } catch (e) {
      console.error(e);
      setDocs(mockDocs); // fallback on error
    } finally {
      setLoadingDocs(false);
    }
  };

  const handleUpload = async () => {
    if (!pickedFile) return;
    try {
      setUploading(true);
      setError(null);
      const formData = new FormData();
      formData.append("file", pickedFile);
      if (family) formData.append("family", family);
      if (version) formData.append("version", version);
      if (effectiveDate) formData.append("effective_date", effectiveDate);

      await uploadDocument(formData);
      
      // reset form
      setPickedFile(null);
      setFamily("");
      setVersion("");
      setEffectiveDate("");
      
      // reload
      await fetchDocs();
    } catch (e: any) {
      console.error("Upload failed", e);
      setError(e.message || "Upload failed. Ensure the backend is running and you are signed in.");
    } finally {
      setUploading(false);
    }
  };

  const handleIndex = async (e: React.MouseEvent, docId: string) => {
    e.preventDefault(); // prevent link navigation
    try {
      setError(null);
      setDocs(prev => prev.map(d => d.doc_id === docId ? { ...d, status: "indexing" } : d));
      await indexDocument(docId, "all");
      await fetchDocs();
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Indexing failed.");
      await fetchDocs();
    }
  };

  return (
    <AppShell
      eyebrow="Chapter II · Library"
      title={<>The shelf, <em className="italic text-cobalt-500">curated.</em></>}
      breadcrumb={["Workspace", "Documents"]}
    >
      <p className="max-w-2xl text-ink-soft text-lg mb-10 -mt-2">A working library of {docs.length} PDFs across {new Set(docs.map(d=>d.family)).size} families. Drag one in to begin reading it.</p>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); setPickedFile(e.dataTransfer.files[0] ?? null); }}
        className={`relative grain rounded-lg border-2 border-dashed transition-all p-14 text-center ${dragOver ? "border-cobalt-500 bg-cobalt-50/40" : "border-rule bg-card"}`}
      >
        <div className="mx-auto h-12 w-12 rounded-full bg-ink text-paper grid place-items-center"><Upload className="h-5 w-5" /></div>
        <h3 className="font-display text-3xl italic mt-5">Lay a PDF on the table.</h3>
        <p className="text-sm text-ink-soft mt-2">
          Drop it here or <label className="text-cobalt-500 cursor-pointer ink-link">browse from your computer
            <input type="file" accept="application/pdf" className="hidden" onChange={(e) => setPickedFile(e.target.files?.[0] ?? null)} />
          </label>
        </p>
        <p className="font-mono text-[10px] uppercase tracking-widest text-ink-soft mt-3">PDF · up to 100MB · auto-indexed in ~14s</p>
      </div>

      {pickedFile && (
        <div className="paper-card rounded-lg p-6 mt-6 animate-fade-in">
          <div className="flex items-center justify-between">
            <div>
              <div className="eyebrow">New document</div>
              <p className="font-display text-2xl italic mt-1">{pickedFile.name}</p>
            </div>
            <span className="eyebrow text-cobalt-500">Ready to index</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div><Label className="eyebrow">Family</Label><Input value={family} onChange={e => setFamily(e.target.value)} placeholder="leave_policy" className="mt-2 bg-paper-tint border-rule" /></div>
            <div><Label className="eyebrow">Version</Label><Input value={version} onChange={e => setVersion(e.target.value)} placeholder="2026" className="mt-2 bg-paper-tint border-rule" /></div>
            <div><Label className="eyebrow">Effective date</Label><Input value={effectiveDate} onChange={e => setEffectiveDate(e.target.value)} type="date" className="mt-2 bg-paper-tint border-rule" /></div>
          </div>
          <div className="flex justify-end gap-2 mt-6">
            <Button variant="ghost" onClick={() => setPickedFile(null)} disabled={uploading}>Cancel</Button>
            <Button onClick={handleUpload} disabled={uploading} className="bg-ink text-paper hover:bg-cobalt-700">
              {uploading ? "Uploading..." : "Upload & index"}
            </Button>
          </div>
          {error && <div className="mt-4 p-3 bg-destructive/10 text-destructive text-xs font-mono uppercase tracking-widest rounded">{error}</div>}
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
      
      {!pickedFile && error && (
        <div className="mt-6 p-4 bg-destructive/10 text-destructive text-sm font-mono uppercase tracking-widest rounded-lg flex items-center justify-between">
          <span>{error}</span>
          <Button variant="ghost" size="sm" onClick={() => setError(null)}>Dismiss</Button>
        </div>
      )}

      <div className="divide-y divide-rule">
        {loadingDocs ? (
           <div className="py-7 text-center text-ink-soft font-mono text-sm">Loading documents...</div>
        ) : docs.map((d, i) => (
          <Link key={d.id || d.doc_id} href={`/inspector?doc_id=${d.doc_id || d.id}`} className="group grid grid-cols-12 gap-6 items-baseline py-7 hover:bg-paper-tint/40 transition-colors px-2 -mx-2 rounded">
            <span className="col-span-1 font-mono text-[10px] text-ink-soft tracking-widest pt-2">№ {String(i + 1).padStart(2, "0")}</span>
            <div className="col-span-12 md:col-span-6">
              <h3 className="font-display text-2xl leading-tight text-ink group-hover:italic transition-all">{d.title || d.doc_id}</h3>
              <p className="text-sm text-ink-soft mt-2 line-clamp-1 leading-relaxed">{d.preview || "No preview available."}</p>
              <div className="mt-3 font-mono text-[10px] uppercase tracking-widest text-ink-soft flex items-center gap-3">
                <span>{d.family || "N/A"}</span><span>·</span><span>v{d.version || "1"}</span><span>·</span><span>eff. {d.effectiveDate || "N/A"}</span>
              </div>
            </div>
            <div className="col-span-4 md:col-span-2 text-center md:text-right">
              <div className="font-display text-3xl italic text-ink leading-none">{d.pages || d.total_pages || 0}</div>
              <div className="eyebrow mt-1">pages</div>
            </div>
            <div className="col-span-4 md:col-span-1 text-center md:text-right">
              <div className="font-display text-3xl italic text-cobalt-500 leading-none">{d.chunks || d.total_chunks || 0}</div>
              <div className="eyebrow mt-1">chunks</div>
            </div>
            <div className="col-span-4 md:col-span-2 md:text-right flex md:justify-end items-center gap-3">
              <StatusDot s={d.status || (d.hybrid ? "indexed" : "indexing")} />
              <button 
                onClick={(e) => handleIndex(e, d.doc_id || d.id)} 
                className="p-1 text-ink-soft hover:text-cobalt-500 transition-colors" 
                title="Force Re-index"
              >
                <RefreshCw className={`h-3 w-3 ${d.status === "indexing" ? "animate-spin" : ""}`} />
              </button>
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
