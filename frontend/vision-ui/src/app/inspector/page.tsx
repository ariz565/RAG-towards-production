"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { AppShell } from "@/components/layout/app-shell";
import { Button } from "@/components/ui/button";
import { getDocuments, getTree, getPageText, getHealth } from "@/lib/api";
import { Database, FileText, ChevronRight, Hash } from "lucide-react";

export default function InspectorPage() {
  const searchParams = useSearchParams();
  const initialDoc = searchParams.get("doc_id") || "";
  
  const [docs, setDocs] = useState<any[]>([]);
  const [docId, setDocId] = useState(initialDoc);
  
  const [tree, setTree] = useState<any>(null);
  const [selectedPage, setSelectedPage] = useState<number | null>(null);
  const [pageText, setPageText] = useState("");
  const [loadingText, setLoadingText] = useState(false);
  
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(console.error);
    getDocuments().then(res => {
      if (res.documents) setDocs(res.documents);
    }).catch(console.error);
  }, []);

  const [treeError, setTreeError] = useState<string | null>(null);

  useEffect(() => {
    if (docId) {
      setTreeError(null);
      getTree(docId).then(setTree).catch(e => {
        // If 503 is returned, it means the tree isn't built yet
        if (e.message.includes("No PageIndex") || e.message.includes("503")) {
          setTreeError("PageIndex not built. Please re-index 'All' on the Documents page.");
        } else {
          console.error(e);
        }
        setTree(null);
      });
    } else {
      setTree(null);
      setTreeError(null);
    }
  }, [docId]);

  useEffect(() => {
    if (docId && selectedPage !== null) {
      setLoadingText(true);
      getPageText(docId, selectedPage).then(res => {
        setPageText(res.text || "");
      }).catch(e => {
        console.error(e);
        setPageText("Failed to load page text.");
      }).finally(() => setLoadingText(false));
    }
  }, [docId, selectedPage]);

  // Recursive tree renderer
  const renderNode = (node: any) => {
    if (!node) return null;
    return (
      <div className="pl-4 mt-2 border-l border-rule/50">
        <div 
          className="group cursor-pointer hover:bg-paper-tint/50 p-2 -ml-2 rounded transition-colors"
          onClick={() => setSelectedPage(node.page_start)}
        >
          <div className="font-display text-lg mb-1 text-ink group-hover:text-cobalt-600 transition-colors">{node.title || "Section"}</div>
          <div className="text-xs font-mono text-ink-soft mb-1">
            Pages {node.page_start} - {node.page_end}
          </div>
        </div>
        {node.children && node.children.length > 0 && (
          <div className="space-y-1">
            {node.children.map((c: any, i: number) => (
              <div key={i}>{renderNode(c)}</div>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <AppShell eyebrow="Chapter V" title="Inspector" breadcrumb={["Workspace", "Inspector"]}>
      <p className="max-w-2xl text-ink-soft text-lg mb-10 -mt-2">Examine the PageIndex tree structure and raw extracted text of any document.</p>
      
      {health && (
        <div className="mb-10 grid grid-cols-2 md:grid-cols-4 gap-px bg-rule border border-rule rounded-lg overflow-hidden">
          <div className="bg-card p-4">
             <div className="eyebrow">Status</div>
             <div className="font-display text-2xl text-emerald-700 italic mt-1">{health.status}</div>
          </div>
          <div className="bg-card p-4">
             <div className="eyebrow">Total Docs</div>
             <div className="font-display text-2xl text-ink italic mt-1">{health.documents}</div>
          </div>
          <div className="bg-card p-4">
             <div className="eyebrow">Vector Space</div>
             <div className="font-display text-2xl text-ink italic mt-1">OK</div>
          </div>
          <div className="bg-card p-4">
             <div className="eyebrow">Version</div>
             <div className="font-display text-2xl text-ink italic mt-1">1.4.2</div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-12 gap-8">
        <div className="col-span-12 lg:col-span-4 space-y-6">
          <div className="paper-card rounded-lg p-6">
            <h3 className="eyebrow mb-4">Select Document</h3>
            <select value={docId} onChange={e => setDocId(e.target.value)} className="w-full px-3 py-2.5 rounded-md border border-rule bg-paper-tint font-mono text-sm">
              <option value="">Select...</option>
              {docs.map(d => (
                <option key={d.doc_id || d.id} value={d.doc_id || d.id}>{d.title || d.doc_id}</option>
              ))}
            </select>
          </div>

          {treeError && (
            <div className="paper-card rounded-lg p-6 bg-destructive/5 border-destructive/20 mt-6">
              <h3 className="eyebrow text-destructive mb-2">Index Missing</h3>
              <p className="text-sm font-mono text-ink-soft">{treeError}</p>
            </div>
          )}

          {tree && !treeError && (
            <div className="paper-card rounded-lg p-6 max-h-[600px] overflow-y-auto mt-6">
              <h3 className="eyebrow mb-4">PageIndex Tree</h3>
              {tree.tree && tree.tree.length > 0 ? (
                <div className="space-y-4">
                  {tree.tree.map((node: any, idx: number) => (
                    <div key={idx}>{renderNode(node)}</div>
                  ))}
                </div>
              ) : (
                <div className="text-sm font-mono text-ink-soft">No tree structure found.</div>
              )}
            </div>
          )}
        </div>

        <div className="col-span-12 lg:col-span-8">
          <div className="paper-card rounded-lg p-6 min-h-[600px] flex flex-col">
            <div className="flex items-center justify-between border-b border-rule pb-4 mb-4">
               <h3 className="font-display text-2xl italic">Page Viewer</h3>
               <div className="flex items-center gap-2">
                 <span className="eyebrow">Page</span>
                 <input 
                   type="number" 
                   value={selectedPage || ""} 
                   onChange={e => setSelectedPage(Number(e.target.value))} 
                   placeholder="#"
                   className="w-16 px-2 py-1 rounded bg-paper-tint border border-rule font-mono text-sm"
                 />
               </div>
            </div>

            <div className="flex-1 bg-paper-tint/30 rounded border border-rule p-6 overflow-y-auto whitespace-pre-wrap font-mono text-sm leading-relaxed text-ink/80 relative">
               {!docId ? (
                 <div className="absolute inset-0 flex flex-col items-center justify-center text-ink-soft">
                   <Database className="h-8 w-8 mb-4 opacity-50" />
                   <span>Select a document and page number</span>
                 </div>
               ) : !selectedPage ? (
                 <div className="absolute inset-0 flex flex-col items-center justify-center text-ink-soft">
                   <Hash className="h-8 w-8 mb-4 opacity-50" />
                   <span>Enter a page number to view raw text</span>
                 </div>
               ) : loadingText ? (
                 <div className="absolute inset-0 flex items-center justify-center">
                   <span className="animate-pulse">Loading...</span>
                 </div>
               ) : (
                 pageText || "No text extracted for this page."
               )}
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
