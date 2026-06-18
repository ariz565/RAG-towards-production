"use client";

import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { AppShell } from "@/components/layout/app-shell";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send, ChevronDown, Copy, ThumbsUp, ThumbsDown, Settings2, Sparkles, RefreshCw, AlertCircle } from "lucide-react";
import { askQuestion, streamAskQuestion, resumeAskQuestion } from "@/lib/api";

const suggestions = [
  "How many weeks of paid maternity leave do eligible employees receive?",
  "Compare maternity and paternity entitlements in detail.",
  "What are the key dates in the Acme Corp MSA?",
  "How much is the remote work stipend, and how is it reimbursed?",
];

type SessionItem = {
  id: string;
  query: string;
  answerText: string;
  citations: any[];
  pipeline_steps: any[];
  trustBadges: any;
  rewrittenQuery?: string | null;
  conversationSummary?: string | null;
  loading: boolean;
  time: number;
};

export default function ChatPage() {
  const [query, setQuery] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [pipelineOpen, setPipelineOpen] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [answerOpen, setAnswerOpen] = useState(true);
  const [streamEnabled, setStreamEnabled] = useState(true);

  // Global Thread tracking
  const [threadId, setThreadId] = useState<string | null>(null);
  
  // Hitl state
  const [clarification, setClarification] = useState<string | null>(null);
  const [clarificationAnswer, setClarificationAnswer] = useState("");

  // History state
  const [sessionHistory, setSessionHistory] = useState<SessionItem[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  const activeItem = sessionHistory.find(s => s.id === activeSessionId);
  const loading = activeItem?.loading || false;

  // Load state from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem("vision_sessionHistory");
    const savedThread = localStorage.getItem("vision_threadId");
    const savedActive = localStorage.getItem("vision_activeSessionId");
    
    if (savedHistory) {
      try { setSessionHistory(JSON.parse(savedHistory)); } catch (e) {}
    }
    if (savedThread) setThreadId(savedThread);
    if (savedActive) setActiveSessionId(savedActive);
  }, []);

  // Save state to localStorage whenever it changes
  useEffect(() => {
    if (sessionHistory.length > 0) {
      localStorage.setItem("vision_sessionHistory", JSON.stringify(sessionHistory));
    } else {
      localStorage.removeItem("vision_sessionHistory");
    }
    
    if (threadId) localStorage.setItem("vision_threadId", threadId);
    else localStorage.removeItem("vision_threadId");
    
    if (activeSessionId) localStorage.setItem("vision_activeSessionId", activeSessionId);
    else localStorage.removeItem("vision_activeSessionId");
  }, [sessionHistory, threadId, activeSessionId]);

  const handleSetActiveSession = (id: string) => {
    setActiveSessionId(id);
    setAnswerOpen(true);
    setSourcesOpen(false);
    setPipelineOpen(false);
  };

  const handleNewTopic = () => {
    setThreadId(null);
    setSessionHistory([]);
    setActiveSessionId(null);
    setQuery("");
  };

  const handleAsk = async (isResume = false) => {
    if (!isResume && !query.trim()) return;
    if (isResume && !clarificationAnswer.trim()) return;

    const userText = isResume ? clarificationAnswer : query;
    const currentQuery = query;
    
    const newItemId = Date.now().toString();
    const newSessionItem: SessionItem = {
      id: newItemId,
      query: userText,
      answerText: "",
      citations: [],
      pipeline_steps: [],
      trustBadges: {},
      loading: true,
      time: Date.now()
    };

    setSessionHistory(prev => [newSessionItem, ...prev]);
    setActiveSessionId(newItemId);
    setAnswerOpen(true);
    if (!isResume) setQuery("");

    try {
      if (isResume) {
        const res = await resumeAskQuestion({ thread_id: threadId, answer: clarificationAnswer });
        processResponse(newItemId, res);
      } else if (streamEnabled) {
        let currentAnswer = "";
        let currentPipeline: any[] = [];
        
        await streamAskQuestion({ query: currentQuery, thread_id: threadId }, (type, data) => {
          if (type === "answer_chunk") {
            currentAnswer += (data.token || data);
            setSessionHistory(prev => prev.map(s => s.id === newItemId ? { ...s, answerText: currentAnswer } : s));
          } else if (type === "answer") {
            setSessionHistory(prev => prev.map(s => s.id === newItemId ? {
              ...s,
              answerText: data.answer,
              citations: data.citations || [],
              trustBadges: {
                confidence: data.confidence,
                grounded: data.grounded,
                refused: data.refused,
                blocked: data.blocked,
                out_of_scope: data.out_of_scope,
                model_used: data.model_used,
                strategy_used: data.strategy_used
              }
            } : s));
          } else if (type.startsWith("node_")) {
            const stepIndex = currentPipeline.findIndex(st => st.node_name === data.node_name);
            if (stepIndex >= 0) {
              currentPipeline[stepIndex] = { ...currentPipeline[stepIndex], ...data };
            } else {
              currentPipeline.push(data);
            }
            setSessionHistory(prev => prev.map(s => s.id === newItemId ? { ...s, pipeline_steps: [...currentPipeline] } : s));
          } else if (type === "pipeline_complete") {
            processResponse(newItemId, data);
          } else if (type === "error") {
            setSessionHistory(prev => prev.map(s => s.id === newItemId ? { ...s, answerText: "[Error]: Stream failed.", loading: false } : s));
          }
        });
      } else {
        const res = await askQuestion({ query: currentQuery, thread_id: threadId });
        processResponse(newItemId, res);
      }
    } catch (e: any) {
      console.error(e);
      setSessionHistory(prev => prev.map(s => s.id === newItemId ? { ...s, answerText: "[Error]: Failed to communicate with the backend.", loading: false } : s));
    } finally {
      setClarificationAnswer("");
    }
  };

  const processResponse = (itemId: string, res: any) => {
    if (res.thread_id) setThreadId(res.thread_id);

    if (res.interrupted && res.clarification) {
      setClarification(res.clarification);
      setSessionHistory(prev => prev.map(s => s.id === itemId ? { ...s, loading: false } : s));
    } else {
      setClarification(null);
      setSessionHistory(prev => prev.map(s => s.id === itemId ? {
        ...s,
        answerText: res.answer || s.answerText,
        citations: res.citations || s.citations,
        pipeline_steps: res.pipeline_steps || s.pipeline_steps,
        rewrittenQuery: res.rewritten_query,
        conversationSummary: res.conversation_summary,
        loading: false,
        trustBadges: {
          confidence: res.confidence !== undefined ? res.confidence : s.trustBadges?.confidence,
          grounded: res.grounded !== undefined ? res.grounded : s.trustBadges?.grounded,
          refused: res.refused !== undefined ? res.refused : s.trustBadges?.refused,
          blocked: res.blocked !== undefined ? res.blocked : s.trustBadges?.blocked,
          out_of_scope: res.out_of_scope !== undefined ? res.out_of_scope : s.trustBadges?.out_of_scope,
          model_used: res.model_used || s.trustBadges?.model_used,
          strategy_used: res.strategy_used || s.trustBadges?.strategy_used,
          total_duration_ms: res.total_duration_ms || s.trustBadges?.total_duration_ms
        }
      } : s));
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk(false);
    }
  };

  return (
    <AppShell
      eyebrow="Chapter III · Inquiry"
      title={<>Ask, and your library <em className="italic text-cobalt-500">answers.</em></>}
      breadcrumb={["Workspace", "Ask"]}
    >
      <div className="grid grid-cols-12 gap-8 h-[calc(100vh-140px)] -mt-4">
        
        {/* LEFT COLUMN: Input and History */}
        <div className="col-span-12 lg:col-span-5 flex flex-col gap-6 overflow-y-auto scrollbar-hide pb-10 pr-2">
          
          <div className="paper-card rounded-lg p-6 grain relative shrink-0">
            <div className="relative">
              <div className="flex items-center justify-between mb-3">
                <div className="eyebrow">Your question</div>
                {threadId && <span className="font-mono text-[9px] text-ink-soft uppercase tracking-widest bg-paper-tint px-2 py-0.5 rounded border border-rule">Thread Active</span>}
              </div>
              <Textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="What would you like to know?"
                rows={4}
                className="resize-none border-0 focus-visible:ring-0 px-0 text-xl font-display leading-snug placeholder:text-ink-soft/50 bg-transparent shadow-none"
              />
              <div className="hairline pt-4 mt-3 flex items-center justify-between">
                <button onClick={() => setAdvanced(!advanced)} className="font-mono text-[10px] uppercase tracking-widest text-ink-soft hover:text-ink inline-flex items-center gap-1.5 transition-colors">
                  <Settings2 className="h-3 w-3" /> Advanced
                </button>
                <div className="flex items-center gap-2">
                  {threadId && sessionHistory.length > 0 && (
                    <Button onClick={handleNewTopic} variant="outline" className="bg-transparent border-rule text-ink hover:bg-paper-tint rounded-md font-mono text-[10px] uppercase tracking-widest transition-colors h-8">
                      New Topic
                    </Button>
                  )}
                  <Button onClick={() => handleAsk(false)} disabled={loading || !query.trim()} className="bg-ink text-paper hover:bg-cobalt-700 rounded-md font-mono text-[10px] uppercase tracking-widest transition-colors h-8">
                    {loading ? "Thinking..." : <>{threadId ? "Follow up" : "Ask"} <Send className="h-3 w-3 ml-1" /></>}
                  </Button>
                </div>
              </div>
            </div>
          </div>

          {advanced && (
            <div className="paper-card rounded-lg p-5 space-y-4 animate-fade-in shrink-0">
              <div className="eyebrow">Parameters</div>
              {[
                { label: "Strategy", opts: ["hybrid", "pageindex", "vector_only", "bm25_only"] },
                { label: "Model", opts: ["gpt-4o-mini", "gpt-4o", "claude-3.5-sonnet", "llama-3.1-70b"] },
              ].map((f) => (
                <div key={f.label} className="grid grid-cols-[120px_1fr] items-center gap-3">
                  <span className="font-mono text-[10px] uppercase tracking-widest text-ink-soft">{f.label}</span>
                  <select className="px-3 py-2 rounded-md border border-rule bg-paper-tint text-sm focus:ring-1 focus:ring-cobalt-500">{f.opts.map(o => <option key={o}>{o}</option>)}</select>
                </div>
              ))}
              <div className="hairline pt-3 flex flex-col gap-2 text-sm">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={streamEnabled} onChange={e => setStreamEnabled(e.target.checked)} className="accent-cobalt-500" /> Stream response
                </label>
              </div>
            </div>
          )}

          {sessionHistory.length === 0 ? (
            <div className="shrink-0">
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
          ) : (
            <div className="shrink-0">
              <div className="eyebrow mb-3">In this session</div>
              <div className="divide-y divide-rule border-y border-rule">
                {sessionHistory.map((s, i) => {
                  const isActive = s.id === activeSessionId;
                  const minutesAgo = Math.floor((Date.now() - s.time) / 60000);
                  const agoLabel = minutesAgo < 1 ? "just now" : `${minutesAgo}m ago`;
                  return (
                    <button 
                      key={s.id} 
                      onClick={() => handleSetActiveSession(s.id)}
                      className={`w-full text-left py-3 flex items-start gap-3 px-2 -mx-2 rounded transition-colors ${isActive ? 'bg-paper-tint border border-rule shadow-sm relative z-10' : 'hover:bg-paper-tint/40'}`}
                    >
                      <span className="font-mono text-[10px] text-ink-soft tracking-widest pt-1">{String(sessionHistory.length - i).padStart(2, "0")}</span>
                      <div className="flex-1 min-w-0">
                        <p className={`font-display text-base leading-snug truncate ${isActive ? 'text-ink' : 'text-ink/70'}`}>{s.query}</p>
                        <p className="font-mono text-[10px] uppercase tracking-widest text-ink-soft mt-1">
                          {agoLabel} {s.trustBadges?.confidence !== undefined ? `· ${Math.round(s.trustBadges.confidence * 100)}%` : ""}
                        </p>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* RIGHT COLUMN: Active Answer */}
        <div className="col-span-12 lg:col-span-7 flex flex-col gap-6 overflow-y-auto scrollbar-hide pb-10 pl-2">
          {!activeItem ? (
            <div className="paper-card rounded-lg p-20 text-center border-dashed h-full flex flex-col items-center justify-center">
              <Sparkles className="h-8 w-8 text-cobalt-300 mx-auto" />
              <p className="font-display text-2xl italic mt-4 text-ink">An answer will appear here.</p>
              <p className="text-ink-soft mt-2 text-sm max-w-sm">Use the left panel to search the document library. Your current session history will be saved dynamically.</p>
            </div>
          ) : (
            <>
              {clarification && activeItem.loading && (
                <div className="paper-card rounded-lg p-6 bg-amber-50/50 border-amber-200 shadow-sm shrink-0">
                   <div className="flex items-center gap-2 mb-3 text-amber-700">
                     <AlertCircle className="h-5 w-5" />
                     <span className="font-mono text-[10px] uppercase tracking-widest">Clarification Needed</span>
                   </div>
                   <h3 className="font-display text-2xl italic mb-4 text-amber-900">{clarification}</h3>
                   <div className="flex gap-3">
                     <input 
                       className="flex h-10 w-full rounded-md border border-amber-200 bg-white px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-amber-500"
                       value={clarificationAnswer} 
                       onChange={e => setClarificationAnswer(e.target.value)} 
                       placeholder="Your answer..." 
                     />
                     <Button onClick={() => handleAsk(true)} disabled={loading} className="bg-amber-600 text-white hover:bg-amber-700 shadow-sm">Resume</Button>
                   </div>
                </div>
              )}

              <article className="paper-card rounded-lg overflow-hidden shrink-0 shadow-sm bg-white overflow-x-hidden mb-2">
                <button onClick={() => setAnswerOpen(!answerOpen)} className="w-full px-10 py-8 flex items-start justify-between hover:bg-paper-tint transition-colors border-b border-rule">
                  <div className="text-left pr-8">
                    <div className="eyebrow mb-3 text-cobalt-500">Question</div>
                    <h3 className="font-display text-3xl leading-tight text-ink">
                      {activeItem.query}
                    </h3>
                  </div>
                  <ChevronDown className={`h-6 w-6 text-ink-soft shrink-0 transition-transform duration-300 mt-2 ${answerOpen ? "rotate-180" : ""}`} />
                </button>

                {answerOpen && (
                  <div className="px-10 py-8">
                    {activeItem.rewrittenQuery && (
                      <div className="mb-6 p-4 bg-cobalt-50/50 border border-cobalt-100 rounded-md">
                        <div className="flex items-center gap-2 mb-1">
                          <Sparkles className="h-4 w-4 text-cobalt-500" />
                          <span className="font-mono text-[10px] uppercase tracking-widest text-cobalt-600">Contextual Rewrite</span>
                        </div>
                        <p className="text-sm text-cobalt-900">{activeItem.rewrittenQuery}</p>
                      </div>
                    )}
                    {activeItem.conversationSummary && (
                      <div className="mb-6 p-4 bg-emerald-50/50 border border-emerald-100 rounded-md">
                        <div className="flex items-center gap-2 mb-1">
                          <RefreshCw className="h-4 w-4 text-emerald-500" />
                          <span className="font-mono text-[10px] uppercase tracking-widest text-emerald-600">Memory Compaction</span>
                        </div>
                        <p className="text-sm text-emerald-900 leading-relaxed">{activeItem.conversationSummary}</p>
                      </div>
                    )}

                    <div>
                      <div className="flex items-center justify-between mb-6">
                        <span className="eyebrow">Answer</span>
                        {(!activeItem.loading && !clarification) && (
                          <div className="flex items-center gap-3 font-mono text-[10px] uppercase tracking-widest bg-paper-tint px-3 py-1.5 rounded-full border border-rule shadow-sm">
                            {activeItem.trustBadges?.confidence !== undefined && <span className="text-emerald-700">● {Math.round(activeItem.trustBadges.confidence * 100)}% confident</span>}
                            {activeItem.trustBadges?.grounded && <span className="text-cobalt-500">● fully grounded</span>}
                            {activeItem.trustBadges?.refused && <span className="text-destructive">● refused</span>}
                            {activeItem.trustBadges?.blocked && <span className="text-destructive">● blocked</span>}
                            {activeItem.trustBadges?.out_of_scope && <span className="text-amber-600">● out of scope</span>}
                          </div>
                        )}
                      </div>

                      <div className="prose prose-base max-w-none prose-headings:font-display prose-strong:text-cobalt-700 prose-strong:font-semibold prose-p:text-ink prose-li:text-ink-soft prose-li:my-1 prose-p:leading-relaxed first-letter:font-display first-letter:text-5xl first-letter:italic first-letter:text-cobalt-500 first-letter:float-left first-letter:mr-2 first-letter:leading-none first-letter:mt-1">
                        {activeItem.loading && !activeItem.answerText ? (
                          <span className="animate-pulse flex items-center gap-2 text-cobalt-600"><RefreshCw className="h-4 w-4 animate-spin" /> Synthesizing...</span>
                        ) : (
                          <ReactMarkdown>{activeItem.answerText}</ReactMarkdown>
                        )}
                      </div>

                      <div className="hairline pt-4 mt-8 flex flex-col md:flex-row md:items-center justify-between gap-4 bg-paper-tint/30 -mx-10 -mb-8 px-10 py-5 rounded-b-lg">
                        <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
                          <div className="flex flex-col">
                            <span className="font-mono text-[9px] uppercase tracking-widest text-ink-soft">Model</span>
                            <span className="font-mono text-[11px] text-ink">{activeItem.trustBadges?.model_used || "..."}</span>
                          </div>
                          <div className="flex flex-col">
                            <span className="font-mono text-[9px] uppercase tracking-widest text-ink-soft">Strategy</span>
                            <span className="font-mono text-[11px] text-ink">{activeItem.trustBadges?.strategy_used || "..."}</span>
                          </div>
                          <div className="flex flex-col">
                            <span className="font-mono text-[9px] uppercase tracking-widest text-ink-soft">Latency</span>
                            <span className="font-mono text-[11px] text-ink">{activeItem.trustBadges?.total_duration_ms ? `${(activeItem.trustBadges.total_duration_ms / 1000).toFixed(2)}s` : "..."}</span>
                          </div>
                          <div className="flex flex-col">
                            <span className="font-mono text-[9px] uppercase tracking-widest text-ink-soft">Tokens</span>
                            <span className="font-mono text-[11px] text-ink">{activeItem.pipeline_steps?.reduce((sum, step) => sum + (step.tokens_used || 0), 0) || "..."}</span>
                          </div>
                          {activeItem.trustBadges?.confidence !== undefined && (
                            <div className="flex flex-col">
                              <span className="font-mono text-[9px] uppercase tracking-widest text-ink-soft">Confidence</span>
                              <span className="font-mono text-[11px] text-emerald-700">{Math.round(activeItem.trustBadges.confidence * 100)}%</span>
                            </div>
                          )}
                        </div>

                        <div className="flex items-center gap-1 shrink-0">
                          <Button variant="ghost" size="sm" className="font-mono text-[10px] uppercase tracking-widest text-ink-soft hover:bg-paper-tint transition-colors"><Copy className="h-3 w-3 mr-1" /> Copy</Button>
                          <Button variant="ghost" size="icon" className="hover:bg-paper-tint transition-colors"><ThumbsUp className="h-3.5 w-3.5 text-ink-soft" /></Button>
                          <Button variant="ghost" size="icon" className="hover:bg-paper-tint transition-colors"><ThumbsDown className="h-3.5 w-3.5 text-ink-soft" /></Button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </article>

              {activeItem.citations && activeItem.citations.length > 0 && (
                <section className="paper-card rounded-lg overflow-hidden shrink-0 shadow-sm bg-white overflow-x-hidden mb-2">
                  <button onClick={() => setSourcesOpen(!sourcesOpen)} className="w-full px-7 py-5 flex items-center justify-between hover:bg-paper-tint transition-colors">
                    <div className="text-left">
                      <div className="eyebrow">Sources</div>
                      <span className="font-display text-xl italic mt-1 block text-ink">Cited in the answer</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="font-mono text-[10px] tracking-widest text-ink-soft bg-paper-tint px-2 py-1 rounded">№ {activeItem.citations.length}</span>
                      <ChevronDown className={`h-4 w-4 text-ink-soft transition-transform duration-300 ${sourcesOpen ? "rotate-180" : ""}`} />
                    </div>
                  </button>
                  {sourcesOpen && (
                    <div className="divide-y divide-rule px-7 pb-7 hairline pt-5 bg-paper-tint/30">
                      {activeItem.citations.map((c: any, i: number) => (
                        <div key={c.id || i} className="py-4 grid grid-cols-[28px_1fr_auto] gap-4 items-baseline">
                          <span className="font-mono text-[10px] text-cobalt-500 tracking-widest pt-1">[{i + 1}]</span>
                          <div className="min-w-0">
                            <p className="font-display text-lg leading-snug text-ink truncate">{c.section || c.section_title || "Document Chunk"}</p>
                            <p className="text-xs text-ink-soft mt-1 leading-relaxed break-words">{c.relevance}</p>
                          </div>
                          <div className="flex gap-1 shrink-0">
                            {(c.pages || c.page_numbers || []).map((p: number) => (
                              <span key={p} className="px-2 py-1 rounded border border-rule bg-paper text-[10px] font-mono text-ink-soft shadow-sm">p.{p}</span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </section>
              )}

              {activeItem.pipeline_steps && activeItem.pipeline_steps.length > 0 && (
                <section className="paper-card rounded-lg overflow-hidden shrink-0 shadow-sm bg-white mb-8 overflow-x-hidden">
                  <button onClick={() => setPipelineOpen(!pipelineOpen)} className="w-full px-7 py-5 flex items-center justify-between hover:bg-paper-tint transition-colors">
                    <div className="text-left">
                      <div className="eyebrow">Trace</div>
                      <span className="font-display text-xl italic mt-1 block text-ink">Pipeline · {activeItem.pipeline_steps.length} steps</span>
                    </div>
                    <ChevronDown className={`h-4 w-4 text-ink-soft transition-transform duration-300 ${pipelineOpen ? "rotate-180" : ""}`} />
                  </button>
                  {pipelineOpen && (
                    <div className="px-7 pb-7 hairline pt-5 bg-paper-tint/30">
                      {activeItem.pipeline_steps.map((s: any, i: number) => (
                        <div key={i} className="mb-4 last:mb-0">
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-mono text-[10px] uppercase tracking-widest text-ink">{s.node_name}</span>
                            <div className="flex items-center gap-3 shrink-0">
                              <span className={`font-mono text-[9px] uppercase tracking-widest ${s.status === 'error' ? 'text-destructive' : 'text-cobalt-500'}`}>{s.status}</span>
                              <span className="font-mono text-[10px] text-ink-soft">{Math.round(s.duration_ms)}ms</span>
                            </div>
                          </div>
                          <p className="text-sm text-ink-soft leading-relaxed break-words whitespace-pre-wrap">{s.thinking || s.result}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </section>
              )}
            </>
          )}
        </div>
      </div>
    </AppShell>
  );
}
