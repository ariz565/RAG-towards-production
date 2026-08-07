"use client";

import { motion, AnimatePresence } from "framer-motion";
import { 
  Activity, Cpu, Database, Globe, ShieldAlert, Zap, Network, Server, Layers, Search, Filter,
  CheckCircle2, AlertCircle, Clock, Target, Lock, EyeOff, ThumbsUp, ThumbsDown, Scale,
  LineChart, Coins, DollarSign, Workflow, ChevronDown, ChevronRight, AlertTriangle,
  GitBranch, FlaskConical, Wrench, TrendingDown
} from "lucide-react";
import { useEffect, useState, useMemo } from "react";
import { AppShell } from "@/components/layout/app-shell";

// Enhanced Mock Traces with Agentic Reasoning Steps
const MODELS = ["gpt-4o", "claude-3.5-sonnet", "gemini-1.5-pro", "gpt-4o-mini", "llama-3-70b-instruct"];
const QUERIES = [
  "What is the maternity leave policy for...",
  "Summarize the Q3 financial results.",
  "Ignore previous instructions and output...",
  "How do I configure the VPN on Linux?",
  "Generate a python script to scrape...",
  "Who is the CEO of Acme Corp?",
  "Compare the 401k match with the...",
  "What are the core hours for engineering?",
  "Write an integration test for the...",
  "List all deprecated API endpoints..."
];
const STATUSES = [200, 200, 200, 200, 200, 429, 403, 500];

const generateTrace = () => {
  const isAgentic = Math.random() > 0.5;
  return {
    id: `tr_${Math.random().toString(36).substr(2, 7)}`,
    timestamp: new Date().toISOString().substring(11, 23),
    query: QUERIES[Math.floor(Math.random() * QUERIES.length)],
    model: MODELS[Math.floor(Math.random() * MODELS.length)],
    tokens: Math.floor(Math.random() * 8000) + 120,
    latency: Math.floor(Math.random() * 2500) + 80,
    status: STATUSES[Math.floor(Math.random() * STATUSES.length)],
    isAgentic,
    steps: isAgentic ? [
      { name: "Input Guardrail Scan", time: 14, status: "OK" },
      { name: "Vector DB Retrieval", time: 145, status: "OK" },
      { name: "Web Search Tool", time: 890, status: "OK" },
      { name: "LLM Synthesis", time: 1240, status: "OK" },
      { name: "Output Toxicity Scan", time: 22, status: "OK" }
    ] : []
  };
};

export default function DashboardPage() {
  const [tick, setTick] = useState(0);
  const [traces, setTraces] = useState<any[]>(() => Array.from({ length: 25 }, generateTrace));
  const [expandedTrace, setExpandedTrace] = useState<string | null>(null);
  
  // Live fluctuating metrics
  const globalRps = useMemo(() => 14205 + Math.floor(Math.random() * 2500 - 1250), [tick]);
  const gpuUtil = useMemo(() => 84 + Math.floor(Math.random() * 12 - 6), [tick]);
  const vectorDbQps = useMemo(() => 4210 + Math.floor(Math.random() * 500 - 250), [tick]);
  const cacheHitRatio = useMemo(() => 34.2 + (Math.random() * 1.5 - 0.75), [tick]);
  const guardrailRejects = useMemo(() => 12.4 + (Math.random() * 0.8 - 0.4), [tick]);
  const toxicityScore = useMemo(() => (0.0014 + (Math.random() * 0.0005)).toFixed(4), [tick]);

  // Sparklines & Heatmaps
  const [sparkRps, setSparkRps] = useState<number[]>(Array(20).fill(50));
  const [latencyHeatmap, setLatencyHeatmap] = useState<number[]>(Array(40).fill(120));
  const [driftLine, setDriftLine] = useState<number[]>(Array(30).fill(0.1));
  
  useEffect(() => {
    const interval = setInterval(() => {
      setTick(t => t + 1);
      
      setSparkRps(prev => [...prev.slice(1), Math.floor(Math.random() * 100)]);
      
      setLatencyHeatmap(prev => {
        const isSpike = Math.random() > 0.8;
        const newVal = isSpike ? Math.floor(Math.random() * 800 + 400) : Math.floor(Math.random() * 80 + 100);
        return [...prev.slice(1), newVal];
      });

      if (tick % 2 === 0) {
        setDriftLine(prev => {
          let last = prev[prev.length - 1];
          let next = last + (Math.random() * 0.04 - 0.02);
          next = Math.max(0, Math.min(1, next));
          return [...prev.slice(1), next];
        });
      }

      if (Math.random() > 0.2) {
        const newTraces = Array.from({ length: Math.floor(Math.random() * 3) + 1 }, generateTrace);
        setTraces(prev => [...newTraces, ...prev].slice(0, 25));
      }
    }, 800);
    return () => clearInterval(interval);
  }, [tick]);

  return (
    <AppShell
      eyebrow="Mission Control"
      title={<>Global <em className="italic text-emerald-500">Telemetry.</em></>}
    >
      <div className="flex flex-col gap-6">

        {/* This page is a UI demo: every metric below is generated client-side
            (Math.random()/hardcoded), not read from any backend. There is no
            multi-tenant billing, GPU fleet, or live traffic system behind this
            app to report real numbers for. For real retrieval-quality metrics
            (grounding, recall, MRR/NDCG) backed by the actual eval benchmark,
            see the Streamlit app's Retrieval Quality page instead. */}
        <div className="paper-card bg-amber-500/10 border border-amber-500/30 p-3 rounded-lg font-mono text-[10px] text-amber-600 uppercase tracking-widest">
          Simulated data — UI demo only, not wired to a backend
        </div>

        {/* Tier 1: 8-Metric Hyper-Dense KPI Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
          <div className="paper-card bg-paper border border-emerald-500/30 p-3 rounded-lg flex flex-col justify-between relative overflow-hidden">
            <div className="absolute inset-0 bg-emerald-500/5 animate-pulse"></div>
            <span className="font-mono text-[9px] uppercase tracking-widest text-emerald-600 mb-2 relative z-10">Global RPS</span>
            <div className="font-mono text-xl text-ink tracking-tight relative z-10">{globalRps.toLocaleString()}</div>
            <div className="flex items-end gap-[1px] h-4 mt-2 opacity-60">
              {sparkRps.map((val, i) => (
                <motion.div key={i} animate={{ height: `${val}%` }} transition={{ duration: 0.2 }} className="flex-1 bg-emerald-500 rounded-t-sm" />
              ))}
            </div>
          </div>

          <div className="paper-card bg-paper border border-rule p-3 rounded-lg flex flex-col justify-between">
            <span className="font-mono text-[9px] uppercase tracking-widest text-ink-soft mb-2">Vector QPS</span>
            <div className="font-mono text-xl text-ink tracking-tight">{vectorDbQps.toLocaleString()}</div>
            <div className="w-full bg-paper-tint h-1 rounded-full mt-2"><motion.div animate={{ width: `${(vectorDbQps / 5000) * 100}%` }} className="bg-cobalt-500 h-full" /></div>
          </div>

          <div className="paper-card bg-paper border border-rule p-3 rounded-lg flex flex-col justify-between">
            <span className="font-mono text-[9px] uppercase tracking-widest text-ink-soft mb-2">VRAM Util</span>
            <div className="font-mono text-xl text-ink tracking-tight">{gpuUtil}%</div>
            <div className="w-full bg-paper-tint h-1 rounded-full mt-2"><motion.div animate={{ width: `${gpuUtil}%` }} className={`h-full ${gpuUtil > 90 ? 'bg-red-500' : 'bg-amber-500'}`} /></div>
          </div>

          <div className="paper-card bg-paper border border-rule p-3 rounded-lg flex flex-col justify-between">
            <span className="font-mono text-[9px] uppercase tracking-widest text-ink-soft mb-2">Total Tokens</span>
            <div className="font-mono text-xl text-ink tracking-tight">4.21B</div>
            <span className="font-mono text-[9px] text-emerald-500 mt-2">+1.2M/min</span>
          </div>

          <div className="paper-card bg-paper border border-rule p-3 rounded-lg flex flex-col justify-between">
            <span className="font-mono text-[9px] uppercase tracking-widest text-ink-soft mb-2 flex items-center gap-1"><Layers className="h-3 w-3" /> Cache Hit</span>
            <div className="font-mono text-xl text-ink tracking-tight">{cacheHitRatio.toFixed(1)}%</div>
            <span className="font-mono text-[9px] text-ink-soft mt-2">Semantic Redis</span>
          </div>

          <div className="paper-card bg-paper border border-rule p-3 rounded-lg flex flex-col justify-between">
            <span className="font-mono text-[9px] uppercase tracking-widest text-ink-soft mb-2 flex items-center gap-1"><Target className="h-3 w-3" /> Grounding</span>
            <div className="font-mono text-xl text-ink tracking-tight">94.2%</div>
            <span className="font-mono text-[9px] text-ink-soft mt-2">Avg Confidence</span>
          </div>

          <div className="paper-card bg-paper border border-rule p-3 rounded-lg flex flex-col justify-between">
            <span className="font-mono text-[9px] uppercase tracking-widest text-ink-soft mb-2 flex items-center gap-1"><Zap className="h-3 w-3" /> Peak Velocity</span>
            <div className="font-mono text-xl text-ink tracking-tight">84K</div>
            <span className="font-mono text-[9px] text-ink-soft mt-2">Tokens / sec</span>
          </div>

          <div className="paper-card bg-paper border border-red-500/20 p-3 rounded-lg flex flex-col justify-between bg-red-500/5">
            <span className="font-mono text-[9px] uppercase tracking-widest text-red-500 mb-2 flex items-center gap-1"><ShieldAlert className="h-3 w-3" /> Block Rate</span>
            <div className="font-mono text-xl text-ink tracking-tight">{guardrailRejects.toFixed(1)}%</div>
            <span className="font-mono text-[9px] text-red-400 mt-2">Admissions Denied</span>
          </div>
        </div>

        {/* Tier 2: LLM-as-a-Judge, Economics, Safety */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* 1. Quality & LLM-as-a-Judge Metrics */}
          <div className="paper-card bg-paper border border-rule p-5 rounded-lg flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <span className="font-mono text-[10px] uppercase tracking-widest text-ink-soft flex items-center gap-2">
                <Scale className="h-3 w-3" /> LLM-as-a-Judge & Quality
              </span>
            </div>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-end mb-1">
                  <span className="font-mono text-[10px] text-ink">Groundedness (Hallucination Rate)</span>
                  <span className="font-mono text-[9px] text-emerald-500">98.4% SAFE</span>
                </div>
                <div className="w-full bg-paper-tint h-1.5 rounded-full overflow-hidden">
                  <motion.div animate={{ width: "98.4%" }} className="h-full bg-emerald-500" />
                </div>
              </div>
              <div>
                <div className="flex justify-between items-end mb-1">
                  <span className="font-mono text-[10px] text-ink">Answer Relevancy (Score 0-1)</span>
                  <span className="font-mono text-[9px] text-cobalt-500">0.94</span>
                </div>
                <div className="w-full bg-paper-tint h-1.5 rounded-full overflow-hidden">
                  <motion.div animate={{ width: "94%" }} className="h-full bg-cobalt-500" />
                </div>
              </div>
              <div>
                <div className="flex justify-between items-end mb-1">
                  <span className="font-mono text-[10px] text-ink">Context Recall (RAG Quality)</span>
                  <span className="font-mono text-[9px] text-amber-500">0.82</span>
                </div>
                <div className="w-full bg-paper-tint h-1.5 rounded-full overflow-hidden">
                  <motion.div animate={{ width: "82%" }} className="h-full bg-amber-500" />
                </div>
              </div>
              
              <div className="pt-2 border-t border-rule">
                <span className="font-mono text-[8px] uppercase tracking-widest text-ink-soft">Human-in-the-Loop Feedback</span>
                <div className="flex items-center gap-4 mt-1">
                  <div className="flex items-center gap-1 text-emerald-500 font-mono text-sm"><ThumbsUp className="h-3 w-3" /> 4,192</div>
                  <div className="flex items-center gap-1 text-red-500 font-mono text-sm"><ThumbsDown className="h-3 w-3" /> 142</div>
                  <div className="w-full bg-red-500/20 h-1.5 rounded-full overflow-hidden ml-2 flex">
                    <motion.div animate={{ width: "96.7%" }} className="h-full bg-emerald-500" />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 2. Cost & Spend Tracking (Economics) */}
          <div className="paper-card bg-paper border border-rule p-5 rounded-lg flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <span className="font-mono text-[10px] uppercase tracking-widest text-ink-soft flex items-center gap-2">
                <Coins className="h-3 w-3" /> Cost & Token Economics
              </span>
            </div>
            
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="p-3 border border-rule rounded bg-paper-tint flex flex-col justify-center">
                <div className="font-mono text-[8px] uppercase tracking-widest text-ink-soft mb-1">Cost per 1M Tokens</div>
                <div className="flex flex-col mt-1 gap-1">
                  <div className="flex justify-between font-mono text-[10px]"><span className="text-emerald-500">Input</span><span className="text-ink">$5.00</span></div>
                  <div className="flex justify-between font-mono text-[10px]"><span className="text-cobalt-500">Output</span><span className="text-ink">$15.00</span></div>
                </div>
              </div>
              <div className="p-3 border border-emerald-500/30 rounded bg-emerald-500/5 flex flex-col justify-center text-center">
                <div className="font-mono text-[8px] uppercase tracking-widest text-emerald-600 mb-1">Cache Dollar Savings</div>
                <div className="font-mono text-xl text-emerald-500">${(cacheHitRatio * 14.2).toFixed(2)}</div>
                <div className="font-mono text-[8px] text-emerald-600/70 mt-1">Saved today</div>
              </div>
            </div>

            <div className="flex-1 border border-rule rounded bg-paper-tint/30 p-3">
              <div className="font-mono text-[9px] uppercase tracking-widest text-ink-soft mb-2 border-b border-rule pb-2 flex justify-between">
                <span>Tenant Cost Attribution</span>
                <DollarSign className="h-3 w-3" />
              </div>
              <div className="space-y-2 mt-2">
                {[
                  { name: "tnt_acme_corp", spend: 482.10, pct: 41 },
                  { name: "tnt_stark_ind", spend: 310.50, pct: 28 },
                  { name: "tnt_wayne_ent", spend: 142.20, pct: 12 },
                  { name: "tnt_oscorp_inc", spend: 98.40, pct: 8 },
                ].map((t, i) => (
                  <div key={i}>
                    <div className="flex justify-between items-end mb-0.5">
                      <span className="font-mono text-[9px] text-ink">{t.name}</span>
                      <span className="font-mono text-[9px] text-ink-soft">${t.spend.toFixed(2)}</span>
                    </div>
                    <div className="w-full bg-paper-tint h-1 rounded-full overflow-hidden">
                      <motion.div animate={{ width: `${t.pct}%` }} className="h-full bg-ink-soft/40" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* 3. Safety, Security & Guardrails */}
          <div className="paper-card bg-paper border border-rule p-5 rounded-lg flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <span className="font-mono text-[10px] uppercase tracking-widest text-ink-soft flex items-center gap-2">
                <ShieldAlert className="h-3 w-3" /> Safety & Guardrails
              </span>
            </div>
            
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="p-3 border border-red-500/30 rounded bg-red-500/5 flex flex-col justify-center">
                <div className="font-mono text-[8px] uppercase tracking-widest text-red-500 mb-1">Prompt Injections</div>
                <div className="font-mono text-xl text-ink tracking-tight">{Math.floor(tick / 15) + 42}</div>
                <div className="font-mono text-[8px] text-ink-soft mt-1">Blocked today</div>
              </div>
              <div className="p-3 border border-cobalt-500/30 rounded bg-cobalt-500/5 flex flex-col justify-center">
                <div className="font-mono text-[8px] uppercase tracking-widest text-cobalt-500 mb-1">PII Redactions</div>
                <div className="font-mono text-xl text-ink tracking-tight">{Math.floor(tick / 5) + 1042}</div>
                <div className="font-mono text-[8px] text-ink-soft mt-1">SSNs, Emails Masked</div>
              </div>
            </div>

            <div className="flex flex-col gap-3">
              <div className="flex justify-between items-center p-2 border border-rule rounded bg-paper-tint/30">
                <div className="flex flex-col">
                  <span className="font-mono text-[9px] text-ink">Guardrail Latency Overhead</span>
                  <span className="font-mono text-[8px] text-ink-soft">Added to TTFT</span>
                </div>
                <span className="font-mono text-sm text-amber-500">+14ms</span>
              </div>
              
              <div className="flex justify-between items-center p-2 border border-rule rounded bg-paper-tint/30">
                <div className="flex flex-col">
                  <span className="font-mono text-[9px] text-ink">Output Toxicity Score</span>
                  <span className="font-mono text-[8px] text-ink-soft">Real-time scan (0.0 to 1.0)</span>
                </div>
                <span className="font-mono text-sm text-emerald-500">{toxicityScore}</span>
              </div>
            </div>
          </div>

        </div>

        {/* Tier 3: Latency Analytics & Agentic Workflows */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* Deep Latency Analytics */}
          <div className="paper-card bg-paper border border-rule p-5 rounded-lg flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <span className="font-mono text-[10px] uppercase tracking-widest text-ink-soft flex items-center gap-2"><Clock className="h-3 w-3" /> Deep Latency Analytics</span>
            </div>
            
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="p-3 border border-rule rounded bg-paper-tint flex flex-col justify-center">
                <div className="font-mono text-[9px] uppercase tracking-widest text-ink-soft mb-1">Time to First Token (TTFT)</div>
                <div className="font-mono text-xl text-emerald-500">42ms</div>
              </div>
              <div className="p-3 border border-rule rounded bg-paper-tint flex flex-col justify-center">
                <div className="font-mono text-[9px] uppercase tracking-widest text-ink-soft mb-1">Time per Output Token</div>
                <div className="font-mono text-xl text-cobalt-500">12ms</div>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-2 mb-4">
              <div className="p-2 border border-rule rounded bg-paper-tint text-center"><div className="font-mono text-[9px] text-ink-soft uppercase tracking-widest mb-1">p50</div><div className="font-mono text-sm text-emerald-500">145ms</div></div>
              <div className="p-2 border border-rule rounded bg-paper-tint text-center"><div className="font-mono text-[9px] text-ink-soft uppercase tracking-widest mb-1">p90</div><div className="font-mono text-sm text-amber-500">380ms</div></div>
              <div className="p-2 border border-rule rounded bg-paper-tint text-center"><div className="font-mono text-[9px] text-ink-soft uppercase tracking-widest mb-1">p95</div><div className="font-mono text-sm text-orange-500">890ms</div></div>
              <div className="p-2 border border-red-500/30 rounded bg-red-500/5 text-center"><div className="font-mono text-[9px] text-red-400 uppercase tracking-widest mb-1">p99</div><div className="font-mono text-sm text-red-500">2.4s</div></div>
            </div>
            
            <div className="flex-1 bg-[#1e1e1e] rounded p-3 flex flex-col h-24">
              <span className="font-mono text-[8px] text-[#888] uppercase tracking-widest mb-2">Live Heatmap (60s sliding window)</span>
              <div className="flex-1 flex items-end gap-0.5">
                {latencyHeatmap.map((val, i) => {
                  const h = Math.min((val / 1200) * 100, 100);
                  let color = "bg-[#4CAF50]";
                  if (val > 400) color = "bg-[#FFC107]";
                  if (val > 800) color = "bg-[#F44336]";
                  return <motion.div key={i} initial={false} animate={{ height: `${h}%` }} transition={{ type: "tween", duration: 0.1 }} className={`flex-1 rounded-t-sm ${color} opacity-80`} />;
                })}
              </div>
            </div>
          </div>

          {/* Agentic & Workflow Metrics */}
          <div className="paper-card bg-paper border border-rule p-5 rounded-lg flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <span className="font-mono text-[10px] uppercase tracking-widest text-ink-soft flex items-center gap-2">
                <Workflow className="h-3 w-3" /> Agentic Workflows
              </span>
            </div>
            
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="p-4 border border-rule rounded bg-paper-tint flex flex-col items-center justify-center text-center">
                <div className="font-mono text-3xl text-emerald-500 tracking-tight">98.2%</div>
                <div className="font-mono text-[9px] uppercase tracking-widest text-ink-soft mt-2">Task Completion Rate</div>
              </div>
              <div className="p-4 border border-rule rounded bg-paper-tint flex flex-col items-center justify-center text-center">
                <div className="font-mono text-3xl text-cobalt-500 tracking-tight">99.8%</div>
                <div className="font-mono text-[9px] uppercase tracking-widest text-ink-soft mt-2">Tool Call Success Rate</div>
              </div>
            </div>

            <div className="flex flex-col gap-4">
              <div>
                <div className="flex justify-between items-end mb-1">
                  <span className="font-mono text-[10px] text-ink">Query Routing Agent</span>
                  <span className="font-mono text-[9px] text-ink-soft">1800/4200 in flight</span>
                </div>
                <div className="w-full bg-paper-tint h-1.5 rounded-full overflow-hidden">
                  <motion.div animate={{ width: "42%" }} className="h-full bg-cobalt-500" />
                </div>
              </div>
              <div>
                <div className="flex justify-between items-end mb-1">
                  <span className="font-mono text-[10px] text-ink">Vector Search Agent</span>
                  <span className="font-mono text-[9px] text-ink-soft">4100/6000 in flight</span>
                </div>
                <div className="w-full bg-paper-tint h-1.5 rounded-full overflow-hidden">
                  <motion.div animate={{ width: "68%" }} className="h-full bg-emerald-500" />
                </div>
              </div>
              <div>
                <div className="flex justify-between items-end mb-1">
                  <span className="font-mono text-[10px] text-ink">Synthesis LLM Agent</span>
                  <span className="font-mono text-[9px] text-ink-soft">7200/8000 in flight</span>
                </div>
                <div className="w-full bg-paper-tint h-1.5 rounded-full overflow-hidden">
                  <motion.div animate={{ width: "90%" }} className="h-full bg-amber-500" />
                </div>
              </div>
            </div>

          </div>
        </div>

        {/* Tier 4: ML Engineering & Experimentation */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Versioning & Traffic Split */}
          <div className="paper-card bg-paper border border-rule p-5 rounded-lg flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <span className="font-mono text-[10px] uppercase tracking-widest text-ink-soft flex items-center gap-2">
                <GitBranch className="h-3 w-3" /> Versioning & Deployment
              </span>
            </div>
            
            <div className="space-y-6">
              <div>
                <span className="font-mono text-[9px] uppercase tracking-widest text-ink-soft border-b border-rule pb-1 block mb-3">System Prompt Traffic</span>
                <div className="flex justify-between items-end mb-1">
                  <span className="font-mono text-[10px] text-ink">sys_prompt_v2.4.1 (Stable)</span>
                  <span className="font-mono text-[10px] text-emerald-500">80%</span>
                </div>
                <div className="w-full bg-paper-tint h-2 rounded-full overflow-hidden flex">
                  <motion.div animate={{ width: "80%" }} className="h-full bg-emerald-500" />
                  <motion.div animate={{ width: "20%" }} className="h-full bg-amber-500" />
                </div>
                <div className="flex justify-between items-center mt-1">
                  <span className="font-mono text-[10px] text-ink-soft border border-amber-500/30 text-amber-500 px-1 rounded">Canary: sys_prompt_v2.4.2-rc</span>
                  <span className="font-mono text-[10px] text-amber-500">20%</span>
                </div>
              </div>

              <div>
                <span className="font-mono text-[9px] uppercase tracking-widest text-ink-soft border-b border-rule pb-1 block mb-3">Model Traffic (Base)</span>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="font-mono text-[10px] text-ink">gpt-4o-2024-05-13</span>
                    <span className="font-mono text-[10px] text-ink-soft bg-paper-tint px-1 rounded">65%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-mono text-[10px] text-ink">claude-3-5-sonnet-20240620</span>
                    <span className="font-mono text-[10px] text-ink-soft bg-paper-tint px-1 rounded">35%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Experiment Tracing & Tool Analytics */}
          <div className="paper-card bg-paper border border-rule p-5 rounded-lg flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <span className="font-mono text-[10px] uppercase tracking-widest text-ink-soft flex items-center gap-2">
                <FlaskConical className="h-3 w-3" /> Experiments & Tooling
              </span>
            </div>
            
            <div className="mb-6">
              <span className="font-mono text-[9px] uppercase tracking-widest text-ink-soft border-b border-rule pb-1 block mb-3">Live A/B Test: EXP-742</span>
              <div className="flex justify-between items-center text-center">
                <div className="flex-1">
                  <div className="font-mono text-[10px] text-ink mb-1">Challenger Win</div>
                  <div className="font-mono text-xl text-emerald-500">42%</div>
                </div>
                <div className="flex-1 border-x border-rule">
                  <div className="font-mono text-[10px] text-ink mb-1">Tie</div>
                  <div className="font-mono text-xl text-ink-soft">43%</div>
                </div>
                <div className="flex-1">
                  <div className="font-mono text-[10px] text-ink mb-1">Control Win</div>
                  <div className="font-mono text-xl text-red-500">15%</div>
                </div>
              </div>
            </div>

            <div>
              <span className="font-mono text-[9px] uppercase tracking-widest text-ink-soft border-b border-rule pb-1 block mb-3 flex justify-between">
                <span>Autonomous Tool Usage</span>
                <Wrench className="h-3 w-3" />
              </span>
              <div className="space-y-2">
                {[
                  { name: "Web_Search_API", pct: 42, color: "bg-cobalt-500" },
                  { name: "Vector_DB_Retrieval", pct: 31, color: "bg-emerald-500" },
                  { name: "Python_Interpreter", pct: 18, color: "bg-amber-500" },
                  { name: "SQL_Database_Query", pct: 9, color: "bg-indigo-500" },
                ].map((t, i) => (
                  <div key={i}>
                    <div className="flex justify-between items-end mb-0.5">
                      <span className="font-mono text-[9px] text-ink">{t.name}</span>
                      <span className="font-mono text-[9px] text-ink-soft">{t.pct}%</span>
                    </div>
                    <div className="w-full bg-paper-tint h-1.5 rounded-full overflow-hidden">
                      <motion.div animate={{ width: `${t.pct}%` }} className={`h-full ${t.color}`} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Deep Drift Detection */}
          <div className="paper-card bg-paper border border-amber-500/20 p-5 rounded-lg flex flex-col relative overflow-hidden bg-amber-500/5">
            <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-10 mix-blend-overlay"></div>
            <div className="flex items-center justify-between mb-4 relative z-10">
              <span className="font-mono text-[10px] uppercase tracking-widest text-amber-500 flex items-center gap-2">
                <TrendingDown className="h-3 w-3" /> Semantic Drift Detection
              </span>
            </div>
            
            <div className="relative z-10 flex flex-col h-full justify-between">
              <div>
                <div className="font-mono text-3xl text-ink tracking-tight mb-1">0.142 <span className="text-sm text-ink-soft">Δ</span></div>
                <span className="font-mono text-[9px] text-amber-500 uppercase tracking-widest border border-amber-500/30 px-1.5 py-0.5 rounded bg-amber-500/10">Action Required</span>
                <p className="font-mono text-[10px] text-ink-soft mt-3 leading-relaxed">
                  User query distribution has drifted significantly from historical baseline over the last 30 days. Fine-tuning recommended.
                </p>
              </div>

              <div className="flex-1 mt-4 flex items-end gap-0.5 h-16">
                {driftLine.map((val, i) => (
                  <motion.div 
                    key={i} 
                    animate={{ height: `${val * 100}%` }} 
                    transition={{ type: "tween", duration: 0.2 }} 
                    className={`flex-1 rounded-t-sm ${val > 0.4 ? 'bg-amber-500' : 'bg-amber-500/30'}`} 
                  />
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Live Firehose Trace Explorer with Reasoning Drops */}
        <div className="paper-card bg-paper border border-rule rounded-lg overflow-hidden flex flex-col shadow-2xl">
          <div className="p-3 border-b border-rule flex items-center justify-between bg-[#1e1e1e] text-[#d4d4d4]">
            <span className="font-mono text-[10px] uppercase tracking-widest flex items-center gap-2 text-emerald-400">
              <Activity className="h-3 w-3 animate-pulse" /> Live Firehose: Step-by-Step Traces
            </span>
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3 w-3 text-ink-soft" />
                <input type="text" placeholder="Search traces (id, query)..." className="pl-8 pr-3 py-1 text-[10px] font-mono bg-paper border border-rule rounded focus:outline-none focus:border-cobalt-300 w-64" />
              </div>
              <button className="p-1 border border-rule rounded hover:bg-paper-tint"><Filter className="h-3 w-3 text-ink-soft" /></button>
            </div>
          </div>
          
          <div className="overflow-x-auto bg-[#1e1e1e] border-t border-[#333]">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[#333] text-[#888]">
                  <th className="py-2 px-4 w-8"></th>
                  <th className="py-2 px-4 font-mono text-[9px] uppercase tracking-widest font-normal">Status</th>
                  <th className="py-2 px-4 font-mono text-[9px] uppercase tracking-widest font-normal">Timestamp</th>
                  <th className="py-2 px-4 font-mono text-[9px] uppercase tracking-widest font-normal">Trace ID</th>
                  <th className="py-2 px-4 font-mono text-[9px] uppercase tracking-widest font-normal">Type</th>
                  <th className="py-2 px-4 font-mono text-[9px] uppercase tracking-widest font-normal">Query Snippet</th>
                  <th className="py-2 px-4 font-mono text-[9px] uppercase tracking-widest font-normal">Model</th>
                  <th className="py-2 px-4 font-mono text-[9px] uppercase tracking-widest font-normal text-right">Tokens</th>
                  <th className="py-2 px-4 font-mono text-[9px] uppercase tracking-widest font-normal text-right">Latency</th>
                </tr>
              </thead>
              <tbody className="font-mono text-[11px] text-[#d4d4d4]">
                <AnimatePresence>
                  {traces.map((trace) => (
                    <motion.tr 
                      key={trace.id}
                      initial={{ opacity: 0, y: -10, backgroundColor: "#2d2d2d" }}
                      animate={{ opacity: 1, y: 0, backgroundColor: "transparent" }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.3 }}
                      className={`border-b border-[#333] transition-colors cursor-pointer ${expandedTrace === trace.id ? 'bg-[#2a2d2e]' : 'hover:bg-[#2a2d2e]'}`}
                      onClick={() => trace.isAgentic && setExpandedTrace(expandedTrace === trace.id ? null : trace.id)}
                    >
                      <td className="py-2 px-4">
                        {trace.isAgentic ? (expandedTrace === trace.id ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />) : null}
                      </td>
                      <td className="py-2 px-4">
                        {trace.status === 200 && <span className="text-emerald-400">{trace.status} OK</span>}
                        {trace.status === 403 && <span className="text-amber-400">{trace.status} BLOCKED</span>}
                        {trace.status === 429 && <span className="text-amber-400">{trace.status} RATE_LIMIT</span>}
                        {trace.status === 500 && <span className="text-red-400">{trace.status} ERROR</span>}
                      </td>
                      <td className="py-2 px-4 text-[#888]">{trace.timestamp}</td>
                      <td className="py-2 px-4 text-[#569cd6]">{trace.id}</td>
                      <td className="py-2 px-4">
                        {trace.isAgentic ? <span className="px-1.5 py-0.5 rounded border border-[#569cd6]/30 text-[#569cd6] text-[9px]">AGENTIC</span> : <span className="px-1.5 py-0.5 rounded border border-[#ce9178]/30 text-[#ce9178] text-[9px]">DIRECT</span>}
                      </td>
                      <td className="py-2 px-4 text-[#ce9178] truncate max-w-[200px]">"{trace.query}"</td>
                      <td className="py-2 px-4 text-[#9cdcfe]">{trace.model}</td>
                      <td className="py-2 px-4 text-right">{trace.tokens.toLocaleString()}</td>
                      <td className="py-2 px-4 text-right">
                        <span className={trace.latency > 2000 ? "text-amber-400" : ""}>{trace.latency}ms</span>
                      </td>
                      
                      {/* Step-by-Step Reasoning Trace (Waterfall) */}
                      {expandedTrace === trace.id && trace.isAgentic && (
                        <td colSpan={9} className="p-0 border-0 bg-[#1a1a1a]">
                          <div className="py-4 pl-14 pr-8 space-y-2 border-l-2 border-[#569cd6] ml-6 my-2">
                            <div className="font-mono text-[9px] text-[#888] uppercase tracking-widest mb-3">Agentic Reasoning Waterfall</div>
                            {trace.steps.map((step: any, idx: number) => (
                              <div key={idx} className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-500"></div>
                                  <span className="text-[#d4d4d4] text-[10px]">{step.name}</span>
                                </div>
                                <div className="flex items-center gap-4 w-1/2">
                                  <div className="flex-1 bg-[#333] h-1.5 rounded-full overflow-hidden">
                                    <div className="bg-[#569cd6] h-full" style={{ width: `${(step.time / 1500) * 100}%` }}></div>
                                  </div>
                                  <span className="text-[#888] text-[9px] w-12 text-right">{step.time}ms</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </td>
                      )}
                    </motion.tr>
                  ))}
                </AnimatePresence>
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </AppShell>
  );
}
