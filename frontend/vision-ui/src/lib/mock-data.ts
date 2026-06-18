export type DocStatus = "indexed" | "indexing" | "failed";

export interface MockDoc {
  id: string;
  title: string;
  family: string;
  version: string;
  effectiveDate: string;
  pages: number;
  chunks: number;
  status: DocStatus;
  uploadedAt: string;
  lastAsked?: string;
  preview: string;
}

export const mockDocs: MockDoc[] = [
  {
    id: "doc_mat_2026",
    title: "Maternity Leave Policy 2026",
    family: "leave_policy",
    version: "2026",
    effectiveDate: "2026-01-01",
    pages: 12,
    chunks: 142,
    status: "indexed",
    uploadedAt: "2026-06-10T09:14:00Z",
    lastAsked: "2026-06-17T08:21:00Z",
    preview: "Eligible employees are entitled to 26 weeks of paid maternity leave, beginning…",
  },
  {
    id: "doc_pat_2026",
    title: "Paternity Leave Policy 2026",
    family: "leave_policy",
    version: "2026",
    effectiveDate: "2026-01-01",
    pages: 8,
    chunks: 94,
    status: "indexed",
    uploadedAt: "2026-06-09T15:42:00Z",
    lastAsked: "2026-06-17T07:02:00Z",
    preview: "Fathers and non-birthing partners receive 8 weeks of paid leave, which may be…",
  },
  {
    id: "doc_remote",
    title: "Remote Work Handbook",
    family: "workplace",
    version: "v3.2",
    effectiveDate: "2026-04-15",
    pages: 34,
    chunks: 412,
    status: "indexed",
    uploadedAt: "2026-05-22T11:00:00Z",
    lastAsked: "2026-06-16T18:30:00Z",
    preview: "Remote-first roles are eligible for a $1,200 annual home-office stipend, with…",
  },
  {
    id: "doc_msa",
    title: "Master Services Agreement — Acme Corp",
    family: "contracts",
    version: "2025-Q4",
    effectiveDate: "2025-10-01",
    pages: 48,
    chunks: 587,
    status: "indexing",
    uploadedAt: "2026-06-17T09:55:00Z",
    preview: "This Master Services Agreement is entered into between the Parties as of the…",
  },
];

export interface MockQuery {
  id: string;
  query: string;
  doc: string;
  ago: string;
  confidence: number;
  answer: string;
}

export const mockQueries: MockQuery[] = [
  {
    id: "q1",
    query: "How many weeks of paid maternity leave do eligible employees receive?",
    doc: "Maternity Leave Policy 2026",
    ago: "2h ago",
    confidence: 0.92,
    answer: "Eligible employees receive **26 weeks** of fully paid maternity leave, with an option to extend by an additional 13 weeks unpaid.",
  },
  {
    id: "q2",
    query: "Compare maternity vs paternity leave duration",
    doc: "Leave Policy 2026 (multi-doc)",
    ago: "5h ago",
    confidence: 0.81,
    answer: "Maternity leave is **26 weeks paid**; paternity / partner leave is **8 weeks paid**. Both can be extended unpaid up to a combined 12 months.",
  },
  {
    id: "q3",
    query: "What is the home office stipend amount?",
    doc: "Remote Work Handbook",
    ago: "yesterday",
    confidence: 0.68,
    answer: "Remote-first employees receive a **$1,200 annual stipend** for home office equipment, reimbursed quarterly.",
  },
];

export interface PipelineStep {
  id: string;
  name: string;
  duration: number;
  tokens?: number;
  status: "ok" | "skipped" | "error";
  thinking?: string;
}

export const mockPipeline: PipelineStep[] = [
  { id: "s1", name: "Query understanding", duration: 142, tokens: 86, status: "ok", thinking: "Detected intent: factual lookup. Rewrote query for retrieval." },
  { id: "s2", name: "Hybrid retrieval (BM25 + vector)", duration: 384, status: "ok", thinking: "Retrieved 18 candidates, reranked to top 5." },
  { id: "s3", name: "Answer generation", duration: 1308, tokens: 1240, status: "ok", thinking: "Drafted answer with 3 inline citations. Grounded check passed." },
];

export interface Citation {
  id: string;
  section: string;
  pages: number[];
  relevance: string;
}

export const mockCitations: Citation[] = [
  { id: "c1", section: "§3.1 Paid maternity entitlement", pages: [4, 5], relevance: "States the 26-week paid entitlement." },
  { id: "c2", section: "§3.4 Extensions and unpaid leave", pages: [7], relevance: "Defines optional 13-week unpaid extension." },
  { id: "c3", section: "Appendix A — Eligibility", pages: [11], relevance: "Clarifies who qualifies as 'eligible employee'." },
];
