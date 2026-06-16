# Agents Lab

The agentic half of the role — a tool-calling **ReAct** loop, **memory**, and
**multi-agent handoff** — as a standalone, offline-runnable reference. The "LLM"
is injected; a dependency-free `RuleBasedLLM` drives the loop deterministically so
everything runs with zero deps. The corpus-search tool **reuses the retrieval
lab** (agentic RAG).

## Concepts

| Piece | What / why | Code |
|---|---|---|
| **Tool (+ metadata/flags)** | `fn(str)->str` + JSON-schema params + flags (`dangerous`/`read_only`/`idempotent`) + `timeout`/`rate_limit` | [base.py](base.py), [tools.py](tools.py) |
| **Secure runner** | Guarded execution: validate → allow/deny → rate-limit → **HITL approval** → timeout → output cap + injection-scan, with **structured per-call logging** | [runtime.py](runtime.py), [security.py](security.py) |
| **ReAct loop** | Thought → Action → Observation, repeat until Final Answer | [react.py](react.py) |
| **Step budget** | Hard cap so the agent can't loop forever (`stopped_reason="budget"`) | [react.py](react.py) |
| **Memory** | Buffer (last N turns) or rolling Summary (bounded context) | [memory.py](memory.py) |
| **Multi-agent** | Supervisor routes a task to the right specialist (handoff) | [multiagent.py](multiagent.py) |
| **MCP** | Spec-aligned Model Context Protocol server + client (JSON-RPC 2.0) | [mcp.py](mcp.py) |

## Secure tool calling ([runtime.py](runtime.py) + [security.py](security.py))

Every tool call flows through one guarded path that returns `{ok, output, error, meta}`
and emits a structured log line (an OTel-style span):

```
exist? → allow/deny → validate args → rate-limit → approval(HITL) → execute(timeout) → cap + injection-scan output
```

- **Metadata/flags** on each tool: JSON-schema `parameters`, `dangerous`,
  `read_only`, `idempotent`, `timeout_s`, `rate_limit_per_min`, `version`.
- **HITL approval**: `dangerous`/`requires_approval` tools are **blocked** unless a
  human approves (`approval_fn`) — routine read-only tools auto-run.
- **Input validation**: size + dangerous-pattern denylist + schema (type/enum/length).
- **Output validation**: cap size + **scan for indirect prompt-injection** (tool
  output is untrusted) — flags surface in `meta`.
- **Logging**: each call → `{call_id, tool, ok, blocked, reason, latency_ms, output_injection}`.

```bash
python -m agents.cli --risky --task "delete document d5" --show-log            # → BLOCKED (needs approval)
python -m agents.cli --risky --approve --task "delete document d5" --show-log  # → executes, logged
```

## MCP — Model Context Protocol ([mcp.py](mcp.py))

The open standard (Anthropic → Linux Foundation, 2025) so any client talks to any
tool/data server over **JSON-RPC 2.0**. This is a minimal, runnable implementation:

- **Server**: registers `tools` (actions), `resources` (read-only context); handles
  `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`;
  **validates every call's args server-side** before executing.
- **Client + in-process transport** (swap for stdio / Streamable HTTP in prod).
- **Bridge**: `server.add_agent_tool(tool)` exposes an agents-lab Tool over MCP.
- **Security baked into the docstring**: tool poisoning, the lethal trifecta,
  OAuth 2.1 + least-privilege scopes, validate-server-side.

```bash
python -m agents.mcp     # initialize → list tools → call (valid + rejected) → read resource
```

## The ReAct loop (what production agents actually do)

```
Question → [ Thought → Action(tool) → Observation ]* → Final Answer
                         ▲___________________________│  (bounded by step budget)
```

Hardening shown here: a **step budget** (no infinite loops), **tool-error capture**
(the agent sees the error and can recover), and a **full transcript** for
observability — the three things that separate a demo agent from a reliable one.

## Quick start

```bash
cd agent-backend/admission-guide

python -m agents.cli --agent react --task "What is 3 * (4 + 1)?"
python -m agents.cli --agent react --task "What GPA do I need for computer science?"
python -m agents.cli --agent supervisor --task "How many credits does the CS degree need?"
```

```python
from agents import build_react_agent, BufferMemory

agent = build_react_agent(memory=BufferMemory())   # inject llm=<real model> for production
print(agent.run("What scholarships are available?").answer)
```

## Memory types
- **BufferMemory** — exact last-N turns; simple, but context grows.
- **SummaryMemory** — rolling LLM summary; bounded context for long chats.
- **(prod) semantic memory** — embed past turns into a vector store and retrieve
  the relevant ones: literally the embeddings + retrieval labs applied to history.

## Multi-agent handoff
A `Supervisor` routes each task to a specialist (`math` with a calculator, `knowledge`
with corpus search). Routing uses an injected LLM, or an offline keyword router.
Scales past one bloated prompt; each agent stays focused and debuggable.

## How this connects to the rest
- Tools can be **anything** — the corpus-search tool *is* the retrieval lab, so the
  agent does real RAG as one action ("agentic RAG").
- Swap `RuleBasedLLM` → a real model and the loop is unchanged.
- For production reliability (durable runs, HITL interrupts, checkpoints,
  guardrails, evals), see the main app — `app/services/pipeline.py` and `evals/`.

## Interview notes
- **ReAct vs plan-and-execute vs reflection** — ReAct interleaves reasoning+acting;
  plan-and-execute plans all steps up front; reflection adds a self-critique pass.
- **Why budgets/guardrails** — agents fail by looping, over-calling tools, or
  following injected instructions in tool output; bound steps, validate tool I/O,
  treat tool/retrieved content as untrusted.
- **Memory** — buffer (short), summary (long), semantic (retrieval over history).
- **Multi-agent** — use it for *separation of concerns*, not for show; more agents = more failure surface.

## References
- ReAct (Yao et al., 2022); Reflexion; LangGraph agent patterns; OpenAI Agents SDK.
- Tool calling + JSON-schema design; HITL for dangerous tools (Composio, CallSphere).
- Agent observability / structured tool spans (OpenTelemetry, Langfuse, AgentOps).
- **MCP**: modelcontextprotocol.io spec; JSON-RPC 2.0; stdio + Streamable HTTP; donated to the Linux Foundation (Dec 2025).
- **MCP security**: tool poisoning (Invariant Labs), the lethal trifecta, OAuth 2.1 + least privilege; OWASP Agentic Top-10.

The production counterpart is this repo's `app/` (LangGraph pipeline with
checkpointing, HITL interrupts, grounding, injection/PII guardrails, and eval gating).
