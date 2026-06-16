"""Agents lab — tool-calling ReAct loop, memory, and multi-agent handoff.

Standalone study package (not wired into the app). Runs offline via the
dependency-free RuleBasedLLM; inject a real LLM for production. The corpus-search
tool reuses the retrieval lab (agentic RAG).
"""

from agents.base import AgentResult, LLM, Memory, RuleBasedLLM, Step, Tool, ToolResult
from agents.factory import build_react_agent, build_supervisor, get_agent
from agents.mcp import InProcessTransport, MCPClient, MCPServer
from agents.memory import BufferMemory, SummaryMemory
from agents.multiagent import Specialist, Supervisor, SupervisorResult
from agents.react import ReActAgent
from agents.runtime import SecureToolRunner, ToolCall
from agents.security import scan_output_for_injection, validate_arg
from agents.tools import default_toolset, make_calculator, make_corpus_search, make_delete_doc, risky_toolset

__all__ = [
    "Tool",
    "ToolResult",
    "Step",
    "AgentResult",
    "LLM",
    "RuleBasedLLM",
    "Memory",
    "ReActAgent",
    "BufferMemory",
    "SummaryMemory",
    "Supervisor",
    "Specialist",
    "SupervisorResult",
    "get_agent",
    "build_react_agent",
    "build_supervisor",
    "default_toolset",
    "risky_toolset",
    "make_calculator",
    "make_corpus_search",
    "make_delete_doc",
    "SecureToolRunner",
    "ToolCall",
    "validate_arg",
    "scan_output_for_injection",
    "MCPServer",
    "MCPClient",
    "InProcessTransport",
]
