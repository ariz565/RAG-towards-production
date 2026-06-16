"""Tool-call security primitives.

Implements the controls the 2026 guidance converges on — *layered*, since no
single control is enough:
- **Input validation** against the tool's parameter spec (type/enum/length) +
  a denylist of dangerous argument patterns.
- **Output validation / injection scan**: tool/MCP output is *untrusted* — scan
  it for indirect prompt-injection before it re-enters the agent's context.
- **Approval policy**: dangerous / non-idempotent tools require explicit human
  approval (HITL); routine read-only tools auto-proceed.

(OWASP Agentic Top-10: Tool Misuse, Goal Hijack, Supply-Chain/Tool-Poisoning.)
"""

from __future__ import annotations

import json
import re

MAX_ARG_CHARS = 4000
MAX_OUTPUT_CHARS = 8000

# Dangerous argument patterns (command/SQL/code injection, path traversal).
_DANGEROUS_ARG = [
    r"\brm\s+-rf\b", r";\s*drop\s+table", r"\b__import__\b", r"\bsubprocess\b",
    r"\beval\s*\(", r"\bos\.system\b", r"\.\./", r"\bcurl\s+.*\|\s*sh\b",
]
_DANGEROUS_ARG_RE = [re.compile(p, re.IGNORECASE) for p in _DANGEROUS_ARG]

# Indirect prompt-injection patterns that may appear inside tool/MCP output.
_INJECTION = [
    r"ignore\s+(all\s+|the\s+)?previous\s+instructions",
    r"disregard\s+(the\s+)?(above|previous)",
    r"you\s+are\s+now\b", r"reveal\s+(your\s+)?(system\s+)?(prompt|instructions)",
    r"</?\s*system\s*>", r"\bexfiltrate\b", r"send\s+.*\bto\s+https?://",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION]


def validate_arg(tool, arg) -> tuple[bool, str]:
    """Validate a tool argument: size, dangerous patterns, and schema (if any)."""
    raw = arg if isinstance(arg, str) else json.dumps(arg, default=str)
    if len(raw) > MAX_ARG_CHARS:
        return False, f"argument exceeds {MAX_ARG_CHARS} chars"
    for rx in _DANGEROUS_ARG_RE:
        if rx.search(raw):
            return False, f"argument matched dangerous pattern /{rx.pattern}/"

    # Structured-arg schema check (type/enum/required/max_length).
    params = getattr(tool, "parameters", {}) or {}
    if params and isinstance(arg, dict):
        for name, spec in params.items():
            if spec.get("required") and name not in arg:
                return False, f"missing required parameter '{name}'"
            if name in arg:
                val = arg[name]
                if "enum" in spec and val not in spec["enum"]:
                    return False, f"'{name}' must be one of {spec['enum']}"
                if spec.get("type") == "string" and not isinstance(val, str):
                    return False, f"'{name}' must be a string"
                if spec.get("max_length") and isinstance(val, str) and len(val) > spec["max_length"]:
                    return False, f"'{name}' exceeds max_length {spec['max_length']}"
    return True, ""


def scan_output_for_injection(text: str) -> list[str]:
    """Return injection patterns found in (untrusted) tool output."""
    return [rx.pattern for rx in _INJECTION_RE if rx.search(text or "")]


def cap_output(text: str) -> tuple[str, bool]:
    if text and len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + " …[truncated]", True
    return text, False


def needs_approval(tool) -> bool:
    return bool(getattr(tool, "dangerous", False) or getattr(tool, "requires_approval", False))
