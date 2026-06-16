"""Secure tool execution + structured per-call logging.

Every tool call goes through one guarded path that enforces the layered controls
and emits a structured log line (an OTel-style span you can replay):

    exist? → allow/deny → validate args → rate-limit → approval(HITL) →
    execute(timeout) → cap + injection-scan output

Returns a rich ToolResult whose `meta` carries call_id, latency, block reason, and
any injection flags — the {ok, data, error, meta} shape evals/observability want.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Callable

from agents.base import Tool, ToolResult
from agents.security import cap_output, needs_approval, scan_output_for_injection, validate_arg

logger = logging.getLogger("agents.tools")

ApprovalFn = Callable[[Tool, str], bool]


@dataclass
class ToolCall:
    call_id: str
    tool: str
    session_id: str
    ok: bool
    blocked: bool
    reason: str
    latency_ms: float
    output_injection: list = field(default_factory=list)
    truncated: bool = False
    ts: float = 0.0


class SecureToolRunner:
    def __init__(
        self,
        tools: list[Tool],
        *,
        allow: list[str] | None = None,
        deny: list[str] | None = None,
        approval_fn: ApprovalFn | None = None,
        auto_approve_dangerous: bool = False,
        log_enabled: bool = True,
    ):
        self.tools = {t.name: t for t in tools}
        self.allow = set(allow) if allow else None
        self.deny = set(deny or [])
        self.approval_fn = approval_fn
        self.auto_approve_dangerous = auto_approve_dangerous
        self.log_enabled = log_enabled
        self.calls: list[ToolCall] = []
        self._recent: dict[str, deque] = defaultdict(deque)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def call(self, name: str, arg, session_id: str = "default") -> ToolResult:
        call_id = uuid.uuid4().hex[:8]
        t0 = time.perf_counter()
        meta: dict = {"call_id": call_id, "tool": name, "session_id": session_id}

        def finish(ok, output="", error="", blocked=False, reason="", inj=None, trunc=False):
            latency = round((time.perf_counter() - t0) * 1000, 1)
            inj = inj or []
            meta.update(latency_ms=latency, blocked=blocked, reason=reason,
                        output_injection=inj, truncated=trunc)
            rec = ToolCall(call_id, name, session_id, ok, blocked, reason or error,
                           latency, inj, trunc, time.time())
            self.calls.append(rec)
            if self.log_enabled:
                logger.info(json.dumps(asdict(rec)))
            return ToolResult(ok=ok, output=output, error=error, meta=meta)

        # 1) existence + allow/deny policy
        if name not in self.tools:
            return finish(False, error=f"unknown tool '{name}'", blocked=True, reason="unknown tool")
        if name in self.deny or (self.allow is not None and name not in self.allow):
            return finish(False, error=f"tool '{name}' not permitted", blocked=True, reason="denied by policy")
        tool = self.tools[name]

        # 2) input validation
        ok, reason = validate_arg(tool, arg)
        if not ok:
            return finish(False, error=reason, blocked=True, reason=reason)

        # 3) rate limit (sliding 60s window)
        if tool.rate_limit_per_min:
            now = time.time()
            window = self._recent[name]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= tool.rate_limit_per_min:
                return finish(False, error="rate limit exceeded", blocked=True, reason="rate limited")
            window.append(now)

        # 4) human approval for dangerous / non-idempotent tools
        if needs_approval(tool):
            approved = self.approval_fn(tool, arg) if self.approval_fn else self.auto_approve_dangerous
            if not approved:
                return finish(False, error="requires human approval", blocked=True, reason="approval required")

        # 5) execute with a timeout (best-effort isolation)
        try:
            future = self._executor.submit(tool.fn, arg)
            output = str(future.result(timeout=tool.timeout_s))
        except concurrent.futures.TimeoutError:
            return finish(False, error=f"timed out after {tool.timeout_s}s", blocked=True, reason="timeout")
        except Exception as e:
            return finish(False, error=f"{type(e).__name__}: {e}", reason="tool error")

        # 6) output validation — cap size + scan for indirect injection
        output, truncated = cap_output(output)
        injection = scan_output_for_injection(output)
        return finish(True, output=output, inj=injection, trunc=truncated)

    def log_dicts(self) -> list[dict]:
        return [asdict(c) for c in self.calls]
