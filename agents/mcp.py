"""Model Context Protocol (MCP) — a minimal, spec-aligned implementation.

MCP is the "USB-C for AI tools": one open protocol so any client (Claude Desktop,
Cursor, Claude Code, your agent) can talk to any tool/data server. Donated by
Anthropic to the Linux Foundation (Dec 2025); 13k+ servers, 97M+ SDK downloads.

What MCP actually is
--------------------
- **Transport**: JSON-RPC 2.0 messages (requests with id+method+params, responses
  with result|error, and one-way notifications). Carried over **stdio** (local,
  same machine) or **Streamable HTTP** (remote, multi-client; added Nov 2025).
- **Three primitives** a server exposes on connect:
    * tools     — actions the model can invoke (have side effects)
    * resources — read-only context (files, rows, docs) addressed by URI
    * prompts   — reusable templates
- **Handshake**: client `initialize` → server advertises capabilities → client
  discovers (`tools/list`, `resources/list`) → invokes (`tools/call`, `resources/read`).

Security (why MCP needs hardening — researched)
-----------------------------------------------
- **Tool poisoning**: a server can hide instructions in a tool *description* that
  the model reads but humans don't (Invariant Labs PoC). → Pin + hash tool
  descriptions, review on change, render them to users.
- **The lethal trifecta**: private-data access + untrusted content + external
  comms = exfiltration risk. Don't grant all three to one agent path.
- **Least privilege**: OAuth 2.1 + PKCE, capability-scoped + short-lived tokens,
  incremental scope consent (2026 spec). Run each tool with minimum permission.
- **Validate server-side**: never trust the client — validate every `tools/call`
  argument against the tool's `inputSchema` before executing (done below).

This module implements a server + client + in-process transport so the protocol
runs offline; swap the transport for stdio/HTTP in production.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

PROTOCOL_VERSION = "2025-11-05"

# JSON-RPC error codes
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# ── JSON-RPC helpers ─────────────────────────────────────────────────

def _ok(req_id, result) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id, code, message) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _validate(arguments: dict, input_schema: dict) -> str | None:
    """Server-side argument validation against a JSON-Schema-ish inputSchema."""
    props = input_schema.get("properties", {})
    for name in input_schema.get("required", []):
        if name not in arguments:
            return f"missing required argument '{name}'"
    for name, val in arguments.items():
        spec = props.get(name)
        if not spec:
            continue
        t = spec.get("type")
        if t == "string" and not isinstance(val, str):
            return f"'{name}' must be a string"
        if t in ("number", "integer") and not isinstance(val, (int, float)):
            return f"'{name}' must be numeric"
        if "enum" in spec and val not in spec["enum"]:
            return f"'{name}' must be one of {spec['enum']}"
    return None


# ── Server ───────────────────────────────────────────────────────────


@dataclass
class _ToolEntry:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], str]


@dataclass
class _ResourceEntry:
    uri: str
    name: str
    mime_type: str
    reader: Callable[[], str]


@dataclass
class MCPServer:
    name: str = "vision-mcp"
    version: str = "1.0.0"
    _tools: dict[str, _ToolEntry] = field(default_factory=dict)
    _resources: dict[str, _ResourceEntry] = field(default_factory=dict)

    # registration -----------------------------------------------------
    def add_tool(self, name, description, input_schema, handler) -> None:
        self._tools[name] = _ToolEntry(name, description, input_schema, handler)

    def add_resource(self, uri, name, reader, mime_type="text/plain") -> None:
        self._resources[uri] = _ResourceEntry(uri, name, mime_type, reader)

    def add_agent_tool(self, tool) -> None:
        """Bridge an agents-lab Tool (single string input) onto MCP."""
        schema = {"type": "object",
                  "properties": {"input": {"type": "string"}},
                  "required": ["input"]}
        self.add_tool(tool.name, tool.description, schema,
                      lambda args: str(tool.fn(args.get("input", ""))))

    # request handling -------------------------------------------------
    def handle(self, request: dict) -> dict | None:
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {}) or {}

        if request.get("jsonrpc") != "2.0" or not method:
            return _err(req_id, INVALID_REQUEST, "invalid JSON-RPC request")

        if method == "initialize":
            return _ok(req_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": self.name, "version": self.version},
            })
        if method == "tools/list":
            return _ok(req_id, {"tools": [
                {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
                for t in self._tools.values()
            ]})
        if method == "tools/call":
            return self._call_tool(req_id, params)
        if method == "resources/list":
            return _ok(req_id, {"resources": [
                {"uri": r.uri, "name": r.name, "mimeType": r.mime_type}
                for r in self._resources.values()
            ]})
        if method == "resources/read":
            return self._read_resource(req_id, params)

        return _err(req_id, METHOD_NOT_FOUND, f"method not found: {method}")

    def _call_tool(self, req_id, params) -> dict:
        name = params.get("name")
        arguments = params.get("arguments", {}) or {}
        tool = self._tools.get(name)
        if not tool:
            return _err(req_id, INVALID_PARAMS, f"unknown tool: {name}")
        # SECURITY: validate args server-side before executing.
        problem = _validate(arguments, tool.input_schema)
        if problem:
            return _err(req_id, INVALID_PARAMS, problem)
        try:
            output = tool.handler(arguments)
            return _ok(req_id, {"content": [{"type": "text", "text": str(output)}], "isError": False})
        except Exception as e:
            # Tool errors are returned as result.isError, not protocol errors.
            return _ok(req_id, {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}],
                                "isError": True})

    def _read_resource(self, req_id, params) -> dict:
        uri = params.get("uri")
        res = self._resources.get(uri)
        if not res:
            return _err(req_id, INVALID_PARAMS, f"unknown resource: {uri}")
        return _ok(req_id, {"contents": [
            {"uri": res.uri, "mimeType": res.mime_type, "text": str(res.reader())}
        ]})


# ── Transport + Client ───────────────────────────────────────────────


class InProcessTransport:
    """Direct client↔server call (no network). Swap for stdio/HTTP in prod."""

    def __init__(self, server: MCPServer):
        self.server = server

    def request(self, message: dict) -> dict:
        return self.server.handle(message)


class MCPClient:
    def __init__(self, transport: InProcessTransport):
        self.transport = transport
        self._id = 0

    def _rpc(self, method: str, params: dict | None = None) -> Any:
        self._id += 1
        resp = self.transport.request(
            {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        )
        if "error" in resp:
            raise RuntimeError(f"MCP error {resp['error']['code']}: {resp['error']['message']}")
        return resp["result"]

    def initialize(self) -> dict:
        return self._rpc("initialize", {"protocolVersion": PROTOCOL_VERSION})

    def list_tools(self) -> list[dict]:
        return self._rpc("tools/list")["tools"]

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self._rpc("tools/call", {"name": name, "arguments": arguments})

    def list_resources(self) -> list[dict]:
        return self._rpc("resources/list")["resources"]

    def read_resource(self, uri: str) -> str:
        contents = self._rpc("resources/read", {"uri": uri})["contents"]
        return contents[0]["text"] if contents else ""


# ── Offline demo ─────────────────────────────────────────────────────


def _demo_server() -> MCPServer:
    server = MCPServer()
    server.add_tool(
        "add", "Add two numbers.",
        {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
         "required": ["a", "b"]},
        lambda args: args["a"] + args["b"],
    )
    server.add_resource(
        "doc://admissions/gpa", "CS GPA policy",
        lambda: "The Computer Science program requires a minimum GPA of 3.2.",
    )
    return server


def main() -> int:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    client = MCPClient(InProcessTransport(_demo_server()))
    info = client.initialize()
    print(f"initialized: {info['serverInfo']} (protocol {info['protocolVersion']})")
    print("tools:", [t["name"] for t in client.list_tools()])
    print("call add(2,3):", client.call_tool("add", {"a": 2, "b": 3}))
    try:  # server-side validation rejects malformed args
        client.call_tool("add", {"a": 2})
    except RuntimeError as e:
        print("call add(bad): rejected →", e)
    print("resources:", [r["uri"] for r in client.list_resources()])
    print("read:", client.read_resource("doc://admissions/gpa"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
