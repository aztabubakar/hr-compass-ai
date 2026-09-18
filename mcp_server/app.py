"""The shared MCPServer instance.

Kept in its own module (separate from `tools.py`, which registers tools on
it via decorators, and from `server.py`, which runs it) purely to avoid a
circular import: `tools.py` needs to import `server` to decorate functions
with `@server.tool()`, and `server.py` needs to import `tools` so those
decorators actually execute and register the tools before the server
starts listening.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

server = MCPServer(
    name="hr-compass-ai",
    title="HR Compass AI MCP Server",
    instructions=(
        "Tools for the Solara Dynamics, Inc. HR assistant: policy retrieval "
        "(RAG), synthetic employee/PTO/benefits lookups, deterministic "
        "policy-compliance evaluation, and mock (non-destructive) HR "
        "actions. All employee and ticket data is synthetic."
    ),
)
