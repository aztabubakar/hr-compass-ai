"""A thin async MCP client for HR Compass AI.

This package talks to `mcp_server` **only** through the Model Context
Protocol (stdio transport, via the official `mcp` SDK's `ClientSession`).
It never imports `mcp_server.tools` or calls a tool implementation function
directly — every tool invocation is a real MCP `tools/call` request over
the wire (well, over a pipe) to a separate server process. See
`client.py` and design-and-evaluation.md's "MCP server/client separation"
section for why that separation matters.
"""
