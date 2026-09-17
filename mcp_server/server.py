"""Entrypoint: run the HR Compass AI MCP server over stdio.

    python -m mcp_server.server

Importing `mcp_server.tools` (even though nothing in this module calls it
directly) is required and deliberate: that import is what executes the
`@server.tool()` decorators and registers all 8 tools on the shared
`server` instance from `mcp_server.app` before `run_stdio_async()` starts
listening.
"""
from __future__ import annotations

import anyio

from mcp_server import tools  # noqa: F401  (registers tools as a side effect)
from mcp_server.app import server


def main() -> None:
    anyio.run(server.run_stdio_async)


if __name__ == "__main__":
    main()
