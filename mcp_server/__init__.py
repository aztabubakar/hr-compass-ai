"""The HR Compass AI MCP server: exposes RAG retrieval and synthetic HR
data as real MCP tools over stdio.

Note on package naming: this package is deliberately *not* named `mcp`
(even though the pre-existing `mcp/` directory in this repository is the
assignment's nominal "MCP directory") because the official Python MCP SDK
is itself imported as `import mcp`. A local top-level package also named
`mcp` would, depending on the exact runtime/sys.path configuration, either
be shadowed by or shadow the real SDK, and its submodules would collide
outright with real SDK submodules (`mcp.server` is already used by the
SDK). `mcp/README.md` documents this and points here. See
design-and-evaluation.md's MCP Architecture section for the full
rationale.
"""
