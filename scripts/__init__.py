"""Deterministic, tool-level proof scripts for HR Compass AI.

Nothing in this package runs an LLM. Each workflow function here calls
real MCP tools (over stdio, via `mcp_client.MCPClient`) in a fixed
sequence to prove the MCP tool layer can support the two required demo
workflows *before* the Phase 3 agent orchestrator exists to choose that
sequence itself.
"""
