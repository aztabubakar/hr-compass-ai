# MCP

**Status: implemented (Phase 2).** The real MCP server and client code
lives in `mcp_server/` and `mcp_client/` at the repository root, **not**
in this directory. This is a deliberate naming decision, not an
oversight: the official Python MCP SDK is itself imported as `import mcp`
(`pip install mcp`), and `mcp.server` is already a real subpackage of that
SDK, so a local top-level package also named `mcp` cannot safely hold
modules like `mcp/server.py` without colliding with the SDK. See
`design-and-evaluation.md`'s "MCP Architecture" section for the full
explanation.

- `mcp_server/` — the MCP server (8 tools: `search_policy_documents`,
  `get_policy_section`, `lookup_employee_profile`, `check_pto_balance`,
  `lookup_benefits_status`, `check_policy_compliance`, `draft_hr_email`,
  `create_mock_hr_ticket`). Run with `python -m mcp_server.server`.
- `mcp_client/` — the async MCP client (`mcp_client.client.MCPClient`),
  used by `scripts/demo_workflows.py` and the Phase 2 test suite, and by
  the Phase 3 agent orchestrator once it exists.
- `core/trace.py` — the operational trace model shared by both.

See `README.md` "MCP server and client" for how to run them and
`design-and-evaluation.md` for architecture, tool schemas, safety, and the
expected call sequence for both demo workflows.
