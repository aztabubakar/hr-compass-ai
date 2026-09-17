# AI Tooling

## Claude Code
Claude Code is being used as an AI-assisted software engineering tool to scaffold, implement, test, debug, and document the application. Keep a concise development log here describing major tasks delegated to Claude Code, what worked well, what required human review/correction, and important design decisions retained by the student.

Do not claim that AI-generated code was manually written. The student remains responsible for reviewing correctness, security, architecture, testing, and the submitted work.

## Development log

### Phase 1 — project foundation + RAG foundation
Delegated to Claude Code: inspecting the starter spec/architecture docs and
producing an implementation plan; inventing the fictional company (Solara
Dynamics, Inc.) and authoring the full 12-document synthetic policy corpus
(as structured content plus a Markdown/HTML/PDF generator); authoring the
synthetic structured HR datasets (employees, PTO, benefits, tickets) with
deliberate edge cases and the deterministic demo employees for the two
required workflows; implementing the RAG ingestion pipeline (multi-format
parsing, cleaning, chunking, embeddings, a persisted FAISS vector store, and
a top-k retrieval interface); and writing the automated test suite (38
tests, all passing).

What worked well: Claude Code caught its own dependency mistake mid-build —
the default `pip install sentence-transformers` pulled a ~5.9 GB
CUDA-enabled PyTorch install in the sandbox environment, which it diagnosed
and replaced with `fastembed` (same embedding model, ONNX Runtime, no
torch/CUDA, ~376 MB total), documenting the tradeoff in
`design-and-evaluation.md` rather than silently swapping libraries.

What required human review: the choice to use `fastembed` instead of the
CLAUDE.md-suggested `sentence-transformers` package, and the corresponding
retrieval-quality tradeoff of the offline "hashing" fallback backend used
only for tests, should be reviewed by the student before Phase 2 builds on
top of it — see `design-and-evaluation.md` Sections 2 and 4.4, and Section 8
for open decisions (LLM provider, chat UI shape, MCP transport) the student
should make explicitly for Phase 2.

Retained by the student: verifying the fictional company/policy details are
internally consistent and reviewing the synthetic data for realism before
it is used in the Phase 2 agent demo and evaluation set.

### Phase 2 — MCP tool layer
Delegated to Claude Code: reading the Phase 1 code (not assuming its
interfaces) before adding anything on top of it; implementing a real MCP
server (`mcp_server/`, 8 tools) and an async MCP client (`mcp_client/`)
using the official `mcp` Python SDK over stdio; a deterministic
policy-compliance rule engine (`mcp_server/compliance.py`) built from the
specific numeric thresholds already present in the Phase 1 policy corpus;
a confirmation-gated mock ticket store; a structured operational-trace
model (`core/trace.py`); two deterministic, non-agentic proof scripts for
the PTO and remote-work workflows (`scripts/demo_workflows.py`); and 44
new automated tests, all exercising the server through the real protocol
(no in-process shortcuts).

What worked well, verified empirically rather than assumed: before writing
any server code, Claude Code installed the `mcp` SDK and ran small
throwaway scripts to confirm exactly how it behaves — (1) whether a local
`mcp/` directory (already present in the starter repo) would collide with
the installed `mcp` package (it does not, when that local directory has no
`__init__.py`, but its submodules still can't be reached because `mcp.server`
is already a real SDK subpackage — which is why the actual code lives in
`mcp_server/`/`mcp_client/` instead), (2) that a tool must return a typed
pydantic model rather than a bare `dict` for the SDK to populate structured,
machine-readable output, and (3) that raising the SDK's `ToolError` with a
specific message reaches the client cleanly (no stack trace), while an
unexpected exception is generically masked — which shaped where `ToolError`
is used deliberately versus where a normal "no record found" result is
returned instead. This upfront empirical check caught what would otherwise
have been a confusing, hard-to-diagnose import bug.

What required a mid-implementation fix: the MCP SDK's `StdioServerParameters`
defaults to a security-filtered minimal environment for a launched
subprocess (not the full parent environment), which silently broke the
demo scripts' ability to pass `EMBEDDING_BACKEND=hashing`/`RAG_INDEX_DIR`
through to the server process. Claude Code diagnosed this from the actual
SDK source rather than guessing, and changed `MCPClient`'s default to
inherit the full parent environment, documenting why this is appropriate
for a first-party, trusted server (and would not be appropriate for an
arbitrary third-party MCP server).

What requires human review: the `check_policy_compliance` rule table's
business-day approximation of the policy corpus's calendar-day interstate
thresholds (see design-and-evaluation.md Section 6.5) is a documented
simplification, not a precise reproduction of the policy text, and should
be revisited before Phase 3. The choice of `mcp` SDK version (2.2.0, whose
server API — `MCPServer`, renamed from the older `FastMCP` — is quite new)
should also be sanity-checked against whatever version the grading/CI
environment resolves.

Retained by the student: confirming the deterministic compliance
thresholds match the intended policy interpretation, and deciding the
Phase 3 items flagged in design-and-evaluation.md Section 9 (LLM provider,
chat UI framework, deployment topology).
