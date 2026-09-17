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

### Phase 3 — LLM agent + web application

Delegated to Claude Code: fixing a known simplification from Phase 2 (the
remote-work compliance check compared duration in business days even where
Solara's own policy text states thresholds in calendar days) by re-reading
the literal policy wording and rewriting `mcp_server/compliance.py` to use
exact calendar-day comparisons for the calendar-day-defined tiers, with a
business-day conversion applied only where the policy itself states a
business-day threshold; a provider-agnostic `LLMProvider` abstraction
(`llm/`) with a Groq implementation over the OpenAI-compatible `openai`
SDK and a scripted `FakeLLMProvider` for offline tests; an `AgentOrchestrator`
(`agent/orchestrator.py`) that discovers MCP tools through the real MCP
client (no direct imports of tool implementations), runs a bounded
LLM-driven tool-calling loop, and returns a structured `AgentResponse`
(answer, citations, snippets, trace, clarification/confirmation/escalation
flags); a versioned system prompt (`agent/prompts.py`); a FastAPI
application (`app/`) exposing `/health`, `/chat`, and `/api/tools`, with
ephemeral in-memory conversation state; a static HTML/CSS/JS chat UI
(`app/static/index.html`) showing citations, retrieved snippets, and a
collapsible operational-trace panel; a manual `scripts/agent_smoke_test.py`
for live Groq verification (not run in CI); and 28 new automated tests
covering the compliance boundary fix, agentic tool selection across the
required behavior scenarios, and the API layer.

What worked well: the orchestrator contains no topic-specific branching —
tool sequencing is driven entirely by whatever the LLM (real or scripted)
decides to call, which the agent-orchestrator tests verify directly by
scripting different tool sequences per scenario (policy-only, PTO,
interstate remote work, international remote work, benefits, ambiguous
request, missing/invalid employee ID, out-of-corpus question) and asserting
the orchestrator faithfully executes each without any scenario-specific
code path. The confirmation-gating design (a code-level check in the
orchestrator, not just a prompt instruction, that downgrades a premature
`confirmed=true` call on `create_mock_hr_ticket` back to `confirmed=false`
unless the prior turn actually asked for confirmation) worked as intended
and is covered by tests in both directions. Citations are compiled only
from real MCP tool results, never parsed from model text, so they cannot
be fabricated by construction.

What required a mid-implementation fix or human review: a real bug in the
tool-call loop, found through code review rather than a failing test — the
first version wrapped an entire batch of tool calls in one try/except,
so a single failing call would skip the required per-call "tool" role
responses for the remaining calls in that batch and corrupt the
OpenAI-format conversation on the next turn; fixed by moving the
try/except inside the loop, per call. A genuine production bug was also
found via a failing test: `app/main.py`'s `/health` endpoint originally read
`RagConfig().index_dir`, but `RagConfig` is a frozen dataclass whose field
defaults are computed once at module import time, so it never reflected
`RAG_INDEX_DIR` overrides set later (e.g. by tests, or by any process that
sets the env var after import); fixed to read `os.getenv("RAG_INDEX_DIR", ...)`
directly at call time. Two other test-only issues were found and fixed:
a FastAPI dependency-resolution-order case where a 503 from the
LLM-not-configured dependency preempted an expected 422 validation error
(fixed by giving that test a configured fake LLM so only the validation
path was exercised), and a `TestClient` default (`raise_server_exceptions=True`)
that re-raised an unhandled exception into the test process instead of
letting the app's global exception handler convert it to a clean 500
(fixed by setting `raise_server_exceptions=False` to observe
production-like behavior). All four fixes were self-diagnosed and
self-verified against the test suite; no human was available to review
them during this session.

No live Groq testing was performed in this phase: no `GROQ_API_KEY` was
available in this development environment. `scripts/agent_smoke_test.py`
was written and exercised only for its no-key exit path (printing setup
guidance and returning cleanly); the automated test suite exercises the
agent orchestrator exclusively through `FakeLLMProvider`. This is stated
here explicitly rather than implied otherwise.

Retained by the student: obtaining a Groq API key and running the live
smoke test before relying on this system for a real demo; reviewing the
system prompt (`agent/prompts.py`) for tone and scope before any
recording; and deciding the Phase 4 items flagged in
design-and-evaluation.md Section 10 (deployment to Render, any additional
LLM provider, authentication/multi-tenancy if ever needed beyond a course
demo).
