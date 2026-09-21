# HR Compass AI

An agentic HR assistant for a fictional company, **Solara Dynamics, Inc.**,
built for the Quantic **AI Engineering Techniques and Architectures**
project. See `PROJECT_SPEC.md` for the authoritative assignment
requirements and `docs/architecture.md` for the system architecture
diagram (Mermaid + text) and component overview.

**Project status: Phase 4 — evaluation, CI/CD, deployment configuration.**
Phase 1 delivered the policy corpus, synthetic HR datasets, and RAG
pipeline. Phase 2 added a real MCP server (8 tools) and MCP client. Phase 3
added a real LLM-driven agent (**Google Gemini** by default, Groq fully
supported as an alternative — see "Configuring the LLM" below) that
dynamically discovers and calls those MCP tools — choosing which ones and
in what order itself, not from a hard-coded sequence — plus a FastAPI web
app (`/health`, `/chat`, `/api/tools`) and a browser chat UI. Phase 4 added
a 25-case evaluation harness (deterministic + live modes, a RAG top-k
ablation), GitHub Actions CI (tests, MCP/app smoke tests, deterministic
evaluation, test-gated deployment trigger), and Render deployment
configuration. **All of that is complete in code and tests; actually
deploying to Render and running the live-model evaluation both require
resources (a Render account, a `GEMINI_API_KEY`) not available in this
development environment** — see `deployed.md` and
`design-and-evaluation.md` section 12 ("Remaining Work") for the precise,
honest list of what's left.

## The company

Solara Dynamics, Inc. is a fictional ~460-person B2B SaaS company (product:
"Solara Flow") headquartered in Austin, TX with offices in Denver, CO and
Raleigh, NC, plus a distributed remote workforce and a small international
footprint via Employer-of-Record (EOR) arrangements in the UK and Canada.
Full company and policy details are documented in `design-and-evaluation.md`.

All employee data, tickets, and policy text in this repository are
**entirely synthetic**. No real company, person, or legal advice is
represented.

## Repository layout

```
policies/         Synthetic HR policy corpus (12 documents, md/html/pdf)
  content.py         Canonical policy text (single source of truth)
  generate_corpus.py Renders content.py into the actual corpus files
mock_data/         Synthetic structured HR data (employees, PTO, benefits, tickets)
  models.py          Pydantic schema
  loader.py          Loads + cross-validates the JSON files
core/              Shared, dependency-light utilities (paths, operational trace model)
rag/               RAG ingestion + retrieval pipeline (Phase 1)
mcp_server/        MCP server: 8 tools wrapping rag/ + mock_data/ (Phase 2)
mcp_client/        Async MCP client (Phase 2)
llm/               Provider-agnostic LLM abstraction (Gemini default, Groq alternative, FakeLLMProvider for tests) (Phase 3)
agent/             LLM agent orchestrator: bounded tool-calling loop, prompts, structured output (Phase 3)
app/               FastAPI web app (/health, /chat, /api/tools) + browser chat UI (Phase 3)
scripts/           Deterministic demo workflows + MCP/app/agent/deployment smoke tests (Phase 2/3/4)
mcp/               MCP docs pointer only — see mcp/README.md for why the code lives in mcp_server/mcp_client
evaluation/        25-case evaluation harness: cases, metrics, offline/live runners, ablation (Phase 4)
tests/             Automated tests (pytest)
docs/              Architecture diagram and acceptance-checklist reference docs
.github/workflows/ GitHub Actions CI: tests, smoke tests, deterministic evaluation, deploy gate (Phase 4)
```

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # edit if you want to override defaults
```

## Building the RAG index

The policy corpus (`policies/*.md`, `*.html`, `*.pdf`) is committed to the
repo, but the vector index built from it is a derived build artifact and is
**not** committed (see `.gitignore`). Build it locally with:

```bash
python -m rag.ingest
```

This parses every policy document, cleans and chunks it, embeds the chunks,
and writes a persisted FAISS index + metadata to `rag_index/` (configurable
via `RAG_INDEX_DIR`-equivalent `index_dir` in `rag/config.py`).

By default this uses the `fastembed` backend (`sentence-transformers/all-MiniLM-L6-v2`,
served via ONNX Runtime), which downloads a small model file from Hugging
Face on first use and therefore **requires normal internet access**. If you
are working offline, or want a fast/deterministic build with no network
calls, use the dependency-free hashing fallback instead:

```bash
EMBEDDING_BACKEND=hashing python -m rag.ingest
```

The hashing backend is lexical (bag-of-words), not semantic, so retrieval
quality is noticeably lower — it exists for tests, offline development, and
as an emergency fallback, not as the production default. See
`rag/embeddings.py` and `design-and-evaluation.md` for the full rationale.

### Querying the index

```python
from rag.retrieval import retrieve

response = retrieve("Can I work from another state for six weeks?", top_k=5)
for r in response.results:
    print(f"[{r.score:.3f}] {r.doc_id} / {r.section}: {r.snippet}")
```

`retrieve()` (and `PolicyRetriever.query()`) return, for each result: the
document ID, title, section heading, source file path, a citation snippet,
and a relevance score — everything the MCP `search_policy_documents` tool
(see "MCP server and client" below) or a future agent needs to ground and
cite an answer. `response.has_sufficient_evidence` flags whether the top
score cleared `RAG_MIN_RELEVANCE_SCORE`, a first, retrieval-level guardrail
against answering from weak or irrelevant matches.

## Regenerating the policy corpus

The corpus files under `policies/*.md|*.html|*.pdf` are generated, not
hand-edited. To change policy content, edit `policies/content.py` (the
single source of truth for all 12 documents) and re-run:

```bash
python -m policies.generate_corpus
```

## MCP server and client

The MCP server (`mcp_server/`) exposes 8 tools over stdio using the
official Python MCP SDK (`mcp.server.mcpserver.MCPServer`); the MCP client
(`mcp_client/`) connects to it using the same SDK's `ClientSession`. Code
lives in `mcp_server/`/`mcp_client/` rather than in `mcp/` — see
`mcp/README.md` for why (short version: the SDK package is itself called
`mcp`, and `mcp.server` is already one of its real subpackages).

### Start the MCP server standalone

```bash
python -m mcp_server.server
```

This blocks, speaking MCP over stdin/stdout, until the process is killed —
it's meant to be launched by a client (or by an MCP-compatible tool like
the [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector)),
not run interactively. `Ctrl+C` to stop it.

### Inspect available tools / run the MCP client smoke test

```bash
python -m scripts.mcp_smoke_test
```

Builds its own hermetic RAG index (hashing backend, to a temp directory)
so it needs no prior setup, launches the server as a subprocess, discovers
all 8 tools, calls one RAG-backed tool (`search_policy_documents`) and one
structured-data tool (`lookup_employee_profile`), and confirms an unknown
employee ID raises a clean tool-level error rather than crashing. Prints a
PASS/FAIL-style trace and exits non-zero on any failure, which is what
lets `.github/workflows/ci.yml` use it as a real CI gate, not just a
manual print. Example output:

```
Building a hermetic RAG index (hashing backend) at /tmp/hr_compass_mcp_smoke_xxxx ...
Connected to MCP server. Discovering tools ...
OK: discovered 8 tools, including all 8 required ones.
Calling a RAG-backed tool: search_policy_documents ...
OK: 3 result(s) returned.
Calling a structured-data-backed tool: lookup_employee_profile ...
OK: looked up Jordan Vance.
Confirming a clean tool-level error for an unknown employee ID ...
OK: unknown employee ID raised a clean MCPToolError, not a crash.

MCP discovery/tool-call smoke test PASSED.
```

### Example: calling a tool from your own code

```python
import anyio
from mcp_client.client import MCPClient

async def main():
    async with MCPClient() as client:
        tools = await client.list_tools()
        result = await client.call_tool("check_pto_balance", {"employee_id": "EMP-1001"})
        print(result)

anyio.run(main)
```

`MCPClient` launches the server subprocess itself (no separate process to
start first) and inherits your shell's environment by default, so setting
`EMBEDDING_BACKEND`/`RAG_INDEX_DIR` before running your own script works
exactly as you'd expect (`scripts/mcp_smoke_test.py` above builds and
points at its own temp index instead, so it needs no such setup).

### Run the two deterministic demo workflows

These prove the MCP tool layer supports both required workflows with real,
citation-bearing tool calls, independent of the LLM agent; the tool call
sequence is hard-coded in `scripts/demo_workflows.py` on purpose, as a
tool-layer smoke test and a source of fixtures — the deployed agent below
does **not** use this hard-coded sequence; it decides its own.

```bash
python -m scripts.run_pto_demo                # EMP-1001, 3 days (happy path)
python -m scripts.run_pto_demo EMP-1009 3 2    # edge case: insufficient balance
python -m scripts.run_remote_work_demo         # EMP-1002 (interstate) and EMP-1003 (international), six weeks
```

Each prints a JSON operational trace (see `core/trace.py`) and the final
`check_policy_compliance` result, including policy citations spanning
multiple documents for the remote-work case.

## Configuring the LLM (Gemini / Groq)

The agent needs a real LLM to run, via a thin provider abstraction (`llm/`)
so `agent/` never depends on a specific vendor SDK — set `LLM_PROVIDER` to
pick one, no code changes needed.

**Gemini (default/primary in production):**

```bash
# 1. Get a free API key: https://aistudio.google.com/apikey
# 2. Copy .env.example to .env and set:
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
# GEMINI_MODEL is optional; defaults to gemini-2.5-flash-lite if unset.
```

**Groq (fully supported alternative):**

```bash
# 1. Get a free API key: https://console.groq.com/keys
# 2. Copy .env.example to .env and set:
LLM_PROVIDER=groq
GROQ_API_KEY=your-key-here
# GROQ_MODEL is optional; a sensible current default is used if unset.
```

Without the selected provider's API key set, the app still starts and
`/health` still works (`llm_configured: false`, `provider` reflects
`LLM_PROVIDER`); `/chat` returns a clean `503` instead of crashing, and
picking an unrecognized `LLM_PROVIDER` value fails clearly at startup
rather than silently falling back to another provider. The automated test
suite never needs a real key — it always uses
`llm.fake_provider.FakeLLMProvider`, a deterministic scripted stand-in (see
"Running tests" below).

Both `GeminiProvider` and `GroqProvider` retry a timeout, connection
error, or rate limit/quota error up to twice with backoff (honoring the
API's `Retry-After` header when given); an invalid request, an auth
failure, or a context-length-exceeded request is never retried, since
retrying an unchanged bad request fails identically every time — and for
Gemini specifically, the SDK's own built-in retry behavior is explicitly
disabled (`retry_options=HttpRetryOptions(attempts=1)`) so the two retry
layers can never stack into the kind of multi-minute-pending-request
problem a naive integration would risk. Every LLM turn and MCP tool call
is timed, and that latency is part of the operational trace shown in the
"Agent Activity" panel. An end-to-end `AGENT_RUN_TIMEOUT_SECONDS` (default
60s, see `.env.example`) bounds the whole `/chat` request — covering every
LLM turn and tool call in the agent's bounded loop, not just one LLM
call — so a stuck upstream call can never hang the request (or the
browser's "thinking" state) indefinitely; it fails with a clean `504`
instead. See `design-and-evaluation.md` section 7.10 for the full
write-up on the Groq-era retry/latency work, including what caused a
production PTO request to need ~8 tool calls and how citation precision,
LLM-payload size, and retry behavior were fixed — the same fixes apply
identically to Gemini, since they live in provider-agnostic code
(`agent/tool_results.py`, `agent/orchestrator.py`).

## Running the web app

```bash
python -m rag.ingest              # build the RAG index, if you haven't yet
uvicorn app.main:app --reload
```

Then open <http://localhost:8000> for the chat UI. The MCP server is
launched and connected automatically when the app starts (one shared,
long-lived subprocess — see `app/main.py`'s `lifespan()`), and closed when
the app shuts down; you do not need to start `mcp_server.server` yourself.

Endpoints:

- `GET /health` — `{"status": "ok", "mcp": "connected"|"unavailable", "rag": "ok"|"not_built", "llm_configured": true|false}`.
- `POST /chat` — `{"message": "...", "conversation_id": "...?"}` -> answer, citations, snippets, operational trace, clarification/confirmation/escalation flags, and the `conversation_id` to continue the conversation.
- `GET /api/tools` — the discovered MCP tools and their input schemas (debugging/demo use).

### Example demo prompts

Try these in the chat UI (also listed there as clickable examples):

- *"What is Solara's paid holiday policy?"* — policy-only, grounded, no employee lookup.
- *"My employee ID is EMP-1001. Can I take three days of PTO next week?"* — PTO workflow.
- *"I'm EMP-1002. Can I work from Colorado for six weeks?"* — remote work, interstate.
- *"I'm EMP-1003. Can I work from Portugal for six weeks?"* — remote work, international.
- *"EMP-1004 — am I enrolled in medical benefits?"* — structured-data lookup.
- *"Can I work somewhere else next month?"* — ambiguous; the agent asks a clarifying question.
- *"How much PTO do I have?"* — missing employee ID; the agent asks for it.
- *"What is the company's stock option vesting schedule?"* — out-of-corpus; the agent says so rather than guessing.
- *"I want to report harassment by a coworker."* — sensitive issue; grounded reporting guidance, offers (does not create) a mock HR case.

## Smoke tests

```bash
python -m scripts.mcp_smoke_test          # MCP discovery + tool calls (CI-safe, no API key)
python -m scripts.app_smoke_test          # FastAPI app-start + /health, /api/tools, / (CI-safe, no API key)
python -m scripts.agent_smoke_test        # live agent test: real LLM call (Gemini/Groq per LLM_PROVIDER)
python -m scripts.deployment_smoke_test <url>   # smoke test an already-deployed URL, e.g. on Render
```

`scripts/mcp_smoke_test.py` and `scripts/app_smoke_test.py` are what
`.github/workflows/ci.yml` runs on every push/PR — each builds its own
hermetic hashing-backend RAG index, needs no LLM API key, and exits
non-zero on failure so CI can gate on them. `scripts/agent_smoke_test.py`
is the only one of the four that makes a real LLM call: it prints setup
guidance and exits cleanly (code 0) if the configured provider's API key
isn't set rather than failing, is meant for a human to run locally to
sanity-check the live path, and is **never** run in CI. `scripts/deployment_smoke_test.py`
takes a base URL (see "Deploying to Render" below) and checks `/`,
`/health`, and `/api/tools`; pass `--test-chat` to also exercise `POST
/chat`, gated on that deployment's own `/health` reporting
`llm.configured: true`.

## Running tests

```bash
pytest
```

145 tests: 38 from Phase 1 (policy parsing, chunking, vector store,
retrieval, HR data integrity), 65 from Phase 2 (44 covering the MCP server/
client/all 8 tools/both demo workflows through the real protocol, plus 21
new compliance-engine boundary regression tests — see "Exact calendar-day
compliance fix" in `design-and-evaluation.md`), 28 from Phase 3 (17
agent-orchestration tests proving genuine model-driven tool selection for
every required behavior, plus 11 FastAPI endpoint tests), and 14 from
Phase 4 (13 covering the evaluation harness itself — case loading, scoring
rules, and a handful of cases run through the real orchestrator/MCP layer
— plus 1 new `/health` degraded-status test). 219 tests pass as of the
Gemini migration, the difference made up of `tests/test_gemini_provider.py`
(29 tests covering `GeminiProvider` end to end against mocked
`google-genai` objects — see its module docstring for the full scenario
list) plus a handful of smaller additions elsewhere (e.g. the `/chat`
end-to-end timeout test in `tests/test_api.py`).

The suite is fully offline and deterministic: it builds a real index over
the actual Solara Dynamics corpus once per session using the hashing
embedding backend (never fastembed/network), every MCP-level test spawns a
genuine `mcp_server.server` subprocess over stdio, and every agent-level
test uses `llm.fake_provider.FakeLLMProvider` — never a real Gemini/Groq
call — so the whole suite needs no API key and no network access beyond
what was already true in Phase 2 (this also includes
`tests/test_gemini_provider.py`, which uses hand-built `google-genai`
response/exception objects, never a live Gemini call). Expect the full run
to take roughly 60-80 seconds,
mostly subprocess spin-up time. A `network` pytest marker is reserved for
an opt-in test that exercises the real `fastembed` model download; it is
excluded by default (see `pytest.ini`) since this project's build/CI
environment cannot always reach Hugging Face.

## Demo employees (for the two required workflows)

The synthetic dataset in `mock_data/` includes employees deliberately set
up to exercise the assignment's two required demo workflows — provable both
at the MCP tool level (see "Run the two deterministic demo workflows"
above) and end-to-end through the real LLM agent (see "Example demo
prompts" above):

- **PTO workflow** ("Can I take three days of PTO next week?"):
  `EMP-1001` (Jordan Vance) has an ample PTO balance (happy path);
  `EMP-1009` (Elena Popescu) has an insufficient balance (edge case); and
  `EMP-1006` (Sam Okafor) is a new hire within the first-30-days
  restricted window (edge case).
- **Remote-work workflow** ("Can I work from another state/country for six
  weeks?"): `EMP-1002` (Priya Natarajan) is based in Austin, TX and asks
  about working from Denver, CO (interstate case); `EMP-1003` (Marcus
  Whitfield) is based in Raleigh, NC and asks about working from Lisbon,
  Portugal (international case).

See `design-and-evaluation.md` for the full roster and the policy
thresholds these scenarios are designed to exercise.

## Evaluation

```bash
python -m evaluation.run_offline          # deterministic, offline, CI-safe (25 cases)
python -m evaluation.run_offline --max-cases 5   # fast subset
python -m evaluation.run_live             # real LLM model (Gemini/Groq per LLM_PROVIDER), never run in CI
python -m evaluation.ablation             # RAG retrieval ablation: top_k=3 vs top_k=5
```

25 structured cases (`evaluation/cases.json`) across 15 categories
(straightforward/multi-document/employee-data policy questions, PTO and
remote-work workflows, benefits, expense compliance, ambiguous requests,
missing/invalid employee IDs, out-of-scope questions, sensitive
escalation, action-confirmation safety, and tool-failure/incomplete-
evidence cases), scored for groundedness, citation accuracy/completeness,
tool-selection accuracy, workflow completion, clarification/escalation
accuracy, action-safety pass rate, and p50/p95 latency. **`run_offline`
proves the agent/MCP/safety architecture works — it does not measure real
LLM quality, since its "model" is a scripted `FakeLLMProvider`; only
`run_live` (not run in this development environment — no `GEMINI_API_KEY`
was available) measures that.** Full methodology, metrics definitions,
documented limitations, and a manual live-model validation checklist are
in `evaluation/README.md`; results and analysis are in
`design-and-evaluation.md` section 10.

## Continuous integration

`.github/workflows/ci.yml` runs on every push/PR: installs dependencies,
an import/build check, the full `pytest` suite, `scripts/mcp_smoke_test.py`,
`scripts/app_smoke_test.py`, `evaluation.run_offline`, and
`evaluation.ablation` — all offline, no LLM API key or other external
service required. A second `deploy` job triggers a Render deploy hook only
after the test job succeeds and only on a push to `main`, so deployment is
genuinely gated on tests passing once the one-time setup in the job's
comments (turning off Render's own Auto-Deploy, setting the
`RENDER_DEPLOY_HOOK_URL` secret) is done — see `design-and-evaluation.md`
section 9.3 for the full explanation of why that setup step is required
for the gating claim to be true.

## Deploying to Render

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT   # the production start command
```

`render.yaml` defines a single free-tier Render "web" service running the
whole app (FastAPI + agent + MCP client/server subprocess + RAG index) in
one process — see `design-and-evaluation.md` section 9.4-9.5 for the full
design, including why the RAG index is built at Render *build* time
(`buildCommand: pip install -r requirements.txt && python -m rag.ingest`)
rather than on first request. To deploy: connect this repo in the Render
dashboard (it will detect `render.yaml`), set `GEMINI_API_KEY` in the
Render dashboard (never commit it; changing it later requires a manual
redeploy/restart for the new value to take effect), and once live,
smoke-test it:

```bash
python -m scripts.deployment_smoke_test https://your-app.onrender.com
```

`deployed.md` has the (currently placeholder) record of the actual
deployed URL, date, and commit once this has been done — see
`design-and-evaluation.md` section 12 for exactly what deployment steps
remain (none of them can be completed from this development environment,
which has no Render account or `GEMINI_API_KEY`).

## Requirements coverage

`docs/requirements-coverage-matrix.md` maps every item in
`docs/acceptance-checklist.md` to its implementation file, test/evidence,
and documentation section, with an honest status
(complete / pending-live-validation / pending-deployment).

## AI tooling

This project was built with Claude Code as an AI pair-programming tool.
See `ai-tooling.md` for the development log.
