# HR Compass AI

An agentic HR assistant for a fictional company, **Solara Dynamics, Inc.**,
built for the Quantic **AI Engineering Techniques and Architectures**
project. See `PROJECT_SPEC.md` for the authoritative assignment
requirements and `docs/architecture.md` for the target system design.

**Project status: Phase 2 — MCP tool layer.** Phase 1 delivered the policy
corpus, synthetic HR datasets, and RAG pipeline. Phase 2 adds a real MCP
server (8 tools) and MCP client, wired to the Phase 1 RAG/data modules, plus
deterministic tool-level proof scripts for the two required demo workflows.
The LLM agent orchestrator, chat UI, CI/CD, deployment, and evaluation
harness are **not yet implemented** — see "Remaining work" below and
`design-and-evaluation.md` for the full plan.

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
scripts/           Deterministic tool-level demo workflows + smoke test (Phase 2)
mcp/               MCP docs pointer only — see mcp/README.md for why the code lives in mcp_server/mcp_client
agent/             LLM agent orchestrator (Phase 3, not yet implemented)
evaluation/        Evaluation harness and cases (Phase 3)
tests/             Automated tests (pytest)
docs/              Architecture and acceptance-checklist reference docs
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

This launches the server as a subprocess, lists all 8 discovered tools
with their input schemas, and makes one real tool call
(`lookup_employee_profile`), printing everything. Example output:

```
Discovered 8 tools:

- search_policy_documents: Search the Solara Dynamics policy corpus and return ranked, citable evidence...
    input_schema: {"properties": {"query": {...}, "top_k": {...}, "document_id": {...}}, ...}
- get_policy_section: ...
- lookup_employee_profile: ...
- check_pto_balance: ...
- lookup_benefits_status: ...
- check_policy_compliance: ...
- draft_hr_email: ...
- create_mock_hr_ticket: ...

Calling lookup_employee_profile(employee_id='EMP-1001')...
{
  "employee_id": "EMP-1001",
  "full_name": "Jordan Vance",
  ...
}
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
start first) and inherits your shell's environment by default, so
`EMBEDDING_BACKEND=hashing python -m scripts.mcp_smoke_test` works exactly
as you'd expect.

### Run the two deterministic demo workflows

These prove the MCP tool layer supports both required workflows with real,
citation-bearing tool calls — no LLM agent yet (that's Phase 3); the tool
call sequence is hard-coded in `scripts/demo_workflows.py`.

```bash
python -m scripts.run_pto_demo                # EMP-1001, 3 days (happy path)
python -m scripts.run_pto_demo EMP-1009 3 2    # edge case: insufficient balance
python -m scripts.run_remote_work_demo         # EMP-1002 (interstate) and EMP-1003 (international)
```

Each prints a JSON operational trace (see `core/trace.py`) and the final
`check_policy_compliance` result, including policy citations spanning
multiple documents for the remote-work case.

## Running tests

```bash
pytest
```

82 tests: the 38 from Phase 1 (policy parsing, chunking, vector store,
retrieval, HR data integrity) plus 44 from Phase 2 covering the MCP server,
client, all 8 tools (through the real protocol), and both demo workflows.
The suite is fully offline and deterministic — it builds a real index over
the actual Solara Dynamics corpus once per session using the hashing
embedding backend (never fastembed/network), and every MCP test spawns a
genuine `mcp_server.server` subprocess over stdio rather than calling tool
functions in-process, so at least one test proves real MCP discovery and
invocation through the protocol as required. Expect the full run to take
roughly 45-60 seconds due to the ~20 subprocess spins. A `network` pytest
marker is reserved for an opt-in test that exercises the real `fastembed`
model download; it is excluded by default (see `pytest.ini`) since this
project's build/CI environment cannot always reach Hugging Face.

## Demo employees (for the two required workflows)

The synthetic dataset in `mock_data/` includes employees deliberately set
up to exercise the assignment's two required demo workflows, already
provable at the MCP tool level today (see "Run the two deterministic demo
workflows" above) and to be driven by an LLM agent in Phase 3:

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

## Remaining work (Phase 3+)

Not yet implemented, per `PROJECT_SPEC.md`: the LLM agent orchestrator
(choosing which MCP tools to call and synthesizing a final natural-language
answer — Phase 2 proves the tool layer works with a hard-coded call
sequence, not an agent's decisions), web chat UI (`/chat`, `/health`),
CI/CD (GitHub Actions), free-tier deployment, and the 20-30 case evaluation
harness with groundedness/citation/tool-selection/safety metrics and an
ablation. See `design-and-evaluation.md` "Remaining Work" for the detailed
plan.

## AI tooling

This project was built with Claude Code as an AI pair-programming tool.
See `ai-tooling.md` for the development log.
