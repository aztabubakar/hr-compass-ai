# HR Compass AI

An agentic HR assistant for a fictional company, **Solara Dynamics, Inc.**,
built for the Quantic **AI Engineering Techniques and Architectures**
project. See `PROJECT_SPEC.md` for the authoritative assignment
requirements and `docs/architecture.md` for the target system design.

**Project status: Phase 1 — project foundation + RAG foundation.** This
phase delivers the policy corpus, synthetic HR datasets, and a working
retrieval-augmented-generation (RAG) ingestion/retrieval pipeline. The MCP
server/client, agent orchestrator, chat UI, CI/CD, deployment, and
evaluation harness are **not yet implemented** — see "Remaining work" below
and `design-and-evaluation.md` for the full plan.

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
core/              Shared, dependency-light utilities (e.g. core/paths.py)
rag/               RAG ingestion + retrieval pipeline (this phase's focus)
agent/             Agent orchestrator (Phase 2)
mcp/               MCP server/client (Phase 2)
evaluation/        Evaluation harness and cases (Phase 2)
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
and a relevance score — everything a future MCP `search_policy_documents`
tool or agent needs to ground and cite an answer. `response.has_sufficient_evidence`
flags whether the top score cleared `RAG_MIN_RELEVANCE_SCORE`, a first,
retrieval-level guardrail against answering from weak or irrelevant matches.

## Regenerating the policy corpus

The corpus files under `policies/*.md|*.html|*.pdf` are generated, not
hand-edited. To change policy content, edit `policies/content.py` (the
single source of truth for all 12 documents) and re-run:

```bash
python -m policies.generate_corpus
```

## Running tests

```bash
pytest
```

38 tests cover policy parsing (all three formats, plus integration tests
against the real corpus), chunking and citation-metadata preservation,
vector store persistence, single- and multi-document retrieval, and
synthetic HR data referential integrity. The default test run is fully
offline and deterministic (it uses the hashing embedding backend and small
fixture documents, never the real corpus's fastembed-built index). A
`network` pytest marker is reserved for an opt-in test that exercises the
real `fastembed` model download; it is excluded by default (see
`pytest.ini`) since this project's build/CI environment cannot always reach
Hugging Face.

## Demo employees (for the two Phase 2 workflows)

The synthetic dataset in `mock_data/` includes employees deliberately set
up to exercise the assignment's two required demo workflows once the agent
orchestrator (Phase 2) exists:

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

## Remaining work (Phase 2+)

Not yet implemented, per `PROJECT_SPEC.md`: MCP server + client (>=5 tools),
agent orchestrator with tool discovery/invocation and operational tracing,
web chat UI (`/chat`, `/health`), CI/CD (GitHub Actions), free-tier
deployment, and the 20-30 case evaluation harness with groundedness/
citation/tool-selection/safety metrics and an ablation. See
`design-and-evaluation.md` "Remaining Work" for the detailed plan.

## AI tooling

This project was built with Claude Code as an AI pair-programming tool.
See `ai-tooling.md` for the development log.
