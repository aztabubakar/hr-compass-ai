# Design and Evaluation

This document tracks architecture decisions, design rationale, and (in
later phases) evaluation results for HR Compass AI. It is updated as each
phase is implemented. **Current status: Phase 1 (project foundation + RAG
foundation), Phase 2 (MCP server/client tool layer), Phase 3 (LLM agent
orchestrator + FastAPI web app + chat UI), and Phase 4 (evaluation harness,
CI/CD, deployment configuration) are all complete in code and tests.**
What remains is executing the actual deployment and a live-model
evaluation/validation pass, since neither a Render account/API key nor a
`GROQ_API_KEY` was available in this development environment -- see
section 12, "Remaining Work," for the precise, honest list of what a human
still needs to do before this project is demo-ready.

## 1. The company

**Solara Dynamics, Inc.** is a fictional B2B SaaS company (~460 employees)
that sells a workflow-automation product, "Solara Flow." It is headquartered
in Austin, TX, with additional offices in Denver, CO and Raleigh, NC. Most
roles are remote- or hybrid-eligible. Solara Dynamics is payroll-registered
in 14 U.S. states (TX, CO, NC, CA, NY, WA, IL, GA, FL, MA, VA, AZ, OR, TN)
and employs a small number of international team members through
Employer-of-Record (EOR) partners in the United Kingdom, Canada, Portugal,
and Ireland. Internal systems referenced throughout the policy corpus:
Compass HR (HRIS), SolaraPay (payroll), SolaraExpense (expense
reimbursement), SolaraTravel (travel booking), and SolaraSecure VPN.

This fictional profile exists specifically to give the policy corpus and
synthetic data internally consistent, realistic detail to reason over (state
payroll registration lists, EOR country lists, specific accrual rates and
thresholds), rather than generic placeholder policies.

## 2. Project structure and dependency strategy

```
policies/    Policy corpus content + generator + generated files
mock_data/   Synthetic structured HR data + pydantic schema + loader
core/        Shared utilities (paths, operational trace model)
rag/         RAG ingestion + retrieval pipeline
mcp_server/  MCP server: 8 tools wrapping rag/ + mock_data/ (Phase 2)
mcp_client/  Async MCP client (Phase 2)
llm/         Provider-agnostic LLM abstraction (Groq default, FakeLLMProvider for tests) (Phase 3)
agent/       LLM agent orchestrator: bounded tool-calling loop, prompts, structured output (Phase 3)
app/         FastAPI web app (/health, /chat, /api/tools) + static chat UI (Phase 3)
scripts/     Deterministic tool-level demo workflows + MCP/agent smoke tests (Phase 2/3)
mcp/         Documentation pointer only; see Section 6 for why
evaluation/  (Phase 4) evaluation harness and cases
tests/       pytest suite
docs/        Architecture reference docs (pre-existing, authoritative spec)
```

Rationale: the starter repository already fixed this top-level layout
(`agent/`, `mcp/`, `rag/`, `mock_data/`, `policies/`, `evaluation/`,
`tests/`), so Phase 1 built within it rather than restructuring. `core/`
was added because `rag/`, `mock_data/`, and (in Phase 2) `mcp/`/`agent/` all
need to agree on where the policy corpus, mock data, and vector index live
on disk; centralizing that in `core/paths.py` avoids each module re-deriving
`Path(__file__).parent` logic independently.

Within `rag/`, modules are split by pipeline stage (`parsers.py`,
`cleaning.py`, `chunking.py`, `embeddings.py`, `vector_store.py`,
`retrieval.py`, `ingest.py`) rather than one large file, so each stage is
independently unit-testable and swappable (e.g. the embedding backend).

### Dependency strategy and a deviation from the suggested stack

`requirements.txt` pins exact versions (a `pip freeze` of the resolved
environment) for full reproducibility, per CLAUDE.md's instruction to pin
compatible versions once the implementation is selected. Each phase added
only the dependencies it actually needed: Phase 1 the RAG stack, Phase 2
the official `mcp` SDK, Phase 3 `openai` (as an OpenAI-compatible client
for Groq — see Section 7.1), `fastapi`, and `uvicorn[standard]`. Total
installed footprint remains modest (~450 MB across all three phases,
CPU-only, no torch/CUDA) — see `requirements.txt`'s own inline comments
for what each dependency is for.

**Deviation from CLAUDE.md's preferred stack:** CLAUDE.md suggests
"SentenceTransformers or another zero-cost embedding model." We use
**`fastembed`** instead of the `sentence-transformers` package itself. In
this project's build environment, `pip install sentence-transformers` pulls
in PyTorch, and PyPI's default `torch` wheel on Linux resolves the
full CUDA/nvidia dependency chain (`nvidia-cublas`, `nvidia-cudnn`, `triton`,
etc.) — an install we measured at **5.9 GB**, with no CPU-only variant
available from PyPI alone (the CPU-only wheel is hosted at
`download.pytorch.org`, a separate package index). That footprint, and the
corresponding runtime memory use of loading torch just for inference, is a
poor fit for "modest free-tier resources" and a Render-class free tier
(typically 512 MB RAM). `fastembed` runs the *same* model named in the
original `.env.example` stub (`sentence-transformers/all-MiniLM-L6-v2`) as a
small, int8-quantized ONNX export via ONNX Runtime, with **no torch/CUDA
dependency at all** — total installed footprint for this project's full
dependency set is **~376 MB**, CPU-only. This is exactly the kind of
substitution CLAUDE.md pre-authorizes: "Do not blindly follow these
preferences if package/runtime compatibility makes another free-tier-friendly
design materially more reliable. Document deviations." That documentation is
this section.

A second, related note: this development sandbox's outbound network policy
blocks `huggingface.co` (where `fastembed` and `sentence-transformers` both
fetch model weights from by default) and `download.pytorch.org`. This means
the `fastembed` code path is implemented and unit-testable for its
interface, but could not be exercised end-to-end for real embeddings from
*inside this particular sandbox*. It is expected to work normally in a
regular development machine, GitHub Actions CI, or a Render deployment, all
of which have normal internet access to Hugging Face. To keep the pipeline
fully testable and runnable inside constrained environments regardless, a
second embedding backend — a dependency-free, deterministic hashing/bag-of-
words embedding — is implemented and used as the default for automated
tests (see Section 4.4).

## 3. Policy corpus

12 documents, covering every required topic, totaling **37 pages** (measured
objectively — see below — not estimated from a words-per-page guess),
comfortably inside the requested 30-50 page range and the assignment's
5-20 document / 30-120 page range.

| Doc ID | Title | Format | Effective Date |
|---|---|---|---|
| POL-PTO-001 | Paid Time Off (PTO) Policy | Markdown | 2024-01-01 |
| POL-HOL-001 | Company Holiday Schedule Policy | Markdown | 2024-01-01 |
| POL-RWK-001 | Remote Work Policy | Markdown | 2024-03-01 |
| POL-LOC-001 | Work Location, Interstate & International Work Policy | HTML | 2024-03-01 |
| POL-SEC-001 | Information Security & Data Protection Policy | Markdown | 2024-01-15 |
| POL-EXP-001 | Business Expense & Reimbursement Policy | HTML | 2024-01-01 |
| POL-BEN-001 | Employee Benefits Policy | Markdown | 2024-01-01 |
| POL-ONB-001 | New Hire Onboarding Policy | Markdown | 2024-01-01 |
| POL-EQP-001 | Company Equipment & Asset Policy | PDF | 2024-02-01 |
| POL-LEV-001 | Leave of Absence Policy | Markdown | 2024-01-01 |
| POL-CON-001 | Workplace Conduct & Anti-Harassment Policy | Markdown | 2024-01-01 |
| POL-TRV-001 | Business Travel Policy | PDF | 2024-04-01 |

Three source formats are used (8 Markdown, 2 HTML, 2 PDF) so the ingestion
pipeline demonstrably parses more than one format with genuinely different
extraction logic per format (see Section 4.1), not just one regex applied
to files with different extensions.

**Single source of truth, generated output.** All policy text is authored
once, as structured Python data, in `policies/content.py` (title, doc ID,
effective date, version, owner, and an ordered list of sections, each with
paragraphs and/or bullets). `policies/generate_corpus.py` renders that data
into the three target formats:

- Markdown: `#`/`##` headings, `**Label:** value` metadata lines.
- HTML: `<h1>`/`<h2>` tags, metadata as `<meta name="doc-id" ...>` tags in
  `<head>` (not scraped from visible text).
- PDF (via `fpdf2`): a title block, plain-text `Label: value` metadata
  lines, and section headings rendered as literal `SECTION N: <heading>`
  text.

This guarantees the three formats never drift out of sync with each other,
and that document IDs/titles/effective dates are always present and
well-formed (a generation-time invariant, checked again at parse time — see
`rag/parsers.py`).

**Why `SECTION N: <heading>` for PDF instead of relying on bold/large
font?** `pypdf`'s text extraction returns a flat text stream and does not
reliably expose font weight/size for structural inference. Rather than
depend on brittle font-based heuristics (or add a heavier layout-aware PDF
library), the corpus generator writes the heading structure directly into
the extracted text as a literal, numbered marker — mirroring how many real
scanned or exported policy PDFs are already numbered in body text. This
keeps the PDF parser simple, robust, and honest about what PDF text
extraction can and cannot recover.

**Objective page count.** Rather than estimate "pages" from a
words-per-page guess (which varies widely with formatting), the generator
includes `generate_corpus.estimate_total_pages()`, which renders *every*
document (regardless of its assigned committed format) through the same
`fpdf2` PDF layout used for the two PDF policies, and sums real page counts.
Running `python -m policies.generate_corpus` prints this figure; at the time
of writing it is **37 pages** for all 12 documents.

## 4. RAG ingestion and retrieval design

### 4.1 Parsing (multi-format)

Each format uses a different, format-appropriate strategy (`rag/parsers.py`):

- **Markdown**: line-oriented regex over `#`/`##` headings and
  `**Label:** value` metadata — fast and sufficient since structure is
  literally encoded in the text syntax.
- **HTML**: `BeautifulSoup` walks real `<h1>`/`<h2>` elements and reads
  `<meta name="...">` tags, rather than scraping visible text for
  metadata-looking strings.
- **PDF**: `pypdf` extracts text; a `SECTION N: <heading>` marker (written
  by the generator, see Section 3) is located with regex, since font-based
  structure is not reliably recoverable from extracted PDF text.

All three converge on the same `ParsedDocument`/`ParsedSection` shape, so
every later stage (cleaning, chunking, embedding) is format-agnostic.
Parsing raises a descriptive `ParseError` if a document is missing its
document ID, title, or effective date — a corpus document without complete
citation metadata fails loudly at ingest time rather than silently
producing uncitable chunks later.

### 4.2 Cleaning

`rag/cleaning.py` applies Unicode NFKC normalization (so visually-identical
characters compare and embed consistently) and collapses redundant
whitespace, without altering intentional single line breaks (which matter
for bullet lists). Cleaning is applied after parsing, once markup has
already been stripped, so it stays deliberately small.

### 4.3 Chunking strategy (justification)

Chunking (`rag/chunking.py`) uses the **policy section** as the primary
chunk boundary, not a fixed character window:

1. Each section in this corpus is already a coherent, self-contained
   topical unit (e.g. "Carryover and Payout Rules") — exactly the
   granularity an HR policy question is usually asked and answered at, and
   the natural unit to cite ("POL-PTO-001, Section 7"). Chunking at
   arbitrary offsets would routinely cut a rule in half.
2. A section is only split further if it exceeds `max_chunk_chars`
   (default 900 characters), so a single embedding vector still represents
   a focused span of text.
3. When a section must be split, the split happens on sentence/bullet
   boundaries only (regex sentence splitting, with bullet lines kept
   atomic), and a configurable overlap (default 150 characters) is carried
   into the next sub-chunk so a fact near a split point isn't orphaned from
   its context.
4. The process is deterministic: the same corpus text always produces
   byte-identical chunks and chunk IDs (`{doc_id}::sec{n}::chunk{m}`) on
   every run — verified by `tests/test_chunking.py`.

Character counts, not token counts, are used as the size unit to avoid a
tokenizer dependency; the constants were chosen empirically against this
corpus's plain English, not against a specific tokenizer's vocabulary. The
141 chunks produced from the 12-document corpus average roughly 380
characters each — most sections fit in a single chunk, and only the longer
sections (e.g. PTO's multi-part rules) split into two or three.

Every chunk retains: `doc_id`, `doc_title`, `section_heading`,
`source_path`, `source_format`, `effective_date`, a stable `chunk_id`, the
chunk `text`, and a `snippet` (the chunk text, truncated to ~220 characters
at a word boundary) for citation display.

### 4.4 Embeddings

Two interchangeable backends implement the same tiny interface
(`rag/embeddings.py`):

- **`fastembed` (production default)** — real local dense sentence
  embeddings (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions) via
  ONNX Runtime. See Section 2 for why this replaces the
  `sentence-transformers` package itself.
- **`hashing` (test/offline fallback)** — a dependency-free (beyond numpy),
  fully deterministic bag-of-words feature-hashing embedding: each
  lowercased word token is MD5-hashed into one of 384 buckets with a
  hash-derived sign, and the resulting vector is L2-normalized. No model
  weights, network access, or GPU are required, and the same text always
  produces the exact same vector on any machine. This is what the default
  `pytest` run uses (see `tests/conftest.py`), so tests are fast, hermetic,
  and reproducible in any CI environment regardless of its network policy.
  It is deliberately **not** the production default — it captures lexical
  overlap, not meaning, so semantically-phrased-but-differently-worded
  queries retrieve worse (see the ablation note in Section 7).

The backend and model actually used to build an index are recorded in that
index's `manifest.json` and re-checked when the index is loaded for
querying (`VectorStore.load(..., expected_backend=...)`), so an index can
never be silently queried with mismatched vectors from a different backend.

### 4.5 Vector store

A small persisted FAISS index (`rag/vector_store.py`):

- `IndexFlatIP` (exact inner-product search over L2-normalized vectors,
  i.e. exact cosine similarity), not an approximate index. At this corpus's
  scale (141 chunks now; the full 30-120-page allowance would still only be
  on the order of a few hundred to low thousands of chunks), exact search
  over 384-dimensional vectors is a few milliseconds and well under a
  megabyte of memory — there is no accuracy/latency tradeoff to make yet,
  and exact search keeps results perfectly deterministic, which matters for
  reproducible evaluation in Phase 2.
- FAISS (rather than a heavier vector database such as Chroma) was chosen
  because it is a single library with no background server process and a
  small dependency footprint (`faiss-cpu` alone, no additional services),
  fitting the "modest free-tier resources" constraint.
- Chunk metadata is stored alongside the FAISS index as a plain JSON
  sidecar (`metadata.json`), keyed by row position, so the index directory
  is human-inspectable without special tooling. `manifest.json` records the
  embedding backend/model/dimension used to build it.
- The index directory (`rag_index/` by default) is a build artifact and is
  **not committed** to the repository (see `.gitignore`); it is built
  locally or in CI/deployment with `python -m rag.ingest`.

### 4.6 Retrieval interface

`rag/retrieval.py` exposes `PolicyRetriever` (and a module-level
`retrieve()` convenience function) as the *only* way the rest of the
application should read from the index. Given a plain-text query and a
`top_k`, it returns a `RetrievalResponse` containing, per result: `doc_id`,
`title`, `section`, `source_path`, a `snippet`, the raw chunk `text`, and a
relevance `score`. A first, retrieval-level guardrail is included:
`has_sufficient_evidence` is `False` when the top result's score falls below
`RAG_MIN_RELEVANCE_SCORE` (default 0.2), so a future agent can decline to
answer from weak evidence rather than fabricating a grounded-sounding
response from a poor match. (Full groundedness/citation checking at the
response level is an agent-orchestrator concern for Phase 2; this is the
retrieval-side half of that guardrail chain.)

## 5. Synthetic structured HR data

`mock_data/*.json` (loaded and cross-validated by `mock_data/loader.py`
into pydantic models in `mock_data/models.py`):

- `departments.json` — 9 departments.
- `locations.json` — 9 work locations (3 Solara offices, 4 U.S. remote
  home-location states, 2 international EOR locations), each flagged
  `payroll_registered` to align with POL-LOC-001's registered-state list.
- `employees.json` — **20 employees** spanning every employment type
  (`full_time`, `part_time`, `contractor`, `intern`, `eor_international`),
  every work arrangement (`onsite`, `hybrid`, `remote`), every status
  (`active`, `on_leave`, `terminated`), a full manager hierarchy rooted at
  one CEO, and multiple departments.
- `pto_balances.json` — 12 records, one per PTO-**accrual**-eligible
  employee (Director+ employees on the Flexible PTO plan intentionally have
  *no* balance record, matching POL-PTO-001 Section 4's "no balance
  tracked" rule — this is itself exercised by
  `test_edge_case_employees_are_present`).
- `benefits.json` — 16 records reflecting POL-BEN-001's eligibility rules
  (e.g. the part-time employee has `benefits_eligible: false` and
  `medical_plan: null`; the newest hire is eligible but still inside the
  30-day waiting period).
- `hr_tickets.json` — 12 mock tickets across PTO, Benefits, Work Location,
  Onboarding, IT Equipment, Payroll, and Leave categories, referencing real
  employee IDs.

### Deterministic demo employees

| Employee | Scenario | Purpose |
|---|---|---|
| `EMP-1001` Jordan Vance | PTO, happy path | Ample balance (48 hrs) for a 3-day (24 hr) request |
| `EMP-1009` Elena Popescu | PTO, edge case | Balance (4 hrs) insufficient for a 3-day request |
| `EMP-1006` Sam Okafor | PTO, edge case | New hire inside the first-30-days restricted window |
| `EMP-1002` Priya Natarajan | Remote work, interstate | Austin, TX -> wants to work from Denver, CO for 6 weeks |
| `EMP-1003` Marcus Whitfield | Remote work, international | Raleigh, NC -> wants to work from Lisbon, Portugal for 6 weeks |

These IDs are intentionally referenced by number in this document and in
`README.md` so Phase 2's agent workflows and evaluation cases can rely on
them without re-deriving which employee fits which scenario. Six weeks
(~30 business days) was chosen deliberately: it exceeds the Remote Work
Policy's 20-business-day no-approval "Work From Anywhere" allowance, so both
demo questions require the agent to reason across *multiple* policies
(POL-RWK-001 -> POL-LOC-001, and for the international case also
POL-SEC-001) rather than answering from a single lookup — see
`policies/content.py`'s POL-LOC-001 "Worked Examples" section, which spells
out the expected outcome for exactly these two cases.

## 6. MCP Architecture (Phase 2)

### 6.1 SDK, transport, and topology

The official Python MCP SDK (`pip install mcp`, version 2.2.0 at the time
of writing) is used for both sides: `mcp.server.mcpserver.MCPServer` for
the server, `mcp.ClientSession` + `mcp.client.stdio` for the client. One
server process is launched per client connection; **stdio** transport was
chosen over HTTP/SSE because:

- It matches `.env.example`'s pre-existing `MCP_TRANSPORT=stdio` default
  from the starter repository.
- It requires no network port, no auth layer, and no separate process
  supervision — the client owns the server's entire lifecycle (it spawns
  the subprocess and tears it down on `close()`), which is the simplest
  possible topology for a single agent talking to a single, co-located
  tool server, and fits "modest free-tier resources" better than running
  an always-on second service.
- It is what the MCP SDK's own `stdio_client`/`StdioServerParameters`
  helpers are built around, so no protocol-framing code had to be written.

The SDK's HTTP/SSE transport (and its `starlette`/`uvicorn` dependencies)
is pulled in by `pip install mcp` but is not used; the design decision is
stdio-only for this project.

### 6.2 A naming collision, and why the code lives in `mcp_server/`/`mcp_client/`

The starter repository already contains an `mcp/` directory (with only a
`README.md`). The official SDK is imported as `import mcp`. Empirically
verifying this in the project's own environment (see the development log
in `ai-tooling.md`) showed that a local top-level directory named `mcp/`
containing **no** `__init__.py` does not shadow the installed SDK package
(Python's import system finds the real, installed regular package first
and stops there). However, that also means such a directory cannot safely
*contain* importable modules either: `mcp.server` is already a real
subpackage of the installed SDK (`mcp.server.mcpserver`, etc.), so a file
at `mcp/server.py` would either be unreachable via `import mcp.server`
(that name already resolves to the SDK) or, if an `__init__.py` were added
to force it to be a regular package, would risk shadowing the SDK outright
depending on sys.path ordering across different invocation contexts
(`python -m`, pytest, a future deployment entrypoint). Rather than depend
on that fragile ordering, the actual server and client code lives in two
distinctly-named top-level packages, `mcp_server/` and `mcp_client/`,
which cannot collide with anything in the `mcp` distribution by
construction. `mcp/README.md` was updated to point here rather than
silently leaving stale instructions in place.

### 6.3 Server/client separation

`mcp_server/` and `mcp_client/` do not import each other. The client
(`mcp_client/client.py`) only ever calls `ClientSession.list_tools()` /
`.call_tool()` — real MCP `tools/list`/`tools/call` requests sent over the
stdio pipe — and never imports anything from `mcp_server.tools`. This was
verified concretely, not just by code review: `tests/test_mcp_server.py`
and `tests/test_mcp_tools.py` spawn a real `python -m mcp_server.server`
subprocess for every test and talk to it only through `MCPClient`, and
`tests/test_mcp_workflows.py`'s two demo workflows are driven entirely
through that same client. If the client ever imported a tool function
directly, removing that import would break nothing at type-check time —
the tests are what actually prove the separation holds at runtime.

### 6.4 Tool schemas

Every tool is a plain, type-hinted Python function registered with
`@server.tool()`; the SDK derives each tool's JSON Schema `input_schema`
from the function signature (via pydantic) automatically — no schema is
hand-written. Every tool *returns* a `pydantic.BaseModel` subclass from
`mcp_server/schemas.py`, never a bare `dict`: empirically, only a typed
return value causes the SDK to populate both `CallToolResult.content`
(human-readable JSON text) and `CallToolResult.structured_content` (a
machine-readable dict) with an accompanying `output_schema`; a bare `dict`
return produced text-only output with no schema when tested directly
against this SDK version. `mcp_client.MCPClient.call_tool()` returns
`structured_content` to its caller for exactly this reason.

| Tool | Key inputs | Returns |
|---|---|---|
| `search_policy_documents` | `query`, `top_k`, `document_id?` | Ranked `PolicySearchHit[]` (doc id/title/section/source path/snippet/score) + `has_sufficient_evidence` |
| `get_policy_section` | `document_id`, `section` | Full section text + citation metadata |
| `lookup_employee_profile` | `employee_id` | Minimized HR-relevant profile (no email/work-authorization) |
| `check_pto_balance` | `employee_id` | Balance + accrual fields, or a `has_tracked_balance=false` explanation |
| `lookup_benefits_status` | `employee_id` | Enrollment/eligibility fields, or a `has_record=false` explanation |
| `check_policy_compliance` | `employee_id`, `scenario_type`, scenario-specific fields | `status`, `reasons`, `citations`, `required_approvals`, `escalation_required`, `facts_considered`, `rule_basis` |
| `draft_hr_email` | `employee_id`, `recipient`, `subject_intent`, `key_facts?`, `requested_action?` | `to`/`subject`/`body`, `is_draft=true` |
| `create_mock_hr_ticket` | `employee_id`, `category`, `summary`, `details?`, `priority?`, `confirmed` | `status` (`preview`/`created`), `ticket_id` (only when created) |

`search_policy_documents` and `get_policy_section` call straight into the
Phase 1 `rag.retrieval`/`rag.parsers` modules; `lookup_employee_profile`,
`check_pto_balance`, and `lookup_benefits_status` call straight into
Phase 1's `mock_data.loader`. None of the Phase 1 logic is duplicated in
`mcp_server/`.

### 6.5 `check_policy_compliance`: facts vs. deterministic rules vs. future LLM interpretation

This tool was, per the assignment's own instruction, designed carefully to
avoid hallucinating policy rules. It draws on exactly three distinct kinds
of input, and its `rule_basis` output field says which combination
produced the result:

1. **Facts retrieved from policy (citations).** `mcp_server/compliance.py`
   runs 2-3 targeted `rag.retrieval` queries per scenario type and attaches
   every resulting chunk (deduplicated) as a `ComplianceCitation`. If *no*
   query for a scenario clears `RagConfig.min_relevance_score`, the tool
   returns `status="insufficient_evidence"` and `rule_basis="insufficient_evidence"`
   **regardless of what the rule table below would have said** — a
   deterministic rule result is only ever surfaced alongside at least one
   supporting citation.
2. **Deterministic business rules.** A fixed table in
   `mcp_server/compliance.py` (`evaluate_pto_request`,
   `evaluate_remote_work_request`), hand-coded from the specific numeric
   thresholds already written into the policy corpus: the PTO accrual/
   balance arithmetic and notice-day tiers (POL-PTO-001), the 20-business-
   day "Work From Anywhere" allowance (POL-RWK-001), the interstate 30/90-
   *calendar*-day approval tiers and payroll-registered-state list
   (POL-LOC-001), and the 30-business-day international EOR allowance and
   approved-country list (POL-LOC-001). Encoding these once means every
   compliance check applies them consistently, and — as of this phase —
   in the exact unit the policy itself states each one in. Earlier
   (Phase 2), the interstate tiers were evaluated against a business-day
   *approximation* of POL-LOC-001's calendar-day thresholds; that
   approximation has been replaced with exact calendar-day evaluation.
   See "Exact calendar-day compliance fix" (Section 6.10) below.
3. **LLM interpretation.** Turning a free-form employee question ("can I
   take three days next week?") into this tool's typed arguments, and
   turning a `ComplianceResult` back into natural-language prose, is the
   agent's job (`agent/orchestrator.py`, via the LLM's tool-calling
   decisions) — this module never sees the user's original wording. This
   tool's input is deliberately a small, fixed set of typed fields per
   `scenario_type` (not a generic free-text or nested-facts blob) so its
   JSON Schema stays simple for tool discovery and so both the Phase 2
   proof scripts and the Phase 3 agent can call it the same way.

### 6.6 Mock action safety

`draft_hr_email` never sends anything — it is a pure function returning a
`to`/`subject`/`body` triple with `is_draft=true`. `create_mock_hr_ticket`
is the one tool with real (if synthetic) side effects, so it gets the
strictest treatment, per the assignment's explicit safety requirement:

- Without `confirmed=true`, the call is a **preview**: no file is read or
  written, and `ticket_id` is `None`. Verified directly by
  `tests/test_mcp_tools.py::test_create_mock_ticket_preview_never_writes_runtime_file`.
- With `confirmed=true`, exactly one record is appended to
  `core.paths.RUNTIME_TICKETS_PATH` (default
  `mock_data/runtime_tickets.json`, gitignored) — never to the committed
  `mock_data/hr_tickets.json` fixture, so running the server (or its
  tests) never mutates committed synthetic data. No real external
  ticketing system is ever contacted.
- Ticket ID generation (`mcp_server/tickets.py`) continues the numbering
  already present in the committed fixture data, so IDs look like a
  natural continuation.

### 6.7 Error-handling strategy

Verified empirically against this SDK version (see `ai-tooling.md`):
raising `mcp.server.mcpserver.exceptions.ToolError("specific message")`
inside a tool produces a clean `CallToolResult(is_error=True, content=[...])`
containing exactly that message — no Python traceback reaches the client
(the traceback is only ever logged server-side, to stderr). This project
draws a deliberate line between two kinds of "failure":

- **The tool cannot do its job** (unknown employee ID, unknown document
  ID, unknown policy section, unknown `scenario_type`, the RAG index not
  having been built yet, an unknown tool name entirely) → raise
  `ToolError` with a specific, actionable message. `mcp_client.MCPClient`
  turns this into an `MCPToolError` for its caller, distinct from a
  transport-level `MCPClientError`.
- **The tool did its job and the honest answer involves an absence** (a
  Director has no tracked PTO balance because they're on the Flexible
  plan; a contractor has no benefits record because they're not eligible;
  a compliance check can't find supporting evidence) → return a normal,
  successful, typed result whose fields explain why. This is not an error
  and must not be raised as one.
- **Malformed arguments** (missing a required field, wrong type) are
  caught automatically by the SDK's pydantic-based schema validation
  before the tool function even runs, producing a clean validation-error
  message — no custom handling was needed for this case.
- **Server startup/connection failure** (bad command, subprocess exits
  immediately, handshake fails) is caught in `MCPClient.connect()` and
  re-raised as `MCPClientError` with the underlying cause attached; the
  client's internal `AsyncExitStack` is always unwound first, so a failed
  connect never leaks a half-started subprocess.

### 6.8 Tool discovery

`mcp_client.MCPClient.list_tools()` calls the real MCP `tools/list`
method and returns the SDK's own `mcp.types.Tool` objects (name,
description, `input_schema`, `output_schema`) — discovery is never
hard-coded on the client side. `tests/test_mcp_server.py::test_tool_discovery_exposes_all_required_tools`
and `::test_tool_schemas_are_valid_json_schema_objects` assert on the
actual discovered set and schemas, not on a fixture.

### 6.9 Expected MCP call sequence for the two demo workflows

Both sequences below are implemented as ordinary (non-agentic) async
functions in `scripts/demo_workflows.py`, each MCP call made through
`MCPClient`, each step recorded as a `core.trace.TraceStep`. They exist to
prove the tool layer *can* support these workflows; a Phase 3 agent will
choose this sequence itself rather than have it hard-coded.

**Workflow A — PTO** (`run_pto_workflow`, e.g. `EMP-1001`, 3 days):
`lookup_employee_profile` → `check_pto_balance` →
`search_policy_documents` → `check_policy_compliance`
(`scenario_type="pto_request"`) → `draft_hr_email`. Verified outcome for
`EMP-1001`: `status="compliant"`, citing POL-PTO-001. For the edge case
`EMP-1009` (insufficient balance): `status="non_compliant"`.

**Workflow B — Remote work** (`run_remote_work_workflow`, e.g. `EMP-1002`
to Colorado, or `EMP-1003` to Portugal, both for six weeks / 30 business
days): `lookup_employee_profile` → `search_policy_documents`
(POL-RWK-001) → `search_policy_documents` (POL-LOC-001) →
`search_policy_documents` (POL-SEC-001) → `check_policy_compliance`
(`scenario_type="remote_work_request"`) → `draft_hr_email`. Verified
outcome for both `EMP-1002` (interstate, Colorado) and `EMP-1003`
(international, Portugal): `status="requires_approval"`, each citing
evidence from **at least two** distinct policy documents (typically
POL-LOC-001 + POL-RWK-001 + POL-SEC-001 together) — the required
multi-document retrieval case. A third scenario (`EMP-1003` to an
unsupported country, e.g. Thailand) returns `status="non_compliant"`.

See `README.md` "Run the two deterministic demo workflows" to reproduce
these locally, and `tests/test_mcp_workflows.py` for the automated
assertions on both.

### 6.10 Exact calendar-day compliance fix (Phase 3)

Phase 2's `check_policy_compliance` accepted a `duration_business_days`
argument for the remote-work scenario and compared it against
`INTERSTATE_RELOCATION_TIER_BUSINESS_DAYS = 65` — an approximation of
POL-LOC-001 Section 4's real threshold, which is stated in **calendar**
days (30/90), not business days. This was flagged as a known
simplification and has now been fixed:

- The tool's input is now `duration_calendar_days` — one authoritative,
  real-world duration (e.g. "six weeks" = 42), matching how a person
  would actually describe a request and avoiding a caller having to do
  unit conversion themselves.
- POL-LOC-001 Section 4's interstate tiers (`INTERSTATE_NO_APPROVAL_CALENDAR_DAYS = 30`,
  `INTERSTATE_PAYROLL_APPROVAL_CALENDAR_DAYS = 90`) are now evaluated
  directly against `duration_calendar_days` — no unit conversion, no
  approximation.
- POL-RWK-001 Section 6's 20-business-day "Work From Anywhere" allowance
  and POL-LOC-001 Section 6's 30-business-day international allowance
  remain evaluated in business days, because *the policy itself* states
  those two thresholds in business days ("Up to 20 business days...";
  "6 consecutive weeks (30 business days)..."). A business-day count is
  now derived from `duration_calendar_days` only for these two checks,
  via `mcp_server.compliance.calendar_to_business_days()`, using a plain
  5-business-days-per-7-calendar-days ratio — which is exactly the ratio
  the policy's own text implies (6 weeks = 42 calendar days = 30 business
  days). No threshold is silently reinterpreted into the wrong unit.
- A genuine edge case this surfaced: a domestic request can be short
  enough to clear POL-LOC-001's 30-calendar-day "no additional approval"
  tier while still exceeding POL-RWK-001's 20-business-day allowance (a
  narrow window around ~28-30 calendar days). Rather than picking one
  policy over the other, the rule now evaluates both literally: POL-LOC-001's
  own ≤30-calendar-day tier governs the *approval outcome* (no additional
  approval required), while the reasons text separately notes when the
  request would need to be routed as a formal Work Location Change
  Request per POL-RWK-001 rather than informal manager notice. See
  `evaluate_remote_work_request`'s docstring and
  `test_interstate_short_stay_exceeding_work_from_anywhere_business_days_still_compliant`
  in `tests/test_compliance_rules.py`.
- 21 new regression tests (`tests/test_compliance_rules.py`) check both
  the international and interstate boundaries directly and exhaustively:
  30 vs. 31 calendar days (compliant -> requires Payroll/P&C approval), 90
  vs. 91 calendar days (Payroll/P&C tier -> relocation/VP tier), and 42
  vs. 43 calendar days for the international business-day allowance (30
  vs. 31 business days, requires-approval -> non-compliant) — plus a test
  that the policy corpus's own "Worked Examples" section (six weeks to
  Colorado -> the 31-90-day tier, "approvable with review") matches the
  code's actual output.
- No policy content was changed to make this fix easier; the fix makes
  the code match the policy, not the other way around.

## 7. Agent, LLM Provider, and Web Application (Phase 3)

### 7.1 LLM provider abstraction

`agent/` depends only on `llm.base.LLMProvider` (`async def chat(messages,
tools) -> LLMResponse`) — a small, OpenAI-tool-calling-shaped interface
(`ChatMessage`, `ToolCall`, `ToolDefinition`, `LLMResponse`), never on a
specific vendor SDK. Two implementations exist:

- **`llm.groq_provider.GroqProvider`** (production default). **Groq** was
  chosen as the primary provider because it has a genuinely free tier with
  fast inference for open-weight, tool-calling-capable models (Llama 3.3
  by default), fitting this project's free-tier-compatibility requirement.
  Implemented via the official `openai` Python SDK pointed at Groq's
  OpenAI-compatible endpoint (`https://api.groq.com/openai/v1`) rather
  than a Groq-specific SDK or a bespoke HTTP client — Groq (and most other
  low-cost inference providers: OpenAI itself, Together, Fireworks, a
  local vLLM/Ollama server) already speak this same API shape, so adding
  another provider later is a thin subclass that changes `base_url` and
  which env vars it reads, not a rewrite. Configuration is entirely via
  environment variables: `GROQ_API_KEY` (required to construct it — raises
  `LLMConfigError` with setup instructions if missing, at construction
  time, before any network call), `GROQ_MODEL` (optional; falls back to a
  documented current default), `LLM_TEMPERATURE` (default `0.2` — low/
  near-deterministic, appropriate for HR policy answers where consistent
  grounding matters more than creative variety), `LLM_TIMEOUT_SECONDS`
  (default `30`). Runtime failures (timeout, auth rejected, rate limit,
  any other API error) are caught and re-raised as a single
  `LLMProviderError` with a specific, clean message — never a raw SDK
  exception.
- **`llm.fake_provider.FakeLLMProvider`** (tests only). A deterministic,
  scripted stand-in: constructed with a list of pre-built `LLMResponse`s,
  it pops and returns the next one on each `chat()` call and records
  exactly what it was sent. No network call, ever. This is what makes it
  possible to *prove* the agent's tool-selection loop is genuinely
  model-driven rather than hard-coded: the test picks which tool the
  "model" decides to call next, and asserts the orchestrator correctly
  acted on that decision — see Section 7.3 and `tests/test_agent_orchestrator.py`.

`llm.config.get_provider()` constructs the configured real provider (only
"groq" is currently supported; requesting anything else raises
`LLMConfigError`) and is never called at import time — importing `llm`,
`agent`, or `app.main` never requires an API key or makes a network call;
only actually handling a `/chat` request (or running
`scripts/agent_smoke_test.py`) does. This is what keeps the automated test
suite free of any Groq dependency.

### 7.2 Agent architecture: a bounded tool-calling loop, not a workflow

**`AgentOrchestrator.run()` (`agent/orchestrator.py`) contains no
`if PTO: call A; call B` branching, for any topic.** On every iteration it
gives the LLM the MCP tool schemas (discovered fresh each `run()` call via
`mcp_client.MCPClient.list_tools()` — real MCP `tools/list`, never
hard-coded) and the conversation so far, and the LLM alone decides whether
to call a tool (which one, with what arguments), ask a clarifying
question, or give a final answer. The orchestrator's job is only to:

1. Discover tools through the real MCP client.
2. Send the conversation + tool schemas to the LLM provider.
3. If the model requests one or more tool calls, execute each via
   `mcp_client.MCPClient.call_tool()` — a genuine MCP `tools/call` request
   over stdio, never a direct import of `mcp_server.tools` — record an
   operational trace step (`core.trace.TraceRecorder`), extract any
   citable evidence (`agent/tool_results.py`), and feed the real result
   back to the model as a `"tool"`-role message.
4. Repeat, up to a hard cap of `DEFAULT_MAX_ITERATIONS = 10` (configurable
   per-`AgentOrchestrator` instance) LLM calls, to guarantee termination
   even if a model gets stuck calling tools indefinitely. Hitting the cap
   produces a graceful, `escalated=True` response rather than hanging or
   erroring.
5. Once the model responds with no further tool calls, parse its final
   text for the two reserved control prefixes (Section 7.4) and return a
   typed `agent.schemas.AgentResponse`.

The Phase 2 deterministic scripts (`scripts/demo_workflows.py`) still
exist and still hard-code their call sequence — they remain useful as a
tool-layer smoke test, a source of realistic fixtures, and a way to
demonstrate the MCP layer independent of any one LLM's behavior — but the
orchestrator never uses or calls into them, and they do not influence how
`AgentOrchestrator` behaves in any way.

**What the LLM receives**, each turn: the system prompt (Section 7.5), the
conversation so far (including prior tool results as `"tool"`-role
messages, each JSON-serialized straight from the MCP call — no Python
object representations or internal details), and the discovered tool
schemas. **What it does not receive**: anything about how a tool is
implemented, database/file paths, or this codebase's module structure.

### 7.3 Tool discovery and selection: proof, not just design

Because "the agent chooses tools dynamically" is a specific, checkable
claim, `tests/test_agent_orchestrator.py` is built to prove it rather than
just exercise it: each test constructs a `FakeLLMProvider` scripted with a
plausible-but-arbitrary sequence of tool-call decisions for its scenario
(a different combination of tools, and a different order, per test — e.g.
`search -> nothing else` for a policy-only question, four tools in
sequence for PTO, five for international remote work), runs it against a
*real* `mcp_server.server` subprocess, and asserts the orchestrator
executed exactly that sequence and produced correctly-shaped output. Since
`AgentOrchestrator` has no knowledge of "PTO" or "remote work" as concepts
anywhere in its code, the only way these tests can pass is if the loop
generically executes whatever the model (real or fake) decides.

### 7.4 Structured output and the CLARIFY:/CONFIRM: convention

`agent.schemas.AgentResponse` (a pydantic model) is the typed contract
returned by every `run()` call and re-exposed by `POST /chat`
(`app.models.ChatResponse`, which adds `conversation_id`):
`answer`, `citations` (`document_id`/`document_title`/`section`/`snippet`/
`source_path`), `snippets`, `trace`, `requires_clarification`,
`requires_confirmation`, `escalated`.

**Citations are never parsed from the model's prose.** They are compiled
by `agent/tool_results.py` directly from the actual MCP tool results seen
during the turn (`search_policy_documents`'s hits,
`get_policy_section`'s content, `check_policy_compliance`'s citations),
deduplicated, and attached to the response independent of what the model
chose to say. This is what "never fabricate citations" means concretely
here: even if the model's prose mentions a document ID incorrectly or not
at all, `AgentResponse.citations` only ever contains evidence a real tool
call actually returned.

**`requires_clarification`/`requires_confirmation`** are derived from two
reserved prefixes the system prompt asks the model to use on its final
answer — `CLARIFY: <question>` and `CONFIRM: <proposed action>` — which
`agent.orchestrator._parse_control_prefix()` recognizes and strips
(case-insensitively) before the text is shown to a user. This is a small,
explicit, testable convention rather than trying to infer intent from
free-form prose after the fact (e.g. guessing whether a sentence "is a
question"), and it is exactly what `FakeLLMProvider`-scripted tests
exercise directly.

**`escalated`** is derived structurally, from the trace (any step whose
`escalation` field is set — currently set when `check_policy_compliance`
returns `escalation_required=true`, when tool discovery or an MCP call
fails at the transport level, or when the iteration cap is hit) — not from
another prefix the model has to remember to add, so it stays reliable even
for cases (an MCP outage, a runaway loop) the model never gets a chance to
comment on.

### 7.5 Prompting strategy

The system prompt (`agent/prompts.py`, `SYSTEM_PROMPT`, versioned via
`SYSTEM_PROMPT_VERSION`) is kept in its own module, in source control,
separate from orchestration logic, so wording changes are reviewable on
their own. It explicitly instructs the model (see the file for exact
wording) to: use policy retrieval when a claim needs policy evidence and
employee-data tools when a claim needs a specific employee's facts (and
both together when a request needs both, e.g. a concrete PTO request);
never fabricate an employee record or a policy; use
`check_policy_compliance` for concrete PTO/remote-work scenarios rather
than trying to do the threshold arithmetic itself from raw policy text;
read and react to tool failures instead of presenting them as success;
ask for a missing employee ID or ask one concise clarifying question
when genuinely ambiguous, rather than guessing; say plainly when the
corpus doesn't contain a policy rather than inventing an answer, and
optionally suggest contacting HR; for a sensitive workplace issue, ground
in the conduct/reporting policy, make clear it isn't a substitute for an
HR investigation, and never create a ticket without explicit confirmation;
stay in scope and redirect politely otherwise; distinguish a retrieved
policy fact from an added suggestion; never send an email (`draft_hr_email`
only ever drafts) and never call `create_mock_hr_ticket` with
`confirmed=true` without the user's explicit prior yes; give a plain-text
answer with no narrated tool use and no visible step-by-step reasoning
(only the two reserved prefixes are special); output operational facts
only through the trace, never through prose.

**Prompt v1.1.0 (production fix):** a live test surfaced that the agent
would check PTO balance and retrieve the PTO policy for a concrete request
like "three days of PTO next week," but stop there and hedge the notice
question back to the user ("if you can meet the 10-business-day notice
requirement") instead of resolving "next week" and calling
`check_policy_compliance`. Two changes fixed this: (1)
`agent/orchestrator.py` now sends a second, dynamically-generated system
message every turn (`_current_date_context()`) stating the real current
date (UTC) — without it the model has no grounded "today" to resolve a
relative phrase against; (2) `SYSTEM_PROMPT` gained explicit rules (4a/4b)
requiring a compliance check whenever a PTO request has enough information
to resolve a timeframe, a documented convention for resolving an
ambiguous relative phrase to its earliest plausible date (a conservative
minimum-notice estimate), and an explicit ban on ever describing a result
as "approved"/"granted" (compliance status is not human sign-off). Rule 3a
was added at the same time for an unrelated but related-in-symptom issue:
the model was observed inventing a `get_policy_section` section name
("Request Procedure") instead of using an exact heading already seen or
returned by a prior tool error's "available sections" list. None of this
is enforced as a hard-coded PTO branch in `agent/orchestrator.py` — it
remains prompt guidance for an LLM that still decides tool sequencing
itself; correctness of the model's own date arithmetic on a live call is
a live-validation concern (see Section 10.2), not something the offline,
scripted test suite can prove — the offline tests (`tests/test_agent_orchestrator.py`,
category B1-B5) prove only that the orchestrator faithfully executes a
compliance check when one is scripted, and that the resulting trace/answer
correctly distinguish balance sufficiency, policy compliance, and (never
claimed) approval.

**Prompt v1.2.0 (production fix):** a live test of the required remote-work
demo workflow (EMP-1002, Colorado, six weeks) showed the model calling only
`check_policy_compliance` and getting a *correct* result (the tool attaches
its own supporting citations internally) — but skipping the richer,
multi-tool evidence-gathering sequence PROJECT_SPEC's demo workflow #2
describes (lookup employee, retrieve remote-work/location policy, check
compliance). New rule 4c asks the model, generically, to establish employee
context and retrieve the actual policy text before calling
`check_policy_compliance` for any concrete remote-work request, and
explains *why* (grounding the final answer in specific retrieved language
and the employee's context, not just the tool's terse rule result) — it
does not restate any numeric threshold, country list, or approval rule;
those stay exclusively in `mcp_server/compliance.py` and the policy corpus,
per this fix's explicit constraint that compliance truth must never be
duplicated into the prompt. `check_policy_compliance`'s own MCP tool
description (`mcp_server/tools.py`, sent verbatim to the model via
`tools/list`) was given the same framing for the same reason: it's read
independently of the system prompt when the model considers this
specific tool. As with the PTO fix, this is prompt/tool-description
guidance only — `agent/orchestrator.py` still contains no
scenario-specific branching, and whether a live model actually follows a
multi-step sequence is a live-validation concern the offline suite cannot
settle; `tests/test_agent_orchestrator.py`'s remote-work tests (categories
C1-C3, D1, E) prove the orchestrator correctly executes whatever
multi-tool sequence is scripted (different employees/states/countries than
the flagship example, to show this isn't special-cased) and that
citations/escalation/compliance status are reported correctly and without
claiming approval.

### 7.6 Guardrails and action-confirmation safety (prompt + code, not prompt alone)

Groundedness and the "never fabricate" rules above are enforced two ways,
deliberately not relying on prompting alone:

- **Retrieval-level**: `rag.retrieval`'s `has_sufficient_evidence` and
  `check_policy_compliance`'s `rule_basis="insufficient_evidence"`
  override (Section 6.5) are structural — the tool itself declines to
  assert a compliance verdict without real supporting evidence,
  independent of anything the model does with that information.
- **Action confirmation**: for `create_mock_hr_ticket`, the system prompt
  instructs the model never to set `confirmed=true` without the user's
  prior explicit agreement — but `AgentOrchestrator.run()` also enforces
  this **in code**, not just via the prompt. It accepts a
  `prior_turn_requires_confirmation: bool` parameter (read by
  `app/main.py` from `app.state.ConversationStore`, which remembers
  whether the *immediately preceding* assistant turn in this conversation
  was itself a `CONFIRM:` offer) and, before sending any call to a tool in
  `_CONFIRMATION_GATED_TOOLS` (currently just `create_mock_hr_ticket`;
  extending this set is how "the same principle should apply to any
  future action-like tool" is meant to be satisfied) with
  `confirmed=true`, checks that flag. If the immediately preceding turn
  was **not** a confirmation offer, the argument is rewritten to
  `confirmed=false` before the call is made — the model literally cannot
  create a ticket in one shot, no matter what it decides, even if asked
  directly ("please file a ticket") in a single message; the conversation
  must pass through an explicit offer-and-confirm exchange first, matching
  the assignment's stated flow exactly ("I can create a mock HR case...
  Would you like me to proceed?" -> explicit yes -> then, and only then,
  `confirmed=true`). See `tests/test_agent_orchestrator.py::test_ticket_creation_without_prior_confirmation_turn_is_downgraded_to_preview`
  and `::test_ticket_creation_after_confirmation_offer_is_allowed`.
- **Sanitized operational trace**: when the LLM provider itself fails
  (timeout, auth, rate limit, any other API error), `AgentOrchestrator.run()`
  logs the full raw exception server-side (`logging.error(...)`, and
  `GroqProvider` logs its own sanitized-but-detailed record beneath that —
  see Section 9.3's logging design) but records only a fixed, generic
  decision string in the `TraceStep` returned to the client
  ("LLM provider error occurred; unable to reach the language model.") —
  never the raw provider exception text. This matters because the trace is
  rendered directly, un-reinterpreted, by the web UI's "Agent Activity"
  panel (`app/static/index.html`'s `renderTrace()`); an earlier version of
  this trace line embedded the raw exception string, which would have
  surfaced upstream API status/error-code detail to end users. See
  `tests/test_llm_provider_logging.py::test_orchestrator_logs_llm_provider_errors_and_keeps_the_graceful_response`.

### 7.7 Conversation state

`app.state.ConversationStore` is an **explicitly ephemeral, in-memory**
store — no Redis, no database, per this phase's constraints. Per
`conversation_id`, it keeps the final user message and final assistant
answer of each turn (not the intermediate tool-call/tool-result chatter
that produced that answer — replaying that into every future turn would
grow unbounded and mostly restates what's already folded into the answer
text) plus whether the last turn was a confirmation offer (Section 7.6).
State is lost on restart/redeploy and is **not** shared across worker
processes if the app is ever run with more than one — both are documented,
accepted limitations at this project's scale (a single-worker free-tier
deployment, demo-length conversations), not oversights.

### 7.8 FastAPI application

`app/main.py`. The MCP client is connected **once**, at app startup, via a
`lifespan` context manager (`app.state.mcp_client`), and closed at
shutdown — not spawned per request, which would mean a new subprocess for
every chat message. Both MCP and LLM unavailability degrade gracefully
rather than crash the app: if the MCP server can't be reached at startup,
`app.state.mcp_client` is `None` and `/health`/`/chat` report it
accurately (`/chat` returns a clean `503`) instead of the app failing to
boot; `GROQ_API_KEY` being unset is likewise a clean `503` from `/chat`
(via the `get_llm_provider` FastAPI dependency, which is also how tests
substitute `FakeLLMProvider` — `app.dependency_overrides[get_llm_provider] = ...`
— without touching real configuration).

Endpoints:

- **`GET /health`** — `{"status": "ok", "mcp": "connected"|"unavailable", "rag": "ok"|"not_built", "llm_configured": bool}`. Cheap by design (checks cached connection state and whether the index directory exists on disk; does not re-verify liveness with a live MCP call or construct an LLM client on every ping) and never includes secret values.
- **`POST /chat`** — `{"message": str, "conversation_id": str | null}` -> `ChatResponse` (`AgentResponse` + `conversation_id`). Runs the full agent loop for one turn.
- **`GET /api/tools`** — the discovered MCP tools and their JSON schemas, for debugging/demo use.

A global exception handler (`@app.exception_handler(Exception)`) is the
last-resort safety net: anything that reaches it is logged server-side
(with the real traceback, for operators) and answered with a generic
`{"detail": "An unexpected error occurred. Please try again."}` at `500`
— no exception message or traceback ever reaches the client. In practice
this should rarely trigger, since `AgentOrchestrator.run()` already
catches every anticipated failure mode (LLM errors, MCP transport errors,
per-tool errors) and returns a normal `AgentResponse` instead of raising.

### 7.9 Web chat UI

A single self-contained page (`app/static/index.html`; no build step, no
frontend framework — vanilla HTML/CSS/JS, per this phase's "avoid a heavy
frontend framework unless clearly necessary") served by FastAPI's
`StaticFiles(html=True)`, mounted at `/` *after* the API routes so they
take precedence. Includes: HR Compass AI branding and a live status strip
(MCP/RAG/LLM, from `/health`, colored dots); a scrollable message history
for the session; a "thinking" loading state while `/chat` is in flight;
each assistant message rendered with its answer text, a citation-chip row
(click to expand the snippet), and a collapsible "Agent Activity" panel
showing that turn's operational trace (tool name, arguments, status,
summary, escalation — never chain-of-thought, since none of that
information exists in the trace to begin with); a visible "Clarification
needed" badge when `requires_clarification` is set; a visible
"Confirmation needed" badge plus quick Yes/No reply buttons when
`requires_confirmation` is set; an "Escalated" badge when applicable; and
a sidebar of the assignment's example prompts (Section 6.9's two demo
workflows plus the other required behaviors) as one-click starters, useful
both for manual testing and for a screen recording. Built with CSS flexbox
and a single narrow-viewport media query so it stays usable at typical
laptop-recording resolutions.

### 7.10 Efficiency, citation precision, latency, and retry (production fix)

A live production trace of the PTO workflow (EMP-1001, "three days of PTO
next week") showed the agent reach the right answer, but only after
roughly 8 MCP tool calls -- several redundant, reworded
`search_policy_documents` calls on the same PTO topic -- and the *final*
Groq call in that turn then failed. Investigating this (see the note on
what could and couldn't actually be determined, below) led to five related
changes, all generic and none hard-coded to PTO or to this example:

- **Reduce redundant searches (prompt, rule 3b in `agent/prompts.py`):**
  `search_policy_documents` already returns `has_sufficient_evidence`;
  the model wasn't being told to act on it. Rule 3b now says: once a
  search comes back sufficient and on-topic, use it (or fetch the exact
  section by name) and move on, rather than re-searching the same topic
  with a reworded query. This is prompt guidance, not an orchestrator
  branch -- whether a live model actually follows it is a live-validation
  concern this offline suite cannot settle, consistent with every other
  prompt-only fix in this section.
- **Exact-repeat-call deduplication (`agent/orchestrator.py`,
  `tool_call_cache`):** a narrow, purely mechanical safety net underneath
  the prompt fix above. If the model calls the exact same tool with the
  exact same arguments twice in one run, the second call is served from an
  in-memory cache instead of hitting the MCP server again (still recorded
  in the trace, marked as a cache hit, `duration_ms=0`). This only catches
  literal repeats, not "slightly reworded" ones -- judging near-duplicate
  relevance is exactly the kind of semantic call this project deliberately
  leaves to the model rather than hard-coding. Never applied to
  `create_mock_hr_ticket` or `draft_hr_email` (`_NON_IDEMPOTENT_TOOLS`):
  a real, if synthetic, side effect or a fresh-prose request must never be
  silently skipped.
- **Citation precision (`agent/tool_results.py`,
  `_filter_relevant_hits`/`CITATION_RELATIVE_SCORE_THRESHOLD`):**
  `search_policy_documents` returns its full top-k regardless of how much
  weaker the 2nd-5th hits are than the 1st, and the agent used to turn
  every one of them into a citation -- so an unrelated equipment- or
  holiday-schedule chunk that merely filled out the remaining top-k slots
  could show up as a citation on a PTO answer. Citations (and, separately,
  the trace's `result_summary`) now only include hits scoring at least
  `CITATION_RELATIVE_SCORE_THRESHOLD` (0.75) of that call's own best
  score -- a *relative*, per-call threshold, not an absolute or
  topic-specific one, so it applies identically to every
  `search_policy_documents` call regardless of subject. The single best
  hit always survives (its ratio to itself is 1.0), so a call that finds
  nothing good still cites its best available match rather than nothing.
  Multi-document citation recall is preserved by construction: this
  project's real multi-document evidence (the required remote-work demo,
  Section 6/9.2) comes from *separate*, `document_id`-scoped calls, each
  of which trivially clears its own bar; two comparably strong hits within
  a single unscoped call also both survive. `get_policy_section` (one
  exact, named section -- no ambiguity) and `check_policy_compliance`
  (citations from `mcp_server.compliance.gather_citations`'s own small set
  of targeted, scenario-specific queries, not a broad top-k dump) are
  unaffected. Proven with hand-constructed, realistically-separated scores
  in `tests/test_tool_results.py`, not against the real hashing-backend
  retriever used elsewhere in this suite -- see that file's docstring for
  why: the hashing backend's topic separation is documented elsewhere in
  this project as intentionally weak (a dependency-free bag-of-words
  fallback), so it isn't a meaningful way to prove this *filtering logic*
  is correct independent of embedding quality.
- **Smaller LLM-facing context (`agent/tool_results.py`,
  `trim_tool_result_for_llm`):** every tool result is JSON-dumped into the
  conversation and resent, in full, on every subsequent LLM turn --
  several tool calls into one run, this resending compounds
  (`O(n^2)` in total tokens sent across a turn's full call sequence, not
  `O(n)`), which is the concrete mechanism by which "~8 tool calls" can
  make the *last* request in that sequence meaningfully larger than the
  first. The copy of a tool result appended to the conversation now: for
  `search_policy_documents`, is narrowed to the same relevance-filtered
  hits used for citations (the model never even sees, and never pays
  token cost for, a hit already judged irrelevant); for both
  `search_policy_documents` and `get_policy_section`, drops `source_path`
  (a filesystem path the model has no use for -- citation source paths
  are captured separately, from the *original* untrimmed result, before
  trimming). The full, untrimmed result is still what
  `summarize_tool_result`, `citations_from_tool_result`, and the
  operational trace use.
- **Bounded retry for transient Groq failures only
  (`llm/groq_provider.py`):** `GroqProvider.chat()` now retries a timeout,
  a bare connection error, or a rate limit (`openai.APIConnectionError` or
  `openai.RateLimitError`, which covers `APITimeoutError` as a subclass)
  up to `DEFAULT_MAX_RETRIES=2` additional times, honoring the API's own
  `Retry-After` header when present and otherwise backing off
  exponentially from `DEFAULT_RETRY_BASE_DELAY_SECONDS=1.0`s. Never
  retried: `AuthenticationError` (a bad key is bad on every attempt) or
  the generic `APIError` catch-all, which is what a permanent 4xx --
  an invalid model name, or a context-length-exceeded request -- surfaces
  as; retrying an unchanged oversized/invalid request fails identically
  every time, so the fix for that case is a smaller/corrected request
  (the citation-filtering and payload-trimming changes above), not a
  retry. Every attempt, successful or not, is timed; a failed attempt's
  sanitized server log line (`_log_provider_error`, already added in an
  earlier fix) now also carries `attempt=N/M` and `duration_ms`, so a
  future occurrence's actual log line will show, definitively, whether it
  hung near the timeout or failed instantly.
- **Latency instrumentation (`core/trace.py`, `TraceStep.duration_ms`):**
  every LLM turn (`agent/orchestrator.py` records one "decision" step per
  `self.llm.chat()` call) and every MCP tool call (`record_tool_call`) now
  carries how long it took, in the client-visible trace -- rendered in the
  web UI's "Agent Activity" panel (`app/static/index.html`) alongside each
  step. A cache-hit repeat call (above) shows `duration_ms=0` explicitly,
  distinguishing "no real round trip occurred" from "a fast one did."

**What could and couldn't be determined about the specific failure:** this
development environment has no access to this project's actual Render
production logs, so the *exact* logged failure category for that one
request (timeout vs. rate limit vs. context-length-exceeded vs. something
else) could not be read and is not claimed here. What the code review
above does establish: (1) the described symptom -- many tool calls
succeeding, then specifically the *last, largest* LLM call in the same
turn failing -- is exactly the signature a growing, repeatedly-resent
conversation produces, since total tokens sent are cumulative and
monotonically increasing across a turn's calls; (2) the two most
plausible categories given that signature are a Groq free-tier
requests/tokens-per-minute rate limit (`rate_limited`) or a
context-length-exceeded 400 (`api_error`), both of which the sanitized
logging already added (see the earlier "log Groq/LLM provider errors"
fix) would show unambiguously via `status_code`/`error_code` on the next
occurrence; (3) every fix above shrinks the accumulated request size or
the number of calls that contribute to it, which mitigates both candidate
causes regardless of which one it actually was. This is stated plainly
rather than implied otherwise, per this project's standing rule against
claiming a verification that did not occur.

### 7.11 Final-answer citation attribution (over-citation fix)

Live evaluation (Section 10.2) showed a distinct citation-precision problem
from the one Section 7.10 already fixed: even with `_filter_relevant_hits`
in place, several remote-work cases (e.g. EVAL-008/9/10/11) returned an
extra document -- typically `POL-SEC-001` -- alongside the two the case
actually expected. The cause was not noisy top-k retrieval within one
`search_policy_documents` call (7.10's fix); it was that `AgentResponse.citations`
was simply every citation accumulated across the *whole turn's* several
tool calls, with no step narrowing that down to what the final answer
actually used. A model researching a remote-work question may reasonably
check more than one angle (interstate, "Work From Anywhere," or an
adjacent policy) across separate calls before realizing only one or two of
them actually matter to its answer -- retrieving that extra evidence is
normal and even encouraged (rule 4c), but carrying all of it into the
final citations regardless of whether the answer used it is not.

The fix keeps "what was retrieved" and "what was cited" as two explicitly
separate sets:

- **Candidate/retrieved evidence** (`collected_citations` in
  `agent/orchestrator.py`): unchanged -- every citation accumulated from
  tool results over the turn, deduplicated as it accumulates. Retrieval
  itself is not narrowed for precision's sake, so recall is never
  sacrificed here.
- **Final, attributed citations** (`agent/tool_results.py`'s new
  `attribute_citations`): computed once, right after the model's final
  answer text is parsed, this narrows the candidate set down to only the
  documents whose exact ID (matched by the project's `PREFIX-CODE-NUMBER`
  convention, e.g. `POL-LOC-001`, via exact set membership -- never
  substring containment) the answer text actually mentions. This can only
  *remove* a candidate, never add one: a document ID the model writes but
  never actually retrieved cannot appear in the result, since attribution
  only selects from already-retrieved candidates, never constructs a new
  `Citation` from parsed text. If the answer mentions none of the
  candidates' IDs at all -- a real case, since a model can give a fully
  correct answer without repeating an exact ID -- attribution
  conservatively falls back to the full candidate set rather than risk
  erasing valid evidence a case's citation recall depends on.

`agent/prompts.py` rule 2a (`SYSTEM_PROMPT_VERSION` 1.3.0 -> 1.4.0)
complements this at the prompting layer: cite only documents that support
a claim actually being made, never a document merely because it was
retrieved, and prefer the minimum sufficient citation set while still
citing every document a genuinely multi-part answer needs. The code-level
fix does not depend on the model following this guidance perfectly --
attribution runs regardless of what the prompt says -- but the two are
meant to point the same direction rather than working against each other.

Proven in `tests/test_tool_results.py`'s `test_attribution_*` tests:
single-document narrowing, a legitimate multi-document answer (both
citations survive, an unrelated third does not), the no-ID-mentioned
fallback, a fabricated-ID-only answer (fallback, and the fake ID never
appears), a mixed real+fabricated answer (only the real one survives), and
deduplication/ordering. `tests/test_agent_orchestrator.py`'s
`test_remote_work_interstate_multi_document_evidence` was updated in the
same change: it had asserted `len({c.document_id for c in result.citations}) >= 2`
purely because the scripted scenario's tool calls touched two documents --
exactly the pre-fix conflation of "retrieved" with "cited" this section
fixes. Its assertion now checks multi-document *retrieval* via each tool
call's own trace `sources` (unaffected by attribution) separately from
final *citations*, which correctly narrow to the one document the scripted
final answer actually names.

## 8. Testing

`pytest` (131 tests total, all passing).

### Phase 1 (38 tests, well under a second, fully offline)

- `test_parsers.py` — unit tests per format (Markdown, HTML) against small
  fixture documents, plus integration tests against the real corpus
  (document count, format diversity, metadata completeness, required-topic
  coverage).
- `test_chunking.py` — size limits, sentence/bullet-boundary splitting,
  overlap behavior, citation-metadata preservation on every chunk,
  determinism across repeated runs, multi-chunk sections.
- `test_vector_store.py` — add/search correctness and ranking, empty-store
  behavior, dimension/length validation, save/load round-trip, and
  actionable errors for a missing or backend-mismatched index.
- `test_retrieval.py` — end-to-end retrieval (ingest -> query) against a
  tiny two-document fixture corpus using the hashing backend: top-k
  citation fields, a single-document-specific query, a genuine
  **multi-document retrieval** case (a query that surfaces chunks from both
  fixture documents), a low-relevance query correctly flagged as
  insufficient evidence, and determinism.
- `test_mock_data.py` — referential integrity of the synthetic dataset, and
  assertions that the deterministic demo employees actually satisfy their
  intended scenario (sufficient/insufficient PTO balance, correct work
  arrangement/location).

### Phase 2 (44 tests, ~45-60s total — each spins a real server subprocess)

A session-scoped fixture (`tests/conftest.py::real_corpus_hashing_index_dir`)
builds one hashing-backend index over the *real* 12-document corpus, reused
by every MCP test below, so the whole suite stays offline and deterministic
while still exercising the actual Solara Dynamics policies (needed for the
workflow tests to make sense). Every test that talks to the server does so
through `mcp_client.MCPClient` connecting to a genuine `python -m
mcp_server.server` subprocess — never by importing `mcp_server.tools`
directly — which is what satisfies the requirement that at least one
automated test prove real MCP tool discovery/calling through the protocol.

- `test_mcp_server.py` — server starts and the client completes the MCP
  handshake; tool discovery returns all 8 required tools; tool schemas are
  valid JSON Schema objects with the expected properties; an unknown tool
  name returns a clean error without crashing the session; client/server
  shutdown is clean and idempotent; a bad server command is handled as a
  clean `MCPClientError` rather than an unhandled exception.
- `test_mcp_client.py` — using the client before `connect()` raises a
  clean, specific error; manual connect/close without the context manager
  works; `connect()` is idempotent; the async context manager still closes
  the subprocess correctly when the `async with` body raises.
- `test_mcp_tools.py` — every one of the 8 tools, called through the
  protocol: `search_policy_documents` (citation fields, a multi-document
  case, `document_id` filtering), `get_policy_section` (valid + unknown
  document + unknown section), `lookup_employee_profile` (valid, minimized
  fields, invalid ID), `check_pto_balance` (balance present, Flexible-plan
  employee with no tracked balance, invalid ID), `lookup_benefits_status`
  (record present, no-record employee, invalid ID),
  `check_policy_compliance` (PTO happy path, PTO insufficient-balance edge
  case, remote-work interstate, remote-work international, unsupported
  country, invalid `scenario_type`), `draft_hr_email` (never sent),
  `create_mock_hr_ticket` (preview never writes the runtime file; confirmed
  creation appends exactly one record with a generated `TCK-` ID; invalid
  employee ID), and malformed arguments (a missing required field) via a
  raw `call_tool` with an empty argument dict.
- `test_mcp_workflows.py` — the two required demo workflows end-to-end
  through `scripts.demo_workflows`: the PTO happy path (`EMP-1001`,
  `status="compliant"`) and edge case (`EMP-1009`, `status="non_compliant"`),
  and the remote-work interstate (`EMP-1002`) and international
  (`EMP-1003`) cases, asserting the exact tool-call sequence, that the
  final result is citation-bearing, and that evidence spans **at least two
  distinct policy documents** for the remote-work cases.
- `test_compliance_rules.py` — 21 unit-level regression tests for the
  calendar-day/business-day thresholds fixed in Section 6.10, run directly
  against `mcp_server.compliance` (no MCP protocol overhead) so every
  boundary can be checked exhaustively and fast.

### Phase 3 (28 tests)

- `test_agent_orchestrator.py` (17 tests) — the required behaviors A-J
  from this phase's instructions, each run against a real MCP server
  subprocess with a `FakeLLMProvider` scripted to make a *different* tool
  (or no-tool) decision per test, proving the loop is genuinely
  model-driven (Section 7.3): policy-only (no employee lookup), the PTO
  workflow (dynamic multi-tool), remote work interstate and international
  (multi-document evidence + escalation), a benefits lookup, an ambiguous
  request (`CLARIFY:`, no tool calls), a missing employee ID, an invalid
  employee ID (the real MCP error is fed back to the model, not
  fabricated), an out-of-corpus question (asserts the actual tool result
  the model received had `has_sufficient_evidence: false`), a sensitive
  workplace issue (grounded + `CONFIRM:`, ticket never created without
  it), the code-enforced confirmation gate in both directions (Section
  7.6), `draft_hr_email` never claiming to have sent anything, the max-
  iteration cap, a hallucinated/unknown tool name handled gracefully, and
  citation propagation.
- `test_api.py` (11 tests) — `/health`'s exact shape and that it never
  includes a secret value; `/api/tools` lists all 8 required tools;
  request validation (empty message -> `422`); `/chat` without
  `GROQ_API_KEY` configured -> clean `503`; the happy path (`FakeLLMProvider`
  injected via `app.dependency_overrides`) returning answer/citations/
  trace; citations shaped correctly; a conversation continuing across two
  `/chat` calls via `conversation_id`; `requires_confirmation` surfacing
  correctly; an `LLMProviderError` inside the agent loop degrading
  gracefully to a `200` rather than crashing (the orchestrator already
  catches it); a genuinely unexpected exception still reaching the client
  only as a clean, generic `500` (no traceback, no exception class name)
  via the global handler; and the root page serving the chat UI with its
  required elements present.

## 9. CI/CD and Deployment (Phase 4)

### 9.1 Architecture diagram

```
Browser
   |
   v
FastAPI (app/main.py)  --serves--> Static chat UI (app/static/index.html)
   |
   v
Agent Orchestrator (agent/orchestrator.py)
   |                                  \
   v                                   v
LLM Provider (llm/)                MCP Client (mcp_client/client.py)
  Groq (prod) / FakeLLMProvider       | stdio (subprocess)
  (tests/offline eval)                v
                                    MCP Server (mcp_server/server.py, 8 tools)
                                       |            |
                                       v            v
                              RAG index         Synthetic HR data
                          (rag/, FAISS +      (mock_data/*.json via
                           fastembed/hashing)   mock_data/loader.py)
                                       |
                                       v
                          Compliance engine (mcp_server/compliance.py)
```

Every arrow above is a real, load-bearing dependency in the actual code
(not aspirational): `app/main.py`'s `lifespan()` constructs one
`MCPClient` and hands it to a fresh `AgentOrchestrator` per `/chat` call;
`AgentOrchestrator` never imports `mcp_server.tools` directly, only calls
through `MCPClient` over the real MCP stdio protocol to the separate
`mcp_server.server` subprocess; that subprocess is the only thing that
touches `rag/` (via `search_policy_documents`/`get_policy_section`),
`mock_data/` (via the employee/PTO/benefits/ticket tools), and
`mcp_server/compliance.py` (via `check_policy_compliance`). The LLM
provider is swappable (Groq in production, `FakeLLMProvider` in every
automated test and in `evaluation/run_offline.py`) without touching
`agent/` at all, since both speak only `llm/base.py`'s `LLMProvider`
interface.

### 9.2 Expected MCP tool sequence for the two required demo workflows

Not hard-coded anywhere (per `agent/orchestrator.py`'s design -- the LLM
alone decides this), but this is the sequence a well-behaved model
produces, and the exact sequence `evaluation/scripts.py` scripts for
`EVAL-006` (PTO) and `EVAL-008` (remote work) in the deterministic
offline evaluation:

**PTO workflow** ("My employee ID is EMP-1001. Can I take three days of
PTO next week?"):
1. `lookup_employee_profile(employee_id="EMP-1001")`
2. `check_pto_balance(employee_id="EMP-1001")`
3. `search_policy_documents(query="PTO request notice and manager approval", document_id="POL-PTO-001")`
4. `check_policy_compliance(employee_id="EMP-1001", scenario_type="pto_request", requested_days=3, notice_business_days=10)`
5. Final grounded answer, citing POL-PTO-001, no ticket/email created.

**Remote-work workflow** ("I'm EMP-1002. Can I work from Colorado for six
weeks?"):
1. `lookup_employee_profile(employee_id="EMP-1002")`
2. `search_policy_documents(query="remote work Work From Anywhere allowance", document_id="POL-RWK-001")`
3. `search_policy_documents(query="interstate work location payroll registered states", document_id="POL-LOC-001")`
4. `check_policy_compliance(employee_id="EMP-1002", scenario_type="remote_work_request", destination_state="Colorado", duration_calendar_days=42)`
5. Final grounded answer citing both POL-RWK-001 and POL-LOC-001 (a real
   multi-document case), reporting that Payroll and People & Culture
   Team approval is required (escalation).

### 9.3 GitHub Actions CI (`.github/workflows/ci.yml`)

Runs on every `push` and `pull_request`. A single `test` job: checkout,
set up Python 3.11, `pip install -r requirements.txt`, an import/build
check (imports the key modules, then runs `rag.ingest` with the hashing
backend to a scratch directory as a build smoke test), the full offline
`pytest` suite (145 tests as of this phase), `scripts/mcp_smoke_test.py`
(explicit MCP discovery + tool-call check, the specific step
`PROJECT_SPEC.md`/`docs/acceptance-checklist.md` call out separately from
the general test suite), `scripts/app_smoke_test.py` (FastAPI app-start
check via `TestClient` entering the real `lifespan()`), `evaluation.run_offline`
(deterministic evaluation smoke test), and `evaluation.ablation` (the
top-k retrieval ablation, also offline). `EMBEDDING_BACKEND=hashing` is
set once at the workflow level so nothing needs network access to
download embedding model weights; no step sets `GROQ_API_KEY` or calls a
real LLM or any other paid/external service.

A second `deploy` job runs only when `test` succeeds, and only for a
`push` to `main` (never for a `pull_request`, which could come from a
fork). It does not deploy anything itself -- it POSTs to a Render "Deploy
Hook" URL stored as the `RENDER_DEPLOY_HOOK_URL` repository secret, which
triggers Render's own build+deploy. Until that secret is set (a one-time,
human setup step -- see the job's comments in `ci.yml`), the job is a
documented no-op rather than a false claim of gating. **This is the
concrete answer to "deployment must only occur if tests pass":** once
Render's own "Auto-Deploy" setting is turned off (also a one-time human
step, since it's a Render dashboard setting, not something in this repo)
and the deploy hook secret is set, the *only* way this service redeploys
is through this `deploy` job, which cannot run unless `test` already
passed. Without that one-time setup, Render's default behavior (deploying
on every push to the connected branch regardless of CI) would still
apply -- this document does not claim gating exists before that setup is
actually done.

### 9.4 Deployment configuration (`render.yaml`)

A single free-tier Render "web" service (`runtime: python`, `plan: free`)
running the whole architecture from section 9.1 in one process:
`buildCommand: pip install -r requirements.txt && python -m rag.ingest`,
`startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT`
(respects Render's dynamically assigned `$PORT`, per its own
requirements), `healthCheckPath: /health`. Environment variables are
declared by name only (`sync: false` for `GROQ_API_KEY`/`GROQ_MODEL`, set
manually in the Render dashboard, never committed); non-secret values
(`LLM_PROVIDER=groq`, `LLM_TEMPERATURE=0.2`, `EMBEDDING_BACKEND=fastembed`,
`MCP_TRANSPORT=stdio`) are set directly in the file since they carry no
sensitive information.

### 9.5 RAG index build strategy at deployment (Option A: build-time)

Of the three strategies CLAUDE.md/the Phase 4 brief named (A: build at
deploy/build time, B: build on first startup if missing, C: commit a
small generated index), **this project uses Option A**: `render.yaml`'s
`buildCommand` runs `python -m rag.ingest` after installing dependencies,
so every deployed instance starts with a complete, persisted FAISS index
already on disk before `uvicorn` ever runs. This was chosen over Option B
because it means the *first* real user request is never the one paying
for index construction (and, with the `fastembed` backend, a model
download) -- `/health`'s `rag.ready` is `true` from the very first request
after a successful build, and a build failure surfaces clearly in
Render's build log rather than as a confusing first-chat-request error.
Option C (committing a pre-built index) was rejected: a FAISS index is a
binary build artifact tied to a specific embedding backend/model/chunking
configuration, and committing it would let the index silently drift out
of sync with the policy corpus or `rag/config.py` defaults -- exactly the
kind of non-reproducible state `.gitignore`'s existing `rag_index/` /
`rag_index_*/` exclusion (from Phase 1) was written to avoid.

The one accepted cost: the production `EMBEDDING_BACKEND=fastembed`
means the *build* step downloads the `sentence-transformers/all-MiniLM-L6-v2`
ONNX weights from Hugging Face the first time (and on any build where
Render's build cache doesn't happen to have them -- Render's build cache
persistence is not guaranteed across deploys, so this may recur on some
deploys). This happens once per build, at build time, never at request
time, and never repeatedly per-request -- there is no code path in this
project that re-downloads or re-embeds anything during request handling.

### 9.6 Health check hardening (`/health`)

`GET /health` (see `app/main.py`, `app/models.py`) returns:

```json
{
  "status": "ok",
  "rag": {"ready": true},
  "mcp": {"ready": true, "tools_discovered": 8},
  "llm": {"configured": false, "provider": "groq"}
}
```

`status` is `"degraded"` (never a `5xx` -- the process is still up and
should not be killed/restarted by Render's health check over this) if RAG
or MCP isn't ready; `llm.configured` is independently reported without
gating `status`, since a missing `GROQ_API_KEY` means `/chat` will 503 but
does not mean the service itself is unhealthy. `mcp.tools_discovered` is
a live count from an actual `tools/list` call made during the health
check (not a cached/assumed number), so it also detects a wedged MCP
subprocess even if it was reachable at startup. Nothing here exposes a
key, secret, or other sensitive environment value -- `llm.provider` is a
provider *name* (`"groq"`), never a credential; this is asserted directly
in `tests/test_api.py::test_health_returns_expected_shape`.

### 9.7 Deployment smoke test (`scripts/deployment_smoke_test.py`)

```
python -m scripts.deployment_smoke_test https://your-app.onrender.com
python -m scripts.deployment_smoke_test https://your-app.onrender.com --test-chat
```

Checks `/`, `/health`, and `/api/tools` against a live URL using only the
standard library (`urllib.request` -- no extra dependency for a one-off
post-deploy check), and optionally `POST /chat`, gated on `--test-chat`
*and* the deployment's own `/health` reporting `llm.configured: true` (so
it never guesses whether a real Groq call is safe to make against that
deployment's quota). Prints a clear PASS/FAIL per check and a non-zero
exit code on any failure.

## 10. Evaluation (Phase 4)

Full methodology, the 25-case evaluation set, metrics definitions, and
documented limitations live in `evaluation/README.md` (not duplicated
here in full to avoid the two documents drifting out of sync). This
section reports what has actually been run and its numeric results, per
CLAUDE.md's "do not claim work or verification that did not occur."

### 10.1 Deterministic (offline) evaluation results

Run via `python -m evaluation.run_offline`, written to
`evaluation/results/offline_20260917T181502Z.json`. **As documented
extensively in `evaluation/README.md`, this is an architecture-verification
run against `FakeLLMProvider`-scripted "model" decisions, not a measurement
of real LLM quality** -- every script in `evaluation/scripts.py` was
written to do the right thing, so a high pass rate here demonstrates the
orchestration/MCP/safety/citation machinery is wired correctly, not that a
live model will behave this well unprompted.

| Metric | Value |
|---|---|
| Cases run | 25 / 25, 0 errors |
| Groundedness (avg) | 0.96 |
| Citation recall (avg) | 1.00 |
| Citation precision (avg) | 0.81 |
| Tool selection accuracy | 1.00 |
| Workflow completion rate | 1.00 |
| Clarification accuracy | 1.00 |
| Escalation accuracy | 1.00 |
| Action-safety pass rate | 1.00 |
| Latency p50 / p95 (offline, near-instant fake model) | 0.007s / 0.026s |

By category (n, groundedness, tool selection, workflow completion,
clarification accuracy, escalation accuracy, action-safety):

| Category | n | Grounded | Tools | Workflow | Clarify | Escalate | Safety |
|---|---|---|---|---|---|---|---|
| A_policy_simple | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| B_policy_multidoc | 1 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| C_employee_data | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| D_pto_workflow | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| E_remote_interstate | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| F_remote_international | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| G_benefits | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| H_expense_compliance | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| I_ambiguous_clarification | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| J_missing_employee_id | 1 | 1.00 | n/a | 1.00 | 1.00 | 1.00 | n/a |
| K_invalid_employee_id | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| L_out_of_scope | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| M_sensitive_escalation | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| N_action_confirmation_safety | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| O_tool_failure | 2 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

The two sub-1.0 groundedness cells are both explained by the documented
keyword-coverage-proxy limitation (evaluation/README.md), not a real
defect: B_policy_multidoc's single case's scripted answer used "paid as a
holiday" rather than the literal rubric phrase "holiday pay," and one
O_tool_failure case's scripted answer said "I don't have a way to..."
rather than the literal word "can't" -- both are correct answers scored
down by exact keyword matching, illustrating exactly why groundedness here
is documented as a proxy, not a factuality judge.

### 10.2 Live (real-model) evaluation results

**Not run.** `evaluation/run_live.py` requires `GROQ_API_KEY`, which is
not available in this development environment (confirmed before writing
this section). Running it, and completing `evaluation/README.md`'s "Live
model validation checklist," is listed as required, not-yet-done work in
section 12 below. Do not read section 10.1's 1.00s as any indication of
how a real model will score -- see 10.1's opening paragraph.

### 10.3 RAG retrieval ablation: top_k=3 vs. top_k=5

Run via `python -m evaluation.ablation`, written to
`evaluation/results/ablation_topk_20260917T181626Z.json`. Uses the 14
evaluation cases that declare `expected_policy_documents` as a
representative query subset, against the real policy corpus (hashing
embedding backend, fully offline/deterministic).

| top_k | Queries | Doc hit rate | Doc recall (avg) | Mean top score | Mean latency | p95 latency |
|---|---|---|---|---|---|---|
| 3 | 14 | 0.79 | 0.71 | 0.357 | 0.06 ms | 0.07 ms |
| 5 | 14 | 0.86 | 0.86 | 0.357 | 0.04 ms | 0.07 ms |

**Interpretation:** widening retrieval from top_k=3 to top_k=5 measurably
improves both doc hit rate (+0.07) and doc recall (+0.14) on this
corpus/query set -- more of the multi-document and less-obviously-worded
queries (e.g. the parental-leave-and-holiday-pay case, which needs
evidence from two different documents) get their second relevant document
into the candidate set only at top_k=5. `mean_top_score` is identical
between the two settings by construction (it only reflects the single
best-ranked result, which top_k does not change). Latency differences at
this corpus size (141 chunks) are negligible in absolute terms
(sub-millisecond either way, on the hashing backend) and should not be
read as representative of latency on a larger corpus or the `fastembed`
backend. **Conclusion: `RAG_DEFAULT_TOP_K=5` (this project's existing
default, set in `rag/config.py` before this ablation was run) is the
better setting for this corpus** -- the recall improvement is worth the
small latency cost, and `rag/config.py`'s default was already correct;
this ablation is retained as evidence for that choice, not a reason to
change it.

### 10.4 Latency methodology

Per-case latency is measured as wall-clock time around each
`AgentOrchestrator.run()` call (`evaluation/evaluator.py:run_case`), from
just before the first LLM turn to the final structured response,
including every MCP tool round-trip in between. p50/p95 use the
nearest-rank method over the sample (`evaluation/metrics.py:latency_stats`
-- documented there as adequate for a ~25-sample run, not a statistically
robust percentile estimator). Offline-mode latency (section 10.1) reflects
only orchestration + real MCP/RAG overhead, since `FakeLLMProvider`
resolves near-instantly; live-mode latency (once run) will additionally
include the real Groq API round-trip per model turn, which will dominate
the total for any case with more than one tool call. No cold-start vs.
warm-request separation has been measured yet, since no live deployment
exists yet to measure against -- `deployed.md` has a placeholder for this,
and `scripts/deployment_smoke_test.py` can be timed manually (e.g. `time
python -m scripts.deployment_smoke_test <url>`) against a real Render
instance immediately after a cold deploy vs. after a warm request, once
one exists.

## 11. Architecture diagram

See section 9.1 above for the text-form diagram (also embedded in
`README.md`'s architecture overview so it's visible without opening this
document).

## 12. Remaining Work

See `docs/requirements-coverage-matrix.md` for the full item-by-item
mapping of every `docs/acceptance-checklist.md` requirement to its
implementation, test/evidence, documentation, and status (complete /
pending-live-validation / pending-deployment) -- this section summarizes
the same conclusion in prose.

Everything in Phase 1-4's code, tests, CI configuration, and deployment
*configuration* is complete. What remains is executing steps that require
resources not available in this development environment, or a human
decision this document should not make unilaterally:

- **Actually deploy to Render.** `render.yaml` is written and validated as
  YAML but has never been applied against a real Render account (none was
  available in this session). Requires: creating the Render service from
  this repo/blueprint, setting `GROQ_API_KEY` in the Render dashboard,
  turning off Render's "Auto-Deploy" toggle, setting the
  `RENDER_DEPLOY_HOOK_URL` GitHub secret (see section 9.3), and then
  filling in `deployed.md`'s placeholders with the real URL, deploy date,
  and commit SHA.
- **Live Groq evaluation.** `python -m evaluation.run_live` and
  `evaluation/README.md`'s "Live model validation checklist" both require
  `GROQ_API_KEY`, which this environment does not have. This is the only
  way to get a real measurement of model quality (section 10.1's numbers
  are architecture verification only, not model quality, by design).
- **A recorded 7-10 minute demo** covering both required workflows
  through the real UI/agent, MCP tool names/arguments/outputs, citations,
  architecture, deployment, and CI/CD -- best recorded against the actual
  deployed URL once it exists, using the two demo prompts in section 9.2
  and README.md's "Example demo prompts."
- **Confirm the Groq model choice** (`GROQ_MODEL`, defaulting to a current
  Llama 3.3 tool-calling model in `llm/groq_provider.py`) against Groq's
  current free-tier model list and rate limits before the recorded demo,
  since this affects both live evaluation quality and its p50/p95 latency.

## 13. Decisions the student/user may want to weigh in on

- **Groq model choice**: see section 12 above.
- **Chat UI**: this project used a minimal FastAPI-served frontend (no
  Streamlit) since it yields a single-service deployment, simpler for a
  free-tier host, per CLAUDE.md's stated preference for that outcome. Now
  that `render.yaml` targets exactly one Render web service, this choice
  is load-bearing for the deployment design as implemented -- worth
  confirming it's still the preferred direction before deploying.
- **`ConversationStore` at scale**: explicitly documented as in-memory,
  single-process, ephemeral (section 7.7) — fine for this project's scope
  and for Render's free tier (a single instance); would need a real shared
  store (e.g. Redis) before any multi-worker/multi-instance deployment.
- **MCP transport for deployment**: stdio (section 6.1) works well when
  the FastAPI process spawns the MCP server as a child process, which is
  the simplest option for a single free-tier web service and is what's
  implemented and deployed via `render.yaml`. If the agent and chat UI
  ever need to run as genuinely separate deployed services, that would
  call for the SDK's HTTP/SSE transport instead.
- **Render's Auto-Deploy setting**: section 9.3's CI gating only works
  once a human turns this off in the Render dashboard and sets the deploy
  hook secret; until then, Render's own default behavior (deploy on every
  push, independent of CI) still applies. This is a one-time manual step
  this repository's code cannot perform on its own.
