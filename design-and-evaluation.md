# Design and Evaluation

This document tracks architecture decisions, design rationale, and (in
later phases) evaluation results for HR Compass AI. It is updated as each
phase is implemented. **Current status: Phase 1 (project foundation + RAG
foundation) and Phase 2 (MCP server/client tool layer) are complete.**
Phase 3 (the LLM agent orchestrator, chat UI, CI/CD, deployment,
evaluation) has not started; see "Remaining Work" at the end.

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
scripts/     Deterministic tool-level demo workflows + smoke test (Phase 2)
mcp/         Documentation pointer only; see Section 6 for why
agent/       (Phase 3) LLM agent orchestrator
evaluation/  (Phase 3) evaluation harness and cases
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
compatible versions once the implementation is selected. Only the
dependencies actually used in this phase are installed; a commented section
lists what Phase 2 will add (FastAPI, an MCP SDK, a chat UI layer) so the
overall intended stack is visible without installing unused packages now.

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
   day approval tiers and payroll-registered-state list (POL-LOC-001), and
   the 30-business-day international EOR allowance and approved-country
   list (POL-LOC-001). Encoding these once means every compliance check
   applies them consistently. One documented simplification: the tool's
   `duration_business_days` input is compared against a business-day
   approximation of POL-LOC-001's calendar-day interstate tiers (see the
   `INTERSTATE_RELOCATION_TIER_BUSINESS_DAYS` constant and its comment) —
   reconciling calendar vs. business days precisely is left to Phase 3.
3. **Future LLM interpretation (not built yet).** Turning a free-form
   employee question ("can I take three days next week?") into this
   tool's typed arguments, and turning a `ComplianceResult` back into
   natural-language prose, is explicitly the Phase 3 agent's job. This
   tool's input is deliberately a small, fixed set of typed fields per
   `scenario_type` (not a generic free-text or nested-facts blob) so its
   JSON Schema stays simple for tool discovery and so today's Phase 2
   proof scripts can call it directly and deterministically.

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

## 7. Testing

`pytest` (82 tests total, all passing).

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

## 8. Remaining Work (Phase 3+)

Not yet implemented, per `PROJECT_SPEC.md` and `docs/acceptance-checklist.md`:

- Agent orchestrator: intent routing (RAG-only vs. tool-required vs.
  multi-step workflow vs. ambiguous vs. out-of-scope), tool selection via
  the now-existing MCP client, response synthesis, and a concise
  operational trace (tool name, arguments, result summary, retrieved
  sources, escalation/confirmation decisions) with no exposed
  chain-of-thought — `core.trace.TraceRecorder` already exists and is used
  by Phase 2's proof scripts; the agent will use the same model instead of
  a hard-coded call sequence.
- The two end-to-end demo workflows (PTO, remote work) wired through the
  agent and MCP tools, using the demo employees above.
- Web chat UI with `/chat` and `/health` endpoints.
- GitHub Actions CI (install, import/build check, tests, MCP
  discovery/tool-call test) gating deployment; free-tier deployment
  (Render or equivalent).
- Evaluation harness: 20-30 cases with gold answers/rubrics; groundedness,
  citation accuracy, tool-selection accuracy, workflow completion,
  clarification/escalation accuracy, action-safety pass rate, p50/p95
  latency; at least one ablation. A natural first ablation, given Phase 1's
  design, is **hashing vs. fastembed retrieval quality** on the same query
  set, since both backends already share one interface end-to-end.
- `deployed.md` filled in once deployed.

## 9. Decisions the student/user may want to weigh in on for Phase 3

- **LLM provider**: `.env.example` leaves `LLM_PROVIDER`/`LLM_MODEL` blank.
  A free-tier-friendly option should be chosen before the agent
  orchestrator is built (this affects cost and latency evaluation).
- **Chat UI**: CLAUDE.md allows either a Streamlit UI or a minimal
  FastAPI-served frontend; the latter yields a single-service deployment,
  which is simpler for a free-tier host.
- **MCP transport for deployment**: stdio (Section 6.1) works well when
  the agent process spawns the MCP server as a child process, which is the
  simplest option for a single free-tier web service. If the agent and
  chat UI ever need to run as genuinely separate deployed services, that
  would call for switching to the SDK's HTTP/SSE transport instead —
  worth confirming once the Phase 3 deployment topology is chosen.
- **`check_policy_compliance`'s calendar/business-day approximation**
  (Section 6.5, point 2): acceptable as a documented simplification for
  Phase 2's proof scripts, but worth deciding whether Phase 3's agent
  should compute exact calendar-day thresholds from an actual date input
  instead of relying on a caller-supplied `duration_business_days`.
