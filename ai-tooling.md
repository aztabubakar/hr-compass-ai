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
