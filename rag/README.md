# RAG

**Status: implemented (Phase 1).** Multi-format parsing (Markdown/HTML/PDF),
cleaning, deterministic section-based chunking, pluggable embeddings
(`fastembed` production default, deterministic `hashing` test/offline
fallback), a persisted FAISS vector store, and a top-k retrieval interface
with a first relevance guardrail (`has_sufficient_evidence`).

- `config.py` — deterministic, env-overridable settings.
- `models.py` — `ParsedDocument`/`Chunk`/`RetrievalResult` shapes.
- `parsers.py`, `cleaning.py`, `chunking.py`, `embeddings.py`, `vector_store.py` — pipeline stages.
- `ingest.py` — `python -m rag.ingest` builds the index.
- `retrieval.py` — `retrieve(query, top_k)` / `PolicyRetriever` is the intended entry point for callers (e.g. a future MCP tool).

See `design-and-evaluation.md` Section 4 for full design rationale. Not yet
implemented: MCP tool wrapping, agent-level groundedness/citation checking
(Phase 2).
