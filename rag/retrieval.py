"""Top-k policy retrieval interface.

This is intentionally the *only* way the rest of the application (a
future MCP `search_policy_documents` tool, evaluation scripts, etc.)
should read from the vector index — callers ask a question in plain text
and get back ranked, citation-ready evidence; they never touch FAISS or
the embedding backend directly.
"""
from __future__ import annotations

from dataclasses import dataclass

from rag.config import RagConfig
from rag.embeddings import EmbeddingBackend, get_backend
from rag.models import RetrievalResult
from rag.vector_store import VectorStore


@dataclass
class RetrievalResponse:
    query: str
    results: list[RetrievalResult]
    has_sufficient_evidence: bool


class PolicyRetriever:
    """Loads a persisted vector index and answers top-k queries against it."""

    def __init__(self, config: RagConfig | None = None) -> None:
        self.config = config or RagConfig()
        self._store: VectorStore = VectorStore.load(self.config.index_dir)
        # Query with whatever backend/model actually built the index (recorded
        # in its manifest), not necessarily the current environment's default,
        # so retrieval keeps working correctly even if EMBEDDING_BACKEND is
        # later changed without re-running ingestion.
        self._backend: EmbeddingBackend = get_backend(
            self._store.embedding_backend, self._store.embedding_model, self.config.hashing_dimension
        )

    def query(self, text: str, top_k: int | None = None) -> RetrievalResponse:
        top_k = top_k or self.config.default_top_k
        query_vector = self._backend.embed([text])[0]
        raw = self._store.search(query_vector, top_k)
        results = [RetrievalResult(chunk=chunk, score=score) for chunk, score in raw]
        has_sufficient_evidence = bool(results) and results[0].score >= self.config.min_relevance_score
        return RetrievalResponse(query=text, results=results, has_sufficient_evidence=has_sufficient_evidence)

    def __len__(self) -> int:
        return len(self._store)


_default_retriever: PolicyRetriever | None = None


def get_default_retriever() -> PolicyRetriever:
    """Lazily construct a process-wide singleton retriever.

    Convenience for simple callers (a CLI, a quick script, or an MCP tool
    handler) that just want to call `retrieve("...")` without managing a
    `PolicyRetriever` instance themselves.
    """
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = PolicyRetriever()
    return _default_retriever


def retrieve(query: str, top_k: int | None = None) -> RetrievalResponse:
    return get_default_retriever().query(query, top_k)
