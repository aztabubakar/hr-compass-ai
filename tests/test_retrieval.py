"""End-to-end retrieval tests using the deterministic hashing embedding
backend, so they run offline and without downloading model weights.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import RagConfig
from rag.ingest import run_ingest
from rag.retrieval import PolicyRetriever
from rag.vector_store import VectorStoreError


@pytest.fixture
def built_retriever(hashing_config: RagConfig) -> PolicyRetriever:
    run_ingest(hashing_config)
    return PolicyRetriever(hashing_config)


def test_retriever_raises_actionable_error_before_ingest(hashing_config: RagConfig) -> None:
    with pytest.raises(VectorStoreError):
        PolicyRetriever(hashing_config)


def test_query_returns_top_k_results_with_citation_fields(built_retriever: PolicyRetriever) -> None:
    # Distinctive vocabulary that only appears in the "Requesting Time Off"
    # section, so this is a fair test even for the lexical hashing backend
    # (used here purely for deterministic, offline testing; see
    # rag/embeddings.py for why production uses real semantic embeddings
    # instead).
    response = built_retriever.query("business days advance approval manager respond", top_k=3)

    assert response.query
    assert 1 <= len(response.results) <= 3
    top = response.results[0]
    assert top.doc_id == "TEST-PTO-001"
    assert top.section == "Requesting Time Off"
    assert top.snippet
    assert isinstance(top.score, float)


def test_query_is_document_specific_for_a_distinctive_term(built_retriever: PolicyRetriever) -> None:
    response = built_retriever.query("multi-factor authentication password requirements", top_k=2)
    assert response.results[0].doc_id == "TEST-SEC-001"


def test_multi_document_query_surfaces_more_than_one_document(built_retriever: PolicyRetriever) -> None:
    # A query that touches both fixture documents' topics (time off *and*
    # security incident handling) should surface chunks from both.
    response = built_retriever.query(
        "What are the policies on requesting time off and reporting a security incident?",
        top_k=4,
    )
    doc_ids = {r.doc_id for r in response.results}
    assert doc_ids == {"TEST-PTO-001", "TEST-SEC-001"}


def test_unrelated_query_is_flagged_as_low_confidence(built_retriever: PolicyRetriever) -> None:
    # No token overlap at all with either fixture document -> hashing
    # backend similarity should fall below the configured threshold.
    response = built_retriever.query("xyzzy quokka bicycle nebula", top_k=3)
    assert response.has_sufficient_evidence is False


def test_top_k_is_respected(built_retriever: PolicyRetriever) -> None:
    response = built_retriever.query("policy", top_k=1)
    assert len(response.results) == 1


def test_retrieval_is_deterministic_across_queries(built_retriever: PolicyRetriever) -> None:
    first = built_retriever.query("carryover of unused time off days", top_k=3)
    second = built_retriever.query("carryover of unused time off days", top_k=3)
    assert [r.chunk.chunk_id for r in first.results] == [r.chunk.chunk_id for r in second.results]
    assert [r.score for r in first.results] == [r.score for r in second.results]
