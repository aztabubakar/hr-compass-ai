"""Tests for the persisted FAISS-backed vector store."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rag.models import Chunk
from rag.vector_store import VectorStore, VectorStoreError


def _make_chunk(chunk_id: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title=f"Title for {doc_id}",
        section_heading="Test Section",
        source_path=f"/fake/{doc_id}.md",
        source_format="md",
        effective_date="2024-01-01",
        chunk_index=0,
        text=text,
        snippet=text[:50],
    )


def _unit(vector: list[float]) -> np.ndarray:
    arr = np.array(vector, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_add_and_search_returns_nearest_neighbor_first() -> None:
    store = VectorStore(dimension=3, embedding_backend="test", embedding_model="test-model")
    chunks = [
        _make_chunk("a", "DOC-A", "alpha"),
        _make_chunk("b", "DOC-B", "beta"),
        _make_chunk("c", "DOC-C", "gamma"),
    ]
    vectors = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])])
    store.add(chunks, vectors)

    results = store.search(_unit([0.9, 0.1, 0]), top_k=2)

    assert len(results) == 2
    assert results[0][0].chunk_id == "a"  # closest to [1,0,0]
    assert results[0][1] > results[1][1]  # scores sorted descending


def test_search_on_empty_store_returns_empty_list() -> None:
    store = VectorStore(dimension=3, embedding_backend="test", embedding_model="test-model")
    assert store.search(_unit([1, 0, 0]), top_k=5) == []


def test_add_rejects_mismatched_lengths() -> None:
    store = VectorStore(dimension=3, embedding_backend="test", embedding_model="test-model")
    chunks = [_make_chunk("a", "DOC-A", "alpha")]
    vectors = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])])  # 2 vectors, 1 chunk
    with pytest.raises(VectorStoreError):
        store.add(chunks, vectors)


def test_add_rejects_wrong_dimension() -> None:
    store = VectorStore(dimension=3, embedding_backend="test", embedding_model="test-model")
    chunks = [_make_chunk("a", "DOC-A", "alpha")]
    vectors = np.stack([_unit([1, 0])])  # wrong dimension
    with pytest.raises(VectorStoreError):
        store.add(chunks, vectors)


def test_save_and_load_round_trip_preserves_chunks_and_search(tmp_path: Path) -> None:
    store = VectorStore(dimension=3, embedding_backend="hashing", embedding_model="hashing")
    chunks = [
        _make_chunk("a", "DOC-A", "alpha content"),
        _make_chunk("b", "DOC-B", "beta content"),
    ]
    vectors = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])])
    store.add(chunks, vectors)

    index_dir = tmp_path / "index"
    store.save(index_dir)

    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "metadata.json").exists()
    assert (index_dir / "manifest.json").exists()

    loaded = VectorStore.load(index_dir)
    assert len(loaded) == 2
    results = loaded.search(_unit([1, 0, 0]), top_k=1)
    assert results[0][0].chunk_id == "a"
    assert results[0][0].doc_title == "Title for DOC-A"  # metadata survived the round trip


def test_load_missing_index_raises_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(VectorStoreError, match="rag.ingest"):
        VectorStore.load(tmp_path / "does-not-exist")


def test_load_with_mismatched_backend_raises(tmp_path: Path) -> None:
    store = VectorStore(dimension=3, embedding_backend="hashing", embedding_model="hashing")
    store.add([_make_chunk("a", "DOC-A", "alpha")], np.stack([_unit([1, 0, 0])]))
    store.save(tmp_path / "index")

    with pytest.raises(VectorStoreError, match="fastembed"):
        VectorStore.load(tmp_path / "index", expected_backend="fastembed")
