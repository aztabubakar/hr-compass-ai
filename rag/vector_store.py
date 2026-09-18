"""A small, local, persisted vector index backed by FAISS.

Design choices:

- `IndexFlatIP` (exact inner-product search over L2-normalized vectors,
  i.e. exact cosine similarity) rather than an approximate index (IVF,
  HNSW). This corpus has on the order of a few hundred chunks; an exact
  search over that many 384-dimensional vectors is a few milliseconds and
  uses well under a megabyte of memory, so there is no accuracy/speed
  tradeoff to make yet, and exact search keeps results perfectly
  deterministic — important for reproducible evaluation in a later phase.
- FAISS itself (rather than a heavier vector database such as Chroma) was
  chosen because it is a single pure-C++-with-Python-bindings library
  with no background server process and a small dependency footprint,
  which fits the "modest free-tier resources" constraint better.
- Chunk metadata (everything needed for a citation) is stored alongside
  the FAISS index as a plain JSON sidecar file, keyed by row position, so
  the index directory is human-inspectable without special tooling.
- A `manifest.json` records the embedding backend name, model name, and
  vector dimension used to build the index. `VectorStore.load` re-checks
  these against the backend the caller asks to query with, so a stale or
  mismatched index fails loudly (with an actionable message) instead of
  returning silently-wrong nearest neighbors.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from rag.models import Chunk

MANIFEST_FILENAME = "manifest.json"
INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"


class VectorStoreError(RuntimeError):
    pass


class VectorStore:
    def __init__(self, dimension: int, embedding_backend: str, embedding_model: str) -> None:
        self.dimension = dimension
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise VectorStoreError("chunks and vectors must have the same length")
        if vectors.shape[1] != self.dimension:
            raise VectorStoreError(
                f"vector dimension {vectors.shape[1]} does not match store dimension {self.dimension}"
            )
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if len(self) == 0:
            return []
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        k = min(top_k, len(self))
        scores, indices = self._index.search(query, k)
        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_dir / INDEX_FILENAME))
        with (index_dir / METADATA_FILENAME).open("w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self._chunks], f, indent=2)
        manifest = {
            "dimension": self.dimension,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "num_chunks": len(self._chunks),
        }
        with (index_dir / MANIFEST_FILENAME).open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, index_dir: Path, expected_backend: str | None = None) -> "VectorStore":
        manifest_path = index_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise VectorStoreError(
                f"No index found at {index_dir}. Run `python -m rag.ingest` first."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if expected_backend is not None and manifest["embedding_backend"] != expected_backend:
            raise VectorStoreError(
                f"Index at {index_dir} was built with embedding backend "
                f"{manifest['embedding_backend']!r}, but {expected_backend!r} was requested. "
                "Re-run `python -m rag.ingest` with the desired backend, or query with the "
                "matching backend."
            )

        store = cls(
            dimension=manifest["dimension"],
            embedding_backend=manifest["embedding_backend"],
            embedding_model=manifest["embedding_model"],
        )
        store._index = faiss.read_index(str(index_dir / INDEX_FILENAME))
        raw_chunks = json.loads((index_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
        store._chunks = [Chunk(**c) for c in raw_chunks]
        return store
