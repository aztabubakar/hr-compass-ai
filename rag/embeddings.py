"""Pluggable embedding backends.

Two implementations share the same tiny interface (`embed`, `dimension`,
`name`) so the vector store and retrieval code never need to know which
one produced a given vector — the backend name and dimension are recorded
in the persisted index manifest (see vector_store.py) and re-checked at
query time so an index built with one backend can't silently be queried
with mismatched vectors from another.

- `FastEmbedBackend` (production default): real local dense sentence
  embeddings computed by `fastembed`, which runs the model through ONNX
  Runtime. This is the "zero-cost/local embedding model" called for by
  the project spec. We use `fastembed` instead of the `sentence-transformers`
  package suggested in CLAUDE.md's preferred stack because, in this
  project's build environment, pip's default `torch` wheel resolution
  pulled the full CUDA/nvidia dependency chain (observed multi-gigabyte
  install) with no way to force a CPU-only wheel from PyPI alone.
  `fastembed` serves the same model named in `.env.example`
  (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions) as a small
  int8-quantized ONNX export, with a CPU-only footprint of a few hundred
  megabytes total — a better fit for a free-tier host. This is a
  documented deviation from the suggested stack, made for free-tier
  reliability as CLAUDE.md's engineering rules explicitly permit.
- `HashingEmbeddingBackend`: a dependency-free (beyond numpy), fully
  deterministic bag-of-words feature-hashing embedding. It does not need
  model weights, network access, or a GPU, so it is used by the default
  automated test suite (fast, hermetic, reproducible in CI) and is
  available as an explicit fallback (`EMBEDDING_BACKEND=hashing`) if a
  deployment environment cannot reach the model host. It is deliberately
  *not* the production default: it captures lexical overlap, not
  semantic meaning, so retrieval quality is lower than a real embedding
  model.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    name: str
    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), dimension) float32 array of unit vectors."""
        ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbeddingBackend:
    """Deterministic feature-hashing embedding (no model weights needed).

    Each lowercased word token is hashed (MD5, a stable hash across
    processes and Python versions, unlike the builtin `hash()`) into a
    bucket in `[0, dimension)`, with a second hash bit choosing +1/-1 so
    opposite tokens don't only ever add constructively. The resulting
    bag-of-words vector is L2-normalized. There is no randomness anywhere
    in this process, so the same text always produces the exact same
    vector on any machine.
    """

    name = "hashing"

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed_one(t) for t in texts]) if texts else np.zeros((0, self.dimension), dtype=np.float32)


class FastEmbedBackend:
    """Local ONNX-based dense embeddings via the `fastembed` package."""

    name = "fastembed"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only when fastembed is missing
            raise RuntimeError(
                "fastembed is not installed. Install it (see requirements.txt) or set "
                "EMBEDDING_BACKEND=hashing to use the offline fallback backend."
            ) from exc

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        # fastembed does not expose dimension as a plain attribute across
        # versions; derive it once from a throwaway embedding instead of
        # hardcoding a number that could silently drift from the model.
        self.dimension = len(next(self._model.embed(["dimension probe"])))

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def get_backend(backend_name: str, model_name: str, hashing_dimension: int = 384) -> EmbeddingBackend:
    if backend_name == "hashing":
        return HashingEmbeddingBackend(dimension=hashing_dimension)
    if backend_name == "fastembed":
        return FastEmbedBackend(model_name=model_name)
    raise ValueError(f"Unknown embedding backend {backend_name!r}; expected 'fastembed' or 'hashing'")
