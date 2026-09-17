"""End-to-end ingestion pipeline: parse -> clean -> chunk -> embed -> index.

Run with:

    python -m rag.ingest                     # uses RagConfig defaults
    EMBEDDING_BACKEND=hashing python -m rag.ingest   # offline/dev/test build

Deterministic given the same corpus files and embedding backend: parsing,
cleaning, and chunking have no randomness, and the two supported embedding
backends (fastembed, hashing) both produce a fixed vector for a fixed
input text.
"""
from __future__ import annotations

import argparse
import logging
import time

from rag.chunking import chunk_corpus
from rag.config import RagConfig
from rag.embeddings import get_backend
from rag.parsers import parse_corpus
from rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("rag.ingest")


def run_ingest(config: RagConfig | None = None) -> VectorStore:
    config = config or RagConfig()
    start = time.monotonic()

    docs = parse_corpus(config.policies_dir)
    logger.info("Parsed %d documents from %s", len(docs), config.policies_dir)

    chunks = chunk_corpus(docs, config)
    logger.info("Produced %d chunks (max_chunk_chars=%d, overlap=%d)",
                len(chunks), config.max_chunk_chars, config.chunk_overlap_chars)

    backend = get_backend(config.embedding_backend, config.embedding_model, config.hashing_dimension)
    logger.info("Embedding with backend=%s model=%s dim=%d", backend.name,
                getattr(backend, "model_name", "n/a"), backend.dimension)
    vectors = backend.embed([c.text for c in chunks])

    store = VectorStore(dimension=backend.dimension, embedding_backend=backend.name,
                         embedding_model=getattr(backend, "model_name", backend.name))
    store.add(chunks, vectors)
    store.save(config.index_dir)

    elapsed = time.monotonic() - start
    logger.info("Saved index with %d vectors to %s (%.2fs)", len(store), config.index_dir, elapsed)
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["fastembed", "hashing"], default=None,
                         help="Override EMBEDDING_BACKEND for this run.")
    args = parser.parse_args()

    config = RagConfig()
    if args.backend:
        config = RagConfig(**{**config.__dict__, "embedding_backend": args.backend})

    run_ingest(config)


if __name__ == "__main__":
    main()
