"""Deterministic configuration for the RAG ingestion and retrieval pipeline.

All tunables live here (rather than scattered through the pipeline) and
default to values that are fixed/deterministic and free-tier friendly, per
CLAUDE.md's engineering rules. Values can be overridden by environment
variables (loaded from `.env` via python-dotenv) without changing code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from core.paths import DEFAULT_RAG_INDEX_DIR, POLICIES_DIR

load_dotenv()  # no-op if .env is absent; never overrides already-set env vars


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw) if raw else default


@dataclass(frozen=True)
class RagConfig:
    """Immutable settings for one ingestion/retrieval pipeline instance."""

    policies_dir: Path = _env_path("RAG_POLICIES_DIR", POLICIES_DIR)
    index_dir: Path = _env_path("RAG_INDEX_DIR", DEFAULT_RAG_INDEX_DIR)

    # Chunking. Sizes are in characters, not tokens, to avoid taking a
    # tokenizer dependency in this module (see chunking.py for rationale).
    max_chunk_chars: int = _env_int("RAG_MAX_CHUNK_CHARS", 900)
    chunk_overlap_chars: int = _env_int("RAG_CHUNK_OVERLAP_CHARS", 150)

    # Embeddings. EMBEDDING_BACKEND selects the implementation:
    #   "fastembed" (default): real local dense embeddings via ONNX Runtime.
    #   "hashing": dependency-free deterministic fallback used by the
    #              default test suite so tests run fast, offline, and
    #              without downloading model weights.
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "fastembed")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    hashing_dimension: int = _env_int("HASHING_EMBEDDING_DIM", 384)

    # Retrieval.
    default_top_k: int = _env_int("RAG_DEFAULT_TOP_K", 5)
    # Below this cosine-similarity score, retrieved evidence is flagged as
    # low-confidence so a calling agent can decline to answer from it
    # rather than fabricating a grounded-sounding response. This is a
    # first, retrieval-level guardrail; response-level groundedness
    # checking is implemented by the agent orchestrator in a later phase.
    min_relevance_score: float = _env_float("RAG_MIN_RELEVANCE_SCORE", 0.2)


DEFAULT_CONFIG = RagConfig()
