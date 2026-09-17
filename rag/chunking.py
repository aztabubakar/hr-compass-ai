"""Deterministic, citation-preserving chunking.

Chunking strategy (justification):

1. The primary chunk boundary is the policy **section**, not a fixed
   character window. Each section in this corpus is already a coherent,
   self-contained topical unit (e.g. "Carryover and Payout Rules"), which
   is exactly the granularity an HR policy question is usually asked and
   answered at, and it is the natural unit to cite ("POL-PTO-001, Section
   7"). Chunking at arbitrary character offsets would routinely cut a
   rule in half and produce citations that point at a fragment instead of
   a coherent policy statement.
2. A section is only split further if it exceeds `max_chunk_chars`
   (default 900 characters, roughly 150-200 tokens for this corpus' plain
   English), so that a single embedding vector still represents a
   semantically focused span of text and stays comfortably within the
   effective context window of a small local embedding model.
3. When a section must be split, the split happens on sentence/bullet
   boundaries (never mid-sentence), and a small overlap (default 150
   characters) is carried into the next sub-chunk so a fact that happens
   to fall near a split point is not orphaned from its context.
4. Chunking is purely a function of the corpus text and these two size
   constants, with no randomness, so re-running ingestion on unchanged
   input files always produces byte-identical chunks and chunk IDs.

Character counts (not token counts) are used as the size unit so this
module has no tokenizer dependency; `max_chunk_chars`/`chunk_overlap_chars`
in `RagConfig` were chosen empirically against this corpus' English
prose, not against a specific tokenizer's vocabulary.
"""
from __future__ import annotations

import re

from rag.cleaning import clean_text
from rag.config import RagConfig
from rag.models import Chunk, ParsedDocument

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")

SNIPPET_MAX_CHARS = 220


def split_into_units(text: str) -> list[str]:
    """Split section text into sentence- or bullet-level units.

    Bullet lines (starting with "- ") are kept atomic (never merged with
    neighboring prose into one unit) since each bullet is typically an
    independent rule.
    """
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            units.append(line)
        else:
            units.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(line) if s.strip())
    return units


def chunk_section_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split `text` into <= max_chars pieces on sentence/bullet boundaries."""
    if len(text) <= max_chars:
        return [text] if text else []

    units = split_into_units(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        added_len = len(unit) + (1 if current else 0)
        if current and current_len + added_len > max_chars:
            chunks.append(" ".join(current))
            # Carry trailing units (most recent first) into the overlap,
            # up to overlap_chars, so context survives the split.
            overlap: list[str] = []
            overlap_len = 0
            for prior_unit in reversed(current):
                extra = len(prior_unit) + (1 if overlap else 0)
                if overlap_len + extra > overlap_chars:
                    break
                overlap.insert(0, prior_unit)
                overlap_len += extra
            current = overlap
            current_len = overlap_len
            added_len = len(unit) + (1 if current else 0)
        current.append(unit)
        current_len += added_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def _make_snippet(text: str) -> str:
    if len(text) <= SNIPPET_MAX_CHARS:
        return text
    return text[:SNIPPET_MAX_CHARS].rsplit(" ", 1)[0] + "..."


def chunk_document(doc: ParsedDocument, config: RagConfig | None = None) -> list[Chunk]:
    """Produce citation-ready chunks for one parsed document."""
    config = config or RagConfig()
    chunks: list[Chunk] = []
    global_index = 0

    for section in doc.sections:
        cleaned = clean_text(section.text)
        pieces = chunk_section_text(cleaned, config.max_chunk_chars, config.chunk_overlap_chars)
        for sub_index, piece in enumerate(pieces):
            chunk_id = f"{doc.doc_id}::sec{section.order_index}::chunk{sub_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    section_heading=section.heading,
                    source_path=str(doc.source_path),
                    source_format=doc.source_format,
                    effective_date=doc.effective_date,
                    chunk_index=global_index,
                    text=piece,
                    snippet=_make_snippet(piece),
                )
            )
            global_index += 1

    return chunks


def chunk_corpus(docs: list[ParsedDocument], config: RagConfig | None = None) -> list[Chunk]:
    config = config or RagConfig()
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, config))
    return all_chunks
