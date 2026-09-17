"""Tests for chunking and citation metadata preservation."""
from __future__ import annotations

import re
from pathlib import Path

from rag.chunking import chunk_corpus, chunk_document, chunk_section_text
from rag.config import RagConfig
from rag.parsers import parse_document, parse_markdown


def test_chunk_section_text_returns_whole_text_when_short() -> None:
    text = "A short section that fits in one chunk."
    pieces = chunk_section_text(text, max_chars=900, overlap_chars=150)
    assert pieces == [text]


def test_chunk_section_text_splits_long_text_on_sentence_boundaries() -> None:
    sentence = "This is a distinct filler sentence about policy details. "
    long_text = sentence * 40  # well over max_chars
    pieces = chunk_section_text(long_text, max_chars=300, overlap_chars=50)

    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= 300 + 50  # allow one unit beyond the soft cap
        # No piece should start or end mid-sentence.
        assert piece.strip().endswith(".")


def test_chunk_section_text_overlap_shares_content_between_pieces() -> None:
    sentence_template = "Sentence number {i} is here for overlap testing purposes only. "
    long_text = "".join(sentence_template.format(i=i) for i in range(30))
    pieces = chunk_section_text(long_text, max_chars=200, overlap_chars=80)

    assert len(pieces) >= 2
    # The overlap window should mean consecutive chunks share at least one
    # whole sentence rather than picking up exactly where the previous one
    # left off. Compare by sentence number (extracted with a regex) rather
    # than raw substrings, since split(". ") is sensitive to trailing
    # punctuation on the last sentence of each piece.
    first_numbers = set(re.findall(r"number (\d+)", pieces[0]))
    second_numbers = set(re.findall(r"number (\d+)", pieces[1]))
    assert first_numbers & second_numbers


def test_chunk_document_preserves_citation_metadata(tiny_corpus_dir: Path) -> None:
    doc = parse_markdown(tiny_corpus_dir / "test-pto-001.md")
    chunks = chunk_document(doc, RagConfig(max_chunk_chars=900, chunk_overlap_chars=150))

    assert len(chunks) == 2  # one chunk per short section
    for chunk in chunks:
        assert chunk.doc_id == "TEST-PTO-001"
        assert chunk.doc_title == "Sample Time Off Policy"
        assert chunk.effective_date == "2024-01-01"
        assert chunk.source_path.endswith("test-pto-001.md")
        assert chunk.section_heading in {"Requesting Time Off", "Carryover Rules"}
        assert chunk.snippet  # never empty
        assert chunk.text  # never empty

    # Chunk IDs are unique and stable (derived from doc_id/section/chunk index).
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(cid.startswith("TEST-PTO-001::sec") for cid in ids)


def test_chunk_document_is_deterministic_across_runs(tiny_corpus_dir: Path) -> None:
    doc = parse_markdown(tiny_corpus_dir / "test-pto-001.md")
    config = RagConfig(max_chunk_chars=900, chunk_overlap_chars=150)

    chunks_a = chunk_document(doc, config)
    chunks_b = chunk_document(doc, config)

    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert [c.text for c in chunks_a] == [c.text for c in chunks_b]


def test_long_section_is_split_into_multiple_chunks(tmp_path: Path) -> None:
    from tests.conftest import FIXTURE_MARKDOWN_LONG_SECTION

    path = tmp_path / "test-trv-001.md"
    path.write_text(FIXTURE_MARKDOWN_LONG_SECTION, encoding="utf-8")
    doc = parse_document(path)

    chunks = chunk_document(doc, RagConfig(max_chunk_chars=500, chunk_overlap_chars=100))
    booking_chunks = [c for c in chunks if c.section_heading == "Booking Travel"]

    assert len(booking_chunks) > 1
    # All sub-chunks of the same section still carry identical doc-level metadata.
    assert len({c.doc_id for c in booking_chunks}) == 1
    assert len({c.section_heading for c in booking_chunks}) == 1


def test_chunk_corpus_aggregates_all_documents(tiny_corpus_dir: Path) -> None:
    from rag.parsers import parse_corpus

    docs = parse_corpus(tiny_corpus_dir)
    chunks = chunk_corpus(docs, RagConfig())

    doc_ids_in_chunks = {c.doc_id for c in chunks}
    assert doc_ids_in_chunks == {"TEST-PTO-001", "TEST-SEC-001"}
