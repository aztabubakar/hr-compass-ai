"""Tests for format-specific policy document parsing."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.paths import POLICIES_DIR
from rag.parsers import ParseError, parse_corpus, parse_document, parse_html, parse_markdown


def test_parse_markdown_extracts_metadata_and_sections(tiny_corpus_dir: Path) -> None:
    doc = parse_markdown(tiny_corpus_dir / "test-pto-001.md")

    assert doc.doc_id == "TEST-PTO-001"
    assert doc.title == "Sample Time Off Policy"
    assert doc.effective_date == "2024-01-01"
    assert doc.source_format == "md"
    assert [s.heading for s in doc.sections] == ["Requesting Time Off", "Carryover Rules"]
    assert "5 business days" in doc.sections[0].text
    assert "December 31st" in doc.sections[1].text


def test_parse_html_extracts_metadata_and_sections(tiny_corpus_dir: Path) -> None:
    doc = parse_html(tiny_corpus_dir / "test-sec-001.html")

    assert doc.doc_id == "TEST-SEC-001"
    assert doc.title == "Sample Security Policy"
    assert doc.effective_date == "2024-02-01"
    assert doc.source_format == "html"
    assert [s.heading for s in doc.sections] == ["Password Requirements", "Incident Reporting"]
    assert "14 characters" in doc.sections[0].text
    assert "24 hours" in doc.sections[1].text


def test_parse_markdown_missing_metadata_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.md"
    bad.write_text("# A Policy With No Metadata\n\n## Section One\n\nSome text.\n", encoding="utf-8")
    with pytest.raises(ParseError):
        parse_markdown(bad)


def test_parse_document_dispatches_by_extension(tiny_corpus_dir: Path) -> None:
    md_doc = parse_document(tiny_corpus_dir / "test-pto-001.md")
    html_doc = parse_document(tiny_corpus_dir / "test-sec-001.html")
    assert md_doc.source_format == "md"
    assert html_doc.source_format == "html"


def test_parse_document_rejects_unsupported_extension(tmp_path: Path) -> None:
    unsupported = tmp_path / "policy.docx"
    unsupported.write_text("irrelevant", encoding="utf-8")
    with pytest.raises(ParseError):
        parse_document(unsupported)


# ---------------------------------------------------------------------------
# Integration tests against the real, generated policy corpus.
# ---------------------------------------------------------------------------
def test_real_corpus_parses_without_errors() -> None:
    docs = parse_corpus(POLICIES_DIR)
    assert 10 <= len(docs) <= 12  # corpus target: 10-12 documents


def test_real_corpus_uses_at_least_two_formats() -> None:
    docs = parse_corpus(POLICIES_DIR)
    formats = {d.source_format for d in docs}
    assert len(formats) >= 2
    assert formats == {"md", "html", "pdf"}


def test_real_corpus_documents_have_stable_metadata() -> None:
    docs = parse_corpus(POLICIES_DIR)
    doc_ids = [d.doc_id for d in docs]

    # Stable, unique document IDs.
    assert len(doc_ids) == len(set(doc_ids))
    for doc_id in doc_ids:
        assert doc_id.startswith("POL-")

    for doc in docs:
        assert doc.title, f"{doc.doc_id} missing a title"
        assert doc.effective_date, f"{doc.doc_id} missing an effective date"
        # ISO 8601 date, e.g. 2024-01-01.
        assert len(doc.effective_date) == 10 and doc.effective_date[4] == "-"
        assert len(doc.sections) >= 5, f"{doc.doc_id} has suspiciously few sections"


def test_real_corpus_required_topics_are_covered() -> None:
    docs = parse_corpus(POLICIES_DIR)
    doc_ids = {d.doc_id for d in docs}
    required_topics = {
        "POL-PTO-001",  # PTO
        "POL-HOL-001",  # holidays
        "POL-RWK-001",  # remote work
        "POL-LOC-001",  # work location / interstate / international
        "POL-SEC-001",  # information security
        "POL-EXP-001",  # expenses and reimbursement
        "POL-BEN-001",  # employee benefits
        "POL-ONB-001",  # onboarding
        "POL-EQP-001",  # company equipment
        "POL-LEV-001",  # leave
        "POL-CON-001",  # workplace conduct
        "POL-TRV-001",  # travel
    }
    assert required_topics <= doc_ids
