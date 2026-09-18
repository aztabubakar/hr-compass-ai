"""Shared data structures passed between ingestion pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedSection:
    """One heading + body extracted from a source document."""

    heading: str
    text: str
    order_index: int


@dataclass
class ParsedDocument:
    """A single policy document after format-specific parsing.

    `doc_id`, `title`, and `effective_date` come from the document's own
    metadata (front-matter lines, HTML <meta> tags, or the plain-text
    metadata block in a PDF) rather than being inferred from the filename,
    so citations remain correct even if a file is renamed.
    """

    doc_id: str
    title: str
    effective_date: str
    source_path: Path
    source_format: str  # "md" | "html" | "pdf"
    sections: list[ParsedSection] = field(default_factory=list)


@dataclass
class Chunk:
    """A citation-ready unit of retrievable text."""

    chunk_id: str
    doc_id: str
    doc_title: str
    section_heading: str
    source_path: str
    source_format: str
    effective_date: str
    chunk_index: int
    text: str
    snippet: str


@dataclass
class RetrievalResult:
    """One retrieved chunk plus its relevance score for a specific query."""

    chunk: Chunk
    score: float

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def title(self) -> str:
        return self.chunk.doc_title

    @property
    def section(self) -> str:
        return self.chunk.section_heading

    @property
    def snippet(self) -> str:
        return self.chunk.snippet
