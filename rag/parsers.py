"""Format-specific parsers for the policy corpus.

Each source format uses a genuinely different extraction strategy rather
than one regex reused everywhere, because each format preserves structure
differently:

- Markdown: heading and metadata syntax (``#``/``##``, ``**Label:**``) is
  literally in the text, so a line-oriented regex parser is sufficient and
  fastest.
- HTML: structure lives in real tags, so we use BeautifulSoup to walk
  ``<h1>``/``<h2>`` elements and their sibling content, and read metadata
  from ``<meta name="...">`` tags rather than scraping visible text.
- PDF: `pypdf` text extraction discards font styling (bold/size), so
  heading structure cannot be recovered from formatting. Instead, the
  corpus generator writes an explicit ``SECTION N: <heading>`` marker as
  literal text for each heading (mirroring how many real-world policy
  PDFs number their sections), and this parser looks for that marker.

All three parsers converge on the same `ParsedDocument`/`ParsedSection`
shape so the rest of the pipeline (cleaning, chunking, embedding) never
needs to know which format a document came from.
"""
from __future__ import annotations

import re
from pathlib import Path

import pypdf
from bs4 import BeautifulSoup

from rag.models import ParsedDocument, ParsedSection

SUPPORTED_EXTENSIONS = {".md": "md", ".html": "html", ".pdf": "pdf"}


class ParseError(ValueError):
    """Raised when a source document is missing required metadata."""


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
_MD_TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_MD_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_MD_META_RE = re.compile(r"^\*\*([A-Za-z ]+):\*\*\s*(.+?)\s*$", re.MULTILINE)


def parse_markdown(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8")

    title_match = _MD_TITLE_RE.search(text)
    if not title_match:
        raise ParseError(f"{path}: no top-level '# Title' heading found")
    title = title_match.group(1).strip()

    metadata = {label.strip().lower(): value.strip() for label, value in _MD_META_RE.findall(text)}
    doc_id = metadata.get("document id")
    effective_date = metadata.get("effective date")
    if not doc_id or not effective_date:
        raise ParseError(f"{path}: missing Document ID and/or Effective Date metadata")

    headings = list(_MD_HEADING_RE.finditer(text))
    sections: list[ParsedSection] = []
    for index, match in enumerate(headings):
        heading = match.group(1).strip()
        start = match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = text[start:end].strip()
        sections.append(ParsedSection(heading=heading, text=body, order_index=index))

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        effective_date=effective_date,
        source_path=path,
        source_format="md",
        sections=sections,
    )


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def parse_html(path: Path) -> ParsedDocument:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")

    def meta(name: str) -> str | None:
        tag = soup.find("meta", attrs={"name": name})
        return tag.get("content") if tag else None

    doc_id = meta("doc-id")
    effective_date = meta("effective-date")
    if not doc_id or not effective_date:
        raise ParseError(f"{path}: missing <meta name='doc-id'/'effective-date'> tags")

    h1 = soup.find("h1")
    if not h1:
        raise ParseError(f"{path}: no <h1> title found")
    title = h1.get_text(strip=True)

    sections: list[ParsedSection] = []
    for index, h2 in enumerate(soup.find_all("h2")):
        heading = h2.get_text(strip=True)
        body_parts: list[str] = []
        for sibling in h2.find_next_siblings():
            if sibling.name == "h2":
                break
            body_parts.append(sibling.get_text(" ", strip=True))
        sections.append(
            ParsedSection(heading=heading, text="\n\n".join(p for p in body_parts if p), order_index=index)
        )

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        effective_date=effective_date,
        source_path=path,
        source_format="html",
        sections=sections,
    )


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
_PDF_SECTION_RE = re.compile(r"^SECTION\s+\d+:\s*(.+)$", re.MULTILINE)
_PDF_META_RE = re.compile(r"^(Document ID|Effective Date|Version|Owner):\s*(.+?)\s*$", re.MULTILINE)


def parse_pdf(path: Path) -> ParsedDocument:
    reader = pypdf.PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    metadata = {label.strip().lower(): value.strip() for label, value in _PDF_META_RE.findall(text)}
    doc_id = metadata.get("document id")
    effective_date = metadata.get("effective date")
    if not doc_id or not effective_date:
        raise ParseError(f"{path}: missing 'Document ID:'/'Effective Date:' metadata lines")

    # The title is rendered as the first non-empty line of the PDF, before
    # the metadata block; pypdf's extraction generally preserves reading
    # order for this simple single-column layout.
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    title = first_line

    headings = list(_PDF_SECTION_RE.finditer(text))
    sections: list[ParsedSection] = []
    for index, match in enumerate(headings):
        heading = match.group(1).strip()
        start = match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = text[start:end].strip()
        sections.append(ParsedSection(heading=heading, text=body, order_index=index))

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        effective_date=effective_date,
        source_path=path,
        source_format="pdf",
        sections=sections,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
_PARSERS = {"md": parse_markdown, "html": parse_html, "pdf": parse_pdf}


def parse_document(path: Path) -> ParsedDocument:
    fmt = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
    if fmt is None:
        raise ParseError(f"{path}: unsupported file extension {path.suffix!r}")
    return _PARSERS[fmt](path)


_NON_POLICY_STEMS = {"readme"}


def parse_corpus(policies_dir: Path) -> list[ParsedDocument]:
    """Parse every generated policy file directly under `policies_dir`.

    Files with a supported extension are treated as corpus documents
    except for known non-policy files (``README.md``) that also live in
    this directory; Python source files (the ``content.py``/
    ``generate_corpus.py`` generator, ``__init__.py``) are already
    excluded by extension.
    """
    paths = sorted(
        p
        for p in policies_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.stem.lower() not in _NON_POLICY_STEMS
    )
    return [parse_document(p) for p in paths]
