"""Renders the canonical policy content in ``content.py`` into the actual
Markdown, HTML, and PDF files that make up the policy corpus under
``policies/``.

Run with:

    python -m policies.generate_corpus

Each :class:`~policies.content.PolicyDoc` is rendered using a
format-specific layout so that the RAG ingestion pipeline's per-format
parsers (see ``rag/parsers.py``) have something realistic to parse:

- Markdown: ``#``/``##`` headings and ``**Label:** value`` metadata lines.
- HTML: semantic ``<h1>``/``<h2>`` headings and ``<meta name="..." ...>``
  metadata tags in the document ``<head>``.
- PDF: a title block rendered with larger/bold fonts, plain-text
  ``Label: value`` metadata lines, and section headings rendered as
  literal ``SECTION N: <heading>`` text so they survive PDF text
  extraction (font styling is not reliably recoverable from extracted
  PDF text, so the heading marker is encoded directly in the text).

Regenerating the corpus is deterministic: given the same ``content.py``,
byte-identical output is produced every time (no timestamps, random
ordering, or non-deterministic library behavior are used).
"""
from __future__ import annotations

import datetime
import html
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from policies.content import POLICIES, PolicyDoc, Section

POLICIES_DIR = Path(__file__).resolve().parent


def _slug(doc: PolicyDoc) -> str:
    return doc.doc_id.lower()


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def render_markdown(doc: PolicyDoc) -> str:
    lines = [
        f"# {doc.title}",
        "",
        f"**Document ID:** {doc.doc_id}  ",
        f"**Effective Date:** {doc.effective_date}  ",
        f"**Version:** {doc.version}  ",
        f"**Owner:** {doc.owner}",
        "",
        doc.summary,
        "",
    ]
    for section in doc.sections:
        lines.append(f"## {section.heading}")
        lines.append("")
        for paragraph in section.paragraphs:
            lines.append(paragraph)
            lines.append("")
        for bullet in section.bullets:
            lines.append(f"- {bullet}")
        if section.bullets:
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
def render_html(doc: PolicyDoc) -> str:
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{html.escape(doc.title)}</title>",
        f'<meta name="doc-id" content="{html.escape(doc.doc_id)}">',
        f'<meta name="effective-date" content="{html.escape(doc.effective_date)}">',
        f'<meta name="version" content="{html.escape(doc.version)}">',
        f'<meta name="owner" content="{html.escape(doc.owner)}">',
        "</head>",
        "<body>",
        f"<h1>{html.escape(doc.title)}</h1>",
        f'<p class="summary">{html.escape(doc.summary)}</p>',
    ]
    for section in doc.sections:
        parts.append(f"<h2>{html.escape(section.heading)}</h2>")
        for paragraph in section.paragraphs:
            parts.append(f"<p>{html.escape(paragraph)}</p>")
        if section.bullets:
            parts.append("<ul>")
            for bullet in section.bullets:
                parts.append(f"<li>{html.escape(bullet)}</li>")
            parts.append("</ul>")
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------
class _PolicyPDF(FPDF):
    pass


def render_pdf(doc: PolicyDoc, output_path: Path) -> int:
    """Render `doc` to a PDF at `output_path`. Returns the page count."""
    pdf = _PolicyPDF(format="Letter")
    # Fixed creation date so regenerating the corpus from unchanged content
    # produces byte-identical PDFs (fpdf2 otherwise stamps the real wall-clock
    # time into both the PDF metadata and the derived file ID).
    pdf.set_creation_date(datetime.datetime(2024, 1, 1, 0, 0, 0))
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 20, 20)
    pdf.add_page()

    def line(text: str, height: float = 6) -> None:
        # Explicitly reset the cursor to the left margin after every cell;
        # fpdf2's default post-cell cursor position for a full-width
        # (w=0) multi_cell is the right margin, which otherwise starves
        # the next call of horizontal space.
        pdf.multi_cell(0, height, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "B", 16)
    line(doc.title, 9)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    for label, value in (
        ("Document ID", doc.doc_id),
        ("Effective Date", doc.effective_date),
        ("Version", doc.version),
        ("Owner", doc.owner),
    ):
        line(f"{label}: {value}")
    pdf.ln(2)

    pdf.set_font("Helvetica", "I", 10)
    line(doc.summary)
    pdf.ln(4)

    for index, section in enumerate(doc.sections, start=1):
        pdf.set_font("Helvetica", "B", 12)
        line(f"SECTION {index}: {section.heading}", 8)
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 10)
        for paragraph in section.paragraphs:
            line(paragraph)
            pdf.ln(1)
        for bullet in section.bullets:
            line(f"- {bullet}")
        pdf.ln(3)

    pdf.output(str(output_path))
    return pdf.page_no()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def generate(output_dir: Path = POLICIES_DIR) -> None:
    total_pdf_pages = 0
    for doc in POLICIES:
        slug = _slug(doc)
        if doc.fmt == "md":
            (output_dir / f"{slug}.md").write_text(render_markdown(doc), encoding="utf-8")
        elif doc.fmt == "html":
            (output_dir / f"{slug}.html").write_text(render_html(doc), encoding="utf-8")
        elif doc.fmt == "pdf":
            total_pdf_pages += render_pdf(doc, output_dir / f"{slug}.pdf")
        else:
            raise ValueError(f"Unknown format {doc.fmt!r} for {doc.doc_id}")
    print(f"Generated {len(POLICIES)} policy documents into {output_dir}")


def estimate_total_pages(output_dir: Path = POLICIES_DIR) -> int:
    """Render every document to a throwaway PDF (regardless of its real
    output format) purely to get an objective, reproducible page count for
    the whole corpus, rather than guessing from a words-per-page constant.
    """
    import tempfile

    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for doc in POLICIES:
            pages = render_pdf(doc, tmp_path / f"{_slug(doc)}.pdf")
            total += pages
    return total


if __name__ == "__main__":
    generate()
    print(f"Estimated total corpus length if every document were paginated as a PDF: "
          f"{estimate_total_pages()} pages")
