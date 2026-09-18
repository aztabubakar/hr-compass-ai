"""Process-wide caches for the synthetic HR dataset and parsed policy docs.

Both are small and static for the lifetime of one server process (the JSON
files and the policy corpus don't change while the server runs), so they
are loaded once and reused across tool calls rather than re-reading disk
on every call. Neither cache duplicates any logic from `mock_data/` or
`rag/` — they call straight through to `mock_data.loader.load_all()` and
`rag.parsers.parse_corpus()` respectively.
"""
from __future__ import annotations

from mock_data.loader import load_all
from mock_data.models import HRDataset
from rag.config import RagConfig
from rag.models import ParsedDocument
from rag.parsers import parse_corpus

_dataset: HRDataset | None = None
_parsed_docs: dict[str, ParsedDocument] | None = None


def get_dataset() -> HRDataset:
    global _dataset
    if _dataset is None:
        _dataset = load_all()
    return _dataset


def get_parsed_docs() -> dict[str, ParsedDocument]:
    global _parsed_docs
    if _parsed_docs is None:
        docs = parse_corpus(RagConfig().policies_dir)
        _parsed_docs = {d.doc_id: d for d in docs}
    return _parsed_docs


def reset_caches() -> None:
    """Used by tests that swap the underlying data/corpus between cases."""
    global _dataset, _parsed_docs
    _dataset = None
    _parsed_docs = None
