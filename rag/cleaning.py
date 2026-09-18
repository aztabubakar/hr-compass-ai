"""Text cleaning applied uniformly after format-specific parsing.

Kept intentionally small: by the time text reaches this module, parsing
has already stripped markup, so cleaning only needs to normalize
whitespace and unicode, not undo formatting.
"""
from __future__ import annotations

import re
import unicodedata

_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    """Normalize unicode and collapse redundant whitespace.

    - NFKC unicode normalization so visually-identical characters (e.g.
      curly vs straight quotes introduced by a source editor) compare and
      embed consistently.
    - Collapse runs of 3+ blank lines to a single blank line, and runs of
      2+ spaces/tabs to one, without altering intentional single line
      breaks (which matter for bullet lists).
    """
    text = unicodedata.normalize("NFKC", text)
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()
