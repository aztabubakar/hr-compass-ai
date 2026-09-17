# Policy Corpus

**Status: implemented (Phase 1).** 12 documents for the fictional company
Solara Dynamics, Inc., totaling 37 pages (see `design-and-evaluation.md`
Section 3 for the full list, format breakdown, and page-count methodology).

- `content.py` — canonical policy text (single source of truth).
- `generate_corpus.py` — renders `content.py` into the actual `.md`/`.html`/`.pdf`
  files in this directory. Run `python -m policies.generate_corpus` after
  editing `content.py`.
- `pol-*.md` / `pol-*.html` / `pol-*.pdf` — the generated corpus, parsed by
  `rag/parsers.py`.
