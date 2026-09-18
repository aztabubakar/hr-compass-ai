"""Shared pytest fixtures.

Unit tests for parsing/chunking/retrieval use small, self-contained
fixture documents (defined here) rather than the real policy corpus, so
they keep passing unchanged if the real corpus content is edited later.
Separate integration tests (in test_parsers.py) run against the real
`policies/` directory to check the corpus itself.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.paths import POLICIES_DIR
from rag.config import RagConfig
from rag.ingest import run_ingest

FIXTURE_MARKDOWN = """# Sample Time Off Policy

**Document ID:** TEST-PTO-001
**Effective Date:** 2024-01-01
**Version:** 1.0
**Owner:** Test Team

A short sample policy about time off, used only for automated tests.

## Requesting Time Off

Employees request time off through the test system at least 5 business days in advance. Approval comes from the employee's manager. Managers should respond within 2 business days.

## Carryover Rules

Up to 5 unused days may be carried into the next year. Anything beyond that is forfeited on December 31st, unless state law requires otherwise.
"""

FIXTURE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sample Security Policy</title>
<meta name="doc-id" content="TEST-SEC-001">
<meta name="effective-date" content="2024-02-01">
<meta name="version" content="1.0">
<meta name="owner" content="Test Team">
</head>
<body>
<h1>Sample Security Policy</h1>
<p class="summary">A short sample security policy, used only for automated tests.</p>
<h2>Password Requirements</h2>
<p>All test accounts require a password of at least 14 characters and multi-factor authentication.</p>
<h2>Incident Reporting</h2>
<p>Report suspected incidents to the test security team within 24 hours of discovery.</p>
</body>
</html>
"""

# A long section (repeated distinct sentences) to exercise chunk splitting
# and overlap logic in test_chunking.py.
_LONG_SENTENCES = " ".join(f"This is unique filler sentence number {i} about travel." for i in range(1, 40))
FIXTURE_MARKDOWN_LONG_SECTION = f"""# Sample Travel Policy

**Document ID:** TEST-TRV-001
**Effective Date:** 2024-03-01
**Version:** 1.0
**Owner:** Test Team

A short sample travel policy with one deliberately long section.

## Booking Travel

{_LONG_SENTENCES}
"""


@pytest.fixture
def tiny_corpus_dir(tmp_path: Path) -> Path:
    """A small multi-format, multi-document corpus for retrieval tests."""
    (tmp_path / "test-pto-001.md").write_text(FIXTURE_MARKDOWN, encoding="utf-8")
    (tmp_path / "test-sec-001.html").write_text(FIXTURE_HTML, encoding="utf-8")
    return tmp_path


@pytest.fixture
def hashing_config(tmp_path: Path, tiny_corpus_dir: Path) -> RagConfig:
    """A RagConfig pointed at the tiny fixture corpus, using the offline,
    dependency-free hashing embedding backend so tests never need network
    access or model downloads."""
    return RagConfig(
        policies_dir=tiny_corpus_dir,
        index_dir=tmp_path / "index",
        embedding_backend="hashing",
    )


# ---------------------------------------------------------------------------
# MCP fixtures (Phase 2). These build a real index over the *actual* Solara
# Dynamics corpus (not the tiny fixture above) using the hashing backend,
# since the MCP tools/workflows are meant to be exercised against the real
# policies. Session-scoped so the ~12-document corpus is only embedded once
# for the whole test run.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def real_corpus_hashing_index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    index_dir = tmp_path_factory.mktemp("mcp_rag_index")
    run_ingest(RagConfig(policies_dir=POLICIES_DIR, index_dir=index_dir, embedding_backend="hashing"))
    return index_dir


@pytest.fixture
def mcp_server_env(real_corpus_hashing_index_dir: Path, tmp_path: Path) -> dict[str, str]:
    """Environment for launching the MCP server subprocess in tests:
    hashing backend + the pre-built real-corpus index (both offline and
    deterministic), plus a per-test isolated runtime-tickets file so
    ticket-creation tests never touch each other or the real project data.
    """
    return {
        **os.environ,
        "EMBEDDING_BACKEND": "hashing",
        "RAG_INDEX_DIR": str(real_corpus_hashing_index_dir),
        "HR_COMPASS_RUNTIME_TICKETS_PATH": str(tmp_path / "runtime_tickets.json"),
    }
