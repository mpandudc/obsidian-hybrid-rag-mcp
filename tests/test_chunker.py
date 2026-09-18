import pytest
from src.chunker import chunk_markdown, clean_frontmatter, extract_summary, extract_title


def test_clean_frontmatter():
    content = "---\ntitle: Test\ntags: [test]\n---\n# Actual Content\nBody text."
    cleaned = clean_frontmatter(content)
    assert cleaned.startswith("# Actual Content")


def test_extract_title():
    content = "# My Obsidian Note\nSome text."
    assert extract_title(content, fallback="fallback") == "My Obsidian Note"

    content_no_h1 = "## Section 2\nText"
    assert extract_title(content_no_h1, fallback="fallback") == "fallback"


def test_chunk_markdown():
    sample_md = """---
tags: [architecture, homelab]
---
# System Architecture

Introduction paragraph about the system architecture.

## Vector Database

SQLite vec is used for dense vector retrieval.
It is lightweight and in-process.

## Lexical Engine

FTS5 BM25 is used for exact term matching.
"""
    title = extract_title(sample_md, fallback="Architecture")
    summary = extract_summary(sample_md)
    chunks = chunk_markdown(sample_md, title=title, summary=summary)

    assert len(chunks) >= 2
    headings = [c.heading for c in chunks]
    assert "Vector Database" in headings
    assert "Lexical Engine" in headings
    for c in chunks:
        assert c.title == "System Architecture"
        assert c.line_start > 0
        assert c.line_end >= c.line_start
