from obsidian_hybrid_rag_mcp.chunker import chunk_markdown, extract_title
from obsidian_hybrid_rag_mcp.markdown import (
    extract_section,
    link_targets,
    normalize_tags,
    parse_frontmatter,
)

FENCED = """---
title: Setup
tags: [Infra, "#proxmox"]
status: Verified
---
# Real Title

Intro text for the note.

## Setup

```bash
# install deps
pip install foo
# run it
python main.py
```

~~~
# also code
~~~

## Next

After the code. [[README]] and [[cuantum-overview#Strategy|alias]]

```
[[not-a-link]]
```
"""


def test_code_fence_comments_are_not_headings():
    chunks = chunk_markdown(FENCED, "Real Title", "", min_chunk_chars=1)
    headings = [c.heading for c in chunks]
    assert headings == ["Real Title", "Setup", "Next"]
    setup = next(c for c in chunks if c.heading == "Setup")
    assert "# install deps" in setup.text and "python main.py" in setup.text and "# also code" in setup.text


def test_extract_title_ignores_fenced_h1_and_falls_back_to_frontmatter():
    assert extract_title("```bash\n# comment\n```\n# Real\n", "f") == "Real"
    assert extract_title("---\ntitle: From FM\n---\n## Only H2\n", "f") == "From FM"
    assert extract_title("## Only H2\n", "f") == "f"


def test_chunks_exclude_frontmatter_and_keep_exact_lines():
    chunks = chunk_markdown(FENCED, "Real Title", "", min_chunk_chars=1)
    lines = FENCED.splitlines()
    assert all("tags:" not in c.text for c in chunks)
    for c in chunks:
        source = "\n".join(lines[c.line_start - 1:c.line_end])
        for line in c.text.splitlines():
            assert line in source


def test_line_numbers_accurate_when_splitting_big_sections():
    body = "# T\n\n## S\n\n" + "\n\n".join(f"para {i} " + "word " * 60 for i in range(40))
    chunks = chunk_markdown(body, "T", "", chunk_char_limit=700)
    lines = body.splitlines()
    assert len(chunks) > 5
    for c in chunks:
        assert len(c.text) <= 700
        assert lines[c.line_start - 1].rstrip() == c.text.splitlines()[0].rstrip()
        assert lines[c.line_end - 1].rstrip() == c.text.splitlines()[-1].rstrip()


def test_blob_lines_dropped_from_chunks():
    blob = "[[null,null," + "x" * 50000 + "]]"
    text = f"# Note\n\nReal content here about macro.\n\n{blob}\n\nMore real content.\n"
    chunks = chunk_markdown(text, "Note", "", max_line_chars=10000)
    joined = "\n".join(c.text for c in chunks)
    assert "null,null" not in joined and "Real content" in joined and "More real content" in joined


def test_frontmatter_tags_status_and_links():
    fm = parse_frontmatter(FENCED)
    assert normalize_tags(fm["tags"]) == ["infra", "proxmox"]
    assert normalize_tags("a, #b c") == ["a", "b", "c"]
    assert str(fm["status"]).lower() == "verified"
    assert link_targets(FENCED) == ["README", "cuantum-overview"]  # fenced [[not-a-link]] ignored


def test_extract_section_is_fence_aware():
    lines = FENCED.splitlines()
    start, end = extract_section(lines, "Setup")
    assert lines[start] == "## Setup"
    assert lines[end] == "## Next"
    assert extract_section(lines, "install deps") is None


def test_short_note_still_produces_a_chunk():
    chunks = chunk_markdown("# A\n\nmomentum alpha\n", "A", "")
    assert len(chunks) == 1 and "momentum alpha" in chunks[0].text
    long_note = "# T\n\n## Tiny\n\nx\n\n## Real\n\n" + "real content " * 10
    assert [c.heading for c in chunk_markdown(long_note, "T", "")] == ["Real"]
