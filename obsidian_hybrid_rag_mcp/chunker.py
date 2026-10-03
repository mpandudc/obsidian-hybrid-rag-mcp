"""
Heading-aware Markdown chunking engine for Obsidian vaults.

Sections split on real headings (# through ####, never on `#` comments inside
fenced code). Oversized sections are packed paragraph by paragraph; oversized
paragraphs split by lines and overlong lines are hard-sliced, so no chunk ever
exceeds `chunk_char_limit`. Every chunk keeps exact source line numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from obsidian_hybrid_rag_mcp.markdown import (
    extract_summary,
    extract_title,
    frontmatter_span,
    iter_headings,
)

__all__ = [
    "MarkdownChunk", "chunk_markdown", "clean_frontmatter", "extract_summary", "extract_title", "split_oversized",
]


@dataclass
class MarkdownChunk:
    title: str
    heading: str
    text: str
    summary: str
    line_start: int
    line_end: int


def clean_frontmatter(content: str) -> str:
    """Strip YAML frontmatter from document header."""
    lines = content.splitlines()
    span = frontmatter_span(lines)
    return "\n".join(lines[span:]).lstrip() if span else content


def split_oversized(text: str, limit: int) -> list[str]:
    """Split a paragraph longer than `limit` by lines, then hard-slice any single overlong line.

    Guarantees every piece is <= limit, so one giant paragraph (no blank lines) can never
    become a single huge chunk that blows up embedding memory.
    """
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for line in text.splitlines():
        while len(line) > limit:
            if buf:
                pieces.append("\n".join(buf))
                buf, buf_len = [], 0
            pieces.append(line[:limit])
            line = line[limit:]
        if buf_len + len(line) + 1 > limit and buf:
            pieces.append("\n".join(buf))
            buf, buf_len = [], 0
        buf.append(line)
        buf_len += len(line) + 1
    if buf:
        pieces.append("\n".join(buf))
    return pieces


# A piece is (line_start, line_end, text), 1-based inclusive line numbers.
Piece = tuple[int, int, str]


def _paragraphs(numbered: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    paras, cur = [], []
    for lineno, line in numbered:
        if line.strip():
            cur.append((lineno, line))
        elif cur:
            paras.append(cur)
            cur = []
    if cur:
        paras.append(cur)
    return paras


def _split_paragraph(para: list[tuple[int, str]], limit: int) -> list[Piece]:
    text = "\n".join(line for _, line in para)
    if len(text) <= limit:
        return [(para[0][0], para[-1][0], text)]
    pieces: list[Piece] = []
    buf: list[tuple[int, str]] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            pieces.append((buf[0][0], buf[-1][0], "\n".join(line for _, line in buf)))
        buf, buf_len = [], 0

    for lineno, line in para:
        while len(line) > limit:
            flush()
            pieces.append((lineno, lineno, line[:limit]))
            line = line[limit:]
        if buf and buf_len + len(line) + 1 > limit:
            flush()
        buf.append((lineno, line))
        buf_len += len(line) + 1
    flush()
    return pieces


def _make_chunk(title: str, heading: str, summary: str, pieces: list[Piece]) -> MarkdownChunk:
    return MarkdownChunk(
        title=title,
        heading=heading,
        text="\n\n".join(p[2] for p in pieces).strip(),
        summary=summary,
        line_start=pieces[0][0],
        line_end=pieces[-1][1],
    )


def chunk_markdown(
    content: str,
    title: str,
    summary: str,
    chunk_char_limit: int = 1500,
    min_chunk_chars: int = 30,
    max_line_chars: int | None = None,
) -> list[MarkdownChunk]:
    """Split markdown by headings, then pack paragraphs into chunks of at most `chunk_char_limit`.

    Frontmatter is excluded (line numbers still refer to the original file). Lines longer than
    `max_line_chars` (e.g. pasted JSON/base64 blobs) are dropped from the indexed text.
    """
    lines = content.splitlines()
    start = frontmatter_span(lines)
    heading_at = {i: text for i, _, text in iter_headings(lines, start=start, max_level=4)}

    sections: list[tuple[str, list[tuple[int, str]]]] = []
    heading, numbered = "Overview", []
    for i in range(start, len(lines)):
        if i in heading_at:
            if numbered:
                sections.append((heading, numbered))
            heading, numbered = heading_at[i], []
        line = lines[i]
        if max_line_chars and len(line) > max_line_chars:
            line = ""
        numbered.append((i + 1, line))
    if numbered:
        sections.append((heading, numbered))

    sectioned = [
        (heading, [p for para in _paragraphs(numbered) for p in _split_paragraph(para, chunk_char_limit)])
        for heading, numbered in sections
    ]
    # Tiny sections are noise in a long note, but a short note must still be searchable.
    if any(sum(len(p[2].strip()) for p in pieces) >= min_chunk_chars for _, pieces in sectioned):
        sectioned = [(h, ps) for h, ps in sectioned if sum(len(p[2].strip()) for p in ps) >= min_chunk_chars]

    chunks: list[MarkdownChunk] = []
    for heading, pieces in sectioned:
        if not pieces:
            continue
        buf: list[Piece] = []
        buf_len = 0
        for piece in pieces:
            extra = len(piece[2]) + (2 if buf else 0)
            if buf and buf_len + extra > chunk_char_limit:
                chunks.append(_make_chunk(title, heading, summary, buf))
                buf, buf_len, extra = [], 0, len(piece[2])
            buf.append(piece)
            buf_len += extra
        if buf:
            chunks.append(_make_chunk(title, heading, summary, buf))
    return chunks
