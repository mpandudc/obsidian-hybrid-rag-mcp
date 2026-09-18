"""
Heading-aware Markdown chunking engine for Obsidian vaults.
Preserves hierarchy, YAML frontmatter, wikilinks, and line-level traceability.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional


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
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2].lstrip()
    return content


def extract_title(content: str, fallback: str) -> str:
    """Extract first H1 heading, or fall back to note filename."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def extract_summary(content: str, max_chars: int = 280) -> str:
    """Extract lead narrative summary paragraph, skipping frontmatter and headings."""
    cleaned = clean_frontmatter(content)
    lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
        if len(" ".join(lines)) >= 200:
            break
    summary = " ".join(lines)
    return summary[:max_chars] + "..." if len(summary) > max_chars else summary


def chunk_markdown(
    content: str,
    title: str,
    summary: str,
    chunk_char_limit: int = 1200,
    min_chunk_chars: int = 30,
) -> List[MarkdownChunk]:
    """
    Split markdown document by natural headings (# through ####).
    If a section exceeds chunk_char_limit, it gracefully splits on double-newlines (paragraphs).
    """
    lines = content.splitlines()
    raw_chunks: List[dict] = []
    current_heading = "Overview"
    current_lines: List[str] = []
    start_line = 1

    heading_regex = re.compile(r"^(#{1,4})\s+(.+)$")

    for idx, line in enumerate(lines, start=1):
        match = heading_regex.match(line)
        if match:
            if current_lines:
                text = "\n".join(current_lines).strip()
                if len(text) >= min_chunk_chars:
                    raw_chunks.append({
                        "heading": current_heading,
                        "text": text,
                        "line_start": start_line,
                        "line_end": idx - 1,
                    })
                current_lines = []
            current_heading = match.group(2).strip()
            start_line = idx
        current_lines.append(line)

    if current_lines:
        text = "\n".join(current_lines).strip()
        if len(text) >= min_chunk_chars:
            raw_chunks.append({
                "heading": current_heading,
                "text": text,
                "line_start": start_line,
                "line_end": len(lines),
            })

    # Subdivide large chunks while preserving line offsets
    final_chunks: List[MarkdownChunk] = []
    for c in raw_chunks:
        if len(c["text"]) <= chunk_char_limit:
            final_chunks.append(
                MarkdownChunk(
                    title=title,
                    heading=c["heading"],
                    text=c["text"],
                    summary=summary,
                    line_start=c["line_start"],
                    line_end=c["line_end"],
                )
            )
        else:
            paragraphs = c["text"].split("\n\n")
            sub_buf: List[str] = []
            current_len = 0
            sub_start = c["line_start"]
            for p in paragraphs:
                if current_len + len(p) > chunk_char_limit and sub_buf:
                    sub_text = "\n\n".join(sub_buf).strip()
                    sub_lines = len(sub_text.splitlines())
                    final_chunks.append(
                        MarkdownChunk(
                            title=title,
                            heading=c["heading"],
                            text=sub_text,
                            summary=summary,
                            line_start=sub_start,
                            line_end=sub_start + sub_lines - 1,
                        )
                    )
                    sub_start = sub_start + sub_lines
                    sub_buf = [p]
                    current_len = len(p)
                else:
                    sub_buf.append(p)
                    current_len += len(p)
            if sub_buf:
                sub_text = "\n\n".join(sub_buf).strip()
                sub_lines = len(sub_text.splitlines())
                final_chunks.append(
                    MarkdownChunk(
                        title=title,
                        heading=c["heading"],
                        text=sub_text,
                        summary=summary,
                        line_start=sub_start,
                        line_end=sub_start + sub_lines - 1,
                    )
                )

    return final_chunks
