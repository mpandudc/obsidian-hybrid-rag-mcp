"""
Markdown primitives shared by the chunker, indexer and server.

Every heading scan here is fence-aware: a `# comment` inside a ``` or ~~~ code
block is code, not a heading.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import yaml

FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")
WIKILINK_RE = re.compile(r"(!?)\[\[([^\]|#^]+)([#^][^\]|]*)?(\|[^\]]*)?\]\]")


def frontmatter_span(lines: list[str]) -> int:
    """Number of leading lines taken by a YAML frontmatter block (0 if none)."""
    if not lines or lines[0].strip() != "---":
        return 0
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            return i + 1
    return 0


def parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter into a dict. Invalid or missing frontmatter returns {}."""
    lines = content.splitlines()
    span = frontmatter_span(lines)
    if not span:
        return {}
    try:
        data = yaml.safe_load("\n".join(lines[1:span - 1]))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def normalize_tags(value) -> list[str]:
    """Frontmatter tags as a lowercase list: accepts a list, 'a, b' or '#a #b'."""
    if value is None:
        return []
    items = value if isinstance(value, list) else re.split(r"[,\s]+", str(value))
    tags = []
    for item in items:
        tag = str(item).strip().strip("\"'").lstrip("#").strip().lower()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def iter_code_mask(lines: list[str], start: int = 0) -> Iterator[tuple[int, bool]]:
    """Yield (index, inside_code) for each line from `start`; fence lines count as code."""
    fence = None
    for i in range(start, len(lines)):
        m = FENCE_RE.match(lines[i])
        if fence is None:
            if m:
                fence = m.group(1)
            yield i, fence is not None
            continue
        if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) \
                and not lines[i].strip()[len(m.group(1)):].strip():
            fence = None
        yield i, True


def iter_headings(lines: list[str], start: int = 0, max_level: int = 6) -> Iterator[tuple[int, int, str]]:
    """Yield (index, level, text) for real markdown headings, skipping fenced code."""
    for i, in_code in iter_code_mask(lines, start):
        if in_code:
            continue
        m = HEADING_RE.match(lines[i])
        if m and len(m.group(1)) <= max_level:
            yield i, len(m.group(1)), m.group(2).strip()


def extract_title(content: str, fallback: str) -> str:
    """First H1 outside code blocks, else frontmatter `title`, else `fallback`."""
    lines = content.splitlines()
    for _, _level, text in iter_headings(lines, start=frontmatter_span(lines), max_level=1):
        return text
    title = parse_frontmatter(content).get("title")
    return str(title).strip() if title else fallback


def extract_section(lines: list[str], heading: str) -> tuple[int, int] | None:
    """Line range [start, end) of the first section whose heading contains `heading`."""
    target = heading.strip().lstrip("#").strip().lower()
    start, level = None, 0
    for i, lvl, text in iter_headings(lines, start=frontmatter_span(lines)):
        if start is None:
            if target in text.lower():
                start, level = i, lvl
        elif lvl <= level:
            return start, i
    return (start, len(lines)) if start is not None else None


def list_headings(lines: list[str]) -> list[str]:
    return [f"{'#' * lvl} {text}" for _, lvl, text in iter_headings(lines, start=frontmatter_span(lines))]


def iter_wikilinks(content: str) -> Iterator[tuple[int, re.Match]]:
    """Yield (line_index, match) for wikilinks outside fenced code.

    Match groups: 1 embed '!', 2 target, 3 '#heading' / '^block', 4 '|alias'.
    """
    lines = content.splitlines()
    for i, in_code in iter_code_mask(lines):
        if in_code:
            continue
        for m in WIKILINK_RE.finditer(lines[i]):
            yield i, m


def link_targets(content: str) -> list[str]:
    seen = []
    for _, m in iter_wikilinks(content):
        target = m.group(2).strip()
        if target and target not in seen:
            seen.append(target)
    return seen


def extract_summary(content: str, max_chars: int = 280) -> str:
    """Frontmatter `summary`/`description`, else the lead prose (skipping headings, quotes and code)."""
    fm = parse_frontmatter(content)
    for key in ("summary", "description"):
        if fm.get(key):
            text = " ".join(str(fm[key]).split())
            return text[:max_chars] + "..." if len(text) > max_chars else text
    lines = content.splitlines()
    picked: list[str] = []
    for i, in_code in iter_code_mask(lines, frontmatter_span(lines)):
        stripped = lines[i].strip()
        if in_code or not stripped or stripped.startswith(("#", ">")):
            continue
        picked.append(stripped)
        if len(" ".join(picked)) >= 200:
            break
    summary = " ".join(picked)
    return summary[:max_chars] + "..." if len(summary) > max_chars else summary
