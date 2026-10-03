"""
Wikilink graph helpers: resolution, link reports, vault lint and link-rewriting moves.

Resolution follows Obsidian: a target with a '/' is matched as a vault path, a bare
target by file name (ambiguous names resolve to the shortest path).
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from obsidian_hybrid_rag_mcp.markdown import (
    WIKILINK_RE,
    frontmatter_span,
    iter_code_mask,
    iter_wikilinks,
    link_targets,
    mask_inline_code,
    parse_frontmatter,
)
from obsidian_hybrid_rag_mcp.vaultfs import HIDDEN_DIRS, iter_notes

DEFAULT_STATUS_VALUES = "draft,active,blocked,approved,verified,completed,falsified,superseded,archived"


def is_hub(rel_or_target: str) -> bool:
    stem = Path(rel_or_target).name.lower().removesuffix(".md")
    return stem in ("readme", "index", "inbox", "hub") or stem.endswith(("index", "-hub")) or "overview" in stem


class LinkIndex:
    """Snapshot of vault note paths for resolving wikilink targets."""

    def __init__(self, vault: Path):
        self.vault = vault
        self.notes: list[str] = []
        self.by_stem: dict[str, list[str]] = defaultdict(list)
        self.by_path: dict[str, str] = {}
        self.files: set[str] = set()
        for _root, dirs, names in os.walk(vault):
            dirs[:] = [d for d in dirs if d not in HIDDEN_DIRS and not d.startswith(".")]
            for name in names:
                self.files.add(name.lower())
        for _, rel in iter_notes(vault):
            self.notes.append(rel)
            noext = rel[:-3].lower()
            self.by_path[noext] = rel
            self.by_stem[noext.rsplit("/", 1)[-1]].append(rel)
        for paths in self.by_stem.values():
            paths.sort(key=lambda p: (p.count("/"), p))
        # Notes linked from the root README (the vault's map of content) are hubs too.
        self.moc: set[str] = set()
        readme = self.by_path.get("readme")
        if readme:
            try:
                text = (vault / readme).read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            self.moc = {r for r in (self.resolve(t) for t in link_targets(text)) if r}

    def is_hub_note(self, rel: str) -> bool:
        return is_hub(rel) or rel in self.moc

    def is_hub_target(self, target: str) -> bool:
        if is_hub(target):
            return True
        rel = self.resolve(target)
        return rel is not None and rel in self.moc

    def resolve(self, target: str) -> str | None:
        """Note rel path for a link target, or None (attachments resolve to None too)."""
        low = target.strip().replace("\\", "/").lower().removesuffix(".md")
        if low in self.by_path:
            return self.by_path[low]
        if "/" in low:
            for noext, rel in self.by_path.items():
                if noext.endswith("/" + low):
                    return rel
            return None
        hits = self.by_stem.get(low)
        return hits[0] if hits else None

    def exists(self, target: str) -> bool:
        suffix = Path(target).suffix.lower()
        if suffix and suffix != ".md":
            return Path(target).name.lower() in self.files
        return self.resolve(target) is not None

    def stem_is_unique(self, rel: str) -> bool:
        return len(self.by_stem.get(rel[:-3].rsplit("/", 1)[-1].lower(), [])) == 1


def link_report(vault: Path, content: str, rel_path: str) -> str:
    """Report broken wikilinks and a missing hub link (vault convention: no orphan notes)."""
    targets = link_targets(content)
    index = LinkIndex(vault)
    broken = [t for t in targets if not index.exists(t)]
    notes = []
    if not targets:
        notes.append("no [[wikilinks]] at all — link the note to its hub and related notes")
    elif not index.is_hub_note(rel_path) and not any(index.is_hub_target(t) for t in targets):
        notes.append("no link to a hub note ([[README]], an *INDEX note, or a project *overview*)")
    if broken:
        notes.append("broken links (no matching note): " + ", ".join(f"[[{b}]]" for b in broken))
    if not notes:
        return f"Link check: OK ({len(targets)} links)."
    return "Link check WARNING: " + "; ".join(notes) + "."


def backlinks(vault: Path, target_rel: str, limit: int = 50) -> list[str]:
    index = LinkIndex(vault)
    hits = []
    for full, rel in iter_notes(vault):
        if rel == target_rel:
            continue
        try:
            text = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        seen_lines = set()
        for lineno, m in iter_wikilinks(text):
            if lineno in seen_lines or index.resolve(m.group(2)) != target_rel:
                continue
            seen_lines.add(lineno)
            hits.append(f"- `{rel}` L{lineno + 1}: {lines[lineno].strip()[:200]}")
            if len(hits) >= limit:
                return hits
    return hits


@dataclass
class LintReport:
    total: int = 0
    broken: list[tuple[str, str]] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    no_hub: list[str] = field(default_factory=list)
    no_frontmatter: list[str] = field(default_factory=list)
    bad_status: list[tuple[str, str]] = field(default_factory=list)
    oversized: list[tuple[str, int]] = field(default_factory=list)
    junk_lines: list[tuple[str, int]] = field(default_factory=list)


def lint_vault(vault: Path, folder: str = "", status_values: set[str] | None = None,
               max_bytes: int = 512 * 1024, max_line_chars: int = 10000) -> LintReport:
    """Whole-vault link graph; findings reported only for notes under `folder`."""
    index = LinkIndex(vault)
    inbound: dict[str, int] = defaultdict(int)
    report = LintReport()
    prefix = f"{folder.strip('/')}/" if folder.strip("/") else ""
    for full, rel in iter_notes(vault):
        try:
            raw = full.read_bytes()
        except OSError:
            continue
        text = raw.decode("utf-8", errors="replace")
        in_scope = rel.startswith(prefix)
        lines = text.splitlines()
        # Drop blob lines first: a pasted JSON dump starting with '[[' is not a wikilink.
        clean = "\n".join(ln for ln in lines if len(ln) <= max_line_chars)
        targets = link_targets(clean)
        for t in targets:
            resolved = index.resolve(t)
            if resolved and resolved != rel:
                inbound[resolved] += 1
            elif resolved is None and not index.exists(t) and in_scope:
                report.broken.append((rel, t))
        if not in_scope:
            continue
        report.total += 1
        if not index.is_hub_note(rel) and not any(index.is_hub_target(t) for t in targets):
            report.no_hub.append(rel)
        if not frontmatter_span(lines):
            report.no_frontmatter.append(rel)
        elif status_values is not None:
            status = parse_frontmatter(text).get("status")
            if status is not None and str(status).strip().lower() not in status_values:
                report.bad_status.append((rel, str(status)))
        if len(raw) > max_bytes:
            report.oversized.append((rel, len(raw)))
        longest = max((len(ln) for ln in lines), default=0)
        if longest > max_line_chars:
            report.junk_lines.append((rel, longest))
    report.orphans = [rel for rel in index.notes
                      if rel.startswith(prefix) and inbound[rel] == 0 and rel.lower() != "readme.md"]
    return report


def _new_target(old_target: str, new_rel: str, index_after: LinkIndex) -> str:
    """Link text for the moved note: keep path style if the old link used a path."""
    new_noext = new_rel[:-3]
    if "/" in old_target or not index_after.stem_is_unique(new_rel):
        return new_noext
    return new_noext.rsplit("/", 1)[-1]


def rewrite_links(text: str, old_rel: str, new_rel: str, index_before: LinkIndex,
                  index_after: LinkIndex) -> tuple[str, int]:
    """Point every wikilink that resolved to `old_rel` at `new_rel` (fenced and inline code untouched)."""
    lines = text.split("\n")
    count = 0
    code = dict(iter_code_mask(lines))

    def repl(m: re.Match) -> str:
        nonlocal count
        if index_before.resolve(m.group(2)) != old_rel:
            return m.group(0)
        count += 1
        target = _new_target(m.group(2).strip(), new_rel, index_after)
        return f"{m.group(1)}[[{target}{m.group(3) or ''}{m.group(4) or ''}]]"

    for i, line in enumerate(lines):
        if code.get(i) or "[[" not in line:
            continue
        out, pos = [], 0
        for m in WIKILINK_RE.finditer(mask_inline_code(line)):
            out.append(line[pos:m.start()])
            out.append(repl(WIKILINK_RE.match(line, m.start())))
            pos = m.end()
        lines[i] = "".join(out) + line[pos:]
    return "\n".join(lines), count
