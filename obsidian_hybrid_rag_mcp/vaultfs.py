"""
Vault filesystem helpers shared by the indexer and the server.

All note paths handed around are POSIX-style relative paths ('server/setup.md')
on every OS, so SQL prefix filters and tool output behave the same on Windows.
"""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import os
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

HIDDEN_DIRS = {".git", ".obsidian", ".trash", ".templates", ".smart-env", ".trash-bin"}
IGNORE_FILE = ".vaultignore"
BACKUP_DIR = Path(".trash") / "vault-mcp"


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def is_hidden(rel: str) -> bool:
    return any(part.startswith(".") or part in HIDDEN_DIRS for part in Path(rel).parts)


def iter_notes(vault: Path) -> Iterator[tuple[Path, str]]:
    """Yield (absolute_path, posix_rel_path) for every visible markdown note, sorted."""
    for root, dirs, files in os.walk(vault):
        dirs[:] = sorted(d for d in dirs if d not in HIDDEN_DIRS and not d.startswith("."))
        for name in sorted(files):
            if name.endswith(".md") and not name.startswith("."):
                full = Path(root) / name
                yield full, rel_posix(full, vault)


def load_ignore(vault: Path) -> list[str]:
    """Patterns from `<vault>/.vaultignore` (gitignore-like globs, '#' comments)."""
    try:
        text = (vault / IGNORE_FILE).read_text(encoding="utf-8")
    except OSError:
        return []
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def is_ignored(rel: str, patterns: list[str]) -> bool:
    """Match a POSIX rel path against .vaultignore patterns.

    'dir/' ignores a folder, a pattern with '/' matches the full path (fnmatch, '*' crosses '/'),
    a pattern without '/' matches the file name.
    """
    name = rel.rsplit("/", 1)[-1]
    for pat in patterns:
        if pat.endswith("/"):
            if rel.startswith(pat.lstrip("/")):
                return True
        elif "/" in pat:
            if fnmatch.fnmatchcase(rel, pat.lstrip("/")):
                return True
        elif fnmatch.fnmatchcase(name, pat):
            return True
    return False


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_matches(actual: str, expected: str) -> bool:
    expected = expected.strip().lower()
    return len(expected) >= 8 and actual.startswith(expected)


def atomic_write(path: Path, text: str) -> None:
    """Write via a temp file in the same folder + os.replace, so readers (Obsidian, sync
    plugins) never see a half-written note."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def backup_note(vault: Path, rel: str) -> str:
    """Copy a note to .trash/vault-mcp/<rel>.<timestamp>.md before it is overwritten."""
    src = vault / rel
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest_rel = (BACKUP_DIR / f"{rel.removesuffix('.md')}.{stamp}.md").as_posix()
    atomic_write(vault / dest_rel, src.read_text(encoding="utf-8", errors="replace"))
    return dest_rel


def resolve_note(vault: Path, rel_path: str, must_exist: bool) -> tuple[Path | None, str | None, str | None]:
    """Map a user-supplied note path to a file inside the vault: (path, posix_rel, error).

    Paths escaping the vault or touching hidden folders are rejected. For must_exist lookups
    a bare note name is resolved by filename; an ambiguous name lists the candidates.
    """
    clean = rel_path.strip().replace("\\", "/").strip("/")
    if not clean:
        return None, None, "Error: empty note path."
    if not clean.endswith(".md"):
        clean += ".md"
    candidate = (vault / clean).resolve()
    if not candidate.is_relative_to(vault):
        return None, None, f"Error: Access denied. Path `{rel_path}` is outside the vault."
    rel = rel_posix(candidate, vault)
    if is_hidden(rel):
        return None, None, f"Error: Access denied. `{rel_path}` is inside a hidden/system folder."
    if candidate.exists() or not must_exist:
        return candidate, rel, None

    name = Path(clean).name
    matches = [r for _, r in iter_notes(vault) if r.rsplit("/", 1)[-1] == name]
    if len(matches) == 1:
        return (vault / matches[0]).resolve(), matches[0], None
    if len(matches) > 1:
        listing = "\n".join(f"- `{m}`" for m in matches)
        return None, None, f"Error: `{rel_path}` is ambiguous. Use one of these paths:\n{listing}"
    return None, None, f"Error: Note `{rel_path}` not found in vault."


class FileLock:
    """Non-blocking cross-process lock on a file (fcntl on POSIX, msvcrt on Windows)."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")  # noqa: SIM115 - held open for the lock's lifetime
        try:
            try:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except ImportError:
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is None:
            return
        with contextlib.suppress(OSError, ImportError):
            try:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except ImportError:
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        self._fh.close()
        self._fh = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
