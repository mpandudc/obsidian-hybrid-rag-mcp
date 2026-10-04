#!/usr/bin/env python3
"""
FastMCP server for hybrid retrieval and safe editing of an Obsidian vault.

Search: SQLite FTS5 (BM25) + sqlite-vec dense vectors (BAAI/bge-m3), fused with
Reciprocal Rank Fusion and reranked by a Jina v2 cross-encoder.
Editing: write / append / exact-string edit / move, all confined to the vault, written
atomically, with optional optimistic-concurrency hashes, a wikilink report and a
re-index after every change.

Re-indexing (VAULT_INDEX_MODE):
  inprocess (default) - a background thread in this process reuses the already-loaded
                        embedding model, so a write never loads a second bge-m3 copy.
  command             - run VAULT_INDEXER_COMMAND (e.g. a memory-capped wrapper script).
  off                 - never re-index from the server (cron only).
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import os
import shlex
import sqlite3
import subprocess
import sys
import threading
import time
import warnings
from collections import OrderedDict
from pathlib import Path

# Offline & thread settings
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.filterwarnings("ignore")

from fastmcp import FastMCP  # noqa: E402

from obsidian_hybrid_rag_mcp import indexer  # noqa: E402
from obsidian_hybrid_rag_mcp.links import (  # noqa: E402
    DEFAULT_STATUS_VALUES,
    LinkIndex,
    backlinks,
    link_report,
    lint_vault,
    rewrite_links,
)
from obsidian_hybrid_rag_mcp.markdown import (  # noqa: E402
    extract_section,
    frontmatter_span,
    list_headings,
)
from obsidian_hybrid_rag_mcp.search import build_fts_query, diversify, search  # noqa: E402, F401
from obsidian_hybrid_rag_mcp.vaultfs import (  # noqa: E402
    atomic_write,
    backup_note,
    file_hash,
    hash_matches,
    iter_notes,
    resolve_note,
)


def _env_path(*names: str, default: Path) -> Path:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser().resolve()
    return default.expanduser().resolve()


def _env_float(name: str) -> float | None:
    value = os.environ.get(name, "").strip()
    return float(value) if value else None


# Paths & Models
VAULT_PATH = _env_path("OBSIDIAN_VAULT_PATH", "VAULT_PATH", default=Path.home() / "vaults" / "pandu-second-brain")
DB_PATH = _env_path("VAULT_INDEX_DB", "INDEX_DB_PATH", default=Path.home() / ".hermes" / "vault-index.db")
FASTEMBED_CACHE_DIR = _env_path("FASTEMBED_CACHE_DIR", default=Path.home() / ".cache" / "fastembed")
EMBED_MODEL_NAME = indexer.EMBED_MODEL_NAME
RERANK_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
MAX_CHUNKS_PER_NOTE = int(os.environ.get("VAULT_MAX_CHUNKS_PER_NOTE", "2"))
# Cross-encoder logit floor; unset = no floor. Calibrate with `vault-eval` before enabling.
MIN_RERANK_SCORE = _env_float("VAULT_MIN_RERANK_SCORE")
# Cross-encoder budget: rerank cost on CPU is ~linear in candidates x characters.
# vault-eval on 3 vCPU, 30 queries: 20x1500 MRR 0.97 / 11.9 s, 12x800 0.93 / 4.8 s, 8x600 0.96 / 2.6 s.
RERANK_CHARS = int(os.environ.get("VAULT_RERANK_CHARS", "600"))
RERANK_POOL = int(os.environ.get("VAULT_RERANK_POOL", "8"))


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))  # respects the container's cpuset
    except AttributeError:
        return os.cpu_count() or 2


RERANK_THREADS = int(os.environ.get("VAULT_RERANK_THREADS", str(min(4, _available_cpus()))))
READ_MAX_CHARS = int(os.environ.get("VAULT_READ_MAX_CHARS", "8000"))
STATUS_VALUES = {s.strip().lower() for s in os.environ.get("VAULT_STATUS_VALUES", DEFAULT_STATUS_VALUES).split(",")
                 if s.strip()}

# Re-index configuration. INDEXER_RUNNER is the legacy name of VAULT_INDEXER_COMMAND.
INDEXER_COMMAND = os.environ.get("VAULT_INDEXER_COMMAND") or os.environ.get("INDEXER_RUNNER", "")
INDEX_MODE = (os.environ.get("VAULT_INDEX_MODE") or ("command" if INDEXER_COMMAND else "inprocess")).lower()
INDEX_MODES = ("inprocess", "command", "off")
LOCK_RETRY_SECONDS = float(os.environ.get("VAULT_INDEX_LOCK_RETRY", "5"))
LOCK_RETRY_MAX = 60

mcp = FastMCP("vault")
_embed_model = None
_reranker = None

# Model idle auto-unload management (drops RAM footprint when idle)
MODEL_IDLE_TIMEOUT = int(os.environ.get("MODEL_IDLE_TIMEOUT", "300"))
_unload_timer: threading.Timer | None = None
_model_lock = threading.Lock()
_write_lock = threading.Lock()
_last_search_at: float | None = None


def _unload_models():
    global _embed_model, _reranker, _unload_timer
    if _worker.busy:  # the indexer thread holds the model; unloading now would force a second copy
        schedule_model_unload()
        return
    with _model_lock:
        if _embed_model is None and _reranker is None:
            _unload_timer = None
            return
        _embed_model = None
        _reranker = None
        _unload_timer = None
    gc.collect()
    with contextlib.suppress(OSError, AttributeError):  # malloc_trim is glibc-only
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    print("[vault-mcp] Models unloaded due to idle timeout; RAM trimmed to OS.", file=sys.stderr)


def schedule_model_unload():
    global _unload_timer, _last_search_at
    _last_search_at = time.time()
    with _model_lock:
        if _unload_timer is not None:
            _unload_timer.cancel()
        _unload_timer = threading.Timer(MODEL_IDLE_TIMEOUT, _unload_models)
        _unload_timer.daemon = True
        _unload_timer.start()


# In-memory LRU cache for search results. Keys include the index version
# (DB + WAL mtime), so any indexer commit invalidates cached results.
_SEARCH_CACHE_MAX_SIZE = 128
_SEARCH_CACHE_TTL = 300
_search_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()


def cache_get(key: tuple):
    with _cache_lock:
        if key in _search_cache:
            val, ts = _search_cache[key]
            if time.time() - ts < _SEARCH_CACHE_TTL:
                _search_cache.move_to_end(key)
                return val
            del _search_cache[key]
    return None


def cache_set(key: tuple, val: str):
    with _cache_lock:
        if key in _search_cache:
            del _search_cache[key]
        elif len(_search_cache) >= _SEARCH_CACHE_MAX_SIZE:
            _search_cache.popitem(last=False)
        _search_cache[key] = (val, time.time())


def cache_clear():
    with _cache_lock:
        _search_cache.clear()


def index_version() -> float:
    """Latest modification time of the index DB or its WAL (changes on every indexer commit)."""
    stamps = []
    for p in (DB_PATH, Path(f"{DB_PATH}-wal")):
        with contextlib.suppress(OSError):
            stamps.append(p.stat().st_mtime)
    return max(stamps) if stamps else 0.0


# ---------------------------------------------------------------------------
# Re-indexing
# ---------------------------------------------------------------------------

class IndexWorker:
    """Background re-index thread that reuses this process's embedding model.

    `request()` never blocks. Requests arriving during a pass mark the worker dirty, so
    exactly one more pass runs afterwards instead of the change being dropped.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._dirty = False
        self._thread: threading.Thread | None = None
        self.busy = False
        self.last_stats: dict | None = None
        self.last_error: str | None = None
        self.last_run_at: float | None = None

    def request(self) -> None:
        with self._lock:
            self._dirty = True
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="vault-indexer", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        retries = 0
        while True:
            with self._lock:
                if not self._dirty:
                    self._thread = None
                    return
                self._dirty = False
            self.busy = True
            try:
                stats = indexer.build_index(VAULT_PATH, DB_PATH, embedder_factory=lambda: get_embed_model())
                if stats.get("locked"):
                    retries += 1
                    if retries <= LOCK_RETRY_MAX:
                        with self._lock:
                            self._dirty = True
                        time.sleep(LOCK_RETRY_SECONDS)
                    else:
                        self.last_error = "index lock held by another indexer; gave up retrying"
                else:
                    retries = 0
                    self.last_stats, self.last_error = stats, None
            except Exception as e:  # noqa: BLE001 - keep the server alive; surfaced in vault_status
                self.last_error = f"{type(e).__name__}: {e}"
            finally:
                self.busy = False
                self.last_run_at = time.time()
                cache_clear()

    def wait(self, timeout: float = 30.0) -> bool:
        """Block until idle (tests / shutdown)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._thread is None:
                    return True
            time.sleep(0.02)
        return False


_worker = IndexWorker()


def _indexer_command() -> list:
    return shlex.split(INDEXER_COMMAND)


def trigger_background_reindex() -> str:
    """Queue an incremental re-index according to INDEX_MODE; returns a short status line."""
    if INDEX_MODE == "inprocess":
        _worker.request()
        return "Re-index queued (in-process)."
    if INDEX_MODE == "command":
        if not INDEXER_COMMAND:
            return "Re-index NOT triggered: VAULT_INDEX_MODE=command but VAULT_INDEXER_COMMAND is empty."
        # If an indexer is already running, it sees the flag and makes another pass.
        with contextlib.suppress(OSError):
            indexer.dirty_path(DB_PATH).touch()
        try:
            subprocess.Popen(  # noqa: S603 - operator-configured command, not user input
                _indexer_command(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
            )
        except OSError as e:
            return f"Re-index command failed to start ({e}); the dirty flag is set for the next run."
        return "Re-index triggered (command)."
    return "Re-index skipped (VAULT_INDEX_MODE=off)."


def indexer_running() -> bool:
    """In-process worker busy, or an external python indexer process alive (Linux /proc)."""
    if _worker.busy:
        return True
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if not argv or b"python" not in Path(argv[0].decode(errors="ignore")).name.encode():
            continue
        joined = b" ".join(argv)
        if b"vault-indexer" in joined or b".indexer" in joined:
            return True
    return False


# ---------------------------------------------------------------------------
# Models & DB
# ---------------------------------------------------------------------------

def get_embed_model():
    global _embed_model
    with _model_lock:
        if _embed_model is None:
            _embed_model = indexer.load_embedder()  # same seq-length cap as the CLI indexer
    schedule_model_unload()
    return _embed_model


def get_reranker():
    global _reranker
    with _model_lock:
        if _reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _reranker = TextCrossEncoder(
                RERANK_MODEL_NAME,
                cache_dir=str(FASTEMBED_CACHE_DIR),
                local_files_only=True,
                threads=RERANK_THREADS,
            )
    schedule_model_unload()
    return _reranker


def get_db_connection(db_path: Path | None = None):
    import sqlite_vec
    target_path = db_path or DB_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Vault index database not found at {target_path}")
    conn = sqlite3.connect(f"file:{target_path.as_posix()}?mode=ro", uri=True)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


def get_db(db_path_str: str):
    return get_db_connection(Path(db_path_str).expanduser().resolve())


serialize_vector = serialize_f32 = indexer.serialize_f32


# ---------------------------------------------------------------------------
# Note helpers
# ---------------------------------------------------------------------------

def read_note(path: Path) -> str:
    """Note text with original line endings preserved (no universal-newline translation)."""
    return path.read_bytes().decode("utf-8", errors="replace")


def update_frontmatter_timestamp(text: str) -> str:
    """Set `updated:` in existing frontmatter; notes without frontmatter are left untouched."""
    lines = text.split("\n")
    span = frontmatter_span(lines)
    if not span:
        return text
    stamp = f'updated: "{time.strftime("%Y-%m-%d %H:%M:%S")}"'
    for i in range(1, span - 1):
        if lines[i].startswith("updated:"):
            lines[i] = stamp + ("\r" if lines[i].endswith("\r") else "")
            return "\n".join(lines)
    eol = "\r" if lines[span - 1].endswith("\r") else ""
    lines.insert(span - 1, stamp + eol)
    return "\n".join(lines)


def _check_hash(path: Path, expected_hash: str) -> str | None:
    if expected_hash and not hash_matches(file_hash(path), expected_hash):
        return ("Error: note changed since you read it (expected_hash mismatch). "
                "Re-read it with vault_read and retry.")
    return None


def _after_change(rel: str, content: str) -> str:
    cache_clear()
    return f"{trigger_background_reindex()}\n{link_report(VAULT_PATH, content, rel)}"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

# Last model failure per stage ("ExcType: message"), cleared on the next success.
# search() falls back silently, so these are what tell the agent and vault_status.
_model_errors: dict[str, str] = {}
_call = threading.local()  # failures during the current vault_search call


def _tracked(stage: str, fn):
    try:
        result = fn()
    except Exception as e:
        _model_errors[stage] = f"{type(e).__name__}: {e}"
        getattr(_call, "failures", {})[stage] = _model_errors[stage]
        raise
    _model_errors.pop(stage, None)
    return result


def _embed_query(query: str) -> bytes:
    return _tracked("embedder", lambda: serialize_f32(
        get_embed_model().encode([query], normalize_embeddings=True)[0]))


def _rerank(query: str, docs: list[str]) -> list[float]:
    return _tracked("reranker", lambda: list(get_reranker().rerank(query, docs)))


def _degradation_note(failures: dict[str, str]) -> str:
    """Footer for results produced without a model the requested mode needed."""
    notes = []
    if "embedder" in failures:
        notes.append(f"semantic search unavailable ({failures['embedder']}); keyword results only")
    if "reranker" in failures:
        notes.append(f"reranker unavailable ({failures['reranker']}); RRF order")
    return ("\n\n> ⚠️ " + " · ".join(notes)) if notes else ""


@mcp.tool()
def vault_search(query: str, limit: int = 5, mode: str = "hybrid", folder: str = "", tags: str = "",
                 status: str = "") -> str:
    """Search the Obsidian vault by meaning and keywords. Prefer this over grep for concept questions.

    Two-tier hybrid retrieval: BM25 (FTS5) + BGE-M3 vectors, fused with RRF and
    reranked by a cross-encoder. At most 2 chunks per note are returned.

    Args:
        query: Search keywords or natural language question (Indonesian or English).
        limit: Max results to return (default 5, max 15).
        mode: 'hybrid' (RRF + cross-encoder), 'keyword' (FTS5 BM25 only), or 'semantic' (BGE-M3 vector only).
        folder: Optional folder filter within the vault (e.g. 'server', 'projects/cuantum').
        tags: Optional comma-separated frontmatter tags; notes must have all of them (e.g. 'cuantum, quant').
        status: Optional comma-separated frontmatter status values; notes must match one (e.g. 'verified').
    """
    limit = max(1, min(limit, 15))
    if mode not in ("hybrid", "keyword", "semantic"):
        mode = "hybrid"
    cache_key = (query.strip().lower(), limit, mode, folder.strip("/ "), tags.lower(), status.lower(),
                 index_version())
    cached_res = cache_get(cache_key)
    if cached_res is not None:
        if _embed_model is not None or _reranker is not None:
            schedule_model_unload()
        return cached_res

    _call.failures = {}
    conn = get_db_connection()
    try:
        outcome = search(
            conn, query, limit=limit, mode=mode, folder=folder, tags=tags, status=status,
            embed_fn=_embed_query, rerank_fn=_rerank, per_note=MAX_CHUNKS_PER_NOTE,
            min_score=MIN_RERANK_SCORE, rerank_chars=RERANK_CHARS, rerank_pool=RERANK_POOL,
        )
    finally:
        conn.close()
    failures, _call.failures = _call.failures, {}
    note = _degradation_note(failures)

    if not outcome.hits:
        if outcome.best_rejected_score is not None:
            res = (f"No sufficiently relevant notes for '{query}' (best rerank score "
                   f"{outcome.best_rejected_score:.2f} < floor {MIN_RERANK_SCORE}). Try other words or mode='keyword'.")
        else:
            res = f"No matching notes found in vault for query: '{query}'."
        if note:  # degraded answers are not cached: a recovered model answers properly next time
            return res + note
        cache_set(cache_key, res)
        return res

    results = []
    for rank, h in enumerate(outcome.hits, start=1):
        snippet = h.chunk_text.strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        score_label = f"Score: {h.score:.3f}" if h.reranked else f"RRF: {h.score:.4f}"
        header = f"### [{rank}] [[{Path(h.rel_path).stem}]] > {h.heading} ({score_label})"
        meta = f"File: `{h.rel_path}` (Lines {h.line_start}-{h.line_end})"
        results.append(f"{header}\n{meta}\n\n{snippet}\n")

    res = "\n---\n".join(results)
    if note:
        res += note
    else:
        cache_set(cache_key, res)
    if _embed_model is not None or _reranker is not None:
        schedule_model_unload()
    return res


# ---------------------------------------------------------------------------
# Read / list / backlinks / recent
# ---------------------------------------------------------------------------

@mcp.tool()
def vault_read(rel_path: str, heading: str = "", start_line: int = 1, max_chars: int = 0) -> str:
    """Read a note (or one heading section) in pages of at most `max_chars` characters.

    The header shows the line range, total lines and a sha256 prefix; pass that hash as
    `expected_hash` to the write tools to refuse edits if the note changed meanwhile.

    Args:
        rel_path: Relative path ('server/homeserver-setup.md') or a unique note name ('homeserver-setup').
        heading: Optional heading text to return only that section (fenced code is never a heading).
        start_line: 1-based line to start from (use the value suggested at the end of a truncated page).
        max_chars: Page size (default 8000, max 50000).
    """
    target_file, clean_rel, error = resolve_note(VAULT_PATH, rel_path, must_exist=True)
    if error:
        return error
    try:
        content = read_note(target_file)
    except OSError as e:
        return f"Error reading file: {e}"
    max_chars = max(500, min(max_chars or READ_MAX_CHARS, 50000))
    lines = content.splitlines()
    lo, hi = 0, len(lines)
    if heading:
        rng = extract_section(lines, heading)
        if rng is None:
            heads = list_headings(lines)
            shown = "\n".join(f"- {h}" for h in heads[:60]) or "(no headings)"
            return f"Heading `{heading}` not found in `{clean_rel}`. Available headings:\n{shown}"
        lo, hi = rng
    lo = max(lo, start_line - 1)
    if lo >= hi:
        return f"`{clean_rel}` has {len(lines)} lines; start_line={start_line} is past the end of the selection."

    out, used, end = [], 0, lo
    while end < hi:
        line = lines[end]
        if out and used + len(line) + 1 > max_chars:
            break
        if not out and len(line) > max_chars:
            out.append(line[:max_chars] + f" … [line truncated, {len(line) - max_chars} more chars]")
            end += 1
            break
        out.append(line)
        used += len(line) + 1
        end += 1
    digest = file_hash(target_file)[:16]
    scope = f" > `{heading}`" if heading else ""
    header = f"# File: `{clean_rel}`{scope} — lines {lo + 1}-{end} of {len(lines)} · sha256 {digest}"
    body = "\n".join(out)
    if end < hi:
        remaining = sum(len(ln) + 1 for ln in lines[end:hi])
        more = f'heading="{heading}", ' if heading else ""
        body += (f"\n\n… [truncated: {remaining} more chars in lines {end + 1}-{hi}]. "
                 f"Continue with vault_read({more}start_line={end + 1}).")
    return f"{header}\n\n{body}"


@mcp.tool()
def vault_list(folder: str = "", limit: int = 200) -> str:
    """List notes in the vault (or one folder, recursively) with their titles. Use instead of ls/find.

    Args:
        folder: Optional folder within the vault (e.g. 'projects/cuantum'). Empty lists the whole vault.
        limit: Max notes to list (default 200, max 1000).
    """
    limit = max(1, min(limit, 1000))
    folder_clean = folder.strip().replace("\\", "/").strip("/")
    if folder_clean:
        base = (VAULT_PATH / folder_clean).resolve()
        if not base.is_relative_to(VAULT_PATH) or not base.is_dir():
            return f"Error: folder `{folder}` not found in vault."
    titles = {}
    with contextlib.suppress(sqlite3.Error, FileNotFoundError):  # titles are optional; fall back to stems
        conn = get_db_connection()
        titles = {r["rel_path"]: r["title"] for r in conn.execute("SELECT rel_path, title FROM notes")}
        conn.close()

    prefix = f"{folder_clean}/" if folder_clean else ""
    notes = [rel for _, rel in iter_notes(VAULT_PATH) if rel.startswith(prefix)]
    lines = [f"- `{rel}` — {titles.get(rel, Path(rel).stem)}" for rel in notes[:limit]]
    more = f"\n… {len(notes) - limit} more (raise limit or narrow folder)" if len(notes) > limit else ""
    scope = f"`{folder_clean}/`" if folder_clean else "vault"
    return f"### {len(notes)} notes in {scope}\n" + "\n".join(lines) + more


@mcp.tool()
def vault_backlinks(note: str, limit: int = 50) -> str:
    """List notes that link to a note via [[wikilinks]], with the linking line.

    Args:
        note: Note path or name (e.g. 'server/homeserver-setup.md' or 'homeserver-setup').
        limit: Max backlinks to return (default 50).
    """
    _target_file, target_rel, error = resolve_note(VAULT_PATH, note, must_exist=True)
    if error:
        return error
    hits = backlinks(VAULT_PATH, target_rel, limit)
    if not hits:
        return f"No backlinks found for `{target_rel}` (orphan note)."
    return f"### Backlinks to `{target_rel}` ({len(hits)})\n" + "\n".join(hits)


@mcp.tool()
def vault_recent(limit: int = 20, folder: str = "", days: int = 0) -> str:
    """List the most recently modified notes (newest first).

    Args:
        limit: Max notes (default 20, max 200).
        folder: Optional folder filter (e.g. 'projects/cuantum').
        days: Only notes modified in the last N days (0 = no limit).
    """
    limit = max(1, min(limit, 200))
    prefix = folder.strip().replace("\\", "/").strip("/")
    prefix = f"{prefix}/" if prefix else ""
    cutoff = time.time() - days * 86400 if days > 0 else 0
    entries = []
    for full, rel in iter_notes(VAULT_PATH):
        if not rel.startswith(prefix):
            continue
        with contextlib.suppress(OSError):
            mtime = full.stat().st_mtime
            if mtime >= cutoff:
                entries.append((mtime, rel))
    entries.sort(reverse=True)
    if not entries:
        return "No notes match."
    lines = [f"- {time.strftime('%Y-%m-%d %H:%M', time.localtime(m))} `{rel}`" for m, rel in entries[:limit]]
    return f"### {min(limit, len(entries))} of {len(entries)} recently modified notes\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Write / append / edit / move
# ---------------------------------------------------------------------------

@mcp.tool()
def vault_write(rel_path: str, content: str, title: str = "", tags: str = "", overwrite: bool = False,
                expected_hash: str = "") -> str:
    """Create a markdown note (or replace one with overwrite=true; the old version is backed up to
    .trash/vault-mcp/). Adds frontmatter, reports broken/missing hub [[wikilinks]] and re-indexes.

    Args:
        rel_path: Note path within the vault (e.g. 'projects/my-idea.md').
        content: Markdown content. Link related notes and the hub with [[wikilinks]].
        title: Optional frontmatter title (defaults to file stem).
        tags: Optional comma-separated tags (e.g. 'project, quant, research').
        overwrite: Replace an existing note. Prefer vault_edit / vault_append for partial changes.
        expected_hash: Optional sha256 prefix from vault_read; refuses to overwrite a note changed since.
    """
    target_file, clean_rel, error = resolve_note(VAULT_PATH, rel_path, must_exist=False)
    if error:
        return error

    final_title = title.strip() or target_file.stem
    tag_list = [t.strip().lstrip("#") for t in tags.split(",") if t.strip()] if tags else []
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    if not content.startswith("---"):
        fm_lines = ["---", f'title: "{final_title}"']
        if tag_list:
            fm_lines.append(f"tags: [{', '.join(tag_list)}]")
        fm_lines += [f'created: "{now_str}"', f'updated: "{now_str}"', "---\n"]
        full_content = "\n".join(fm_lines) + content.lstrip()
    else:
        full_content = update_frontmatter_timestamp(content)

    backup = ""
    with _write_lock:
        if target_file.exists():
            if not overwrite:
                return (f"Error: `{clean_rel}` already exists. Use vault_edit / vault_append for changes, "
                        "or overwrite=true to replace it (a backup is kept).")
            if err := _check_hash(target_file, expected_hash):
                return err
            try:
                backup = backup_note(VAULT_PATH, clean_rel)
            except OSError as e:
                return f"Error: could not back up `{clean_rel}` before overwriting: {e}"
        try:
            atomic_write(target_file, full_content)
        except OSError as e:
            return f"Error writing file `{clean_rel}`: {e}"
    note = f" Previous version backed up to `{backup}`." if backup else ""
    report = _after_change(clean_rel, full_content)
    return f"Successfully wrote `{clean_rel}` ({len(full_content)} bytes).{note}\n{report}"


@mcp.tool()
def vault_append(rel_path: str, content: str, heading: str = "", expected_hash: str = "") -> str:
    """Append content to an existing note, optionally under a heading (created at the end if missing).
    Reports broken/missing hub [[wikilinks]] and re-indexes.

    Args:
        rel_path: Note path or unique note name.
        content: Text to append. Use [[wikilinks]] for connected concepts.
        heading: Optional heading under which to append.
        expected_hash: Optional sha256 prefix from vault_read; refuses if the note changed since.
    """
    target_file, clean_rel, error = resolve_note(VAULT_PATH, rel_path, must_exist=True)
    if error:
        return error
    appended_text = content.strip()
    with _write_lock:
        if err := _check_hash(target_file, expected_hash):
            return err
        try:
            existing = read_note(target_file)
        except OSError as e:
            return f"Error reading file `{clean_rel}`: {e}"
        lines = existing.split("\n")
        rng = extract_section(lines, heading) if heading else None
        if rng is not None:
            start, end = rng
            insert_at = end
            while insert_at > start + 1 and not lines[insert_at - 1].strip():
                insert_at -= 1
            block = ["", appended_text]
            if insert_at < len(lines) and lines[insert_at].strip():
                block.append("")
            lines[insert_at:insert_at] = block
            new_content = "\n".join(lines)
            if not new_content.endswith("\n"):
                new_content += "\n"
        elif heading:
            new_content = existing.rstrip() + f"\n\n## {heading}\n\n" + appended_text + "\n"
        else:
            new_content = existing.rstrip() + "\n\n" + appended_text + "\n"
        new_content = update_frontmatter_timestamp(new_content)
        try:
            atomic_write(target_file, new_content)
        except OSError as e:
            return f"Error updating file `{clean_rel}`: {e}"
    report = _after_change(clean_rel, new_content)
    return f"Successfully appended to `{clean_rel}`.\n{report}"


@mcp.tool()
def vault_edit(rel_path: str, old_text: str, new_text: str, replace_all: bool = False,
               expected_hash: str = "") -> str:
    """Replace an exact text span in a note (use instead of patch/sed on vault files).
    Fails if old_text is missing, or appears more than once without replace_all.
    Reports broken/missing hub [[wikilinks]] and re-indexes.

    Args:
        rel_path: Note path or unique note name.
        old_text: Exact text to replace (include enough context to be unique).
        new_text: Replacement text.
        replace_all: Replace every occurrence instead of requiring a unique match.
        expected_hash: Optional sha256 prefix from vault_read; refuses if the note changed since.
    """
    target_file, clean_rel, error = resolve_note(VAULT_PATH, rel_path, must_exist=True)
    if error:
        return error
    if not old_text:
        return "Error: old_text must not be empty."
    with _write_lock:
        if err := _check_hash(target_file, expected_hash):
            return err
        try:
            existing = read_note(target_file)
        except OSError as e:
            return f"Error reading file `{clean_rel}`: {e}"
        # Agents send LF; match CRLF notes too.
        if "\r\n" in existing and "\r\n" not in old_text:
            old_text, new_text = old_text.replace("\n", "\r\n"), new_text.replace("\n", "\r\n")
        count = existing.count(old_text)
        if count == 0:
            return f"Error: old_text not found in `{clean_rel}`. Re-read the note and copy the exact text."
        if count > 1 and not replace_all:
            return f"Error: old_text appears {count} times in `{clean_rel}`. Add more context or set replace_all=true."
        new_content = existing.replace(old_text, new_text) if replace_all else existing.replace(old_text, new_text, 1)
        new_content = update_frontmatter_timestamp(new_content)
        try:
            atomic_write(target_file, new_content)
        except OSError as e:
            return f"Error updating file `{clean_rel}`: {e}"
    report = _after_change(clean_rel, new_content)
    replaced = count if replace_all else 1
    return f"Successfully edited `{clean_rel}` ({replaced} replacement(s)).\n{report}"


@mcp.tool()
def vault_move(src: str, dst: str, update_links: bool = True) -> str:
    """Move/rename a note and rewrite every [[wikilink]] that pointed at it (aliases, #headings
    and ![[embeds]] are kept). Use this instead of creating a copy, so no links break.

    Args:
        src: Current note path or unique note name.
        dst: New path within the vault (e.g. 'projects/cuantum/plans/plan-x.md').
        update_links: Rewrite links in other notes (default true).
    """
    src_file, src_rel, error = resolve_note(VAULT_PATH, src, must_exist=True)
    if error:
        return error
    dst_file, dst_rel, error = resolve_note(VAULT_PATH, dst, must_exist=False)
    if error:
        return error
    if dst_file.exists():
        return f"Error: `{dst_rel}` already exists."
    with _write_lock:
        before = LinkIndex(VAULT_PATH)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(src_file, dst_file)
        except OSError as e:
            return f"Error moving `{src_rel}` to `{dst_rel}`: {e}"
        files_changed = links_changed = 0
        failed = []
        if update_links:
            after = LinkIndex(VAULT_PATH)
            for full, rel in iter_notes(VAULT_PATH):
                try:
                    text = read_note(full)
                except OSError:
                    continue
                if "[[" not in text:
                    continue
                new_text, n = rewrite_links(text, src_rel, dst_rel, before, after)
                if n:
                    try:
                        atomic_write(full, new_text)
                    except OSError:
                        failed.append(rel)
                        continue
                    files_changed += 1
                    links_changed += n
    cache_clear()
    status = trigger_background_reindex()
    msg = f"Moved `{src_rel}` → `{dst_rel}`. Rewrote {links_changed} link(s) in {files_changed} note(s)."
    if failed:
        msg += " Failed to update: " + ", ".join(f"`{f}`" for f in failed)
    return f"{msg}\n{status}"


# ---------------------------------------------------------------------------
# Lint / status
# ---------------------------------------------------------------------------

@mcp.tool()
def vault_lint(folder: str = "", limit: int = 30) -> str:
    """Audit vault hygiene: broken [[links]], orphan notes (no inbound links), notes without a hub
    link or frontmatter, frontmatter `status` outside the allowed set, oversized notes and blob lines.

    Args:
        folder: Optional folder to report on (the link graph always spans the whole vault).
        limit: Max items listed per finding type (default 30).
    """
    r = lint_vault(VAULT_PATH, folder, STATUS_VALUES, indexer.MAX_FILE_BYTES, indexer.MAX_LINE_CHARS)

    def section(title: str, items: list) -> str:
        if not items:
            return ""
        shown = "\n".join(f"- {i}" for i in items[:limit])
        more = f"\n- … {len(items) - limit} more" if len(items) > limit else ""
        return f"\n#### {title} ({len(items)})\n{shown}{more}\n"

    body = "".join([
        section("Broken links", [f"`{n}` → [[{t}]]" for n, t in r.broken]),
        section("Orphans (no inbound links)", [f"`{n}`" for n in r.orphans]),
        section("No hub link", [f"`{n}`" for n in r.no_hub]),
        section("No frontmatter", [f"`{n}`" for n in r.no_frontmatter]),
        section(f"Status not in {{{', '.join(sorted(STATUS_VALUES))}}}", [f"`{n}`: {s}" for n, s in r.bad_status]),
        section("Oversized (skipped by indexer)", [f"`{n}` ({b // 1024} KB)" for n, b in r.oversized]),
        section("Blob lines (dropped from index)", [f"`{n}` (longest line {c} chars)" for n, c in r.junk_lines]),
    ])
    scope = f"`{folder.strip('/')}/`" if folder.strip("/") else "vault"
    return f"### Vault lint — {r.total} notes in {scope}\n" + (body or "\nNo issues found.\n")


def index_freshness():
    """Compare vault files to the index: (pending_rel_paths, deleted_rel_paths)."""
    conn = get_db_connection()
    indexed = {r["rel_path"]: r["mtime"] for r in conn.execute("SELECT rel_path, mtime FROM notes")}
    conn.close()
    pending, current = [], set()
    for full, rel in iter_notes(VAULT_PATH):
        current.add(rel)
        try:
            mtime = full.stat().st_mtime
        except OSError:
            continue
        if rel not in indexed or abs(indexed[rel] - mtime) > 0.001:
            pending.append(rel)
    deleted = [rel for rel in indexed if rel not in current]
    return pending, deleted


@mcp.tool()
def vault_status() -> str:
    """Report index health: note/chunk counts, notes waiting to be indexed, skipped notes, indexer
    activity, model RAM state."""
    if not DB_PATH.exists():
        return f"Database not found at `{DB_PATH}`. Run the indexer first."

    conn = get_db_connection()
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version != indexer.SCHEMA_VERSION:
        conn.close()
        return (f"### Vault Index Status\n- **Health:** OUTDATED SCHEMA — index v{version}, server expects "
                f"v{indexer.SCHEMA_VERSION}. Run the indexer once (it rebuilds automatically).")
    notes_count = conn.execute("SELECT count(*) FROM notes WHERE skip_reason IS NULL").fetchone()[0]
    skipped = conn.execute("SELECT rel_path, skip_reason FROM notes WHERE skip_reason IS NOT NULL").fetchall()
    chunks_count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    vec_count = conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
    conn.close()
    pending, deleted = index_freshness()
    running = indexer_running()
    last_write = index_version()
    age_min = (time.time() - last_write) / 60 if last_write else None
    db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)

    if running:
        health = f"INDEXING ({len(pending)} note(s) still pending)"
    elif pending or deleted:
        health = (f"STALE — {len(pending)} changed/new and {len(deleted)} deleted note(s) not in index; "
                  "search may miss them")
    elif chunks_count != vec_count:
        health = f"INCONSISTENT — {chunks_count} chunks vs {vec_count} vectors (run a rebuild)"
    else:
        health = "OK — index matches vault"

    model_loaded = _embed_model is not None or _reranker is not None
    if model_loaded and _last_search_at:
        remaining = max(0, int(MODEL_IDLE_TIMEOUT - (time.time() - _last_search_at)))
        model_status_str = f"loaded (auto-unload in {remaining}s)"
    else:
        model_status_str = "unloaded (next semantic search pays a cold-load)"

    pending_list = ""
    if pending and not running:
        shown = ", ".join(f"`{p}`" for p in pending[:10])
        pending_list = f"\n- **Pending notes:** {shown}{' …' if len(pending) > 10 else ''}"
    skipped_list = ""
    if skipped:
        shown = "; ".join(f"`{r['rel_path']}` ({r['skip_reason']})" for r in skipped[:10])
        skipped_list = f"\n- **Skipped ({len(skipped)}):** {shown}{' …' if len(skipped) > 10 else ''}"
    worker = ""
    if INDEX_MODE == "inprocess" and (_worker.last_error or _worker.last_stats):
        worker = f"\n- **Last in-process index:** {_worker.last_error or _worker.last_stats}"
    last_write_str = f"{age_min:.0f} min ago" if age_min is not None else "never"
    floor = "off" if MIN_RERANK_SCORE is None else MIN_RERANK_SCORE
    model_errors = "".join(
        f"\n- ⚠️ {label}: unavailable — {_model_errors[stage]}"
        for stage, label in (("embedder", "Embedder"), ("reranker", "Reranker")) if stage in _model_errors
    )

    return f"""### Vault Index Status
- **Health:** {health}
- **Indexed:** {notes_count} notes, {chunks_count} chunks, {vec_count} vectors ({db_size_mb:.1f} MB){skipped_list}
- **Last index write:** {last_write_str}
- **Indexer running:** {'yes' if running else 'no'} (mode: {INDEX_MODE}){pending_list}{worker}
- **Models:** `{EMBED_MODEL_NAME}` + `{RERANK_MODEL_NAME}` — {model_status_str}; rerank floor: {floor}{model_errors}
- **Search cache:** {len(_search_cache)}/{_SEARCH_CACHE_MAX_SIZE}
"""


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def validate_config(transport: str, host: str, allow_remote: bool) -> str | None:
    """Return an error message for unsafe or incomplete configuration, else None."""
    if INDEX_MODE not in INDEX_MODES:
        return f"VAULT_INDEX_MODE must be one of {INDEX_MODES}, got '{INDEX_MODE}'."
    if INDEX_MODE == "command" and not INDEXER_COMMAND:
        return "VAULT_INDEX_MODE=command requires VAULT_INDEXER_COMMAND (e.g. a memory-capped wrapper script)."
    if transport != "stdio" and host not in LOOPBACK_HOSTS and not allow_remote:
        return (f"Refusing to bind {host}: the write tools have no authentication. "
                "Bind 127.0.0.1, or pass --allow-remote behind your own auth proxy.")
    return None


def main(argv: list[str] | None = None):
    import argparse
    parser = argparse.ArgumentParser(description="Obsidian Hybrid RAG FastMCP server")
    parser.add_argument("--transport", default=os.environ.get("MCP_TRANSPORT", "stdio"),
                        choices=["stdio", "sse", "http", "streamable-http"])
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"),
                        help="Bind address for network transports. Keep 127.0.0.1: write tools have no auth.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", "8765")))
    parser.add_argument("--allow-remote", action="store_true",
                        default=os.environ.get("MCP_ALLOW_REMOTE", "") == "1",
                        help="Allow binding a non-loopback address (only behind an authenticating proxy).")
    args = parser.parse_args(argv)

    if error := validate_config(args.transport, args.host, args.allow_remote):
        print(f"[vault-mcp] {error}", file=sys.stderr)
        sys.exit(2)
    if args.transport != "stdio":
        mcp.run(transport=args.transport, host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
