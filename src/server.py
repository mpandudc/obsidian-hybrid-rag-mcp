#!/usr/bin/env python3
"""
FastMCP server for hybrid retrieval and safe editing of an Obsidian vault.

Search: SQLite FTS5 (BM25) + sqlite-vec dense vectors (BAAI/bge-m3), fused with
Reciprocal Rank Fusion and reranked by a Jina v2 cross-encoder.
Editing: write / append / exact-string edit, all confined to the vault, with a
wikilink report and a background re-index after every change.

The file is self-contained (no package imports) so it can also be deployed as a
single script. All paths are configurable through environment variables.
"""

import contextlib
import ctypes
import gc
import os
import re
import shlex
import shutil
import sqlite3
import struct
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

from fastmcp import FastMCP


def _env_path(*names: str, default: Path) -> Path:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser().resolve()
    return default.expanduser().resolve()


# Paths & Models
VAULT_PATH = _env_path("OBSIDIAN_VAULT_PATH", "VAULT_PATH", default=Path.home() / "vaults" / "pandu-second-brain")
DB_PATH = _env_path("VAULT_INDEX_DB", "INDEX_DB_PATH", default=Path.home() / ".hermes" / "vault-index.db")
FASTEMBED_CACHE_DIR = _env_path("FASTEMBED_CACHE_DIR", default=Path.home() / ".cache" / "fastembed")
INDEXER_LOCK = Path(os.environ.get("VAULT_INDEXER_LOCK", "/tmp/vault-indexer.lock"))
# Command that runs one incremental index pass (e.g. a memory-capped wrapper script).
INDEXER_COMMAND = os.environ.get("VAULT_INDEXER_COMMAND", "")
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024
RERANK_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
MAX_CHUNKS_PER_NOTE = int(os.environ.get("VAULT_MAX_CHUNKS_PER_NOTE", "2"))
HIDDEN_DIRS = {".git", ".obsidian", ".trash", ".templates", ".smart-env", ".trash-bin"}

mcp = FastMCP("vault")
_embed_model = None
_reranker = None

# Model idle auto-unload management (drops RAM footprint when idle)
MODEL_IDLE_TIMEOUT = int(os.environ.get("MODEL_IDLE_TIMEOUT", "300"))
_unload_timer: threading.Timer | None = None
_model_lock = threading.Lock()
_last_search_at: float | None = None


def _unload_models():
    global _embed_model, _reranker, _unload_timer
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
_search_cache = OrderedDict()


def cache_get(key: tuple):
    if key in _search_cache:
        val, ts = _search_cache[key]
        if time.time() - ts < _SEARCH_CACHE_TTL:
            _search_cache.move_to_end(key)
            return val
        del _search_cache[key]
    return None


def cache_set(key: tuple, val: str):
    if key in _search_cache:
        del _search_cache[key]
    elif len(_search_cache) >= _SEARCH_CACHE_MAX_SIZE:
        _search_cache.popitem(last=False)
    _search_cache[key] = (val, time.time())


def cache_clear():
    _search_cache.clear()


def index_version() -> float:
    """Latest modification time of the index DB or its WAL (changes on every indexer commit)."""
    stamps = []
    for p in (DB_PATH, Path(f"{DB_PATH}-wal")):
        try:
            stamps.append(p.stat().st_mtime)
        except OSError:
            pass
    return max(stamps) if stamps else 0.0


def _indexer_command() -> list:
    if INDEXER_COMMAND:
        return shlex.split(INDEXER_COMMAND)
    cmd = [sys.executable, "-m", "src.indexer", "--vault-path", str(VAULT_PATH), "--db-path", str(DB_PATH)]
    if shutil.which("flock"):
        cmd = ["flock", "-n", str(INDEXER_LOCK)] + cmd
    return cmd


def trigger_background_reindex():
    """Trigger a non-blocking incremental indexer run."""
    with contextlib.suppress(OSError):  # missing indexer command: cron/next write retries
        subprocess.Popen(
            _indexer_command(),
            cwd=str(Path(__file__).resolve().parent.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def indexer_running() -> bool:
    """Detect a running indexer process (python running the indexer) without touching its lock."""
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if not argv or b"python" not in Path(argv[0].decode(errors="ignore")).name.encode():
            continue
        joined = b" ".join(argv)
        if b"vault-indexer" in joined or b"src.indexer" in joined:
            return True
    return False


def get_embed_model():
    global _embed_model
    with _model_lock:
        if _embed_model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(4)
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu", local_files_only=True)
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
                threads=2,
            )
    schedule_model_unload()
    return _reranker


def get_db_connection(db_path: Path | None = None):
    import sqlite_vec
    target_path = db_path or DB_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Vault index database not found at {target_path}")
    conn = sqlite3.connect(f"file:{target_path}?mode=ro", uri=True)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


def get_db(db_path_str: str):
    return get_db_connection(Path(db_path_str).expanduser().resolve())


def serialize_vector(vec):
    return struct.pack(f"{len(vec)}f", *vec)


serialize_f32 = serialize_vector


# ---------------------------------------------------------------------------
# Vault path helpers
# ---------------------------------------------------------------------------

def _is_hidden(rel: Path) -> bool:
    return any(part.startswith(".") or part in HIDDEN_DIRS for part in rel.parts)


def iter_vault_notes():
    """Yield (absolute_path, rel_path_str) for every visible markdown note."""
    for root, dirs, files in os.walk(VAULT_PATH):
        dirs[:] = sorted(d for d in dirs if d not in HIDDEN_DIRS and not d.startswith("."))
        for name in sorted(files):
            if name.endswith(".md") and not name.startswith("."):
                full = Path(root) / name
                yield full, str(full.relative_to(VAULT_PATH))


def resolve_note(rel_path: str, must_exist: bool):
    """Map a user-supplied note path to a file inside the vault.

    Returns (path, rel_path, error). Paths that escape the vault are rejected.
    For must_exist lookups a bare note name is resolved by filename; an
    ambiguous name returns the candidate list instead of guessing.
    """
    clean = rel_path.strip().strip("/")
    if not clean:
        return None, None, "Error: empty note path."
    if not clean.endswith(".md"):
        clean += ".md"
    candidate = (VAULT_PATH / clean).resolve()
    if not candidate.is_relative_to(VAULT_PATH):
        return None, None, f"Error: Access denied. Path `{rel_path}` is outside the vault."
    if _is_hidden(candidate.relative_to(VAULT_PATH)):
        return None, None, f"Error: Access denied. `{rel_path}` is inside a hidden/system folder."
    if candidate.exists() or not must_exist:
        return candidate, str(candidate.relative_to(VAULT_PATH)), None

    name = Path(clean).name
    matches = [rel for _, rel in iter_vault_notes() if Path(rel).name == name]
    if len(matches) == 1:
        return (VAULT_PATH / matches[0]).resolve(), matches[0], None
    if len(matches) > 1:
        listing = "\n".join(f"- `{m}`" for m in matches)
        return None, None, f"Error: `{rel_path}` is ambiguous. Use one of these paths:\n{listing}"
    return None, None, f"Error: Note `{rel_path}` not found in vault."


WIKILINK_RE = re.compile(r"!?\[\[([^\]|#^]+)(?:[#^][^\]|]*)?(?:\|[^\]]*)?\]\]")


def _link_targets(content: str) -> list:
    seen = []
    for m in WIKILINK_RE.finditer(content):
        target = m.group(1).strip()
        if target and target not in seen:
            seen.append(target)
    return seen


def _is_hub(target: str) -> bool:
    stem = Path(target).name.lower()
    return stem in ("readme", "index") or stem.endswith("index") or "overview" in stem


def link_report(content: str, rel_path: str) -> str:
    """Report broken wikilinks and a missing hub link (vault convention: no orphan notes)."""
    targets = _link_targets(content)
    stems, paths, files = set(), set(), set()
    for root, dirs, names in os.walk(VAULT_PATH):
        dirs[:] = [d for d in dirs if d not in HIDDEN_DIRS and not d.startswith(".")]
        for name in names:
            rel = str((Path(root) / name).relative_to(VAULT_PATH))
            files.add(name.lower())
            if name.endswith(".md"):
                stems.add(name[:-3].lower())
                paths.add(rel[:-3].lower())

    broken = []
    for t in targets:
        low = t.lower()
        if Path(t).suffix and Path(t).suffix.lower() != ".md":
            ok = Path(t).name.lower() in files
        else:
            low = low.removesuffix(".md")
            ok = low in paths or Path(low).name in stems
        if not ok:
            broken.append(t)

    notes = []
    if not targets:
        notes.append("no [[wikilinks]] at all — link the note to its hub and related notes")
    elif not _is_hub(Path(rel_path).stem) and not any(_is_hub(t) for t in targets):
        notes.append("no link to a hub note ([[README]], an *INDEX note, or a project *overview*)")
    if broken:
        notes.append("broken links (no matching note): " + ", ".join(f"[[{b}]]" for b in broken))
    if not notes:
        return f"Link check: OK ({len(targets)} links)."
    return "Link check WARNING: " + "; ".join(notes) + "."


def update_frontmatter_timestamp(text: str) -> str:
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm_body = parts[1]
            if re.search(r"^updated:.*$", fm_body, flags=re.MULTILINE):
                fm_body = re.sub(r"^updated:.*$", f'updated: "{now_str}"', fm_body, flags=re.MULTILINE)
            else:
                fm_body = fm_body.rstrip() + f'\nupdated: "{now_str}"\n'
            return f"---{fm_body}---" + parts[2]
    return text


def _after_change(rel: str, content: str) -> str:
    cache_clear()
    trigger_background_reindex()
    return link_report(content, rel)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

# Function words that only add noise to an OR-joined BM25 query.
STOPWORDS = {
    # Indonesian
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada", "ini", "itu", "atau", "adalah",
    "juga", "ada", "tidak", "gak", "nggak", "bisa", "akan", "sudah", "udah", "dalam", "saja", "aja",
    "apa", "bagaimana", "gimana", "kenapa", "mengapa", "kapan", "dimana", "mana", "siapa", "nya",
    "lalu", "jadi", "karena", "kalau", "kalo", "agar", "supaya", "oleh", "sebagai", "tentang", "cara",
    # English
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "were",
    "be", "it", "this", "that", "how", "what", "why", "when", "where", "which", "who", "do", "does",
    "can", "i", "my", "we", "you", "about", "from", "by", "as", "at",
}


def build_fts_query(query: str) -> str:
    terms = [t for t in "".join(c if c.isalnum() or c in " _-" else " " for c in query).split() if t]
    terms = [t.strip("-_") for t in terms if t.strip("-_")]
    content = [t for t in terms if t.lower() not in STOPWORDS]
    terms = content or terms
    # Quote each term so FTS5 treats '-' and keywords (AND/OR/NOT/NEAR) literally.
    return " OR ".join('"' + t + '"' for t in terms)


def diversify(scored_items: list, rel_of: dict, limit: int, per_note: int) -> list:
    """Keep score order but allow at most `per_note` chunks from the same note."""
    picked, counts = [], {}
    for item in scored_items:
        rel = rel_of[item[0]]
        if counts.get(rel, 0) >= per_note:
            continue
        counts[rel] = counts.get(rel, 0) + 1
        picked.append(item)
        if len(picked) >= limit:
            break
    return picked


@mcp.tool()
def vault_search(query: str, limit: int = 5, mode: str = "hybrid", folder: str = "") -> str:
    """Search the Obsidian vault by meaning and keywords. Prefer this over grep for concept questions.

    Two-tier hybrid retrieval: BM25 (FTS5) + BGE-M3 vectors, fused with RRF and
    reranked by a cross-encoder. At most 2 chunks per note are returned.

    Args:
        query: Search keywords or natural language question (Indonesian or English).
        limit: Max results to return (default 5, max 15).
        mode: 'hybrid' (RRF + cross-encoder), 'keyword' (FTS5 BM25 only), or 'semantic' (BGE-M3 vector only).
        folder: Optional folder filter within the vault (e.g. 'server', 'projects/cuantum').
    """
    limit = max(1, min(limit, 15))
    folder_filter = folder.strip("/ ")
    if mode not in ("hybrid", "keyword", "semantic"):
        mode = "hybrid"
    cache_key = (query.strip().lower(), limit, mode, folder_filter, index_version())

    cached_res = cache_get(cache_key)
    if cached_res is not None:
        if _embed_model is not None or _reranker is not None:
            schedule_model_unload()
        return cached_res

    conn = get_db_connection()
    folder_like = f"{folder_filter}/%"
    fetch_n = limit * 6  # headroom for per-note diversity

    fts_candidates = []
    if mode in ("hybrid", "keyword"):
        fts_query = build_fts_query(query)
        if fts_query:
            try:
                if folder_filter:
                    rows = conn.execute(
                        "SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ? AND rel_path LIKE ? ORDER BY rank LIMIT ?;",
                        (fts_query, folder_like, fetch_n),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?;",
                        (fts_query, fetch_n),
                    ).fetchall()
                fts_candidates = [(row["rowid"], row["rank"]) for row in rows]
            except sqlite3.Error:
                fts_candidates = []

    vec_candidates = []
    if mode in ("hybrid", "semantic"):
        try:
            embed_model = get_embed_model()
            q_blob = serialize_vector(embed_model.encode([query], normalize_embeddings=True)[0])
            # KNN runs before the folder filter, so oversample to keep enough in-folder hits.
            knn_k = min(4096, max(300, fetch_n * 25)) if folder_filter else fetch_n
            rows = conn.execute(
                "SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? ORDER BY distance LIMIT ?;",
                (q_blob, knn_k),
            ).fetchall()
            raw_vec = [(row["rowid"], row["distance"]) for row in rows]
            if folder_filter and raw_vec:
                placeholders = ",".join("?" for _ in raw_vec)
                valid_ids = {
                    r["id"] for r in conn.execute(
                        f"SELECT id FROM chunks WHERE id IN ({placeholders}) AND rel_path LIKE ?",
                        [r[0] for r in raw_vec] + [folder_like],
                    ).fetchall()
                }
                raw_vec = [r for r in raw_vec if r[0] in valid_ids]
            vec_candidates = raw_vec[:fetch_n]
        except Exception:  # noqa: BLE001 - model load/encode can fail many ways; keep keyword hits
            vec_candidates = []

    # Reciprocal Rank Fusion (RRF) k=60
    k = 60
    if mode == "keyword":
        ranked_lists = [fts_candidates]
    elif mode == "semantic":
        ranked_lists = [vec_candidates]
    else:
        ranked_lists = [fts_candidates, vec_candidates]
    rrf_scores = {}
    for candidates in ranked_lists:
        for rank, (cid, _) in enumerate(candidates, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)

    if not rrf_scores:
        conn.close()
        res = f"No matching notes found in vault for query: '{query}'."
        cache_set(cache_key, res)
        return res

    pool_size = max(limit * 4, 16)
    candidate_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:pool_size]
    placeholders = ",".join("?" for _ in candidate_ids)
    chunk_map = {
        row["id"]: row for row in conn.execute(
            f"SELECT id, rel_path, title, heading, summary, chunk_text, line_start, line_end "
            f"FROM chunks WHERE id IN ({placeholders});",
            candidate_ids,
        ).fetchall()
    }
    conn.close()

    # Stage 2: cross-encoder rerank (falls back to RRF order on failure)
    ordered_cids = [cid for cid in candidate_ids if cid in chunk_map]
    scored_items = [(cid, rrf_scores[cid]) for cid in ordered_cids]
    if mode == "hybrid" and ordered_cids:
        try:
            reranker = get_reranker()
            doc_texts = [
                f"Title: {chunk_map[cid]['title']} > {chunk_map[cid]['heading']}\n{chunk_map[cid]['chunk_text'][:800]}"
                for cid in ordered_cids
            ]
            rerank_scores = list(reranker.rerank(query, doc_texts))
            scored_items = sorted(
                ((cid, float(s)) for cid, s in zip(ordered_cids, rerank_scores)),
                key=lambda x: x[1], reverse=True,
            )
        except Exception:  # noqa: BLE001, S110 - reranker failure keeps the RRF order
            pass

    rel_of = {cid: chunk_map[cid]["rel_path"] for cid in ordered_cids}
    top_results = diversify(scored_items, rel_of, limit, MAX_CHUNKS_PER_NOTE)

    results = []
    for rank, (cid, score) in enumerate(top_results, start=1):
        c = chunk_map[cid]
        snippet = c["chunk_text"].strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        score_label = f"Score: {score:.3f}" if mode == "hybrid" else f"RRF: {score:.4f}"
        header = f"### [{rank}] [[{Path(c['rel_path']).stem}]] > {c['heading']} ({score_label})"
        meta = f"File: `{c['rel_path']}` (Lines {c['line_start']}-{c['line_end']})"
        results.append(f"{header}\n{meta}\n\n{snippet}\n")

    res = "\n---\n".join(results)
    cache_set(cache_key, res)
    if _embed_model is not None or _reranker is not None:
        schedule_model_unload()
    return res


# ---------------------------------------------------------------------------
# Read / list / backlinks
# ---------------------------------------------------------------------------

def _extract_section(content: str, heading: str):
    target_clean = heading.strip().lower()
    in_section, section_level, section_lines = False, 0, []
    heading_regex = re.compile(r"^(#{1,6})\s+(.+)$")
    for line in content.splitlines():
        match = heading_regex.match(line)
        if match:
            level = len(match.group(1))
            if in_section:
                if level <= section_level:
                    break
                section_lines.append(line)
            elif target_clean in match.group(2).strip().lower():
                in_section, section_level = True, level
                section_lines.append(line)
        elif in_section:
            section_lines.append(line)
    return section_lines


@mcp.tool()
def vault_read(rel_path: str, heading: str = "") -> str:
    """Read a full note or one heading section from the vault.

    Args:
        rel_path: Relative path ('server/homeserver-setup.md') or a unique note name ('homeserver-setup').
        heading: Optional heading text to return only that section.
    """
    target_file, clean_rel, error = resolve_note(rel_path, must_exist=True)
    if error:
        return error
    try:
        content = target_file.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError) as e:
        return f"Error reading file: {e}"

    if not heading:
        return f"# File: `{clean_rel}`\n\n{content}"
    section_lines = _extract_section(content, heading)
    if section_lines:
        return f"# File: `{clean_rel}` > Heading: `{heading}`\n\n" + "\n".join(section_lines)
    return f"Heading `{heading}` not found in `{clean_rel}`. Returning whole note:\n\n{content}"


@mcp.tool()
def vault_list(folder: str = "", limit: int = 200) -> str:
    """List notes in the vault (or one folder, recursively) with their titles. Use instead of ls/find.

    Args:
        folder: Optional folder within the vault (e.g. 'projects/cuantum'). Empty lists the whole vault.
        limit: Max notes to list (default 200, max 1000).
    """
    limit = max(1, min(limit, 1000))
    folder_clean = folder.strip().strip("/")
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
    notes = [rel for _, rel in iter_vault_notes() if rel.startswith(prefix)]
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
    _target_file, target_rel, error = resolve_note(note, must_exist=True)
    if error:
        return error
    stem = Path(target_rel).stem.lower()
    path_noext = target_rel[:-3].lower()
    hits = []
    for full, rel in iter_vault_notes():
        if rel == target_rel or len(hits) >= limit:
            continue
        try:
            text = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            found = False
            for m in WIKILINK_RE.finditer(line):
                t = m.group(1).strip().lower()
                t = t.removesuffix(".md")
                if t == path_noext or Path(t).name == stem:
                    found = True
                    break
            if found:
                hits.append(f"- `{rel}` L{lineno}: {line.strip()[:200]}")
                if len(hits) >= limit:
                    break
    if not hits:
        return f"No backlinks found for `{target_rel}` (orphan note)."
    return f"### Backlinks to `{target_rel}` ({len(hits)})\n" + "\n".join(hits)


# ---------------------------------------------------------------------------
# Write / append / edit
# ---------------------------------------------------------------------------

@mcp.tool()
def vault_write(rel_path: str, content: str, title: str = "", tags: str = "") -> str:
    """Create or overwrite a markdown note. Adds frontmatter, reports broken/missing hub [[wikilinks]],
    clears the search cache and triggers background re-indexing.

    Args:
        rel_path: Note path within the vault (e.g. 'projects/my-idea.md').
        content: Markdown content. Link related notes and the hub with [[wikilinks]].
        title: Optional frontmatter title (defaults to file stem).
        tags: Optional comma-separated tags (e.g. 'project, quant, research').
    """
    target_file, clean_rel, error = resolve_note(rel_path, must_exist=False)
    if error:
        return error
    target_file.parent.mkdir(parents=True, exist_ok=True)

    final_title = title.strip() or target_file.stem
    tag_list = [t.strip().lstrip("#") for t in tags.split(",") if t.strip()] if tags else []
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    if not content.startswith("---"):
        fm_lines = ["---", f"title: \"{final_title}\""]
        if tag_list:
            fm_lines.append(f"tags: [{', '.join(tag_list)}]")
        fm_lines += [f"created: \"{now_str}\"", f"updated: \"{now_str}\"", "---\n"]
        full_content = "\n".join(fm_lines) + content.lstrip()
    else:
        full_content = update_frontmatter_timestamp(content)

    try:
        target_file.write_text(full_content, encoding="utf-8")
    except (OSError, UnicodeError) as e:
        return f"Error writing file `{clean_rel}`: {e}"
    report = _after_change(clean_rel, full_content)
    return f"Successfully wrote `{clean_rel}` ({len(full_content)} bytes). Re-index triggered.\n{report}"


@mcp.tool()
def vault_append(rel_path: str, content: str, heading: str = "") -> str:
    """Append content to an existing note, optionally under a heading (created at the end if missing).
    Reports broken/missing hub [[wikilinks]], clears the search cache and triggers re-indexing.

    Args:
        rel_path: Note path or unique note name.
        content: Text to append. Use [[wikilinks]] for connected concepts.
        heading: Optional heading under which to append.
    """
    target_file, clean_rel, error = resolve_note(rel_path, must_exist=True)
    if error:
        return error
    try:
        existing = target_file.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError) as e:
        return f"Error reading file `{clean_rel}`: {e}"

    appended_text = content.strip()
    if not heading:
        new_content = existing.rstrip() + "\n\n" + appended_text + "\n"
    else:
        lines = existing.splitlines()
        target_clean = heading.strip().lower()
        heading_regex = re.compile(r"^(#{1,6})\s+(.+)$")
        insert_idx, section_level, in_section = -1, 0, False
        for idx, line in enumerate(lines):
            match = heading_regex.match(line)
            if match:
                level = len(match.group(1))
                if in_section:
                    if level <= section_level:
                        insert_idx = idx
                        break
                elif target_clean in match.group(2).strip().lower():
                    in_section, section_level = True, level
                    insert_idx = idx + 1
            elif in_section:
                insert_idx = idx + 1
        if in_section and insert_idx != -1:
            lines.insert(insert_idx, "\n" + appended_text + "\n")
            new_content = "\n".join(lines) + "\n"
        else:
            new_content = existing.rstrip() + f"\n\n## {heading}\n\n" + appended_text + "\n"

    new_content = update_frontmatter_timestamp(new_content)
    try:
        target_file.write_text(new_content, encoding="utf-8")
    except (OSError, UnicodeError) as e:
        return f"Error updating file `{clean_rel}`: {e}"
    report = _after_change(clean_rel, new_content)
    return f"Successfully appended to `{clean_rel}`. Re-index triggered.\n{report}"


@mcp.tool()
def vault_edit(rel_path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
    """Replace an exact text span in a note (use instead of patch/sed on vault files).
    Fails if old_text is missing, or appears more than once without replace_all.
    Reports broken/missing hub [[wikilinks]], clears the search cache and triggers re-indexing.

    Args:
        rel_path: Note path or unique note name.
        old_text: Exact text to replace (include enough context to be unique).
        new_text: Replacement text.
        replace_all: Replace every occurrence instead of requiring a unique match.
    """
    target_file, clean_rel, error = resolve_note(rel_path, must_exist=True)
    if error:
        return error
    if not old_text:
        return "Error: old_text must not be empty."
    try:
        existing = target_file.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError) as e:
        return f"Error reading file `{clean_rel}`: {e}"

    count = existing.count(old_text)
    if count == 0:
        return f"Error: old_text not found in `{clean_rel}`. Re-read the note and copy the exact text."
    if count > 1 and not replace_all:
        return f"Error: old_text appears {count} times in `{clean_rel}`. Add more context or set replace_all=true."

    new_content = existing.replace(old_text, new_text) if replace_all else existing.replace(old_text, new_text, 1)
    new_content = update_frontmatter_timestamp(new_content)
    try:
        target_file.write_text(new_content, encoding="utf-8")
    except (OSError, UnicodeError) as e:
        return f"Error updating file `{clean_rel}`: {e}"
    report = _after_change(clean_rel, new_content)
    replaced = count if replace_all else 1
    return f"Successfully edited `{clean_rel}` ({replaced} replacement(s)). Re-index triggered.\n{report}"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def index_freshness():
    """Compare vault files to the index: (pending_rel_paths, deleted_rel_paths)."""
    conn = get_db_connection()
    indexed = {r["rel_path"]: r["mtime"] for r in conn.execute("SELECT rel_path, mtime FROM notes")}
    conn.close()
    pending, current = [], set()
    for full, rel in iter_vault_notes():
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
    """Report index health: note/chunk counts, notes waiting to be indexed, indexer activity, model RAM state."""
    if not DB_PATH.exists():
        return f"Database not found at `{DB_PATH}`. Run the indexer first."

    conn = get_db_connection()
    notes_count = conn.execute("SELECT count(*) FROM notes").fetchone()[0]
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
    last_write_str = f"{age_min:.0f} min ago" if age_min is not None else "never"

    return f"""### Vault Index Status
- **Health:** {health}
- **Indexed:** {notes_count} notes, {chunks_count} chunks, {vec_count} vectors ({db_size_mb:.1f} MB)
- **Last index write:** {last_write_str}
- **Indexer running:** {'yes' if running else 'no'}{pending_list}
- **Models:** `{EMBED_MODEL_NAME}` + `{RERANK_MODEL_NAME}` — {model_status_str}
- **Search cache:** {len(_search_cache)}/{_SEARCH_CACHE_MAX_SIZE}
"""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Obsidian Hybrid RAG FastMCP server")
    parser.add_argument("--transport", default=os.environ.get("MCP_TRANSPORT", "stdio"),
                        choices=["stdio", "sse", "http", "streamable-http"])
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"),
                        help="Bind address for network transports. Keep 127.0.0.1: write tools have no auth.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", "8765")))
    args = parser.parse_args()

    if args.transport in {"sse", "http", "streamable-http"}:
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            print(f"[vault-mcp] WARNING: binding to {args.host} exposes unauthenticated write tools.", file=sys.stderr)
        mcp.run(transport=args.transport, host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
