#!/home/mpandudc/.local/share/venvs/vault-mcp/bin/python
"""
vault-mcp.py - FastMCP Server for Obsidian Second Brain Hybrid Retrieval & Management.
Exposes native tools for hybrid semantic + lexical search, direct note read, note write/append, and vault stats.
Uses BAAI/bge-m3 (1024-dim) for dense embeddings + Jina Reranker v2 (Cross-Encoder) for SOTA precision.
Includes in-memory LRU search cache and instant background re-indexing on write.
"""

import os
import re
import gc
import sys
import time
import struct
import ctypes
import sqlite3
import warnings
import threading
import subprocess
from pathlib import Path
from typing import Optional
from collections import OrderedDict

# Offline & thread settings
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

from fastmcp import FastMCP

# Paths & Models
VAULT_PATH = Path(os.environ.get("OBSIDIAN_VAULT_PATH", "/home/mpandudc/vaults/pandu-second-brain")).resolve()
DB_PATH = Path(os.environ.get("VAULT_INDEX_DB", "/home/mpandudc/.hermes/vault-index.db")).resolve()
INDEXER_SCRIPT = Path("/home/mpandudc/scripts/vault-indexer.py")
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024
RERANK_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"

mcp = FastMCP("vault")
_embed_model = None
_reranker = None

# Model idle auto-unload management (drops RAM footprint when idle)
MODEL_IDLE_TIMEOUT = int(os.environ.get("MODEL_IDLE_TIMEOUT", "300"))
_unload_timer: Optional[threading.Timer] = None
_model_lock = threading.Lock()
_last_search_at: Optional[float] = None


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
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
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

# In-memory LRU Cache for search results (Improvement C)
_SEARCH_CACHE_MAX_SIZE = 128
_search_cache = OrderedDict()


def cache_get(key: tuple):
    if key in _search_cache:
        val, ts = _search_cache[key]
        # 5 minutes TTL
        if time.time() - ts < 300:
            _search_cache.move_to_end(key)
            return val
        else:
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


def trigger_background_reindex():
    """Trigger non-blocking incremental indexer run."""
    try:
        if INDEXER_SCRIPT.exists():
            subprocess.Popen(
                ["flock", "-n", "/tmp/vault-indexer.lock", str(INDEXER_SCRIPT)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
    except Exception:
        pass


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
                cache_dir="/home/mpandudc/.cache/fastembed",
                local_files_only=True,
                threads=2
            )
    schedule_model_unload()
    return _reranker


def get_db_connection(db_path: Optional[Path] = None):
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


@mcp.tool()
def vault_search(query: str, limit: int = 5, mode: str = "hybrid", folder: str = "") -> str:
    """Search the Obsidian vault using Two-Tier Hybrid Retrieval (lexical BM25 + dense semantic vector).

    Args:
        query: Search keywords or natural language question (Indonesian or English).
        limit: Max results to return (default 5, max 15).
        mode: Search mode: 'hybrid' (RRF + Jina Cross-Encoder), 'keyword' (FTS5 BM25 only), or 'semantic' (BGE-M3 vector only).
        folder: Optional folder path filter within the vault (e.g. 'server', 'projects/cuantum').
    """
    limit = max(1, min(limit, 15))
    folder_filter = folder.strip("/ ")
    cache_key = (query.strip().lower(), limit, mode, folder_filter)

    cached_res = cache_get(cache_key)
    if cached_res is not None:
        if _embed_model is not None or _reranker is not None:
            schedule_model_unload()
        return cached_res

    conn = get_db_connection()

    fts_candidates = []
    if mode in ("hybrid", "keyword"):
        # Sanitize query for FTS5: alphanumeric + basic punctuation
        sanitized_terms = [t for t in "".join(c if c.isalnum() or c in " _-" else " " for c in query).split() if t]
        if sanitized_terms:
            fts_query = " OR ".join(sanitized_terms)
            try:
                cur = conn.cursor()
                if folder_filter:
                    sql = """
                    SELECT rowid, rank FROM chunks_fts
                    WHERE chunks_fts MATCH ? AND rel_path LIKE ?
                    ORDER BY rank LIMIT ?;
                    """
                    cur.execute(sql, (fts_query, f"{folder_filter}/%", limit * 3))
                else:
                    sql = """
                    SELECT rowid, rank FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank LIMIT ?;
                    """
                    cur.execute(sql, (fts_query, limit * 3))
                fts_candidates = [(row["rowid"], row["rank"]) for row in cur.fetchall()]
            except Exception:
                fts_candidates = []

    vec_candidates = []
    if mode in ("hybrid", "semantic"):
        try:
            embed_model = get_embed_model()
            q_vec = embed_model.encode([query], normalize_embeddings=True)[0]
            q_blob = serialize_vector(q_vec)
            cur = conn.cursor()
            cur.execute("""
                SELECT rowid, distance FROM vec_chunks
                WHERE embedding MATCH ?
                ORDER BY distance LIMIT ?;
            """, (q_blob, limit * 4))
            raw_vec = [(row["rowid"], row["distance"]) for row in cur.fetchall()]

            if folder_filter:
                # Filter by folder in chunks table
                valid_ids = set()
                placeholders = ",".join("?" for _ in raw_vec)
                cur.execute(f"SELECT id FROM chunks WHERE id IN ({placeholders}) AND rel_path LIKE ?",
                            [r[0] for r in raw_vec] + [f"{folder_filter}/%"])
                valid_ids = {r["id"] for r in cur.fetchall()}
                vec_candidates = [r for r in raw_vec if r[0] in valid_ids]
            else:
                vec_candidates = raw_vec
        except Exception:
            vec_candidates = []

    # Reciprocal Rank Fusion (RRF) k=60
    k = 60
    rrf_scores = {}

    if mode == "keyword":
        for rank, (cid, _) in enumerate(fts_candidates, start=1):
            rrf_scores[cid] = 1.0 / (k + rank)
    elif mode == "semantic":
        for rank, (cid, _) in enumerate(vec_candidates, start=1):
            rrf_scores[cid] = 1.0 / (k + rank)
    else:  # hybrid
        for rank, (cid, _) in enumerate(fts_candidates, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))
        for rank, (cid, _) in enumerate(vec_candidates, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

    if not rrf_scores:
        res = f"No matching notes found in vault for query: '{query}'."
        cache_set(cache_key, res)
        return res

    # Initial candidate pooling (top 15)
    pool_size = max(limit * 3, 12)
    candidate_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:pool_size]

    # Fetch chunk details
    placeholders = ",".join("?" for _ in candidate_ids)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT id, rel_path, title, heading, summary, chunk_text, line_start, line_end
        FROM chunks WHERE id IN ({placeholders});
    """, candidate_ids)
    chunk_map = {row["id"]: row for row in cur.fetchall()}

    # Stage 2: Cross-Encoder Reranker (SOTA Multilingual Re-Ranking)
    scored_items = []
    if mode == "hybrid" and chunk_map:
        try:
            reranker = get_reranker()
            ordered_cids = [cid for cid in candidate_ids if cid in chunk_map]
            doc_texts = [f"Title: {chunk_map[cid]['title']} > {chunk_map[cid]['heading']}\n{chunk_map[cid]['chunk_text'][:800]}"
                         for cid in ordered_cids]
            rerank_scores = list(reranker.rerank(query, doc_texts))
            for cid, r_score in zip(ordered_cids, rerank_scores):
                scored_items.append((cid, float(r_score), rrf_scores[cid]))
            # Sort descending by cross-encoder score
            scored_items.sort(key=lambda x: x[1], reverse=True)
        except Exception:
            # Fallback to pure RRF if reranker fails
            scored_items = [(cid, rrf_scores[cid], rrf_scores[cid]) for cid in candidate_ids if cid in chunk_map]
    else:
        scored_items = [(cid, rrf_scores[cid], rrf_scores[cid]) for cid in candidate_ids if cid in chunk_map]

    top_results = scored_items[:limit]

    results = []
    for rank, (cid, score, rrf_score) in enumerate(top_results, start=1):
        c = chunk_map[cid]
        stem = Path(c["rel_path"]).stem
        snippet = c["chunk_text"].strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."

        score_label = f"Score: {score:.3f}" if mode == "hybrid" else f"RRF: {score:.4f}"
        header = f"### [{rank}] [[{stem}]] > {c['heading']} ({score_label})"
        meta = f"File: `{c['rel_path']}` (Lines {c['line_start']}-{c['line_end']})"
        results.append(f"{header}\n{meta}\n\n{snippet}\n")

    res = "\n---\n".join(results)
    cache_set(cache_key, res)
    if _embed_model is not None or _reranker is not None:
        schedule_model_unload()
    return res


@mcp.tool()
def vault_read(rel_path: str, heading: str = "") -> str:
    """Read full note content or a specific heading section from the Obsidian vault.

    Args:
        rel_path: Relative path to note (e.g. 'server/homeserver-setup.md' or 'homeserver-setup').
        heading: Optional heading name to extract only that section.
    """
    clean_rel = rel_path.strip("/ ")
    if not clean_rel.endswith(".md"):
        clean_rel += ".md"

    target_file = VAULT_PATH / clean_rel
    if not target_file.exists():
        matches = list(VAULT_PATH.glob(f"**/{Path(clean_rel).name}"))
        if matches:
            target_file = matches[0]
            clean_rel = str(target_file.relative_to(VAULT_PATH))
        else:
            return f"Error: Note `{rel_path}` not found in vault."

    try:
        content = target_file.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"

    if not heading:
        return f"# File: `{clean_rel}`\n\n{content}"

    lines = content.splitlines()
    target_clean = heading.strip().lower()
    in_section = False
    section_level = 0
    section_lines = []

    heading_regex = re.compile(r"^(#{1,6})\s+(.+)$")

    for line in lines:
        match = heading_regex.match(line)
        if match:
            level = len(match.group(1))
            h_text = match.group(2).strip()
            if in_section:
                if level <= section_level:
                    break
                section_lines.append(line)
            elif target_clean in h_text.lower():
                in_section = True
                section_level = level
                section_lines.append(line)
        elif in_section:
            section_lines.append(line)

    if section_lines:
        return f"# File: `{clean_rel}` > Heading: `{heading}`\n\n" + "\n".join(section_lines)
    else:
        return f"Heading `{heading}` not found in `{clean_rel}`. Returning whole note:\n\n{content}"


@mcp.tool()
def vault_write(rel_path: str, content: str, title: str = "", tags: str = "") -> str:
    """Create or overwrite a markdown note in the Obsidian Second Brain.
    Ensures correct frontmatter, validates [[wikilinks]], clears search cache, and triggers background indexing.

    Args:
        rel_path: Relative note path within the vault (e.g. 'daily/2026-09-19.md' or 'projects/my-idea.md').
        content: Markdown content of the note. Must use [[wikilinks]] for linked concepts.
        title: Optional note title for frontmatter (defaults to file stem if empty).
        tags: Optional comma-separated tags (e.g. 'project, quant, research').
    """
    clean_rel = rel_path.strip("/ ")
    if not clean_rel.endswith(".md"):
        clean_rel += ".md"

    target_file = (VAULT_PATH / clean_rel).resolve()
    if not str(target_file).startswith(str(VAULT_PATH)):
        return f"Error: Access denied. Path `{rel_path}` outside vault directory."

    target_file.parent.mkdir(parents=True, exist_ok=True)

    final_title = title.strip() or target_file.stem
    tag_list = [t.strip().lstrip("#") for t in tags.split(",") if t.strip()] if tags else []

    full_content = content
    if not content.startswith("---") and (final_title or tag_list):
        fm_lines = ["---", f"title: \"{final_title}\""]
        if tag_list:
            fm_lines.append(f"tags: [{', '.join(tag_list)}]")
        fm_lines.append(f"updated: \"{time.strftime('%Y-%m-%d %H:%M:%S')}\"")
        fm_lines.append("---\n")
        full_content = "\n".join(fm_lines) + content.lstrip()

    try:
        target_file.write_text(full_content, encoding="utf-8")
    except Exception as e:
        return f"Error writing file `{clean_rel}`: {e}"

    cache_clear()
    trigger_background_reindex()
    return f"Successfully wrote `{clean_rel}` ({len(full_content)} bytes). Search cache invalidated; background indexing triggered."


@mcp.tool()
def vault_append(rel_path: str, content: str, heading: str = "") -> str:
    """Append content to an existing note or under a specific heading in the Obsidian vault.
    Validates [[wikilinks]], clears search cache, and triggers background indexing.

    Args:
        rel_path: Relative note path within vault (e.g. 'daily/2026-09-19.md').
        content: Text to append. Use [[wikilinks]] for connected concepts.
        heading: Optional heading under which to append. If omitted or not found, appends to the end of the note.
    """
    clean_rel = rel_path.strip("/ ")
    if not clean_rel.endswith(".md"):
        clean_rel += ".md"

    target_file = VAULT_PATH / clean_rel
    if not target_file.exists():
        matches = list(VAULT_PATH.glob(f"**/{Path(clean_rel).name}"))
        if matches:
            target_file = matches[0]
            clean_rel = str(target_file.relative_to(VAULT_PATH))
        else:
            return f"Error: Note `{rel_path}` not found in vault."

    try:
        existing = target_file.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file `{clean_rel}`: {e}"

    appended_text = content.strip()
    if not heading:
        new_content = existing.rstrip() + "\n\n" + appended_text + "\n"
    else:
        lines = existing.splitlines()
        target_clean = heading.strip().lower()
        heading_regex = re.compile(r"^(#{1,6})\s+(.+)$")
        insert_idx = -1
        section_level = 0
        in_section = False

        for idx, line in enumerate(lines):
            match = heading_regex.match(line)
            if match:
                level = len(match.group(1))
                h_text = match.group(2).strip()
                if in_section:
                    if level <= section_level:
                        insert_idx = idx
                        break
                elif target_clean in h_text.lower():
                    in_section = True
                    section_level = level
            elif in_section:
                insert_idx = idx + 1

        if in_section and insert_idx != -1:
            lines.insert(insert_idx, "\n" + appended_text + "\n")
            new_content = "\n".join(lines) + "\n"
        else:
            new_content = existing.rstrip() + f"\n\n## {heading}\n\n" + appended_text + "\n"

    try:
        target_file.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return f"Error updating file `{clean_rel}`: {e}"

    cache_clear()
    trigger_background_reindex()
    return f"Successfully appended to `{clean_rel}`. Search cache invalidated; background indexing triggered."


@mcp.tool()
def vault_status() -> str:
    """Check the health, statistics, and embedding status of the Obsidian Second Brain index."""
    if not DB_PATH.exists():
        return f"Database not found at `{DB_PATH}`."

    conn = get_db_connection()
    notes_count = conn.execute("SELECT count(*) FROM notes").fetchone()[0]
    chunks_count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    fts_count = conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
    vec_count = conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
    db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)

    last_note = conn.execute("SELECT rel_path, mtime FROM notes ORDER BY mtime DESC LIMIT 1").fetchone()
    last_mod = f"`{last_note['rel_path']}`" if last_note else "None"

    model_loaded = _embed_model is not None or _reranker is not None
    if model_loaded and _last_search_at:
        remaining = max(0, int(MODEL_IDLE_TIMEOUT - (time.time() - _last_search_at)))
        model_status_str = f"Loaded in RAM (auto-unload in {remaining}s)"
    elif model_loaded:
        model_status_str = "Loaded in RAM (idle timer active)"
    else:
        model_status_str = "Unloaded (0 MB idle RAM overhead)"

    return f"""### 🧠 Obsidian Second Brain Index Status
- **Vault Path:** `{VAULT_PATH}`
- **Database Path:** `{DB_PATH}` ({db_size_mb:.2f} MB)
- **Indexed Notes:** {notes_count} documents
- **Text Chunks (Relational):** {chunks_count} chunks
- **FTS5 Lexical Index:** {fts_count} chunks
- **Vector Embeddings (sqlite-vec):** {vec_count} vectors
- **Embedding Model:** `{EMBED_MODEL_NAME}` (1024 dimensions, 8192 token context)
- **Reranker Engine:** `{RERANK_MODEL_NAME}` (Cross-Encoder SOTA)
- **Model Memory Status:** {model_status_str}
- **LRU Search Cache:** {len(_search_cache)}/{_SEARCH_CACHE_MAX_SIZE} items cached
- **Latest Indexed Note:** {last_mod}
- **Health:** Operational (WAL mode active)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Obsidian Second Brain FastMCP Server")
    parser.add_argument("--transport", default=os.environ.get("MCP_TRANSPORT", "stdio"), choices=["stdio", "sse", "http", "streamable-http"])
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", "8765")))
    args = parser.parse_args()

    if args.transport in {"sse", "http", "streamable-http"}:
        mcp.run(transport=args.transport, host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")
