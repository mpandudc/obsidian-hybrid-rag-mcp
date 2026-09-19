"""
FastMCP Stdio Server for Obsidian Hybrid RAG.
Features Two-Stage Retrieval: FTS5 BM25 + BGE-M3 Dense Vectors + Jina Reranker v2 Cross-Encoder.
Includes LRU Search Cache and note authoring tools (vault_write, vault_append).
"""

from __future__ import annotations
import os
import re
import sqlite3
import struct
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sqlite_vec
import torch
from fastembed.rerank.cross_encoder import TextCrossEncoder
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

from src.indexer import build_index

DEFAULT_VAULT_PATH = os.getenv("VAULT_PATH", str(Path.home() / "vaults" / "pandu-second-brain"))
DEFAULT_DB_PATH = os.getenv("INDEX_DB_PATH", str(Path.home() / ".hermes" / "vault-index.db"))
EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"

mcp = FastMCP("obsidian-hybrid-rag")

# Lazy model singletons
_EMBED_MODEL: Optional[SentenceTransformer] = None
_RERANK_MODEL: Optional[TextCrossEncoder] = None

# LRU Search Cache (in-memory)
_SEARCH_CACHE_MAX_SIZE = 128
_SEARCH_CACHE: OrderedDict[Tuple[str, int, Optional[str]], List[Dict[str, Any]]] = OrderedDict()


def cache_get(key: Tuple[str, int, Optional[str]]) -> Optional[List[Dict[str, Any]]]:
    if key in _SEARCH_CACHE:
        _SEARCH_CACHE.move_to_end(key)
        return _SEARCH_CACHE[key]
    return None


def cache_put(key: Tuple[str, int, Optional[str]], value: List[Dict[str, Any]]) -> None:
    if key in _SEARCH_CACHE:
        _SEARCH_CACHE.move_to_end(key)
    _SEARCH_CACHE[key] = value
    if len(_SEARCH_CACHE) > _SEARCH_CACHE_MAX_SIZE:
        _SEARCH_CACHE.popitem(last=False)


def cache_clear() -> None:
    _SEARCH_CACHE.clear()


def get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        torch.set_num_threads(4)
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _EMBED_MODEL


def get_rerank_model() -> TextCrossEncoder:
    global _RERANK_MODEL
    if _RERANK_MODEL is None:
        _RERANK_MODEL = TextCrossEncoder(model_name=RERANK_MODEL_NAME, threads=2)
    return _RERANK_MODEL


def get_db(db_path_str: str) -> sqlite3.Connection:
    p = Path(db_path_str).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Vault index database not found at {p}")
    # Open in URI read-only mode to avoid write locks and ensure safe concurrent reads
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


def serialize_f32(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _trigger_background_sync() -> None:
    """Trigger non-blocking background index sync if indexer script or tool exists."""
    indexer_script = Path.home() / "scripts" / "vault-indexer.py"
    if indexer_script.exists():
        subprocess.Popen(
            ["flock", "-n", "/tmp/vault-indexer.lock", str(indexer_script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


@mcp.tool()
def search_vault(
    query: str,
    top_k: int = 5,
    heading_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search your Obsidian knowledge base using state-of-the-art Hybrid RAG.
    Combines SQLite FTS5 (BM25 lexical), BAAI/bge-m3 (1024-dim dense semantic vector),
    Reciprocal Rank Fusion (RRF), and Jina Reranker v2 Cross-Encoder scoring.
    Features an in-memory LRU cache for repeat queries.

    Args:
        query: Natural language question or search phrase
        top_k: Number of highest-ranking passages to return (default 5)
        heading_filter: Optional substring filter for document section headers
    """
    cache_key = (query.strip().lower(), top_k, heading_filter.lower() if heading_filter else None)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    db_path = Path(DEFAULT_DB_PATH).expanduser().resolve()
    if not db_path.exists():
        return [{"error": f"Database file not found at {db_path}. Please run indexer first."}]

    conn = get_db(str(db_path))

    # Stage 1A: Lexical search via SQLite FTS5 (BM25)
    fts_results: List[Dict[str, Any]] = []
    try:
        # Sanitize query for FTS5 syntax
        clean_q = "".join(c if c.isalnum() or c.isspace() else " " for c in query).strip()
        if clean_q:
            fts_query = " OR ".join(f'"{word}"*' for word in clean_q.split() if word)
            cur = conn.execute("""
                SELECT
                    c.id AS chunk_id,
                    c.note_id,
                    c.heading,
                    c.text,
                    c.line_start,
                    c.line_end,
                    n.rel_path,
                    n.title,
                    bm25(fts_chunks) AS fts_score
                FROM fts_chunks f
                JOIN chunks c ON c.id = f.rowid
                JOIN notes n ON n.id = c.note_id
                WHERE fts_chunks MATCH ?
                ORDER BY fts_score ASC
                LIMIT 25;
            """, (fts_query,))
            fts_results = [dict(r) for r in cur.fetchall()]
    except Exception:
        fts_results = []

    # Stage 1B: Dense Vector search via BGE-M3 (1024-dim cosine distance)
    vec_results: List[Dict[str, Any]] = []
    try:
        embed_model = get_embed_model()
        q_emb = embed_model.encode([query], normalize_embeddings=True)[0]
        q_bytes = serialize_f32(q_emb.tolist())

        cur = conn.execute("""
            SELECT
                c.id AS chunk_id,
                c.note_id,
                c.heading,
                c.text,
                c.line_start,
                c.line_end,
                n.rel_path,
                n.title,
                v.distance AS cosine_dist
            FROM vec_chunks v
            JOIN chunks c ON c.id = v.chunk_id
            JOIN notes n ON n.id = c.note_id
            WHERE v.embedding MATCH ? AND k = 25
            ORDER BY v.distance ASC;
        """, (q_bytes,))
        vec_results = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        return [{"error": f"Vector retrieval error: {str(e)}"}]

    # Stage 1C: Reciprocal Rank Fusion (RRF, k=60)
    rrf_scores: Dict[int, float] = {}
    candidate_meta: Dict[int, Dict[str, Any]] = {}
    k_rrf = 60.0

    for rank, item in enumerate(fts_results):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_rrf + rank + 1))
        candidate_meta[cid] = item

    for rank, item in enumerate(vec_results):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_rrf + rank + 1))
        if cid not in candidate_meta:
            candidate_meta[cid] = item

    if not candidate_meta:
        return []

    # Sort candidates by combined RRF score and take top 15 for Stage 2 Reranking
    sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    top_candidates = [candidate_meta[cid] for cid in sorted_cids[:15]]

    if heading_filter:
        h_lower = heading_filter.lower()
        top_candidates = [c for c in top_candidates if h_lower in c["heading"].lower()]

    if not top_candidates:
        return []

    # Stage 2: Cross-Encoder Reranking via Jina Reranker v2
    reranker = get_rerank_model()
    docs_to_rerank = [f"{c['title']} > {c['heading']}\n{c['text']}" for c in top_candidates]
    rerank_scores = list(reranker.rerank(query=query, documents=docs_to_rerank))

    scored_candidates = []
    for cand, score in zip(top_candidates, rerank_scores):
        scored_candidates.append({
            "rel_path": cand["rel_path"],
            "title": cand["title"],
            "heading": cand["heading"],
            "lines": f"L{cand['line_start']}-L{cand['line_end']}",
            "relevance_score": round(float(score), 4),
            "text": cand["text"],
        })

    # Sort descending by cross-encoder relevance score
    scored_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
    result = scored_candidates[:top_k]
    cache_put(cache_key, result)
    return result


@mcp.tool()
def get_note(
    rel_path: str,
    offset_line: int = 1,
    limit_lines: int = 250,
) -> Dict[str, Any]:
    """
    Retrieve full markdown content of a specific note with line number pagination.

    Args:
        rel_path: Note path relative to vault root (e.g. 'trading/cuantum-setup.md')
        offset_line: Line number to start reading from (1-indexed)
        limit_lines: Maximum lines to return in single request
    """
    vault_path = Path(DEFAULT_VAULT_PATH).expanduser().resolve()
    target = (vault_path / rel_path).resolve()

    # Prevent directory traversal attacks
    if not str(target).startswith(str(vault_path)):
        return {"error": "Access denied: Path is outside vault directory."}

    if not target.exists() or not target.is_file():
        return {"error": f"Note file not found: {rel_path}"}

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        all_lines = content.splitlines()
        total_lines = len(all_lines)

        start = max(1, offset_line) - 1
        end = min(total_lines, start + limit_lines)
        sliced = all_lines[start:end]

        formatted = "\n".join(f"{idx + 1:4d} | {line}" for idx, line in enumerate(sliced, start=start))
        return {
            "rel_path": rel_path,
            "total_lines": total_lines,
            "start_line": start + 1,
            "end_line": end,
            "has_more": end < total_lines,
            "content": formatted,
        }
    except Exception as e:
        return {"error": f"Failed reading note: {str(e)}"}


@mcp.tool()
def write_note(
    rel_path: str,
    content: str,
    title: str = "",
    tags: str = "",
) -> Dict[str, Any]:
    """
    Create or overwrite a markdown note in the Obsidian Second Brain.
    Ensures YAML frontmatter with title & tags, invalidates search cache, and triggers background index sync.

    Args:
        rel_path: Note path relative to vault root (e.g. 'research/carf-tax.md')
        content: Note body. Must use [[wikilinks]] for linked concepts.
        title: Optional document title for YAML frontmatter
        tags: Optional comma-separated tags (e.g. 'crypto, tax, compliance')
    """
    clean_rel = rel_path.strip("/ ")
    if not clean_rel.endswith(".md"):
        clean_rel += ".md"

    vault_path = Path(DEFAULT_VAULT_PATH).expanduser().resolve()
    target = (vault_path / clean_rel).resolve()

    if not str(target).startswith(str(vault_path)):
        return {"status": "error", "error": f"Access denied: Path `{rel_path}` outside vault directory."}

    target.parent.mkdir(parents=True, exist_ok=True)

    final_title = title.strip() or target.stem
    tag_list = [t.strip().lstrip("#") for t in tags.split(",") if t.strip()] if tags else []

    full_content = content
    if not content.startswith("---") and (final_title or tag_list):
        fm_lines = ["---", f'title: "{final_title}"']
        if tag_list:
            fm_lines.append(f"tags: [{', '.join(tag_list)}]")
        fm_lines.append(f"updated: {Path(__file__).stat().st_mtime if Path(__file__).exists() else ''}")
        fm_lines.extend(["---", "", ""])
        full_content = "\n".join(fm_lines) + content

    target.write_text(full_content, encoding="utf-8")

    # Invalidate cache & trigger background indexing
    cache_clear()
    _trigger_background_sync()

    wikilink_count = len(re.findall(r"\[\[([^\]]+)\]\]", full_content))
    return {
        "status": "success",
        "rel_path": clean_rel,
        "bytes_written": len(full_content.encode("utf-8")),
        "wikilinks_detected": wikilink_count,
        "message": f"Note `{clean_rel}` written successfully.",
    }


@mcp.tool()
def append_note(
    rel_path: str,
    content: str,
    heading: str = "",
) -> Dict[str, Any]:
    """
    Append text to an existing note or under a specific heading in the Obsidian Second Brain.

    Args:
        rel_path: Note path relative to vault root (e.g. 'daily/2026-09-19.md')
        content: Text to append. Use [[wikilinks]] for linked concepts.
        heading: Optional heading name under which to append content.
    """
    clean_rel = rel_path.strip("/ ")
    if not clean_rel.endswith(".md"):
        clean_rel += ".md"

    vault_path = Path(DEFAULT_VAULT_PATH).expanduser().resolve()
    target = (vault_path / clean_rel).resolve()

    if not str(target).startswith(str(vault_path)):
        return {"status": "error", "error": f"Access denied: Path `{rel_path}` outside vault directory."}

    if not target.exists():
        matches = list(vault_path.glob(f"**/{Path(clean_rel).name}"))
        if matches:
            target = matches[0]
            clean_rel = str(target.relative_to(vault_path))
        else:
            return {"status": "error", "error": f"Note `{rel_path}` not found in vault."}

    try:
        existing = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"status": "error", "error": f"Failed reading note: {str(e)}"}

    if not heading:
        new_content = existing.rstrip() + "\n\n" + content.strip() + "\n"
    else:
        lines = existing.splitlines()
        target_clean = heading.strip().lower()
        idx_to_insert = -1
        heading_level = 0
        h_regex = re.compile(r"^(#{1,6})\s+(.+)$")

        for i, line in enumerate(lines):
            m = h_regex.match(line)
            if m:
                lvl = len(m.group(1))
                h_name = m.group(2).strip()
                if idx_to_insert != -1:
                    if lvl <= heading_level:
                        idx_to_insert = i
                        break
                elif target_clean in h_name.lower():
                    heading_level = lvl
                    idx_to_insert = i + 1

        if idx_to_insert != -1:
            lines.insert(idx_to_insert, "\n" + content.strip())
            new_content = "\n".join(lines) + "\n"
        else:
            new_content = existing.rstrip() + f"\n\n## {heading}\n\n" + content.strip() + "\n"

    target.write_text(new_content, encoding="utf-8")

    cache_clear()
    _trigger_background_sync()

    return {
        "status": "success",
        "rel_path": clean_rel,
        "heading_targeted": heading or "end_of_file",
        "message": f"Appended content to `{clean_rel}`.",
    }


@mcp.tool()
def sync_vault() -> Dict[str, Any]:
    """
    Trigger incremental sync on the Obsidian vault.
    Scans for created, modified, or deleted notes and updates dense vectors & FTS5 index.
    """
    vault_path = Path(DEFAULT_VAULT_PATH).expanduser().resolve()
    db_path = Path(DEFAULT_DB_PATH).expanduser().resolve()

    try:
        build_index(vault_path=vault_path, db_path=db_path, rebuild=False, batch_size=15)
        cache_clear()
        return {
            "status": "success",
            "message": "Incremental sync completed successfully.",
            "vault_path": str(vault_path),
            "db_path": str(db_path),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Obsidian Hybrid RAG FastMCP Server")
    parser.add_argument(
        "--transport",
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        choices=["stdio", "sse", "http", "streamable-http"],
        help="MCP transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MCP_HOST", "127.0.0.1"),
        help="Host for HTTP/SSE daemon (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_PORT", "8765")),
        help="Port for HTTP/SSE daemon (default: 8765)",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload BGE-M3 and Jina Reranker models into RAM at startup",
    )
    args = parser.parse_args()

    if args.preload:
        print("[obsidian-hybrid-rag] Pre-warming BGE-M3 and Jina Reranker models...", file=sys.stderr)
        get_embed_model()
        get_rerank_model()
        print("[obsidian-hybrid-rag] Models pre-warmed successfully.", file=sys.stderr)

    if args.transport in {"sse", "http", "streamable-http"}:
        mcp.run(transport=args.transport, host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()