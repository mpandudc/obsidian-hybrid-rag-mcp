"""
Two-stage hybrid retrieval over the vault index.

Stage 1: FTS5 BM25 (weighted: heading > title > body > note summary) and sqlite-vec KNN,
fused with Reciprocal Rank Fusion. Stage 2: optional cross-encoder rerank with an optional
score floor, then per-note diversity. Kept free of MCP/model globals so the server and the
eval harness share it.
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

RRF_K = 60
# bm25() weights per chunks_fts column: rel_path, title, heading, summary, chunk_text, line_start, line_end.
# `summary` is note-level text repeated in every chunk, so it only nudges ranking.
FTS_WEIGHTS = (0.0, 2.0, 3.0, 0.2, 1.0, 0.0, 0.0)

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


@dataclass
class Hit:
    chunk_id: int
    rel_path: str
    title: str
    heading: str
    chunk_text: str
    line_start: int
    line_end: int
    score: float
    reranked: bool


@dataclass
class SearchOutcome:
    hits: list[Hit]
    reranked: bool = False
    best_rejected_score: float | None = None  # set when every candidate fell under min_score


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


def folder_bounds(folder: str) -> tuple[str, str] | None:
    """Half-open [lo, hi) range of rel paths under `folder` ('/' + 1 == '0')."""
    clean = folder.strip().replace("\\", "/").strip("/ ")
    if not clean:
        return None
    return clean + "/", clean + "0"


def split_csv(value: str) -> list[str]:
    return [v.strip().lstrip("#").lower() for v in value.split(",") if v.strip().lstrip("#")]


def note_filter(tags: str, status: str) -> tuple[str, list]:
    """SQL predicate on `rel_path` for notes having ALL `tags` and ANY of `status`."""
    conds, params = [], []
    for tag in split_csv(tags):
        conds.append("tags LIKE ?")
        params.append(f"%,{tag},%")
    statuses = split_csv(status)
    if statuses:
        conds.append(f"status IN ({','.join('?' for _ in statuses)})")
        params.extend(statuses)
    if not conds:
        return "", []
    where = " AND ".join(conds)  # fixed column predicates; values are bound parameters
    return f"rel_path IN (SELECT rel_path FROM notes WHERE {where})", params  # noqa: S608


def vec_has_rel_path(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'vec_chunks'").fetchone()
    return bool(row) and "rel_path" in (row[0] or "")


def _fts(conn, query: str, bounds, nf_sql: str, nf_params: list, n: int) -> list[int]:
    fts_query = build_fts_query(query)
    if not fts_query:
        return []
    where, params = ["chunks_fts MATCH ?"], [fts_query]
    if bounds:
        where.append("rel_path >= ? AND rel_path < ?")
        params.extend(bounds)
    if nf_sql:
        where.append(nf_sql)
        params.extend(nf_params)
    weights = ", ".join(str(w) for w in FTS_WEIGHTS)
    sql = (f"SELECT rowid, bm25(chunks_fts, {weights}) AS score FROM chunks_fts "  # noqa: S608 - constants only
           f"WHERE {' AND '.join(where)} ORDER BY score LIMIT ?")
    try:
        return [r[0] for r in conn.execute(sql, [*params, n]).fetchall()]
    except sqlite3.Error:
        return []


def _vec(conn, q_blob: bytes, bounds, nf_sql: str, nf_params: list, n: int) -> list[int]:
    native = bounds is not None and vec_has_rel_path(conn)
    post_filter = bool(nf_sql) or (bounds is not None and not native)
    k = min(4096, max(300, n * 25)) if post_filter else n
    sql = "SELECT rowid FROM vec_chunks WHERE embedding MATCH ? AND k = ?"
    params: list = [q_blob, k]
    if native:
        sql += " AND rel_path >= ? AND rel_path < ?"
        params.extend(bounds)
    ids = [r[0] for r in conn.execute(sql + " ORDER BY distance", params).fetchall()]
    if post_filter and ids:
        conds, cparams = [], []
        if bounds is not None and not native:
            conds.append("rel_path >= ? AND rel_path < ?")
            cparams.extend(bounds)
        if nf_sql:
            conds.append(nf_sql)
            cparams.extend(nf_params)
        placeholders = ",".join("?" for _ in ids)
        keep = {r[0] for r in conn.execute(
            f"SELECT id FROM chunks WHERE id IN ({placeholders}) AND {' AND '.join(conds)}",  # noqa: S608
            [*ids, *cparams])}
        ids = [i for i in ids if i in keep]
    return ids[:n]


def search(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 5,
    mode: str = "hybrid",
    folder: str = "",
    tags: str = "",
    status: str = "",
    embed_fn: Callable[[str], bytes] | None = None,
    rerank_fn: Callable[[str, list[str]], list[float]] | None = None,
    per_note: int = 2,
    min_score: float | None = None,
    rerank_chars: int = 1500,
) -> SearchOutcome:
    """Run retrieval. `embed_fn(query) -> float32 blob`; `rerank_fn(query, docs) -> scores`.
    Either may raise: semantic falls back to keyword hits, rerank falls back to RRF order."""
    bounds = folder_bounds(folder)
    nf_sql, nf_params = note_filter(tags, status)
    fetch_n = limit * 6  # headroom for per-note diversity

    ranked_lists = []
    if mode in ("hybrid", "keyword"):
        ranked_lists.append(_fts(conn, query, bounds, nf_sql, nf_params, fetch_n))
    if mode in ("hybrid", "semantic") and embed_fn is not None:
        # Model load/encode can fail many ways; keep the keyword hits.
        with contextlib.suppress(Exception):
            ranked_lists.append(_vec(conn, embed_fn(query), bounds, nf_sql, nf_params, fetch_n))

    rrf: dict[int, float] = {}
    for ids in ranked_lists:
        for rank, cid in enumerate(ids, start=1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    if not rrf:
        return SearchOutcome([])

    pool = sorted(rrf, key=lambda c: rrf[c], reverse=True)[:max(limit * 4, 16)]
    placeholders = ",".join("?" for _ in pool)
    sql = f"SELECT id, rel_path, title, heading, chunk_text, line_start, line_end FROM chunks WHERE id IN ({placeholders})"  # noqa: S608, E501
    rows = {r[0]: r for r in conn.execute(sql, pool).fetchall()}
    ordered = [cid for cid in pool if cid in rows]
    scored = [(cid, rrf[cid]) for cid in ordered]

    reranked = False
    if mode == "hybrid" and rerank_fn is not None and ordered:
        try:
            docs = [f"Title: {rows[c][2]} > {rows[c][3]}\n{rows[c][4][:rerank_chars]}" for c in ordered]
            scores = [float(s) for s in rerank_fn(query, docs)]
            scored = sorted(zip(ordered, scores, strict=True), key=lambda x: x[1], reverse=True)
            reranked = True
        except Exception:  # noqa: BLE001, S110 - reranker failure keeps the RRF order
            pass

    best_rejected = None
    if reranked and min_score is not None:
        kept = [s for s in scored if s[1] >= min_score]
        if not kept and scored:
            best_rejected = scored[0][1]
        scored = kept

    rel_of = {cid: rows[cid][1] for cid in ordered}
    hits = [
        Hit(cid, rows[cid][1], rows[cid][2], rows[cid][3], rows[cid][4], rows[cid][5], rows[cid][6], score, reranked)
        for cid, score in diversify(scored, rel_of, limit, per_note)
    ]
    return SearchOutcome(hits, reranked, best_rejected)
