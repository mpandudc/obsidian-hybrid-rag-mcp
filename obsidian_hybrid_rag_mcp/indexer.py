"""
Incremental vault indexer: SQLite FTS5 (BM25) + sqlite-vec dense vectors (BAAI/bge-m3).

Schema (PRAGMA user_version = SCHEMA_VERSION; an older index is rebuilt automatically):
  notes(rel_path, title, summary, tags, status, mtime, content_hash, skip_reason, chunk_count)
  chunks(id, rel_path, title, heading, summary, chunk_text, line_start, line_end)
  chunks_fts  - FTS5 over title/heading/summary/chunk_text (rowid = chunks.id)
  vec_chunks  - vec0 float[1024] cosine + `rel_path` metadata column (rowid = chunks.id)
  meta(key, value) - config fingerprint; a changed fingerprint re-processes every note

Notes are skipped (recorded with `skip_reason`, no chunks) when matched by `.vaultignore`,
larger than VAULT_MAX_FILE_BYTES, or opted out with frontmatter `index: false`. Lines longer
than VAULT_MAX_LINE_CHARS (pasted JSON/base64 blobs) are dropped from the indexed text.

Memory safety: chunks are hard-capped by the chunker, the embedder's max_seq_length is capped,
and every note is committed on its own so an interrupted run keeps its progress. Only one
indexer runs at a time (`<db>.lock`); a writer that finds it busy touches `<db>.dirty` and the
running CLI indexer makes another pass before exiting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import struct
import sys
import time
from collections.abc import Callable
from pathlib import Path

from obsidian_hybrid_rag_mcp.chunker import chunk_markdown
from obsidian_hybrid_rag_mcp.markdown import extract_summary, extract_title, normalize_tags, parse_frontmatter
from obsidian_hybrid_rag_mcp.vaultfs import FileLock, is_ignored, iter_notes, load_ignore

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SCHEMA_VERSION = 2
DEFAULT_VAULT_PATH = (os.getenv("OBSIDIAN_VAULT_PATH") or os.getenv("VAULT_PATH")
                      or str(Path.home() / "vaults" / "pandu-second-brain"))
DEFAULT_DB_PATH = (os.getenv("VAULT_INDEX_DB") or os.getenv("INDEX_DB_PATH")
                   or str(Path.home() / ".hermes" / "vault-index.db"))
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024
CHUNK_CHAR_LIMIT = int(os.getenv("VAULT_CHUNK_CHAR_LIMIT", "1500"))
# bge-m3 supports 8192 tokens, but attention memory grows quadratically; 1024 keeps CPU indexing small.
EMBED_MAX_SEQ_LENGTH = int(os.getenv("VAULT_EMBED_MAX_SEQ_LENGTH", "1024"))
EMBED_BATCH_SIZE = int(os.getenv("VAULT_EMBED_BATCH_SIZE", "8"))
MAX_FILE_BYTES = int(os.getenv("VAULT_MAX_FILE_BYTES", str(512 * 1024)))
MAX_LINE_CHARS = int(os.getenv("VAULT_MAX_LINE_CHARS", "10000"))
DIRTY_MAX_PASSES = 10


def serialize_f32(vec: list[float]) -> bytes:
    """Pack floats into raw float32 bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def lock_path(db_path: Path) -> Path:
    return Path(f"{db_path}.lock")


def dirty_path(db_path: Path) -> Path:
    return Path(f"{db_path}.dirty")


def connect(db_path: Path) -> sqlite3.Connection:
    import sqlite_vec
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-65536;")  # 64 MB
    return conn


def drop_all(conn: sqlite3.Connection) -> None:
    with conn:
        for table in ("vec_chunks", "chunks_fts", "chunks", "notes", "meta"):
            conn.execute(f"DROP TABLE IF EXISTS {table};")


def init_db(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rel_path TEXT UNIQUE,
                title TEXT,
                summary TEXT,
                tags TEXT,
                status TEXT,
                mtime REAL,
                content_hash TEXT,
                skip_reason TEXT,
                chunk_count INTEGER DEFAULT 0
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rel_path TEXT,
                title TEXT,
                heading TEXT,
                summary TEXT,
                chunk_text TEXT,
                line_start INTEGER,
                line_end INTEGER
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_rel_path ON chunks(rel_path);")
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                rel_path UNINDEXED, title, heading, summary, chunk_text,
                line_start UNINDEXED, line_end UNINDEXED,
                tokenize = 'unicode61'
            );
        """)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                rowid INTEGER PRIMARY KEY,
                embedding float[{EMBED_DIM}] distance_metric=cosine,
                rel_path text
            );
        """)
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")


def schema_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version;").fetchone()[0]


def has_tables(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT count(*) FROM sqlite_master WHERE name = 'notes'").fetchone()[0] > 0


def delete_note(conn: sqlite3.Connection, rel_path: str) -> None:
    conn.execute("DELETE FROM vec_chunks WHERE rowid IN (SELECT id FROM chunks WHERE rel_path = ?);", (rel_path,))
    conn.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE rel_path = ?);", (rel_path,))
    conn.execute("DELETE FROM chunks WHERE rel_path = ?;", (rel_path,))
    conn.execute("DELETE FROM notes WHERE rel_path = ?;", (rel_path,))


def scan_vault(vault_path: Path):
    """Yield (rel_path, abs_path) for every visible markdown note (POSIX rel paths)."""
    for full, rel in iter_notes(vault_path):
        yield rel, full


def load_embedder():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu", local_files_only=True)
    model.max_seq_length = EMBED_MAX_SEQ_LENGTH
    return model


def config_fingerprint(ignore: list[str]) -> str:
    cfg = {"chunk": CHUNK_CHAR_LIMIT, "max_bytes": MAX_FILE_BYTES, "max_line": MAX_LINE_CHARS,
           "ignore": ignore, "model": EMBED_MODEL_NAME}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def skip_reason(rel: str, size: int, fm: dict, ignore: list[str]) -> str | None:
    if is_ignored(rel, ignore):
        return "ignored (.vaultignore)"
    if size > MAX_FILE_BYTES:
        return f"too large ({size // 1024} KB > {MAX_FILE_BYTES // 1024} KB)"
    if str(fm.get("index", "")).strip().lower() in ("false", "no", "0", "off"):
        return "opted out (index: false)"
    return None


def build_index(vault_path: Path, db_path: Path, rebuild: bool = False, embedder=None,
                embedder_factory: Callable | None = None) -> dict:
    """Run one incremental (or full) index pass. Returns counters; {'locked': True} if another
    indexer holds the lock."""
    lock = FileLock(lock_path(db_path))
    if not lock.acquire():
        return {"locked": True}
    try:
        return _build_index(vault_path, db_path, rebuild, embedder, embedder_factory)
    finally:
        lock.release()


def _build_index(vault_path, db_path, rebuild, embedder, embedder_factory) -> dict:
    start = time.time()
    conn = connect(db_path)
    if rebuild or (has_tables(conn) and schema_version(conn) != SCHEMA_VERSION):
        if not rebuild:
            print(f"[indexer] schema v{schema_version(conn)} != v{SCHEMA_VERSION}: rebuilding index", flush=True)
        drop_all(conn)
    init_db(conn)

    ignore = load_ignore(vault_path)
    fingerprint = config_fingerprint(ignore)
    row = conn.execute("SELECT value FROM meta WHERE key = 'config'").fetchone()
    config_changed = row is not None and row[0] != fingerprint

    existing = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT rel_path, mtime, content_hash FROM notes")}
    to_process, current = [], set()
    unchanged = 0
    for rel, full in scan_vault(vault_path):
        current.add(rel)
        try:
            mtime = full.stat().st_mtime
            raw = full.read_bytes()
        except OSError as e:
            print(f"[indexer] skip {rel}: {e}", file=sys.stderr)
            continue
        chash = hashlib.sha256(raw).hexdigest()
        prev = existing.get(rel)
        if not config_changed and prev and prev[1] == chash and abs(prev[0] - mtime) < 0.001:
            unchanged += 1
            continue
        to_process.append((rel, full, mtime, chash, raw))

    deleted = [rel for rel in existing if rel not in current]
    with conn:
        for rel in deleted:
            delete_note(conn, rel)
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('config', ?)", (fingerprint,))

    indexed = skipped = 0
    for i, (rel, full, mtime, chash, raw) in enumerate(to_process, start=1):
        content = raw.decode("utf-8", errors="replace")
        small = len(raw) <= MAX_FILE_BYTES
        fm = parse_frontmatter(content) if small else {}
        title = extract_title(content, fallback=full.stem) if small else full.stem
        reason = skip_reason(rel, len(raw), fm, ignore)
        tags = normalize_tags(fm.get("tags"))
        status = str(fm.get("status") or "").strip().lower()
        chunks, vectors, summary = [], [], ""
        if reason is None:
            summary = extract_summary(content)
            chunks = chunk_markdown(content, title, summary, chunk_char_limit=CHUNK_CHAR_LIMIT,
                                    max_line_chars=MAX_LINE_CHARS)
            if chunks:
                if embedder is None:
                    embedder = embedder_factory() if embedder_factory else load_embedder()
                texts = [f"Document: {title} > {c.heading}\n{c.text}" for c in chunks]
                vectors = embedder.encode(texts, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True,
                                          show_progress_bar=False)
        # One transaction per note: an interrupted run keeps everything committed so far.
        with conn:
            delete_note(conn, rel)
            conn.execute(
                "INSERT INTO notes (rel_path, title, summary, tags, status, mtime, content_hash, skip_reason, "
                "chunk_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (rel, title, summary, "," + ",".join(tags) + "," if tags else "", status, mtime, chash, reason,
                 len(chunks)),
            )
            for c, vec in zip(chunks, vectors, strict=True):
                cur = conn.execute(
                    "INSERT INTO chunks (rel_path, title, heading, summary, chunk_text, line_start, line_end) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?);",
                    (rel, title, c.heading, summary, c.text, c.line_start, c.line_end),
                )
                cid = cur.lastrowid
                conn.execute(
                    "INSERT INTO chunks_fts (rowid, rel_path, title, heading, summary, chunk_text, line_start, "
                    "line_end) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                    (cid, rel, title, c.heading, summary, c.text, c.line_start, c.line_end),
                )
                conn.execute("INSERT INTO vec_chunks (rowid, embedding, rel_path) VALUES (?, ?, ?);",
                             (cid, serialize_f32([float(x) for x in vec]), rel))
        if reason:
            skipped += 1
            print(f"[indexer] {i}/{len(to_process)} {rel}: skipped, {reason}", flush=True)
        else:
            indexed += 1
            print(f"[indexer] {i}/{len(to_process)} {rel}: {len(chunks)} chunks", flush=True)

    conn.close()
    stats = {"indexed": indexed, "skipped": skipped, "unchanged": unchanged, "deleted": len(deleted),
             "seconds": round(time.time() - start, 2)}
    print(f"[indexer] done: {stats}", flush=True)
    return stats


def run_until_clean(vault_path: Path, db_path: Path, rebuild: bool = False, embedder=None) -> dict:
    """Index, then repeat while writers flagged `<db>.dirty` during the pass."""
    flag = dirty_path(db_path)
    stats: dict = {}
    for _ in range(DIRTY_MAX_PASSES):
        flag.unlink(missing_ok=True)
        stats = build_index(vault_path, db_path, rebuild=rebuild, embedder=embedder)
        if stats.get("locked"):
            # Leave the flag so the indexer holding the lock makes another pass.
            flag.touch()
            print("[indexer] another indexer holds the lock; flagged it dirty", flush=True)
            return stats
        rebuild = False
        if not flag.exists():
            break
    return stats


def main():
    parser = argparse.ArgumentParser(description="Obsidian Hybrid RAG indexer (BGE-M3 + SQLite FTS5 + sqlite-vec)")
    parser.add_argument("--vault-path", default=DEFAULT_VAULT_PATH, help="Obsidian vault root")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="SQLite index database file")
    parser.add_argument("--rebuild", action="store_true", help="Drop all index tables and rebuild")
    args = parser.parse_args()
    run_until_clean(
        vault_path=Path(args.vault_path).expanduser().resolve(),
        db_path=Path(args.db_path).expanduser().resolve(),
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
