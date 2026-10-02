"""
Incremental vault indexer: SQLite FTS5 (BM25) + sqlite-vec dense vectors (BAAI/bge-m3).

Schema is the one queried by src/server.py:
  notes(rel_path, title, summary, mtime, content_hash)
  chunks(id, rel_path, title, heading, summary, chunk_text, line_start, line_end)
  chunks_fts  - FTS5 over title/heading/summary/chunk_text (rowid = chunks.id)
  vec_chunks  - vec0 float[1024] cosine (rowid = chunks.id)

Memory safety: chunks are hard-capped by the chunker, the embedder's max_seq_length
is capped, and every note is committed on its own so an interrupted run keeps its
progress. Run it inside a memory-capped cgroup/scope in production.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import struct
import sys
import time
from pathlib import Path

from src.chunker import chunk_markdown, extract_summary, extract_title

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEFAULT_VAULT_PATH = (os.getenv("OBSIDIAN_VAULT_PATH") or os.getenv("VAULT_PATH")
                      or str(Path.home() / "vaults" / "pandu-second-brain"))
DEFAULT_DB_PATH = (os.getenv("VAULT_INDEX_DB") or os.getenv("INDEX_DB_PATH")
                   or str(Path.home() / ".hermes" / "vault-index.db"))
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024
CHUNK_CHAR_LIMIT = int(os.getenv("VAULT_CHUNK_CHAR_LIMIT", "2500"))
# bge-m3 supports 8192 tokens, but attention memory grows quadratically; 1024 keeps CPU indexing small.
EMBED_MAX_SEQ_LENGTH = int(os.getenv("VAULT_EMBED_MAX_SEQ_LENGTH", "1024"))
EMBED_BATCH_SIZE = int(os.getenv("VAULT_EMBED_BATCH_SIZE", "8"))
SKIP_DIRS = {".git", ".obsidian", ".trash", ".templates", ".smart-env", ".trash-bin"}


def serialize_f32(vec: list[float]) -> bytes:
    """Pack floats into raw float32 bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


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


def init_db(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rel_path TEXT UNIQUE,
                title TEXT,
                summary TEXT,
                mtime REAL,
                content_hash TEXT
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
                embedding float[{EMBED_DIM}] distance_metric=cosine
            );
        """)


def delete_note(conn: sqlite3.Connection, rel_path: str) -> None:
    conn.execute("DELETE FROM vec_chunks WHERE rowid IN (SELECT id FROM chunks WHERE rel_path = ?);", (rel_path,))
    conn.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE rel_path = ?);", (rel_path,))
    conn.execute("DELETE FROM chunks WHERE rel_path = ?;", (rel_path,))
    conn.execute("DELETE FROM notes WHERE rel_path = ?;", (rel_path,))


def scan_vault(vault_path: Path):
    """Yield (rel_path, abs_path) for every visible markdown note."""
    for root, dirs, files in os.walk(vault_path):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            if name.endswith(".md") and not name.startswith("."):
                full = Path(root) / name
                yield str(full.relative_to(vault_path)), full


def load_embedder():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu", local_files_only=True)
    model.max_seq_length = EMBED_MAX_SEQ_LENGTH
    return model


def build_index(vault_path: Path, db_path: Path, rebuild: bool = False, embedder=None) -> dict:
    """Run an incremental (or full) index pass. Returns counters."""
    start = time.time()
    conn = connect(db_path)
    if rebuild:
        with conn:
            for table in ("vec_chunks", "chunks_fts", "chunks", "notes"):
                conn.execute(f"DROP TABLE IF EXISTS {table};")
    init_db(conn)

    existing = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT rel_path, mtime, content_hash FROM notes")}
    to_process, current = [], set()
    unchanged = 0
    for rel, full in scan_vault(vault_path):
        current.add(rel)
        try:
            mtime = full.stat().st_mtime
            content = full.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"[indexer] skip {rel}: {e}", file=sys.stderr)
            continue
        chash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        prev = existing.get(rel)
        if prev and prev[1] == chash and abs(prev[0] - mtime) < 0.001:
            unchanged += 1
            continue
        to_process.append((rel, full, mtime, chash, content))

    deleted = [rel for rel in existing if rel not in current]
    with conn:
        for rel in deleted:
            delete_note(conn, rel)

    if to_process and embedder is None:
        embedder = load_embedder()

    indexed = 0
    for i, (rel, full, mtime, chash, content) in enumerate(to_process, start=1):
        title = extract_title(content, fallback=full.stem)
        summary = extract_summary(content)
        chunks = chunk_markdown(content, title, summary, chunk_char_limit=CHUNK_CHAR_LIMIT)
        texts = [f"Document: {title} > {c.heading}\n{c.text}" for c in chunks]
        vectors = embedder.encode(texts, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True,
                                  show_progress_bar=False) if texts else []
        # One transaction per note: an interrupted run keeps everything committed so far.
        with conn:
            delete_note(conn, rel)
            conn.execute(
                "INSERT INTO notes (rel_path, title, summary, mtime, content_hash) VALUES (?, ?, ?, ?, ?);",
                (rel, title, summary, mtime, chash),
            )
            for c, vec in zip(chunks, vectors):
                cur = conn.execute(
                    "INSERT INTO chunks (rel_path, title, heading, summary, chunk_text, line_start, line_end) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?);",
                    (rel, title, c.heading, summary, c.text, c.line_start, c.line_end),
                )
                cid = cur.lastrowid
                conn.execute(
                    "INSERT INTO chunks_fts (rowid, rel_path, title, heading, summary, chunk_text, line_start, line_end) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                    (cid, rel, title, c.heading, summary, c.text, c.line_start, c.line_end),
                )
                conn.execute("INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?);",
                             (cid, serialize_f32([float(x) for x in vec])))
        indexed += 1
        print(f"[indexer] {i}/{len(to_process)} {rel}: {len(chunks)} chunks", flush=True)

    conn.close()
    stats = {"indexed": indexed, "unchanged": unchanged, "deleted": len(deleted),
             "seconds": round(time.time() - start, 2)}
    print(f"[indexer] done: {stats}", flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Obsidian Hybrid RAG indexer (BGE-M3 + SQLite FTS5 + sqlite-vec)")
    parser.add_argument("--vault-path", default=DEFAULT_VAULT_PATH, help="Obsidian vault root")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="SQLite index database file")
    parser.add_argument("--rebuild", action="store_true", help="Drop all index tables and rebuild")
    args = parser.parse_args()
    build_index(
        vault_path=Path(args.vault_path).expanduser().resolve(),
        db_path=Path(args.db_path).expanduser().resolve(),
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
