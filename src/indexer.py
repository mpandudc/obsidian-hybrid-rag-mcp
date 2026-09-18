"""
High-performance vault indexer using BAAI/bge-m3 dense embeddings and SQLite FTS5.
Supports incremental sync, mini-batching, and memory-safe commit loops.
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
from typing import List, Tuple

import numpy as np
import sqlite_vec
import torch
from sentence_transformers import SentenceTransformer

from src.chunker import chunk_markdown, clean_frontmatter, extract_summary, extract_title

DEFAULT_VAULT_PATH = os.getenv("VAULT_PATH", str(Path.home() / "vaults" / "pandu-second-brain"))
DEFAULT_DB_PATH = os.getenv("INDEX_DB_PATH", str(Path.home() / ".hermes" / "vault-index.db"))
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024


def serialize_f32(vec: List[float]) -> bytes:
    """Pack Python float array into raw IEEE 754 float32 bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def get_file_hash(path: Path) -> str:
    """Generate SHA256 hex digest of file contents for change detection."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with WAL mode, FTS5, and sqlite-vec virtual table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    # Optimization pragmas
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -64000;")  # 64MB cache

    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rel_path TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                file_hash TEXT NOT NULL,
                mtime REAL NOT NULL,
                indexed_at REAL NOT NULL
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id INTEGER NOT NULL,
                heading TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
            );
        """)

        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                text,
                heading,
                title,
                content='chunks',
                content_rowid='id',
                tokenize='porter unicode61'
            );
        """)

        # Triggers to keep FTS5 in sync with chunks table
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO fts_chunks(rowid, text, heading, title)
                VALUES (new.id, new.text, new.heading, (SELECT title FROM notes WHERE id = new.note_id));
            END;
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO fts_chunks(fts_chunks, rowid, text, heading, title)
                VALUES('delete', old.id, old.text, old.heading, (SELECT title FROM notes WHERE id = old.note_id));
            END;
        """)

        # sqlite-vec virtual table
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{EMBED_DIM}]
            );
        """)

    return conn


def scan_vault_files(vault_path: Path) -> List[Path]:
    """Recursively collect all non-hidden markdown files."""
    files = []
    for root, dirs, filenames in os.walk(vault_path):
        # Exclude hidden directories (like .obsidian, .git)
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".md") and not fn.startswith("."):
                files.append(Path(root) / fn)
    return sorted(files)


def build_index(
    vault_path: Path,
    db_path: Path,
    rebuild: bool = False,
    batch_size: int = 15,
) -> None:
    """Run incremental or full indexing pipeline."""
    torch.set_num_threads(4)

    print(f"[*] Connecting to database: {db_path}")
    conn = init_db(db_path)

    if rebuild:
        print("[!] Rebuild requested. Purging existing tables...")
        with conn:
            conn.execute("DELETE FROM vec_chunks;")
            conn.execute("DELETE FROM chunks;")
            conn.execute("DELETE FROM notes;")
            conn.execute("DELETE FROM fts_chunks;")
        print("[+] Tables wiped.")

    print(f"[*] Scanning vault at: {vault_path}")
    all_files = scan_vault_files(vault_path)
    print(f"[+] Discovered {len(all_files)} markdown notes.")

    # Check indexed status
    cur = conn.cursor()
    cur.execute("SELECT rel_path, file_hash, mtime FROM notes")
    existing = {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    to_process: List[Tuple[Path, str, str, float]] = []
    for f in all_files:
        rel = str(f.relative_to(vault_path))
        mtime = f.stat().st_mtime
        f_hash = get_file_hash(f)
        if rel in existing:
            prev_hash, _ = existing[rel]
            if prev_hash == f_hash:
                continue  # Note untouched
        to_process.append((f, rel, f_hash, mtime))

    # Detect deleted notes
    current_rel_set = {str(f.relative_to(vault_path)) for f in all_files}
    deleted_notes = [rel for rel in existing if rel not in current_rel_set]
    if deleted_notes:
        print(f"[-] Pruning {len(deleted_notes)} deleted notes from index...")
        with conn:
            for rel in deleted_notes:
                note_id = conn.execute("SELECT id FROM notes WHERE rel_path = ?", (rel,)).fetchone()
                if note_id:
                    nid = note_id[0]
                    chunk_ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE note_id = ?", (nid,)).fetchall()]
                    for cid in chunk_ids:
                        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
                    conn.execute("DELETE FROM chunks WHERE note_id = ?", (nid,))
                    conn.execute("DELETE FROM notes WHERE id = ?", (nid,))

    if not to_process:
        print("[+] Vault index is fully up to date. Zero changes detected.")
        conn.close()
        return

    print(f"[*] Loading SOTA Embedding Model: {EMBED_MODEL_NAME} (dimension {EMBED_DIM})...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    print("[+] Model loaded into RAM.")

    total = len(to_process)
    print(f"[*] Indexing {total} notes in mini-batches of {batch_size}...")

    start_time = time.time()
    for b_idx in range(0, total, batch_size):
        batch = to_process[b_idx : b_idx + batch_size]
        t0 = time.time()

        batch_chunks_data = []
        batch_notes_records = []

        for f_path, rel, f_hash, mtime in batch:
            try:
                raw_text = f_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"[!] Warning reading {rel}: {e}")
                continue

            title = extract_title(raw_text, fallback=f_path.stem)
            summary = extract_summary(raw_text)
            chunks = chunk_markdown(raw_text, title=title, summary=summary)

            batch_notes_records.append((rel, title, summary, f_hash, mtime, time.time(), chunks))

        # Flatten chunk texts for vectorized batch encoding
        all_chunk_texts: List[str] = []
        chunk_map: List[Tuple[int, int, str, int, int]] = []

        for note_idx, (rel, title, summary, f_hash, mtime, now_ts, chunks) in enumerate(batch_notes_records):
            for c_idx, c in enumerate(chunks):
                text_to_embed = f"{title} > {c.heading}\n{c.text}"
                all_chunk_texts.append(text_to_embed)
                chunk_map.append((note_idx, c_idx, c.heading, c.line_start, c.line_end))

        if not all_chunk_texts:
            continue

        # Compute dense embeddings in single batched matrix operation
        embeddings = embed_model.encode(
            all_chunk_texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Atomic commit per mini-batch
        with conn:
            for note_idx, (rel, title, summary, f_hash, mtime, now_ts, chunks) in enumerate(batch_notes_records):
                # Clean up existing note records if any
                prev_id = conn.execute("SELECT id FROM notes WHERE rel_path = ?", (rel,)).fetchone()
                if prev_id:
                    nid = prev_id[0]
                    cids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE note_id = ?", (nid,)).fetchall()]
                    for cid in cids:
                        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
                    conn.execute("DELETE FROM chunks WHERE note_id = ?", (nid,))
                    conn.execute("DELETE FROM notes WHERE id = ?", (nid,))

                cur = conn.execute("""
                    INSERT INTO notes (rel_path, title, summary, file_hash, mtime, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (rel, title, summary, f_hash, mtime, now_ts))
                note_id = cur.lastrowid

                # Match embeddings back to this note
                for i, (n_i, c_i, heading, l_start, l_end) in enumerate(chunk_map):
                    if n_i == note_idx:
                        chunk_obj = chunks[c_i]
                        c_cur = conn.execute("""
                            INSERT INTO chunks (note_id, heading, chunk_index, text, line_start, line_end)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (note_id, heading, c_i, chunk_obj.text, l_start, l_end))
                        chunk_id = c_cur.lastrowid

                        emb_bytes = serialize_f32(embeddings[i].tolist())
                        conn.execute("""
                            INSERT INTO vec_chunks (chunk_id, embedding)
                            VALUES (?, ?)
                        """, (chunk_id, emb_bytes))

        elapsed_b = time.time() - t0
        processed_count = min(b_idx + batch_size, total)
        print(f"[{processed_count}/{total}] Processed {len(batch)} notes ({len(all_chunk_texts)} chunks) in {elapsed_b:.2f}s")

    total_time = time.time() - start_time
    print(f"[+] Indexing completed successfully in {total_time:.2f}s.")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Obsidian Hybrid RAG Indexer (BGE-M3 + SQLite FTS5)")
    parser.add_argument("--vault-path", default=DEFAULT_VAULT_PATH, help="Path to Obsidian vault root directory")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to SQLite target database file")
    parser.add_argument("--rebuild", action="store_true", help="Wipe database and rebuild from scratch")
    parser.add_argument("--batch-size", type=int, default=15, help="Number of notes to process per batch commit")
    args = parser.parse_args()

    build_index(
        vault_path=Path(args.vault_path).expanduser().resolve(),
        db_path=Path(args.db_path).expanduser().resolve(),
        rebuild=args.rebuild,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
