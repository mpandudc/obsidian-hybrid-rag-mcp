import sqlite3

from conftest import FakeEmbedder


def _count(db, table):
    import sqlite_vec
    conn = sqlite3.connect(str(db))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    conn.close()
    return n


def test_incremental_index_and_delete(vault):
    idx, root, db = vault["indexer"], vault["root"], vault["db"]
    assert _count(db, "notes") == 5  # hidden .obsidian note skipped
    assert _count(db, "chunks") == _count(db, "vec_chunks") == _count(db, "chunks_fts")

    stats = idx.build_index(root, db, embedder=FakeEmbedder())
    assert stats["indexed"] == 0 and stats["unchanged"] == 5

    (root / "other" / "notes.md").unlink()
    stats = idx.build_index(root, db, embedder=FakeEmbedder())
    assert stats["deleted"] == 1
    assert _count(db, "notes") == 4
    assert _count(db, "chunks") == _count(db, "vec_chunks")


def test_giant_paragraph_is_bounded(vault):
    idx, root, db = vault["indexer"], vault["root"], vault["db"]
    giant = "# Rangkuman\n\n[[README]]\n\n" + "\n".join(f"kalimat panjang nomor {i} " * 20 for i in range(400))
    (root / "rangkuman.md").write_text(giant, encoding="utf-8")
    idx.build_index(root, db, embedder=FakeEmbedder())
    conn = sqlite3.connect(str(db))
    longest = conn.execute(
        "SELECT max(length(chunk_text)) FROM chunks WHERE rel_path = 'rangkuman.md'"
    ).fetchone()[0]
    conn.close()
    assert longest <= idx.CHUNK_CHAR_LIMIT
