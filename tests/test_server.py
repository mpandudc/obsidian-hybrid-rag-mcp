import os
import time

import pytest
from conftest import FakeEmbedder, call

from obsidian_hybrid_rag_mcp.server import build_fts_query, diversify, get_db, serialize_f32


def test_serialize_f32():
    data = serialize_f32([0.1, 0.2, 0.3, -0.4])
    assert isinstance(data, bytes)
    assert len(data) == 4 * 4


def test_get_db_non_existent():
    with pytest.raises(FileNotFoundError):
        get_db("/path/to/definitely/non_existent_vault.db")


def test_fts_query_drops_stopwords_and_quotes_terms():
    q = build_fts_query("bagaimana cara backup homeserver yang aman?")
    assert '"backup"' in q and '"homeserver"' in q and '"aman"' in q
    assert '"yang"' not in q and '"bagaimana"' not in q
    # A query made only of stopwords still searches for something
    assert build_fts_query("the and") == '"the" OR "and"'
    # FTS5 keywords / hyphens are quoted, not interpreted
    assert build_fts_query("bge-m3 NOT") == '"bge-m3" OR "NOT"'


def test_diversify_caps_chunks_per_note():
    items = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)]
    rel_of = {1: "a.md", 2: "a.md", 3: "a.md", 4: "b.md"}
    assert [i for i, _ in diversify(items, rel_of, limit=5, per_note=2)] == [1, 2, 4]


def test_search_keyword_and_hybrid(vault):
    s = vault["server"]
    res = call(s.vault_search, "vzdump backup rclone", mode="keyword")
    assert "server/homeserver-setup.md" in res
    res = call(s.vault_search, "nightly vzdump backup offsite", mode="hybrid")  # reranker disabled -> RRF fallback
    assert "homeserver-setup" in res.splitlines()[0]


def test_search_folder_filter_semantic(vault):
    s = vault["server"]
    res = call(s.vault_search, "risk limits position sizing bot", mode="semantic", folder="projects/cuantum")
    assert "projects/cuantum/" in res
    assert "server/" not in res and "other/" not in res


def test_search_cache_invalidated_by_index_change(vault):
    s = vault["server"]
    first = call(s.vault_search, "tomato gardening", mode="keyword")
    assert "other/notes.md" in first
    note = vault["root"] / "server" / "new-note.md"
    note.write_text("# New\n\n[[README]]\n\ntomato tomato tomato gardening greenhouse\n", encoding="utf-8")
    time.sleep(0.05)
    vault["indexer"].build_index(vault["root"], vault["db"], embedder=FakeEmbedder())
    second = call(s.vault_search, "tomato gardening", mode="keyword")
    assert "server/new-note.md" in second


def test_read_blocks_traversal_and_hidden(vault):
    s = vault["server"]
    assert "outside the vault" in call(s.vault_read, "../outside.md")
    assert "hidden/system folder" in call(s.vault_read, ".obsidian/secret")
    assert "Access denied" in call(s.vault_write, "../../evil.md", "x")
    assert not (vault["tmp"] / "evil.md").exists()


def test_read_ambiguous_name_lists_candidates(vault):
    s = vault["server"]
    res = call(s.vault_read, "notes")
    assert "ambiguous" in res
    assert "projects/cuantum/notes.md" in res and "other/notes.md" in res
    assert "Proxmox host" in call(s.vault_read, "homeserver-setup", heading="Proxmox")


def test_write_reports_links(vault):
    s = vault["server"]
    ok = call(s.vault_write, "projects/idea.md", "# Idea\n\nSee [[README]] and [[cuantum-overview]].\n")
    assert "Link check: OK" in ok
    bad = call(s.vault_write, "projects/idea2.md", "# Idea 2\n\nSee [[does-not-exist]].\n")
    assert "broken links" in bad and "[[does-not-exist]]" in bad
    assert "no link to a hub note" in bad
    orphan = call(s.vault_write, "projects/idea3.md", "# Idea 3\n\nNo links here.\n")
    assert "no [[wikilinks]] at all" in orphan
    written = (vault["root"] / "projects" / "idea.md").read_text(encoding="utf-8")
    assert written.startswith("---\ntitle:") and "created:" in written


def test_append_under_existing_empty_heading(vault):
    s = vault["server"]
    p = vault["root"] / "server" / "log.md"
    p.write_text("# Log\n\n[[README]]\n\n## Today\n", encoding="utf-8")
    call(s.vault_append, "server/log.md", "- did a thing", heading="Today")
    text = p.read_text(encoding="utf-8")
    assert text.count("## Today") == 1
    assert text.index("- did a thing") > text.index("## Today")


def test_edit_requires_unique_match(vault):
    s = vault["server"]
    p = vault["root"] / "server" / "edit.md"
    p.write_text("# Edit\n\n[[README]]\n\nalpha beta\nalpha gamma\n", encoding="utf-8")
    assert "not found" in call(s.vault_edit, "server/edit.md", "zeta", "x")
    assert "appears 2 times" in call(s.vault_edit, "server/edit.md", "alpha", "omega")
    assert "Successfully edited" in call(s.vault_edit, "server/edit.md", "alpha beta", "omega beta")
    assert "Successfully edited" in call(s.vault_edit, "server/edit.md", "alpha", "omega", replace_all=True)
    text = p.read_text(encoding="utf-8")
    assert "alpha" not in text and text.count("omega") == 2


def test_list_and_backlinks(vault):
    s = vault["server"]
    listing = call(s.vault_list, "projects/cuantum")
    assert "2 notes" in listing and "Cuantum Overview" in listing
    assert "secret" not in call(s.vault_list)
    assert "not found" in call(s.vault_list, "../")
    back = call(s.vault_backlinks, "cuantum-overview")
    assert "README.md" in back and "projects/cuantum/notes.md" in back
    assert "orphan" in call(s.vault_backlinks, "other/notes.md")


def test_status_reports_stale_and_ok(vault, monkeypatch):
    s = vault["server"]
    monkeypatch.setattr(s, "indexer_running", lambda: False)  # host may run a real indexer
    assert "OK — index matches vault" in call(s.vault_status)
    monkeypatch.setattr(s, "indexer_running", lambda: True)
    assert "INDEXING" in call(s.vault_status)
    monkeypatch.setattr(s, "indexer_running", lambda: False)
    p = vault["root"] / "server" / "homeserver-setup.md"
    p.write_text(p.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")
    os.utime(p, (time.time() + 5, time.time() + 5))
    status = call(s.vault_status)
    assert "STALE" in status and "server/homeserver-setup.md" in status
