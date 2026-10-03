import os
import sqlite3
import time

from conftest import FakeEmbedder, call

from obsidian_hybrid_rag_mcp import eval as vault_eval
from obsidian_hybrid_rag_mcp.vaultfs import FileLock, atomic_write, is_ignored


def _rows(db, sql, *params):
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _reindex(vault):
    return vault["indexer"].build_index(vault["root"], vault["db"], embedder=FakeEmbedder())


# ---------------------------------------------------------------- vault_read


def test_read_pages_large_notes(vault):
    s = vault["server"]
    big = "# Big\n\n[[README]]\n\n" + "\n".join(f"line {i} " + "x" * 80 for i in range(400))
    (vault["root"] / "big.md").write_text(big, encoding="utf-8")
    page = call(s.vault_read, "big", max_chars=2000)
    assert len(page) < 3000
    assert "lines 1-" in page and "of 404" in page and "sha256" in page
    assert "Continue with vault_read(start_line=" in page
    nxt = int(page.rsplit("start_line=", 1)[1].split(")")[0])
    page2 = call(s.vault_read, "big", start_line=nxt, max_chars=2000)
    assert f"lines {nxt}-" in page2


def test_read_missing_heading_lists_headings_not_whole_note(vault):
    s = vault["server"]
    res = call(s.vault_read, "homeserver-setup", heading="Nope")
    assert "not found" in res and "## Proxmox" in res and "## Backup" in res
    assert "vzdump" not in res


def test_read_single_giant_line_is_cut(vault):
    s = vault["server"]
    (vault["root"] / "blob.md").write_text("# Blob\n" + "z" * 100000 + "\n", encoding="utf-8")
    res = call(s.vault_read, "blob", start_line=2, max_chars=1000)
    assert len(res) < 1500 and "line truncated" in res


# ---------------------------------------------------------------- writes


def test_write_refuses_overwrite_and_backs_up(vault):
    s = vault["server"]
    res = call(s.vault_write, "server/homeserver-setup.md", "# New\n\n[[README]]\n")
    assert "already exists" in res
    res = call(s.vault_write, "server/homeserver-setup.md", "# New\n\n[[README]]\n", overwrite=True)
    assert "backed up to `.trash/vault-mcp/server/homeserver-setup." in res
    backups = list((vault["root"] / ".trash" / "vault-mcp" / "server").glob("homeserver-setup.*.md"))
    assert len(backups) == 1 and "vzdump" in backups[0].read_text(encoding="utf-8")


def test_expected_hash_blocks_lost_updates(vault):
    s = vault["server"]
    page = call(s.vault_read, "homeserver-setup")
    digest = page.split("sha256 ", 1)[1].split()[0]
    p = vault["root"] / "server" / "homeserver-setup.md"
    p.write_text(p.read_text(encoding="utf-8") + "\nedited on phone\n", encoding="utf-8")
    res = call(s.vault_edit, "homeserver-setup", "Nightly", "Daily", expected_hash=digest)
    assert "changed since you read it" in res
    fresh = call(s.vault_read, "homeserver-setup").split("sha256 ", 1)[1].split()[0]
    assert "Successfully edited" in call(s.vault_edit, "homeserver-setup", "Nightly", "Daily", expected_hash=fresh)
    assert "changed since" in call(s.vault_append, "homeserver-setup", "x", expected_hash=digest)


def test_atomic_write_leaves_no_temp_files(tmp_path):
    p = tmp_path / "n.md"
    atomic_write(p, "a\r\nb\n")
    assert p.read_bytes() == b"a\r\nb\n"
    assert [f.name for f in tmp_path.iterdir()] == ["n.md"]


def test_edit_preserves_crlf(vault):
    s = vault["server"]
    p = vault["root"] / "crlf.md"
    p.write_bytes(b"# CRLF\r\n\r\n[[README]]\r\n\r\nalpha\r\nbeta\r\n")
    assert "Successfully edited" in call(s.vault_edit, "crlf", "alpha\nbeta", "gamma\ndelta")
    assert p.read_bytes() == b"# CRLF\r\n\r\n[[README]]\r\n\r\ngamma\r\ndelta\r\n"


def test_append_under_heading_ignores_fenced_comment(vault):
    s = vault["server"]
    p = vault["root"] / "fence.md"
    p.write_text("# F\n\n[[README]]\n\n## Steps\n\n```bash\n# Steps here\necho hi\n```\n\n## Other\n\nx\n",
                 encoding="utf-8")
    call(s.vault_append, "fence", "- new step", heading="Steps")
    text = p.read_text(encoding="utf-8")
    assert text.index("echo hi") < text.index("- new step") < text.index("## Other")
    assert text.count("## Steps") == 1


# ---------------------------------------------------------------- re-indexing


def test_inprocess_reindex_after_write(vault, monkeypatch):
    s = vault["server"]
    monkeypatch.setattr(s, "INDEX_MODE", "inprocess")
    res = call(s.vault_write, "server/fresh.md", "# Fresh\n\n[[README]]\n\nzanzibar quokka protocol\n")
    assert "Re-index queued (in-process)" in res
    assert s._worker.wait(10)
    assert s._worker.last_error is None
    assert "server/fresh.md" in call(s.vault_search, "zanzibar quokka", mode="keyword")


def test_worker_retries_when_lock_held(vault, monkeypatch):
    s = vault["server"]
    monkeypatch.setattr(s, "INDEX_MODE", "inprocess")
    monkeypatch.setattr(s, "LOCK_RETRY_SECONDS", 0.05)
    lock = FileLock(vault["indexer"].lock_path(vault["db"]))
    assert lock.acquire()
    try:
        (vault["root"] / "late.md").write_text("# Late\n\n[[README]]\n\nlatecomer wombat\n", encoding="utf-8")
        s.trigger_background_reindex()
        time.sleep(0.3)
        assert s._worker._thread is not None  # still retrying, change not dropped
    finally:
        lock.release()
    assert s._worker.wait(10)
    assert "late.md" in call(s.vault_search, "latecomer wombat", mode="keyword")


def test_command_mode_without_command_fails_closed(vault, monkeypatch):
    s = vault["server"]
    monkeypatch.setattr(s, "INDEX_MODE", "command")
    monkeypatch.setattr(s, "INDEXER_COMMAND", "")
    assert "requires VAULT_INDEXER_COMMAND" in s.validate_config("stdio", "127.0.0.1", False)
    assert "NOT triggered" in s.trigger_background_reindex()


def test_legacy_indexer_runner_env_is_honoured(vault, monkeypatch):
    import importlib
    monkeypatch.delenv("VAULT_INDEX_MODE", raising=False)
    monkeypatch.setenv("INDEXER_RUNNER", "/home/x/run-vault-indexer.sh")
    s = importlib.reload(vault["server"])
    assert s.INDEX_MODE == "command" and s.INDEXER_COMMAND == "/home/x/run-vault-indexer.sh"


def test_remote_bind_refused_without_flag(vault):
    s = vault["server"]
    assert "Refusing to bind 0.0.0.0" in s.validate_config("sse", "0.0.0.0", False)
    assert s.validate_config("sse", "0.0.0.0", True) is None
    assert s.validate_config("sse", "127.0.0.1", False) is None


def test_unload_deferred_while_indexing(vault, monkeypatch):
    s = vault["server"]
    s._embed_model = object()
    monkeypatch.setattr(s._worker, "busy", True)
    s._unload_models()
    assert s._embed_model is not None
    monkeypatch.setattr(s._worker, "busy", False)
    s._unload_models()
    assert s._embed_model is None
    if s._unload_timer:
        s._unload_timer.cancel()


def test_cli_indexer_repeats_when_flagged_dirty(vault, monkeypatch):
    idx = vault["indexer"]
    calls = []
    real = idx.build_index

    def fake(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            idx.dirty_path(vault["db"]).touch()  # a writer flagged during the first pass
        return real(*a, **k)

    monkeypatch.setattr(idx, "build_index", fake)
    idx.run_until_clean(vault["root"], vault["db"], embedder=FakeEmbedder())
    assert len(calls) == 2 and not idx.dirty_path(vault["db"]).exists()


# ---------------------------------------------------------------- indexer filters & schema


def test_skip_rules_recorded(vault, monkeypatch):
    idx, root, db = vault["indexer"], vault["root"], vault["db"]
    monkeypatch.setattr(idx, "MAX_FILE_BYTES", 2000)
    (root / "huge.md").write_text("# Huge\n" + "word " * 1000, encoding="utf-8")
    (root / "private.md").write_text("---\nindex: false\n---\n# Private\n\nsecret plans\n", encoding="utf-8")
    (root / ".vaultignore").write_text("# comment\nclippings/\n*.draft.md\n", encoding="utf-8")
    (root / "clippings").mkdir()
    (root / "clippings" / "video.md").write_text("# Video\n\npromo text\n", encoding="utf-8")
    (root / "idea.draft.md").write_text("# Draft\n\ndraft text\n", encoding="utf-8")
    _reindex(vault)
    reasons = dict(_rows(db, "SELECT rel_path, skip_reason FROM notes WHERE skip_reason IS NOT NULL"))
    assert reasons["huge.md"].startswith("too large")
    assert reasons["private.md"].startswith("opted out")
    assert reasons["clippings/video.md"].startswith("ignored")
    assert reasons["idea.draft.md"].startswith("ignored")
    assert _rows(db, "SELECT count(*) FROM chunks WHERE rel_path IN ('huge.md','private.md')")[0][0] == 0
    status = call(vault["server"].vault_status)
    assert "OK — index matches vault" in status and "Skipped (4)" in status


def test_ignore_patterns():
    pats = ["clippings/", "resources/**/Livro_*.md", "*.tmp.md"]
    assert is_ignored("clippings/a.md", pats)
    assert is_ignored("resources/academic/courses/x/Livro_Macro.md", pats)
    assert is_ignored("deep/a.tmp.md", pats)
    assert not is_ignored("resources/general/x.md", pats)


def test_old_schema_is_rebuilt(vault):
    db = vault["db"]
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()
    assert "OUTDATED SCHEMA" in call(vault["server"].vault_status)
    stats = _reindex(vault)
    assert stats["indexed"] == 5
    assert _rows(db, "PRAGMA user_version")[0][0] == vault["indexer"].SCHEMA_VERSION


def test_config_change_reprocesses(vault):
    (vault["root"] / ".vaultignore").write_text("other/\n", encoding="utf-8")
    stats = _reindex(vault)
    assert stats["unchanged"] == 0 and stats["skipped"] == 1


def test_tags_and_status_stored(vault):
    (vault["root"] / "tagged.md").write_text(
        "---\ntags: [Quant, cuantum]\nstatus: VERIFIED\n---\n# Tagged\n\n[[README]]\n\nmomentum signal study\n",
        encoding="utf-8")
    _reindex(vault)
    assert _rows(vault["db"], "SELECT tags, status FROM notes WHERE rel_path='tagged.md'") == [
        (",quant,cuantum,", "verified")]


# ---------------------------------------------------------------- search


def test_search_tag_and_status_filters(vault):
    root = vault["root"]
    (root / "a.md").write_text("---\ntags: [quant]\nstatus: verified\n---\n# A\n\nmomentum alpha signal\n",
                               encoding="utf-8")
    (root / "b.md").write_text("---\ntags: [quant]\nstatus: draft\n---\n# B\n\nmomentum alpha signal\n",
                               encoding="utf-8")
    (root / "c.md").write_text("# C\n\nmomentum alpha signal\n", encoding="utf-8")
    _reindex(vault)
    s = vault["server"]
    for mode in ("keyword", "semantic"):
        res = call(s.vault_search, "momentum alpha signal", mode=mode, tags="quant")
        assert "`a.md`" in res and "`b.md`" in res and "`c.md`" not in res
        res = call(s.vault_search, "momentum alpha signal", mode=mode, tags="quant", status="verified")
        assert "`a.md`" in res and "`b.md`" not in res


def test_folder_filter_uses_vec_metadata(vault):
    from obsidian_hybrid_rag_mcp.search import vec_has_rel_path
    s = vault["server"]
    conn = s.get_db_connection()
    assert vec_has_rel_path(conn)
    conn.close()
    res = call(s.vault_search, "trading bot strategy risk", mode="semantic", folder="projects/cuantum")
    assert "projects/cuantum/" in res and "server/" not in res


def test_heading_match_outranks_summary_only_match(vault):
    root = vault["root"]
    # The lead summary is copied into every chunk of a note; it must not beat a real heading match.
    noisy = "# Noisy\n\nkangaroo appears in the lead summary.\n\n" + "\n\n".join(
        f"## Part {i}\n\nunrelated filler text {i}" for i in range(6))
    (root / "noisy.md").write_text(noisy, encoding="utf-8")
    (root / "target.md").write_text("# Target\n\n## Kangaroo Care\n\nfeeding schedule\n", encoding="utf-8")
    _reindex(vault)
    res = call(vault["server"].vault_search, "kangaroo", mode="keyword", limit=3)
    assert res.splitlines()[0].startswith("### [1] [[target]]")


def test_rerank_floor_and_full_chunk(vault, monkeypatch):
    s = vault["server"]
    seen = []

    class Reranker:
        def rerank(self, query, docs):
            seen.extend(docs)
            return [5.0 if "vzdump" in d and "vzdump" in query else -8.0 for d in docs]

    monkeypatch.setattr(s, "get_reranker", lambda: Reranker())
    monkeypatch.setattr(s, "MIN_RERANK_SCORE", -2.0)
    res = call(s.vault_search, "nightly vzdump backup", mode="hybrid")
    assert "homeserver-setup" in res and "tomato" not in res and "Score: 5.000" in res
    res = call(s.vault_search, "tomato gardening plants", mode="hybrid")
    assert "No sufficiently relevant notes" in res and "-8.00" in res
    assert seen


# ---------------------------------------------------------------- lint / move / recent


def test_lint_reports_hygiene(vault):
    root = vault["root"]
    (root / "orphan.md").write_text("---\nstatus: VERIFIED_AUDITED\n---\n# Orphan\n\n[[ghost-note]]\n",
                                    encoding="utf-8")
    (root / "junk.md").write_text("# Junk\n\n[[README]]\n\n[[null,null," + "x" * 20000 + "\n", encoding="utf-8")
    s = vault["server"]
    res = call(s.vault_lint)
    assert "`orphan.md` → [[ghost-note]]" in res
    assert "Orphans" in res and "- `orphan.md`" in res
    assert "`orphan.md`: VERIFIED_AUDITED" in res
    assert "`junk.md` (longest line" in res
    assert "null,null" not in res  # blob lines are not parsed as links
    assert "orphan.md" not in call(s.vault_lint, folder="server")


def test_move_rewrites_links(vault):
    s = vault["server"]
    root = vault["root"]
    (root / "linker.md").write_text(
        "# Linker\n\n[[README]] [[cuantum-overview]] [[projects/cuantum/cuantum-overview#Strategy|plan]] "
        "![[cuantum-overview]]\n\n```\n[[cuantum-overview]]\n```\n", encoding="utf-8")
    res = call(s.vault_move, "cuantum-overview", "projects/cuantum/archive/cuantum-hub-old.md")
    assert "Rewrote 5 link(s) in 3 note(s)" in res
    text = (root / "linker.md").read_text(encoding="utf-8")
    assert "[[README]] [[cuantum-hub-old]]" in text
    assert "[[projects/cuantum/archive/cuantum-hub-old#Strategy|plan]]" in text
    assert "![[cuantum-hub-old]]" in text
    assert "```\n[[cuantum-overview]]\n```" in text  # code untouched
    assert "[[cuantum-hub-old]]" in (root / "README.md").read_text(encoding="utf-8")
    assert not (root / "projects" / "cuantum" / "cuantum-overview.md").exists()
    assert "already exists" in call(s.vault_move, "other/notes.md", "README.md")


def test_recent_orders_by_mtime(vault):
    s = vault["server"]
    p = vault["root"] / "other" / "notes.md"
    os.utime(p, (time.time() + 100, time.time() + 100))
    res = call(s.vault_recent, limit=2)
    assert res.splitlines()[1].endswith("`other/notes.md`")


# ---------------------------------------------------------------- eval


def test_eval_metrics(vault, tmp_path):
    golden = tmp_path / "golden.json"
    golden.write_text('[{"query": "vzdump rclone backup", "expected": ["homeserver-setup"]},'
                      ' {"query": "tomato gardening", "expected": ["other/notes.md"]},'
                      ' {"query": "zzzz qqqq", "expected": ["README"]}]', encoding="utf-8")
    items = vault_eval.load_golden(golden)
    conn = vault["server"].get_db_connection()
    r = vault_eval.evaluate(conn, items, "keyword", 3)
    conn.close()
    assert abs(r["hit_at_k"] - 2 / 3) < 1e-9 and abs(r["mrr"] - 2 / 3) < 1e-9
    assert r["misses"][0]["query"] == "zzzz qqqq"
    assert "| keyword | 0.67 |" in vault_eval.format_report([r])


def test_notes_linked_from_readme_count_as_hubs(vault):
    root = vault["root"]
    (root / "README.md").write_text("# Hub\n\n[[homeserver-setup]] [[cuantum-overview]] [[project-status]]\n",
                                    encoding="utf-8")
    (root / "project-status.md").write_text("# Status\n\n[[README]]\n", encoding="utf-8")
    (root / "child.md").write_text("# Child\n\nPart of [[project-status]].\n", encoding="utf-8")
    no_hub = vault["server"].lint_vault(root).no_hub
    assert "child.md" not in no_hub and "project-status.md" not in no_hub
    assert "Link check: OK" in call(vault["server"].vault_write, "child2.md", "# C2\n\nSee [[project-status]].\n")


def test_move_skips_links_in_inline_code(vault):
    root = vault["root"]
    (root / "doc.md").write_text("# Doc\n\n[[README]] `[[cuantum-overview]]` [[cuantum-overview]]\n", encoding="utf-8")
    call(vault["server"].vault_move, "cuantum-overview", "projects/cuantum/co2.md")
    assert (root / "doc.md").read_text(encoding="utf-8") == "# Doc\n\n[[README]] `[[cuantum-overview]]` [[co2]]\n"


def test_model_failures_are_reported(vault, monkeypatch):
    s = vault["server"]

    def broken_reranker():
        raise RuntimeError("model not in cache_dir")

    def broken_embedder():
        raise OSError("bge-m3 weights missing")

    monkeypatch.setattr(s, "get_reranker", broken_reranker)
    res = call(s.vault_search, "nightly vzdump backup", mode="hybrid")
    assert "homeserver-setup" in res  # still answers from RRF
    assert "reranker unavailable" in res and "model not in cache_dir" in res
    assert "Reranker: unavailable — RuntimeError: model not in cache_dir" in call(s.vault_status)

    monkeypatch.setattr(s, "get_embed_model", broken_embedder)
    res = call(s.vault_search, "vzdump rclone", mode="semantic")
    assert "semantic search unavailable" in res and "bge-m3 weights missing" in res
    assert "Embedder: unavailable — OSError: bge-m3 weights missing" in call(s.vault_status)

    monkeypatch.setattr(s, "get_embed_model", lambda: FakeEmbedder())
    call(s.vault_search, "vzdump rclone backup", mode="semantic")
    assert "Embedder: unavailable" not in call(s.vault_status)
