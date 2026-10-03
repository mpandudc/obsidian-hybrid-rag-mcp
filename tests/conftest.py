import hashlib
import importlib
import math

import pytest


class FakeEmbedder:
    """Deterministic bag-of-words vectors: shared words => higher cosine similarity."""

    dim = 1024

    def encode(self, texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False):
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in text.lower().split():
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


NOTES = {
    "README.md": "# Hub\n\nLinks: [[homeserver-setup]] [[cuantum-overview]]\n",
    "server/homeserver-setup.md": (
        "# Homeserver Setup\n\nHub: [[README]]\n\n## Proxmox\n\nProxmox host runs LXC containers for hermes.\n\n"
        "## Backup\n\nNightly vzdump backup uploads offsite to Google Drive with rclone.\n"
    ),
    "projects/cuantum/cuantum-overview.md": (
        "# Cuantum Overview\n\n[[README]]\n\n## Strategy\n\nTrading bot strategy uses funding rate carry.\n"
    ),
    "projects/cuantum/notes.md": (
        "# Cuantum Notes\n\n[[cuantum-overview]]\n\nRisk limits and position sizing for the bot.\n"
    ),
    "other/notes.md": "# Other Notes\n\n[[README]]\n\nUnrelated gardening notes about tomato plants.\n",
    ".obsidian/secret.md": "# hidden\n",
}


@pytest.fixture()
def vault(tmp_path, monkeypatch):
    root = tmp_path / "vault"
    for rel, text in NOTES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    (tmp_path / "outside.md").write_text("# outside secret\n", encoding="utf-8")
    db = tmp_path / "index.db"
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(root))
    monkeypatch.setenv("VAULT_INDEX_DB", str(db))
    monkeypatch.setenv("VAULT_INDEX_MODE", "off")
    monkeypatch.delenv("VAULT_INDEXER_COMMAND", raising=False)
    monkeypatch.delenv("INDEXER_RUNNER", raising=False)

    from obsidian_hybrid_rag_mcp import indexer
    indexer.build_index(root, db, embedder=FakeEmbedder())

    from obsidian_hybrid_rag_mcp import server
    server = importlib.reload(server)
    monkeypatch.setattr(server, "get_embed_model", lambda: FakeEmbedder())

    def no_reranker():
        raise RuntimeError("reranker disabled in tests")

    monkeypatch.setattr(server, "get_reranker", no_reranker)
    return {"root": root, "db": db, "server": server, "indexer": indexer, "tmp": tmp_path}


def call(tool, *args, **kwargs):
    """FastMCP may wrap decorated functions in a Tool object; call the underlying function."""
    fn = getattr(tool, "fn", tool)
    return fn(*args, **kwargs)
