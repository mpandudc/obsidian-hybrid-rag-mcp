# Obsidian Hybrid RAG MCP Server

[![CI](https://github.com/mpandudc/obsidian-hybrid-rag-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/mpandudc/obsidian-hybrid-rag-mcp/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/Model%20Context%20Protocol-FastMCP-brightgreen.svg)](https://modelcontextprotocol.io/)
[![Embedding](https://img.shields.io/badge/Bi--Encoder-BAAI%2Fbge--m3-orange.svg)](https://huggingface.co/BAAI/bge-m3)
[![Reranker](https://img.shields.io/badge/Cross--Encoder-Jina%20Reranker%20v2-purple.svg)](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)
[![Vector Store](https://img.shields.io/badge/Vector%20DB-sqlite--vec-yellowgreen.svg)](https://github.com/asg017/sqlite-vec)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A local Model Context Protocol (MCP) server that gives agents **two-stage hybrid search** and **safe editing tools** over an Obsidian markdown vault.

Search combines SQLite FTS5 (BM25) and `sqlite-vec` dense vectors (`BAAI/bge-m3`), fuses them with Reciprocal Rank Fusion and reranks with a `jinaai/jina-reranker-v2` cross-encoder — all in one Python process and one SQLite file, without a vector database daemon or a RAG framework.

---

## 🏛️ Architecture Overview

```
                          ┌──────────────────────────┐
                          │   Obsidian Vault (.md)   │
                          └─────────────┬────────────┘
                 .vaultignore / size cap │ (fence-aware heading chunker)
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │       Single Embedded SQLite Database        │
                 │  ┌────────────────────┐ ┌──────────────────┐ │
                 │  │    SQLite FTS5     │ │    sqlite-vec    │ │
                 │  │ (weighted BM25)    │ │ (1024-dim + path │ │
                 │  │                    │ │  metadata)       │ │
                 │  └─────────┬──────────┘ └─────────┬────────┘ │
                 └────────────┼──────────────────────┼──────────┘
                              └──────────┬───────────┘
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │ Stage 1: Reciprocal Rank Fusion (RRF, k=60)  │
                 │ + folder / tag / status filters              │
                 └───────────────────────┬──────────────────────┘
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │ Stage 2: Cross-Encoder Reranker (optional    │
                 │ score floor) + max 2 chunks per note         │
                 └───────────────────────┬──────────────────────┘
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │   FastMCP (stdio / SSE / streamable HTTP)    │
                 │  (Hermes Agent / Claude Desktop / Cursor)    │
                 └──────────────────────────────────────────────┘
```

---

## ✨ Key Features

1. **In-process, single-file index.** FTS5 and `sqlite-vec` live in one `vault-index.db`. The index schema is versioned (`PRAGMA user_version`); an outdated index is rebuilt automatically.
2. **Fence-aware, line-exact chunking.** Sections split on real headings only — a `# comment` inside a fenced code block is code, not a heading. Chunks never exceed `VAULT_CHUNK_CHAR_LIMIT` and report exact source line ranges. Frontmatter is parsed (tags, status) but not embedded.
3. **Junk-resistant indexing.** Notes matched by `.vaultignore`, larger than `VAULT_MAX_FILE_BYTES`, or marked `index: false` are recorded as skipped. Lines longer than `VAULT_MAX_LINE_CHARS` (pasted JSON / base64 blobs) are dropped from the indexed text.
4. **Memory-safe re-indexing.** By default the server re-indexes in a background thread that **reuses the already-loaded embedding model**, so a write never loads a second `bge-m3`. Only one indexer runs at a time (`<db>.lock`); changes made during a pass trigger exactly one more pass instead of being dropped.
5. **Safe editing for agents.** Writes are atomic (temp file + rename), confined to the vault, refuse to overwrite unless asked (with a backup in `.trash/vault-mcp/`), support optimistic concurrency (`expected_hash`), keep CRLF line endings, and return a `[[wikilink]]` report.
6. **Vault hygiene tools.** `vault_lint` finds broken links, orphans, missing hub links / frontmatter and off-vocabulary `status:` values; `vault_move` renames a note and rewrites every link to it.
7. **Measurable retrieval.** `vault-eval` reports hit@k, recall@k and MRR per mode on a golden query set, plus the reranker score distribution to calibrate a relevance floor.

---

## 🚀 Installation & Quickstart

Python 3.10–3.12. [`uv`](https://github.com/astral-sh/uv) recommended.

```bash
git clone https://github.com/mpandudc/obsidian-hybrid-rag-mcp.git
cd obsidian-hybrid-rag-mcp
uv venv .venv && source .venv/bin/activate

# CPU-only torch first, then the package with the model extras
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[models]"
```

The core install (`pip install -e .`) is enough for keyword search and the editing / lint tools; semantic and hybrid search and indexing need the `models` extra.

Download the models once (the server runs with `HF_HUB_OFFLINE=1`):

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('jinaai/jina-reranker-v2-base-multilingual')"
```

Build the index:

```bash
vault-indexer --vault-path "/path/to/vault" --rebuild   # full build
vault-indexer --vault-path "/path/to/vault"             # incremental
```

### Environment configuration

| Variable | Default | Purpose |
|---|---|---|
| `OBSIDIAN_VAULT_PATH` (or `VAULT_PATH`) | `~/vaults/pandu-second-brain` | Vault root |
| `VAULT_INDEX_DB` (or `INDEX_DB_PATH`) | `~/.hermes/vault-index.db` | SQLite index file |
| `VAULT_INDEX_MODE` | `inprocess` (`command` if `VAULT_INDEXER_COMMAND` is set) | `inprocess` / `command` / `off` — how write tools re-index |
| `VAULT_INDEXER_COMMAND` (legacy `INDEXER_RUNNER`) | — | Command for `command` mode (e.g. a memory-capped wrapper). Required in that mode; the server refuses to start without it |
| `VAULT_MAX_FILE_BYTES` | `524288` | Notes above this size are skipped |
| `VAULT_MAX_LINE_CHARS` | `10000` | Longer lines are dropped from indexed text |
| `VAULT_CHUNK_CHAR_LIMIT` | `1500` | Max characters per chunk |
| `VAULT_EMBED_MAX_SEQ_LENGTH` | `1024` | Token cap per chunk for bge-m3 |
| `VAULT_EMBED_BATCH_SIZE` | `8` | Encode batch size |
| `VAULT_MAX_CHUNKS_PER_NOTE` | `2` | Result diversity cap per note |
| `VAULT_RERANK_CHARS` | chunk limit | Characters of each chunk shown to the reranker |
| `VAULT_MIN_RERANK_SCORE` | unset (off) | Drop reranked results below this score — calibrate with `vault-eval` |
| `VAULT_READ_MAX_CHARS` | `8000` | Default `vault_read` page size |
| `VAULT_STATUS_VALUES` | `draft,active,approved,verified,completed,falsified,superseded,archived` | Allowed frontmatter `status` values for `vault_lint` |
| `MODEL_IDLE_TIMEOUT` | `300` | Seconds before models are unloaded from RAM |
| `FASTEMBED_CACHE_DIR` | `~/.cache/fastembed` | Reranker model cache |
| `MCP_ALLOW_REMOTE` | unset | `1` allows binding a non-loopback address |

### `.vaultignore`

Optional file in the vault root; one glob per line, `#` for comments:

```gitignore
# a folder
clippings/
# a path pattern
resources/**/Livro_*.md
# a file-name pattern
*.draft.md
```

Skipped notes stay readable through `vault_read`; they are only left out of search. `vault_status` lists them with the reason.

---

## 🔌 MCP Client Configuration

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "obsidian-vault": {
      "command": "/path/to/obsidian-hybrid-rag-mcp/.venv/bin/vault-mcp",
      "env": {
        "OBSIDIAN_VAULT_PATH": "/path/to/your/obsidian-vault",
        "VAULT_INDEX_DB": "/path/to/vault-index.db"
      }
    }
  }
}
```

### Shared SSE daemon (recommended for multi-agent setups)

One daemon means one model copy for every agent profile:

```ini
# ~/.config/systemd/user/vault-mcp.service
[Unit]
Description=Obsidian Hybrid RAG FastMCP Daemon (SSE)
After=network.target

[Service]
Type=simple
ExecStart=/path/to/.venv/bin/vault-mcp --transport sse --host 127.0.0.1 --port 8765
Environment=OBSIDIAN_VAULT_PATH=/path/to/vault
# In-process re-indexing shares the daemon's model, so cap the daemon itself:
MemoryMax=4G
MemorySwapMax=512M
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
hermes config set mcp_servers.vault.url http://127.0.0.1:8765/sse
hermes config set mcp_servers.vault.transport sse
```

> **Security:** the write tools have no authentication. The server refuses to bind anything but loopback unless you pass `--allow-remote` (or `MCP_ALLOW_REMOTE=1`) — only do that behind an authenticating proxy.

Cron keeps the index fresh for edits made outside the MCP (Obsidian on phone/PC):

```cron
*/30 * * * * systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=512M /path/to/.venv/bin/vault-indexer
```

The CLI indexer and the daemon share `<db>.lock`, so they never index concurrently.

---

## 🛠️ MCP Tools

| Tool | Purpose |
|---|---|
| `vault_search(query, limit=5, mode="hybrid", folder="", tags="", status="")` | Hybrid / `keyword` / `semantic` search. `folder` filters natively in the vector index; `tags` (all must match) and `status` (any) filter on frontmatter. |
| `vault_read(rel_path, heading="", start_line=1, max_chars=8000)` | Paged read of a note or section. Header shows the line range and a sha256 prefix; a missing heading lists the available headings instead of dumping the note. |
| `vault_list(folder="", limit=200)` | Notes and titles under a folder. |
| `vault_recent(limit=20, folder="", days=0)` | Most recently modified notes. |
| `vault_backlinks(note, limit=50)` | Notes linking to a note, with the linking line. |
| `vault_write(rel_path, content, title="", tags="", overwrite=False, expected_hash="")` | Create a note with frontmatter; replacing one needs `overwrite=true` and backs it up. |
| `vault_append(rel_path, content, heading="", expected_hash="")` | Append at the end or under a heading (created if missing). |
| `vault_edit(rel_path, old_text, new_text, replace_all=False, expected_hash="")` | Exact-string replace; refuses missing or ambiguous matches. |
| `vault_move(src, dst, update_links=True)` | Rename/move a note and rewrite every wikilink to it (aliases, `#heading`, `![[embeds]]` kept; code blocks untouched). |
| `vault_lint(folder="", limit=30)` | Broken links, orphans, missing hub link / frontmatter, invalid `status`, oversized notes, blob lines. |
| `vault_status()` | `OK` / `INDEXING` / `STALE` / `INCONSISTENT` / `OUTDATED SCHEMA`, counts, skipped notes, index mode, model RAM state. |

---

## 📏 Evaluating retrieval

Write a golden set (see [`eval/golden.example.json`](eval/golden.example.json)) and run:

```bash
vault-eval --golden eval/golden.json --k 5
```

It prints hit@k / recall@k / MRR per mode, every miss, and the reranker score distribution of relevant vs irrelevant results. Use the relevant-score p10 to pick `VAULT_MIN_RERANK_SCORE`, and re-run after changing chunk size, weights or models.

Reference run on the author's vault (233 notes, 30 queries from `eval/golden.example.json`, k=5):

| mode | hit@5 | recall@5 | MRR |
|---|---|---|---|
| keyword | 0.97 | 0.95 | 0.79 |
| semantic | 1.00 | 1.00 | 0.92 |
| hybrid (with reranker) | 1.00 | 1.00 | 0.97 |

Relevant results scored −0.90 … 2.01 (p10 0.07); irrelevant ones −3.52 … 1.82 (median 0.01). A floor of `-1.0` kept every relevant hit.

---

## 🔁 Upgrading from 1.x

- The package moved from `src` to `obsidian_hybrid_rag_mcp`: use the `vault-mcp` / `vault-indexer` entry points (or `python -m obsidian_hybrid_rag_mcp.server`).
- The index schema is now v2. The first indexer run rebuilds it automatically (`vault_status` says `OUTDATED SCHEMA` until then).
- `vault_write` no longer overwrites silently: pass `overwrite=true`.
- Re-indexing defaults to in-process. To keep an external wrapper, set `VAULT_INDEX_MODE=command` and `VAULT_INDEXER_COMMAND` (the legacy `INDEXER_RUNNER` is still read).
- Binding a non-loopback address now requires `--allow-remote`.

---

## 🧪 Development

```bash
uv pip install -e ".[dev]"
ruff check .
pytest
```

The tests use a deterministic fake embedder, so they need neither torch nor the models.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

Developed by [Muhammad Pandu Dwi Cahyo](https://github.com/mpandudc).
