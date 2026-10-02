# Obsidian Hybrid RAG MCP Server

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/Model%20Context%20Protocol-FastMCP-brightgreen.svg)](https://modelcontextprotocol.io/)
[![Embedding](https://img.shields.io/badge/Bi--Encoder-BAAI%2Fbge--m3-orange.svg)](https://huggingface.co/BAAI/bge-m3)
[![Reranker](https://img.shields.io/badge/Cross--Encoder-Jina%20Reranker%20v2-purple.svg)](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)
[![Vector Store](https://img.shields.io/badge/Vector%20DB-sqlite--vec-yellowgreen.svg)](https://github.com/asg017/sqlite-vec)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A production-grade, zero-bloat Model Context Protocol (MCP) server providing **Two-Stage Hybrid RAG** (Retrieval-Augmented Generation) for Obsidian markdown knowledge vaults.

Engineered with **SOTA dual-model AI** (`BAAI/bge-m3` 1024-dim dense embeddings + `jinaai/jina-reranker-v2` cross-encoder) and **in-process C-extensions** (`sqlite-vec` + SQLite FTS5 BM25), eliminating heavy external vector database daemons (Qdrant, Milvus, Chroma) and bulky abstraction frameworks (LangChain, LlamaIndex).

---

## 🏛️ Architecture Overview

```
                          ┌──────────────────────────┐
                          │   Obsidian Vault (.md)   │
                          └─────────────┬────────────┘
                                        │ (Heading-Aware Chunker)
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │       Single Embedded SQLite Database        │
                 │  ┌────────────────────┐ ┌──────────────────┐ │
                 │  │    SQLite FTS5     │ │    sqlite-vec    │ │
                 │  │   (BM25 Lexical)   │ │ (1024-dim Dense) │ │
                 │  └─────────┬──────────┘ └─────────┬────────┘ │
                 └────────────┼──────────────────────┼──────────┘
                              │                      │
                   [BM25 Lexical Hits]   [Cosine Distance Hits]
                              │                      │
                              └──────────┬───────────┘
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │ Stage 1: Reciprocal Rank Fusion (RRF, k=60)  │
                 │ Merges lexical + semantic into Top 15 pool   │
                 └───────────────────────┬──────────────────────┘
                                         │
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │ Stage 2: Cross-Encoder Neural Reranker       │
                 │ (jina-reranker-v2-base-multilingual ONNX)    │
                 │ Deep query-document attention scoring        │
                 └───────────────────────┬──────────────────────┘
                                         │
                                         ▼ Top K Results
                 ┌──────────────────────────────────────────────┐
                 │           FastMCP Stdio Interface            │
                 │  (Hermes Agent / Claude Desktop / Cursor)    │
                 └──────────────────────────────────────────────┘
```

---

## ✨ Key Features

1. **Anti-Bloat, In-Process Architecture**:
   - Runs entirely inside a single Python process and a single SQLite file (`vault-index.db`).
   - Native C-extension vector similarity via `sqlite-vec`. Zero Docker containers, zero network ports, zero external vector service management.

2. **State-of-the-Art Dual-Model Pipeline**:
   - **Dense Bi-Encoder**: `BAAI/bge-m3` (1024 dimensions, multilingual). The model supports 8,192 tokens, but the indexer caps `max_seq_length` at 1,024 because attention memory grows quadratically on CPU.
   - **Cross-Encoder Reranker**: `jinaai/jina-reranker-v2-base-multilingual` executed via FastEmbed ONNX Runtime with 2 execution threads.

3. **Heading-Aware Hierarchy Chunking**:
   - Preserves document outline semantics (`#`, `##`, `###`, `####`).
   - Context is injected with parent breadcrumbs (`Title > Heading > Subheading`) so vector embeddings never lose high-level context.
   - Preserves line-number traceability (`L45-L78`) for fast file jumps.

4. **Two-Stage Hybrid Search with Reciprocal Rank Fusion (RRF)**:
   - **Stage 1A**: SQLite FTS5 evaluates exact keyword matches, code identifiers, and acronyms using BM25.
   - **Stage 1B**: `sqlite-vec` retrieves semantically analogous concepts across multilingual vocabularies.
   - **Fusion**: RRF ($score = \sum \frac{1}{60 + rank}$) merges candidates into the top 15 pool.
   - **Stage 2**: Cross-encoder computes pairwise joint attention between the query and candidate passages, returning precision-ranked results.

5. **Safe vault editing for agents**:
   - Write tools are confined to the vault (path traversal and hidden folders rejected), return a `[[wikilink]]` report (broken links, missing hub link) and trigger a background re-index.
   - `vault_edit` gives agents an exact-string replace so they never need `patch`/`sed` on vault files.

6. **Memory-safe indexing**:
   - Oversized paragraphs are split by lines / hard-sliced (no chunk above `VAULT_CHUNK_CHAR_LIMIT`, default 2,500 chars).
   - One transaction per note: an interrupted run keeps its progress.
   - Run the indexer in a memory-capped cgroup in production (see below).

---

## ⚡ Performance & Benchmarks

Tested on Linux x86_64, 4 vCPU (Intel Xeon / AMD EPYC), 4 GB RAM:

| Metric | Measured Value | Note |
|---|---|---|
| **Query Latency (Stage 1 Hybrid)** | **~28 ms** | FTS5 BM25 + sqlite-vec 1024-dim |
| **Query Latency (Stage 2 Rerank)** | **~245 ms** | Jina Reranker v2 ONNX 15 candidates |
| **End-to-End Query Latency** | **< 280 ms** | Models already loaded (warm) |
| **Cold model load** | **~3-30 s** | First semantic query after idle unload; depends on CPU contention |
| **Keyword mode** | **< 10 ms** | FTS5 only, no model load |
| **Incremental Sync Time** | **< 0.1 s** | Unmodified notes bypassed via SHA256 hashes |
| **Peak Inference Memory (RSS)** | **~2.4 GB** | PyTorch CPU (BGE-M3) + FastEmbed ONNX |
| **Indexing Throughput** | **~8-10 chunks/sec** | 4-thread CPU batch matrix computation |

---

## 🚀 Installation & Quickstart

### 1. Prerequisites

- Python 3.10, 3.11, or 3.12
- Recommended: [`uv`](https://github.com/astral-sh/uv) package manager for ultra-fast virtualenv management.

### 2. Clone and Setup

```bash
git clone https://github.com/mpandudc/obsidian-hybrid-rag-mcp.git
cd obsidian-hybrid-rag-mcp

# Create dedicated virtualenv
uv venv .venv
source .venv/bin/activate

# Install dependencies (CPU optimized)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
```

### 3. Environment Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OBSIDIAN_VAULT_PATH` (or `VAULT_PATH`) | `~/vaults/pandu-second-brain` | Vault root |
| `VAULT_INDEX_DB` (or `INDEX_DB_PATH`) | `~/.hermes/vault-index.db` | SQLite index file |
| `VAULT_INDEXER_COMMAND` | `flock -n /tmp/vault-indexer.lock python -m src.indexer ...` | Command run after every write tool (point it at a memory-capped wrapper) |
| `VAULT_MAX_CHUNKS_PER_NOTE` | `2` | Result diversity cap per note |
| `MODEL_IDLE_TIMEOUT` | `300` | Seconds before models are unloaded from RAM |
| `VAULT_CHUNK_CHAR_LIMIT` | `2500` | Indexer: max characters per chunk |
| `VAULT_EMBED_MAX_SEQ_LENGTH` | `1024` | Indexer: token cap per chunk for bge-m3 |
| `VAULT_EMBED_BATCH_SIZE` | `8` | Indexer: encode batch size |
| `FASTEMBED_CACHE_DIR` | `~/.cache/fastembed` | Reranker model cache |

```bash
export OBSIDIAN_VAULT_PATH="/path/to/your/obsidian-vault"
export VAULT_INDEX_DB="$HOME/.config/obsidian-mcp/vault-index.db"
```

### 4. Build Initial Index

Run the indexer to parse markdown notes, generate BGE-M3 dense embeddings, and populate FTS5 indices:

```bash
# Full initial build
vault-indexer --vault-path "/path/to/your/obsidian-vault" --rebuild

# Or incremental sync
vault-indexer --vault-path "/path/to/your/obsidian-vault"
```

---

## 🔌 MCP Client Configuration

### Claude Desktop (`claude_desktop_config.json`)

On macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`  
On Windows: `%APPDATA%\Claude\claude_desktop_config.json`  
On Linux: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "obsidian-vault": {
      "command": "/path/to/obsidian-hybrid-rag-mcp/.venv/bin/python",
      "args": ["-m", "src.server"],
      "env": {
        "VAULT_PATH": "/path/to/your/obsidian-vault",
        "INDEX_DB_PATH": "/path/to/obsidian-hybrid-rag-mcp/vault-index.db"
      }
    }
  }
}
```

### Hermes Agent CLI / Ecosystem

**Standard Stdio:**
```bash
hermes mcp add vault --command "/path/to/obsidian-hybrid-rag-mcp/.venv/bin/python -m src.server"
```

**Shared SSE Daemon Mode (Recommended for Multi-Agent setups):**
Run one daemon so every profile/session shares a single model copy (loaded lazily, unloaded after `MODEL_IDLE_TIMEOUT`):
```bash
vault-mcp --transport sse --host 127.0.0.1 --port 8765
```
> **Security:** the write tools have no authentication. Keep `--host 127.0.0.1` (the default); the server prints a warning if bound elsewhere.

Register with Hermes:
```bash
hermes config set mcp_servers.vault.url http://127.0.0.1:8765/sse
hermes config set mcp_servers.vault.transport sse
```

Or configure via `systemd` user service (`~/.config/systemd/user/vault-mcp.service`):
```ini
[Unit]
Description=Obsidian Hybrid RAG FastMCP Daemon (SSE)
After=network.target

[Service]
Type=simple
ExecStart=/path/to/.venv/bin/python -m src.server --transport sse --host 127.0.0.1 --port 8765
Environment=VAULT_INDEXER_COMMAND=/path/to/run-vault-indexer.sh
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

---

## 🛠️ MCP Tools Exposed

| Tool | Purpose |
|---|---|
| `vault_search(query, limit=5, mode="hybrid", folder="")` | Hybrid BM25 + bge-m3 + cross-encoder search. `mode`: `hybrid` / `keyword` (instant, no model) / `semantic`. Stopwords (ID/EN) dropped from the BM25 query, folder filter oversamples vector KNN, max 2 chunks per note. |
| `vault_read(rel_path, heading="")` | Full note or one heading section. Unique bare names resolve; ambiguous names return candidates. |
| `vault_list(folder="", limit=200)` | Notes and titles under a folder. |
| `vault_backlinks(note, limit=50)` | Notes linking to a note, with the linking line. |
| `vault_write(rel_path, content, title="", tags="")` | Create/overwrite with frontmatter (`created`/`updated`). |
| `vault_append(rel_path, content, heading="")` | Append at the end or under a heading (heading created if missing). |
| `vault_edit(rel_path, old_text, new_text, replace_all=False)` | Exact-string replace; refuses missing or ambiguous matches. |
| `vault_status()` | `OK` / `INDEXING` / `STALE` (with pending notes) / `INCONSISTENT`, counts, last index write, model RAM state. |

Write tools return a link report, clear the search cache and trigger `VAULT_INDEXER_COMMAND`. The search cache is also keyed on the index DB mtime, so results never outlive an indexer commit.

### Production memory cap (systemd)

```bash
flock -n /tmp/vault-indexer.lock   systemd-run --user --scope --quiet -p MemoryMax=3G -p MemorySwapMax=512M -p CPUQuota=150% --nice=10   python -m src.indexer
```

---

## 🧪 Testing

Run the included test suite:

```bash
pytest tests/
```

---

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).

Developed with ❤️ by [Muhammad Pandu Dwi Cahyo](https://github.com/mpandudc).
