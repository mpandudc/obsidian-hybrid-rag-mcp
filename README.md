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
   - **Dense Bi-Encoder**: `BAAI/bge-m3` (1024 dimensions, 8,192 token context window, multi-lingual SOTA).
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

5. **Sub-millisecond IPC**:
   - FastMCP Stdio communication with JSON-RPC over stdin/stdout. Zero HTTP overhead.

---

## ⚡ Performance & Benchmarks

Tested on Linux x86_64, 4 vCPU (Intel Xeon / AMD EPYC), 4 GB RAM:

| Metric | Measured Value | Note |
|---|---|---|
| **Query Latency (Stage 1 Hybrid)** | **~28 ms** | FTS5 BM25 + sqlite-vec 1024-dim |
| **Query Latency (Stage 2 Rerank)** | **~245 ms** | Jina Reranker v2 ONNX 15 candidates |
| **End-to-End Query Latency** | **< 280 ms** | Cold cache to final JSON response |
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

Create a `.env` or set environment variables:

```bash
# Path to your local Obsidian vault root
export VAULT_PATH="/path/to/your/obsidian-vault"

# Path to the persistent SQLite index database
export INDEX_DB_PATH="$HOME/.config/obsidian-mcp/vault-index.db"
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
Run as a background daemon or systemd service to keep models pre-warmed and share a single ~2.1 GB RAM footprint across multiple profiles or sessions:
```bash
# Start daemon listening on SSE
vault-mcp --transport sse --host 127.0.0.1 --port 8765 --preload
```

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
ExecStart=/path/to/.venv/bin/python -m src.server --transport sse --host 127.0.0.1 --port 8765 --preload
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

---

## 🛠️ MCP Tools Exposed

### 1. `search_vault`
Executes Two-Stage Hybrid RAG query across your notes.
- **Parameters**:
  - `query` (*string*, required): Natural language question or search phrase.
  - `top_k` (*integer*, default: `5`): Number of highest-ranked passages.
  - `heading_filter` (*string*, optional): Substring filter for specific note headings (e.g. `Architecture`, `Runbook`).
- **Response**: Array of candidate objects with `title`, `rel_path`, `heading`, `lines`, `relevance_score`, and markdown `text`.

### 2. `get_note`
Reads full note content with line numbering and bounded pagination to prevent context window overflow.
- **Parameters**:
  - `rel_path` (*string*, required): Relative note path (e.g., `projects/cuantum.md`).
  - `offset_line` (*integer*, default: `1`): Start line number.
  - `limit_lines` (*integer*, default: `250`): Maximum lines to return.

### 3. `sync_vault`
Triggers incremental background synchronization to detect newly modified or deleted notes.
- **Parameters**: None.

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
