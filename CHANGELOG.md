# Changelog

## Unreleased

### Changed
- Hybrid search reranks only the top `VAULT_RERANK_POOL` (default 8) RRF candidates, each cut to `VAULT_RERANK_CHARS` (default 600, was 1500). On a 3-vCPU host this took hybrid latency from 11.9 s to 2.6 s at MRR 0.96 (was 0.97). Reranker threads follow the process cpuset (`VAULT_RERANK_THREADS`).
- `vault-eval` reports avg / p95 latency (after a warm-up query) and accepts `--rerank-pool` / `--rerank-chars`.

### Fixed
- A missing or broken embedding / reranker model made `semantic` and `hybrid` search silently fall back to keyword hits or RRF order. Results now end with a warning naming the failure, degraded results are not cached, and `vault_status` lists the unavailable model with its error until it loads again.

## 2.0.0

### Fixed
- `#` comments inside fenced code blocks were treated as headings: they split code blocks into separate chunks and could become a note's title. Heading detection is now fence-aware everywhere (chunker, title, `vault_read` sections, `vault_append`).
- `vault_read` returned the whole note when a heading was not found, and had no size limit. It now pages output (`start_line`, `max_chars`) and lists the available headings instead.
- Note paths used the OS separator, so on Windows folder filters, `vault_list` and result paths were wrong. All paths are POSIX now.
- Notes whose sections were all shorter than 30 characters produced no chunks and could never be found.
- Chunk line ranges drifted when a section was split; ranges are now exact.
- A write that arrived while an indexer was already running was skipped until the next cron run. Writers now flag the index dirty and the running indexer makes one more pass.
- `VAULT_INDEXER_COMMAND` unset silently fell back to an indexer without a memory cap. Re-indexing now defaults to an in-process worker; `command` mode refuses to start without a command. The legacy `INDEXER_RUNNER` variable is read too.

### Changed
- **Package renamed** from `src` to `obsidian_hybrid_rag_mcp`. Use the `vault-mcp`, `vault-indexer` and `vault-eval` entry points.
- **Index schema v2** (tags, status, skip reason, `rel_path` metadata on vectors). Older indexes are rebuilt automatically.
- In-process re-indexing reuses the server's loaded embedding model instead of starting a second process with its own copy, and idle unload waits while indexing.
- Writes are atomic (temp file + rename) and keep CRLF line endings.
- `vault_write` refuses to overwrite unless `overwrite=true`, and backs up the old version to `.trash/vault-mcp/`.
- Write tools accept `expected_hash` (from `vault_read`) to refuse edits to a note that changed in the meantime.
- Binding a non-loopback address requires `--allow-remote`.
- BM25 is weighted (heading 3, title 2, body 1, note summary 0.2) so a note-level summary repeated in every chunk no longer drowns real matches.
- Default chunk size is 1500 characters, and the reranker sees the whole chunk (was the first 800 characters).
- Folder filters run inside the vector KNN (sqlite-vec metadata column) instead of oversampling and filtering afterwards.
- Heavy model dependencies moved to the `models` extra; unused `rich`, `pydantic` and `numpy` pins removed.

### Added
- Indexer skip rules: `.vaultignore`, `VAULT_MAX_FILE_BYTES`, frontmatter `index: false`; overlong lines (`VAULT_MAX_LINE_CHARS`) are dropped from indexed text. Skipped notes are listed by `vault_status`.
- `vault_search` filters `tags` and `status`.
- Optional reranker score floor `VAULT_MIN_RERANK_SCORE`.
- Tools: `vault_lint`, `vault_move` (rewrites links), `vault_recent`. Notes linked from the root README count as hubs; `[[links]]` inside inline code are ignored (as in Obsidian); `blocked` is an allowed status.
- `vault-eval` retrieval evaluation (hit@k, recall@k, MRR, rerank score distribution) and an example golden set.
- CI on Ubuntu and Windows, ruff configuration.
