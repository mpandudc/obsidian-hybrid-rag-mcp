"""
Retrieval evaluation: hit@k, recall@k and MRR over a golden query set, per search mode,
plus the cross-encoder score distribution of relevant vs irrelevant results (use it to
choose VAULT_MIN_RERANK_SCORE).

Golden file (JSON): [{"query": "...", "expected": ["server/homeserver-setup.md", "other-note"]}, ...]
`expected` entries may be vault paths (with or without .md) or bare note names.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from obsidian_hybrid_rag_mcp.search import search


def load_golden(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all("query" in d and d.get("expected") for d in data):
        raise ValueError("golden file must be a JSON list of {query, expected: [...]}")
    return data


def is_expected(rel: str, expected: list[str]) -> bool:
    noext = rel.removesuffix(".md").lower()
    stem = noext.rsplit("/", 1)[-1]
    for e in expected:
        e = e.strip().replace("\\", "/").lower().removesuffix(".md")
        if e == noext or ("/" not in e and e == stem):
            return True
    return False


def evaluate(conn, golden: list[dict], mode: str, k: int, embed_fn: Callable | None = None,
             rerank_fn: Callable | None = None, rerank_pool: int = 8, rerank_chars: int = 600) -> dict:
    hits_at_k = 0
    recall_sum = 0.0
    rr_sum = 0.0
    misses = []
    rel_scores: list[float] = []
    irrel_scores: list[float] = []
    latencies: list[float] = []
    if golden:  # warm-up: model loading is not query latency
        search(conn, golden[0]["query"], limit=k, mode=mode, embed_fn=embed_fn, rerank_fn=rerank_fn,
               rerank_pool=rerank_pool, rerank_chars=rerank_chars)
    for item in golden:
        expected = item["expected"]
        started = time.perf_counter()
        outcome = search(conn, item["query"], limit=k, mode=mode, embed_fn=embed_fn, rerank_fn=rerank_fn,
                         per_note=1, rerank_pool=rerank_pool, rerank_chars=rerank_chars)
        latencies.append((time.perf_counter() - started) * 1000)
        paths = [h.rel_path for h in outcome.hits]
        first = next((i for i, p in enumerate(paths, start=1) if is_expected(p, expected)), None)
        found = {p for p in paths if is_expected(p, expected)}
        if first:
            hits_at_k += 1
            rr_sum += 1.0 / first
        else:
            misses.append({"query": item["query"], "expected": expected, "got": paths[:3]})
        recall_sum += min(1.0, len(found) / len(expected))
        if outcome.reranked:
            for h in outcome.hits:
                (rel_scores if is_expected(h.rel_path, expected) else irrel_scores).append(h.score)
    n = len(golden) or 1
    return {
        "mode": mode, "k": k, "queries": len(golden),
        "hit_at_k": hits_at_k / n, "recall_at_k": recall_sum / n, "mrr": rr_sum / n,
        "misses": misses, "relevant_scores": rel_scores, "irrelevant_scores": irrel_scores,
        "avg_ms": statistics.mean(latencies) if latencies else 0.0,
        "p95_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0.0,
    }


def _quantiles(values: list[float]) -> str:
    if len(values) < 2:
        return ", ".join(f"{v:.2f}" for v in values) or "-"
    q = statistics.quantiles(values, n=10)
    return f"min {min(values):.2f} · p10 {q[0]:.2f} · median {statistics.median(values):.2f} · max {max(values):.2f}"


def format_report(results: list[dict]) -> str:
    out = ["| mode | hit@k | recall@k | MRR | avg ms | p95 ms |", "|---|---|---|---|---|---|"]
    out += [f"| {r['mode']} | {r['hit_at_k']:.2f} | {r['recall_at_k']:.2f} | {r['mrr']:.2f} | "
            f"{r['avg_ms']:.0f} | {r['p95_ms']:.0f} |" for r in results]
    for r in results:
        if r["relevant_scores"] or r["irrelevant_scores"]:
            out.append(f"\nRerank scores ({r['mode']}): relevant [{_quantiles(r['relevant_scores'])}], "
                       f"irrelevant [{_quantiles(r['irrelevant_scores'])}]")
            out.append("A floor (VAULT_MIN_RERANK_SCORE) a bit under the relevant p10 drops most noise.")
        for m in r["misses"]:
            out.append(f"- miss [{r['mode']}] {m['query']!r}: expected {m['expected']}, got {m['got']}")
    return "\n".join(out)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Evaluate vault retrieval quality against a golden query set")
    parser.add_argument("--golden", required=True, type=Path, help="JSON list of {query, expected}")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--modes", default="keyword,semantic,hybrid")
    parser.add_argument("--rerank-pool", type=int, default=None, help="Override VAULT_RERANK_POOL")
    parser.add_argument("--rerank-chars", type=int, default=None, help="Override VAULT_RERANK_CHARS")
    args = parser.parse_args(argv)

    from obsidian_hybrid_rag_mcp import server  # loads env config (VAULT_INDEX_DB, models)

    golden = load_golden(args.golden)
    conn = server.get_db_connection()
    try:
        results = [
            evaluate(conn, golden, mode.strip(), args.k, embed_fn=server._embed_query, rerank_fn=server._rerank,
                     rerank_pool=args.rerank_pool or server.RERANK_POOL,
                     rerank_chars=args.rerank_chars or server.RERANK_CHARS)
            for mode in args.modes.split(",") if mode.strip()
        ]
    finally:
        conn.close()
    print(format_report(results))


if __name__ == "__main__":
    main()
