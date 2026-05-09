"""User-facing search and forget CLIs.

`/memory-search "query"` — surface the L2 vector store as a slash command
so the user can introspect what Hermes remembers. Returns the top-k
matching memories with content, tier, source_ref, and distance.

`/memory-forget <id>` — surgical removal. Tombstones a specific memory
(audit log retains the receipt). Re-forgetting is a no-op.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))


# Allow tier filter list reused by search/forget
DEFAULT_SEARCH_TIERS = ("verified", "probationary", "candidate")


def _sanitize_id(value: str) -> str:
    return str(value).replace("'", "''")


def search(
    store,
    query: str,
    k: int = 5,
    min_confidence: float = 0.0,
    include_tiers: tuple[str, ...] = DEFAULT_SEARCH_TIERS,
) -> list[dict[str, Any]]:
    """Run vector search against L2; return user-facing record dicts.

    Excludes tombstoned by default. min_confidence drops results whose
    cosine-similarity is below threshold (0 = return everything LanceDB
    returns). Each result has: id, content, tier, source_ref, distance,
    confidence, writer.
    """
    if not query or not query.strip():
        return []

    vector = store._embedder.embed_one(query)
    tier_filter = ", ".join(f"'{t}'" for t in include_tiers)

    try:
        rows = (
            store.table.search(vector)
            .where(f"tier IN ({tier_filter})")
            .limit(k)
            .to_list()
        )
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    for r in rows:
        distance = float(r.get("_distance", 1.0))
        # cosine similarity = 1 - distance for normalized embeddings
        sim = 1.0 - distance
        if sim < min_confidence:
            continue
        results.append({
            "id": r.get("id", ""),
            "content": (r.get("content") or "")[:500],  # truncate for readability
            "tier": r.get("tier", "?"),
            "source_ref": r.get("source_ref", "?"),
            "writer": r.get("writer", "?"),
            "confidence": float(r.get("confidence") or 0.0),
            "distance": round(distance, 4),
            "similarity": round(sim, 4),
        })
    return results


def format_search_output(results: list[dict], query: str) -> str:
    """Human-readable formatting for /memory-search output."""
    if not results:
        return f"No results for: {query!r}"

    lines = [f"# Memory search results for: {query!r}\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. [{r['id'][:8]}] tier={r['tier']} similarity={r['similarity']}")
        lines.append(f"   writer={r['writer']} source={r['source_ref']}")
        lines.append(f"   {r['content']}")
        lines.append("")
    return "\n".join(lines)


def forget(store, memory_id: str, reason: str = "user requested") -> dict[str, Any]:
    """Tombstone a memory by id. Returns dict with tombstoned count.

    Idempotent: forgetting an already-tombstoned or missing id is a no-op.
    Audit log records the operation so the receipt survives.
    """
    if not memory_id:
        return {"tombstoned": 0, "reason": "empty id"}

    rows = store.scan_v2_lean()
    target = next((r for r in rows if r["id"] == memory_id), None)

    if target is None:
        return {"tombstoned": 0, "reason": "id not found"}
    if target.get("tier") == "tombstoned":
        return {"tombstoned": 0, "reason": "already tombstoned"}

    safe_id = _sanitize_id(memory_id)
    with store.lock:
        store.table.update(
            where=f"id = '{safe_id}'",
            values={"tier": "tombstoned"},
        )

    from memory.write_gate import _audit
    _audit(
        store, memory_id=memory_id, op="user_forget",
        who="user",
        why=f"tombstoned by user: {reason[:200]}",
        before=target.get("tier", "unknown"),
        after="tombstoned",
    )

    return {"tombstoned": 1, "id": memory_id}


# ---- CLI entrypoints ----

def main_search() -> int:
    """python -m memory.search_cli search 'query'"""
    ap = argparse.ArgumentParser(description="Hermes memory search")
    ap.add_argument("query", help="Natural-language query")
    ap.add_argument("-k", "--top-k", type=int, default=5)
    ap.add_argument("--min-confidence", type=float, default=0.0,
                     help="Drop results below this cosine similarity")
    ap.add_argument("--include-tombstoned", action="store_true")
    ap.add_argument("--json", action="store_true",
                     help="Output JSON instead of formatted text")
    args = ap.parse_args()

    from memory.embeddings import EmbeddingService
    from memory.store import MemoryStore

    store = MemoryStore(embedder=EmbeddingService())
    tiers = (
        DEFAULT_SEARCH_TIERS + ("tombstoned",)
        if args.include_tombstoned else DEFAULT_SEARCH_TIERS
    )
    results = search(
        store, query=args.query, k=args.top_k,
        min_confidence=args.min_confidence,
        include_tiers=tiers,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_search_output(results, args.query))
    return 0


def main_forget() -> int:
    """python -m memory.search_cli forget <id> --reason '...'"""
    ap = argparse.ArgumentParser(description="Hermes memory forget (tombstone)")
    ap.add_argument("memory_id", help="Memory id to tombstone")
    ap.add_argument("--reason", default="user requested",
                     help="Reason recorded in audit log")
    args = ap.parse_args()

    from memory.embeddings import EmbeddingService
    from memory.store import MemoryStore

    store = MemoryStore(embedder=EmbeddingService())
    result = forget(store, memory_id=args.memory_id, reason=args.reason)
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    """Dispatcher: python -m memory.search_cli {search,forget} ..."""
    if len(sys.argv) < 2:
        print("Usage: python -m memory.search_cli {search,forget} ...", file=sys.stderr)
        return 1
    op = sys.argv.pop(1)
    if op == "search":
        return main_search()
    elif op == "forget":
        return main_forget()
    else:
        print(f"Unknown op: {op}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
