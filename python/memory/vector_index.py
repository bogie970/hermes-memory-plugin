"""LanceDB vector index management.

Without a vector index, LanceDB does linear scan on every search. That's
fine under ~5k records (sub-100ms), painful at 10k, broken at 100k.

This module builds an IVF_PQ index on the `vector` column when the row
count crosses INDEX_THRESHOLD. Called from /cleanup and exposed as a
direct command for manual rebuilds.
"""

from __future__ import annotations

import os
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from memory.store import MemoryStore


# Build the index once the corpus crosses this. Default chosen so 51 records
# (today's production state) doesn't trigger; 10k records does.
INDEX_THRESHOLD = int(os.environ.get("HERMES_INDEX_THRESHOLD", "5000"))


def _row_count(store: MemoryStore) -> int:
    """Cheap count via LanceDB pushdown — doesn't materialize rows."""
    try:
        return store.table.count_rows()
    except Exception:
        # Fallback: scan
        return len(store.scan_v2_lean())


def has_vector_index(store: MemoryStore) -> bool:
    """Return True if a vector index exists on the table.

    LanceDB's API for this is `list_indices()` which returns a list of
    index descriptors. We just check if any covers the vector column.
    """
    try:
        if hasattr(store.table, "list_indices"):
            indices = store.table.list_indices()
            for idx in indices:
                cols = getattr(idx, "columns", None) or idx.get("columns", [])
                if "vector" in cols:
                    return True
        return False
    except Exception:
        return False


def should_build_index(store: MemoryStore, threshold: int = INDEX_THRESHOLD) -> bool:
    """Decide whether to build/rebuild the vector index."""
    n = _row_count(store)
    return n > threshold and not has_vector_index(store)


def ensure_vector_index(
    store: MemoryStore,
    threshold: int = INDEX_THRESHOLD,
    force: bool = False,
) -> dict[str, Any]:
    """Build the vector index if needed.

    Returns a status dict: {"action": "built"|"skipped"|"error",
                            "row_count": N, "reason": "...", ...}.
    """
    n = _row_count(store)
    if not force and n <= threshold:
        return {
            "action": "skipped",
            "row_count": n,
            "threshold": threshold,
            "reason": f"row count {n} <= threshold {threshold}",
        }
    if not force and has_vector_index(store):
        return {
            "action": "skipped",
            "row_count": n,
            "reason": "index already exists",
        }

    # Build IVF_PQ index on the vector column.
    # num_partitions heuristic: sqrt(N) clamped to [16, 256]
    num_partitions = max(16, min(256, int(n ** 0.5)))
    started = datetime.now(timezone.utc).isoformat()
    try:
        with store.lock:
            store.table.create_index(
                vector_column_name="vector",
                num_partitions=num_partitions,
                # PQ subvector count — must divide embedding dim cleanly
                # 768 / 16 = 48 -> 16 subvectors of 48 dims each
                num_sub_vectors=16,
                replace=True,
            )
        return {
            "action": "built",
            "row_count": n,
            "num_partitions": num_partitions,
            "started": started,
            "completed": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {
            "action": "error",
            "row_count": n,
            "reason": str(e)[:300],
        }


def vector_index_status(store: MemoryStore) -> dict[str, Any]:
    """One-glance status for /memory-stats integration."""
    return {
        "row_count": _row_count(store),
        "has_index": has_vector_index(store),
        "threshold": INDEX_THRESHOLD,
        "should_build": should_build_index(store),
    }


# ---- CLI entrypoint ----

def main() -> int:
    """python -m memory.vector_index [--force]"""
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Hermes vector index management")
    ap.add_argument("--force", action="store_true",
                     help="Build index regardless of threshold")
    ap.add_argument("--threshold", type=int, default=INDEX_THRESHOLD)
    ap.add_argument("--status", action="store_true",
                     help="Just print status, don't build")
    args = ap.parse_args()

    from memory.embeddings import EmbeddingService

    store = MemoryStore(embedder=EmbeddingService())
    if args.status:
        print(json.dumps(vector_index_status(store), indent=2))
    else:
        result = ensure_vector_index(store, threshold=args.threshold,
                                       force=args.force)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
