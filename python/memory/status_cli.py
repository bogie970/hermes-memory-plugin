"""CLI for /maintenance-status — prints recent maintenance runs and store stats."""

from __future__ import annotations

import argparse
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Memory maintenance status")
    ap.add_argument("--limit", type=int, default=10, help="Recent runs to show")
    ap.add_argument("--stats", action="store_true",
                     help="Also include LanceDB tier counts")
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
    from memory.maintenance_log import format_status, read_recent

    runs = read_recent(args.limit)
    print(format_status(runs))

    if args.stats:
        from memory.embeddings import EmbeddingService
        from memory.store import MemoryStore

        store = MemoryStore(embedder=EmbeddingService())
        rows = store.scan_v2()
        from collections import Counter
        tiers = Counter(r.get("tier") for r in rows)
        print("\n# Store stats")
        print(f"- total: {len(rows)}")
        for tier, count in sorted(tiers.items()):
            print(f"- {tier}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
