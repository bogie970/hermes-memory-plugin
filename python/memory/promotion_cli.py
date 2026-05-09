"""CLI wrapper for memory.promotion.run_daily — invoked by /consolidate-memory.

Logs the run via maintenance_log so /maintenance-status can show history.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description="Sonnet daily promotion pass")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--dry-run", action="store_true",
                     help="Don't actually call the LLM; just count eligible candidates")
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
    from aisys.memory import promotion
    from aisys.memory.embeddings import EmbeddingService
    from aisys.memory.maintenance_log import log_run
    from aisys.memory.store import MemoryStore

    store = MemoryStore(embedder=EmbeddingService())

    if args.dry_run:
        rows = store.scan_v2()
        eligible = sum(1 for r in rows if promotion._eligible_for_review(r))
        print(json.dumps({"dry_run": True, "eligible_candidates": eligible,
                          "total_candidates": sum(1 for r in rows if r.get("tier") == "candidate")}))
        return 0

    t0 = time.monotonic()
    error: str | None = None
    try:
        result = promotion.run_daily(store, model=args.model)
        summary = {
            "processed": result.processed,
            "promoted": result.promoted,
            "rejected": result.rejected,
            "held": result.held,
            "errors": result.errors,
        }
    except Exception as e:
        error = str(e)[:300]
        summary = {}
        # Also record to error sentinel for /memory-stats surfacing
        try:
            from aisys.memory.error_sentinel import record_error
            record_error(source="promotion_cli", error=error, context=f"model={args.model}")
        except Exception:
            pass
    duration = time.monotonic() - t0

    log_run(job="sonnet_daily", duration_s=duration, summary=summary, error=error)

    print(json.dumps({
        "duration_s": round(duration, 2),
        "summary": summary,
        "error": error,
    }))
    return 0 if error is None else 1


if __name__ == "__main__":
    sys.exit(main())
