"""CLI for /maintenance-status — recent runs, store stats, quota burn, errors."""

from __future__ import annotations

import argparse
import pathlib
import sys


# Rough Sonnet/Opus per-call cost estimate for quota awareness.
# Very approximate: each adjudication ~5k input + ~1k output.
APPROX_TOKENS_PER_SONNET_CALL = 6000
APPROX_TOKENS_PER_OPUS_CALL = 6000


def _summarize_quota(runs: list[dict]) -> dict:
    """Aggregate per-job call counts and rough token totals."""
    by_job: dict[str, dict] = {}
    for r in runs:
        job = r.get("job", "?")
        summary = r.get("summary") or {}
        # Each run might process N candidates -> N model calls
        calls = (
            summary.get("processed", 0)
            + summary.get("promoted", 0)  # already counted in processed but defensive
        )
        # Don't double-count; use processed as proxy
        calls = summary.get("processed", 0)
        slot = by_job.setdefault(job, {"runs": 0, "calls": 0, "errors": 0})
        slot["runs"] += 1
        slot["calls"] += int(calls)
        if r.get("error"):
            slot["errors"] += 1
    # Estimate tokens
    for job, slot in by_job.items():
        if "sonnet" in job:
            slot["est_tokens"] = slot["calls"] * APPROX_TOKENS_PER_SONNET_CALL
        elif "opus" in job:
            slot["est_tokens"] = slot["calls"] * APPROX_TOKENS_PER_OPUS_CALL
        else:
            slot["est_tokens"] = 0
    return by_job


def main() -> int:
    ap = argparse.ArgumentParser(description="Memory maintenance status")
    ap.add_argument("--limit", type=int, default=10, help="Recent runs to show")
    ap.add_argument("--stats", action="store_true",
                     help="Also include LanceDB tier counts + contradictions")
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
    from memory.error_sentinel import read_recent as read_errors
    from memory.maintenance_log import format_status, read_recent

    runs = read_recent(args.limit)
    print(format_status(runs))

    # Quota burn summary
    if runs:
        print("\n# Quota burn (estimated)")
        for job, slot in _summarize_quota(runs).items():
            tokens = slot.get("est_tokens", 0)
            print(f"- {job}: {slot['runs']} runs, ~{slot['calls']} calls, ~{tokens:,} tokens, {slot['errors']} errors")

    # Recent silent errors from sentinel
    errors = read_errors(limit=5)
    if errors:
        print("\n# Recent hook/CLI errors (silent failures)")
        for e in errors:
            print(f"- {e['ts']} {e['source']}: {e['error'][:100]}")
            if e.get("context"):
                print(f"    context: {e['context']}")

    if args.stats:
        from memory.embeddings import EmbeddingService
        from memory.store import MemoryStore

        store = MemoryStore(embedder=EmbeddingService())
        rows = store.scan_v2_lean()
        from collections import Counter
        tiers = Counter(r.get("tier") for r in rows)
        print("\n# Store stats")
        print(f"- total: {len(rows)}")
        for tier, count in sorted(tiers.items()):
            print(f"- {tier}: {count}")

        # Pending contradictions awaiting user review
        pending_review = [r for r in rows
                          if r.get("contradiction_state") == "user_review"]
        if pending_review:
            print(f"\n# Pending contradictions ({len(pending_review)})")
            for r in pending_review[:5]:
                print(f"- [{r['id'][:8]}] tier={r['tier']}: {r.get('content', '')[:100]}")
        else:
            print("\n# Pending contradictions: 0")

    return 0


if __name__ == "__main__":
    sys.exit(main())
