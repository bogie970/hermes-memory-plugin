"""Memory tiering: archive cold memories, promote hot ones, merge duplicates.

Tiering runs periodically (every 1-2 hours) to keep the active memory
set focused and deduplicated. Three operations:

1. Archive: move cold, low-importance memories to L3
2. Promote: bring back archived memories that are being actively accessed
3. Merge: find near-duplicate memories and consolidate them
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from memory.config import (
    ARCHIVE_MIN_AGE_DAYS,
    ARCHIVE_MIN_ACCESS_COUNT,
    L3_IMPORTANCE_THRESHOLD,
    ACCESS_PROMOTE_THRESHOLD,
    ACCESS_PROMOTE_IMPORTANCE,
    MERGE_SIMILARITY_THRESHOLD,
)
from memory.store import MemoryStore

log = logging.getLogger("memory.tiering")


@dataclass
class TieringResult:
    """Summary of a tiering pass."""
    archived: int = 0
    promoted: int = 0
    archive_checked: int = 0
    promote_checked: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class MergeResult:
    """Summary of a merge pass."""
    total_checked: int = 0
    merged: int = 0
    linked: int = 0
    errors: list[str] = field(default_factory=list)


def run_tiering(store: MemoryStore) -> TieringResult:
    """Run a single tiering pass: archive cold, promote hot.

    Archive criteria (ALL must be true):
    - importance < L3_IMPORTANCE_THRESHOLD (0.15)
    - age > ARCHIVE_MIN_AGE_DAYS (60 days)
    - access_count < ARCHIVE_MIN_ACCESS_COUNT (3)

    Promote criteria (ALL must be true):
    - currently archived
    - access_count >= ACCESS_PROMOTE_THRESHOLD (5)
    - importance >= ACCESS_PROMOTE_IMPORTANCE (0.75)
    """
    result = TieringResult()
    now = datetime.now(timezone.utc)

    # --- Archive pass ---
    try:
        active_memories = store.get_all(include_archived=False)
    except Exception as e:
        log.error("Failed to load active memories: %s", e)
        result.errors.append(f"Archive load: {e}")
        return result

    result.archive_checked = len(active_memories)

    for mem in active_memories:
        try:
            age_days = (now - mem.created_at).total_seconds() / 86400

            if (
                mem.importance < L3_IMPORTANCE_THRESHOLD
                and age_days > ARCHIVE_MIN_AGE_DAYS
                and mem.access_count < ARCHIVE_MIN_ACCESS_COUNT
            ):
                store.archive(mem.id)
                result.archived += 1
                log.debug(
                    "Archived %s: importance=%.2f, age=%dd, accesses=%d",
                    mem.id[:8], mem.importance, int(age_days), mem.access_count,
                )
        except Exception as e:
            log.warning("Archive check failed for %s: %s", mem.id[:8], e)
            result.errors.append(f"Archive {mem.id[:8]}: {e}")

    # --- Promote pass ---
    try:
        archived_memories = store.get_all(include_archived=True)
        archived_memories = [m for m in archived_memories if m.archived]
    except Exception as e:
        log.error("Failed to load archived memories: %s", e)
        result.errors.append(f"Promote load: {e}")
        return result

    result.promote_checked = len(archived_memories)

    for mem in archived_memories:
        try:
            if (
                mem.access_count >= ACCESS_PROMOTE_THRESHOLD
                and mem.importance >= ACCESS_PROMOTE_IMPORTANCE
            ):
                store.unarchive(mem.id)
                result.promoted += 1
                log.debug(
                    "Promoted %s: importance=%.2f, accesses=%d",
                    mem.id[:8], mem.importance, mem.access_count,
                )
        except Exception as e:
            log.warning("Promote check failed for %s: %s", mem.id[:8], e)
            result.errors.append(f"Promote {mem.id[:8]}: {e}")

    log.info(
        "Tiering: %d archived (of %d active), %d promoted (of %d archived)",
        result.archived, result.archive_checked,
        result.promoted, result.promote_checked,
    )
    return result


def run_merge_pass(store: MemoryStore) -> MergeResult:
    """Find and merge near-duplicate memories.

    Scans active memories pairwise via vector similarity. For each pair
    with cosine similarity > MERGE_SIMILARITY_THRESHOLD:
    - If one is clearly a superset (>= 20% longer + same core content):
      keep the longer one, archive the shorter
    - If both have unique info: link them (don't merge)
    - If nearly identical: keep higher importance, archive the other

    Uses the store's search() to find nearest neighbors per memory,
    rather than O(n^2) pairwise comparison.
    """
    result = MergeResult()

    try:
        active_memories = store.get_all(include_archived=False)
    except Exception as e:
        log.error("Failed to load memories for merge: %s", e)
        result.errors.append(f"Load: {e}")
        return result

    result.total_checked = len(active_memories)

    if len(active_memories) < 2:
        return result

    # Track IDs already processed (archived or linked) this pass
    processed: set[str] = set()

    for mem in active_memories:
        if mem.id in processed:
            continue

        try:
            # Search for similar memories (top 3 to find duplicates)
            results = store.search(
                query=mem.content,
                k=3,
                namespaces=[mem.namespace] if mem.namespace else None,
            )

            for scored in results:
                other = scored.record
                if other.id == mem.id or other.id in processed:
                    continue

                similarity = scored.relevance
                if similarity < MERGE_SIMILARITY_THRESHOLD:
                    continue

                # Found a near-duplicate pair
                len_ratio = len(mem.content) / max(len(other.content), 1)

                if len_ratio >= 1.2:
                    # mem is the superset — archive other
                    _merge_into(store, keeper=mem, loser=other)
                    processed.add(other.id)
                    result.merged += 1
                elif len_ratio <= 0.8:
                    # other is the superset — archive mem
                    _merge_into(store, keeper=other, loser=mem)
                    processed.add(mem.id)
                    result.merged += 1
                    break  # mem is gone, stop checking its neighbors
                elif similarity >= 0.95:
                    # Nearly identical — keep higher importance
                    if mem.importance >= other.importance:
                        _merge_into(store, keeper=mem, loser=other)
                        processed.add(other.id)
                    else:
                        _merge_into(store, keeper=other, loser=mem)
                        processed.add(mem.id)
                        break
                    result.merged += 1
                else:
                    # Both have unique info — link them
                    _link_pair(store, mem, other)
                    result.linked += 1

        except Exception as e:
            log.warning("Merge check failed for %s: %s", mem.id[:8], e)
            result.errors.append(f"Merge {mem.id[:8]}: {e}")

    log.info(
        "Merge pass: %d checked, %d merged, %d linked",
        result.total_checked, result.merged, result.linked,
    )
    return result


def _merge_into(store: MemoryStore, keeper: "MemoryRecord", loser: "MemoryRecord") -> None:
    """Merge loser into keeper: combine tags, bump importance, archive loser."""
    # Combine tags
    new_tags = list(set(keeper.tags + loser.tags))
    if new_tags != keeper.tags:
        store.update_tags(keeper.id, new_tags)

    # Take the higher importance
    if loser.importance > keeper.importance:
        store.update_importance(keeper.id, loser.importance)

    # Archive the loser
    store.archive(loser.id)

    log.debug(
        "Merged %s into %s (archived loser)",
        loser.id[:8], keeper.id[:8],
    )


def _link_pair(store: MemoryStore, a: "MemoryRecord", b: "MemoryRecord") -> None:
    """Link two memories bidirectionally without merging."""
    a_links = list(set(a.links + [b.id]))
    b_links = list(set(b.links + [a.id]))

    if a_links != a.links:
        store.update_links(a.id, a_links)
    if b_links != b.links:
        store.update_links(b.id, b_links)

    log.debug("Linked %s <-> %s", a.id[:8], b.id[:8])
