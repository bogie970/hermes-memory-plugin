"""Memory consolidation: importance decay and maintenance.

Handles time-based decay of memory importance scores. Memories that
haven't been accessed recently lose importance, making them candidates
for archival. Protected categories (user feedback) are exempt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from memory.config import IMPORTANCE_DECAY_RATE
from memory.store import MemoryStore

log = logging.getLogger("memory.consolidation")

# Floor: never decay below this value
IMPORTANCE_FLOOR = 0.1

# Categories exempt from decay
DECAY_EXEMPT_CATEGORIES = {"feedback"}
DECAY_EXEMPT_MIN_IMPORTANCE = 0.8


@dataclass
class DecayResult:
    """Summary of a decay pass."""
    total_checked: int = 0
    decayed: int = 0
    skipped_exempt: int = 0
    skipped_recent: int = 0
    errors: list[str] = field(default_factory=list)


def decay_importance(store: MemoryStore, dry_run: bool = False) -> DecayResult:
    """Apply time-based importance decay to all active memories.

    Formula: importance *= IMPORTANCE_DECAY_RATE ^ weeks_since_last_access

    Rules:
    - Only decay memories not accessed in the past week
    - Never decay below IMPORTANCE_FLOOR (0.1)
    - Never decay user corrections (category="feedback" with importance >= 0.8)

    Args:
        store: MemoryStore instance.
        dry_run: If True, calculate but don't write changes.

    Returns:
        DecayResult with counts.
    """
    result = DecayResult()
    now = datetime.now(timezone.utc)

    try:
        all_memories = store.get_all(include_archived=False)
    except Exception as e:
        log.error("Failed to load memories for decay: %s", e)
        result.errors.append(f"Load failed: {e}")
        return result

    result.total_checked = len(all_memories)

    for mem in all_memories:
        try:
            # Check exemption: feedback with high importance
            if (
                mem.category in DECAY_EXEMPT_CATEGORIES
                and mem.importance >= DECAY_EXEMPT_MIN_IMPORTANCE
            ):
                result.skipped_exempt += 1
                continue

            # Calculate weeks since last access
            delta = now - mem.last_accessed
            weeks = delta.total_seconds() / (7 * 24 * 3600)

            # Skip if accessed within the past week
            if weeks < 1.0:
                result.skipped_recent += 1
                continue

            # Apply decay: importance *= rate ^ weeks
            new_importance = mem.importance * (IMPORTANCE_DECAY_RATE ** weeks)
            new_importance = max(IMPORTANCE_FLOOR, new_importance)

            # Skip if effectively unchanged (< 0.01 difference)
            if abs(new_importance - mem.importance) < 0.01:
                result.skipped_recent += 1
                continue

            if not dry_run:
                store.update_importance(mem.id, round(new_importance, 4))

            result.decayed += 1
            log.debug(
                "Decayed %s: %.3f -> %.3f (%.1f weeks idle)",
                mem.id[:8], mem.importance, new_importance, weeks,
            )

        except Exception as e:
            log.warning("Decay failed for %s: %s", mem.id[:8], e)
            result.errors.append(f"Decay {mem.id[:8]}: {e}")

    log.info(
        "Decay pass: %d checked, %d decayed, %d exempt, %d recent/unchanged",
        result.total_checked, result.decayed,
        result.skipped_exempt, result.skipped_recent,
    )
    return result


def run_consolidation(store: MemoryStore, dry_run: bool = False) -> DecayResult:
    """Run a full consolidation pass (currently just decay).

    Entry point for the periodic consolidation job. Wraps decay_importance
    and can be extended with additional consolidation steps.
    """
    return decay_importance(store, dry_run=dry_run)
