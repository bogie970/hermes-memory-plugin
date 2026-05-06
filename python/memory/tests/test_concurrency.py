"""Phase C — concurrency tests.

Verifies the existing FileLock on MemoryStore prevents lost updates
when multiple threads write simultaneously.

Run: pytest aisys/memory/tests/test_concurrency.py -v
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


@pytest.fixture
def v2_store(lance_store_v1):
    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)
    return lance_store_v1


def test_parallel_writes_no_lost_updates(v2_store):
    """5 threads × 20 writes each = 100 unique records, no lost updates."""
    from memory.write_gate import write_memory

    errors = []

    def worker(idx):
        try:
            for i in range(20):
                write_memory(
                    store=v2_store,
                    content=f"thread_{idx}_record_{i}",
                    writer="user",
                    provenance="user_stated",
                    source_ref=f"t:{idx}:{i}",
                    confidence=1.0,
                )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # Each thread writes 20 distinct records => 100 total
    assert v2_store.count() == 100


def test_parallel_writes_no_duplicate_ids(v2_store):
    """No two records share the same id under concurrent writes."""
    from memory.write_gate import write_memory

    def worker(idx):
        for i in range(15):
            write_memory(
                store=v2_store,
                content=f"unique_t{idx}_r{i}",
                writer="user",
                provenance="user_stated",
                source_ref=f"t:{idx}:{i}",
                confidence=1.0,
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = v2_store.scan_v2()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids detected"


def test_concurrent_dedup_bumps_seen_count_correctly(v2_store):
    """When 10 threads all write the same content, seen_count == 10."""
    from memory.write_gate import write_memory

    rec_id_holder = []

    def worker():
        rec_id = write_memory(
            store=v2_store,
            content="shared content for dedup test",
            writer="user",
            provenance="user_stated",
            source_ref="shared_t",
            confidence=1.0,
        )
        rec_id_holder.append(rec_id)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads got the same record id (dedup worked)
    assert len(set(rec_id_holder)) == 1
    # And seen_count reflects all 10 attempts
    rec = next(r for r in v2_store.scan_v2() if r["id"] == rec_id_holder[0])
    assert rec["seen_count"] == 10
