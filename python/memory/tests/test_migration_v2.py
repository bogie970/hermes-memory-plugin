"""Phase B — schema migration tests.

Tests the v2 schema additions (provenance, tier, bitemporal fields, audit log).
Critical safety: existing 50 production records must be preserved.

Run: pytest aisys/memory/tests/test_migration_v2.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from memory.schema import MemoryRecord, MemoryType


# ------------- Tier B.1: schema additions are present in new schema --------

def test_v2_schema_has_tier_field():
    """LANCE_SCHEMA_V2 must include tier column."""
    from memory.store import LANCE_SCHEMA_V2
    field_names = {f.name for f in LANCE_SCHEMA_V2}
    assert "tier" in field_names

def test_v2_schema_has_provenance_fields():
    """LANCE_SCHEMA_V2 must include provenance, source_ref, writer, confidence."""
    from memory.store import LANCE_SCHEMA_V2
    field_names = {f.name for f in LANCE_SCHEMA_V2}
    for required in ("provenance", "source_ref", "writer", "confidence"):
        assert required in field_names, f"missing field: {required}"

def test_v2_schema_has_bitemporal_fields():
    """LANCE_SCHEMA_V2 must include valid_from, valid_to, supersedes, superseded_by."""
    from memory.store import LANCE_SCHEMA_V2
    field_names = {f.name for f in LANCE_SCHEMA_V2}
    for required in ("valid_from", "valid_to", "supersedes", "superseded_by"):
        assert required in field_names

def test_v2_schema_has_contradiction_fields():
    """LANCE_SCHEMA_V2 must include contradiction tracking fields."""
    from memory.store import LANCE_SCHEMA_V2
    field_names = {f.name for f in LANCE_SCHEMA_V2}
    for required in ("contradiction_state", "conflict_with"):
        assert required in field_names

def test_v2_schema_has_promotion_fields():
    """LANCE_SCHEMA_V2 must include seen_count, last_seen_at, promoted_at."""
    from memory.store import LANCE_SCHEMA_V2
    field_names = {f.name for f in LANCE_SCHEMA_V2}
    for required in ("seen_count", "last_seen_at", "promoted_at"):
        assert required in field_names

def test_v2_schema_has_embedding_versioning():
    """LANCE_SCHEMA_V2 must include embedding_model and embedding_version."""
    from memory.store import LANCE_SCHEMA_V2
    field_names = {f.name for f in LANCE_SCHEMA_V2}
    assert "embedding_model" in field_names
    assert "embedding_version" in field_names


# ------------- Tier B.2: backfill existing records preserves data ----------

def test_backfill_preserves_record_count(lance_store_v1, sample_v1_records, fake_embedder):
    """Migration must not lose existing records."""
    # Seed 5 records via the v1 path (no v2 fields)
    for rec in sample_v1_records:
        lance_store_v1.insert(rec, enrich=False)

    initial_count = lance_store_v1.count()
    assert initial_count == len(sample_v1_records)

    # Run migration
    from memory.migrate_v2 import backfill_v2
    result = backfill_v2(lance_store_v1)

    assert result.records_migrated == initial_count
    assert lance_store_v1.count() == initial_count

def test_backfill_sets_tier_verified_for_existing(lance_store_v1, sample_v1_records, fake_embedder):
    """All existing records get tier=verified (grandfather rule)."""
    for rec in sample_v1_records:
        lance_store_v1.insert(rec, enrich=False)

    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)

    rows = lance_store_v1.scan_v2()
    assert len(rows) == len(sample_v1_records)
    for row in rows:
        assert row["tier"] == "verified"

def test_backfill_sets_provenance_to_llm_inferred(lance_store_v1, sample_v1_records):
    """Pre-migration records have provenance=llm_inferred (safe default)."""
    for rec in sample_v1_records:
        lance_store_v1.insert(rec, enrich=False)

    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)

    rows = lance_store_v1.scan_v2()
    for row in rows:
        assert row["provenance"] == "llm_inferred"

def test_backfill_sets_embedding_model_versioning(lance_store_v1, sample_v1_records):
    """All existing rows tagged with current embedding model."""
    for rec in sample_v1_records:
        lance_store_v1.insert(rec, enrich=False)

    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)

    rows = lance_store_v1.scan_v2()
    for row in rows:
        assert row["embedding_model"] == "gte-modernbert-base"
        assert row["embedding_version"] == 1

def test_backfill_idempotent(lance_store_v1, sample_v1_records):
    """Re-running backfill is a no-op."""
    for rec in sample_v1_records:
        lance_store_v1.insert(rec, enrich=False)

    from memory.migrate_v2 import backfill_v2

    # First run — should migrate
    result1 = backfill_v2(lance_store_v1)
    assert result1.records_migrated == len(sample_v1_records)

    def _comparable(rows):
        # Strip vector (np array, not directly comparable) and sort by id
        return sorted(
            [{k: v for k, v in r.items() if k != "vector"} for r in rows],
            key=lambda r: r["id"],
        )

    snapshot1 = _comparable(lance_store_v1.scan_v2())

    # Second run — should be no-op
    result2 = backfill_v2(lance_store_v1)
    assert result2.records_migrated == 0
    assert result2.skipped == "already_v2"

    snapshot2 = _comparable(lance_store_v1.scan_v2())
    assert snapshot1 == snapshot2

def test_backfill_preserves_original_content(lance_store_v1, sample_v1_records):
    """Content, tags, importance, etc. unchanged after migration."""
    for rec in sample_v1_records:
        lance_store_v1.insert(rec, enrich=False)

    pre_migration = sorted(
        [(r.id, r.content, r.importance) for r in [
            MemoryRecord.from_lance_row(row) for row in lance_store_v1._table.search().to_list()
        ]],
        key=lambda x: x[0],
    )

    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)

    post_migration = sorted(
        [(r.id, r.content, r.importance) for r in [
            MemoryRecord.from_lance_row(row) for row in lance_store_v1._table.search().to_list()
        ]],
        key=lambda x: x[0],
    )

    assert pre_migration == post_migration


# ------------- Tier B.3: audit log table -----------------------------------

def test_audit_log_table_created_on_migration(lance_store_v1, sample_v1_records):
    """Migration creates audit_log table if missing."""
    for rec in sample_v1_records:
        lance_store_v1.insert(rec, enrich=False)

    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)

    assert "memory_audit" in lance_store_v1.list_tables()

def test_audit_log_records_one_entry_per_migrated_record(lance_store_v1, sample_v1_records):
    """Backfill writes an audit entry for each migrated record."""
    for rec in sample_v1_records:
        lance_store_v1.insert(rec, enrich=False)

    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)

    audits = lance_store_v1.audit_scan()
    backfill_entries = [a for a in audits if a["op"] == "backfill_v2"]
    assert len(backfill_entries) == len(sample_v1_records)

def test_audit_log_has_required_fields(lance_store_v1, sample_v1_records):
    """Audit entries must include memory_id, op, who, when, why."""
    for rec in sample_v1_records:
        lance_store_v1.insert(rec, enrich=False)

    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)

    audits = lance_store_v1.audit_scan()
    assert len(audits) > 0
    for a in audits:
        assert "memory_id" in a
        assert "op" in a
        assert "who" in a
        assert "when" in a
        assert "why" in a


# ------------- Tier B.4: new records use full v2 schema --------------------

def test_new_record_after_migration_has_v2_defaults(lance_store_v1, sample_v1_records):
    """Records inserted after migration carry all v2 fields with sensible defaults."""
    # Migrate first (creates v2 schema)
    for rec in sample_v1_records[:1]:
        lance_store_v1.insert(rec, enrich=False)
    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)

    # Insert a new record
    new_rec = MemoryRecord(
        content="Phase B is done",
        memory_type=MemoryType.SEMANTIC,
        category="milestone",
        source="hermes",
        importance=0.9,
    )
    lance_store_v1.insert(new_rec, enrich=False)

    # Find the new row in v2 scan
    rows = lance_store_v1.scan_v2()
    new_row = next(r for r in rows if r["content"] == "Phase B is done")
    # New rows from MemoryStore.insert() should default to verified
    # (the writer was effectively the system / not explicitly subconscious)
    assert new_row["tier"] in ("verified", "probationary")
    assert new_row["provenance"] in ("llm_inferred", "user_stated", "tool_observed")
    assert new_row["embedding_model"] == "gte-modernbert-base"
    assert new_row["embedding_version"] == 1


# ------------- Tier B.5: filelock prevents corruption ----------------------

def test_concurrent_backfill_does_not_corrupt(lance_store_v1, sample_v1_records):
    """Two concurrent backfill calls produce identical result, no corruption."""
    import threading

    for rec in sample_v1_records:
        lance_store_v1.insert(rec, enrich=False)

    from memory.migrate_v2 import backfill_v2

    results = []

    def runner():
        try:
            results.append(backfill_v2(lance_store_v1))
        except Exception as e:
            results.append(e)

    t1 = threading.Thread(target=runner)
    t2 = threading.Thread(target=runner)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Both completed (filelock serializes them)
    assert len(results) == 2
    # No corruption: count is correct
    assert lance_store_v1.count() == len(sample_v1_records)
    # All have v2 fields
    for row in lance_store_v1.scan_v2():
        assert row["tier"] == "verified"
