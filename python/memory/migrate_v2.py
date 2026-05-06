"""Schema migration v1 -> v2: add provenance, tier, bitemporal, and audit log fields.

Idempotent: safe to run multiple times. Detects already-migrated tables.
Preserves all existing data — backfills new columns with sensible defaults.

Usage:
    cd hermes
    python -m memory.migrate_v2

Or programmatically:
    from memory.migrate_v2 import backfill_v2
    result = backfill_v2(memory_store)
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from memory.store import (
    AUDIT_TABLE_NAME,
    LANCE_AUDIT_SCHEMA,
    V2_FIELDS,
    MemoryStore,
)


# ---- Defaults for backfilled rows ----

V2_DEFAULTS = {
    "tier": "verified",                 # grandfather existing data as verified
    "provenance": "llm_inferred",       # safe default; we don't know origin
    "source_ref": "pre-migration",
    "writer": "manual",
    "confidence": 0.5,                  # neutral
    "valid_to": "",                     # currently valid
    "supersedes": "",
    "superseded_by": "",
    "contradiction_state": "clean",
    "conflict_with": "[]",
    "seen_count": 1,
    "promoted_at": "",                  # not promoted via Sonnet job
    "embedding_model": "gte-modernbert-base",
    "embedding_version": 1,
}


@dataclass
class MigrationResult:
    """Outcome of a backfill_v2() run."""
    records_migrated: int = 0
    audit_entries_written: int = 0
    columns_added: list[str] | None = None
    skipped: str | None = None  # set when no-op (e.g., "already_v2")

    def __post_init__(self):
        if self.columns_added is None:
            self.columns_added = []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_has_v2_fields(table) -> bool:
    """Check if the memory table already has v2 columns."""
    schema_field_names = {f.name for f in table.schema}
    v2_field_names = {f.name for f in V2_FIELDS}
    return v2_field_names.issubset(schema_field_names)


def _ensure_audit_table(db) -> None:
    """Create the memory_audit table if missing."""
    if AUDIT_TABLE_NAME not in list(db.table_names()):
        db.create_table(AUDIT_TABLE_NAME, schema=LANCE_AUDIT_SCHEMA)


def _add_v2_columns_to_table(table, existing_rows: list[dict]) -> list[str]:
    """Add v2 columns to an existing v1 table.

    LanceDB's `add_columns` takes a function or pyarrow expression.
    For a backfill we compute the values per-row and pass via merge_insert.
    """
    columns_added = []
    schema_field_names = {f.name for f in table.schema}

    # Check which v2 fields are missing
    missing_fields = [f for f in V2_FIELDS if f.name not in schema_field_names]
    if not missing_fields:
        return []

    # Build a dict per row: {id: {new_field: default_value, ...}}
    # Use add_columns with a SQL expression (cheapest path)
    # LanceDB add_columns syntax: dict of column_name -> SQL expression string
    sql_expressions = {}
    for field in missing_fields:
        default = V2_DEFAULTS.get(field.name)
        if field.type == pa.utf8():
            # SQL string literal
            sql_expressions[field.name] = f"'{default}'"
        elif field.type == pa.float32():
            sql_expressions[field.name] = f"CAST({default} AS FLOAT)"
        elif field.type == pa.int32():
            sql_expressions[field.name] = f"CAST({default} AS INT)"
        else:
            sql_expressions[field.name] = f"'{default}'"
        columns_added.append(field.name)

    # Special cases: valid_from = created_at, last_seen_at = last_accessed
    if "valid_from" in sql_expressions:
        sql_expressions["valid_from"] = "created_at"
    if "last_seen_at" in sql_expressions:
        sql_expressions["last_seen_at"] = "last_accessed"

    table.add_columns(sql_expressions)
    return columns_added


def _write_audit_entries(db, memory_ids: list[str]) -> int:
    """Write one backfill_v2 audit entry per migrated record."""
    if not memory_ids:
        return 0

    audit_table = db.open_table(AUDIT_TABLE_NAME)
    now = _now_iso()
    rows = [
        {
            "id": str(uuid.uuid4()),
            "memory_id": mid,
            "op": "backfill_v2",
            "who": "manual",
            "when": now,
            "why": "schema migration v1->v2: added tier, provenance, bitemporal fields",
            "before": "",
            "after": json.dumps(V2_DEFAULTS),
        }
        for mid in memory_ids
    ]
    audit_table.add(rows)
    return len(rows)


def backfill_v2(store: MemoryStore) -> MigrationResult:
    """Migrate a v1 LanceDB table to v2 schema.

    Steps:
      1. Check if already v2 (idempotent)
      2. Snapshot existing rows
      3. Ensure memory_audit table exists
      4. Add v2 columns to memory table (LanceDB add_columns — metadata only)
      5. Write audit entries for each backfilled record
      6. Return MigrationResult

    Acquires the store's file lock so concurrent migration calls serialize safely.

    Args:
        store: MemoryStore instance pointing at the table to migrate.

    Returns:
        MigrationResult with counts and skip reason if applicable.
    """
    with store.lock:
        # Step 1: idempotency check
        if _table_has_v2_fields(store.table):
            return MigrationResult(skipped="already_v2")

        # Step 2: snapshot existing data
        existing_rows = store.table.to_pandas().to_dict(orient="records")
        memory_ids = [row["id"] for row in existing_rows]

        # Step 3: ensure audit table
        _ensure_audit_table(store.db)

        # Step 4: add v2 columns
        columns_added = _add_v2_columns_to_table(store.table, existing_rows)

        # Step 5: write audit entries
        audit_count = _write_audit_entries(store.db, memory_ids)

        return MigrationResult(
            records_migrated=len(memory_ids),
            audit_entries_written=audit_count,
            columns_added=columns_added,
        )


# ---- CLI entrypoint ----

def main():
    """Run migration on the user's actual LanceDB store."""
    from memory.store import MemoryStore
    from memory.embeddings import EmbeddingService

    print("Initializing MemoryStore...")
    store = MemoryStore(embedder=EmbeddingService())

    print(f"Pre-migration count: {store.count()}")
    print(f"Pre-migration tables: {store.list_tables()}")

    print("Running backfill_v2...")
    result = backfill_v2(store)

    if result.skipped:
        print(f"  Skipped: {result.skipped}")
    else:
        print(f"  Migrated {result.records_migrated} records")
        print(f"  Wrote {result.audit_entries_written} audit entries")
        print(f"  Added columns: {', '.join(result.columns_added)}")

    print(f"Post-migration tables: {store.list_tables()}")
    print("Done.")


if __name__ == "__main__":
    main()
