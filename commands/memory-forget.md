---
description: Tombstone a specific memory by id. Audit log retains the receipt; idempotent.
---

Surgical removal of a memory you found wrong via /memory-search. Tombstones
the record (excludes from default retrieval) but preserves an audit-log
entry recording who/when/why.

Usage: /memory-forget <memory-id> "<reason>"

The first arg is the memory id (from /memory-search output, the [xxxxxxxx]
prefix is the first 8 chars). The optional reason is recorded in the audit
log.

!python -m memory.search_cli forget $ARGUMENTS
