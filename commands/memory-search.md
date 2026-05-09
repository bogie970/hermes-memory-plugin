---
description: Search the Hermes memory store directly. Returns top-k matches with tier, source, and content.
---

Query the L2 vector store for memories matching a natural-language query.
Returns the top-5 hits with their content, tier, source_ref, and similarity
score.

Usage: /memory-search how did we set up the cleanup pass

Excludes tombstoned memories by default.

!python -m memory.search_cli search "$ARGUMENTS"
