# Changelog

## [2.0.0] - 2026-05-06 — v2 rebuild as Hermes Memory Plugin

Comprehensive rebuild from Letta cross-session-agent paradigm to local-first
hierarchical memory. Forked from `letta-ai/claude-subconscious`. See
`hermes/docs/memory/MASTER_PLAN.md` for the full rationale and design.

### Architecture

- **4-layer memory model**: L1.recent (verbatim recent transcript) + L1.notes
  (Haiku-managed triple-pack) + L2 (LanceDB vector store, cross-session) +
  Archive (cold tier). Patterns layer (5 letta-style blocks) preserved as
  complementary identity layer. Audit log as append-only ops history.

- **Tiered write pipeline**: every memory has tier `candidate` /
  `probationary` / `verified` / `tombstoned`. `subconscious_haiku` writers
  forced to `candidate`; user-stated facts go directly to `verified`. Sonnet
  daily promotion runs candidate → probationary based on re-encounter.

- **Bitemporal facts** (Zep-inspired): never destructive delete. Supersession
  sets `valid_to` and `superseded_by` link. Old chains preserved.

- **Triple-pack format**: dense LLM-readable notation for L1.notes
  (`#id @subj :pred obj ^prov.`) with edit ops (`+ ~ - x`) and cross-refs
  (`@entity`, `L2:hex`, `#id`).

### Added

- L1 manager with eviction, segmentation via Haiku, fallback path
- TS hooks: `l1_watch.ts` (Stop async), `precompact_safety.ts` (PreCompact sync)
- Recall@k eval harness with fixture synthesis (`memory.eval.harness`,
  `memory.eval.synthesize`)
- Daily Sonnet promotion job (`memory.promotion.run_daily`)
- Bitemporal contradiction adjudication (`memory.contradictions`)
- Maintenance log + `/maintenance-status` `/memory-stats` `/consolidate-memory`
  `/dream` slash commands
- Schema migration v1 → v2 (idempotent backfill, audit log table)
- File-locking on writes (FileLock around dedup-check + insert)
- Filesystem grounding: candidate-demote on missing code refs
- Adversarial test suite covering 10 failure modes (memory poisoning,
  hallucination amplification, embedding drift, parser fuzz, etc.)

### Changed

- **Plugin name**: `claude-subconscious` → `hermes-memory-plugin` (avoids
  conflict with upstream Letta repo)
- **Embedding model**: `all-MiniLM-L6-v2` (384-dim) → `gte-modernbert-base`
  (768-dim)
- **Provider**: Letta Cloud API → local LanceDB + claude-cli for Haiku
  invocations (uses Max plan, no API key)
- **Plugin python tree**: now bundles canonical hermes/aisys/memory/ — no
  longer requires sibling hermes repo

### Removed

- ~3,400 LOC of Letta cloud paradigm: agent_config, letta_api_url,
  send_worker_sdk, pretool_sync, compaction_guard, conversation_utils.test
- `Subconscious.af` (Letta agent template)
- All `isLocalMode()` branches (single execution path now)

### Security

- Marker file injection defense (strict filename pattern, 64KB cap, XML escape)
- SQL sanitization on dedup where-clause
- Subconscious dedup cannot boost verified seen_count (poisoning prevention)
- Concurrent promotion serialized via FileLock

### Tests

- 108/108 pass on bundled tree, ~60s full run
- Property-based fuzzing on triple-pack parser via Hypothesis
- 11 adversarial chaos tests (poisoning, race conditions, fuzz, contradiction
  cycles, oversized content)

### Cross-platform

- `.gitattributes` added (CRLF protection for `silent-launcher.exe`)
- TS hooks use OS-portable `path.delimiter` for PYTHONPATH
- Windows / macOS / Linux install scripts kept in sync

---

## [1.1.0] - 2026-01-28 (legacy Letta upstream)

### Added

- **PreToolUse hook for mid-workflow context injection** - New lightweight hook that checks for Letta agent updates before each tool use. Addresses "workflow drift" in long workflows by injecting new messages and memory block diffs mid-stream. Silent no-op if nothing changed.

- **Letta Code GitHub Action** - `@letta-code` can now respond to issues and PRs in this repository.

- **LETTA_BASE_URL support** - Self-hosted Letta servers can now be configured via environment variable.

- **Windows compatibility** - Fixed `npx spawn ENOENT` error on Windows.

- **Linux tmpfs workaround** - Documented workaround for `EXDEV` error when `/tmp` is on a different filesystem.

### Changed

- **Session start sync** - CLAUDE.md now syncs at session start for fresh agent/conversation IDs.

- **Default model** - Changed default agent model to GLM 4.7 (free tier on Letta Cloud).

- **Automatic model detection** - Plugin now queries available models and auto-selects if configured model is unavailable.

### Fixed

- **Plugin install syntax** - Updated README with correct marketplace install commands.

- **Conversation message ordering** - Fixed message fetch to correctly show newest messages first.

- **Conversation URL** - Links now point to agent view with conversation query param.

### Security

- **Sanitized default agent** - Removed user-specific data from bundled `Subconscious.af` file.

---

## [1.0.0] - 2026-01-16 (legacy Letta upstream)

Initial release.

### Features

- Bidirectional sync between Claude Code and Letta agents
- Memory blocks sync to `.claude/CLAUDE.md`
- Session transcripts sent to Letta agent asynchronously
- Conversation isolation per Claude Code session
- Auto-import default Subconscious agent if no agent configured
- Memory block diffs shown on changes
- New messages from Letta agent injected into context

### Hooks

- `SessionStart` - Notify agent of new session
- `UserPromptSubmit` - Sync memory before each prompt
- `Stop` - Send transcript after each response
