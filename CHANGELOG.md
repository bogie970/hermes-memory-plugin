# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — 2026-05-11/12 production debugging session

### Fixed

- **`a7331e8` (2026-05-11) — ConPTY silent-launcher stdin bypass.** THIS WAS THE ROOT FIX for the silent-hook bug that had been masquerading as a dozen smaller problems. `hooks/silent-npx.cjs` now detects when stdin is piped and bypasses `silent-launcher.exe` entirely — ConPTY was swallowing the piped JSON payload from Claude Code. Symptom prior to fix: Stop / PreCompact hooks fired but produced no log entries, no LanceDB writes, no observable side-effect.

- **`0f1f788` (2026-05-11) — Hook stdin timeout 2000ms → 30000ms, max bytes 4MB → 64MB.** Claude Code now ships 10+ MB transcript payloads on Stop; the previous caps silently truncated input. Defense in depth — the ConPTY fix above was the real cause, but these limits were also too low for current Claude Code behavior.

- **`10ab4e4` (2026-05-11) — Log volume fix in `transcript_utils.ts`.** Replaced per-message log entries with a single summary line per invocation; added 10MB log rotation in `stop_capture.ts`. `send_messages.log` was previously growing at multi-MB-per-day rates.

- **`7f7742` (2026-05-11) — Plugin pointed at empty `~/.hermes` instead of populated hermes repo.** Resolution path fix.

- **`da9d473` (2026-05-11) — Sync `llm_caller.py` + `runner.py` from hermes.** Brings in: `DETACHED_PROCESS` flag on the `claude --print` subprocess and `LETTA_MODE=off` / `HERMES_MODE=off` env vars so the nested Claude invocation does not re-trigger plugin hooks; file-lock singleton via `portalocker.lock` in `runner.py` replacing the old fragile PID check.

  **Hallucination caveat (transparency):** The hermes-side commits motivating this sync (`5da9367`, `753d44b`) were originally diagnosed as fixing a "recursive runner chain." That observation was wrong — what looked like recursion was the diagnostic command's own bash/powershell subprocess tree being misidentified. The code is still good defensive practice (detaching nested Claude subprocesses and using a file-lock singleton are both correct), but the original bug report driving the change did not exist. Documented here rather than silently rewritten.

### Added

- **`f1b8796` (2026-05-11) — `sync-cache.ps1` / `sync-cache.sh` scripts.** Propagate edits from the source repo to the active plugin cache (`~/.claude/plugins/cache/...`) so changes take effect without a full re-install. Closes the dev-loop gap where source edits had no effect because the cache copy was stale.

- **`2cad254` (2026-05-11) — `portalocker` added to `requirements-memory.txt`.** Required by the file-lock singleton in `runner.py`.

- **`2da71ed` (2026-05-11) — Embedding daemon + vector index management.** Long-lived embedding process replaces per-call spawn; vector index lifecycle (create / compact / verify) is now explicit.

---

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
