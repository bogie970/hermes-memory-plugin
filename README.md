# Hermes Memory Plugin

A local-first hierarchical memory plugin for [Claude Code](https://claude.ai/code).
Keeps a single session alive longer and accumulates project knowledge across sessions —
no cloud, no API key, runs entirely on your machine.

> Forked from [`letta-ai/claude-subconscious`](https://github.com/letta-ai/claude-subconscious).
> The TypeScript hook plumbing came from there. Everything below the hooks (LanceDB store,
> tiered intelligence, write gate, contradiction model, eval harness, adversarial suite)
> is built fresh for the local-first single-user case.

## What it does

Claude Code forgets between sessions and lossily compacts when context fills.
Hermes Memory adds:

- **L1.notes** — Haiku-managed structured notes that compress older context inline
- **L2** — LanceDB vector store with semantic recall across all your sessions
- **Archive** — cold storage tier, queryable on demand
- **Patterns** — slow-moving identity layer (the 5 letta-style blocks)
- **Audit log** — append-only operation history for forensic rollback

A 4-tier write pipeline (`candidate → probationary → verified`, with `tombstoned`
for retracts) prevents memory poisoning. A bitemporal contradiction model
(Zep-inspired) lets new claims supersede old ones without losing history.

Maintenance runs through three tiers of intelligence:

- **Haiku** — real-time on every Stop hook (curate, embed, tag)
- **Sonnet** — daily promotion pass via `/consolidate-memory`
- **Opus** — weekly audit (planned)

All maintenance uses your Claude Max plan via `claude --print` — no API key needed.

## Architecture

```
┌─ L1.recent ─── verbatim recent transcript (untouched)
├─ L1.notes ──── triple-pack format, Haiku-managed
├─ L2 ────────── LanceDB vectors, cross-session, auto-injected
├─ Archive ───── LanceDB cold tier, query on demand
├─ Patterns ──── identity layer (cross-session, slow-moving)
└─ Audit log ─── append-only ops history
```

When the active transcript crosses 60% of the context limit, the `l1_watch.ts`
Stop-hook spawns the Python L1 manager. It picks a safe cut point (never
splitting tool_use pairs, exempting last 20 turns), sends the older half to
Haiku for semantic segmentation, vectorizes each chunk into L2 with
`source="l1_evict"`, and writes a placeholder marker. On your next prompt,
`sync_letta_memory.ts` injects the marker so Claude knows the content was
moved and how to retrieve it.

## Install

```bash
git clone https://github.com/bogie970/hermes-memory-plugin
cd hermes-memory-plugin
./install.ps1   # Windows PowerShell
# or
./install.sh    # macOS / Linux
```

The installer:

1. Creates `~/.hermes/` (data, venv, logs, models)
2. Installs Python deps (LanceDB, sentence-transformers, etc.)
3. Pre-downloads the embedding model (`gte-modernbert-base`, ~500MB)
4. Installs Node deps for the TS hooks
5. Writes `hermes.config.json` with absolute paths

Then register the plugin in `~/.claude/settings.json`:

```json
{
  "extraKnownMarketplaces": {
    "hermes-memory": {
      "source": {
        "source": "directory",
        "path": "/absolute/path/to/hermes-memory-plugin"
      }
    }
  },
  "enabledPlugins": {
    "hermes-memory@hermes-memory": true
  }
}
```

Restart Claude Code, run `/reload-plugins`, you should see the Hermes Memory
banner on session start.

## Slash commands

- `/maintenance-status` — show recent runs + tier breakdown
- `/memory-stats` — quick tier counts
- `/consolidate-memory` — manually trigger Sonnet daily promotion
- `/dream` — full maintenance cycle (dry-run + promote + status)

## Configuration

Tunable via env or `hermes.config.json`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `HERMES_L1_TRIGGER_FRACTION` | 0.6 | When to evict (fraction of context limit) |
| `HERMES_L1_EVICT_FRACTION` | 0.5 | How much to evict (oldest fraction) |
| `HERMES_L1_PIN_RECENT` | 20 | Turns exempt from eviction |
| `HERMES_L1_CONTEXT_LIMIT` | 200000 | Context window size |
| `LETTA_MODE` | full | `full` / `whisper` / `off` |

## Development

```bash
# Run the test suite
PYTHONPATH=hermes/aisys python -m pytest hermes/aisys/memory/tests/ -v

# Phase B+C+D+E+F+G+H combined: 108 tests, ~1 minute
```

## License

MIT. Original Letta scaffold MIT-licensed; Hermes additions also MIT.
