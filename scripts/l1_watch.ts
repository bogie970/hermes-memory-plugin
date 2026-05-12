#!/usr/bin/env npx tsx
/**
 * L1 watcher — Stop hook async sibling.
 *
 * Estimates current transcript token count. If above threshold, spawns
 * the Python L1 manager CLI detached. Never blocks; never non-zero exits.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { readTranscript } from './transcript_utils.ts';
import { buildPythonSubprocessEnv, getMode, getTempStateDir, readBoundedStdinJson, recordHookError } from './conversation_utils.ts';
import { getConfig } from './config.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CONTEXT_LIMIT = parseInt(process.env.HERMES_L1_CONTEXT_LIMIT ?? '200000');
const TRIGGER_FRACTION = parseFloat(process.env.HERMES_L1_TRIGGER_FRACTION ?? '0.6');
const EVICT_FRACTION = parseFloat(process.env.HERMES_L1_EVICT_FRACTION ?? '0.5');
const PIN_RECENT = parseInt(process.env.HERMES_L1_PIN_RECENT ?? '20');
const CHARS_PER_TOKEN = 3.5;
// Cooldown after a successful eviction. Prevents firing Haiku on every Stop
// hook while transcript stays past the 60% mark.
const COOLDOWN_MS = parseInt(process.env.HERMES_L1_COOLDOWN_MS ?? '300000');  // 5 min

const TEMP_STATE_DIR = getTempStateDir();
const LOG_FILE = path.join(TEMP_STATE_DIR, 'l1_watch.log');

interface HookInput {
  session_id: string;
  transcript_path: string;
  cwd: string;
  hook_event_name?: string;
  stop_hook_active?: boolean;
}

function ensureLogDir(): void {
  if (!fs.existsSync(TEMP_STATE_DIR)) {
    fs.mkdirSync(TEMP_STATE_DIR, { recursive: true });
  }
}

function log(msg: string): void {
  ensureLogDir();
  fs.appendFileSync(LOG_FILE, `[${new Date().toISOString()}] ${msg}\n`);
}

async function readHookInput(): Promise<HookInput> {
  const v = await readBoundedStdinJson<HookInput>(30000);
  if (!v) throw new Error('empty or oversized stdin');
  return v;
}

function sanitizeCwd(cwd: string): string {
  return cwd.replace(/[\\/:]/g, '-').replace(/^-+/, '');
}

function getMarkerDir(cwd: string): string {
  return path.join(os.homedir(), '.claude', 'projects', sanitizeCwd(cwd), 'l1_markers');
}

/**
 * Returns true if a marker file in the dir was written within COOLDOWN_MS.
 * Markers (active or .consumed-*) count — we just want to know "did an
 * eviction happen recently."
 */
function recentEviction(markerDir: string, cooldownMs: number): boolean {
  if (!fs.existsSync(markerDir)) return false;
  const cutoff = Date.now() - cooldownMs;
  try {
    for (const name of fs.readdirSync(markerDir)) {
      if (!name.startsWith('l1_evicted_')) continue;
      const stat = fs.statSync(path.join(markerDir, name));
      if (stat.mtimeMs >= cutoff) return true;
    }
  } catch {
    // ignore — fall through
  }
  return false;
}

/**
 * Estimate in-context tokens from a Claude Code JSONL transcript.
 *
 * The naive heuristic (file.size / 3.5) overcounts massively because the
 * JSONL file accumulates EVERY message ever written in the session,
 * including raw tool_result payloads that get replaced by compaction
 * summaries but never deleted from disk.
 *
 * Strategy:
 *   1. Walk lines newest -> oldest, find the most recent compaction
 *      marker. Claude Code writes these as a `user` entry with
 *      `isCompactSummary: true` (also accept generic `type: summary`
 *      or top-level `summary` field for forward-compat).
 *   2. Sum content/tool_use/tool_result/summary chars from that index
 *      forward — everything before is already replaced by the summary
 *      and should not be double-counted.
 *   3. Divide by CHARS_PER_TOKEN.
 *
 * If no compaction marker exists (early/short sessions), count all lines.
 */
function estimateTokens(transcriptPath: string): number {
  try {
    const data = fs.readFileSync(transcriptPath, 'utf-8');
    const lines = data.split('\n').filter((l) => l.trim());

    // Find the most recent compaction marker.
    let summaryIdx = -1;
    for (let i = lines.length - 1; i >= 0; i--) {
      try {
        const entry = JSON.parse(lines[i]);
        if (
          entry.isCompactSummary === true ||
          entry.type === 'summary' ||
          (typeof entry.summary === 'string' && entry.summary.length > 0)
        ) {
          summaryIdx = i;
          break;
        }
      } catch {
        continue;
      }
    }

    const startIdx = summaryIdx >= 0 ? summaryIdx : 0;
    let totalChars = 0;
    for (let i = startIdx; i < lines.length; i++) {
      try {
        const entry = JSON.parse(lines[i]);
        // Pull content from the common shapes we see in CC transcripts.
        // message.content can be a string or an array of typed blocks
        // (text, tool_use, tool_result, thinking, image). JSON.stringify
        // gives us a conservative char count over the whole payload.
        const payload =
          entry.message?.content ??
          entry.content ??
          entry.summary ??
          '';
        const s =
          typeof payload === 'string' ? payload : JSON.stringify(payload);
        totalChars += s.length;
      } catch {
        continue;
      }
    }

    return Math.ceil(totalChars / CHARS_PER_TOKEN);
  } catch {
    return 0;
  }
}

// Exported for ad-hoc verification scripts.
export { estimateTokens };

function spawnL1Manager(
  pythonPath: string,
  pythonDir: string,
  transcriptPath: string,
  markerDir: string,
  sessionId: string,
): void {
  const env = buildPythonSubprocessEnv({ PYTHONPATH: pythonDir });
  const child = spawn(pythonPath, [
    '-m', 'memory.l1_manager_cli',
    '--transcript', transcriptPath,
    '--marker-dir', markerDir,
    '--session-id', sessionId,
    '--evict-fraction', String(EVICT_FRACTION),
    '--pin-recent', String(PIN_RECENT),
  ], {
    detached: true,
    stdio: 'ignore',
    cwd: pythonDir,
    env,
    windowsHide: true,
  });
  child.unref();
  log(`spawned l1_manager_cli (pid ${child.pid})`);
}

async function main(): Promise<void> {
  log('='.repeat(40));
  log('l1_watch start');

  if (getMode() === 'off') {
    log('mode=off, skipping');
    process.exit(0);
  }

  let hookInput: HookInput;
  try {
    hookInput = await readHookInput();
  } catch (e) {
    log(`stdin parse failed: ${e}`);
    process.exit(0);
  }

  if (hookInput.stop_hook_active) {
    log('stop_hook_active, skipping to prevent loop');
    process.exit(0);
  }

  const tokens = estimateTokens(hookInput.transcript_path);
  const threshold = CONTEXT_LIMIT * TRIGGER_FRACTION;
  log(`tokens=${tokens} threshold=${threshold}`);

  if (tokens < threshold) {
    log('below threshold, skipping eviction');
    process.exit(0);
  }

  // Plugin-bundled python tree (the canonical install)
  const cfg = getConfig();
  const pythonDir = (cfg as any).pythonDir || path.resolve(path.dirname(__filename), '..', 'python');

  const markerDir = getMarkerDir(hookInput.cwd);
  try {
    fs.mkdirSync(markerDir, { recursive: true });
  } catch (e) {
    log(`mkdir markerDir failed: ${e}`);
    process.exit(0);
  }

  // Cooldown: don't re-fire if we evicted within the cooldown window.
  // Prevents Stop-hook hammering when transcript stays past 60% of context.
  if (recentEviction(markerDir, COOLDOWN_MS)) {
    log(`recent eviction within ${COOLDOWN_MS}ms — skipping`);
    process.exit(0);
  }

  try {
    spawnL1Manager(
      cfg.pythonPath,
      pythonDir,
      hookInput.transcript_path,
      markerDir,
      hookInput.session_id,
    );
    log('eviction spawned, exiting');
  } catch (e) {
    log(`spawn failed: ${e}`);
  }
  process.exit(0);
}

main().catch((e) => {
  log(`unhandled: ${e}`);
  recordHookError('l1_watch.ts', e);
  process.exit(0);
});
