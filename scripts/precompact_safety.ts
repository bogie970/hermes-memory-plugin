#!/usr/bin/env npx tsx
/**
 * PreCompact safety net — synchronous fallback.
 *
 * If the async L1 watcher hasn't run recently, do a synchronous eviction
 * before Claude Code's own compaction destroys the context.
 *
 * NEVER blocks compaction. Always exits 0 with {"continue": true}.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import { getMode, getTempStateDir } from './conversation_utils.ts';
import { getConfig } from './config.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const RECENT_MARKER_WINDOW_MS = 5 * 60 * 1000;  // 5 minutes
const SYNC_TIMEOUT_MS = 20_000;
const EVICT_FRACTION = parseFloat(process.env.HERMES_L1_EVICT_FRACTION ?? '0.5');
const PIN_RECENT = parseInt(process.env.HERMES_L1_PIN_RECENT ?? '20');

const TEMP_STATE_DIR = getTempStateDir();
const LOG_FILE = path.join(TEMP_STATE_DIR, 'precompact_safety.log');

interface HookInput {
  session_id: string;
  transcript_path: string;
  cwd: string;
  hook_event_name?: string;
  compaction_trigger?: 'manual' | 'auto';
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
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('readable', () => {
      let chunk;
      while ((chunk = process.stdin.read()) !== null) {
        data += chunk;
      }
    });
    process.stdin.on('end', () => {
      try {
        resolve(JSON.parse(data));
      } catch (e) {
        reject(new Error(`stdin parse: ${e}`));
      }
    });
    process.stdin.on('error', reject);
  });
}

function sanitizeCwd(cwd: string): string {
  return cwd.replace(/[\\/:]/g, '-').replace(/^-+/, '');
}

function getMarkerDir(cwd: string): string {
  return path.join(os.homedir(), '.claude', 'projects', sanitizeCwd(cwd), 'l1_markers');
}

function hasRecentMarker(markerDir: string): boolean {
  if (!fs.existsSync(markerDir)) return false;
  const cutoff = Date.now() - RECENT_MARKER_WINDOW_MS;
  try {
    for (const name of fs.readdirSync(markerDir)) {
      if (!name.startsWith('l1_evicted_') || !name.endsWith('.md')) continue;
      const full = path.join(markerDir, name);
      const stat = fs.statSync(full);
      if (stat.mtimeMs >= cutoff) return true;
    }
  } catch (e) {
    log(`hasRecentMarker error: ${e}`);
  }
  return false;
}

function runSyncEviction(
  pythonPath: string,
  hermesRoot: string,
  transcriptPath: string,
  markerDir: string,
  sessionId: string,
): void {
  const pythonDir = path.join(hermesRoot, 'aisys');
  const env = {
    ...process.env,
    PYTHONPATH: `${pythonDir}${path.delimiter}${hermesRoot}`,
    HERMES_ROOT: hermesRoot,
  };
  try {
    const result = spawnSync(pythonPath, [
      '-m', 'aisys.memory.l1_manager_cli',
      '--transcript', transcriptPath,
      '--marker-dir', markerDir,
      '--session-id', sessionId,
      '--evict-fraction', String(EVICT_FRACTION),
      '--pin-recent', String(PIN_RECENT),
    ], {
      timeout: SYNC_TIMEOUT_MS,
      encoding: 'utf-8',
      cwd: hermesRoot,
      env,
      windowsHide: true,
    });
    log(`sync eviction status=${result.status} stdout=${(result.stdout || '').slice(0, 200)} stderr=${(result.stderr || '').slice(0, 200)}`);
  } catch (e) {
    log(`sync eviction error: ${e}`);
  }
}

function emitContinue(): void {
  // PreCompact hook never blocks. Always pass through.
  process.stdout.write(JSON.stringify({ continue: true }) + '\n');
  process.exit(0);
}

async function main(): Promise<void> {
  log('='.repeat(40));
  log('precompact_safety start');

  if (getMode() === 'off') {
    log('mode=off, passing through');
    return emitContinue();
  }

  let hookInput: HookInput;
  try {
    hookInput = await readHookInput();
  } catch (e) {
    log(`stdin parse failed: ${e}`);
    return emitContinue();
  }

  log(`trigger=${hookInput.compaction_trigger} session=${hookInput.session_id}`);

  const markerDir = getMarkerDir(hookInput.cwd);
  if (hasRecentMarker(markerDir)) {
    log('recent marker found, watcher already ran — skipping');
    return emitContinue();
  }

  const cfg = getConfig();
  const hermesRoot = (
    process.env.HERMES_ROOT
    || (cfg as any).hermesRoot
    || path.resolve(path.dirname(__filename), '..', '..', 'hermes')
  );

  try {
    fs.mkdirSync(markerDir, { recursive: true });
  } catch (e) {
    log(`mkdir markerDir failed: ${e}`);
    return emitContinue();
  }

  runSyncEviction(
    cfg.pythonPath,
    hermesRoot,
    hookInput.transcript_path,
    markerDir,
    hookInput.session_id,
  );

  emitContinue();
}

main().catch((e) => {
  log(`unhandled: ${e}`);
  emitContinue();
});
