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
import { getMode, getTempStateDir } from './conversation_utils.ts';
import { getConfig } from './config.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CONTEXT_LIMIT = parseInt(process.env.HERMES_L1_CONTEXT_LIMIT ?? '200000');
const TRIGGER_FRACTION = parseFloat(process.env.HERMES_L1_TRIGGER_FRACTION ?? '0.6');
const EVICT_FRACTION = parseFloat(process.env.HERMES_L1_EVICT_FRACTION ?? '0.5');
const PIN_RECENT = parseInt(process.env.HERMES_L1_PIN_RECENT ?? '20');
const CHARS_PER_TOKEN = 3.5;

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
        reject(new Error(`Failed to parse stdin: ${e}`));
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

function estimateTokens(transcriptPath: string): number {
  try {
    const stats = fs.statSync(transcriptPath);
    return Math.ceil(stats.size / CHARS_PER_TOKEN);
  } catch {
    return 0;
  }
}

function spawnL1Manager(
  pythonPath: string,
  pythonDir: string,
  transcriptPath: string,
  markerDir: string,
  sessionId: string,
): void {
  const env = {
    ...process.env,
    PYTHONPATH: pythonDir,
  };
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
  process.exit(0);
});
