#!/usr/bin/env npx tsx
/**
 * SessionStart hook — local-only.
 *
 * Shows the Hermes Memory banner, initializes session state, and scrubs any
 * legacy <letta> sections from CLAUDE.md.
 *
 * No Letta cloud calls.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  cleanLettaFromClaudeMd,
  getMode,
  getTempStateDir,
  expandPath,
} from './conversation_utils.ts';
import { getLocalConversationId } from './local_store.ts';

const TEMP_STATE_DIR = getTempStateDir();
const LOG_FILE = path.join(TEMP_STATE_DIR, 'session_start.log');

interface HookInput {
  session_id: string;
  cwd: string;
  hook_event_name?: string;
}

function getDurableStateDir(cwd: string): string {
  const raw = process.env.LETTA_HOME || cwd;
  const base = process.env.LETTA_HOME ? expandPath(raw) : raw;
  return path.join(base, '.letta', 'claude');
}

function getSyncStateFile(cwd: string, sessionId: string): string {
  return path.join(getDurableStateDir(cwd), `session-${sessionId}.json`);
}

function ensureLogDir(): void {
  if (!fs.existsSync(TEMP_STATE_DIR)) {
    fs.mkdirSync(TEMP_STATE_DIR, { recursive: true });
  }
}

function ensureDurableStateDir(cwd: string): void {
  const dir = getDurableStateDir(cwd);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function log(message: string): void {
  ensureLogDir();
  const timestamp = new Date().toISOString();
  const logLine = `[${timestamp}] ${message}\n`;
  fs.appendFileSync(LOG_FILE, logLine);
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
        reject(new Error(`Failed to parse hook input: ${e}`));
      }
    });
    process.stdin.on('error', reject);
  });
}

function saveSessionState(cwd: string, sessionId: string, conversationId: string): void {
  ensureDurableStateDir(cwd);
  const state = {
    sessionId,
    conversationId,
    lastProcessedIndex: -1,
    startedAt: new Date().toISOString(),
  };
  fs.writeFileSync(getSyncStateFile(cwd, sessionId), JSON.stringify(state, null, 2), 'utf-8');
}

async function main(): Promise<void> {
  log('='.repeat(60));
  log('session_start started');

  const mode = getMode();
  log(`Mode: ${mode}`);
  if (mode === 'off') {
    log('Mode is off, exiting');
    process.exit(0);
  }

  // Banner: Unix uses /dev/tty, Windows uses stderr
  let tty: { write(s: string): boolean; end?(): void } | null = null;
  if (process.platform === 'win32') {
    tty = process.stderr;
  } else {
    try {
      const stream = fs.createWriteStream('/dev/tty');
      stream.on('error', () => { tty = null; });
      tty = stream;
    } catch {
      // TTY not available
    }
  }

  const writeTty = (text: string) => {
    if (tty) tty.write(text);
  };

  try {
    log('Reading hook input from stdin...');
    const hookInput = await readHookInput();
    log(`Hook input: session_id=${hookInput.session_id}, cwd=${hookInput.cwd}`);

    // Hermes Memory banner
    writeTty('\n');
    writeTty('\x1b[1m  Hermes Memory\x1b[0m \x1b[2m(local)\x1b[0m\n');
    writeTty('\x1b[35m');
    writeTty('  ▐\x1b[31m▛\x1b[35m███\x1b[31m▜\x1b[35m▌\n');
    writeTty(' ▝▜█████▛▘\n');
    writeTty('   ▘▘ ▝▝\n');
    writeTty('\x1b[0m');
    writeTty('\x1b[2m');
    writeTty(`  Mode:    ${mode}\n`);
    writeTty('  Storage: LanceDB + pattern blocks (file-backed)\n');
    writeTty('\x1b[0m\n');
    if (tty && tty.end && tty !== process.stderr) tty.end();

    const conversationId = getLocalConversationId(hookInput.session_id);
    saveSessionState(hookInput.cwd, hookInput.session_id, conversationId);

    // Legacy CLAUDE.md scrub (one-shot migration; safe to call repeatedly)
    cleanLettaFromClaudeMd(hookInput.cwd);
    const homeDir = process.env.HOME || os.homedir();
    if (homeDir !== hookInput.cwd) {
      cleanLettaFromClaudeMd(homeDir);
    }

    log('session_start completed');

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    log(`ERROR: ${errorMessage}`);

    writeTty('\r\x1b[K');
    writeTty('\x1b[31m');
    writeTty(`  Hermes Memory error: ${errorMessage}\n`);
    writeTty('\x1b[0m');
    if (tty && tty.end && tty !== process.stderr) tty.end();

    process.exit(1);
  }
}

main();
