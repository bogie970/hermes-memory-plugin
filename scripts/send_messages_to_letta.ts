#!/usr/bin/env npx tsx
/**
 * Stop hook — local-only.
 *
 * On every Stop event:
 *   1. Read transcript JSONL
 *   2. Diff against last processed index
 *   3. Write payload JSON
 *   4. Spawn local Python worker (subconscious) detached
 *
 * No Letta cloud calls. No LETTA_API_KEY required.
 */

import * as fs from 'fs';
import * as path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import {
  loadSyncState,
  saveSyncState,
  getSyncStateFile,
  getMode,
  getTempStateDir,
} from './conversation_utils.ts';
import {
  readTranscript,
  formatMessagesForLetta,
  formatAsXmlTranscript,
} from './transcript_utils.ts';
import { getConfig } from './config.ts';

const hermesConfig = getConfig();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TEMP_STATE_DIR = getTempStateDir();
const LOG_FILE = path.join(TEMP_STATE_DIR, 'send_messages.log');

interface HookInput {
  session_id: string;
  transcript_path: string;
  stop_hook_active?: boolean;
  cwd: string;
  hook_event_name?: string;
}

function ensureLogDir(): void {
  if (!fs.existsSync(TEMP_STATE_DIR)) {
    fs.mkdirSync(TEMP_STATE_DIR, { recursive: true });
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

async function main(): Promise<void> {
  log('='.repeat(60));
  log('stop_capture started');

  const mode = getMode();
  log(`Mode: ${mode}`);
  if (mode === 'off') {
    log('Mode is off, exiting');
    process.exit(0);
  }

  try {
    log('Reading hook input from stdin...');
    const hookInput = await readHookInput();
    log(`Hook input: session=${hookInput.session_id} transcript=${hookInput.transcript_path} cwd=${hookInput.cwd}`);

    if (hookInput.stop_hook_active) {
      log('Stop hook already active, exiting to prevent loop');
      process.exit(0);
    }

    log(`Reading transcript from: ${hookInput.transcript_path}`);
    const messages = await readTranscript(hookInput.transcript_path, log);
    log(`Found ${messages.length} messages in transcript`);

    if (messages.length === 0) {
      log('No messages found, exiting');
      process.exit(0);
    }

    const typeCounts: Record<string, number> = {};
    for (const msg of messages) {
      const key = msg.type || msg.role || 'unknown';
      typeCounts[key] = (typeCounts[key] || 0) + 1;
    }
    log(`Message types: ${JSON.stringify(typeCounts)}`);

    const state = loadSyncState(hookInput.cwd, hookInput.session_id, log);

    const newMessages = formatMessagesForLetta(messages, state.lastProcessedIndex, log);

    if (newMessages.length === 0) {
      log('No new messages to send after formatting');
      process.exit(0);
    }

    // Spawn local Python worker (subconscious)
    const pluginRoot = path.resolve(__dirname, '..');
    const pythonDir = path.join(pluginRoot, 'python');
    if (!fs.existsSync(path.join(pythonDir, 'memory'))) {
      log('ERROR: python/memory/ not found — run install.ps1');
      console.error('Subconscious: Python modules missing. Run the install script.');
      process.exit(1);
    }

    const transcriptXml = formatAsXmlTranscript(newMessages);
    const stateFile = getSyncStateFile(hookInput.cwd, hookInput.session_id);

    const localPayload = {
      sessionId: hookInput.session_id,
      cwd: hookInput.cwd,
      stateFile,
      newLastProcessedIndex: messages.length - 1,
      transcriptXml,
    };

    const payloadFile = path.join(TEMP_STATE_DIR, `local-payload-${hookInput.session_id}-${Date.now()}.json`);
    fs.writeFileSync(payloadFile, JSON.stringify(localPayload), 'utf-8');
    log(`Wrote local payload to ${payloadFile} (${transcriptXml.length} chars XML)`);

    const workerScript = path.join(__dirname, 'local_worker.py');
    const pythonCmd = hermesConfig.pythonPath;

    const workerEnv = { ...process.env, PYTHONPATH: pythonDir };

    const child = spawn(pythonCmd, [workerScript, payloadFile], {
      detached: true,
      stdio: 'ignore',
      cwd: pythonDir,
      env: workerEnv,
      windowsHide: true,
    });
    child.unref();

    log(`Spawned local worker (PID: ${child.pid})`);
    log('Hook completed (local worker running in background)');
    process.exit(0);

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    log(`ERROR: ${errorMessage}`);
    if (error instanceof Error && error.stack) {
      log(`Stack trace: ${error.stack}`);
    }
    console.error(`Error in stop_capture: ${errorMessage}`);
    process.exit(1);
  }
}

main();
