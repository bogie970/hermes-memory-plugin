#!/usr/bin/env npx tsx
/**
 * Send Messages to Letta Script
 * 
 * Sends Claude Code conversation messages to a Letta agent.
 * This script is designed to run as a Claude Code Stop hook.
 * 
 * Environment Variables:
 *   LETTA_API_KEY - API key for Letta authentication
 *   LETTA_AGENT_ID - Agent ID to send messages to
 * 
 * Hook Input (via stdin):
 *   - session_id: Current session ID
 *   - transcript_path: Path to conversation JSONL file
 *   - stop_hook_active: Whether stop hook is already active
 * 
 * Exit Codes:
 *   0 - Success
 *   1 - Non-blocking error
 * 
 * Log file: $TMPDIR/letta-claude-sync-$UID/send_messages.log
 */

import * as fs from 'fs';
import * as path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { getAgentId } from './agent_config.ts';
import {
  loadSyncState,
  saveSyncState,
  getOrCreateConversation,
  getSyncStateFile,
  spawnSilentWorker,
  getMode,
  getTempStateDir,
  getSdkToolsMode,
} from './conversation_utils.ts';
import {
  readTranscript,
  formatMessagesForLetta,
  formatAsXmlTranscript,
} from './transcript_utils.ts';
import { isLocalMode } from './local_store.ts';

// ESM-compatible __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const TEMP_STATE_DIR = getTempStateDir();
const LOG_FILE = path.join(TEMP_STATE_DIR, 'send_messages.log');

interface HookInput {
  session_id: string;
  transcript_path: string;
  stop_hook_active?: boolean;
  cwd: string;
  hook_event_name?: string;
}

/**
 * Ensure temp log directory exists
 */
function ensureLogDir(): void {
  if (!fs.existsSync(TEMP_STATE_DIR)) {
    fs.mkdirSync(TEMP_STATE_DIR, { recursive: true });
  }
}

/**
 * Log message to file
 */
function log(message: string): void {
  ensureLogDir();
  const timestamp = new Date().toISOString();
  const logLine = `[${timestamp}] ${message}\n`;
  fs.appendFileSync(LOG_FILE, logLine);
}

/**
 * Read hook input from stdin
 */
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


/**
 * Main function
 */
async function main(): Promise<void> {
  log('='.repeat(60));
  log('send_messages_to_letta.ts started');

  const mode = getMode();
  log(`Mode: ${mode}`);
  if (mode === 'off') {
    log('Mode is off, exiting');
    process.exit(0);
  }
  
  // Get environment variables
  const apiKey = process.env.LETTA_API_KEY;

  log(`LETTA_API_KEY: ${apiKey ? 'set (' + apiKey.substring(0, 10) + '...)' : 'NOT SET'}`);

  try {
    // Read hook input
    log('Reading hook input from stdin...');
    const hookInput = await readHookInput();
    log(`Hook input received:`);
    log(`  session_id: ${hookInput.session_id}`);
    log(`  transcript_path: ${hookInput.transcript_path}`);
    log(`  stop_hook_active: ${hookInput.stop_hook_active}`);
    log(`  hook_event_name: ${hookInput.hook_event_name}`);
    log(`  cwd: ${hookInput.cwd}`);
    
    // Prevent infinite loops if stop hook is already active
    if (hookInput.stop_hook_active) {
      log('Stop hook already active, exiting to prevent loop');
      process.exit(0);
    }

    // Read transcript
    log(`Reading transcript from: ${hookInput.transcript_path}`);
    const messages = await readTranscript(hookInput.transcript_path, log);
    log(`Found ${messages.length} messages in transcript`);

    if (messages.length === 0) {
      log('No messages found, exiting');
      process.exit(0);
    }

    // Log message types found
    const typeCounts: Record<string, number> = {};
    for (const msg of messages) {
      const key = msg.type || msg.role || 'unknown';
      typeCounts[key] = (typeCounts[key] || 0) + 1;
    }
    log(`Message types: ${JSON.stringify(typeCounts)}`);

    // Load sync state (from durable storage)
    const state = loadSyncState(hookInput.cwd, hookInput.session_id, log);

    // Format new messages
    const newMessages = formatMessagesForLetta(messages, state.lastProcessedIndex, log);

    if (newMessages.length === 0) {
      log('No new messages to send after formatting');
      process.exit(0);
    }

    // ── LOCAL MODE: Spawn Python worker ──
    if (isLocalMode()) {
      const hermesRoot = process.env.HERMES_ROOT;
      if (!hermesRoot) {
        log('ERROR: HERMES_ROOT not set — cannot spawn local worker');
        console.error('Subconscious: HERMES_ROOT environment variable not set');
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

      // Spawn the Python bridge script
      const workerScript = path.join(__dirname, 'local_worker.py');
      const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';

      const workerEnv = { ...process.env, HERMES_ROOT: hermesRoot };

      const child = spawn(pythonCmd, [workerScript, payloadFile], {
        detached: true,
        stdio: 'ignore',
        cwd: hookInput.cwd,
        env: workerEnv,
        windowsHide: true,
      });
      child.unref();

      log(`Spawned local worker (PID: ${child.pid})`);
      log('Hook completed (local worker running in background)');
      process.exit(0);
    }

    // ── CLOUD MODE: Original Letta API flow ──
    if (!apiKey) {
      log('No LETTA_API_KEY and not in local mode — exiting');
      process.exit(0);
    }

    const agentId = await getAgentId(apiKey, log);
    log(`Using agent: ${agentId}`);

    // Get or create conversation for this session
    const conversationId = await getOrCreateConversation(apiKey, agentId, hookInput.session_id, hookInput.cwd, state, log);
    log(`Using conversation: ${conversationId}`);

    // Save state now (with conversation ID) so it persists even if worker fails
    saveSyncState(hookInput.cwd, state, log);

    // Build the message payload (same format as sendBatchToConversation)
    const transcriptEntries = newMessages.map(m => {
      const role = m.role === 'user' ? 'user' : m.role === 'assistant' ? 'claude_code' : 'system';
      const escaped = m.text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      return `<message role="${role}">\n${escaped}\n</message>`;
    }).join('\n');

    const userMessage = `<claude_code_session_update>
<session_id>${hookInput.session_id}</session_id>

<transcript>
${transcriptEntries}
</transcript>

<instructions>
You may provide commentary or guidance for Claude Code. Your response will be added to Claude's context window on the next prompt. Use this to:
- Offer observations about the user's work
- Provide reminders or context from your memory
- Suggest approaches or flag potential issues
- Send async messages/guidance to Claude Code

Write your response as if speaking directly to Claude Code.
</instructions>
</claude_code_session_update>`;

    // Send via Letta Code SDK (Sub gets client-side tools)
    const sdkToolsMode = getSdkToolsMode();
    log(`SDK tools mode: ${sdkToolsMode}`);

    const payloadFile = path.join(TEMP_STATE_DIR, `payload-${hookInput.session_id}-${Date.now()}.json`);
    const stateFile = getSyncStateFile(hookInput.cwd, hookInput.session_id);

    const sdkPayload = {
      agentId,
      conversationId,
      sessionId: hookInput.session_id,
      message: userMessage,
      stateFile,
      newLastProcessedIndex: messages.length - 1,
      cwd: hookInput.cwd,
      sdkToolsMode,
    };
    fs.writeFileSync(payloadFile, JSON.stringify(sdkPayload), 'utf-8');
    log(`Wrote SDK payload to ${payloadFile}`);

    const workerScript = path.join(__dirname, 'send_worker_sdk.ts');
    const child = spawnSilentWorker(workerScript, payloadFile, hookInput.cwd);
    log(`Spawned SDK worker (PID: ${child.pid})`);

    log('Hook completed (worker running in background)');

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    log(`ERROR: ${errorMessage}`);
    if (error instanceof Error && error.stack) {
      log(`Stack trace: ${error.stack}`);
    }
    console.error(`Error sending messages to Letta: ${errorMessage}`);
    process.exit(1);
  }
}

// Run main function
main();
