#!/usr/bin/env npx tsx
/**
 * Session Start Hook Script
 *
 * Notifies Letta agent when a new Claude Code session begins.
 * This script is designed to run as a Claude Code SessionStart hook.
 *
 * Environment Variables:
 *   LETTA_API_KEY - API key for Letta authentication
 *   LETTA_AGENT_ID - Agent ID to send messages to
 *
 * Hook Input (via stdin):
 *   - session_id: Current session ID
 *   - cwd: Current working directory
 *   - hook_event_name: "SessionStart"
 *
 * Exit Codes:
 *   0 - Success
 *   1 - Non-blocking error
 *
 * Log file: $TMPDIR/letta-claude-sync-$UID/session_start.log
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { getAgentId } from './agent_config.js';
import {
  cleanLettaFromClaudeMd,
  createConversation,
  fetchAgent,
  getMode,
  getTempStateDir,
  getSdkToolsMode,
  expandPath,
} from './conversation_utils.js';
import { buildLettaApiUrl } from './letta_api_url.js';
import { isLocalMode, getLocalAgent, getLocalConversationId } from './local_store.js';

// Configuration
const TEMP_STATE_DIR = getTempStateDir();
const LOG_FILE = path.join(TEMP_STATE_DIR, 'session_start.log');

interface HookInput {
  session_id: string;
  cwd: string;
  hook_event_name?: string;
}

interface ConversationEntry {
  conversationId: string;
  agentId: string;
}

// Support both old format (string) and new format (object) for backward compatibility
interface ConversationsMap {
  [sessionId: string]: string | ConversationEntry;
}

interface Conversation {
  id: string;
  agent_id: string;
  created_at?: string;
}

// Durable storage in .letta directory
// If LETTA_HOME is set, use that as the base instead of cwd
function getDurableStateDir(cwd: string): string {
  const raw = process.env.LETTA_HOME || cwd;
  const base = process.env.LETTA_HOME ? expandPath(raw) : raw;
  return path.join(base, '.letta', 'claude');
}

function getConversationsFile(cwd: string): string {
  return path.join(getDurableStateDir(cwd), 'conversations.json');
}

function getSyncStateFile(cwd: string, sessionId: string): string {
  return path.join(getDurableStateDir(cwd), `session-${sessionId}.json`);
}

/**
 * Ensure directories exist
 */
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
 * Load conversations mapping
 */
function loadConversationsMap(cwd: string): ConversationsMap {
  const filePath = getConversationsFile(cwd);
  if (fs.existsSync(filePath)) {
    try {
      return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    } catch (e) {
      log(`Failed to load conversations map: ${e}`);
    }
  }
  return {};
}

/**
 * Save conversations mapping
 */
function saveConversationsMap(cwd: string, map: ConversationsMap): void {
  ensureDurableStateDir(cwd);
  fs.writeFileSync(getConversationsFile(cwd), JSON.stringify(map, null, 2), 'utf-8');
}

/**
 * Save session state
 */
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

/**
 * Send session start message to Letta
 */
async function sendSessionStartMessage(
  apiKey: string,
  conversationId: string,
  sessionId: string,
  cwd: string
): Promise<void> {
  const url = buildLettaApiUrl(`/conversations/${conversationId}/messages`);

  const projectName = path.basename(cwd);
  const timestamp = new Date().toISOString();

  const sdkToolsMode = getSdkToolsMode();
  const toolAccessDescription = sdkToolsMode === 'full'
    ? 'Full tool access enabled — you can Read, Grep, Glob, Edit, Write, Bash, and search the web.'
    : sdkToolsMode === 'read-only'
    ? 'Read-only tool access — you can Read, Grep, Glob files and search the web. No writes.'
    : 'Listen-only mode — no client-side tools. You can only update your memory blocks.';

  const message = `<claude_code_session_start>
<project>${projectName}</project>
<path>${cwd}</path>
<session_id>${sessionId}</session_id>
<timestamp>${timestamp}</timestamp>
<sdk_tools_mode>${sdkToolsMode}</sdk_tools_mode>

<context>
A new Claude Code session has begun. I'll be sending you updates as the session progresses.

Tool access: ${toolAccessDescription}
${sdkToolsMode !== 'off' ? `Use your tools to explore the codebase at ${cwd} when processing transcripts.` : ''}
</context>
</claude_code_session_start>`;

  log(`Sending session start message to conversation ${conversationId}`);

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      messages: [{ role: 'user', content: message }],
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to send message: ${response.status} ${errorText}`);
  }

  // Consume stream minimally
  const reader = response.body?.getReader();
  if (reader) {
    try {
      await reader.read();
    } finally {
      reader.cancel();
    }
  }

  log(`Session start message sent successfully`);
}

/**
 * Main function
 */
async function main(): Promise<void> {
  log('='.repeat(60));
  log('session_start.ts started');

  const mode = getMode();
  log(`Mode: ${mode}`);
  if (mode === 'off') {
    log('Mode is off, exiting');
    process.exit(0);
  }

  const apiKey = process.env.LETTA_API_KEY;

  // Try to open TTY for user-visible output (bypasses Claude's capture)
  // Skip on Windows — /dev/tty resolves to C:\dev\tty which doesn't exist
  let tty: fs.WriteStream | null = null;
  if (process.platform !== 'win32') {
    try {
      tty = fs.createWriteStream('/dev/tty');
      tty.on('error', () => { tty = null; });
    } catch {
      // TTY not available
    }
  }

  const writeTty = (text: string) => {
    if (tty) tty.write(text);
  };

  try {
    // Read hook input first (needed by both modes)
    log('Reading hook input from stdin...');
    const hookInput = await readHookInput();
    log(`Hook input: session_id=${hookInput.session_id}, cwd=${hookInput.cwd}`);

    if (isLocalMode()) {
      // ── LOCAL MODE ──
      log('Local mode (no LETTA_API_KEY) — using local file storage');

      writeTty('\n');
      writeTty('\x1b[1m  Claude Subconscious\x1b[0m \x1b[2m(local)\x1b[0m\n');
      writeTty('\x1b[35m');
      writeTty('  ▐\x1b[31m▛\x1b[35m███\x1b[31m▜\x1b[35m▌\n');
      writeTty(' ▝▜█████▛▘\n');
      writeTty('   ▘▘ ▝▝\n');
      writeTty('\x1b[0m');
      writeTty('\x1b[2m');
      writeTty(`  Mode:    ${mode}\n`);
      writeTty('  Storage: local blocks (file-backed)\n');
      writeTty('\x1b[0m\n');
      if (tty) tty.end();

      const conversationId = getLocalConversationId(hookInput.session_id);
      saveSessionState(hookInput.cwd, hookInput.session_id, conversationId);

      cleanLettaFromClaudeMd(hookInput.cwd);
      const homeDir = process.env.HOME || os.homedir();
      if (homeDir !== hookInput.cwd) {
        cleanLettaFromClaudeMd(homeDir);
      }

      log('Local mode session_start completed');

    } else {
      // ── CLOUD MODE (original Letta API flow) ──
      writeTty('\n');
      writeTty('\x1b[1m  Claude Subconscious\x1b[0m\n');
      writeTty('\n');
      writeTty('\x1b[35m');
      writeTty('  ▐\x1b[31m▛\x1b[35m███\x1b[31m▜\x1b[35m▌\n');
      writeTty(' ▝▜█████▛▘\n');
      writeTty('   ▘▘ ▝▝\n');
      writeTty('\x1b[0m');
      writeTty('\x1b[2m  Connecting...\x1b[0m');

      const agentId = await getAgentId(apiKey!, log);
      const agent = await fetchAgent(apiKey!, agentId);
      const agentName = agent.name || 'Unnamed Agent';
      const modelHandle = (agent as any).llm_config?.handle || (agent as any).llm_config?.model || 'unknown';

      writeTty('\r\x1b[K');
      writeTty('\n  Agent information:\n');
      writeTty('\x1b[1m');
      writeTty(`  ${agentName}\n`);
      writeTty('\x1b[0m');
      writeTty('\x1b[2m');
      writeTty(`  ${agentId}\n`);
      writeTty('\n');

      const sdkTools = process.env.LETTA_SDK_TOOLS || 'read-only';
      const baseUrl = process.env.LETTA_BASE_URL || 'https://api.letta.com';
      writeTty(`  Model:      ${modelHandle}\n`);
      writeTty(`  Mode:       ${mode}\n`);
      writeTty(`  SDK Tools:  ${sdkTools}\n`);
      if (process.env.LETTA_BASE_URL) {
        writeTty(`  Server:     ${baseUrl}\n`);
      }
      if (process.env.LETTA_HOME) {
        writeTty(`  Home:       ${expandPath(process.env.LETTA_HOME)}\n`);
      }
      writeTty('\n');
      writeTty('  Learn about configuration settings:\n');
      writeTty('  github.com/letta-ai/claude-subconscious\n');
      writeTty('\x1b[0m');
      writeTty('\n');

      const conversationsMap = loadConversationsMap(hookInput.cwd);
      let conversationId: string;
      const cached = conversationsMap[hookInput.session_id];

      if (cached) {
        const entry = typeof cached === 'string'
          ? { conversationId: cached, agentId: null as string | null }
          : cached;

        if (entry.agentId && entry.agentId !== agentId) {
          log(`Agent ID changed (${entry.agentId} -> ${agentId}), clearing stale conversation`);
          delete conversationsMap[hookInput.session_id];
          conversationId = await createConversation(apiKey!, agentId, log);
          conversationsMap[hookInput.session_id] = { conversationId, agentId };
          saveConversationsMap(hookInput.cwd, conversationsMap);
        } else if (!entry.agentId) {
          log(`Upgrading old format entry, creating new conversation`);
          delete conversationsMap[hookInput.session_id];
          conversationId = await createConversation(apiKey!, agentId, log);
          conversationsMap[hookInput.session_id] = { conversationId, agentId };
          saveConversationsMap(hookInput.cwd, conversationsMap);
        } else {
          conversationId = entry.conversationId;
          log(`Reusing existing conversation: ${conversationId}`);
        }
      } else {
        conversationId = await createConversation(apiKey!, agentId, log);
        conversationsMap[hookInput.session_id] = { conversationId, agentId };
        saveConversationsMap(hookInput.cwd, conversationsMap);
      }

      saveSessionState(hookInput.cwd, hookInput.session_id, conversationId);

      log('Cleaning up any legacy CLAUDE.md content...');
      cleanLettaFromClaudeMd(hookInput.cwd);
      const homeDir = process.env.HOME || os.homedir();
      if (homeDir !== hookInput.cwd) {
        cleanLettaFromClaudeMd(homeDir);
      }
      log('CLAUDE.md cleanup done');

      const isHosted = !process.env.LETTA_BASE_URL;
      if (isHosted) {
        const convUrl = `https://app.letta.com/agents/${agentId}?conversation=${conversationId}`;
        writeTty('\x1b[2m');
        writeTty('  View the subconscious agent:\n');
        writeTty(`  ${convUrl}\n`);
        writeTty('\x1b[0m');
        writeTty('\n');
      }

      writeTty('\x1b[2m');
      writeTty('  Come talk to us on Discord:\n');
      writeTty('  https://discord.gg/letta\n');
      writeTty('\x1b[0m');
      writeTty('\n');

      if (tty) tty.end();

      await sendSessionStartMessage(apiKey!, conversationId, hookInput.session_id, hookInput.cwd);
      log('Completed successfully');
    }

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    log(`ERROR: ${errorMessage}`);

    writeTty('\r\x1b[K');
    writeTty('\x1b[31m');
    writeTty(`  Subconscious error: ${errorMessage}\n`);
    writeTty('\x1b[0m');
    if (tty) tty.end();

    process.exit(1);
  }
}

main();
