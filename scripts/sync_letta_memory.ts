#!/usr/bin/env tsx
/**
 * Letta Memory Sync Script
 * 
 * Syncs Letta agent memory blocks to the project's CLAUDE.md file.
 * This script is designed to run as a Claude Code UserPromptSubmit hook.
 * 
 * Environment Variables:
 *   LETTA_API_KEY - API key for Letta authentication
 *   LETTA_AGENT_ID - Agent ID to fetch memory blocks from
 *   CLAUDE_PROJECT_DIR - Project directory (set by Claude Code)
 *   LETTA_DEBUG - Set to "1" to enable debug logging to stderr
 * 
 * Exit Codes:
 *   0 - Success
 *   1 - Non-blocking error (logged to stderr)
 *   2 - Blocking error (prevents prompt processing)
 */

import * as fs from 'fs';
import * as path from 'path';
import * as readline from 'readline';
import { spawnSync } from 'child_process';
import { getAgentId } from './agent_config.ts';
import { buildLettaApiUrl } from './letta_api_url.ts';
import {
  loadSyncState,
  saveSyncState,
  getOrCreateConversation,
  lookupConversation,
  SyncState,
  Agent,
  MemoryBlock,
  fetchAgent,
  escapeXmlContent,
  sanitizeBlockLabel,
  formatAllBlocksForStdout,
  cleanLettaFromClaudeMd,
  getMode,
  getTempStateDir,
} from './conversation_utils.ts';
import { isLocalMode, getLocalAgent, consumeWhispers } from './local_store.ts';
import { checkCompactionRisk, formatCompactionWarning } from './compaction_guard.ts';
import { getConfig } from './config.ts';

// Configuration — loaded from hermes.config.json, env vars, or defaults
const hermesConfig = getConfig();
const DEBUG = hermesConfig.debug;

function debug(...args: unknown[]): void {
  if (DEBUG) {
    console.error('[sync debug]', ...args);
  }
}

interface LettaMessage {
  id: string;
  message_type: string;
  content?: string;
  text?: string;
  date?: string;
}

interface MessageInfo {
  id: string;
  text: string;
  date: string | null;
}

interface HookInput {
  session_id: string;
  cwd: string;
  prompt?: string;  // User's prompt text (available on UserPromptSubmit)
  transcript_path?: string;  // Path to transcript JSONL
}

// Temp state directory for logs
const TEMP_STATE_DIR = getTempStateDir();

/**
 * Read hook input from stdin
 */
async function readHookInput(): Promise<HookInput | null> {
  return new Promise((resolve) => {
    let input = '';
    const rl = readline.createInterface({ input: process.stdin });
    
    rl.on('line', (line) => {
      input += line;
    });
    
    rl.on('close', () => {
      if (!input.trim()) {
        resolve(null);
        return;
      }
      try {
        resolve(JSON.parse(input));
      } catch {
        resolve(null);
      }
    });

    // Timeout after 1000ms if no input (100ms was too aggressive on loaded machines)
    setTimeout(() => {
      rl.close();
    }, 1000);
  });
}

/**
 * Detect which blocks have changed since last sync
 */
function detectChangedBlocks(
  currentBlocks: MemoryBlock[],
  lastBlockValues: { [label: string]: string } | null
): MemoryBlock[] {
  // First sync - no previous state, don't show all blocks as "changed"
  if (!lastBlockValues) {
    return [];
  }
  
  return currentBlocks.filter(block => {
    const previousValue = lastBlockValues[block.label];
    // Changed if: new block (not in previous) or value differs
    return previousValue === undefined || previousValue !== block.value;
  });
}

/**
 * Compute a simple line-based diff between two strings
 */
function computeDiff(oldValue: string, newValue: string): { added: string[], removed: string[] } {
  const oldLines = oldValue.split('\n').map(l => l.trim()).filter(l => l);
  const newLines = newValue.split('\n').map(l => l.trim()).filter(l => l);
  
  const oldSet = new Set(oldLines);
  const newSet = new Set(newLines);
  
  const added = newLines.filter(line => !oldSet.has(line));
  const removed = oldLines.filter(line => !newSet.has(line));
  
  return { added, removed };
}

/**
 * Format changed blocks for stdout injection with diffs
 */
function formatChangedBlocksForStdout(
  changedBlocks: MemoryBlock[],
  lastBlockValues: { [label: string]: string } | null
): string {
  if (changedBlocks.length === 0) {
    return '';
  }
  
  const nonEmptyChanges = changedBlocks.filter(block => block.value && block.value.trim());
  if (nonEmptyChanges.length === 0) {
    return '';
  }

  const formatted = nonEmptyChanges.map(block => {
    const safeLabel = sanitizeBlockLabel(block.label);
    const previousValue = lastBlockValues?.[block.label];

    // New block - show full content
    if (previousValue === undefined) {
      const escapedContent = escapeXmlContent(block.value || '');
      return `<${safeLabel} status="new">\n${escapedContent}\n</${safeLabel}>`;
    }

    // Existing block - show diff
    const diff = computeDiff(previousValue, block.value || '');

    if (diff.added.length === 0 && diff.removed.length === 0) {
      const escapedContent = escapeXmlContent(block.value || '');
      return `<${safeLabel} status="modified">\n${escapedContent}\n</${safeLabel}>`;
    }

    const diffLines: string[] = [];
    for (const line of diff.removed) {
      diffLines.push(`- ${escapeXmlContent(line)}`);
    }
    for (const line of diff.added) {
      diffLines.push(`+ ${escapeXmlContent(line)}`);
    }

    return `<${safeLabel} status="modified">\n${diffLines.join('\n')}\n</${safeLabel}>`;
  }).join('\n');
  
  return `<letta_memory_update>
<!-- Memory blocks updated since last prompt (showing diff) -->
${formatted}
</letta_memory_update>`;
}

/**
 * Fetch all assistant messages from the conversation history since last seen
 */
async function fetchAssistantMessages(
  apiKey: string, 
  conversationId: string | null,
  lastSeenMessageId: string | null
): Promise<{ messages: MessageInfo[], lastMessageId: string | null }> {
  if (!conversationId) {
    // No conversation yet, return empty
    return { messages: [], lastMessageId: null };
  }

  // Use a high limit because Letta returns multiple entries per logical message
  // (hidden_reasoning + assistant_message pairs), so limit=50 may not reach newest messages
  const url = buildLettaApiUrl(`/conversations/${conversationId}/messages`, {
    limit: 300,
  });

  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    // Don't fail if we can't fetch messages, just return empty
    return { messages: [], lastMessageId: lastSeenMessageId };
  }

  const allMessages: LettaMessage[] = await response.json();

  // Filter to assistant messages only, then sort by date descending (newest first)
  // The API does NOT guarantee newest-first ordering — newer messages can appear at the end
  const assistantMessages = allMessages
    .filter(msg => msg.message_type === 'assistant_message')
    .sort((a, b) => {
      const da = a.date ? new Date(a.date).getTime() : 0;
      const db = b.date ? new Date(b.date).getTime() : 0;
      return db - da; // newest first
    });

  // Find the index of the last seen message
  // Since messages are newest-first, new messages are BEFORE lastSeenIndex (indices 0 to lastSeenIndex-1)
  let endIndex = assistantMessages.length; // Default: return all messages
  if (lastSeenMessageId) {
    const lastSeenIndex = assistantMessages.findIndex(msg => msg.id === lastSeenMessageId);
    if (lastSeenIndex !== -1) {
      // Only return messages newer than the last seen one (before it in the array)
      endIndex = lastSeenIndex;
    }
  }
  debug(`endIndex=${endIndex}, will return messages from index 0 to ${endIndex - 1}`);

  // Get new messages (from 0 to endIndex, which are the newest messages)
  const newMessages: MessageInfo[] = [];
  for (let i = 0; i < endIndex; i++) {
    const msg = assistantMessages[i];
    const text = msg.content || msg.text;
    if (text && typeof text === 'string') {
      newMessages.push({
        id: msg.id,
        text,
        date: msg.date || null,
      });
    }
  }
  debug(`Returning ${newMessages.length} new messages`);

  // Get the last message ID for tracking (the NEWEST message, which is first in the array)
  const lastMessageId = assistantMessages.length > 0
    ? assistantMessages[0].id
    : lastSeenMessageId;
  debug(`Setting lastMessageId=${lastMessageId}`);

  return { messages: newMessages, lastMessageId };
}

/**
 * Format assistant messages for stdout injection
 */
function formatMessagesForStdout(agent: Agent, messages: MessageInfo[]): string {
  if (messages.length === 0) {
    return '';
  }

  const agentName = escapeXmlContent(agent.name || 'Letta Agent');
  const formattedMessages = messages.map((msg, index) => {
    const timestamp = escapeXmlContent(msg.date || 'unknown');
    const msgNum = messages.length > 1 ? ` msg="${index + 1}/${messages.length}"` : '';
    const escapedText = escapeXmlContent(msg.text);
    return `<letta_message from="${agentName}"${msgNum} timestamp="${timestamp}">
${escapedText}
</letta_message>`;
  });

  return formattedMessages.join('\n\n');
}

/**
 * Retrieve query-dependent memories from LanceDB vector store.
 * Calls Python subprocess: memory.query_retrieve
 *
 * Returns XML string with retrieved memories, or empty string on failure.
 * Graceful degradation: any error returns '' so the existing system continues.
 */
function retrieveMemories(prompt: string, cwd: string): string {
  const pythonPath = hermesConfig.pythonPath;
  const pluginRoot = path.resolve(__dirname, '..');
  const pythonDir = path.join(pluginRoot, 'python');

  // Skip trivially short prompts (single words, "y", "ok", etc.)
  if (!prompt || prompt.trim().length < 5) {
    debug('retrieveMemories: prompt too short, skipping');
    return '';
  }

  // Truncate prompt to avoid Windows argv length limits (32k) and
  // because embedding models only use first ~512 tokens anyway.
  const truncatedPrompt = prompt.slice(0, 8000);

  try {
    const t0 = Date.now();
    const result = spawnSync(
      pythonPath,
      ['-m', 'memory.query_retrieve', truncatedPrompt, '--k', '10', '--format', 'xml'],
      {
        cwd: pythonDir,
        timeout: 15000,  // 15s timeout: cold start loads embedding model (~3-9s)
        encoding: 'utf-8',
        env: {
          ...process.env,
          PYTHONPATH: pythonDir,
        },
        windowsHide: true,
      }
    );
    const elapsed = Date.now() - t0;

    if (result.stderr) {
      debug(`retrieveMemories stderr (${elapsed}ms):`, result.stderr.trim());
    }

    if (result.status !== 0) {
      debug(`retrieveMemories: python exited with code ${result.status}`);
      return '';
    }

    const output = (result.stdout || '').trim();
    if (!output || output === '<retrieved_memories count="0"/>') {
      debug(`retrieveMemories: no results (${elapsed}ms)`);
      return '';
    }

    debug(`retrieveMemories: got results (${elapsed}ms), ${output.length} chars`);
    return output;
  } catch (err) {
    debug('retrieveMemories error:', err);
    return '';
  }
}

/**
 * Main function
 */
async function main(): Promise<void> {
  // Check mode
  const mode = getMode();
  if (mode === 'off') {
    process.exit(0);
  }

  const apiKey = process.env.LETTA_API_KEY;
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();

  try {
    const hookInput = await readHookInput();
    const cwd = hookInput?.cwd || projectDir;
    const sessionId = hookInput?.session_id;

    let state: SyncState | null = null;
    if (sessionId) {
      state = loadSyncState(cwd, sessionId);
    }

    let conversationId = state?.conversationId || null;
    if (!conversationId && sessionId) {
      conversationId = lookupConversation(cwd, sessionId);
      if (conversationId && state) {
        state.conversationId = conversationId;
      }
    }
    const lastBlockValues = state?.lastBlockValues || null;
    const lastSeenMessageId = state?.lastSeenMessageId || null;

    let agent: Agent;
    let newMessages: MessageInfo[] = [];
    let lastMessageId: string | null = lastSeenMessageId;

    if (isLocalMode()) {
      agent = getLocalAgent(cwd);
      debug('Local mode — loaded', (agent.blocks || []).length, 'blocks');
      const populated = (agent.blocks || []).filter(b => b.value && b.value.trim());
      debug('Non-empty blocks:', populated.map(b => b.label).join(', ') || 'none');
    } else {
      if (!apiKey) {
        console.error('Error: LETTA_API_KEY environment variable is not set');
        process.exit(1);
      }
      const agentId = await getAgentId(apiKey);
      const [fetchedAgent, messagesResult] = await Promise.all([
        fetchAgent(apiKey, agentId),
        fetchAssistantMessages(apiKey, conversationId, lastSeenMessageId),
      ]);
      agent = fetchedAgent;
      newMessages = messagesResult.messages;
      lastMessageId = messagesResult.lastMessageId;
    }

    // Detect which blocks have changed since last sync
    const changedBlocks = detectChangedBlocks(agent.blocks || [], lastBlockValues);
    debug('Changed blocks:', changedBlocks.length, changedBlocks.map(b => b.label).join(', '));

    // Clean up any existing <letta> section from CLAUDE.md (legacy migration)
    cleanLettaFromClaudeMd(cwd);
    
    // Update state with block values and last seen message ID
    if (state) {
      state.lastBlockValues = {};
      for (const block of agent.blocks || []) {
        state.lastBlockValues[block.label] = block.value;
      }
      // Track the last message we've seen
      if (lastMessageId) {
        state.lastSeenMessageId = lastMessageId;
      }
    }
    
    // Output to stdout - this gets injected before the user's prompt
    // (UserPromptSubmit hooks add stdout to context)
    const outputs: string[] = [];
    
    if (mode === 'full') {
      // Full mode: inject memory blocks + messages
      const isFirstPrompt = !lastBlockValues;
      
      if (isFirstPrompt) {
        outputs.push(formatAllBlocksForStdout(agent, conversationId));
      } else {
        const changedBlocksOutput = formatChangedBlocksForStdout(changedBlocks, lastBlockValues);
        if (changedBlocksOutput) {
          outputs.push(changedBlocksOutput);
        }
      }
    }
    
    // Query-dependent retrieval: search L2 vector store for relevant memories
    if (hookInput?.prompt) {
      const retrievedXml = retrieveMemories(hookInput.prompt, cwd);
      if (retrievedXml) {
        outputs.push(retrievedXml);
      }
    }

    // Both modes: inject messages from Sub
    const messageOutput = formatMessagesForStdout(agent, newMessages);
    outputs.push(messageOutput);

    // Add instruction to acknowledge messages if there are any
    if (newMessages.length > 0) {
      const agentName = agent.name || 'Subconscious';
      outputs.push(`<instruction>Your Subconscious (${agentName}) sent you a message above. Briefly acknowledge what ${agentName} said - just a short note like "Sub notes: [key point]" so the user knows.</instruction>`);
    }
    
    // Inject whispers (one-shot messages from background worker)
    if (isLocalMode()) {
      const whispers = consumeWhispers(cwd);
      debug('Whispers consumed:', whispers.length);
      if (whispers.length > 0) {
        const formatted = whispers.map(w => {
          const escapedText = escapeXmlContent(w.text);
          const escapedId = escapeXmlContent(w.id);
          const escapedTs = escapeXmlContent(w.timestamp);
          return `<subconscious_whisper id="${escapedId}" timestamp="${escapedTs}">\n${escapedText}\n</subconscious_whisper>`;
        }).join('\n');
        outputs.push(formatted);
        const wCount = whispers.length === 1 ? '1 whisper' : `${whispers.length} whispers`;
        outputs.push(`<instruction>Your Subconscious sent ${wCount} above. These are one-time observations — acknowledge briefly inline, e.g. "Sub whispers: [key point]".</instruction>`);
      }
    }

    // Compaction guard: check if background worker is falling behind
    if (sessionId && hookInput?.transcript_path) {
      const compactionStatus = checkCompactionRisk(cwd, sessionId, hookInput.transcript_path, debug);
      const warning = formatCompactionWarning(compactionStatus);
      if (warning) {
        outputs.push(warning);
      }
    }

    const finalOutput = outputs.join('\n\n');
    debug('Final output length:', finalOutput.length, 'chars');
    console.log(finalOutput);

    // Save state
    if (state && sessionId) {
      saveSyncState(cwd, state);
    }
    
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`Error syncing Letta memory: ${errorMessage}`);
    // Exit with code 1 for non-blocking error
    // Change to exit(2) if you want to block prompt processing on sync failures
    process.exit(1);
  }
}

// Run main function
main();
