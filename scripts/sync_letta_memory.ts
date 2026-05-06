#!/usr/bin/env tsx
/**
 * UserPromptSubmit hook — local-only.
 *
 * Injects on every prompt:
 *   1. Memory block diff (5 letta-style blocks, now "patterns" layer)
 *   2. L2 retrieval (vector search via memory.query_retrieve)
 *   3. Whispers (one-shot observations from subconscious worker)
 *
 * No Letta cloud calls. No LETTA_API_KEY required.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as readline from 'readline';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import {
  loadSyncState,
  saveSyncState,
  SyncState,
  Agent,
  MemoryBlock,
  escapeXmlContent,
  sanitizeBlockLabel,
  formatAllBlocksForStdout,
  cleanLettaFromClaudeMd,
  getMode,
  getTempStateDir,
} from './conversation_utils.ts';
import { getLocalAgent, consumeWhispers } from './local_store.ts';
import { getConfig } from './config.ts';

const hermesConfig = getConfig();
const DEBUG = hermesConfig.debug;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function debug(...args: unknown[]): void {
  if (DEBUG) {
    console.error('[sync debug]', ...args);
  }
}

interface HookInput {
  session_id: string;
  cwd: string;
  prompt?: string;
  transcript_path?: string;
}

const TEMP_STATE_DIR = getTempStateDir();

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

    setTimeout(() => {
      rl.close();
    }, 1000);
  });
}

function detectChangedBlocks(
  currentBlocks: MemoryBlock[],
  lastBlockValues: { [label: string]: string } | null
): MemoryBlock[] {
  if (!lastBlockValues) {
    return [];
  }
  return currentBlocks.filter(block => {
    const previousValue = lastBlockValues[block.label];
    return previousValue === undefined || previousValue !== block.value;
  });
}

function computeDiff(oldValue: string, newValue: string): { added: string[], removed: string[] } {
  const oldLines = oldValue.split('\n').map(l => l.trim()).filter(l => l);
  const newLines = newValue.split('\n').map(l => l.trim()).filter(l => l);

  const oldSet = new Set(oldLines);
  const newSet = new Set(newLines);

  const added = newLines.filter(line => !oldSet.has(line));
  const removed = oldLines.filter(line => !newSet.has(line));

  return { added, removed };
}

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

    if (previousValue === undefined) {
      const escapedContent = escapeXmlContent(block.value || '');
      return `<${safeLabel} status="new">\n${escapedContent}\n</${safeLabel}>`;
    }

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

  return `<patterns_update>
<!-- Pattern blocks updated since last prompt (showing diff) -->
${formatted}
</patterns_update>`;
}

function _sanitizeCwd(cwd: string): string {
  return cwd.replace(/[\\/:]/g, '-').replace(/^-+/, '');
}

function _markerDir(cwd: string): string {
  return path.join(os.homedir(), '.claude', 'projects', _sanitizeCwd(cwd), 'l1_markers');
}

/**
 * Read and consume L1-evicted marker files. Renames consumed files
 * to .consumed-<ts>.md (don't delete — useful for postmortem).
 */
function consumeL1Markers(cwd: string): string[] {
  const dir = _markerDir(cwd);
  if (!fs.existsSync(dir)) return [];
  const out: string[] = [];
  let entries: string[] = [];
  try {
    entries = fs.readdirSync(dir);
  } catch {
    return [];
  }
  for (const name of entries) {
    if (!name.startsWith('l1_evicted_') || !name.endsWith('.md')) continue;
    const full = path.join(dir, name);
    try {
      const content = fs.readFileSync(full, 'utf-8').trim();
      if (content) out.push(content);
      const consumed = full.replace(/\.md$/, `.consumed-${Date.now()}.md`);
      fs.renameSync(full, consumed);
    } catch (e) {
      debug('marker read failed:', name, e);
    }
  }
  return out;
}

/**
 * Retrieve query-dependent memories from LanceDB vector store.
 * Calls Python subprocess: memory.query_retrieve
 */
function retrieveMemories(prompt: string, cwd: string): string {
  const pythonPath = hermesConfig.pythonPath;
  const pluginRoot = path.resolve(__dirname, '..');
  const pythonDir = path.join(pluginRoot, 'python');

  if (!prompt || prompt.trim().length < 5) {
    debug('retrieveMemories: prompt too short, skipping');
    return '';
  }

  const truncatedPrompt = prompt.slice(0, 8000);

  try {
    const t0 = Date.now();
    const result = spawnSync(
      pythonPath,
      ['-m', 'memory.query_retrieve', truncatedPrompt, '--k', '10', '--format', 'xml'],
      {
        cwd: pythonDir,
        timeout: 15000,
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

async function main(): Promise<void> {
  const mode = getMode();
  if (mode === 'off') {
    process.exit(0);
  }

  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();

  try {
    const hookInput = await readHookInput();
    const cwd = hookInput?.cwd || projectDir;
    const sessionId = hookInput?.session_id;

    let state: SyncState | null = null;
    if (sessionId) {
      state = loadSyncState(cwd, sessionId);
    }

    const lastBlockValues = state?.lastBlockValues || null;

    // Local-only: load pattern blocks from disk
    const agent: Agent = getLocalAgent(cwd);
    debug('Loaded', (agent.blocks || []).length, 'blocks');
    const populated = (agent.blocks || []).filter(b => b.value && b.value.trim());
    debug('Non-empty blocks:', populated.map(b => b.label).join(', ') || 'none');

    const changedBlocks = detectChangedBlocks(agent.blocks || [], lastBlockValues);
    debug('Changed blocks:', changedBlocks.length, changedBlocks.map(b => b.label).join(', '));

    // Legacy CLAUDE.md scrub (one-shot migration; will remove this in ~6 months)
    cleanLettaFromClaudeMd(cwd);

    if (state) {
      state.lastBlockValues = {};
      for (const block of agent.blocks || []) {
        state.lastBlockValues[block.label] = block.value;
      }
    }

    const outputs: string[] = [];

    if (mode === 'full') {
      const isFirstPrompt = !lastBlockValues;
      if (isFirstPrompt) {
        outputs.push(formatAllBlocksForStdout(agent, null));
      } else {
        const changedBlocksOutput = formatChangedBlocksForStdout(changedBlocks, lastBlockValues);
        if (changedBlocksOutput) {
          outputs.push(changedBlocksOutput);
        }
      }
    }

    // L2 vector store retrieval
    if (hookInput?.prompt) {
      const retrievedXml = retrieveMemories(hookInput.prompt, cwd);
      if (retrievedXml) {
        outputs.push(retrievedXml);
      }
    }

    // L1-evicted markers (Phase D) — emitted by l1_watch / precompact_safety
    const markers = consumeL1Markers(cwd);
    if (markers.length > 0) {
      outputs.push(...markers);
      outputs.push(`<instruction>L1 manager evicted ${markers.length} chunk-block(s) above. Use memory_recall(query, scope="l1_evict") if you need details from the evicted content.</instruction>`);
    }

    // Whispers — one-shot observations from subconscious worker
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

    const finalOutput = outputs.join('\n\n');
    debug('Final output length:', finalOutput.length, 'chars');
    console.log(finalOutput);

    if (state && sessionId) {
      saveSyncState(cwd, state);
    }

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`Error syncing memory: ${errorMessage}`);
    process.exit(1);
  }
}

main();
