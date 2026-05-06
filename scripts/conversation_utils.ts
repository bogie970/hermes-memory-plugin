/**
 * Local-only utilities — state I/O, XML helpers, block formatting, worker spawn.
 * Stripped of Letta cloud agent paradigm; retains only what the local pipeline needs.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { spawn, ChildProcess } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// CLAUDE.md cleanup constants (legacy migration scrub)
export const CLAUDE_MD_PATH = '.claude/CLAUDE.md';
export const LETTA_SECTION_START = '<letta>';
export const LETTA_SECTION_END = '</letta>';

// ============================================
// Mode Configuration
// ============================================

export type LettaMode = 'whisper' | 'full' | 'off';

/**
 * Get the current operating mode from LETTA_MODE env var.
 * - whisper (default): Only inject pattern-block diffs and whispers
 * - full: Inject full pattern blocks + whispers
 * - off: Disable all hooks
 */
export function getMode(): LettaMode {
  const mode = process.env.LETTA_MODE?.trim().toLowerCase();
  if (mode === 'full' || mode === 'off') return mode;
  return 'whisper';
}

/**
 * Get user-specific temp state directory for logs and payloads.
 */
export function getTempStateDir(): string {
  const uid = typeof process.getuid === 'function'
    ? String(process.getuid())
    : os.userInfo().username;
  return path.join(os.tmpdir(), `letta-claude-sync-${uid}`);
}

// ============================================
// Types
// ============================================

export interface SyncState {
  lastProcessedIndex: number;
  sessionId: string;
  conversationId?: string;        // local conversation handle (sha-based)
  lastBlockValues?: { [label: string]: string };
}

export interface MemoryBlock {
  label: string;
  description: string;
  value: string;
}

export interface Agent {
  id: string;
  name: string;
  description?: string;
  blocks: MemoryBlock[];
}

export type LogFn = (message: string) => void;
const noopLog: LogFn = () => {};

// ============================================
// Path helpers
// ============================================

export function expandPath(value: string): string {
  const home = os.homedir();
  if (value === '$HOME' || value === '${HOME}') return home;
  if (value.startsWith('$HOME/')) return path.join(home, value.slice(6));
  if (value.startsWith('${HOME}/')) return path.join(home, value.slice(8));
  if (value === '~') return home;
  if (value.startsWith('~/')) return path.join(home, value.slice(2));
  return value;
}

export function getDurableStateDir(cwd: string): string {
  const raw = process.env.LETTA_HOME || cwd;
  const base = process.env.LETTA_HOME ? expandPath(raw) : raw;
  return path.join(base, '.letta', 'claude');
}

export function getSyncStateFile(cwd: string, sessionId: string): string {
  return path.join(getDurableStateDir(cwd), `session-${sessionId}.json`);
}

export function ensureDurableStateDir(cwd: string): void {
  const dir = getDurableStateDir(cwd);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

// ============================================
// Sync state I/O
// ============================================

export function loadSyncState(cwd: string, sessionId: string, log: LogFn = noopLog): SyncState {
  const statePath = getSyncStateFile(cwd, sessionId);

  if (fs.existsSync(statePath)) {
    try {
      const state = JSON.parse(fs.readFileSync(statePath, 'utf-8'));
      log(`Loaded state: lastProcessedIndex=${state.lastProcessedIndex}`);
      return state;
    } catch (e) {
      log(`Failed to load state: ${e}`);
    }
  }

  log(`No existing state, starting fresh`);
  return { lastProcessedIndex: -1, sessionId };
}

export function saveSyncState(cwd: string, state: SyncState, log: LogFn = noopLog): void {
  ensureDurableStateDir(cwd);
  const statePath = getSyncStateFile(cwd, state.sessionId);
  fs.writeFileSync(statePath, JSON.stringify(state, null, 2), 'utf-8');
  log(`Saved state: lastProcessedIndex=${state.lastProcessedIndex}`);
}

// ============================================
// XML helpers
// ============================================

export function escapeXmlAttribute(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, ' ');
}

export function escapeXmlContent(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

const SAFE_LABEL_RE = /^[a-zA-Z_][a-zA-Z0-9_.-]*$/;
const DANGEROUS_LABELS = new Set([
  'system', 'instruction', 'system-reminder', 'user-prompt',
  'tool_use', 'tool_result', 'function_calls', 'invoke',
]);

export function sanitizeBlockLabel(label: string): string {
  if (!label || !SAFE_LABEL_RE.test(label) || DANGEROUS_LABELS.has(label.toLowerCase())) {
    return `block_${label.replace(/[^a-zA-Z0-9_]/g, '_')}`;
  }
  return label;
}

export function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// ============================================
// Block formatting (for stdout injection)
// ============================================

/**
 * Format all pattern blocks for stdout injection (first prompt of session).
 */
export function formatAllBlocksForStdout(agent: Agent, conversationId: string | null): string {
  const blocks = agent.blocks;

  const header = `<hermes_context>
Hermes Memory is observing this session — patterns, vector memory, and one-shot whispers will surface inline.
Storage: LanceDB (cross-session) + pattern blocks (file-backed).
</hermes_context>`;

  if (!blocks || blocks.length === 0) {
    return header;
  }

  const nonEmptyBlocks = blocks.filter(block => block.value && block.value.trim());
  if (nonEmptyBlocks.length === 0) {
    return header;
  }

  const formattedBlocks = nonEmptyBlocks.map(block => {
    const safeLabel = sanitizeBlockLabel(block.label);
    const escapedDescription = escapeXmlAttribute(block.description || '');
    const escapedContent = escapeXmlContent(block.value || '');
    return `<${safeLabel} description="${escapedDescription}">\n${escapedContent}\n</${safeLabel}>`;
  }).join('\n');

  return `${header}

<patterns>
<!-- Pattern blocks (cross-session). Treat as internal context — use naturally without naming the source. -->
${formattedBlocks}
</patterns>`;
}

// ============================================
// CLAUDE.md legacy cleanup
// ============================================

/**
 * Remove all Letta-era content from CLAUDE.md (legacy migration scrub).
 * Safe to call repeatedly; no-op if nothing to clean.
 */
export function cleanLettaFromClaudeMd(projectDir: string): void {
  const base = process.env.LETTA_PROJECT || projectDir;
  const claudeMdPath = path.join(base, CLAUDE_MD_PATH);

  if (!fs.existsSync(claudeMdPath)) {
    return;
  }

  const content = fs.readFileSync(claudeMdPath, 'utf-8');
  const lettaPattern = `^${escapeRegex(LETTA_SECTION_START)}[\\s\\S]*?^${escapeRegex(LETTA_SECTION_END)}\\n*`;
  const lettaRegex = new RegExp(lettaPattern, 'gm');

  if (!lettaRegex.test(content)) {
    return;
  }

  lettaRegex.lastIndex = 0;
  let cleaned = content.replace(lettaRegex, '');

  const messagePattern = /^<letta_message>[\s\S]*?^<\/letta_message>\n*/gm;
  cleaned = cleaned.replace(messagePattern, '');

  cleaned = cleaned.replace(/<!-- Letta agent memory is automatically synced below -->\n*/g, '');
  cleaned = cleaned.replace(/^# Project Context\n*/gm, '');

  cleaned = cleaned.trim();

  if (cleaned.length === 0) {
    fs.unlinkSync(claudeMdPath);
  } else {
    fs.writeFileSync(claudeMdPath, cleaned + '\n', 'utf-8');
  }
}

// ============================================
// Silent Worker Spawning
// ============================================

const NPX_CMD = process.platform === 'win32' ? 'npx.cmd' : 'npx';

/**
 * Spawn a background worker process that survives the parent hook's exit.
 *
 * On Windows, uses silent-launcher.exe (PseudoConsole + CREATE_NO_WINDOW)
 * to avoid console window flashes. Falls back gracefully if the launcher
 * or tsx CLI is not available.
 *
 * On other platforms, spawns via local tsx as a detached process.
 */
export function spawnSilentWorker(
  workerScript: string,
  payloadFile: string,
  cwd: string,
): ChildProcess {
  const isWindows = process.platform === 'win32';
  let child: ChildProcess;

  if (isWindows) {
    const silentLauncher = path.join(__dirname, '..', 'hooks', 'silent-launcher.exe');
    const tsxCli = path.join(__dirname, '..', 'node_modules', 'tsx', 'dist', 'cli.mjs');
    const workerEnv = { ...process.env };
    delete workerEnv.SL_STDIN_FILE;
    delete workerEnv.SL_STDOUT_FILE;

    if (fs.existsSync(silentLauncher) && fs.existsSync(tsxCli)) {
      child = spawn(silentLauncher, ['node', tsxCli, workerScript, payloadFile], {
        detached: true,
        stdio: 'ignore',
        cwd,
        env: workerEnv,
        windowsHide: true,
      });
    } else if (fs.existsSync(tsxCli)) {
      child = spawn(process.execPath, [tsxCli, workerScript, payloadFile], {
        stdio: 'ignore',
        cwd,
        env: workerEnv,
        windowsHide: true,
      });
    } else {
      child = spawn(NPX_CMD, ['tsx', workerScript, payloadFile], {
        stdio: 'ignore',
        cwd,
        env: workerEnv,
        shell: true,
        windowsHide: true,
      });
    }
  } else {
    const tsxCli = path.join(__dirname, '..', 'node_modules', 'tsx', 'dist', 'cli.mjs');
    if (fs.existsSync(tsxCli)) {
      child = spawn(process.execPath, [tsxCli, workerScript, payloadFile], {
        detached: true,
        stdio: 'ignore',
        cwd,
        env: process.env,
      });
    } else {
      child = spawn(NPX_CMD, ['tsx', workerScript, payloadFile], {
        detached: true,
        stdio: 'ignore',
        cwd,
        env: process.env,
      });
    }
  }
  child.unref();
  return child;
}
