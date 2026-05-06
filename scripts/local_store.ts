/**
 * Local file-based pattern store + whisper queue.
 * Loads/saves the 5 letta-style pattern blocks and consumes one-shot whispers.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { fileURLToPath } from 'url';
import { Agent, MemoryBlock } from './conversation_utils.ts';

const AGENT_NAME = 'Hermes';

function getBlocksFilePath(cwd: string): string {
  const home = process.env.LETTA_HOME
    ? expandHome(process.env.LETTA_HOME)
    : cwd;
  return path.join(home, '.letta', 'claude', 'local_blocks.json');
}

function getSeedBlocksPath(): string {
  const thisDir = path.dirname(fileURLToPath(import.meta.url));
  return path.join(thisDir, '..', 'data', 'local_blocks.json');
}

function expandHome(p: string): string {
  const home = process.env.HOME || process.env.USERPROFILE || '';
  if (p === '~' || p === '$HOME' || p === '${HOME}') return home;
  if (p.startsWith('~/')) return path.join(home, p.slice(2));
  if (p.startsWith('$HOME/')) return path.join(home, p.slice(6));
  if (p.startsWith('${HOME}/')) return path.join(home, p.slice(8));
  return p;
}

interface BlocksFile {
  version: number;
  blocks: Record<string, {
    label: string;
    description: string;
    value: string;
    char_limit: number;
    updated_at: string;
  }>;
}

/**
 * Load memory blocks from local JSON file.
 * If the per-project file doesn't exist, copies from seed template.
 */
export function loadLocalBlocks(cwd: string): MemoryBlock[] {
  const blocksPath = getBlocksFilePath(cwd);

  if (!fs.existsSync(blocksPath)) {
    const seedPath = getSeedBlocksPath();
    if (fs.existsSync(seedPath)) {
      const dir = path.dirname(blocksPath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.copyFileSync(seedPath, blocksPath);
    } else {
      return [];
    }
  }

  try {
    const raw: BlocksFile = JSON.parse(fs.readFileSync(blocksPath, 'utf-8'));
    return Object.values(raw.blocks)
      .filter(b => b.value && b.value.trim().length > 0)
      .map(b => ({
        label: b.label,
        description: b.description,
        value: b.value,
      }));
  } catch {
    return [];
  }
}

/**
 * Returns a fake Agent object backed by local blocks.
 * Drop-in replacement for fetchAgent() in local mode.
 */
export function getLocalAgent(cwd: string): Agent {
  return {
    id: 'local-agent',
    name: AGENT_NAME,
    description: 'Local subconscious (no cloud)',
    blocks: loadLocalBlocks(cwd),
  };
}

/**
 * Deterministic local conversation ID from session ID.
 */
export function getLocalConversationId(sessionId: string): string {
  const hash = crypto.createHash('sha256').update(`local-${sessionId}`).digest('hex').slice(0, 12);
  return `local-conv-${hash}`;
}

interface Whisper {
  id: string;
  text: string;
  timestamp: string;
  priority: string;
}

/**
 * Read and consume whisper messages. Returns whispers and deletes the file.
 * Once-only delivery: reading = consuming.
 */
export function consumeWhispers(cwd: string): Whisper[] {
  const home = process.env.LETTA_HOME
    ? expandHome(process.env.LETTA_HOME)
    : cwd;
  const whispersPath = path.join(home, '.letta', 'claude', 'whispers.json');

  if (!fs.existsSync(whispersPath)) {
    return [];
  }

  // Atomic consume: rename first, then read from the renamed copy.
  // Prevents race where Python writes between our read and unlink.
  const tmpPath = whispersPath + `.consuming-${process.pid}`;
  try {
    fs.renameSync(whispersPath, tmpPath);
  } catch {
    return [];
  }

  try {
    const raw = JSON.parse(fs.readFileSync(tmpPath, 'utf-8'));
    try { fs.unlinkSync(tmpPath); } catch {}
    if (!Array.isArray(raw) || raw.length === 0) {
      return [];
    }
    const validated = raw.filter((w: any) =>
      w && typeof w.text === 'string' && w.text.length > 0 &&
      typeof w.id === 'string' && typeof w.timestamp === 'string'
    );
    return validated.slice(0, 20);
  } catch {
    try { fs.unlinkSync(tmpPath); } catch {}
    return [];
  }
}
