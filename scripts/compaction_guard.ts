/**
 * Compaction Guard
 *
 * Detects when Claude Code is approaching compaction and ensures
 * unprocessed conversation messages are backed up before context
 * is lost. Integrates with sync_letta_memory.ts (UserPromptSubmit hook).
 */

import * as fs from 'fs';
import * as path from 'path';
import {
  loadSyncState,
  getDurableStateDir,
  ensureDurableStateDir,
  LogFn,
} from './conversation_utils.ts';

export interface WorkerHealth {
  alive: boolean;
  lastRun: string | null;
  error: string | null;
}

export interface CompactionStatus {
  safe: boolean;
  messagesBehind: number;
  totalMessages: number;
  lastProcessed: number;
  backupPath?: string;
  workerHealth: WorkerHealth;
  action: 'none' | 'warn' | 'panic_extract' | 'worker_error';
}

/**
 * Count valid JSONL lines (parseable JSON only) to match worker's counting.
 * Loads the full file — acceptable for transcript sizes seen in practice.
 */
function countTranscriptMessages(transcriptPath: string): number {
  if (!fs.existsSync(transcriptPath)) return 0;
  const content = fs.readFileSync(transcriptPath, 'utf-8');
  let count = 0;
  for (const line of content.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      JSON.parse(trimmed);
      count++;
    } catch {
      // Skip malformed lines — matches worker behavior
    }
  }
  return count;
}

/**
 * Check worker health from worker_status.json.
 */
function checkWorkerHealth(cwd: string): WorkerHealth {
  const stateDir = getDurableStateDir(cwd);
  const statusPath = path.join(stateDir, 'worker_status.json');

  if (!fs.existsSync(statusPath)) {
    return { alive: true, lastRun: null, error: null };
  }

  try {
    const status = JSON.parse(fs.readFileSync(statusPath, 'utf-8'));
    return {
      alive: status.success !== false || !status.error,
      lastRun: status.last_run || null,
      error: status.error || null,
    };
  } catch {
    return { alive: true, lastRun: null, error: null };
  }
}

/**
 * Check if a recent backup already exists for this lastProcessedIndex.
 */
function hasRecentBackup(cwd: string, lastProcessedIndex: number): string | null {
  const stateDir = getDurableStateDir(cwd);
  if (!fs.existsSync(stateDir)) return null;

  const marker = path.join(stateDir, 'last_backup_index.txt');
  if (fs.existsSync(marker)) {
    try {
      const saved = parseInt(fs.readFileSync(marker, 'utf-8').trim(), 10);
      if (saved === lastProcessedIndex) {
        // Find the most recent backup file
        const files = fs.readdirSync(stateDir)
          .filter(f => f.startsWith('unprocessed_backup_'))
          .sort()
          .reverse();
        if (files.length > 0) return path.join(stateDir, files[0]);
      }
    } catch { /* ignore */ }
  }
  return null;
}

/**
 * Check how far behind the background worker is and decide what action to take.
 */
export function checkCompactionRisk(
  cwd: string,
  sessionId: string,
  transcriptPath: string,
  log: LogFn = () => {},
): CompactionStatus {
  const state = loadSyncState(cwd, sessionId, log);
  const totalMessages = countTranscriptMessages(transcriptPath);
  const lastProcessed = state.lastProcessedIndex;
  const messagesBehind = Math.max(0, totalMessages - 1 - lastProcessed);
  const workerHealth = checkWorkerHealth(cwd);

  log(`Compaction guard: total=${totalMessages}, lastProcessed=${lastProcessed}, behind=${messagesBehind}`);

  // Check worker health first
  if (workerHealth.error && !workerHealth.alive) {
    return {
      safe: false,
      messagesBehind,
      totalMessages,
      lastProcessed,
      workerHealth,
      action: 'worker_error',
    };
  }

  if (messagesBehind > 20) {
    // Avoid repeated panic extracts for the same state
    const existing = hasRecentBackup(cwd, lastProcessed);
    if (existing) {
      log(`Panic extract skipped — recent backup exists: ${existing}`);
      return {
        safe: false,
        messagesBehind,
        totalMessages,
        lastProcessed,
        backupPath: existing,
        workerHealth,
        action: 'panic_extract',
      };
    }

    const backupPath = panicExtract(cwd, transcriptPath, lastProcessed, log);
    return {
      safe: false,
      messagesBehind,
      totalMessages,
      lastProcessed,
      backupPath,
      workerHealth,
      action: 'panic_extract',
    };
  }

  if (messagesBehind >= 5) {
    return {
      safe: false,
      messagesBehind,
      totalMessages,
      lastProcessed,
      workerHealth,
      action: 'warn',
    };
  }

  return {
    safe: true,
    messagesBehind,
    totalMessages,
    lastProcessed,
    workerHealth,
    action: 'none',
  };
}

/**
 * Emergency backup: write all unprocessed transcript messages to a JSONL file.
 * Returns the path to the backup file.
 */
export function panicExtract(
  cwd: string,
  transcriptPath: string,
  lastProcessedIndex: number,
  log: LogFn = () => {},
): string {
  ensureDurableStateDir(cwd);
  const stateDir = getDurableStateDir(cwd);
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const backupPath = path.join(stateDir, `unprocessed_backup_${timestamp}.jsonl`);

  if (!fs.existsSync(transcriptPath)) {
    log(`Panic extract: transcript not found at ${transcriptPath}`);
    fs.writeFileSync(backupPath, '', 'utf-8');
    return backupPath;
  }

  const content = fs.readFileSync(transcriptPath, 'utf-8');
  const lines = content.split('\n');
  const unprocessed: string[] = [];
  let lineIndex = 0;

  for (const line of lines) {
    if (!line.trim()) continue;
    try {
      JSON.parse(line.trim());
    } catch {
      continue; // Skip malformed — matches counting logic
    }
    if (lineIndex > lastProcessedIndex) {
      unprocessed.push(line.trim());
    }
    lineIndex++;
  }

  fs.writeFileSync(backupPath, unprocessed.join('\n') + '\n', 'utf-8');
  // Record which index we backed up to avoid repeats
  fs.writeFileSync(path.join(stateDir, 'last_backup_index.txt'), String(lastProcessedIndex), 'utf-8');
  log(`Panic extract: backed up ${unprocessed.length} messages to ${backupPath}`);
  return backupPath;
}

/**
 * Format a compaction warning for stdout injection.
 * Returns empty string if no warning needed.
 */
export function formatCompactionWarning(status: CompactionStatus): string {
  if (status.action === 'none') return '';

  if (status.action === 'worker_error') {
    return `<compaction_warning level="critical">
Memory system: background worker has failed. Error: ${status.workerHealth.error || 'unknown'}. Last run: ${status.workerHealth.lastRun || 'never'}. ${status.messagesBehind} messages unprocessed. The memory system may need attention.
</compaction_warning>`;
  }

  if (status.action === 'panic_extract') {
    return `<compaction_warning level="critical">
Memory system: compaction risk — ${status.messagesBehind} messages not yet processed by background worker. Emergency backup saved to ${status.backupPath}. Consider waiting before continuing if possible.
</compaction_warning>`;
  }

  if (status.action === 'warn') {
    return `<compaction_warning level="warning">
Memory system: ${status.messagesBehind} messages pending background processing. If compaction occurs, some context may not be in long-term memory yet.
</compaction_warning>`;
  }

  return '';
}
