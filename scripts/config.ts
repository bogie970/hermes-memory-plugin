/**
 * Centralized configuration loader.
 *
 * Reads hermes.config.json from the plugin root directory.
 * Falls back to environment variables for backward compatibility.
 * Falls back to sensible defaults when nothing is configured.
 *
 * Config file location: <plugin_root>/hermes.config.json
 * Created automatically by install script.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const PLUGIN_ROOT = path.resolve(__dirname, '..');
export const PYTHON_DIR = path.join(PLUGIN_ROOT, 'python');
const CONFIG_PATH = path.join(PLUGIN_ROOT, 'hermes.config.json');
const DEFAULT_DATA_DIR = path.join(os.homedir(), '.hermes');

interface HermesConfig {
  dataDir: string;
  pythonPath: string;
  pythonDir: string;
  mode: 'full' | 'whisper' | 'off';
  debug: boolean;
  venvPath: string;
  // Legacy — kept for backward compat, defaults to PLUGIN_ROOT
  hermesRoot: string;
}

let _cached: HermesConfig | null = null;

function loadConfigFile(): Record<string, unknown> {
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      return JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf-8'));
    }
  } catch {}
  return {};
}

function expandHome(p: string): string {
  if (p.startsWith('~')) {
    return path.join(os.homedir(), p.slice(1));
  }
  return p;
}

export function getConfig(): HermesConfig {
  if (_cached) return _cached;

  const file = loadConfigFile();

  const dataDir = expandHome(
    (file.dataDir as string) || process.env.LETTA_HOME || DEFAULT_DATA_DIR
  );

  const venvPath = expandHome(
    (file.venvPath as string) || path.join(dataDir, 'venv')
  );

  // Use venv python if it exists, otherwise fall back
  let pythonPath = (file.pythonPath as string) || process.env.HERMES_PYTHON || '';
  if (!pythonPath) {
    const venvPython = process.platform === 'win32'
      ? path.join(venvPath, 'Scripts', 'python.exe')
      : path.join(venvPath, 'bin', 'python');
    pythonPath = fs.existsSync(venvPython) ? venvPython : 'python';
  }

  const modeRaw = ((file.mode as string) || process.env.LETTA_MODE || 'full').trim().toLowerCase();
  const mode = (modeRaw === 'full' || modeRaw === 'whisper' || modeRaw === 'off') ? modeRaw : 'full';

  const debug = (file.debug as boolean) || process.env.LETTA_DEBUG === '1' || false;

  _cached = {
    dataDir,
    pythonPath,
    pythonDir: PYTHON_DIR,
    mode,
    debug,
    venvPath,
    hermesRoot: PLUGIN_ROOT,
  };
  return _cached;
}

export function getPythonPath(): string {
  return getConfig().pythonPath;
}

export function getDataDir(): string {
  return getConfig().dataDir;
}

export function getMode(): 'full' | 'whisper' | 'off' {
  return getConfig().mode;
}

export function isDebug(): boolean {
  return getConfig().debug;
}

export function getConfigPath(): string {
  return CONFIG_PATH;
}
