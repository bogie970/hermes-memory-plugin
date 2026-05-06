"""Memory subsystem configuration constants.

Standalone — reads paths from hermes.config.json or defaults to ~/.hermes/.
No dependency on external repos.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

# --- Path resolution ---
_PYTHON_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PLUGIN_ROOT = _PYTHON_ROOT.parent
_CONFIG_FILE = _PLUGIN_ROOT / "hermes.config.json"

def _load_plugin_config() -> dict:
    try:
        if _CONFIG_FILE.exists():
            return json.loads(_CONFIG_FILE.read_text())
    except Exception:
        pass
    return {}

_plugin_cfg = _load_plugin_config()

_default_home = pathlib.Path.home() / ".hermes"
_home_dir = pathlib.Path(
    _plugin_cfg.get("dataDir") or os.environ.get("HERMES_DATA_DIR") or str(_default_home)
)
DATA_DIR = _home_dir / "data"
LANCEDB_DIR = DATA_DIR / "memory_store"
MEMORY_DIR = LANCEDB_DIR
LANCEDB_PATH = str(LANCEDB_DIR)
MEMORY_BACKUPS_DIR = DATA_DIR / "memory_backups"
LOGS_DIR = _home_dir / "logs"

# Ensure python/ is on sys.path so `from memory.xxx` imports work
_python_str = str(_PYTHON_ROOT)
if _python_str not in sys.path:
    sys.path.insert(0, _python_str)

# Ensure data directories exist
for _d in (DATA_DIR, LANCEDB_DIR, MEMORY_BACKUPS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Model config ---
EMBEDDING_MODEL = "Alibaba-NLP/gte-modernbert-base"
EMBEDDING_DIM = 768
EMBEDDING_DEVICE = "cpu"

# Tier thresholds
L1_MAX_TOKENS = 4000
L1_RECENT_SESSIONS = 5
L3_AGE_DAYS = 90
L3_IMPORTANCE_THRESHOLD = 0.15

# Retrieval defaults
DEFAULT_TOP_K = 5
OVERFETCH_MULTIPLIER = 10

# Consolidation
MERGE_SIMILARITY_THRESHOLD = 0.85
ARCHIVE_MIN_AGE_DAYS = 60
ARCHIVE_MIN_ACCESS_COUNT = 3
IMPORTANCE_DECAY_RATE = float(os.environ.get("HERMES_DECAY_RATE", "0.95"))
ACCESS_PROMOTE_THRESHOLD = 5
ACCESS_PROMOTE_IMPORTANCE = 0.75

# Triple-score weights (Park et al. 2023)
ALPHA_RELEVANCE = 0.5
BETA_RECENCY = 0.2
GAMMA_IMPORTANCE = 0.3
RECENCY_DECAY_RATE = 0.01

TABLE_NAME = "memories"
