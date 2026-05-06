#!/usr/bin/env bash
set -euo pipefail

# Hermes Memory System — Installer (macOS / Linux)
#
# Usage:
#   ./install.sh
#   ./install.sh --data-dir ~/my-hermes --python python3.11

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${HOME}/.hermes"
PYTHON_PATH="python3"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --python)   PYTHON_PATH="$2"; shift 2 ;;
    *)          echo "Unknown arg: $1"; exit 1 ;;
  esac
done

step()  { echo -e "\n\033[36m>> $1\033[0m"; }
ok()    { echo -e "   \033[32mOK: $1\033[0m"; }
warn()  { echo -e "   \033[33mWARN: $1\033[0m"; }
err()   { echo -e "   \033[31mERROR: $1\033[0m"; }

echo ""
echo -e "\033[36m============================================\033[0m"
echo -e "\033[36m  Hermes Memory System - Installer\033[0m"
echo -e "\033[36m============================================\033[0m"
echo ""
echo "  Plugin dir:  $SCRIPT_DIR"
echo "  Data dir:    $DATA_DIR"
echo "  Python:      $PYTHON_PATH"
echo ""

# ── Step 1: Directories ──
step "Creating directory structure"
mkdir -p "$DATA_DIR"/{data/memory_store,logs,models,backups}
ok "Directory structure ready"

# ── Step 2: Python check ──
step "Checking Python"
PY_VER=$($PYTHON_PATH --version 2>&1) || { err "Python not found at '$PYTHON_PATH'"; exit 1; }
ok "$PY_VER"

# ── Step 3: Venv ──
VENV_PATH="$DATA_DIR/venv"
step "Creating Python virtual environment"
if [[ ! -f "$VENV_PATH/bin/python" ]]; then
  $PYTHON_PATH -m venv "$VENV_PATH"
  ok "Created venv at $VENV_PATH"
else
  ok "Venv already exists"
fi
VENV_PYTHON="$VENV_PATH/bin/python"

# ── Step 4: Python deps ──
step "Installing Python dependencies"
"$VENV_PYTHON" -m pip install --upgrade pip --quiet
"$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements-memory.txt" --quiet
ok "Python dependencies installed"

# ── Step 5: Embedding model ──
step "Pre-downloading embedding model (~500MB first time)"
"$VENV_PYTHON" -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('Alibaba-NLP/gte-modernbert-base')
r = m.encode(['test'])
print(f'OK: model loaded, embedding dim={len(r[0])}')
" || warn "Model download failed — will download on first use"

# ── Step 6: Node deps ──
step "Installing Node dependencies"
if [[ ! -d "$SCRIPT_DIR/node_modules" ]]; then
  cd "$SCRIPT_DIR" && npm install --quiet
  ok "Node modules installed"
else
  ok "Node modules already present"
fi

# ── Step 7: Verify Python modules ──
step "Verifying Python memory modules"
PYTHON_DIR="$SCRIPT_DIR/python"
if [[ -f "$PYTHON_DIR/memory/store.py" ]]; then
  ok "Python modules found at $PYTHON_DIR"
else
  err "python/memory/store.py not found — repo may be incomplete"
  exit 1
fi

# ── Step 8: Config ──
step "Writing configuration"
CONFIG_PATH="$SCRIPT_DIR/hermes.config.json"
cat > "$CONFIG_PATH" <<EOF
{
  "dataDir": "$DATA_DIR",
  "pythonPath": "$VENV_PYTHON",
  "mode": "full",
  "debug": false,
  "venvPath": "$VENV_PATH"
}
EOF
ok "Config written to $CONFIG_PATH"

# ── Step 9: Add python/ to venv path ──
step "Setting up Python path"
SITE_PACKAGES=$(find "$VENV_PATH" -type d -name "site-packages" | head -1)
printf '%s\n' "$PYTHON_DIR" > "$SITE_PACKAGES/hermes-memory.pth"
ok "Added python/ to venv site-packages"

# ── Done ──
echo ""
echo -e "\033[32m============================================\033[0m"
echo -e "\033[32m  Installation Complete!\033[0m"
echo -e "\033[32m============================================\033[0m"
echo ""
echo "  To activate, run Claude with:"
echo -e "    \033[33mclaude --plugin-dir \"$SCRIPT_DIR\"\033[0m"
echo ""
echo "  Config: $CONFIG_PATH"
echo "  Data:   $DATA_DIR"
echo "  Python: $VENV_PYTHON"
echo ""
