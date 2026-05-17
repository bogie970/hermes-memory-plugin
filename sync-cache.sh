#!/usr/bin/env bash
# Sync hermes-memory plugin source -> Claude Code plugin cache.
# Mirror of sync-cache.ps1 for WSL / bash users.
# Always exits 0.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve cache path: arg > plugin-config.json > default
CACHE="${1:-}"
if [ -z "$CACHE" ]; then
  CFG="$HOME/.claude/plugin-config.json"
  if [ -f "$CFG" ] && command -v jq >/dev/null 2>&1; then
    v=$(jq -r '.hermesMemoryCache // empty' "$CFG" 2>/dev/null || true)
    if [ -n "$v" ]; then CACHE="$v"; fi
  fi
fi
if [ -z "$CACHE" ]; then
  CACHE="$HOME/.claude/plugins/cache/hermes-memory/hermes-memory/1.0.0"
fi

if [ ! -d "$CACHE" ]; then
  echo "Cache path not found: $CACHE"
  echo "(Run install.sh first, or pass cache path as arg 1.)"
  exit 0
fi

echo "Hermes plugin cache sync"
echo "  Source: $SCRIPT_DIR"
echo "  Cache:  $CACHE"
echo

COPIED=0
UNCHANGED=0
MISSING=0
COPIED_FILES=()

sync_file() {
  local src="$1" dst="$2" only_if_newer="${3:-0}"
  [ -f "$src" ] || return 0
  mkdir -p "$(dirname "$dst")"
  if [ -f "$dst" ]; then
    local s d sl dl
    s=$(stat -c %Y "$src" 2>/dev/null || stat -f %m "$src")
    d=$(stat -c %Y "$dst" 2>/dev/null || stat -f %m "$dst")
    sl=$(stat -c %s "$src" 2>/dev/null || stat -f %z "$src")
    dl=$(stat -c %s "$dst" 2>/dev/null || stat -f %z "$dst")
    if [ "$only_if_newer" = "1" ] && [ "$s" -le "$d" ]; then
      UNCHANGED=$((UNCHANGED+1)); return 0
    fi
    if [ "$s" = "$d" ] && [ "$sl" = "$dl" ]; then
      UNCHANGED=$((UNCHANGED+1)); return 0
    fi
  else
    MISSING=$((MISSING+1))
  fi
  cp -p "$src" "$dst"
  COPIED=$((COPIED+1))
  COPIED_FILES+=("${dst#$CACHE/}")
}

sync_glob() {
  local rel_dir="$1" pattern="$2"
  local src_dir="$SCRIPT_DIR/$rel_dir"
  [ -d "$src_dir" ] || return 0
  shopt -s nullglob
  for f in "$src_dir"/$pattern; do
    [ -f "$f" ] || continue
    sync_file "$f" "$CACHE/$rel_dir/$(basename "$f")"
  done
  shopt -u nullglob
}

sync_glob "scripts"            "*.ts"
sync_glob "scripts"            "*.cjs"
sync_glob "scripts"            "*.py"
sync_glob "python/memory"      "*.py"
sync_glob "python/subconscious" "*.py"
sync_glob "commands"           "*.md"

sync_file "$SCRIPT_DIR/hooks/hooks.json" "$CACHE/hooks/hooks.json"
sync_glob "hooks"              "*.cjs"
sync_file "$SCRIPT_DIR/hermes.config.json" "$CACHE/hermes.config.json" 1

echo
echo "Sync complete:"
echo "  Copied:    $COPIED"
echo "  Unchanged: $UNCHANGED"
if [ "$MISSING" -gt 0 ]; then echo "  New (not previously in cache): $MISSING"; fi

if [ "$COPIED" -gt 0 ] && [ "$COPIED" -le 30 ]; then
  echo
  echo "Updated files:"
  for f in "${COPIED_FILES[@]}"; do echo "    $f"; done
fi

exit 0
