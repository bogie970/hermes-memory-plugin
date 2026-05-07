#Requires -Version 5.1
<#
.SYNOPSIS
    Install Hermes Memory System for Claude Code.

.DESCRIPTION
    Sets up everything needed for the hierarchical memory system:
    1. Creates ~/.hermes/ directory structure
    2. Creates Python virtual environment
    3. Installs Python dependencies (LanceDB, sentence-transformers, etc.)
    4. Pre-downloads the embedding model
    5. Writes hermes.config.json
    6. Installs node dependencies
    7. Tells you how to activate the plugin

.EXAMPLE
    .\install.ps1
    .\install.ps1 -DataDir "D:\my-hermes"
    .\install.ps1 -PythonPath "python3.11"
#>

param(
    [string]$DataDir = (Join-Path $env:USERPROFILE ".hermes"),
    [string]$PythonPath = "python"
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Step($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "   OK: $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "   WARN: $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "   ERROR: $msg" -ForegroundColor Red }

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Hermes Memory System - Installer" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Plugin dir:  $ScriptRoot"
Write-Host "  Data dir:    $DataDir"
Write-Host "  Python:      $PythonPath"
Write-Host ""

# ── Step 1: Create directory structure ──
Write-Step "Creating directory structure"
$dirs = @(
    $DataDir,
    (Join-Path $DataDir "data"),
    (Join-Path (Join-Path $DataDir "data") "memory_store"),
    (Join-Path $DataDir "logs"),
    (Join-Path $DataDir "models"),
    (Join-Path $DataDir "backups")
)
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
        Write-OK "Created $d"
    }
}
Write-OK "Directory structure ready"

# ── Step 2: Check Python ──
Write-Step "Checking Python"
try {
    $pyVer = & $PythonPath --version 2>&1
    Write-OK "$pyVer"
} catch {
    Write-Err "Python not found at '$PythonPath'. Install Python 3.10+ and retry."
    exit 1
}

# ── Step 3: Create virtual environment ──
$VenvPath = Join-Path $DataDir "venv"
Write-Step "Creating Python virtual environment"
if (-not (Test-Path (Join-Path (Join-Path $VenvPath "Scripts") "python.exe"))) {
    & $PythonPath -m venv $VenvPath
    Write-OK "Created venv at $VenvPath"
} else {
    Write-OK "Venv already exists at $VenvPath"
}

$VenvPython = Join-Path (Join-Path $VenvPath "Scripts") "python.exe"

# ── Step 4: Install Python dependencies ──
Write-Step "Installing Python dependencies (this may take a few minutes)"
$reqFile = Join-Path $ScriptRoot "requirements-memory.txt"
& $VenvPython -m pip install --upgrade pip --quiet 2>&1 | Out-Null
& $VenvPython -m pip install -r $reqFile --quiet 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Err "pip install failed. Check the output above."
    exit 1
}
Write-OK "Python dependencies installed"

# ── Step 5: Pre-download embedding model ──
Write-Step "Pre-downloading embedding model (first time takes ~500MB)"
$modelScript = @"
import sys
try:
    from sentence_transformers import SentenceTransformer
    # Pinned revision — keep in sync with python/memory/embeddings.py
    model = SentenceTransformer(
        'Alibaba-NLP/gte-modernbert-base',
        revision='e7f32e3c00f91d699e8c43b53106206bcc72bb22',
    )
    result = model.encode(['test'])
    print(f'OK: model loaded, embedding dim={len(result[0])}')
except Exception as e:
    print(f'FAIL: {e}', file=sys.stderr)
    sys.exit(1)
"@
& $VenvPython -c $modelScript
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Embedding model download failed. It will download on first use (slower first prompt)."
} else {
    Write-OK "Embedding model cached"
}

# ── Step 6: Install Node dependencies ──
Write-Step "Installing Node dependencies"
Push-Location $ScriptRoot
try {
    if (-not (Test-Path (Join-Path $ScriptRoot "node_modules"))) {
        npm install --quiet 2>&1 | Out-Null
        Write-OK "Node modules installed"
    } else {
        Write-OK "Node modules already present"
    }
} finally {
    Pop-Location
}

# ── Step 7: Verify Python modules ──
Write-Step "Verifying Python memory modules"
$pythonDir = Join-Path $ScriptRoot "python"
$memoryDir = Join-Path $pythonDir "memory"
if (Test-Path (Join-Path $memoryDir "store.py")) {
    Write-OK "Python modules found at $pythonDir"
} else {
    Write-Err "python/memory/store.py not found — repo may be incomplete"
    exit 1
}

# ── Step 8: Write config ──
Write-Step "Writing configuration"
$configPath = Join-Path $ScriptRoot "hermes.config.json"
$config = @{
    dataDir    = $DataDir
    pythonPath = $VenvPython
    mode       = "full"
    debug      = $false
    venvPath   = $VenvPath
} | ConvertTo-Json -Depth 3

Set-Content -Path $configPath -Value $config -Encoding utf8
Write-OK "Config written to $configPath"

# ── Step 9: Add python/ to venv path ──
Write-Step "Setting up Python path"
$pthFile = Join-Path (Join-Path (Join-Path $VenvPath "Lib") "site-packages") "hermes-memory.pth"
Set-Content -Path $pthFile -Value $pythonDir -Encoding utf8
Write-OK "Added python/ to venv site-packages"

# ── Done ──
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  To activate, run Claude with:" -ForegroundColor White
Write-Host "    claude --plugin-dir `"$ScriptRoot`"" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Or add to your Claude settings (~/.claude/settings.json):" -ForegroundColor White
Write-Host "    { `"plugins`": [`"$($ScriptRoot -replace '\\','\\')`"] }" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Config file: $configPath" -ForegroundColor Gray
Write-Host "  Data dir:    $DataDir" -ForegroundColor Gray
Write-Host "  Python:      $VenvPython" -ForegroundColor Gray
Write-Host ""
