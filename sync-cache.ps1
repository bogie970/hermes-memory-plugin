#Requires -Version 5.1
<#
.SYNOPSIS
    Sync hermes-memory plugin source -> Claude Code plugin cache.

.DESCRIPTION
    Claude Code copies plugins into ~/.claude/plugins/cache/... at install
    time. Subsequent edits to the source tree do NOT propagate automatically,
    and /reload-plugins only re-indexes commands. This script does an
    idempotent copy of the runtime-relevant files so the cache matches source.

    Safe to run repeatedly. Always exits 0 so it never blocks the user.

.EXAMPLE
    pwsh sync-cache.ps1
#>

param(
    [string]$CacheRoot = "",
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# Resolve cache path: arg > plugin-config.json > default
function Resolve-CacheRoot {
    param([string]$Arg)
    if ($Arg) { return $Arg }
    $cfg = Join-Path $env:USERPROFILE ".claude\plugin-config.json"
    if (Test-Path $cfg) {
        try {
            $j = Get-Content $cfg -Raw | ConvertFrom-Json
            if ($j.hermesMemoryCache) { return $j.hermesMemoryCache }
        } catch { }
    }
    return (Join-Path $env:USERPROFILE ".claude\plugins\cache\hermes-memory\hermes-memory\1.0.0")
}

$Cache = Resolve-CacheRoot $CacheRoot

if (-not (Test-Path $Cache)) {
    Write-Host "Cache path not found: $Cache" -ForegroundColor Yellow
    Write-Host "(Run install.ps1 first, or pass -CacheRoot <path>.)" -ForegroundColor Yellow
    exit 0
}

Write-Host "Hermes plugin cache sync" -ForegroundColor Cyan
Write-Host "  Source: $ScriptRoot"
Write-Host "  Cache:  $Cache"
Write-Host ""

$Copied = 0
$Unchanged = 0
$Missing = 0
$CopiedFiles = New-Object System.Collections.ArrayList

function Sync-File {
    param([string]$SrcFile, [string]$DstFile, [bool]$OnlyIfNewer = $false)
    if (-not (Test-Path $SrcFile)) { return }
    $dstDir = Split-Path -Parent $DstFile
    if (-not (Test-Path $dstDir)) {
        New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
    }
    if (Test-Path $DstFile) {
        $s = (Get-Item $SrcFile).LastWriteTimeUtc
        $d = (Get-Item $DstFile).LastWriteTimeUtc
        $sLen = (Get-Item $SrcFile).Length
        $dLen = (Get-Item $DstFile).Length
        if ($OnlyIfNewer -and ($s -le $d)) {
            $script:Unchanged++
            return
        }
        if (($s -eq $d) -and ($sLen -eq $dLen)) {
            $script:Unchanged++
            return
        }
    } else {
        $script:Missing++
    }
    Copy-Item -Path $SrcFile -Destination $DstFile -Force
    $script:Copied++
    [void]$script:CopiedFiles.Add($DstFile.Substring($Cache.Length).TrimStart('\','/'))
    if ($Verbose) { Write-Host "  COPY $DstFile" -ForegroundColor Green }
}

function Sync-Glob {
    param([string]$RelDir, [string]$Pattern)
    $srcDir = Join-Path $ScriptRoot $RelDir
    if (-not (Test-Path $srcDir)) { return }
    Get-ChildItem -Path $srcDir -Filter $Pattern -File | ForEach-Object {
        $dst = Join-Path (Join-Path $Cache $RelDir) $_.Name
        Sync-File -SrcFile $_.FullName -DstFile $dst
    }
}

# Runtime-relevant globs
Sync-Glob "scripts" "*.ts"
Sync-Glob "scripts" "*.cjs"
Sync-Glob "scripts" "*.py"
Sync-Glob "python\memory" "*.py"
Sync-Glob "python\subconscious" "*.py"
Sync-Glob "commands" "*.md"

# Single files
Sync-File (Join-Path $ScriptRoot "hooks\hooks.json") (Join-Path $Cache "hooks\hooks.json")

# Config: only if cache is older (don't clobber a user-installed config)
Sync-File (Join-Path $ScriptRoot "hermes.config.json") (Join-Path $Cache "hermes.config.json") -OnlyIfNewer $true

Write-Host ""
Write-Host "Sync complete:" -ForegroundColor Cyan
Write-Host "  Copied:    $Copied" -ForegroundColor Green
Write-Host "  Unchanged: $Unchanged" -ForegroundColor Gray
if ($Missing -gt 0) { Write-Host "  New (not previously in cache): $Missing" -ForegroundColor Yellow }

if (($Copied -gt 0) -and (-not $Verbose) -and ($Copied -le 30)) {
    Write-Host ""
    Write-Host "Updated files:" -ForegroundColor Gray
    $CopiedFiles | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
}

exit 0
