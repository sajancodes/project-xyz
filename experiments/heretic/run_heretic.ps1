# Run Heretic abliteration against the downloaded 4-bit Qwen model.
# Usage: .\run_heretic.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Heretic = Join-Path $Root ".venv\Scripts\heretic.exe"
Set-Location (Join-Path $Root "experiments\heretic")
& $Heretic
