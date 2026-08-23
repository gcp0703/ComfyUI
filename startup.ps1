<#
.SYNOPSIS
    Launch the azure_worker (Azure Storage Queue <-> ComfyUI + Ollama bridge).

.DESCRIPTION
    Activates the project virtualenv and runs `python -m azure_worker.main`
    from the repo root. Configuration is read from azure_worker\.env, which
    main.py auto-loads via python-dotenv -- this script sets no environment
    variables itself.

    Ollama is assumed to already be running (it starts from the Windows tray);
    this script does not start it.

.EXAMPLE
    .\startup.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# Repo root = this script's directory.
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

$VenvActivate = Join-Path $Root '.venv\Scripts\Activate.ps1'
$EnvFile      = Join-Path $Root 'azure_worker\.env'

if (-not (Test-Path -LiteralPath $VenvActivate)) {
    throw "Virtualenv not found at $VenvActivate. Create it and install azure_worker\requirements.txt first."
}

if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Missing azure_worker\.env. Copy azure_worker\.env.example to azure_worker\.env and fill it in."
}

Write-Host "Activating virtualenv..." -ForegroundColor Cyan
. $VenvActivate

Write-Host "Starting azure_worker (Ctrl+C to stop)..." -ForegroundColor Cyan
python -m azure_worker.main
exit $LASTEXITCODE
