param(
    [string]$BindHost = "0.0.0.0",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw ".venv not found. Run scripts/setup_venv.ps1 first."
}

Set-Location $ProjectRoot
& $VenvPython -m uvicorn app.main:app --host $BindHost --port $Port
