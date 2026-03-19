param(
    [ValidateSet("gpu", "cpu")]
    [string]$Device = "gpu",
    [string]$Wheelhouse = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
$PipExe = Join-Path $VenvPath "Scripts\pip.exe"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not available on PATH. Install Python 3.10+ first."
}

if (-not (Test-Path $VenvPath)) {
    python -m venv $VenvPath
}

& $PythonExe -m pip install --upgrade pip setuptools wheel

$TorchIndex = if ($Device -eq "gpu") { "https://download.pytorch.org/whl/cu121" } else { "https://download.pytorch.org/whl/cpu" }

if ([string]::IsNullOrWhiteSpace($Wheelhouse)) {
    & $PipExe install torch torchvision torchaudio --index-url $TorchIndex
    & $PipExe install -r (Join-Path $ProjectRoot "requirements.txt")
} else {
    & $PipExe install --no-index --find-links $Wheelhouse torch torchvision torchaudio
    & $PipExe install --no-index --find-links $Wheelhouse -r (Join-Path $ProjectRoot "requirements.txt")
}

Write-Host "Virtual environment is ready: $VenvPath"
