# Phase 1 Windows setup: venv + Blackwell-compatible PyTorch (cu128) + Ultralytics (YOLO26 + ByteTrack).
# Usage:  powershell -ExecutionPolicy Bypass -File env\setup_windows.ps1
$ErrorActionPreference = "Stop"

# Resolve repo root (parent of this script's dir) so it works from anywhere.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
Write-Host "Repo root: $RepoRoot" -ForegroundColor Cyan

# Pick a Python launcher.
$py = (Get-Command py -ErrorAction SilentlyContinue)
if ($py) { $PYEXE = "py"; $PYARGS = @("-3") } else { $PYEXE = "python"; $PYARGS = @() }

# 1. Create venv
if (-not (Test-Path "$RepoRoot\.venv")) {
    Write-Host "Creating venv..." -ForegroundColor Cyan
    & $PYEXE @PYARGS -m venv "$RepoRoot\.venv"
} else {
    Write-Host ".venv already exists, reusing." -ForegroundColor Yellow
}
$VPY = "$RepoRoot\.venv\Scripts\python.exe"

# 2. Upgrade pip
& $VPY -m pip install --upgrade pip wheel setuptools

# 3. Perception stack first (ultralytics will pull a CPU torch from PyPI — we override it next).
Write-Host "Installing perception requirements..." -ForegroundColor Cyan
& $VPY -m pip install -r "$RepoRoot\env\requirements.txt"

# 4. PyTorch for Blackwell (RTX 50xx) — CUDA 12.8 wheels. MUST be last: force-reinstall so it
#    replaces whatever CPU-only torch ultralytics dragged in from the default PyPI index.
Write-Host "Installing PyTorch (cu128, Blackwell-compatible) — overriding any CPU torch..." -ForegroundColor Cyan
& $VPY -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 5. Verify CUDA + Ultralytics see the GPU
Write-Host "`n=== Verification ===" -ForegroundColor Green
& $VPY -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
& $VPY -m ultralytics checks

Write-Host "`nDone. Activate with:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
