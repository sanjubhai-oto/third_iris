# Install the SimpleFlight AirSim settings used by the Web UI and scenario scripts.
# Usage:
#   powershell -ExecutionPolicy Bypass -File env\install_airsim_settings.ps1
$ErrorActionPreference = "Stop"

$Source = Join-Path $PSScriptRoot "airsim_settings.simpleflight.json"
$DestDir = Join-Path $env:USERPROFILE "OneDrive\Documents\AirSim"
$Dest = Join-Path $DestDir "settings.json"

if (-not (Test-Path $Source)) {
    throw "Missing settings template: $Source"
}

New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
if (Test-Path $Dest) {
    $Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    Copy-Item -LiteralPath $Dest -Destination "$Dest.bak-$Stamp" -Force
    Write-Host "Backed up existing settings.json to settings.json.bak-$Stamp" -ForegroundColor Yellow
}

Copy-Item -LiteralPath $Source -Destination $Dest -Force
Write-Host "Installed AirSim settings:" -ForegroundColor Green
Write-Host "  $Dest"
Write-Host "Vehicles: Ego + Target SimpleFlight; camera: front_center Scene + DepthPlanar, 1280x720, FOV 90."
