# Set up AlpaSandbox on Windows (CARLA on Windows, model in WSL2).
# Usage (PowerShell):  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 [-CarlaVersion 0.9.16]
param([string]$CarlaVersion = "0.9.16", [string]$Distro = "Ubuntu-24.04")
$ErrorActionPreference = "Stop"

Write-Host "==> Python 3.12 virtual environment in .venv"
py -3.12 -m venv .venv
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install -e . "carla==$CarlaVersion"

Write-Host "==> WSL distros"
wsl.exe -l -v
Write-Host "If $Distro is missing: wsl --install -d $Distro"
Write-Host "Then inside WSL: bash scripts/setup_wsl.sh"

Write-Host "==> Writing alpasandbox.toml (WSL runtime, CARLA backend)"
if (-not (Test-Path alpasandbox.toml)) { Copy-Item examples\wsl.toml alpasandbox.toml }

Write-Host "`nNext: start CarlaUE4.exe, then:"
Write-Host "  .venv\Scripts\Activate.ps1"
Write-Host "  alpasandbox doctor"
Write-Host "  alpasandbox app"
