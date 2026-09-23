param([ValidateSet('cuda', 'cpu')][string]$Device = 'cuda')
$ErrorActionPreference = 'Stop'
$serverRoot = $PSScriptRoot
$pythonPath = Join-Path $serverRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    & py -3.12 -m venv (Join-Path $serverRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
}
& $pythonPath -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'Unable to initialize pip.' }
$torchIndex = if ($Device -eq 'cuda') { 'https://download.pytorch.org/whl/cu126' } else { 'https://download.pytorch.org/whl/cpu' }
& $pythonPath -m pip install 'torch==2.9.0' --index-url $torchIndex --timeout 600
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $pythonPath -m pip install -r (Join-Path $serverRoot 'requirements.txt') --timeout 600
if ($LASTEXITCODE -ne 0) { throw 'Server dependency installation failed.' }
& $pythonPath (Join-Path $serverRoot 'backend.py')
if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }
Write-Host "Ready to start: & '$serverRoot/start.ps1' -Device $Device"
