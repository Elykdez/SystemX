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
& $pythonPath -m pip install 'torch==2.9.0' 'torchvision==0.24.0' --index-url $torchIndex --timeout 600
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $pythonPath -m pip install -r (Join-Path $serverRoot 'requirements.txt') --timeout 600
if ($LASTEXITCODE -ne 0) { throw 'Server dependency installation failed.' }
if ($Device -eq 'cuda') {
    & $pythonPath -m pip install -r (Join-Path $serverRoot 'requirements-cuda.txt') --timeout 600
    if ($LASTEXITCODE -ne 0) { throw 'CUDA acceleration dependency installation failed.' }
}
Write-Host 'Preparing the System One Instant model. Missing files will be downloaded; an existing complete cache is reused.'
& $pythonPath (Join-Path $serverRoot 'model_download.py')
if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }
Write-Host "Ready to start: '$serverRoot/start.bat' -Device $Device"
