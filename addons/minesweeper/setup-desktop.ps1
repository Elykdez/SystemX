$ErrorActionPreference = 'Stop'
$serverRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$pythonPath = Join-Path $serverRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Run '$serverRoot/setup.bat' first." }
& $pythonPath -m pip install -r (Join-Path $PSScriptRoot 'requirements-desktop.txt')
if ($LASTEXITCODE -ne 0) { throw 'Desktop dependency installation failed.' }
Write-Host "Ready. Run '$PSScriptRoot/play-desktop.bat'."
