$ErrorActionPreference = 'Stop'
$serverRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$pythonPath = Join-Path $serverRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Run '$serverRoot/setup.bat' first." }
& $pythonPath -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Minesweeper dependency installation failed.' }
Write-Host "Ready. Start the model with '$serverRoot/start.bat', then run '$PSScriptRoot/play-minesweeper.bat'. Google Chrome must be installed."
