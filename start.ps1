param(
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [string]$BindAddress = '127.0.0.1',
    [int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run setup.ps1 first.' }
$previousOffline = $env:HF_HUB_OFFLINE
try {
    if (-not (Test-Path Env:HF_HUB_OFFLINE)) {
        $cacheStatus = & $pythonPath (Join-Path $PSScriptRoot 'model_download.py') --check-only
        if ($LASTEXITCODE -eq 0) {
            $env:HF_HUB_OFFLINE = '1'
            Write-Host 'Using the downloaded model locally. Loading and warmup may take a minute.'
        } elseif ($LASTEXITCODE -eq 2) {
            Write-Host 'Model files are missing. Startup will download them before loading.'
        } else {
            throw 'Model cache check failed. Run model_download.py --check-only to inspect the error.'
        }
    }
    & $pythonPath (Join-Path $PSScriptRoot 'server.py') --host $BindAddress --port $Port --device $Device
    $serverExitCode = $LASTEXITCODE
} finally {
    $env:HF_HUB_OFFLINE = $previousOffline
}
exit $serverExitCode
