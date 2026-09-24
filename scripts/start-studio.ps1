param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3000
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$StudioRoot = Join-Path $ProjectRoot "studio"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing .venv Python." }
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw "npm was not found." }
if (-not (Test-Path -LiteralPath (Join-Path $StudioRoot "node_modules") -PathType Container)) {
    throw "Studio dependencies are missing. Run npm install in .\studio."
}
foreach ($Port in @($BackendPort, $FrontendPort)) {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $Port is already in use. Choose a different port parameter."
    }
}
if (-not $env:REPOGRAPH_WORKSPACE_ROOT) { $env:REPOGRAPH_WORKSPACE_ROOT = $ProjectRoot }
if (-not $env:REPOGRAPH_STUDIO_DATA_DIR) {
    $env:REPOGRAPH_STUDIO_DATA_DIR = Join-Path ([System.IO.Path]::GetTempPath()) "repograph-studio"
}
$env:REPOGRAPH_STUDIO_ALLOWED_ORIGIN = "http://localhost:$FrontendPort"
$env:NEXT_PUBLIC_REPOGRAPH_API_URL = "http://127.0.0.1:$BackendPort"
$Backend = $null
$Frontend = $null
try {
    $Backend = Start-Process -FilePath $Python -ArgumentList @(
        "-m", "uvicorn", "studio_backend.app:create_app", "--factory",
        "--host", "127.0.0.1", "--port", $BackendPort
    ) -WorkingDirectory $ProjectRoot -PassThru -NoNewWindow
    $Frontend = Start-Process -FilePath "npm.cmd" -ArgumentList @(
        "run", "dev", "--", "--hostname", "127.0.0.1", "--port", $FrontendPort
    ) -WorkingDirectory $StudioRoot -PassThru -NoNewWindow
    Start-Sleep -Seconds 3
    if ($Backend.HasExited) { throw "Studio backend exited during startup." }
    if ($Frontend.HasExited) { throw "Studio frontend exited during startup." }
    Write-Host "RepoGraph Studio: http://localhost:$FrontendPort"
    Write-Host "Backend health: http://127.0.0.1:$BackendPort/api/health"
    Write-Host "Press Ctrl+C to stop both processes."
    Wait-Process -Id $Backend.Id, $Frontend.Id
}
finally {
    foreach ($Process in @($Frontend, $Backend)) {
        if ($null -ne $Process -and -not $Process.HasExited) {
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
