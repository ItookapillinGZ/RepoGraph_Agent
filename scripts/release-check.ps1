param(
    [switch]$SkipStudio,
    [switch]$SkipDocker
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Arguments = @("-m", "repograph", "release-check")
if ($SkipStudio) { $Arguments += "--skip-studio" }
if ($SkipDocker) { $Arguments += "--skip-docker" }
& $Python @Arguments
exit $LASTEXITCODE
