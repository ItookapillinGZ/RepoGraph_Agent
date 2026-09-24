Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "RepoGraph virtual environment was not found. Create .venv and install requirements.txt."
}
& $Python -m repograph doctor
exit $LASTEXITCODE
