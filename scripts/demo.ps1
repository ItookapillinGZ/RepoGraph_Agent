param(
    [ValidateSet("docker", "host")]
    [string]$Sandbox = "docker",
    [switch]$Fresh
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "RepoGraph virtual environment was not found. Run .\scripts\doctor.ps1 after setup."
}
$Arguments = @("-m", "repograph", "demo", "--sandbox", $Sandbox)
if ($Fresh) { $Arguments += "--fresh" }
& $Python @Arguments
exit $LASTEXITCODE
