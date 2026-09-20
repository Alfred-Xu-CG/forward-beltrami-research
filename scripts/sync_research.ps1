param(
    [string]$Message = "docs: update forward Beltrami research"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $root = git rev-parse --show-toplevel
    if ($LASTEXITCODE -ne 0 -or ([IO.Path]::GetFullPath($root) -ne [IO.Path]::GetFullPath($repoRoot))) {
        throw "The script must run inside the canonical D:\QC_optimization repository."
    }
    git add -A
    git diff --cached --check
    $staged = @(git diff --cached --name-only)
    if ($staged.Count -eq 0) {
        Write-Output "No changes to synchronize."
        return
    }
    $large = @($staged | Where-Object { $_ -match '\.(npz|npy|png|html)$' })
    if ($large.Count -gt 0) {
        throw "Large generated files are staged; inspect .gitignore before continuing: $($large -join ', ')"
    }
    git commit -m $Message
    git push origin HEAD
    git status --short
    Write-Output "Synchronized $($staged.Count) files to origin."
}
finally {
    Pop-Location
}
