[CmdletBinding()]
param(
    [string]$SourcePath = (Split-Path -Parent $PSScriptRoot),
    [string]$TargetPath = 'D:\QC_optimization'
)

$ErrorActionPreference = 'Stop'

$source = (Resolve-Path -LiteralPath $SourcePath).Path
$targetFull = [System.IO.Path]::GetFullPath($TargetPath)
if ($source.TrimEnd('\') -ieq $targetFull.TrimEnd('\')) {
    throw 'Source and target paths must be different.'
}

if (Test-Path -LiteralPath $targetFull) {
    $target = (Resolve-Path -LiteralPath $targetFull).Path
    $existing = @(Get-ChildItem -LiteralPath $target -Force)
    if ($existing.Count -gt 0) {
        $marker = Join-Path $target 'pyproject.toml'
        $recognized = (Test-Path -LiteralPath $marker) -and
            ((Get-Content -LiteralPath $marker -Raw) -match 'name\s*=\s*["'']qcopt["'']')
        if (-not $recognized) {
            throw "Target path is not empty and is not a recognized qcopt project: $target"
        }
    }
} else {
    $targetParent = Split-Path -Parent $targetFull
    if (-not (Test-Path -LiteralPath $targetParent)) {
        throw "Target parent does not exist: $targetParent"
    }
    New-Item -ItemType Directory -Path $targetFull | Out-Null
    $target = (Resolve-Path -LiteralPath $targetFull).Path
}

foreach ($entry in Get-ChildItem -LiteralPath $source -Force) {
    Copy-Item -LiteralPath $entry.FullName -Destination $target -Recurse -Force
}

$verifiedFiles = 0
$manifestVerified = $false
$manifestPath = Join-Path $target 'artifacts\manifest.json'
if (Test-Path -LiteralPath $manifestPath) {
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    foreach ($item in $manifest.files) {
        $artifactPath = Join-Path (Join-Path $target 'artifacts') $item.path
        if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
            throw "Manifest artifact is missing after delivery: $($item.path)"
        }
        $file = Get-Item -LiteralPath $artifactPath
        if ($file.Length -ne [long]$item.bytes) {
            throw "Manifest byte count mismatch after delivery: $($item.path)"
        }
        $digest = (Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($digest -ne ([string]$item.sha256).ToLowerInvariant()) {
            throw "Manifest SHA-256 mismatch after delivery: $($item.path)"
        }
        $verifiedFiles++
    }
    $manifestVerified = $true
}

$gitHeadVerified = $null
$sourceGit = Join-Path $source '.git'
$targetGit = Join-Path $target '.git'
if ((Test-Path -LiteralPath $sourceGit) -and (Test-Path -LiteralPath $targetGit)) {
    $sourceHead = & git -C $source rev-parse HEAD 2>$null
    $sourceGitValid = $LASTEXITCODE -eq 0
    $targetHead = & git -C $target rev-parse HEAD 2>$null
    $targetGitValid = $LASTEXITCODE -eq 0
    if ($sourceGitValid -and $targetGitValid) {
        if ($sourceHead -ne $targetHead) {
            throw "Git HEAD mismatch after delivery: source=$sourceHead target=$targetHead"
        }
        $gitHeadVerified = $true
    }
}

[ordered]@{
    source = $source
    target = $target
    manifest_verified = $manifestVerified
    verified_files = $verifiedFiles
    git_head_verified = $gitHeadVerified
} | ConvertTo-Json -Compress
