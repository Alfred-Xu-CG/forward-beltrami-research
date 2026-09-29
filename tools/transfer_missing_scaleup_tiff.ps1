# Recover the already selected ACROBAT raw inputs on D: without recopying complete TIFFs.
$ErrorActionPreference = 'Stop'
$remoteRoot = '/home/ET/zhxu/codex_runs/digital_topology_wsi_20260929/scaleup_tiff_temp'
$localRoot = 'D:\QC_optimization_data\digital_topology_wsi\ACROBAT_train_subset\scaleup102_raw_tiff'
$identity = 'C:/Users/xuzhehao/.ssh/id_ed25519_ai_codex'
$hostName = 'zhxu@ai.math.cuhk.edu.hk'
if (-not (Test-Path -LiteralPath $localRoot -PathType Container)) {
    throw "Missing intended D-drive destination: $localRoot"
}
$listing = & ssh.exe -o BatchMode=yes -o ConnectTimeout=10 -o ClearAllForwardings=yes -J friend-aliyun ai-codex-mihomo-codex "find $remoteRoot -maxdepth 1 -type f -name '*.tiff' -printf '%f %s\n'"
if ($LASTEXITCODE -ne 0) { throw 'Remote input listing failed' }
$copied = 0
$skipped = 0
Push-Location -LiteralPath $localRoot
try {
    foreach ($row in $listing) {
        if ($row -notmatch '^([0-9]+_[A-Za-z0-9]+\.tiff) ([0-9]+)$') {
            throw "Unexpected remote file-list row: $row"
        }
        $name = $Matches[1]
        $expected = [int64]::Parse($Matches[2])
        $localFile = Join-Path $localRoot $name
        $present = Get-Item -LiteralPath $localFile -ErrorAction SilentlyContinue
        if ($null -ne $present -and $present.Length -eq $expected) {
            $skipped++
            continue
        }
        $source = "${hostName}:$remoteRoot/$name"
        $success = $false
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            & scp.exe -O -q -o BatchMode=yes -o ConnectTimeout=10 -o ClearAllForwardings=yes -o "IdentityFile=$identity" -J friend-aliyun $source .
            $present = Get-Item -LiteralPath $localFile -ErrorAction SilentlyContinue
            if ($LASTEXITCODE -eq 0 -and $null -ne $present -and $present.Length -eq $expected) {
                $success = $true
                break
            }
        }
        if (-not $success) { throw "Could not transfer complete $name" }
        $copied++
        if ($copied % 10 -eq 0) { Write-Output "Copied $copied missing TIFFs; skipped $skipped complete files" }
    }
} finally {
    Pop-Location
}
Write-Output "Complete: copied $copied, skipped $skipped, remote count $($listing.Count)"
