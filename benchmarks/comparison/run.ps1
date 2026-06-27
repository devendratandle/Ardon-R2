# Run both accuracy and performance benchmarks; produce a side-by-side
# comparison table on stdout. Requires:
#   - `R` on PATH (CRAN R 4.x or newer with `Rscript`)
#   - `r2.exe` built in release mode (run `cargo build --release` first)
#
# From repo root:
#   pwsh ./bench/r_vs_r2/run.ps1
#
# Output: prints two tables (accuracy + performance) with R, R2, and
# the absolute / relative difference. Exits 0 always; numerical
# disagreement is informational, not an error.

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Split-Path -Parent (Split-Path -Parent $scriptDir)
$r2Exe     = Join-Path $repoRoot "target\release\r2.exe"

if (-not (Test-Path $r2Exe)) {
    Write-Error "r2.exe not found at $r2Exe. Run 'cargo build --release' first."
}
if (-not (Get-Command Rscript -ErrorAction SilentlyContinue)) {
    Write-Error "Rscript not on PATH. Install CRAN R or add it to PATH."
}

function Parse-KeyValues($lines) {
    $h = @{}
    foreach ($l in $lines) {
        if ($l -match '^\s*\[?\d*\]?\s*"?([\w\.]+)=([^"]+?)"?\s*$') {
            $h[$matches[1]] = $matches[2]
        }
    }
    return $h
}

function Run-Pair($name, $rScript, $r2Script) {
    Write-Host ""
    Write-Host "═══ $name ═══" -ForegroundColor Cyan
    Write-Host "Running R: $rScript"
    $rOut  = & Rscript --vanilla $rScript 2>$null
    Write-Host "Running R2: $r2Script"
    $r2Out = & $r2Exe $r2Script 2>$null

    $rMap  = Parse-KeyValues $rOut
    $r2Map = Parse-KeyValues $r2Out

    $allKeys = ($rMap.Keys + $r2Map.Keys) | Sort-Object -Unique
    "{0,-30} {1,-22} {2,-22} {3,-12}" -f "key", "R", "R2", "delta"
    "{0,-30} {1,-22} {2,-22} {3,-12}" -f ("-"*30), ("-"*22), ("-"*22), ("-"*12)
    foreach ($k in $allKeys) {
        $rv  = if ($rMap.ContainsKey($k))  { $rMap[$k]  } else { "<missing>" }
        $r2v = if ($r2Map.ContainsKey($k)) { $r2Map[$k] } else { "<missing>" }
        $delta = ""
        $rNum  = 0.0
        $r2Num = 0.0
        if ([double]::TryParse($rv,  [ref]$rNum) -and [double]::TryParse($r2v, [ref]$r2Num)) {
            $abs_d = [math]::Abs($rNum - $r2Num)
            if ([math]::Abs($rNum) -gt 1e-12) {
                $rel = $abs_d / [math]::Abs($rNum)
                $delta = "{0:G4} ({1:P2})" -f $abs_d, $rel
            } else {
                $delta = "{0:G4}" -f $abs_d
            }
        }
        "{0,-30} {1,-22} {2,-22} {3,-12}" -f $k, $rv, $r2v, $delta
    }
}

Run-Pair "ACCURACY"   (Join-Path $scriptDir "accuracy.R")   (Join-Path $scriptDir "accuracy.r2")
Run-Pair "PERFORMANCE (seconds, lower is better)" (Join-Path $scriptDir "performance.R") (Join-Path $scriptDir "performance.r2")

Write-Host ""
Write-Host "Done. For accuracy, deltas under 1e-3 are typically within R2's stated tolerance."
Write-Host "For performance, R's matrix-multiply numbers depend on which BLAS R is linked to."
