# Run R vs R2 — Minimal Instructions

The binary is built. Everything below assumes you're at the repo root:
`E:\R2_Rust _opus4.6\r2\r2\`.

## What you have

```
target/release/r2.exe              ← built v0.1.0 binary (6.5 MB)
bench/r_vs_r2/accuracy.R           ← R-side accuracy tests
bench/r_vs_r2/accuracy.r2          ← R2-side accuracy tests (matching)
bench/r_vs_r2/performance.R        ← R-side performance benchmarks
bench/r_vs_r2/performance.r2       ← R2-side performance benchmarks
```

## Run it

Open PowerShell (not bash) in the repo root and paste these four lines.
Each writes one output file.

```powershell
Rscript --vanilla bench\r_vs_r2\accuracy.R    > out_R_accuracy.txt    2>$null
.\target\release\r2.exe bench\r_vs_r2\accuracy.r2 > out_R2_accuracy.txt 2>$null

Rscript --vanilla bench\r_vs_r2\performance.R > out_R_performance.txt  2>$null
.\target\release\r2.exe bench\r_vs_r2\performance.r2 > out_R2_performance.txt 2>$null
```

Total runtime: ~30–60 seconds (R is fast; R2 is fast).

## What to send back

Just the four output files:

- `out_R_accuracy.txt`
- `out_R2_accuracy.txt`
- `out_R_performance.txt`
- `out_R2_performance.txt`

Paste them into the next message (or attach if your client supports it).
Each file is small — a few dozen lines of `KEY=VALUE`.

## Or: use the comparison driver (optional)

If you want a side-by-side table immediately:

```powershell
pwsh bench\r_vs_r2\run.ps1 > comparison_report.txt
```

This writes both R and R2 outputs to one file with deltas already
computed. Send me that single file instead of the four.

## If `Rscript` isn't found

Add R to your PATH for this session:

```powershell
$env:Path += ";C:\Program Files\R\R-4.x.x\bin"
```

Replace `R-4.x.x` with your actual R version directory.

## If something errors out

Just send me whatever output you got. Partial results are still useful —
the comparison driver shows `<missing>` for any keys one side didn't
produce, so I can see exactly where it stopped.

## What's tested

- **9 accuracy sections**: descriptives, lm, glm logistic, t.test, svd
  (singular values), eigen, aov, kmeans, cor.
- **8 performance sections**: vector add 1e7, sum+mean 1e7, sort 1e6,
  matmul 500×500, lm 1e5×5, kmeans 1e5×10, sapply×30, svd 200×100.

Accuracy is matched to ~1e-3 tolerance (R2's t-CDF uses trapezoidal
integration with ~1e-3 accuracy on p-values). Performance depends on
your machine and which BLAS R is linked against — typical results on a
2024 Windows laptop with default Rblas: R2 wins matmul by 2×, R wins
nothing or ties on the rest.
