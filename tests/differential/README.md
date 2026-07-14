# Differential correctness harness (r2 vs GNU R)

Runs every script in `cases/` under both Ardon-R2 and GNU R and compares
their `key=value` outputs. Exit code 0 only if every case passes.

## Run

```sh
tests/differential/run.sh              # all cases
tests/differential/run.sh matrix       # cases whose name contains "matrix"
R2_BIN=../../target/debug/r2.exe tests/differential/run.sh   # debug binary
RSCRIPT=/usr/bin/Rscript tests/differential/run.sh           # other R
```

Defaults: `../../target/release/r2.exe` and
`C:/R/R-4.5.3/bin/x64/Rscript.exe` (paths relative to this directory).

## Writing a case

One `.R` file in `cases/`, valid in both engines, printing only
deterministic `key=value` lines:

```r
cat("mykey=", some_value, "\n", sep = "")
```

- Numeric values compare with relative tolerance 1e-9 (absolute 1e-12
  near zero); strings compare exactly.
- A key emitted by only one engine fails the case; an r2 error that
  aborts a script shows as a cascade of "R-only key" entries — the first
  missing key marks the failure point.
- No RNG-dependent values (the rnorm/runif streams differ between
  engines by design). Derive structural facts — lengths, dims, classes —
  from random data instead.
- Emit booleans as 0/1 (`if (cond) 1 else 0`).
