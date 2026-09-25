# The same user-level linear-algebra calls, timed in R and in R2 on the
# same deterministic 1000 x 1000 matrices (no RNG, so both engines factor
# identical inputs). Median of 5 runs, milliseconds.
#   Rscript benchmarks/linalg_vs_r.R
#   r2 benchmarks/linalg_vs_r.R
n <- 1000
A <- matrix(sin(1:(n * n)), n, n)
S <- crossprod(A) + diag(n)          # symmetric positive definite
med_ms <- function(f) {
  t <- numeric(5)
  for (i in 1:5) t[i] <- system.time(f())[["elapsed"]] * 1000
  median(t)
}
# A call the engine lacks reports NA instead of stopping the run.
med_ms_or_na <- function(f) tryCatch(med_ms(f), error = function(e) NA)
cat("matmul=", med_ms_or_na(function() A %*% A), "\n", sep = "")
cat("solve=", med_ms_or_na(function() solve(S)), "\n", sep = "")
cat("chol=", med_ms_or_na(function() chol(S)), "\n", sep = "")
cat("qr=", med_ms_or_na(function() qr(A)), "\n", sep = "")
cat("eigen_sym=", med_ms_or_na(function() eigen(S, symmetric = TRUE)), "\n", sep = "")
cat("svd_values=", med_ms_or_na(function() svd(A, nu = 0, nv = 0)), "\n", sep = "")
cat("svd_full=", med_ms_or_na(function() svd(A)), "\n", sep = "")
