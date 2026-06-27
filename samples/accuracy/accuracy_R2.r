# Numerical accuracy + benchmark — R2 version
# Run in R2: source("E:/R2_Rust _opus4.6/r2/r2/samples/accuracy_R2.r")
#
# Note: R2's sprintf is minimal (no %g specifier), so this version uses
# multi-arg cat() to print labeled values. Precision is bounded by R2's
# fmt_num (~7 significant digits). For 17-digit comparison vs R, see
# accuracy_R.r and compare visually.

cat("=== R2 numerical accuracy + benchmark ===\n\n")

# ── 1. Deterministic test data — identical to accuracy_R.r ──
x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
y <- c(2.1, 4.3, 6.05, 8.2, 9.95, 12.1, 14.05, 16.2, 18.1, 19.9)
big <- (1:1000000) / 1000

ax <- c(10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5)
ay <- c(8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68)

cat("--- 1. Reductions on small vector x[1..10] ---\n")
cat("sum(x)    =", sum(x), "\n")
cat("mean(x)   =", mean(x), "\n")
cat("var(x)    =", var(x), "\n")
cat("sd(x)     =", sd(x), "\n")
cat("min(x)    =", min(x), "\n")
cat("max(x)    =", max(x), "\n")
cat("prod(x)   =", prod(x), "\n")
cat("median(x) =", median(x), "\n")

cat("\n--- 2. Two-vector statistics on (x, y) ---\n")
cat("cor(x, y) =", cor(x, y), "\n")
cat("cov(x, y) =", cov(x, y), "\n")

cat("\n--- 3. Linear regression y ~ x ---\n")
fit <- lm(y ~ x)
co <- coef(fit)
cat("intercept =", co[1], "\n")
cat("slope     =", co[2], "\n")
if (!is.null(fit$r.squared)) {
  cat("r.squared =", fit$r.squared, "\n")
}

cat("\n--- 4. Anscombe quartet ---\n")
cat("mean(ax)   =", mean(ax),   " (expected 9)\n")
cat("var(ax)    =", var(ax),    " (expected 11)\n")
cat("cor(ax,ay) =", cor(ax, ay), " (expected ~0.81642)\n")

cat("\n--- 5. Reductions on 1M-element 'big' (precision check) ---\n")
cat("sum(big)  =", sum(big),  " (analytic 500000500)\n")
cat("mean(big) =", mean(big), " (analytic 500.0005)\n")
cat("max(big)  =", max(big),  " (analytic 1000)\n")

cat("\n--- 6. Benchmarks (1M-element vector, 10 reps each) ---\n")
t0 <- Sys.time()
for (i in 1:10) { s <- sum(big) }
t1 <- Sys.time()
cat("sum(big)  : ", (t1 - t0) / 10, " s/op\n")

t0 <- Sys.time()
for (i in 1:10) { m <- mean(big) }
t1 <- Sys.time()
cat("mean(big) : ", (t1 - t0) / 10, " s/op\n")

t0 <- Sys.time()
for (i in 1:10) { d <- sd(big) }
t1 <- Sys.time()
cat("sd(big)   : ", (t1 - t0) / 10, " s/op\n")

t0 <- Sys.time()
big2 <- big * 2 + 1
for (i in 1:5) { r <- cor(big, big2) }
t1 <- Sys.time()
cat("cor       : ", (t1 - t0) / 5, " s/op\n")

cat("\n=== End of R2 run ===\n")
