# Numerical accuracy + benchmark — R version
# Run in plain R: source("samples/accuracy_R.r")
# Output: high-precision values + timings, side-by-side comparable to accuracy_R2.r.

options(digits = 17)            # show full f64 precision
cat("=== R numerical accuracy + benchmark ===\n\n")

# ── 1. Deterministic test data (no RNG → identical across runs and tools) ──
x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
y <- c(2.1, 4.3, 6.05, 8.2, 9.95, 12.1, 14.05, 16.2, 18.1, 19.9)
big <- (1:1000000) / 1000      # 1M deterministic doubles, range 0.001..1000

# Anscombe's quartet point — known mean=7.5, var=4.12, cor with y_q ≈ 0.816
ax <- c(10,8,13,9,11,14,6,4,12,7,5)
ay <- c(8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68)

cat("--- 1. Reductions on small vector x[1..10] ---\n")
cat(sprintf("sum(x)    = %.17g\n", sum(x)))
cat(sprintf("mean(x)   = %.17g\n", mean(x)))
cat(sprintf("var(x)    = %.17g\n", var(x)))
cat(sprintf("sd(x)     = %.17g\n", sd(x)))
cat(sprintf("min(x)    = %.17g\n", min(x)))
cat(sprintf("max(x)    = %.17g\n", max(x)))
cat(sprintf("prod(x)   = %.17g\n", prod(x)))
cat(sprintf("median(x) = %.17g\n", median(x)))

cat("\n--- 2. Two-vector statistics on (x, y) ---\n")
cat(sprintf("cor(x, y) = %.17g\n", cor(x, y)))
cat(sprintf("cov(x, y) = %.17g\n", cov(x, y)))

cat("\n--- 3. Linear regression y ~ x ---\n")
fit <- lm(y ~ x)
cat(sprintf("intercept = %.17g\n", coef(fit)[1]))
cat(sprintf("slope     = %.17g\n", coef(fit)[2]))
cat(sprintf("r.squared = %.17g\n", summary(fit)$r.squared))

cat("\n--- 4. Anscombe quartet (known answers: mean=7.5, var=4.12) ---\n")
cat(sprintf("mean(ax) = %.17g (expected 9)\n", mean(ax)))
cat(sprintf("var(ax)  = %.17g (expected 11)\n", var(ax)))
cat(sprintf("cor(ax,ay) = %.17g (expected 0.816420516)\n", cor(ax, ay)))

cat("\n--- 5. Reductions on 1M-element 'big' (precision check) ---\n")
cat(sprintf("sum(big)  = %.17g (analytic %.17g)\n", sum(big), 1000000 * 1000.001 / 2))
cat(sprintf("mean(big) = %.17g (analytic 500.0005)\n", mean(big)))
cat(sprintf("max(big)  = %.17g (1000)\n", max(big)))

cat("\n--- 6. Benchmarks (1M-element vector) ---\n")
t1 <- system.time(for(i in 1:10) sum(big))[3] / 10
cat(sprintf("sum(big)  : %.6f s/op\n", t1))
t2 <- system.time(for(i in 1:10) mean(big))[3] / 10
cat(sprintf("mean(big) : %.6f s/op\n", t2))
t3 <- system.time(for(i in 1:10) sd(big))[3] / 10
cat(sprintf("sd(big)   : %.6f s/op\n", t3))
t4 <- system.time(for(i in 1:5) cor(big, big * 2 + 1))[3] / 5
cat(sprintf("cor       : %.6f s/op\n", t4))

cat("\n=== End of R run ===\n")
