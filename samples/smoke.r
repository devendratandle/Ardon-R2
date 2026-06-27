# ─────────────────────────────────────────────────────────────────────
# Ardon-R2 smoke test — runs on BOTH R and R2 with identical code.
#
#   r2      samples/smoke.r      # run on Ardon-R2
#   Rscript samples/smoke.r      # run on CRAN R  (cross-check)
#
# Compare the two outputs side by side: the numbers should match to ~7
# significant figures, and the [time] lines show R2 vs R wall-clock.
# If you see a mismatch or an error, please report it (issue tracker).
# ─────────────────────────────────────────────────────────────────────

# NOTE: timing uses `system.time(EXPR)` directly (not wrapped in a helper)
# because R2 evaluates arguments eagerly — a helper would time nothing.
ok <- function(label, cond) cat(if (isTRUE(cond)) "  [PASS] " else "  [FAIL] ", label, "\n", sep = "")

cat("=== 1. Vectors & descriptive statistics ===\n")
set.seed(1)
x <- rnorm(1000, mean = 10, sd = 3)
cat(sprintf("  mean=%.5f sd=%.5f median=%.5f\n", mean(x), sd(x), median(x)))
ok("var == sd^2", abs(var(x) - sd(x)^2) < 1e-9)
ok("quantile median == median", abs(quantile(x, 0.5)[[1]] - median(x)) < 1e-9)

cat("\n=== 2. Data frame operations ===\n")
df <- data.frame(g = rep(c("a", "b", "c"), each = 50),
                 y = c(rnorm(50, 1), rnorm(50, 5), rnorm(50, 9)))
agg <- aggregate(y ~ g, data = df, FUN = mean)
cat("  group means:", round(agg$y, 3), "\n")
ok("3 groups aggregated", nrow(agg) == 3)

cat("\n=== 3. Linear & generalized linear models ===\n")
fit <- lm(Sepal.Length ~ Petal.Length + Petal.Width, data = iris)
cat("  lm coef:", round(coef(fit), 5), "\n")
iris2 <- iris
iris2$virginica <- as.numeric(iris$Species == "virginica")
g <- glm(virginica ~ Petal.Width, data = iris2, family = "binomial")
cat("  glm coef:", round(coef(g), 4), "\n")

cat("\n=== 4. Hypothesis tests ===\n")
tt <- t.test(iris$Sepal.Length[1:50], iris$Sepal.Length[51:100])
cat(sprintf("  Welch t = %.4f, df = %.3f\n", tt$statistic, tt$parameter))
cat(sprintf("  cor(Petal.L, Petal.W) = %.6f\n",
            cor(iris$Petal.Length, iris$Petal.Width)))

cat("\n=== 5. Linear algebra ===\n")
A <- matrix(c(2, 0, 0, 0, 3, 0, 0, 0, 6), 3, 3)
cat("  eigenvalues:", sort(round(eigen(A)$values, 5)), "\n")
B <- matrix(rnorm(9), 3, 3)
ok("solve(B) %*% B == I", max(abs(solve(B) %*% B - diag(3))) < 1e-9)
s <- svd(matrix(rnorm(200 * 50), 200, 50))
ok("svd: 50 singular values", length(s$d) == 50)

cat("\n=== 6. Performance — wall-clock (compare R vs R2 elapsed) ===\n")
big <- rnorm(1e6)
cat("  sort 1e6:\n");       system.time(sort(big))
cat("  sum 1e7:\n");        system.time(sum(rnorm(1e7)))
Mmul <- matrix(rnorm(500 * 500), 500, 500)
cat("  matmul 500x500:\n"); system.time(Mmul %*% Mmul)

cat("\nSMOKE TEST DONE — compare this output against the other engine.\n")
