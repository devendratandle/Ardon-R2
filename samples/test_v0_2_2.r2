# ================================================================
#  Ardon-R2 — self-test script for the v0.2.2 fixes
#  Run in:  R2Gui (paste), CLI interactive (r2), or CLI (r2 this.r2)
# ================================================================

cat("===== 1. STATS OUTPUT (must be VISIBLE — no <htest model>, no NULL) =====\n")

# Welch t-test, formula form (resolves columns from data=)
t.test(Sepal.Width ~ Species, data = iris[1:100,])

# Two-vector t-test
t.test(c(5.1,4.9,4.7,4.6,5.0), c(6.2,5.9,6.1,6.3,5.8))

# Chi-squared
chisq.test(matrix(c(200,400,300,500), byrow=TRUE, nrow=2))

# ANOVA table
aov(Sepal.Length ~ Species, data = iris)

# Inspectors — these used to print a trailing "NULL"; should be clean now
summary(iris)
str(iris)

cat("\n===== 2. ACCURACY (compare to R) =====\n")

cat("qnorm(0.975) — R: 1.959964 ->", qnorm(0.975), "\n")
cat("lm coefficient p-values (t-distribution):\n")
print(lm(Sepal.Length ~ Sepal.Width, data = iris)$p.values)

A <- matrix(c(4,1,1,3), nrow = 2)
cat("det(A) — should be 11 ->", det(A), "\n")
cat("solve(A) %*% A — should be the identity matrix:\n")
print(solve(A) %*% A)

cat("\n===== 3. GRAPHICS DEVICE =====\n")
cat("GUI: each plot opens in the Graphics window.\n")
cat("CLI interactive: first plot opens the browser viewer.\n")
cat("CLI script (r2 this.r2): plots are saved as .svg files (no popup).\n")

hist(rnorm(1000), col = "red")
plot(1:10, (1:10)^2)
boxplot(rnorm(100))

cat("\n===== done =====\n")
