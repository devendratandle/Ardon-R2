# ============================================================
#  Ardon-R2 v0.2.2 verification script
#  Paste into the R2Gui console (best — exercises interactive
#  multi-line entry) or run: r2 test_v0_2_2.r2
# ============================================================

# --- 1. Multi-line c() with inline comments (parser fix) ---
set.seed(123)
subject  <- factor(rep(1:10, each = 3))
time     <- factor(rep(c("T1","T2","T3"), times = 10))
response <- c(
  rnorm(10, mean = 50, sd = 5),   # T1
  rnorm(10, mean = 55, sd = 5),   # T2
  rnorm(10, mean = 60, sd = 5)    # T3
)
data <- data.frame(subject, time, response)
cat("1) response:", length(response), " data:", nrow(data), "x", ncol(data),
    "  (expect 30 / 30 x 3)\n")

# --- 2. Stats output VISIBLE (no <htest model>, no trailing NULL) ---
t.test(Sepal.Width ~ Species, data = iris[1:100, ])
chisq.test(matrix(c(200,400,300,500), byrow = TRUE, nrow = 2))
aov(Sepal.Length ~ Species, data = iris)
summary(iris)
str(iris)

# --- 3. Accuracy ---
cat("3) qnorm(0.975)=", qnorm(0.975), " (R: 1.959964)\n")
A <- matrix(c(4,1,1,3), nrow = 2)
cat("   det(A)=", det(A), " (expect 11)\n")
print(solve(A) %*% A)
print(lm(Sepal.Length ~ Sepal.Width, data = iris)$p.values)

# --- 4. Fusion correctness ---
v <- as.numeric(1:100)
cat("4) fusion diff (expect 0):", max(abs((v*2+1) - ((v*2)+1))), "\n")

# --- 5. Out-of-core arrow ---
mmap.write(as.numeric(1:1000), "ooc_test.bin")
m <- mmap.col("ooc_test.bin")
cat("5) mmap sum=", sum(m), " mean=", mean(m), " length=", m$length,
    "  (expect 500500 / 500.5 / 1000)\n")

# --- 6. Graphics device (GUI window / CLI browser / script .svg) ---
hist(rnorm(1000), col = "red")
plot(1:10, (1:10)^2)

cat("\n=== script done; clear() next (interactive) ===\n")
