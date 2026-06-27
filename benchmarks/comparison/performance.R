# R-side performance comparison script.
# Each block reports wall-clock seconds in `BENCH=seconds` format.
#
# Usage:  Rscript performance.R > perf_R.txt
# All sizes chosen so a single block runs in 0.05–5 seconds on a 2024 laptop;
# bump N upward by 10× for stress runs.
#
# Notes:
# - Set the same RNG seed at the top so random sizes match across runs.
# - We deliberately call `system.time(...)` per block (real wall-clock).
# - `set.seed` is reseeded inside each randomness-using block so blocks
#   are order-independent.

options(digits = 6)

# ── Warmup pass ──────────────────────────────────────────────────────
# Matches the warmup in `performance.r2` so the comparison times the
# same workload. R doesn't need warmup, but we run it here to keep the
# two scripts symmetric.
set.seed(0)
.warm_a <- rnorm(8); .warm_b <- rnorm(8)
.x <- .warm_a + .warm_b
.s <- sum(.warm_a); .m <- mean(.warm_a); .v <- sort(.warm_a)
.M <- matrix(.warm_a, 4, 2); .C <- .M %*% t(.M)
.df <- data.frame(y = .warm_a, x1 = .warm_b)
.fit <- lm(y ~ x1, data = .df)
.km <- kmeans(matrix(rnorm(40), 8, 5), centers = 2)
.r <- sapply(1:2, function(i) mean(.warm_a))
.sv <- svd(matrix(rnorm(20), 5, 4))

# 1. Element-wise: 1e7 vector add (cold-cache + warm)
n <- 1e7
set.seed(1); a <- rnorm(n); b <- rnorm(n)
t1 <- system.time({ s <- a + b; sum_s <- sum(s) })
cat(sprintf("vec_add_1e7=%.4f\n", t1[3]))

# 2. Sum + mean on 1e7
t1 <- system.time({ s <- sum(a); m <- mean(a) })
cat(sprintf("sum_mean_1e7=%.4f\n", t1[3]))

# 3. Sort on 1e6 doubles
set.seed(2); v <- rnorm(1e6)
t1 <- system.time({ vs <- sort(v) })
cat(sprintf("sort_1e6=%.4f\n", t1[3]))

# 4. Matrix multiply 500×500  (size kept moderate; R may link to MKL/OpenBLAS)
set.seed(3); A <- matrix(rnorm(500*500), 500, 500); B <- matrix(rnorm(500*500), 500, 500)
t1 <- system.time({ C <- A %*% B })
cat(sprintf("matmul_500x500=%.4f\n", t1[3]))

# 5. Linear regression on 1e5 × 5
n <- 1e5
set.seed(4)
X <- matrix(rnorm(n * 5), n, 5)
y <- X %*% c(1, -0.5, 0.2, 0.1, 0.05) + rnorm(n)
df <- data.frame(y = as.numeric(y), x1 = X[,1], x2 = X[,2], x3 = X[,3], x4 = X[,4], x5 = X[,5])
t1 <- system.time({ fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = df) })
cat(sprintf("lm_1e5x5=%.4f\n", t1[3]))

# 6. K-means on 1e5 × 10, k=5
set.seed(5)
M <- matrix(rnorm(1e5 * 10), 1e5, 10)
t1 <- system.time({ km <- kmeans(M, centers = 5, nstart = 1, iter.max = 20) })
cat(sprintf("kmeans_1e5x10_k5=%.4f\n", t1[3]))

# 7. Sapply on iris × scalar function over 30 reps
t1 <- system.time({
  res <- sapply(1:30, function(i) mean(iris$Sepal.Length * i))
})
cat(sprintf("sapply_iris_30=%.4f\n", t1[3]))

# 8. SVD on 200×100
set.seed(6); M2 <- matrix(rnorm(200 * 100), 200, 100)
t1 <- system.time({ s <- svd(M2) })
cat(sprintf("svd_200x100=%.4f\n", t1[3]))
