# R2-side performance comparison script.
# Pair with `performance.R`. Each block reports `BENCH=seconds`.
#
# Usage:  r2 performance.r2 > perf_R2.txt
# Or:     target/release/r2 bench/r_vs_r2/performance.r2

# Helper: tic/toc using Sys.time().
tic <- function() Sys.time()
toc <- function(t0) as.numeric(Sys.time() - t0)
emit <- function(key, sec) cat(key, "=", sec, "\n", sep = "")

# ── Warmup pass ──────────────────────────────────────────────────────
# R2 builtins may compile / cache state on first invocation, which
# inflates first-call timings. Call each measured builtin once with a
# trivial input so the steady-state cost is what gets measured below.
# (R has no equivalent need but adding a warmup section to R as well
# keeps the comparison apples-to-apples.)
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

# Each block clears its allocations via `rm("name")` afterwards so the
# next block starts from a clean heap. Without this, prior R2 vectors
# stay in the global env and inflate subsequent timings ~10× through
# heap fragmentation pressure. R has GC that handles this implicitly.

# 1. Element-wise: 1e7 vector add
n <- 1e7
set.seed(1); a <- rnorm(n); b <- rnorm(n)
t0 <- tic(); s <- a + b; sum_s <- sum(s); emit("vec_add_1e7", toc(t0))
rm("a"); rm("b"); rm("s"); rm("sum_s")

# 2. Sum + mean on 1e7
set.seed(11); a <- rnorm(1e7)
t0 <- tic(); s <- sum(a); m <- mean(a); emit("sum_mean_1e7", toc(t0))
rm("a"); rm("s"); rm("m")

# 3. Sort on 1e6 doubles
set.seed(2); v <- rnorm(1e6)
t0 <- tic(); vs <- sort(v); emit("sort_1e6", toc(t0))
rm("v"); rm("vs")

# 4. Matrix multiply 500×500
set.seed(3)
A <- matrix(rnorm(500 * 500), 500, 500)
B <- matrix(rnorm(500 * 500), 500, 500)
t0 <- tic(); C <- A %*% B; emit("matmul_500x500", toc(t0))
rm("A"); rm("B"); rm("C")

# 5. Linear regression on 1e5 × 5
n <- 1e5
set.seed(4)
x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n); x4 <- rnorm(n); x5 <- rnorm(n)
y <- 1.0 * x1 - 0.5 * x2 + 0.2 * x3 + 0.1 * x4 + 0.05 * x5 + rnorm(n)
df <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5)
t0 <- tic(); fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = df); emit("lm_1e5x5", toc(t0))
rm("x1"); rm("x2"); rm("x3"); rm("x4"); rm("x5"); rm("y"); rm("df"); rm("fit")

# 6. K-means on 1e5 × 10, k=5
set.seed(5)
M <- matrix(rnorm(1e5 * 10), 1e5, 10)
t0 <- tic(); km <- kmeans(M, centers = 5); emit("kmeans_1e5x10_k5", toc(t0))
rm("M"); rm("km")

# 7. Sapply iris × scalar function over 30 reps
t0 <- tic()
res <- sapply(1:30, function(i) mean(iris$Sepal.Length * i))
emit("sapply_iris_30", toc(t0))
rm("res")

# 8. SVD on 200×100
set.seed(6); M2 <- matrix(rnorm(200 * 100), 200, 100)
t0 <- tic(); s <- svd(M2); emit("svd_200x100", toc(t0))
rm("M2"); rm("s")
