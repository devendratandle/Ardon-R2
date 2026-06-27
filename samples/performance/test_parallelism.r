# R2 parallelism + JIT test script
# Run by pasting blocks into the REPL, OR `source("samples/test_parallelism.r")`.
# All operations stay correct on small data; large data triggers Oracle parallel paths.

cat("\n=== 1. Generate simulated data ===\n")
set.seed(42)
n <- 500000
x <- rnorm(n, mean = 10, sd = 2)
y <- rnorm(n, mean = 20, sd = 3)
cat("Generated 2 vectors of length", n, "\n")

cat("\n=== 2. Tier A reductions (Oracle: parallel above ~200K) ===\n")
cat("sum(x)   = "); system.time(s1 <- sum(x));   cat("  ->", s1, "\n")
cat("mean(x)  = "); system.time(m1 <- mean(x));  cat("  ->", m1, "\n")
cat("sd(x)    = "); system.time(s2 <- sd(x));    cat("  ->", s2, "\n")
cat("var(x)   = "); system.time(v1 <- var(x));   cat("  ->", v1, "\n")
cat("min(x)   = "); system.time(mn <- min(x));   cat("  ->", mn, "\n")
cat("max(x)   = "); system.time(mx <- max(x));   cat("  ->", mx, "\n")

cat("\n=== 3. JIT-compiled user functions ===\n")
# Scalar JIT (Phase C.2)
poly <- function(a, b) a*a + 2*a*b + b*b + 1
cat("poly(3, 5) -> "); print(poly(3, 5))   # = 65, runs as native code

# Vector reduction JIT (Phase C.3) — body is sum(v), compiles via extern
my_sum <- function(v) sum(v)
cat("my_sum(x) = "); system.time(r <- my_sum(x)); cat("  ->", r, "\n")

# Element-wise scalar map JIT (Phase C.4-mini)
add_one <- function(v) v + 1
cat("add_one(x[1:5]) -> "); print(add_one(x[1:5]))

# Composed expression JIT (Phase C.4-full part 2)
quad <- function(v) (v - 5) * (v - 5) + 1
cat("quad(c(0, 1, 2, 3)) -> "); print(quad(c(0, 1, 2, 3)))

# Vector + Vector (Phase C.4-full)
vadd <- function(a, b) a + b
cat("vadd(x[1:5], y[1:5]) -> "); print(vadd(x[1:5], y[1:5]))

cat("\n=== 4. Parallel summary on a data frame (per-column fan-out) ===\n")
df <- data.frame(
  height = rnorm(50000, 170, 10),
  weight = rnorm(50000, 70, 12),
  score  = rnorm(50000, 50, 15),
  iq     = rnorm(50000, 100, 15)
)
cat("summary(df):\n"); system.time(summary(df))

cat("\n=== 5. K-means on large matrix (parallel point-to-centroid) ===\n")
M <- matrix(rnorm(40000), nrow = 10000, ncol = 4)
cat("kmeans(M, centers = 3):\n"); system.time(km <- kmeans(M, centers = 3))

cat("\n=== 6. Cross-validation (parallel folds) ===\n")
xm <- matrix(rnorm(2000), nrow = 200, ncol = 10)
yv <- xm[, 1] * 2 + xm[, 2] * -1 + rnorm(200, 0, 0.5)
cat("cv(xm, yv, model = 'lm', k = 5):\n"); system.time(cv(xm, yv, model = "lm", k = 5))

cat("\n=== Done ===\n")
cat("Notes:\n")
cat("  - elapsed times above include Oracle's serial-vs-parallel choice.\n")
cat("  - Run with R2_JIT=0 (env var) to disable JIT and compare.\n")
cat("  - Set seed differs between runs; values vary slightly.\n\n")
