# R2-side benchmark (run with target/release/r2). Mirrors bench_r.R.

# --- deterministic correctness (must match R exactly) ---
Ad <- matrix(c(1,2,3,4,5,6,7,8,9), 3, 3)
cat("det matmul sum   :", sum(Ad %*% Ad), "\n")
cat("det crossprod sum:", sum(crossprod(Ad)), "\n\n")

# --- performance (random; compare wall-clock to R) ---
set.seed(1)
n <- 1024
A <- matrix(rnorm(n*n), n, n); B <- matrix(rnorm(n*n), n, n)
C <- A %*% B
cat("matmul 1024x1024 :\n"); system.time(A %*% B)

m <- 100000; p <- 50
X <- matrix(rnorm(m*p), m, p)
G <- crossprod(X)
cat("crossprod 100kx50:\n"); system.time(crossprod(X))

v <- rnorm(5e7)
cat("sum 5e7          :\n"); system.time(sum(v))

y <- 2 + 3*X[,1] - 1.5*X[,2]
df <- data.frame(y=y, x1=X[,1], x2=X[,2])
cat("lm 100kx2        :\n"); system.time(lm(y ~ x1 + x2, data=df))
cat("lm coef:", coef(lm(y ~ x1 + x2, data=df)), "\n")
