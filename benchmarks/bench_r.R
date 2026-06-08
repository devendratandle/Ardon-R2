# R-side benchmark (run with Rscript). Mirrors bench_r2.R.
cat("R", as.character(getRversion()),
    "| BLAS:", tryCatch(La_library(), error=function(e) "?"), "\n\n")

# --- deterministic correctness (fixed inputs; must match R2 exactly) ---
Ad <- matrix(c(1,2,3,4,5,6,7,8,9), 3, 3)
cat(sprintf("det matmul sum   : %.1f\n", sum(Ad %*% Ad)))
cat(sprintf("det crossprod sum: %.1f\n\n", sum(crossprod(Ad))))

# --- performance (random; compare wall-clock to R2) ---
set.seed(1)
n <- 1024
A <- matrix(rnorm(n*n), n, n); B <- matrix(rnorm(n*n), n, n)
t1 <- system.time(C <- A %*% B)[3]
cat(sprintf("matmul 1024x1024 : %.4f s   %.2f GFLOP/s\n", t1, 2*n^3/t1/1e9))

m <- 100000L; p <- 50L
X <- matrix(rnorm(m*p), m, p)
t2 <- system.time(G <- crossprod(X))[3]
cat(sprintf("crossprod 100kx50: %.4f s\n", t2))

v <- rnorm(5e7)
t3 <- system.time(s <- sum(v))[3]
cat(sprintf("sum 5e7          : %.4f s\n", t3))

y <- 2 + 3*X[,1] - 1.5*X[,2]
df <- data.frame(y=y, x1=X[,1], x2=X[,2])
t4 <- system.time(fit <- lm(y ~ x1 + x2, df))[3]
cat(sprintf("lm 100kx2        : %.4f s   coef=%s\n", t4,
            paste(round(coef(fit),5), collapse=" ")))
