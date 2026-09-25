# User-level linear algebra against R: solve (inverse, matrix and vector
# right-hand sides), chol, svd(nu = 0, nv = 0), and svd's field order.
# n = 150 is past the 64-row base case, so the recursive GEMM solves run.
n <- 150
A <- matrix(sin(1:(n * n)), n, n)
S <- crossprod(A) + diag(n)
b <- cos(1:n)
B <- matrix(cos(1:(n * 3)), n, 3)
kv <- function(k, v) cat(k, "=", v, "\n", sep = "")

Si <- solve(S)
kv("inv.residual.ok", max(abs(Si %*% S - diag(n))) < 1e-9)
kv("inv.11", Si[1, 1])
kv("inv.sum", sum(Si))
x <- solve(S, b)
kv("solve.vec.1", x[1])
kv("solve.vec.sum", sum(x))
X <- solve(A, B)
kv("solve.mat.residual.ok", max(abs(A %*% X - B)) < 1e-8)
kv("solve.mat.sum", sum(X))

R <- chol(S)
kv("chol.upper", sum(abs(R[lower.tri(R)])) == 0)
kv("chol.rebuild.ok", max(abs(t(R) %*% R - S)) < 1e-8)
kv("chol.11", R[1, 1])
kv("chol.nn", R[n, n])

d <- svd(A, nu = 0, nv = 0)
kv("svd.values.names", paste(names(d), collapse = ","))
kv("svd.d1", d$d[1])
kv("svd.dn", d$d[n])
kv("svd.names", paste(names(svd(A)), collapse = ","))
