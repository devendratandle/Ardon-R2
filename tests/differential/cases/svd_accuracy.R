# SVD accuracy on ill-conditioned matrices. The small singular values are
# compared as 4-significant-digit STRINGS ("s1.093e-13"): the harness's
# absolute floor (1e-12) would pass any error on values that small, and a
# method that squares the condition number is off in the 2nd digit here.
kv <- function(k, v) cat(k, "=", v, "\n", sep = "")
h <- 1 / (outer(1:10, 1:10, "+") - 1)
s <- svd(h)
for (i in 1:10) kv(sprintf("hilb.d%02d", i), paste0("s", sprintf("%.3e", s$d[i])))
kv("hilb.d01.full", s$d[1])
kv("hilb.rebuild.ok", max(abs(s$u %*% diag(s$d) %*% t(s$v) - h)) < 1e-12)
kv("hilb.u.orth.ok", max(abs(crossprod(s$u) - diag(10))) < 1e-12)
kv("hilb.v.orth.ok", max(abs(crossprod(s$v) - diag(10))) < 1e-12)
x <- seq(0.1, 1, length.out = 8)
v <- outer(x, 0:7, "^")
dv <- svd(v, nu = 0, nv = 0)$d
for (i in 1:8) kv(sprintf("vand.d%02d", i), paste0("s", sprintf("%.3e", dv[i])))
