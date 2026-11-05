# qr() and its accessors against R: the compact form itself, not only the
# products, because R documents $qr, $qraux, $rank and $pivot.
p <- function(k, v) for (i in seq_along(v)) cat(k, ".", i, "=", sprintf("%.15g", v[i]), "\n", sep = "")

# Past the rank, LINPACK goes on reflecting what is only rounding error: the
# columns it moved aside hold noise below their leading rows, in R as here. They
# are read through qr.R, whose values there are near zero.
show <- function(k, q) {
  d <- dim(q$qr)
  keep <- if (q$rank == min(d)) seq_len(d[2]) else seq_len(q$rank)
  if (length(keep) > 0) { p(paste0(k, ".qr"), q$qr[, keep]); p(paste0(k, ".aux"), q$qraux[keep]) }
  p(paste0(k, ".rank"), q$rank); p(paste0(k, ".pivot"), q$pivot)
}
accessors <- function(k, x, y) {
  q <- qr(x)
  show(k, q)
  p(paste0(k, ".Q"), qr.Q(q)); p(paste0(k, ".Qc"), qr.Q(q, complete = TRUE))
  p(paste0(k, ".R"), qr.R(q)); p(paste0(k, ".Rc"), qr.R(q, complete = TRUE))
  p(paste0(k, ".dimQ"), dim(qr.Q(q))); p(paste0(k, ".dimR"), dim(qr.R(q)))
  p(paste0(k, ".qy"), qr.qy(q, y)); p(paste0(k, ".qty"), qr.qty(q, y))
  p(paste0(k, ".res"), qr.resid(q, y))
  # at rank 0 R returns y as the fitted values and also as the residuals, which
  # cannot both be right; the fit of a zero matrix is 0 here
  if (q$rank > 0) p(paste0(k, ".fit"), qr.fitted(q, y))
  b <- qr.coef(q, y)
  cat(k, ".na=", sum(is.na(b)), "\n", sep = "")
  p(paste0(k, ".coef"), b[!is.na(b)])
}

hilbert <- function(n) { m <- matrix(0, n, n); for (i in 1:n) for (j in 1:n) m[i, j] <- 1 / (i + j - 1); m }
wave <- function(n, m) { x <- matrix(0, n, m); for (i in 1:n) for (j in 1:m) x[i, j] <- sin(i * j + 1) + i / (j + 2); x }

accessors("tall", cbind(1, c(1, 2, 4, 8)), c(1, 3, 2, 5))
accessors("square", matrix(c(2, 1, 1, 3, 2, 1, 1, 0, 4), 3), c(1, 2, 3))
accessors("wide", matrix(c(1, 2, 3, 4, 5, 7, 2, 1), 2), c(1, 2))
accessors("onecol", matrix(c(3, 4, 0), 3), c(1, 2, 3))
accessors("onerow", matrix(c(3, 4), 1), 2)
accessors("oneone", matrix(-5, 1), 2)
accessors("wave", wave(30, 8), sin(1:30))

# condition number 1.5e7. The last qraux is a norm downdated five times, so both
# programs lose it: the truth is |R[6, 6]| = 3.98624e-07, R has 3.98600e-07 and
# this has 3.98607e-07. Everything else stands.
qh <- qr(hilbert(6))
p("hilbert.qr", qh$qr); p("hilbert.aux", qh$qraux[1:5]); p("hilbert.R", qr.R(qh)); p("hilbert.Q", qr.Q(qh))
cat("hilbert.top=", abs(abs(qh$qraux[6]) / abs(qh$qr[6, 6]) - 1) < 1e-3, "\n", sep = "")

# rank deficient: the negligible column moves right, the others keep their order
accessors("collinear", cbind(1:5, 2 * (1:5), c(1, 0, 1, 0, 2)), c(1, 2, 4, 3, 7))
accessors("zerofirst", cbind(0, 1:4, c(2, 1, 2, 5)), c(1, 2, 3, 4))
accessors("twodrop", cbind(1:6, 2:7, 3:8, c(1, 0, 0, 1, 1, 0)), 6:1)
accessors("zero", matrix(0, 3, 2), c(1, 2, 3))

# the tolerance decides the rank
near <- cbind(1:6, 1:6 + 1e-6 * c(1, -1, 2, 0, 1, -2))
show("tol.default", qr(near))
show("tol.loose", qr(near, tol = 1e-3))
show("tol.tight", qr(near, 1e-12))

# a vector is one column; a matrix response keeps its columns
show("vector", qr(c(1, 2, 2)))
qv <- qr(wave(6, 2))
Y <- cbind(1:6, c(1, 0, 1, 0, 1, 0))
p("my.qy", qr.qy(qv, Y)); p("my.qty", qr.qty(qv, Y)); p("my.coef", qr.coef(qv, Y))
p("my.res", qr.resid(qv, Y)); p("my.fit", qr.fitted(qv, Y))
p("my.dim", c(dim(qr.qy(qv, Y)), dim(qr.coef(qv, Y))))

# the first k columns only
qk <- qr(wave(8, 3))
p("fit.k1", qr.fitted(qk, 8:1, k = 1)); p("fit.k2", qr.fitted(qk, 8:1, 2))

# the matrix back, in its original order
back <- list(wave(7, 3), cbind(1:5, 2 * (1:5), c(1, 0, 1, 0, 2)), matrix(c(1, 2, 3, 4, 5, 7, 2, 1), 2))
for (i in 1:3) {
  s <- back[[i]]
  q <- qr(s)
  p(paste0("back", i, ".diff"), qr.X(q, ncol = ncol(s)) - s)
  p(paste0("back", i), qr.X(q, ncol = max(ncol(s), nrow(s))))
}
p("back.more", qr.X(qr(wave(6, 2)), ncol = 4))
p("back.less", qr.X(qr(wave(6, 3)), ncol = 2))
p("back.full", qr.X(qr(wave(6, 2)), complete = TRUE))
p("q.dvec", qr.Q(qr(wave(5, 2)), Dvec = c(2, -1)))

# solving
A <- matrix(c(4, 2, 1, 1, 3, 2, 0, 1, 5), 3)
p("s.b", qr.solve(A, c(1, 2, 3)))
p("s.inv", qr.solve(A))
p("s.mat", qr.solve(A, cbind(1:3, 3:1)))
p("s.ls", qr.solve(wave(9, 3), 1:9))
p("s.q", qr.solve(qr(A), c(1, 2, 3)))
p("s.solve", solve(qr(A), c(1, 2, 3)))
p("s.solveinv", solve(qr(A)))
p("s.wide", qr.solve(matrix(c(1, 2, 3, 4, 5, 7), 2), c(1, 2)))
p("s.tol", round(qr.solve(near, 1:6, tol = 1e-12), 6))

# against what it should agree with
fit <- lm(c(1, 3, 2, 5, 4, 7) ~ c(1, 2, 3, 4, 5, 7))
q <- qr(cbind(1, c(1, 2, 3, 4, 5, 7)))
p("lm.coef", qr.coef(q, c(1, 3, 2, 5, 4, 7)) - coef(fit))
p("lm.res", qr.resid(q, c(1, 3, 2, 5, 4, 7)) - residuals(fit))
p("orth", crossprod(qr.Q(q)))
p("product", qr.Q(q) %*% qr.R(q))

# names
X <- cbind(a = c(1, 2, 3, 4), b = c(2, 4, 6, 8), c = c(1, 0, 1, 3))
rownames(X) <- c("r1", "r2", "r3", "r4")
qn <- qr(X)
cat("cn=", colnames(qn$qr), "rn=", rownames(qn$qr), "\n")
cat("coefn=", names(qr.coef(qn, 1:4)), "\n")
cat("coefm=", rownames(qr.coef(qn, cbind(u = 1:4))), colnames(qr.coef(qn, cbind(u = 1:4))), "\n")
cat("Rn=", colnames(qr.R(qn)), rownames(qr.R(qn)), "\n")
cat("Xn=", colnames(qr.X(qn, ncol = 3)), rownames(qr.X(qn, ncol = 3)), "\n")
cat("yn=", names(qr.qty(qn, c(p = 1, q = 2, r = 3, s = 4))), names(qr.resid(qn, c(p = 1, q = 2, r = 3, s = 4))), "\n")
cat("vecn=", rownames(qr(c(u = 1, v = 2))$qr), "\n")

# an object, and a data frame goes in as its matrix
cat("class=", class(qn), "is=", is.qr(qn), is.qr(X), "names=", names(qn), "\n")
show("df", qr(data.frame(u = c(1, 2, 3), v = c(1, 0, 1))))
p("df.solve", qr.solve(data.frame(u = c(1, 2), v = c(1, 0)), c(1, 2)))
