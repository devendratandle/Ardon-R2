# The beta and gamma families against R. Printed in 15 digits, so the harness's
# 1e-9 tolerance applies. Shapes and tails are kept where R itself is right:
# against a 340-digit reference, R's pbeta(0.75, 1e4, 25, log.p = TRUE) is -Inf
# and qbeta(1e-300, 10, 1e5, lower.tail = FALSE) is 1, both wrong.
p <- function(k, v) for (i in seq_along(v)) cat(k, ".", i, "=", sprintf("%.15g", v[i]), "\n", sep = "")
A <- c(0.01, 0.1, 0.5, 1, 1.5, 2, 3.7, 10, 25, 100, 1000)
X <- c(1e-10, 1e-5, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1 - 1e-5)
Q <- c(1e-100, 1e-20, 1e-5, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-5)
for (i in seq_along(A)) for (j in seq_along(A)) {
  a <- A[i]; b <- A[j]; k <- paste(i, j, sep = ".")
  for (m in seq_along(X)) {
    x <- X[m]; km <- paste(k, m, sep = ".")
    p(paste0("dbeta.", km), dbeta(x, a, b))
    p(paste0("dbeta.log.", km), dbeta(x, a, b, log = TRUE))
    p(paste0("pbeta.", km), pbeta(x, a, b))
    p(paste0("pbeta.up.", km), pbeta(x, a, b, lower.tail = FALSE))
  }
  for (m in seq_along(Q)) {
    q <- Q[m]; km <- paste(k, m, sep = ".")
    p(paste0("qbeta.", km), qbeta(q, a, b))
    p(paste0("qbeta.up.", km), qbeta(q, a, b, lower.tail = FALSE))
  }
}
M <- c(1e-5, 0.01, 0.1, 0.5, 0.9, 1, 1.1, 2, 5, 20)
G <- c(0.01, 0.1, 0.5, 1, 1.5, 2, 3.7, 10, 25, 100, 1000, 1e4)
for (i in seq_along(G)) {
  a <- G[i]
  for (m in seq_along(M)) {
    x <- a * M[m]; km <- paste(i, m, sep = ".")
    p(paste0("dgamma.", km), dgamma(x, a, rate = 2))
    p(paste0("dgamma.log.", km), dgamma(x, a, scale = 0.5, log = TRUE))
    p(paste0("pgamma.", km), pgamma(x, a, 2))
    p(paste0("pgamma.up.", km), pgamma(x, a, 2, lower.tail = FALSE))
    p(paste0("pgamma.log.", km), pgamma(x, a, scale = 3, log.p = TRUE))
  }
  for (m in seq_along(Q)) {
    q <- Q[m]; km <- paste(i, m, sep = ".")
    p(paste0("qgamma.", km), qgamma(q, a, 2))
    p(paste0("qgamma.up.", km), qgamma(q, a, scale = 3, lower.tail = FALSE))
  }
}
# log.p on the way in
p("qbeta.logp", qbeta(-5, 2, 3, log.p = TRUE))
p("qbeta.logp.up", qbeta(-5, 2, 3, lower.tail = FALSE, log.p = TRUE))
p("qgamma.logp", qgamma(-5, 2, log.p = TRUE))
p("pbeta.log.up", pbeta(0.3, 2, 3, lower.tail = FALSE, log.p = TRUE))
# every parameter recycles
p("recycle.x", pbeta(c(0.1, 0.5, 0.9), 2, 3))
p("recycle.shape", pbeta(0.5, c(1, 2, 3), 2))
p("recycle.both", pbeta(c(0.2, 0.7), c(1, 2, 3, 4), 2))
p("recycle.gamma", pgamma(1:4, c(1, 2), c(1, 2, 3, 4)))
p("recycle.scale", qgamma(0.5, 2, scale = c(1, 2)))
p("recycle.dbeta", dbeta(0.3, c(1, 2), c(3, 4)))
# the boundaries
p("dbeta.0", c(dbeta(0, 2, 3), dbeta(0, 1, 3)))
p("dbeta.1", c(dbeta(1, 3, 2), dbeta(1, 3, 1)))
p("dgamma.0", c(dgamma(0, 2), dgamma(0, 1), dgamma(0, 1, rate = 3)))
p("edges", c(pbeta(0, 2, 3), pbeta(1, 2, 3), pbeta(-1, 2, 3), pbeta(2, 2, 3)))
p("edges.q", c(qbeta(0, 2, 3), qbeta(1, 2, 3), qgamma(0, 2)))
cat("inf=", qgamma(1, 2), ",", dbeta(0, 0.5, 2), ",", dgamma(0, 0.5), "\n", sep = "")
# zero and infinite shapes: the limits are point masses
p("limit.p", c(pbeta(0.3, 0, 2), pbeta(0.3, 2, 0), pbeta(0.3, 0, 0), pbeta(0.3, Inf, Inf), pbeta(0.7, Inf, Inf)))
p("limit.q", c(qbeta(0.3, 0, 2), qbeta(0.3, 2, 0), qbeta(0.3, 0, 0), qbeta(0.7, 0, 0), qbeta(0.5, 0, 0), qbeta(0.3, Inf, Inf)))
p("limit.gamma", c(pgamma(1, 0), pgamma(0, 0), qgamma(0.5, 0)))
# what the families are, by name
p("exp", pgamma(2, 1, 0.5) - pexp(2, 0.5))
p("chisq", pgamma(3, 2.5, scale = 2) - pchisq(3, 5))
p("t", pbeta(9 / (9 + 4 * 2.5^2), 4.5, 0.5) - 2 * pt(-2.5, 9))
p("na", c(pbeta(NA, 2, 3), pgamma(1, NA)))
