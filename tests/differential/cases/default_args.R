# Lazy-style default arguments: defaults see other args, evaluate in the
# function's own environment, chain in declaration order; missing().
f <- function(x, y = x * 2) x + y
cat("dep.default=", f(5), "\n", sep = "")
cat("dep.override=", f(5, 1), "\n", sep = "")
g <- function(a, b = length(a)) b
cat("fn.default=", g(c(1, 2, 3)), "\n", sep = "")
k <- function(n, m = n + 1, p = m * 2) p
cat("chained.default=", k(3), "\n", sep = "")
h <- function(x, y) if (missing(y)) -1 else y
cat("missing.true=", h(1), "\n", sep = "")
cat("missing.false=", h(1, 9), "\n", sep = "")
hd <- function(x, y = 7) if (missing(y)) y * 10 else y
cat("missing.defaulted=", hd(1), "\n", sep = "")
cat("missing.supplied=", hd(1, 2), "\n", sep = "")
sq <- function(x, pow = 2) x ^ pow
cat("named.only=", sq(x = 4), "\n", sep = "")
