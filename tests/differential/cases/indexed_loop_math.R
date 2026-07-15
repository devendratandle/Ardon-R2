# J.5 loop-to-vector: indexed accumulation loops must equal their
# vectorized spellings (and R), whether interpreted or JIT-compiled.
va_loop <- function(x) { s <- 0; for (i in 1:length(x)) s <- s + (x[i] - mean(x))^2; s / (length(x) - 1) }
x <- c(2.1, 4.7, 1.3, 8.9, 5.5, 3.2, 7.8, 6.4, 0.9, 9.1)
cat("var.loop=", va_loop(x), "\n", sep = "")
cat("var.builtin=", var(x), "\n", sep = "")
dot_loop <- function(a, b) { s <- 0; for (i in 1:length(a)) s <- s + a[i] * b[i]; s }
y <- c(1.0, 3.5, 2.2, 7.1, 4.9, 2.8, 6.6, 5.9, 1.4, 8.3)
cat("dot.loop=", dot_loop(x, y), "\n", sep = "")
cat("dot.vec=", sum(x * y), "\n", sep = "")
wmean <- function(v, w) { n <- length(v); s <- 0; for (i in 1:n) s <- s + v[i] * w[i]; s / sum(w) }
cat("wmean.loop=", wmean(x, y), "\n", sep = "")
nested <- function(x, w) { b <- 0; for (it in 1:5) { s <- 0; for (i in 1:length(x)) s <- s + x[i] * w[i]; b <- b + s * 0.1 }; b }
cat("nested.loops=", nested(x, y), "\n", sep = "")
recur <- function(c) { s <- 0; for (i in 1:length(c)) s <- s * 2 + c[i]; s }
cat("recurrence.untouched=", recur(c(1, 2, 3, 4)), "\n", sep = "")
# Block-bodied helper composition (J.5b): library-style std() chain.
vmean2 <- function(x) sum(x) / length(x)
vsd2 <- function(x) { m <- vmean2(x); s <- sqrt(sum((x - m)^2) / length(x)); if (s < 1e-12) 1 else s }
std2 <- function(x) { m <- vmean2(x); (x - m) / vsd2(x) }
z <- std2(x)
cat("std.mean.iszero=", if (abs(vmean2(z)) < 1e-9) 1 else 0, "\n", sep = "")
cat("std.sd=", vsd2(z), "\n", sep = "")
cat("std.first=", z[1], "\n", sep = "")
# Store-map loops: y[i] <- expr over a numeric(length(v)) buffer.
sqdev <- function(x) { y <- numeric(length(x)); m <- sum(x)/length(x); for (i in 1:length(x)) y[i] <- (x[i] - m)^2; y }
cat("store.sum=", sum(sqdev(x)), "\n", sep = "")
cat("store.first=", sqdev(x)[1], "\n", sep = "")
wprod <- function(a, b) { y <- numeric(length(a)); for (i in 1:length(a)) y[i] <- a[i] * b[i] + 1; sum(y) }
cat("store.nested=", wprod(x, y), "\n", sep = "")
