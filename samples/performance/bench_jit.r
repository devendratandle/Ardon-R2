# Phase C.2 JIT benchmark — hot scalar loop
# Run with R2_JIT=1 (default, JIT on) and R2_JIT=0 (JIT off) and compare.

f <- function(x, y) x*x + 2*x*y + y*y + 1

s <- 0
n <- 1000000
i <- 1
while (i <= n) {
  s <- s + f(i, 2)
  i <- i + 1
}
cat("sum =", s, "\n")
