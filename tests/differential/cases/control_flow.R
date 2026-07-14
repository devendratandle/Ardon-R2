# Loops, recursion, closures, edge cases R is famously picky about.
s <- 0
for (i in 1:10) s <- s + i * i
cat("for.sumsq=", s, "\n", sep = "")
b <- 5
for (it in 1:0) b <- b + 1   # R iterates 1 THEN 0 -> two iterations
cat("for.desc=", b, "\n", sep = "")
z <- 0
for (i in seq_len(0)) z <- z + 1   # zero iterations
cat("for.empty=", z, "\n", sep = "")
w <- 0
while (w < 7) w <- w + 2
cat("while.val=", w, "\n", sep = "")
k <- 0
repeat { k <- k + 3; if (k > 10) break }
cat("repeat.val=", k, "\n", sep = "")
n <- 0
for (i in 1:10) { if (i %% 2 == 0) next; n <- n + i }
cat("next.oddsum=", n, "\n", sep = "")
fact <- function(n) if (n <= 1) 1 else n * fact(n - 1)
cat("recursion.fact6=", fact(6), "\n", sep = "")
make_add <- function(k) function(x) x + k
add10 <- make_add(10)
cat("closure.add=", add10(5), "\n", sep = "")
f <- function(x, y = 2) x ^ y
cat("default.arg=", f(3), "\n", sep = "")
cat("named.arg=", f(y = 3, x = 2), "\n", sep = "")
cat("ifelse.vec=", paste(ifelse(c(1, 5, 3) > 2, "hi", "lo"), collapse = ","), "\n", sep = "")
cat("switch.val=", switch("b", a = 1, b = 22, c = 3), "\n", sep = "")
