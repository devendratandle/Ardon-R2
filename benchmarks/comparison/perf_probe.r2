# Tight perf probe — no while loops (those have separate overhead).
# Each measurement is a single timed expression.

tic <- function() Sys.time()
toc <- function(t0) as.numeric(Sys.time() - t0)
emit <- function(key, sec) cat(key, "=", sec, "\n", sep = "")

# Tic/toc resolution (already showed ~20μs in first run; reconfirm).
t0 <- tic(); s <- toc(t0); emit("overhead.empty_tic_toc", s)
t0 <- tic(); s <- toc(t0); emit("overhead.empty_tic_toc_2", s)

# Scale the workload — see if it's O(n) or O(constant + n).
# rnorm size varies; tic only the binary op.

set.seed(1); a <- rnorm(1e3); b <- rnorm(1e3)
t0 <- tic(); s <- a + b; emit("vec_add_1e3", toc(t0))

set.seed(1); a <- rnorm(1e4); b <- rnorm(1e4)
t0 <- tic(); s <- a + b; emit("vec_add_1e4", toc(t0))

set.seed(1); a <- rnorm(1e5); b <- rnorm(1e5)
t0 <- tic(); s <- a + b; emit("vec_add_1e5", toc(t0))

set.seed(1); a <- rnorm(1e6); b <- rnorm(1e6)
t0 <- tic(); s <- a + b; emit("vec_add_1e6", toc(t0))

set.seed(1); a <- rnorm(1e7); b <- rnorm(1e7)
t0 <- tic(); s <- a + b; emit("vec_add_1e7", toc(t0))

# Pure sum reductions at same sizes
set.seed(1); a <- rnorm(1e6)
t0 <- tic(); s <- sum(a); emit("sum_1e6", toc(t0))

set.seed(1); a <- rnorm(1e7)
t0 <- tic(); s <- sum(a); emit("sum_1e7", toc(t0))

# Cost of `a + b` only — vs sum(a + b) which forces a materialization
set.seed(1); a <- rnorm(1e7); b <- rnorm(1e7)
t0 <- tic(); s <- sum(a + b); emit("sum_of_add_1e7", toc(t0))
