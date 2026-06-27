# Isolate matmul timing — verify the dgemm kernel speed reaches the engine.

tic <- function() Sys.time()
toc <- function(t0) as.numeric(Sys.time() - t0)
emit <- function(key, sec) cat(key, "=", sec, "\n", sep = "")

set.seed(3)
A <- matrix(rnorm(500 * 500), 500, 500)
B <- matrix(rnorm(500 * 500), 500, 500)

# First call — pays any one-time setup
t0 <- tic(); C <- A %*% B; emit("matmul_500_call1", toc(t0))
# Second call — should be steady-state
t0 <- tic(); C <- A %*% B; emit("matmul_500_call2", toc(t0))
# Third call
t0 <- tic(); C <- A %*% B; emit("matmul_500_call3", toc(t0))
# Average over 5 inline calls without overhead between
t0 <- tic()
C <- A %*% B; C <- A %*% B; C <- A %*% B; C <- A %*% B; C <- A %*% B
emit("matmul_500_5x_avg", toc(t0) / 5)
