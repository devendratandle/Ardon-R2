# Probe SIMD impact on math1-shape workload, 5 runs.

tic <- function() Sys.time()
toc <- function(t0) as.numeric(Sys.time() - t0)
emit <- function(key, sec) cat(key, "=", sec, "\n", sep = "")

set.seed(42)
v <- rnorm(1e6, mean = 0.5, sd = 1.5)
f1 <- function(x) sqrt(x*x + 1)

# Warmup
r <- f1(v); s <- sum(r)
emit("warmup_checksum", s)

# 5 timed runs
for (i in 1:5) {
  t0 <- tic(); r <- f1(v); t1 <- toc(t0)
  emit(paste0("run_", i), t1)
}
