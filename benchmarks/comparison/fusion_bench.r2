# Phase C.9 fusion bench. Demonstrates the JIT map-reduce fusion path:
#   `function(x) sum(sqrt(x*x + 1))` JITs as one fused loop (no
#   intermediate 8 MB vector), versus the inline `sum(sqrt(x*x + 1))`
#   form where the engine evaluates intermediates separately.

set.seed(42)
v <- rnorm(1e7)

tic <- function() Sys.time()
toc <- function(t0) as.numeric(Sys.time() - t0)
emit <- function(key, sec) cat(key, "=", sec, "\n", sep = "")

# Warmup (so first-call JIT compile doesn't skew timing)
.w <- sum(v)

# Fused closure: function(x) sum(sqrt(x*x + 1)) hits Phase C.9 path.
f <- function(x) sum(sqrt(x*x + 1))
t0 <- tic(); s <- f(v); t1 <- toc(t0)
emit("fused_closure", t1)
cat("  result =", s, "\n")

# Unfused inline: engine materialises each intermediate vector.
t0 <- tic(); s2 <- sum(sqrt(v*v + 1)); t1 <- toc(t0)
emit("unfused_inline", t1)
cat("  result =", s2, "\n")
