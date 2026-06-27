set.seed(42)
v <- rnorm(1e7)
.w <- sum(v)

f <- function(x) sum(sqrt(x*x + 1))
t0 <- Sys.time(); s <- f(v); t1 <- as.numeric(Sys.time() - t0)
cat(sprintf("fused_closure=%.4f\n", t1))
cat(sprintf("  result = %.4f\n", s))

t0 <- Sys.time(); s2 <- sum(sqrt(v*v + 1)); t1 <- as.numeric(Sys.time() - t0)
cat(sprintf("unfused_inline=%.4f\n", t1))
cat(sprintf("  result = %.4f\n", s2))
