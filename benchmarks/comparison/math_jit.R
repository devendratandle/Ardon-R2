# R-side math-JIT comparison.
# Five user-defined functions whose bodies are pure scalar arithmetic +
# math calls — exactly the shape M-R2-JIT compiles to native code. The
# script applies each to a 1e6-element vector, prints checksums (for
# accuracy comparison) and wall-clock time (for performance comparison).

options(digits = 12)

set.seed(42)
v <- rnorm(1e6, mean = 0.5, sd = 1.5)
w <- rnorm(1e6, mean = 0.3, sd = 0.8)

# Warmup
.x <- v + w; .s <- sum(.x)

# ── 1. sqrt(x*x + 1) — branchless, one math call per element ──
f1 <- function(x) sqrt(x*x + 1)
t0 <- Sys.time(); r1 <- f1(v); t1 <- as.numeric(Sys.time() - t0)
cat(sprintf("math1.checksum=%.10f\n", sum(r1)))
cat(sprintf("math1.time=%.4f\n", t1))

# ── 2. log(exp(x)) — should be identity (numerical roundtrip) ──
f2 <- function(x) log(exp(x))
t0 <- Sys.time(); r2 <- f2(v); t1 <- as.numeric(Sys.time() - t0)
cat(sprintf("math2.checksum=%.10f\n", sum(r2)))
cat(sprintf("math2.time=%.4f\n", t1))

# ── 3. sin^2 + cos^2 — Pythagorean identity, all ~1.0 ──
f3 <- function(x) sin(x)*sin(x) + cos(x)*cos(x)
t0 <- Sys.time(); r3 <- f3(v); t1 <- as.numeric(Sys.time() - t0)
cat(sprintf("math3.checksum=%.10f\n", sum(r3)))
cat(sprintf("math3.time=%.4f\n", t1))

# ── 4. sqrt(x^2 + y^2) — 2-arg, hypotenuse ──
f4 <- function(x, y) sqrt(x*x + y*y)
t0 <- Sys.time(); r4 <- f4(v, w); t1 <- as.numeric(Sys.time() - t0)
cat(sprintf("math4.checksum=%.10f\n", sum(r4)))
cat(sprintf("math4.time=%.4f\n", t1))

# ── 5. abs(sin(x)) + abs(cos(x)) — multi-call chain ──
f5 <- function(x) abs(sin(x)) + abs(cos(x))
t0 <- Sys.time(); r5 <- f5(v); t1 <- as.numeric(Sys.time() - t0)
cat(sprintf("math5.checksum=%.10f\n", sum(r5)))
cat(sprintf("math5.time=%.4f\n", t1))
