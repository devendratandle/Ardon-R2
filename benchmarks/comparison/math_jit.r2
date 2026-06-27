# R2-side math-JIT comparison — matches `math_jit.R` exactly.
# Each user function body lowers via M-R2-JIT to native machine code
# (no interpreter fallback for these shapes after v0.1.0 Phase C.6).

tic <- function() Sys.time()
toc <- function(t0) as.numeric(Sys.time() - t0)
emit <- function(key, val) cat(key, "=", val, "\n", sep = "")

set.seed(42)
v <- rnorm(1e6, mean = 0.5, sd = 1.5)
w <- rnorm(1e6, mean = 0.3, sd = 0.8)

# Warmup
.x <- v + w; .s <- sum(.x)

# ── 1. sqrt(x*x + 1) ──
f1 <- function(x) sqrt(x*x + 1)
t0 <- tic(); r1 <- f1(v); t1 <- toc(t0)
emit("math1.checksum", sum(r1))
emit("math1.time",     t1)
rm("r1")

# ── 2. log(exp(x)) ──
f2 <- function(x) log(exp(x))
t0 <- tic(); r2 <- f2(v); t1 <- toc(t0)
emit("math2.checksum", sum(r2))
emit("math2.time",     t1)
rm("r2")

# ── 3. sin(x)*sin(x) + cos(x)*cos(x) — Pythagorean identity ──
f3 <- function(x) sin(x)*sin(x) + cos(x)*cos(x)
t0 <- tic(); r3 <- f3(v); t1 <- toc(t0)
emit("math3.checksum", sum(r3))
emit("math3.time",     t1)
rm("r3")

# ── 4. sqrt(x*x + y*y) ──
f4 <- function(x, y) sqrt(x*x + y*y)
t0 <- tic(); r4 <- f4(v, w); t1 <- toc(t0)
emit("math4.checksum", sum(r4))
emit("math4.time",     t1)
rm("r4")

# ── 5. abs(sin(x)) + abs(cos(x)) ──
f5 <- function(x) abs(sin(x)) + abs(cos(x))
t0 <- tic(); r5 <- f5(v); t1 <- toc(t0)
emit("math5.checksum", sum(r5))
emit("math5.time",     t1)
rm("r5")
