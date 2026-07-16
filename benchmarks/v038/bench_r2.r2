# v0.3.8 benchmark — R2 side. Emits key=seconds lines.
# Timing helper: median of 3 runs of a thunk.
bench <- function(label, fn) {
  best <- Inf
  for (r in 1:3) {
    t0 <- as.numeric(Sys.time())
    fn()
    dt <- as.numeric(Sys.time()) - t0
    if (dt < best) best <- dt
  }
  cat(label, "=", best, "\n", sep = "")
}

set.seed(1)
xs <- rnorm(1e6); ys <- rnorm(1e6)
xm <- rnorm(1e7); ym <- rnorm(1e7)

# ── C-internals class: R2 builtins vs R builtins ─────────────────────
bench("cint.vecadd_1e7",  function() { z <- xm + ym })
bench("cint.sum_1e7",     function() { s <- sum(xm) })
bench("cint.sort_1e6",    function() { s <- sort(xs) })
bench("cint.sd_1e6",      function() { s <- sd(xs) })
bench("cint.cor_1e6",     function() { s <- cor(xs, ys) })
A <- matrix(rnorm(500*500), 500, 500); B <- matrix(rnorm(500*500), 500, 500)
bench("cint.matmul_500",  function() { m <- A %*% B })

# ── Native-R2 class: user formulas that JIT-compile ──────────────────
va  <- function(x) sum((x - mean(x))^2) / (length(x) - 1)
cr  <- function(x, y) sum((x-mean(x))*(y-mean(y))) / sqrt(sum((x-mean(x))^2)*sum((y-mean(y))^2))
bench("native.var_formula_1e6", function() { v <- va(xs) })
bench("native.cor_formula_1e6", function() { v <- cr(xs, ys) })

# ── Loop class: the same textbook loop R runs interpreted ────────────
xsmall <- xs[1:1e5]; ysmall <- ys[1:1e5]  # O(n^2) loop.var needs a size R can finish
valoop <- function(x) { s <- 0; for (i in 1:length(x)) s <- s + (x[i] - mean(x))^2; s / (length(x) - 1) }
dotloop <- function(a, b) { s <- 0; for (i in 1:length(a)) s <- s + a[i]*b[i]; s }
gd <- function(x) { b <- 0; for (it in 1:500) b <- b + 0.5*mean(x - b); b }
bench("loop.var_1e5",   function() { v <- valoop(xsmall) })
bench("loop.dot_1e6",   function() { v <- dotloop(xs, ys) })
bench("loop.gd_1e6",    function() { v <- gd(xs) })

# ── Addon-library class: r2sem-style composed helpers ────────────────
vmean <- function(x) sum(x) / length(x)
vsd   <- function(x) { m <- vmean(x); s <- sqrt(sum((x - m)^2) / length(x)); if (s < 1e-12) 1 else s }
std   <- function(x) { m <- vmean(x); (x - m) / vsd(x) }
bench("addon.standardize_1e6", function() { z <- std(xs) })
