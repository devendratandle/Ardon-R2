# Distribution functions against R on a grid, tails included. Values are
# printed to 15 significant digits; tail probabilities are also compared
# as log(p) (log.p = TRUE), because the harness's absolute floor (1e-12)
# would let a p-value of 8e-19 computed as 0 pass. Keys name the call.
kv <- function(k, v) cat(k, "=", sprintf("%.15e", v), "\n", sep = "")
dfs <- c(1, 2, 5, 10, 30, 100)

# t: centre to far tail
for (d in dfs) {
  for (t in c(-7.1, -3, 1e-9, 0.1, 0.5, 0.99, 1.01, 2, 5, 12, 40)) {
    kv(sprintf("pt.t%s.df%s", t, d), pt(t, d))
    kv(sprintf("pt.up.t%s.df%s", t, d), pt(t, d, lower.tail = FALSE))
    kv(sprintf("pt.logup.t%s.df%s", t, d), pt(t, d, lower.tail = FALSE, log.p = TRUE))
    kv(sprintf("dt.t%s.df%s", t, d), dt(t, d))
  }
  for (p in c(1e-10, 0.001, 0.025, 0.5, 0.9, 0.999)) {
    kv(sprintf("qt.p%s.df%s", p, d), qt(p, d))
    kv(sprintf("qt.up.p%s.df%s", p, d), qt(p, d, lower.tail = FALSE))
  }
}

# F
for (d1 in c(1, 3, 10)) {
  for (d2 in c(5, 30, 200)) {
    for (x in c(0.2, 1, 4, 30)) {
      kv(sprintf("pf.x%s.%s.%s", x, d1, d2), pf(x, d1, d2))
      kv(sprintf("pf.logup.x%s.%s.%s", x, d1, d2), pf(x, d1, d2, lower.tail = FALSE, log.p = TRUE))
    }
    for (p in c(0.01, 0.5, 0.95, 0.9999)) kv(sprintf("qf.p%s.%s.%s", p, d1, d2), qf(p, d1, d2))
  }
}

# normal: deep tails
for (z in c(-40, -10, -3, 0, 1.96, 8)) {
  kv(sprintf("pnorm.z%s", z), pnorm(z))
  kv(sprintf("pnorm.log.z%s", z), pnorm(z, log.p = TRUE))
  kv(sprintf("dnorm.z%s", z), dnorm(z))
}
for (p in c(1e-300, 1e-20, 1e-5, 0.3, 0.975)) kv(sprintf("qnorm.p%s", p), qnorm(p))

# binomial and Poisson
for (n in c(10, 200)) for (pr in c(0.03, 0.5)) for (k in c(0, 3, 7)) {
  kv(sprintf("dbinom.%s.%s.%s", k, n, pr), dbinom(k, n, pr))
  kv(sprintf("pbinom.%s.%s.%s", k, n, pr), pbinom(k, n, pr))
  kv(sprintf("pbinom.logup.%s.%s.%s", k, n, pr), pbinom(k, n, pr, lower.tail = FALSE, log.p = TRUE))
}
for (lam in c(0.5, 4, 100)) for (k in c(0, 2, 10, 150)) {
  kv(sprintf("dpois.%s.%s", k, lam), dpois(k, lam))
  kv(sprintf("ppois.%s.%s", k, lam), ppois(k, lam))
  kv(sprintf("ppois.logup.%s.%s", k, lam), ppois(k, lam, lower.tail = FALSE, log.p = TRUE))
}
for (p in c(0.05, 0.5, 0.95)) {
  kv(sprintf("qbinom.%s", p), qbinom(p, 50, 0.3))
  kv(sprintf("qpois.%s", p), qpois(p, 7))
}

# exponential
for (x in c(0.001, 1, 50)) {
  kv(sprintf("pexp.%s", x), pexp(x, 2))
  kv(sprintf("pexp.logup.%s", x), pexp(x, 2, lower.tail = FALSE, log.p = TRUE))
  kv(sprintf("dexp.%s", x), dexp(x, 2))
}
kv("qexp.0.9", qexp(0.9, 2))
