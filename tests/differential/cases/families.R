# The standard families against R: uniform, Weibull, lognormal, Cauchy, logistic,
# geometric, negative binomial, hypergeometric, the F density and quantile, and
# the normal's positional mean and sd. Printed in 15 digits, so the harness's
# 1e-9 tolerance applies.
p <- function(k, v) for (i in seq_along(v)) cat(k, ".", i, "=", sprintf("%.15g", v[i]), "\n", sep = "")
Q <- c(1e-20, 1e-5, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-5)

# the normal took its parameters by name only
p("norm.pos", c(dnorm(1, 2, 3), pnorm(1, 2, 3), qnorm(0.7, 2, 3)))
p("norm.named", c(dnorm(1, mean = 2, sd = 3), pnorm(1, sd = 3, mean = 2), qnorm(0.7, sd = 3)))
p("norm.upper", pnorm(1, 2, 3, FALSE))

# continuous: every function, tail and log form, over parameters and points
check <- function(name, d, pf, qf, xs, ps) {
  for (i in seq_along(ps)) {
    par <- ps[[i]]
    for (j in seq_along(xs)) {
      x <- xs[j]; k <- paste(name, i, j, sep = ".")
      p(paste0(k, ".d"), do.call(d, c(list(x), par)))
      p(paste0(k, ".dl"), do.call(d, c(list(x), par, list(log = TRUE))))
      p(paste0(k, ".p"), do.call(pf, c(list(x), par)))
      p(paste0(k, ".u"), do.call(pf, c(list(x), par, list(lower.tail = FALSE))))
      p(paste0(k, ".pl"), do.call(pf, c(list(x), par, list(log.p = TRUE))))
    }
    for (j in seq_along(Q)) {
      k <- paste(name, i, j, sep = ".")
      p(paste0(k, ".q"), do.call(qf, c(list(Q[j]), par)))
      p(paste0(k, ".qu"), do.call(qf, c(list(Q[j]), par, list(lower.tail = FALSE))))
      p(paste0(k, ".ql"), do.call(qf, c(list(log(Q[j])), par, list(log.p = TRUE))))
    }
  }
}
XP <- c(1e-5, 0.01, 0.3, 1, 1.7, 5, 40, 1000)
XR <- c(-1000, -40, -5, -1, -0.3, 0, 0.3, 1, 5, 40, 1000)
check("weibull", dweibull, pweibull, qweibull, XP, list(list(0.5), list(1), list(1.7, 3), list(10, 0.5)))
check("lnorm", dlnorm, plnorm, qlnorm, XP, list(list(), list(1), list(0.5, 2), list(-3, 0.25)))
check("cauchy", dcauchy, pcauchy, qcauchy, XR, list(list(), list(1), list(2, 3), list(-5, 0.1)))
check("logis", dlogis, plogis, qlogis, XR, list(list(), list(1), list(2, 3), list(-5, 0.1)))
check("unif", dunif, punif, qunif, c(-1, 0, 0.25, 0.5, 1, 2, 3), list(list(), list(0, 2), list(-1, 3)))
check("f", df, pf, qf, XP, list(list(1, 1), list(2, 5), list(3, 12), list(5, 2), list(10, 100), list(200, 30), list(0.5, 4)))

# discrete
checkd <- function(name, d, pf, qf, xs, ps) {
  for (i in seq_along(ps)) {
    par <- ps[[i]]
    for (j in seq_along(xs)) {
      x <- xs[j]; k <- paste(name, i, j, sep = ".")
      p(paste0(k, ".d"), do.call(d, c(list(x), par)))
      p(paste0(k, ".dl"), do.call(d, c(list(x), par, list(log = TRUE))))
      p(paste0(k, ".p"), do.call(pf, c(list(x), par)))
      p(paste0(k, ".u"), do.call(pf, c(list(x), par, list(lower.tail = FALSE))))
      p(paste0(k, ".pl"), do.call(pf, c(list(x), par, list(log.p = TRUE))))
    }
    # not 1e-20 in the upper tail: R adds up from the bottom and cannot hold
    # 1 - 1e-20, so qhyper(1e-20, 100, 900, 50, lower.tail = FALSE) is 24 in R,
    # where exact fractions give P(X > 24) = 9e-14 and the answer 30
    for (j in 2:length(Q)) {
      k <- paste(name, i, j, sep = ".")
      p(paste0(k, ".q"), do.call(qf, c(list(Q[j]), par)))
      p(paste0(k, ".qu"), do.call(qf, c(list(Q[j]), par, list(lower.tail = FALSE))))
    }
  }
}
XD <- c(0, 1, 2, 3, 5, 10, 25, 100)
checkd("geom", dgeom, pgeom, qgeom, XD, list(list(0.01), list(0.2), list(0.5), list(0.9), list(1)))
checkd("nbinom", dnbinom, pnbinom, qnbinom, XD, list(list(1, 0.5), list(3, 0.2), list(0.5, 0.7), list(20, 0.9), list(100, 0.3)))
checkd("nbmu", dnbinom, pnbinom, qnbinom, XD, list(list(3, mu = 2), list(0.5, mu = 10), list(100, mu = 0.01)))
checkd("hyper", dhyper, phyper, qhyper, XD, list(list(10, 7, 8), list(30, 20, 10), list(100, 900, 50), list(5, 5, 0), list(1000, 2000, 300)))

# recycling, positional and named arguments
p("r.unif", punif(c(0.1, 0.5), c(0, 0.2, 0.4), 1))
p("r.weibull", pweibull(1:4, c(1, 2), 2))
p("r.lnorm", plnorm(2, c(0, 1), c(1, 2, 3, 4)))
p("r.cauchy", qcauchy(0.8, 1:2, 2))
p("r.logis", dlogis(1:3, 0, 1:3))
p("r.geom", dgeom(0:3, c(0.2, 0.5)))
p("r.nbinom", dnbinom(0:3, 2:3, 0.5))
p("r.hyper", dhyper(1, 2:3, 3, 2))
p("r.df", df(1:2, 3, 4:5))
p("r.qf", qf(c(0.1, 0.9), 3, 7))
p("named", c(pweibull(2, shape = 1.5, scale = 2), plnorm(2, sdlog = 2), pgeom(3, prob = 0.2), phyper(2, m = 5, n = 5, k = 3)))

# edges and non-whole counts
p("edge.geom", c(dgeom(1.5, 0.3), dgeom(-1, 0.3), pgeom(-1, 0.3), pgeom(2.5, 0.3), qgeom(0, 0.3)))
p("edge.nb", c(dnbinom(1.5, 2, 0.3), dnbinom(0, 0, 0.3), dnbinom(1, 0, 0.3), pnbinom(-1, 2, 0.3), pnbinom(2.5, 2, 0.3)))
p("edge.hyper", c(dhyper(1.5, 5, 5, 3), dhyper(9, 5, 5, 3), phyper(9, 5, 5, 3), phyper(-1, 5, 5, 3)))
p("edge.f", c(df(0, 2, 5), df(0, 3, 5), df(-1, 3, 5), qf(0, 3, 5)))
p("edge.w", c(dweibull(0, 1), dweibull(0, 2), dweibull(-1, 2), pweibull(-1, 2), qweibull(0, 2)))
p("edge.u", c(dunif(0.5, 0, 1), punif(0, 0, 0), qunif(0.5, 2, 2)))
cat("inf=", dweibull(0, 0.5), ",", qf(1, 3, 5), ",", qcauchy(1), ",", qlogis(0), ",", qlnorm(1), ",", qgeom(1, 0.3), ",", qnbinom(1, 2, 0.3), "\n", sep = "")
p("na", c(dunif(NA), pweibull(1, NA), qcauchy(NA), plogis(NA), dhyper(1, NA, 2, 1)))
cat("nan=", is.nan(dunif(1, 2, 1)), is.nan(pweibull(1, -1)), is.nan(dgeom(1, 0)), is.nan(qgeom(2, 0.5)), is.nan(dhyper(1, -1, 2, 1)), "\n")

# the older functions: log = was read nowhere, and a vector parameter kept its
# first value, both silently
check("norm", dnorm, pnorm, qnorm, XR, list(list(), list(1), list(2, 3), list(-5, 0.1)))
check("exp", dexp, pexp, qexp, XP, list(list(), list(0.5), list(3)))
check("t", dt, pt, qt, XR, list(list(1), list(2.5), list(10), list(100), list(1e5)))
checkd("binom", dbinom, pbinom, qbinom, XD, list(list(5, 0.3), list(20, 0.5), list(100, 0.01), list(1000, 0.9), list(0, 0.5), list(10, 0), list(10, 1)))
checkd("pois", dpois, ppois, qpois, XD, list(list(0.1), list(1), list(3.5), list(20), list(1000), list(0)))
p("r.dnorm", dnorm(0, c(0, 1)))
p("r.pnorm", pnorm(0, c(0, 1), 2))
p("r.qnorm", qnorm(0.7, 0, c(1, 2)))
p("r.dexp", dexp(1, c(1, 2)))
p("r.pexp", pexp(1, c(1, 2)))
p("r.qexp", qexp(0.5, c(1, 2)))
p("r.dbinom", dbinom(1, 2:3, 0.5))
p("r.pbinom", pbinom(1, 2:3, 0.5))
p("r.qbinom", qbinom(0.5, 2:3, 0.5))
p("r.dpois", dpois(1, 1:2))
p("r.ppois", ppois(1, 1:2))
p("r.qpois", qpois(0.5, 1:2))
p("r.dt", dt(1, 1:2))
p("r.pt", pt(1, 1:2))
p("r.qt", qt(0.7, 1:2))
p("r.pf", pf(1, 1:2, 3))
p("far", c(pf(1e-5, 200, 30, log.p = TRUE), pt(-1e10, 2, log.p = TRUE), pbinom(0, 1000, 0.9, log.p = TRUE), ppois(0, 1000, log.p = TRUE)))
p("big", c(dpois(1000, 1000), dbinom(5000, 10000, 0.5), dt(1, 1e8)))

# what the families are, by name
p("id.exp", c(pweibull(2, 1, 1 / 0.5) - pexp(2, 0.5), pgeom(3, 0.3) - pnbinom(3, 1, 0.3)))
p("id.t", c(dcauchy(0.7) - dt(0.7, 1), pcauchy(0.7) - pt(0.7, 1)))
p("id.f", qf(0.95, 1, 9) - qt(0.975, 9)^2)
