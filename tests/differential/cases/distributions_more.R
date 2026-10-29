# Distribution functions beyond distributions.R / distribution_grid.R:
# discrete quantiles, exponential, densities, upper tails, far tails.
p <- function(k, v) cat(k, "=", sprintf("%.15g", v), "\n", sep = "")
for (pp in c(0.01, 0.3, 0.5, 0.9, 0.999)) {
  p(paste0("qbinom.", pp), qbinom(pp, 20, 0.35))
  p(paste0("qpois.", pp), qpois(pp, 4.2))
  p(paste0("qexp.", pp), qexp(pp, 1.7))
}
for (x in c(0.1, 1, 2.5, 7)) {
  p(paste0("dexp.", x), dexp(x, 1.7))
  p(paste0("pexp.", x), pexp(x, 1.7))
  p(paste0("dt.", x), dt(x, 4.5))
  p(paste0("dchisq.", x), dchisq(x, 3))
  p(paste0("dnorm.", x), dnorm(x))
}
p("pnorm.upper", pnorm(1.96, lower.tail = FALSE))
p("pnorm.far", pnorm(-30))
p("pnorm.log", pnorm(-30, log.p = TRUE))
p("pt.upper", pt(2.5, 7, lower.tail = FALSE))
p("pchisq.upper", pchisq(25, 3, lower.tail = FALSE))
p("pchisq.far", pchisq(400, 3, lower.tail = FALSE))
p("pf.upper", pf(4.2, 3, 17, lower.tail = FALSE))
p("qnorm.tiny", qnorm(1e-300))
p("qt.small.df", qt(0.975, 1.5))
p("qchisq.small", qchisq(1e-10, 2))
p("qf.0.99", qf(0.99, 4, 9))
p("pbinom.edge", pbinom(20, 20, 0.35))
p("dbinom.0", dbinom(0, 20, 0.35))
p("ppois.0", ppois(0, 4.2))
p("dnorm.mu.sd", dnorm(1.3, mean = 2, sd = 0.4))
p("pnorm.mu.sd", pnorm(1.3, mean = 2, sd = 0.4))
p("qnorm.mu.sd", qnorm(0.2, mean = 2, sd = 0.4))
