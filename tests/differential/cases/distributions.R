# Distribution functions against R: the chi-squared family (both tails,
# density, quantiles) across df from 0.5 to 10000, and the lower.tail /
# log.p flags of every p- and q-function R2 has.
#
# Tiny tails are compared through log.p = TRUE: the harness compares
# values near zero with an absolute tolerance, which would let a wrong
# 1e-22 pass as 0, while their logs are large and compared relatively.
kv <- function(k, v) cat(k, "=", sprintf("%.17g", v), "\n", sep = "")

dfs <- c(0.5, 1, 1.5, 2, 2.005, 3, 5, 10, 50, 100, 1000, 10000)
xs  <- c(0.001, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 300, 1000, 10100)
for (i in seq_along(dfs)) for (j in seq_along(xs)) {
  df <- dfs[i]; x <- xs[j]; k <- paste0("_", i, "_", j)
  kv(paste0("lp", k), pchisq(x, df, log.p = TRUE))
  kv(paste0("lq", k), pchisq(x, df, lower.tail = FALSE, log.p = TRUE))
  kv(paste0("d", k), dchisq(x, df))
}
ps <- c(1e-10, 0.01, 0.5, 0.95, 0.999999)
for (i in seq_along(dfs)) for (j in seq_along(ps)) {
  k <- paste0("_", i, "_", j)
  kv(paste0("q", k), qchisq(ps[j], dfs[i]))
  kv(paste0("qu", k), qchisq(ps[j], dfs[i], lower.tail = FALSE))
}
# tails far below what 1 - p can express
for (i in c(1, 2, 7, 10, 11)) {
  kv(paste0("qtiny_", i), qchisq(1e-100, dfs[i], lower.tail = FALSE))
  kv(paste0("qlog_", i), qchisq(-200, dfs[i], lower.tail = FALSE, log.p = TRUE))
}
# recycling over df
r <- pchisq(3, c(1, 2, 5)); kv("rec_len", length(r)); kv("rec_3", r[3])
r <- qchisq(c(0.1, 0.9), c(2, 20)); kv("recq_1", r[1]); kv("recq_2", r[2])
# chisq.test's p-value, straight from the upper tail
kv("chisq_test_logp", log(chisq.test(matrix(c(900, 100, 100, 900), 2))$p.value))

# lower.tail / log.p in the other families
kv("pnorm_up", pnorm(3, lower.tail = FALSE))
kv("pnorm_up_log", pnorm(10, lower.tail = FALSE, log.p = TRUE))
kv("pnorm_lo_log", pnorm(-40, log.p = TRUE))
kv("pnorm_hi_log", pnorm(10, log.p = TRUE) * 1e20)
kv("qnorm_up", qnorm(1e-30, lower.tail = FALSE))
kv("qnorm_log", qnorm(-50, log.p = TRUE))
kv("pt_up", pt(10, 5, lower.tail = FALSE))
kv("qt_up", qt(0.001, 7, lower.tail = FALSE))
kv("pf_up", pf(50, 3, 10, lower.tail = FALSE))
kv("qf_up", qf(0.05, 4, 20, lower.tail = FALSE))
kv("pexp_up_log", pexp(50, 2, lower.tail = FALSE, log.p = TRUE))
kv("qexp_lo", qexp(1e-10, 2))
kv("qexp_up", qexp(1e-10, 2, lower.tail = FALSE))
kv("pbinom_up", pbinom(3, 10, 0.3, lower.tail = FALSE))
kv("ppois_up", ppois(2, 4, lower.tail = FALSE))
