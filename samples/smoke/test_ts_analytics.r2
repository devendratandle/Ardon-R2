# Phase R.T.4 — TS analytics smoke test.

# AR(1) toy series: x_t = 0.7 * x_{t-1} + noise. Should give r(1) ≈ 0.7.
set.seed(1)
n <- 200
x <- rep(0.0, n)
for (i in 2:n) { x[i] <- 0.7 * x[i-1] + rnorm(1) }

# ACF should decay geometrically from ~0.7.
a <- acf(x, lag.max = 5)
print(a$acf)

# PACF should spike at lag 1 then ~0.
p <- pacf(x, lag.max = 5)
print(p$acf)

# Seasonal decomposition — monthly series with trend + seasonality.
mts <- ts(c(10,12,15,14,13,18,20,19,17,15,13,11,
            12,14,17,16,15,20,22,21,19,17,15,13,
            14,16,19,18,17,22,24,23,21,19,17,15),
          start = c(2020, 1), frequency = 12)
d <- decompose(mts)
print(d$figure)        # seasonal pattern (length 12)
print(d$trend)         # centered MA (some NA at ends)

# Regularity / periodicity.
print(is.regular(mts))                              # TRUE — ts is regular
dates <- as.Date(c("2024-01-02","2024-01-03","2024-01-04","2024-01-05","2024-01-08"))
xx <- xts(c(1,2,3,4,5), order.by = dates)
print(is.regular(xx))                               # FALSE — Mon..Thu then Mon
print(periodicity(xx)$scale)                        # "daily"

# Lag and diff of a ts.
small <- ts(c(10, 12, 15, 14, 13), start = 2020, frequency = 1)
print(diff_ts(small))                              # length 4: differences
print(diff_ts(small, lag = 2))                     # length 3

# Vector lag (NA-pad style).
print(lag(c(10, 20, 30, 40), k = 1))               # NA 10 20 30
print(lag(c(10, 20, 30, 40), k = -1))              # 20 30 40 NA
