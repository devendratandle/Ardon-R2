# Phase R.T.3 — xts() smoke test.

# Build an xts of daily closes for 6 trading days.
dates <- as.Date(c("2024-01-02","2024-01-03","2024-01-04",
                   "2024-01-08","2024-01-09","2024-01-10"))
prices <- c(100.5, 101.3, 99.8, 102.1, 103.4, 102.9)
x <- xts(prices, order.by = dates)
print(x)

# Accessors.
print(is.xts(x))    # TRUE
print(index(x))     # 6 dates
print(coredata(x))  # 6x1 matrix of prices

# Date-string subsetting (xts package's hallmark feature).
print(xts.subset(x, "2024-01-03/2024-01-08"))   # rows 2..4
print(xts.subset(x, "2024-01"))                  # whole month
print(xts.subset(x, "/2024-01-04"))              # from start through Jan 4
print(xts.subset(x, "2024-01-08/"))              # from Jan 8 onward

# first / last
print(first(x, 2))
print(last(x, 2))

# na.locf with a vector containing NAs.
v <- c(1, NA, NA, 5, NA, 7)
print(na.locf(v))

# merge.xts: outer-join two series with overlapping but different dates.
y_dates <- as.Date(c("2024-01-03","2024-01-04","2024-01-05","2024-01-08"))
y <- xts(c(50, 51, 52, 53), order.by = y_dates)
m <- merge.xts(x, y)
print(m)
