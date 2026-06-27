# Phase R.T.5 — period aggregation + Hindu calendar smoke test.

# Daily prices over 2 months.
dates <- as.Date(c("2024-01-02","2024-01-09","2024-01-16","2024-01-23","2024-01-30",
                   "2024-02-06","2024-02-13","2024-02-20","2024-02-27"))
px <- c(100, 102, 99, 105, 103, 108, 107, 110, 109)
x <- xts(px, order.by = dates)

# Monthly aggregation — average per month.
print(to.monthly(x, FUN = "mean"))

# Sum per month.
print(to.monthly(x, FUN = "sum"))

# Last value of each month (close).
print(to.monthly(x, FUN = "last"))

# Hindu calendar
d <- as.Date("2024-03-25")     # Holi 2024 was around this date.
t <- tithi(d)
print(t$tithi)
print(t$name)
print(t$paksha)
print(t$masa)

hd <- hindu.date(d)
print(hd$formatted)
print(hd$saka.year)            # ~1945-1946 Saka
