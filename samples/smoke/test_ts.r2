# Phase R.T.2 — ts() smoke test.

# Monthly series starting Jan 1960, 24 obs (= ends Dec 1961).
x <- ts(1:24, start = c(1960, 1), frequency = 12)
print(x)

# Accessors.
print(start(x))      # c(1960, 1)
print(end(x))        # c(1961, 12)
print(frequency(x))  # 12
print(deltat(x))     # 0.0833...
print(is.ts(x))      # TRUE

# Time index and cycle (period within year).
print(time(x))
print(cycle(x))      # 1..12, 1..12

# Quarterly series.
q <- ts(c(10, 12, 15, 13, 11, 14, 16, 13), start = c(2020, 1), frequency = 4)
print(q)

# Sub-window: Q2 2020 through Q3 2021.
w <- window(q, start = c(2020, 2), end = c(2021, 3))
print(w)
print(start(w))
print(end(w))
