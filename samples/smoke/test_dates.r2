# Phase R.T.1 smoke test — Date / POSIXct primitives.

d1 <- as.Date("2024-03-15")
d2 <- as.Date("2024-03-20")
print(d1)
print(d2)

# Arithmetic flows through numeric: Date - Date and Date + N.
print(d2 - d1)                      # 5 (numeric, days)
print(difftime(d2, d1, units="days"))
print(difftime(d2, d1, units="hours"))

# format() round-trip.
print(format.Date(d1, format="%d/%m/%Y"))

# POSIXct
t1 <- as.POSIXct("2024-03-15 12:34:56")
print(t1)
print(format.POSIXct(t1, format="%Y-%m-%d %H:%M:%S"))

# Vector of dates.
ds <- as.Date(c("2024-01-01","2024-02-01","2024-03-01"))
print(ds)

# Sys.Date / Sys.time work.
sd <- Sys.Date()
print(format.Date(sd))
