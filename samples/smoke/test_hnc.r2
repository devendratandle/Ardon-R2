# Phase R.T.5b — HNC (Hindu National Calendar) smoke test.
# Format: SSSS-MM-P-TT, where Adhik months get an `A` after MM.

cat("\n--- Gudi Padwa 2024 (HNC new year) ---\n")
d1 <- as.Date("2024-04-09")    # Gudi Padwa 2024 (close approx)
h1 <- hnc.date(d1)
print(h1$formatted)             # expect ~1946-01-1-01
print(h1$masa.name)              # Chaitra

cat("\n--- Holi 2024 (Krishna Pratipada) ---\n")
d2 <- as.Date("2024-03-25")
h2 <- hnc.date(d2)
print(h2$formatted)              # expect 1946-01-2-01 (or 1945 depending on Saka cutoff)
print(h2$tithi)                  # expect 1 (Pratipada)

cat("\n--- Adhik Maas detection (rough check) ---\n")
# 2023 had Adhik Shravan from ~Jul 18 to Aug 16.
d3 <- as.Date("2023-08-01")
h3 <- hnc.date(d3)
print(h3$formatted)
print(h3$adhik)                  # may be TRUE (mean-elements approx)

cat("\n--- Tithi short codes ---\n")
t <- tithi(as.Date("2024-03-25"))
print(t$short)                   # "kru-pra"
print(t$masa.short)              # "cai"
print(t$adhik)                   # FALSE
