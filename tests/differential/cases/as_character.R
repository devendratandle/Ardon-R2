# Numbers as text: as.character() and paste() use 15 significant digits and
# scientific notation where it is no wider, as R does. Compared as exact strings.
kv <- function(k, v) cat(k, "=[", paste(v, collapse = "|"), "]\n", sep = "")

kv("third", as.character(1/3))
kv("tenths", paste(0.1 + 0.2))
kv("pi", as.character(pi))
kv("big", as.character(c(123456, 1e5, 1e15, 1234567890123)))
kv("small", as.character(c(0.0001, 0.00012345, 1e-20)))
kv("mixed", paste("x", c(1.5, 2, -3.25e-7)))
kv("ints", as.character(c(1, 10, 100, 1000)))
kv("negzero", as.character(c(-1.5, 2.000001)))
kv("special", as.character(c(NA, NaN, Inf, -Inf)))
kv("paste0_num", paste0("v", 1e5))
m <- matrix(c(1.5, 2.25, 3, 4.125), 2)
kv("sprintf_matrix", sprintf("%.2f", m))
kv("sprintf_matrix_d", sprintf("%g", m * 10))
kv("paste_matrix", paste(m))
