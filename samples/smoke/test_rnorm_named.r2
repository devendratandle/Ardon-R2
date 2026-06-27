# Verify the named-n bug fix.

# Style 1 — your original call: n given as a named arg.
x <- rnorm(mean = 10, sd = 2, n = 100000)
cat("Length of x (should be 100000):", length(x), "\n")
cat("Mean   (should be ~10):", mean(x), "\n")
cat("Stddev (should be ~2):  ", sd(x), "\n")

# Style 2 — positional first.
y <- rnorm(50, mean = 5, sd = 1)
cat("Length of y (should be 50):", length(y), "\n")

# Style 3 — all positional.
z <- rnorm(20)
cat("Length of z (should be 20):", length(z), "\n")

# Same fix applies to runif / rbinom / rpois.
u <- runif(min = 0, max = 100, n = 1000)
cat("Length of u (should be 1000):", length(u), "\n")
