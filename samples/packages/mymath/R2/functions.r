# mymath — sample R2 addon package
# Demonstrates how to create installable R2 packages

# Factorial function
factorial <- function(n) {
    if (n <= 1) return(1)
    n * factorial(n - 1)
}

# Fibonacci
fibonacci <- function(n) {
    if (n <= 1) return(n)
    a <- 0
    b <- 1
    for (i in 2:n) {
        tmp <- b
        b <- a + b
        a <- tmp
    }
    b
}

# Greatest common divisor
gcd <- function(a, b) {
    while (b > 0) {
        tmp <- b
        b <- a %% b
        a <- tmp
    }
    a
}

# Least common multiple
lcm <- function(a, b) {
    a * b / gcd(a, b)
}

# Sigmoid function
sigmoid <- function(x) {
    1 / (1 + exp(-x))
}

# Normalize vector to [0,1]
normalize <- function(x) {
    mn <- min(x)
    mx <- max(x)
    (x - mn) / (mx - mn)
}

# Root mean squared error
rmse <- function(actual, predicted) {
    sqrt(mean((actual - predicted) ^ 2))
}

# Mean absolute error
mae <- function(actual, predicted) {
    mean(abs(actual - predicted))
}
