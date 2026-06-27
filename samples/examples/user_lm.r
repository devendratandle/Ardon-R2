# ═══════════════════════════════════════════════════════════════════════
# Example: Writing your own lm() in R2 language
# This shows how users can build statistical functions WITHOUT Rust
# Heavy math calls .Internal() → runs in Rust
# Everything else is pure R2
# ═══════════════════════════════════════════════════════════════════════

my_lm <- function(x, y) {
    # x is a matrix of predictors, y is response vector
    n <- length(y)
    
    # Add intercept column (column of 1s)
    ones <- rep(1, n)
    X <- cbind(matrix(ones, nrow = n, ncol = 1), x)
    p <- ncol(X)
    
    # Solve normal equations: beta = (X'X)^-1 X'y
    # .Internal calls Rust for the matrix math
    beta <- .Internal("solve_lstsq", X, y)
    
    # Fitted values: y_hat = X * beta
    fitted <- rep(0, n)
    sapply(1:n, function(i) {
        s <- 0
        sapply(1:p, function(j) {
            s <- s + X[i, j] * beta[j]
        })
    })
    
    # For now, simple matrix multiply
    y_hat <- X %*% matrix(beta, nrow = p, ncol = 1)
    
    # Residuals
    residuals <- y - y_hat
    
    # R-squared
    y_mean <- mean(y)
    ss_res <- sum(residuals^2)
    ss_tot <- sum((y - y_mean)^2)
    r_squared <- 1 - ss_res / ss_tot
    
    # Print results
    cat("My Custom Linear Regression\n")
    cat(f"Coefficients: {beta}\n")
    cat(f"R-squared: {r_squared}\n")
    cat(f"Residual SE: {sqrt(ss_res / (n - p))}\n")
    
    beta
}

# Test it:
cat("=== Testing user-defined lm() ===\n")
x <- matrix(mtcars$wt, nrow = 32, ncol = 1)
y <- mtcars$mpg
my_lm(x, y)
