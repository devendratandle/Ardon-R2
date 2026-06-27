# R2 Demo — Performance Benchmark
# Run: source("samples/demo_benchmark.r")
# Then run the same in R to compare

cat("=== R2 Performance Benchmark ===\n\n")

# Matrix multiply
cat("Matrix multiply 1000x1000:\n")
x <- matrix(rnorm(1000000), nrow = 1000, ncol = 1000)
system.time(t(x) %*% x)

# Linear regression
cat("\nlm() with 10K rows, 10 predictors:\n")
set.seed(42)
x <- matrix(rnorm(100000), nrow = 10000, ncol = 10)
y <- rnorm(10000)
system.time(lm(y ~ x))

# Gradient boosting
cat("\nGBM 100 trees on mtcars:\n")
system.time(gbm(mpg ~ ., data = mtcars, ntrees = 100))

cat("\n=== Copy these commands to R for comparison ===\n")
