# R2 Demo — Basic Statistical Computing
# Run: source("samples/demo_basics.r")

cat("=== R2 Basic Statistics Demo ===\n\n")

# Vectors and basic stats
x <- c(23, 45, 12, 67, 34, 89, 56, 78, 90, 11)
cat(f"Mean: {mean(x)}\n")
cat(f"SD: {sd(x)}\n")
cat(f"Median: {median(x)}\n")

# Data frame operations
cat("\n=== Iris Data ===\n")
summary(iris)

cat("\n=== Filtered: Sepal.Length > 7 ===\n")
big <- filter(iris, iris$Sepal.Length > 7)
print(big)

# Linear regression
cat("\n=== Linear Regression ===\n")
model <- lm(mpg ~ wt + hp, data = mtcars)
summary(model)
