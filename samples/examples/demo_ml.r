# R2 Demo — Machine Learning (all built-in, no packages needed)
# Run: source("samples/demo_ml.r")

cat("=== R2 Machine Learning Demo ===\n\n")

# Decision Tree
cat("--- Decision Tree ---\n")
rpart(Petal.Length ~ ., data = iris)

# Random Forest
cat("\n--- Random Forest ---\n")
rf(Petal.Length ~ ., data = iris, ntrees = 50)

# Gradient Boosting
cat("\n--- Gradient Boosting ---\n")
g <- gbm(mpg ~ ., data = mtcars, ntrees = 100)
summary(g)

# K-means Clustering
cat("\n--- K-means ---\n")
x <- matrix(c(iris$Sepal.Length, iris$Sepal.Width, iris$Petal.Length, iris$Petal.Width), nrow = 150, ncol = 4)
km <- kmeans(x, centers = 3)
summary(km)

# PCA
cat("\n--- PCA ---\n")
prcomp(x)

cat("\n=== All 12 ML algorithms available with zero install ===\n")
