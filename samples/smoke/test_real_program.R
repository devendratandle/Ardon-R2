# ============================================================
#  Real-world analysis program — end-to-end smoke test
#  Datasets: iris (factor) + mtcars (regression)
# ============================================================

cat("===== 1. Data inspection =====\n")
cat("iris dim:", dim(iris), "\n")
cat("species levels:", levels(iris$Species), "\n")
cat("mtcars rows:", nrow(mtcars), " cols:", ncol(mtcars), "\n\n")

cat("===== 2. Group summaries (factor-aware) =====\n")
for (sp in levels(iris$Species)) {
    sl <- iris$Sepal.Length[iris$Species == sp]
    cat(sprintf("  %-12s n=%d  mean SL=%.3f  sd=%.3f\n", sp, length(sl), mean(sl), sd(sl)))
}
cat("\n")

cat("===== 3. tapply: mean petal length by species =====\n")
print(tapply(iris$Petal.Length, iris$Species, mean))
cat("\n")

cat("===== 4. Two-sample t-test (setosa vs versicolor) =====\n")
two <- iris[iris$Species != "virginica", ]
tt <- t.test(Sepal.Length ~ Species, data = two)
cat("  t-statistic:", round(tt$statistic, 3), "\n")
cat("  (R reference: t = -10.521)\n\n")

cat("===== 5. Linear regression mpg ~ wt + hp (mtcars) =====\n")
fit <- lm(mpg ~ wt + hp, data = mtcars)
print(summary(fit))
cat("  (R reference coefs: 37.227, -3.878, -0.0318; R^2 = 0.8268)\n\n")

cat("===== 6. Correlation =====\n")
cat("  cor(wt, mpg):", round(cor(mtcars$wt, mtcars$mpg), 4), " (R: -0.8677)\n\n")

cat("===== 7. apply family + user-defined function =====\n")
zscore <- function(v) (v - mean(v)) / sd(v)
z <- sapply(mtcars[c("mpg","wt","hp")], function(col) round(mean(zscore(col)), 10))
cat("  mean z-score of each col (should be ~0):", z, "\n\n")

cat("===== 8. Control flow + accumulation =====\n")
fib <- function(n) {
    a <- 0; b <- 1
    for (i in 1:n) { t <- a + b; a <- b; b <- t }
    a
}
cat("  fib(1..10):", sapply(1:10, fib), "\n\n")

cat("===== 9. String / data wrangling =====\n")
hp_class <- ifelse(mtcars$hp > 150, "high", "low")
cat("  high-hp cars:", sum(hp_class == "high"), " low-hp:", sum(hp_class == "low"), "\n")
cat("  mean mpg high-hp:", round(mean(mtcars$mpg[hp_class == "high"]), 2),
    " low-hp:", round(mean(mtcars$mpg[hp_class == "low"]), 2), "\n\n")

cat("===== 10. Metaprogramming (Phase L) =====\n")
wt <- mtcars$wt; hp <- mtcars$hp   # must exist (R2 is eager; see caveat)
describe <- function(x) cat("  you passed:", deparse(substitute(x)), "\n")
describe(log(wt) + hp^2)
cat("  eval(parse):", eval(parse(text = "sum(1:100)")), " (R: 5050)\n\n")

cat("ALL SECTIONS COMPLETED.\n")
