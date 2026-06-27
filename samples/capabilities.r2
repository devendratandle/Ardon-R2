# ─────────────────────────────────────────────────────────────────────
# Ardon-R2 capabilities tour — features beyond / extending base R.
#
#   r2 samples/capabilities.r2
#
# One file that exercises the R2-specific surface (time series, dates,
# language objects, built-in ML, factors). Self-checking: each section
# prints [PASS]/[FAIL]. If anything fails on your machine, please report it.
# ─────────────────────────────────────────────────────────────────────

ok <- function(label, cond) cat(if (isTRUE(cond)) "  [PASS] " else "  [FAIL] ", label, "\n", sep = "")

cat("=== 1. Dates ===\n")
d <- as.Date("2024-03-15")
cat("  as.Date('2024-03-15') day-count:", as.numeric(d), "\n")
ok("date arithmetic (+30 days)", as.numeric(d + 30) - as.numeric(d) == 30)
ok("difftime in days", as.numeric(as.Date("2024-03-20") - d) == 5)

cat("\n=== 2. Time series — ts() / acf / diff ===\n")
y <- ts(c(112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118),
        start = c(1949, 1), frequency = 12)
cat("  frequency:", frequency(y), " start:", start(y), "\n")
ok("acf runs over the series", length(acf(as.numeric(y), 4)) >= 4)
ok("diff() drops one element", length(diff(as.numeric(y))) == length(y) - 1)

cat("\n=== 3. Language objects (quote / eval / substitute) ===\n")
e <- quote(a * b + 1)
a <- 6; b <- 7
cat("  eval(quote(a*b+1)) =", eval(e), "\n")
ok("eval matches direct", eval(e) == a * b + 1)
ok("deparse round-trips", deparse(quote(x + y)) == "x + y")

cat("\n=== 4. Built-in machine learning (no packages) ===\n")
km <- kmeans(iris[, 1:4], centers = 3)
ok("kmeans: 3 clusters over 150 rows", sum(km$size) == 150)
fitrf <- rf(Species ~ ., data = iris, ntrees = 20)
ok("random forest trains", !is.null(fitrf))
tree <- rpart(Sepal.Length ~ Petal.Length, data = iris)
ok("decision tree trains", !is.null(tree))

cat("\n=== 5. Factors & grouped aggregation ===\n")
f <- factor(c("lo", "hi", "lo", "mid", "hi"), levels = c("lo", "mid", "hi"))
cat("  levels:", levels(f), "\n")
ok("as.integer(factor) gives codes", all(as.integer(f) == c(1, 3, 1, 2, 3)))
tg <- tapply(iris$Sepal.Length, iris$Species, mean)
cat("  mean Sepal.Length by species:", round(as.numeric(tg), 3), "\n")
ok("tapply: 3 species groups", length(tg) == 3)

cat("\nCAPABILITIES TOUR DONE.\n")
