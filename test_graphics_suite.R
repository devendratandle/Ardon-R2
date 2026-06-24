# ============================================================
#  Graphics suite — exercises every plot type + file devices.
#  Run as a script; writes files into ./gtest/ for inspection.
# ============================================================
cat("=== multi-page PDF: 4 high-level plots ===\n")
pdf("gtest_multi.pdf", width = 7, height = 5)
plot(mtcars$wt, mtcars$mpg, main = "MPG vs Weight", xlab = "wt", ylab = "mpg",
     col = "blue", pch = 19)
hist(iris$Sepal.Length, main = "Sepal Length", col = "lightgray")
boxplot(Sepal.Length ~ Species, data = iris, main = "By Species")
barplot(table(mtcars$cyl), main = "Cylinder counts", col = "steelblue")
dev.off()

cat("=== PNG scatter with overlays ===\n")
png("gtest_scatter.png", width = 640, height = 480)
plot(iris$Petal.Length, iris$Petal.Width,
     col = iris$Species, pch = 19, cex = 1.2,
     main = "Iris petals", xlab = "length", ylab = "width")
abline(lm(Petal.Width ~ Petal.Length, data = iris), col = "red", lwd = 2)
legend("topleft", legend = levels(iris$Species), col = 1:3, pch = 19)
text(1.5, 2.0, "annotation")
dev.off()

cat("=== SVG pairs matrix ===\n")
svg("gtest_pairs.svg")
pairs(iris[1:4], col = iris$Species)
dev.off()

cat("=== SVG curve + overlays ===\n")
svg("gtest_curve.svg")
curve(sin(x), from = 0, to = 2 * pi, main = "sin and cos")
curve(cos(x), add = TRUE, col = "red")
points(pi, 0, col = "blue", pch = 19)
lines(c(0, 2*pi), c(0, 0), col = "gray")
dev.off()

cat("done writing graphics files.\n")
