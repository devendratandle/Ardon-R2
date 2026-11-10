# What Gview reads besides the SVG (which crates/r2-repl/tests/ide_protocol.rs
# checks): density's fields, barplot's bar midpoints, and the data and rounding
# its tables show. Whole output.
d <- density(c(1, 2, 3, 4, 10))
names(d)
d$bw
d$n
d$data.name
class(d)
length(d$x)
range(d$x)
density(mtcars$mpg)$bw
density(faithful$eruptions)$bw
density(airquality$Wind, bw = 2)$bw
pdf(NULL)
bp <- barplot(c(a = 1, b = 3, c = 2))
bp
bh <- barplot(c(4, 5), horiz = TRUE)
bh
invisible(dev.off())

round(3.5965625, 6)
round(2.5)
round(0.125, 2)
round(-1.005, 2)
round(123.456, -1)
round(c(0.0045, 2.675), 2)
round(0.1 + 0.2, 15)
sapply(mtcars[, 1:3], mean)
sapply(list(a = 1:3, b = 4:6), sum)
sapply(c(x = 1, y = 2), function(v) v * 2)

dim(iris)
colMeans(iris[, 1:4])
sapply(iris[, 1:4], sd)
table(iris$Species)
colMeans(faithful)
str(ToothGrowth)
levels(ToothGrowth$supp)
