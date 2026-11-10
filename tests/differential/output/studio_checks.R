# Results Ardon Studio found differing from R: two-way table(), wilcox.test,
# the airquality data. Also named matrix indexing and `$` before an R2 keyword.
# Whole output.
t1 <- table(c("a", "b", "a", "b"), c("x", "x", "y", "y"))
t1
t2 <- table(mtcars$cyl, mtcars$gear)
t2
dim(t2)
as.vector(t2)
t2[2, 3]
t2["6", "4"]
sum(t2)
class(t2)
t3 <- table(cyl = mtcars$cyl, gear = mtcars$gear)
t3
names(dimnames(t3))
dimnames(t3)$gear
with(mtcars, table(am, vs))
table(c("long label", "b"), c("x", "yy"))
x <- c("p", "q", "p")
table(x)

m <- matrix(1:6, 2, dimnames = list(c("r1", "r2"), c("a", "b", "c")))
m["r2", "b"]
m["r1", ]
m[, "c"]
m[, c("a", "c")]

w <- wilcox.test(mpg ~ am, data = mtcars, exact = FALSE)
w$p.value
w$statistic
w$method
x <- c(1.1, 2.2, 3.3, 4.4); y <- c(2.5, 3.6, 5.7, 6.8)
e <- wilcox.test(x, y)
e$p.value
e$statistic
e$method
wilcox.test(x, y, alternative = "less")$p.value
wilcox.test(x, y, alternative = "greater")$p.value
wilcox.test(x, y, correct = FALSE, exact = FALSE)$p.value
wilcox.test(c(1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30),
            c(0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29), paired = TRUE)$p.value
s <- wilcox.test(c(-1.2, 2.5, 3.1, 0.4, 5.5, -0.8, 2.2), mu = 0)
s$statistic
s$p.value
s$method
names(s)
wilcox.test(1:60 + 0.5, 41:100, exact = FALSE)$p.value
numeric(0) + 1:3
length(integer(0) > 1)

dim(airquality)
sum(is.na(airquality$Ozone))
sum(is.na(airquality$Solar.R))
mean(airquality$Temp)
colMeans(airquality[, 3:6])
airquality[153, ]
nobs(lm(Ozone ~ Wind, data = airquality))
