# Factor level order: R sorts the default levels (numbers numerically,
# never as strings), NA is never a level, and an explicit `levels=` order
# is kept as given. Grouping functions follow the same order.
j <- function(x) paste(x, collapse = ",")

# character input
x <- c("b", "a", "b", NA, "c")
f <- as.factor(x)
cat("chr.levels=", j(levels(f)), "\n", sep = "")
cat("chr.codes=", j(as.integer(f)), "\n", sep = "")
cat("chr.nlevels=", nlevels(f), "\n", sep = "")
cat("chr.na=", sum(is.na(f)), "\n", sep = "")
cat("chr.factor.levels=", j(levels(factor(x))), "\n", sep = "")
cat("chr.factor.codes=", j(as.integer(factor(x))), "\n", sep = "")

# numeric input sorts numerically: 2 before 10
n <- c(10, 2, 2.5, NA, -1, 100, 2)
cat("num.levels=", j(levels(as.factor(n))), "\n", sep = "")
cat("num.codes=", j(as.integer(as.factor(n))), "\n", sep = "")
cat("num.factor.levels=", j(levels(factor(n))), "\n", sep = "")
cat("num.labels=", j(levels(factor(c(1e5, 0.1 + 0.2, 1/3)))), "\n", sep = "")
cat("num.nan=", j(levels(factor(c(1, NaN, NA)))), "\n", sep = "")
cat("int.levels=", j(levels(as.factor(c(30L, 4L, 4L)))), "\n", sep = "")
cat("lgl.levels=", j(levels(factor(c(TRUE, FALSE, TRUE)))), "\n", sep = "")

# explicit levels keep their order; values outside them become NA
e <- factor(c("lo", "hi", "mid", "hi", "other"), levels = c("lo", "mid", "hi"))
cat("explicit.levels=", j(levels(e)), "\n", sep = "")
cat("explicit.codes=", j(as.integer(e)), "\n", sep = "")
cat("explicit.na=", sum(is.na(e)), "\n", sep = "")
en <- factor(c(1, 3, 2), levels = c(3, 1, 2))
cat("explicit.num.levels=", j(levels(en)), "\n", sep = "")
cat("explicit.num.codes=", j(as.integer(en)), "\n", sep = "")

# a factor: as.factor keeps it, factor() drops unused levels in order
u <- factor(c("a", "c"), levels = c("c", "b", "a"))
cat("asfactor.keep=", j(levels(as.factor(u))), "\n", sep = "")
cat("factor.drop=", j(levels(factor(u))), "\n", sep = "")
cat("factor.drop.codes=", j(as.integer(factor(u))), "\n", sep = "")

# grouping follows the level order and drops NA keys
cat("split.names=", j(names(split(1:5, x))), "\n", sep = "")
cat("split.num.names=", j(names(split(1:4, c(10, 2, 10, 2)))), "\n", sep = "")
cat("split.num.b=", j(split(1:4, c(10, 2, 10, 2))[["10"]]), "\n", sep = "")
cat("split.explicit.names=", j(names(split(1:5, e))), "\n", sep = "")
cat("tapply.names=", j(names(tapply(1:5, x, sum))), "\n", sep = "")
cat("tapply.sums=", j(unlist(tapply(1:5, x, sum))), "\n", sep = "")
cat("table.names=", j(names(table(c(10, 2, NA, 2)))), "\n", sep = "")
cat("table.counts=", j(as.integer(table(c(10, 2, NA, 2)))), "\n", sep = "")
cat("table.chr.names=", j(names(table(x))), "\n", sep = "")

# a character predictor's reference level is the alphabetically first
fit <- lm(y ~ g, data = data.frame(g = c("b", "a", "b", "a"), y = c(1, 2, 3, 5)))
cat("lm.coef.names=", j(names(coef(fit))), "\n", sep = "")
cat("lm.coef.g=", coef(fit)[2], "\n", sep = "")
tt <- t.test(y ~ g, data = data.frame(g = c("b", "b", "b", "a", "a", "a"), y = c(1, 2, 4, 5, 7, 6)))
cat("ttest.est1=", tt$estimate[1], "\n", sep = "")
cat("ttest.est2=", tt$estimate[2], "\n", sep = "")
