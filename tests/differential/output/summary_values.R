fit <- lm(mpg ~ wt, data = mtcars)
s <- summary(fit)
class(s)
s$coefficients
sv <- summary(c(1, 2, 3, 4, 100))
sv
inherits(sv, "summaryDefault")
sv[["Median"]]
summary(c(1.5, 2.25, NA))
a <- anova(fit)
inherits(a, "anova")
st <- system.time(for (i in 1:3) i)
class(st)
