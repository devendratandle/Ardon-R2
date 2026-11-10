# What Ardon Studio's Tables tab reads for APA tables: coefficient names of
# factor and expression terms, cor.test's fields, anova's table, dimnames, nobs.
# Values only; the printed layouts are R2's own. Whole output.
names(coef(lm(mpg ~ wt + factor(cyl), data = mtcars)))
names(coef(lm(mpg ~ factor(cyl), data = mtcars)))
names(coef(lm(mpg ~ wt + I(wt^2) + log(hp), data = mtcars)))
names(coef(glm(am ~ wt + factor(cyl), data = mtcars, family = binomial)))
deparse(quote(x^2 + 1:3))

ct <- cor.test(mtcars$mpg, mtcars$wt)
names(ct)
ct$conf.int
ct$parameter
ct$statistic
ct$estimate
ct$null.value
ct$alternative
cl <- cor.test(mtcars$mpg, mtcars$wt, alternative = "less", conf.level = 0.9)
cl$conf.int
cl$p.value

fit <- lm(mpg ~ wt + hp, data = mtcars)
a <- anova(fit)
names(a)
a$Df
a[["Sum Sq"]]
a[["Mean Sq"]]
a[["F value"]]
a[["Pr(>F)"]]
rownames(a)
a2 <- anova(lm(mpg ~ factor(cyl) + wt, data = mtcars))
rownames(a2)
a2$Df
a2[["F value"]]

m <- matrix(1:4, 2, dimnames = list(c("r1", "r2"), c("c1", "c2")))
dimnames(m)
m
dimnames(data.frame(x = 1:3, g = c("a", "b", "c")))
is.null(dimnames(matrix(1:4, 2)))
nobs(fit)
