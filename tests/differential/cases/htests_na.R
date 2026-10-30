# t.test with missing values. R drops the whole ROW: in the formula form
# through model.frame's na.omit, in the paired form by complete.cases(x, y).
# R2 used to drop NAs from the response alone ("LHS length (7) != RHS
# length (8)") and, paired, to shift every pair after the first NA.
p <- function(k, v) cat(k, "=", sprintf("%.15g", v), "\n", sep = "")
d <- data.frame(score = c(5.1, NA, 6.2, 5.8, 7.0, 6.6, 7.4, 6.9, 6.1),
                grp = c("a", "a", "a", "a", "b", "b", "b", "b", "b"))
t1 <- t.test(score ~ grp, data = d)
p("fna.stat", t1$statistic)
p("fna.df", t1$parameter)
p("fna.p", t1$p.value)
p("fna.lo", t1$conf.int[1])
d2 <- data.frame(score = c(5.1, 5.6, 6.2, 5.8, 7.0, 6.6, 7.4, 6.9, 6.1),
                 grp = c("a", "a", NA, "a", "b", "b", "b", "b", "b"))
t2 <- t.test(score ~ grp, data = d2)
p("gna.stat", t2$statistic)
p("gna.df", t2$parameter)
d3 <- data.frame(score = c(5.1, NA, 6.2, 5.8, 7.0, 6.6, NA, 6.9, 6.1),
                 grp = c("a", "a", NA, "a", "b", "b", "b", "b", "b"))
t3 <- t.test(score ~ grp, data = d3, var.equal = TRUE)
p("both.stat", t3$statistic)
p("both.df", t3$parameter)
p("both.p", t3$p.value)
x <- c(5.1, 4.9, NA, 5.8, 6.0, 5.5, 5.3, 6.4)
y <- c(6.3, 5.8, 7.1, 6.0, NA, 6.6, 5.9, 7.4)
tp <- t.test(x, y, paired = TRUE)
p("pair.stat", tp$statistic)
p("pair.df", tp$parameter)
p("pair.p", tp$p.value)
p("pair.hi", tp$conf.int[2])
tw <- t.test(x, y)
p("welch.na.stat", tw$statistic)
p("welch.na.df", tw$parameter)
# the owner's example: a factor group, NA in the second group's response
du <- data.frame(score = c(81.5, 77, 92.25, NA, 68, 88, 95.5, 73),
                 grp = factor(c("a", "b", "a", "b", "a", "b", "a", "b")))
tu <- t.test(score ~ grp, data = du)
p("owner.stat", tu$statistic)
p("owner.df", tu$parameter)
p("owner.p", tu$p.value)
p("owner.lo", tu$conf.int[1])
p("owner.hi", tu$conf.int[2])
