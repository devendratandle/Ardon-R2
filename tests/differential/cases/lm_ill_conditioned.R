# Standard errors on a near-collinear design. x2 differs from x1 by a
# small quadratic term: kappa(X) ~ 4e5, kappa(X'X) ~ 1.6e11. SEs from
# inverting X'X are off by 1.2e-6 relative (measured in R), which this
# harness's 1e-9 gate rejects; SEs from the QR's R factor (chol2inv) pass.
# confint() = coef +/- t * SE, so it exposes the SEs in both engines.
x1 <- seq(1, 40)
x2 <- x1 + 1e-6 * x1 * x1
y  <- 2 + 0.5 * x1 - 0.25 * x2 + sin(x1)
fit <- lm(y ~ x1 + x2)
co <- coef(fit)
cat("ill.b0=", co[[1]], "\n", sep = "")
cat("ill.b1=", co[[2]], "\n", sep = "")
cat("ill.b2=", co[[3]], "\n", sep = "")
ci <- confint(fit)
cat("ill.ci.b1.lo=", ci[2, 1], "\n", sep = "")
cat("ill.ci.b1.hi=", ci[2, 2], "\n", sep = "")
cat("ill.ci.b2.lo=", ci[3, 1], "\n", sep = "")
cat("ill.ci.b2.hi=", ci[3, 2], "\n", sep = "")
s <- summary(fit)
cat("ill.r2=", s$r.squared, "\n", sep = "")
cat("ill.sigma=", s$sigma, "\n", sep = "")
# Binomial glm near separation: the IRLS weights make X'WX far worse
# conditioned than W^(1/2) X.
z <- c(0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1)
w <- seq(1, 20)
g <- glm(z ~ w, family = binomial)
gc <- coef(g)
cat("sep.b0=", gc[[1]], "\n", sep = "")
cat("sep.b1=", gc[[2]], "\n", sep = "")
cat("sep.dev=", deviance(g), "\n", sep = "")
