# glm beyond the logistic fit in lm_glm.R: poisson and gaussian families,
# the coefficient table (estimate, SE, z/t, p), deviances and AIC.
#
# Not compared: the standard errors of IRLS fits (poisson, binomial). R2
# iterates to the MLE and takes (X'WX)^-1 at its weights; R's glm.fit stops
# on a deviance-change rule and reports the QR of its last weighted step,
# whose weights are one iteration old. The two differ by ~2e-8 relative —
# R's convention, not an error in either — so they would fail the 1e-9
# tolerance. The gaussian table (no IRLS) is compared.
p <- function(k, v) cat(k, "=", sprintf("%.15g", v), "\n", sep = "")
cnt <- c(2, 3, 6, 7, 8, 9, 10, 12, 15, 20)
xx <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
gr <- c(0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
d <- data.frame(cnt = cnt, xx = xx, gr = gr)
gp <- glm(cnt ~ xx + gr, data = d, family = poisson)
cp <- coef(gp)
p("pois.b0", cp[[1]])
p("pois.bx", cp[[2]])
p("pois.bg", cp[[3]])
p("pois.dev", deviance(gp))
p("pois.nulldev", gp$null.deviance)
p("pois.aic", gp$aic)
p("pois.dfres", gp$df.residual)
p("pois.fit1", gp$fitted.values[[1]])
gg <- glm(cnt ~ xx, data = d, family = gaussian)
cg <- coef(gg)
p("gaus.b0", cg[[1]])
p("gaus.bx", cg[[2]])
p("gaus.dev", deviance(gg))
p("gaus.aic", gg$aic)
sg <- summary(gg)$coefficients
p("gaus.se.x", sg[2, 2])
p("gaus.t.x", sg[2, 3])
p("gaus.p.x", sg[2, 4])
gb <- glm(am ~ wt + hp, data = mtcars, family = binomial)
p("bin.b.wt", coef(gb)[[2]])
p("bin.aic", gb$aic)
p("bin.nulldev", gb$null.deviance)
big <- c(55, 60, 72, 80, 95, 110, 130, 150)
bx <- c(1, 2, 3, 4, 5, 6, 7, 8)
gbig <- glm(big ~ bx, data = data.frame(big = big, bx = bx), family = poisson)
p("pois.bigcount.aic", gbig$aic)
p("pois.bigcount.dev", deviance(gbig))
