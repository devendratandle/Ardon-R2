# R-side accuracy comparison script.
# Runs nine standard analyses on built-in datasets and prints results
# in a parseable `KEY=VALUE` format. Pair with `accuracy.r2` for R2.
#
# Usage:  Rscript accuracy.R > out_R.txt
# Then:   diff or eyeball against `out_R2.txt` (numeric tolerance ~1e-3).

options(digits = 12)

# ── 1. Descriptive stats on iris$Sepal.Length ─────────────────────────
x <- iris$Sepal.Length
cat(sprintf("desc.mean=%.10f\n",   mean(x)))
cat(sprintf("desc.median=%.10f\n", median(x)))
cat(sprintf("desc.sd=%.10f\n",     sd(x)))
cat(sprintf("desc.var=%.10f\n",    var(x)))
cat(sprintf("desc.min=%.10f\n",    min(x)))
cat(sprintf("desc.max=%.10f\n",    max(x)))

# ── 2. Linear regression: mpg ~ wt + hp on mtcars ─────────────────────
fit <- lm(mpg ~ wt + hp, data = mtcars)
co  <- coef(fit)
s   <- summary(fit)
cat(sprintf("lm.intercept=%.10f\n",  co["(Intercept)"]))
cat(sprintf("lm.beta_wt=%.10f\n",    co["wt"]))
cat(sprintf("lm.beta_hp=%.10f\n",    co["hp"]))
cat(sprintf("lm.r_squared=%.10f\n",  s$r.squared))
cat(sprintf("lm.adj_r_squared=%.10f\n", s$adj.r.squared))
cat(sprintf("lm.f_stat=%.10f\n",     s$fstatistic[1]))

# ── 3. GLM logistic: am ~ wt + hp on mtcars ───────────────────────────
gfit <- glm(am ~ wt + hp, family = binomial, data = mtcars)
gc   <- coef(gfit)
gs   <- summary(gfit)
cat(sprintf("glm.intercept=%.10f\n", gc["(Intercept)"]))
cat(sprintf("glm.beta_wt=%.10f\n",   gc["wt"]))
cat(sprintf("glm.beta_hp=%.10f\n",   gc["hp"]))
cat(sprintf("glm.null_deviance=%.10f\n",     gfit$null.deviance))
cat(sprintf("glm.residual_deviance=%.10f\n", gfit$deviance))
cat(sprintf("glm.aic=%.10f\n",                gfit$aic))

# ── 4. Two-sample t.test (Welch): setosa vs versicolor petal length ───
x <- iris$Petal.Length[iris$Species == "setosa"]
y <- iris$Petal.Length[iris$Species == "versicolor"]
tt <- t.test(x, y)
cat(sprintf("ttest.statistic=%.10f\n", tt$statistic))
cat(sprintf("ttest.df=%.10f\n",        tt$parameter))
cat(sprintf("ttest.p_value=%.10g\n",   tt$p.value))

# ── 5. SVD: reconstruction on a 5×3 numeric matrix ────────────────────
A <- matrix(c(1,2,3,4,5,  10,20,30,40,50,  100,200,300,400,510), 5, 3)
sv <- svd(A)
A_reconstructed <- sv$u %*% diag(sv$d) %*% t(sv$v)
err <- sqrt(sum((A - A_reconstructed)^2))
cat(sprintf("svd.d_1=%.10f\n", sv$d[1]))
cat(sprintf("svd.d_2=%.10f\n", sv$d[2]))
cat(sprintf("svd.d_3=%.10f\n", sv$d[3]))
cat(sprintf("svd.reconstruction_error=%.10g\n", err))

# ── 6. Eigendecomposition: a known 3×3 symmetric matrix ───────────────
# Avoids cov-on-matrix (R2's cov is two-vector scalar; R's is matrix).
# This 3×3 has eigenvalues approximately {7.288, 2.133, 0.579} per R.
A_eig <- matrix(c(4, 1, 1,  1, 3, 2,  1, 2, 3), 3, 3)
e <- eigen(A_eig)
cat(sprintf("eigen.lambda_1=%.10f\n", e$values[1]))
cat(sprintf("eigen.lambda_2=%.10f\n", e$values[2]))
cat(sprintf("eigen.lambda_3=%.10f\n", e$values[3]))

# ── 7. ANOVA: Sepal.Length by Species ─────────────────────────────────
a <- aov(Sepal.Length ~ Species, data = iris)
asum <- summary(a)[[1]]
cat(sprintf("aov.f_stat=%.10f\n", asum$"F value"[1]))
cat(sprintf("aov.p_value=%.10g\n", asum$"Pr(>F)"[1]))

# ── 8. K-means on iris (4 features, k=3, set.seed for determinism) ────
set.seed(42)
km <- kmeans(iris[, 1:4], centers = 3, nstart = 1)
sizes <- sort(km$size)
cat(sprintf("kmeans.tot_withinss=%.10f\n", km$tot.withinss))
cat(sprintf("kmeans.size_smallest=%d\n", sizes[1]))
cat(sprintf("kmeans.size_largest=%d\n",  sizes[3]))

# ── 9. Correlation: cor(iris$Sepal.Length, iris$Petal.Length) ─────────
cat(sprintf("cor.pearson=%.10f\n", cor(iris$Sepal.Length, iris$Petal.Length)))
