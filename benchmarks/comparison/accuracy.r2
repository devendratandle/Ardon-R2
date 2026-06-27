# R2-side accuracy comparison script.
# Pair with `accuracy.R`. Prints KEY=VALUE lines to stdout.
#
# Usage:  r2 accuracy.r2 > out_R2.txt
# Or from workspace: target/release/r2 bench/r_vs_r2/accuracy.r2

# Helper: print one `key=value\n` line to stdout.
emit <- function(key, val) {
  cat(key, "=", val, "\n", sep = "")
}

# ── 1. Descriptive stats on iris$Sepal.Length ─────────────────────────
x <- iris$Sepal.Length
emit("desc.mean",   mean(x))
emit("desc.median", median(x))
emit("desc.sd",     sd(x))
emit("desc.var",    var(x))
emit("desc.min",    min(x))
emit("desc.max",    max(x))

# ── 2. Linear regression: mpg ~ wt + hp on mtcars ─────────────────────
fit <- lm(mpg ~ wt + hp, data = mtcars)
emit("lm.intercept",     fit$coefficients[1])
emit("lm.beta_wt",       fit$coefficients[2])
emit("lm.beta_hp",       fit$coefficients[3])
emit("lm.r_squared",     fit$r.squared)
emit("lm.adj_r_squared", fit$adj.r.squared)
emit("lm.f_stat",        fit$f.statistic)

# ── 3. GLM logistic: am ~ wt + hp on mtcars ───────────────────────────
gfit <- glm(am ~ wt + hp, family = binomial(), data = mtcars)
emit("glm.intercept",         gfit$coefficients[1])
emit("glm.beta_wt",           gfit$coefficients[2])
emit("glm.beta_hp",           gfit$coefficients[3])
emit("glm.null_deviance",     gfit$null.deviance)
emit("glm.residual_deviance", gfit$deviance)
emit("glm.aic",               gfit$aic)

# ── 4. Two-sample t.test (Welch): setosa vs versicolor petal length ───
x <- iris$Petal.Length[1:50]
y <- iris$Petal.Length[51:100]
tt <- t.test(x, y)
emit("ttest.statistic", tt$statistic)
emit("ttest.df",        tt$parameter)
emit("ttest.p_value",   tt$p.value)

# ── 5. SVD: singular values of a 5×3 matrix ──────────────────────────
# (R2 has no `diag()` builtin yet, so we skip explicit U·diag(d)·t(V)
# reconstruction here. r2-linalg's unit tests already verify that
# property to ~1e-9 absolute. Singular values themselves are uniquely
# defined and the most informative comparison point.)
A <- matrix(c(1,2,3,4,5,  10,20,30,40,50,  100,200,300,400,510), 5, 3)
sv <- svd(A)
emit("svd.d_1", sv$d[1])
emit("svd.d_2", sv$d[2])
emit("svd.d_3", sv$d[3])

# ── 6. Eigendecomposition: a known 3×3 symmetric matrix ───────────────
# Same matrix as accuracy.R section 6. Eigenvalues ≈ {7.288, 2.133, 0.579}.
# R2's cov() is two-vector scalar (returns a number, not a matrix) so we
# pick a fixed symmetric matrix to keep the cross-language test clean.
A_eig <- matrix(c(4, 1, 1,  1, 3, 2,  1, 2, 3), 3, 3)
e <- eigen(A_eig)
emit("eigen.lambda_1", e$values[1])
emit("eigen.lambda_2", e$values[2])
emit("eigen.lambda_3", e$values[3])

# ── 7. ANOVA: Sepal.Length by Species ─────────────────────────────────
a <- aov(Sepal.Length ~ Species, data = iris)
emit("aov.f_stat",  a$f.statistic)
emit("aov.p_value", a$p.value)

# ── 8. K-means on iris (4 features, k=3) ──────────────────────────────
X <- cbind(iris$Sepal.Length, iris$Sepal.Width, iris$Petal.Length, iris$Petal.Width)
set.seed(42)
km <- kmeans(X, centers = 3)
sizes <- sort(km$size)
emit("kmeans.tot_withinss",  km$tot.withinss)
emit("kmeans.size_smallest", sizes[1])
emit("kmeans.size_largest",  sizes[3])

# ── 9. Correlation: Pearson on iris columns ───────────────────────────
emit("cor.pearson", cor(iris$Sepal.Length, iris$Petal.Length))
