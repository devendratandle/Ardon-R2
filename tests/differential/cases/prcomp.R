# prcomp against R: components from the SVD of the centred data. Signs of a
# component are arbitrary, so rotations and scores compare by absolute value.
kv <- function(k, v) cat(k, "=", paste(sprintf("%.10g", as.numeric(v)), collapse = ","), "\n", sep = "")

x <- matrix(c(2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1,
              2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9,
              1.0, 3.0, 2.0, 0.5, 4.0, 1.5, 2.5, 3.5, 0.2, 1.8), 10, 3)
p <- prcomp(x)
kv("sdev", p$sdev)
kv("rot", abs(p$rotation))
kv("x", abs(p$x))
kv("center", p$center)
s <- prcomp(x, scale. = TRUE)
kv("sdev_scaled", s$sdev)
kv("rot_scaled", abs(s$rotation))
kv("scale", s$scale)
# nearly collinear columns: the covariance route lost the small component
z <- cbind(1:20, (1:20) * 2 + 1e-6 * sin(1:20), cos(1:20))
kv("sdev_collinear", prcomp(z)$sdev)
# wider than tall
w <- matrix(sin(1:24), 4, 6)
kv("sdev_wide", prcomp(w)$sdev[1:2])
kv("dim_wide", dim(prcomp(w)$x))
kv("scale_fn", scale(x)[1:5])
kv("scale_nocenter", scale(x, center = FALSE)[1:5])
# kappa(exact = TRUE) is sigma_max / sigma_min, from the singular values
h <- outer(1:5, 1:5, function(i, j) 1 / (i + j - 1))
kv("kappa_hilbert5", kappa(h, exact = TRUE))
kv("kappa_x", kappa(x, exact = TRUE))
