# PLS-SEM (composite path modelling) in pure R2 — MATRIX-BASED.
# The inner/outer estimation is expressed as native matrix ops (scale,
# %*%, cor(matrix), t()), so each iteration is a handful of native calls
# instead of scalar loops. Reflective (Mode A), factorial inner scheme.
#
#   plssem(data, model, R = 200)

plssem <- function(data, model, R = 200) {
  vmean <- function(x) sum(x) / length(x)
  vsd   <- function(x) { m <- vmean(x); s <- sqrt(sum((x - m)^2) / length(x)); if (s < 1e-12) 1 else s }
  std   <- function(x) { m <- vmean(x); (x - m) / vsd(x) }
  strip <- function(s) gsub(" ", "", s)
  tokens <- function(s) {
    parts <- strsplit(s, "+")[[1]]
    out <- c()
    for (p in parts) { z <- strip(p); if (nchar(z) > 0) out <- c(out, z) }
    out
  }

  # ---- parse the model syntax ----
  lines <- strsplit(model, "\n")[[1]]
  cnames <- c(); inds <- list(); s_endo <- c(); s_pred <- list()
  for (ln in lines) {
    if (nchar(strip(ln)) == 0) next
    meas <- strsplit(ln, "=~")[[1]]
    if (length(meas) == 2) {
      cnames <- c(cnames, strip(meas[1]))
      inds[[length(inds) + 1]] <- tokens(meas[2])
    } else {
      st <- strsplit(ln, "~")[[1]]
      if (length(st) == 2) { s_endo <- c(s_endo, strip(st[1])); s_pred[[length(s_pred) + 1]] <- tokens(st[2]) }
    }
  }
  C <- length(cnames)
  cindex <- function(nm) { for (j in 1:C) if (cnames[j] == nm) return(j); 0 }

  preds <- list()
  for (j in 1:C) preds[[j]] <- c()
  if (length(s_endo) > 0) for (e in 1:length(s_endo)) {
    j <- cindex(s_endo[e]); pv <- c()
    for (p in s_pred[[e]]) pv <- c(pv, cindex(p))
    preds[[j]] <- pv
  }
  adjacent <- function(a, b) { if (b %in% preds[[a]]) return(TRUE); if (a %in% preds[[b]]) return(TRUE); FALSE }

  # ---- assemble one raw indicator matrix + complete cases ----
  allnames <- c(); blocks <- list()
  for (j in 1:C) {
    idx <- c()
    for (nm in inds[[j]]) { allnames <- c(allnames, nm); idx <- c(idx, length(allnames)) }
    blocks[[j]] <- idx
  }
  rawcols <- list()
  for (t in 1:length(allnames)) { nm <- allnames[t]; rawcols[[t]] <- data[[nm]] }
  nrow0 <- length(rawcols[[1]])
  ok <- rep(TRUE, nrow0)
  for (t in 1:length(rawcols)) ok <- ok & !is.na(rawcols[[t]])
  keep <- which(ok)
  rawmat <- rawcols[[1]][keep]
  if (length(rawcols) > 1) for (t in 2:length(rawcols)) rawmat <- cbind(rawmat, rawcols[[t]][keep])
  n <- length(keep)

  # ---- one PLS fit on a set of row indices → path coefficients ----
  fit_once <- function(rows) {
    Z <- scale(rawmat[rows, ])
    nn <- length(rows)
    Slist <- list()
    for (j in 1:C) { blk <- Z[, blocks[[j]]]; Slist[[j]] <- std(blk %*% rep(1, length(blocks[[j]]))) }
    for (it in 1:100) {
      S <- Slist[[1]]
      if (C > 1) for (j in 2:C) S <- cbind(S, Slist[[j]])
      Rm <- cor(S)                                  # C×C correlations — one native call
      inner <- list()
      for (j in 1:C) {
        z <- rep(0, nn)
        for (k in 1:C) if (adjacent(j, k)) z <- z + Rm[j, k] * Slist[[k]]
        inner[[j]] <- std(z)
      }
      newS <- list(); d <- 0
      for (j in 1:C) {
        blk <- Z[, blocks[[j]]]
        w <- as.numeric(t(blk) %*% inner[[j]])       # outer weights = t(block) %*% inner
        sc <- std(blk %*% w)
        d <- max(d, 1 - abs(cor(sc, Slist[[j]])))
        newS[[j]] <- sc
      }
      Slist <- newS
      if (d < 1e-7) break
    }
    out <- c()
    for (j in 1:C) {
      pv <- preds[[j]]
      if (length(pv) == 1) {
        p1 <- pv[1]; x <- Slist[[p1]]; y <- Slist[[j]]
        out <- c(out, sum(x * y) / sum(x * x))
      } else if (length(pv) == 2) {
        p1 <- pv[1]; p2 <- pv[2]
        x1 <- Slist[[p1]]; x2 <- Slist[[p2]]; y <- Slist[[j]]
        s11 <- sum(x1 * x1); s22 <- sum(x2 * x2); s12 <- sum(x1 * x2)
        b1 <- sum(x1 * y); b2 <- sum(x2 * y)
        det <- s11 * s22 - s12 * s12
        out <- c(out, (s22 * b1 - s12 * b2) / det, (s11 * b2 - s12 * b1) / det)
      }
    }
    out
  }

  est <- fit_once(1:n)
  labels <- c()
  for (j in 1:C) { pv <- preds[[j]]; if (length(pv) >= 1) for (p in pv) labels <- c(labels, paste0(cnames[p], " -> ", cnames[j])) }

  se <- rep(NA, length(est)); tval <- rep(NA, length(est)); lo <- rep(NA, length(est)); hi <- rep(NA, length(est))
  if (R > 0) {
    np <- length(est)
    # Parallel bootstrap (Phase P): each rep refits on a resample across cores.
    # Per-worker RNG makes sample() independent + reproducible. par.sapply
    # returns an np x R matrix — one column per resample.
    boot <- par.sapply(1:R, function(r) fit_once(sample(1:n, n, replace = TRUE)))
    for (m in 1:np) {
      col <- boot[m, ]
      se[m] <- vsd(col) * sqrt(n / (n - 1)); tval[m] <- est[m] / se[m]
      sc <- sort(col)
      ilo <- round(0.025 * R, 0); if (ilo < 1) ilo <- 1
      ihi <- round(0.975 * R, 0); if (ihi > R) ihi <- R
      lo[m] <- sc[ilo]; hi[m] <- sc[ihi]
    }
  }

  cat("\nPLS-SEM (r2sem library, matrix-based) — Mode A, factorial scheme\n")
  cat("  n =", n, "complete cases | bootstrap reps =", R, "\n\n")
  cat("Structural paths:\n")
  for (m in 1:length(est))
    cat("  ", labels[m], ":  est =", round(est[m], 3), " se =", round(se[m], 3),
        " t =", round(tval[m], 2), " CI = [", round(lo[m], 3), ",", round(hi[m], 3), "]\n")
  invisible(list(labels = labels, est = est, se = se, t = tval, ci_lower = lo, ci_upper = hi, n = n))
}
