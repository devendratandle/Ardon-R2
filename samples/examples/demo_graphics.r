# =====================================================================
# Ardon-R2 — Interactive Graphics Device Demo
# =====================================================================
#
# Walks through every Phase R.G capability one plot at a time. After
# each plot is rendered:
#   1. The browser auto-refreshes and shows it.
#   2. You can save it under any filename you like.
#   3. Hit Enter to move on to the next plot.
#
# Run from the repo root:
#     ./target/release/r2 samples/demo_graphics.r

cat("\n========================================\n")
cat("Ardon-R2 — Interactive Graphics Demo\n")
cat("========================================\n\n")

# ---------------------------------------------------------------------
# Helper: show, optionally save, wait for Enter.
# ---------------------------------------------------------------------
# The default name argument is the filename suggested if the user just
# hits Enter at the save prompt; typing a different name saves under
# that; typing "skip" (or anything that looks like a skip word) skips
# saving entirely and proceeds to the next plot.
inspect <- function(default_name) {
  cat("\n   Plot is live at http://127.0.0.1:8765/\n")
  name <- readline(paste0("   Save as [default: ", default_name, ", or 'skip']: "))
  if (nchar(name) == 0) {
    save_plot(default_name)
    cat("   -> saved as ", default_name, "\n", sep = "")
  } else if (name == "skip" || name == "no" || name == "n") {
    cat("   -> skipped\n")
  } else {
    save_plot(name)
    cat("   -> saved as ", name, "\n", sep = "")
  }
  invisible(readline("   Press Enter for next plot... "))
}

# ---------------------------------------------------------------------
# 0. Launch the live viewer.
# ---------------------------------------------------------------------
dev.view()

cat("\nThis demo will walk through 6 plots. After each one:\n")
cat("  - the browser tab refreshes and shows it\n")
cat("  - you choose a filename (or hit Enter for the default)\n")
cat("  - press Enter again to move to the next plot\n\n")
invisible(readline("Press Enter to begin... "))


# ---------------------------------------------------------------------
# 1. Scatter with overlays
# ---------------------------------------------------------------------
cat("\n[1/6] Scatter plot with overlays (lines, abline, legend)...\n")
x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
y <- c(2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.8, 16.2, 17.9, 20.1)
plot(x, y, main = "Linear trend with overlay", xlab = "x", ylab = "y")
lines(x, y, col = "blue")
abline(a = 0, b = 2, col = "red", lty = 2)
legend("topleft", legend = c("data", "y=2x"), col = c("blue", "red"))
inspect("demo-overlays.svg")


# ---------------------------------------------------------------------
# 2. par() introspection (no plot — informational)
# ---------------------------------------------------------------------
cat("\n[2/6] par() introspection (no plot, just terminal output)...\n")
cat("   par('cex') = ", par("cex"), "\n", sep = "")
cat("   par('lwd') = ", par("lwd"), "\n", sep = "")
cat("   par('col') = ", par("col"), "\n", sep = "")

oldpar <- par(cex = 1.5, lwd = 2.5, col = "darkgreen")
cat("\n   After par(cex=1.5, lwd=2.5, col='darkgreen'):\n")
cat("   par('cex') = ", par("cex"), "\n", sep = "")
cat("   par('col') = ", par("col"), "\n", sep = "")

restored <- par(oldpar)
cat("\n   After par(oldpar):\n")
cat("   par('cex') = ", par("cex"), "  (restored)\n", sep = "")
invisible(readline("   Press Enter for next plot... "))


# ---------------------------------------------------------------------
# 3. 2x2 iris dashboard via par(mfrow=c(2,2))
# ---------------------------------------------------------------------
cat("\n[3/6] 2x2 iris dashboard with par(mfrow=c(2,2))...\n")
dev.off()
par(mfrow = c(2, 2))
plot(iris$Sepal.Length, iris$Sepal.Width,
     main = "Sepal L vs W", xlab = "length", ylab = "width")
hist(iris$Petal.Length, breaks = 12, main = "Petal length")
boxplot(setosa = iris$Sepal.Length[1:50],
        versicolor = iris$Sepal.Length[51:100],
        virginica = iris$Sepal.Length[101:150],
        main = "Sepal length by species")
barplot(c(50, 50, 50), main = "Species counts",
        names.arg = c("setosa", "versicolor", "virginica"))
inspect("iris-overview.svg")


# ---------------------------------------------------------------------
# 4. 3x1 tall stack via par(mfcol=c(3,1))
# ---------------------------------------------------------------------
cat("\n[4/6] 3x1 tall histogram stack with par(mfcol=c(3,1))...\n")
dev.off()
par(mfcol = c(3, 1))
hist(iris$Sepal.Length, main = "Sepal Length")
hist(iris$Petal.Length, main = "Petal Length")
hist(iris$Petal.Width,  main = "Petal Width")
inspect("iris-tall.svg")


# ---------------------------------------------------------------------
# 5. Cursor wrap (5 plots into 2x2 grid)
# ---------------------------------------------------------------------
cat("\n[5/6] Cursor wrap demo (5 plots into 2x2 grid)...\n")
dev.off()
par(mfrow = c(2, 2))
plot(1:10, (1:10) * 2,        main = "linear")
plot(1:10, (1:10)^2,          main = "quadratic")
plot(1:10, sqrt(1:10) * 5,    main = "sqrt")
plot(1:10, log(1:10) * 3,     main = "log")
# 5th plot wraps to panel (1,1).
plot(1:10, (1:10)^2, main = "x^2 (wrapped to panel 1)")
inspect("wrap-demo.svg")


# ---------------------------------------------------------------------
# 6. par(oldpar) round-trip
# ---------------------------------------------------------------------
cat("\n[6/6] par(oldpar) save/restore round-trip...\n")
dev.off()
oldpar <- par(mfrow = c(2, 2), cex = 1.4)
cat("   cex during demo = ", par("cex"), " (expected 1.4)\n", sep = "")
plot(1:5, 1:5, main = "before restore")
plot(1:5, (1:5)^2, main = "before restore")
restored <- par(oldpar)
cat("   cex after restore = ", par("cex"), " (back to 1.0)\n", sep = "")
dev.off()
plot(1:10, sqrt(1:10) * 5, main = "Single panel after restore")
inspect("plot-restored.svg")


# ---------------------------------------------------------------------
# Finale — leave the 2x2 iris dashboard on screen
# ---------------------------------------------------------------------
cat("\n[finale] Re-rendering the 2x2 iris dashboard for the long view...\n")
dev.off()
par(mfrow = c(2, 2))
plot(iris$Sepal.Length, iris$Sepal.Width, main = "Sepal L vs W", xlab = "length", ylab = "width")
hist(iris$Petal.Length, breaks = 12, main = "Petal length")
boxplot(setosa = iris$Sepal.Length[1:50],
        versicolor = iris$Sepal.Length[51:100],
        virginica = iris$Sepal.Length[101:150],
        main = "Sepal length by species")
barplot(c(50, 50, 50), main = "Species counts",
        names.arg = c("setosa", "versicolor", "virginica"))
save_plot("iris-overview-final.svg")

cat("\n========================================\n")
cat("Demo finished. The 2x2 iris dashboard is now in your browser.\n")
cat("Browser: http://127.0.0.1:8765/\n")
cat("\n")
cat("Press Enter to exit.\n")
cat("========================================\n")
invisible(readline(""))
