# Graph gallery — one colourful, labelled example of each Ardon-R2 plot
# type. Saves a PNG per plot into screenshots/ for the README showcase.
#   run:  r2 samples/graph_gallery.r2
set.seed(42)

# 1. Scatter + linear fit + axis labels (las=1 horizontal ticks)
x <- rnorm(60, 10, 3); y <- 2 * x + rnorm(60, 0, 4)
plot(x, y, main = "Scatter + linear fit", xlab = "predictor x",
     ylab = "response y", col = "steelblue", pch = 19, las = 1)
abline(lm(y ~ x), col = "firebrick", lwd = 2)
save.plot("screenshots/01-scatter.png", width = 1000, height = 700)

# 2. Multi-series line plot (matplot)
t <- seq(0, 2 * pi, length.out = 100)
M <- cbind(sin(t), cos(t), sin(2 * t))
matplot(t, M, type = "l", lwd = 2,
        col = c("tomato", "seagreen", "royalblue"),
        main = "Trig waves (matplot)", xlab = "t", ylab = "value", las = 1)
save.plot("screenshots/02-lines.png", width = 1000, height = 700)

# 3. Barplot
counts <- c(23, 41, 18, 35, 29)
barplot(counts, names.arg = c("A", "B", "C", "D", "E"),
        col = c("#E64A19", "#FBC02D", "#388E3C", "#1976D2", "#7B1FA2"),
        main = "Category counts (barplot)", xlab = "group",
        ylab = "count", las = 1)
save.plot("screenshots/03-barplot.png", width = 1000, height = 700)

# 4. Histogram
h <- rnorm(2000, 50, 12)
hist(h, breaks = 30, col = "mediumpurple", border = "white",
     main = "Distribution (hist)", xlab = "value", las = 1)
save.plot("screenshots/04-histogram.png", width = 1000, height = 700)

# 5. Grouped boxplot (las=2 rotates x labels vertical)
boxplot(setosa = iris$Sepal.Length[1:50],
        versicolor = iris$Sepal.Length[51:100],
        virginica = iris$Sepal.Length[101:150],
        col = c("#FF8A65", "#4DB6AC", "#9575CD"),
        main = "Sepal length by species (boxplot)", ylab = "cm", las = 2)
save.plot("screenshots/05-boxplot.png", width = 1000, height = 700)

# 6. Pie chart
pie(c(30, 25, 20, 15, 10),
    labels = c("Rust", "Stats", "ML", "Graphics", "JIT"),
    col = c("#EF5350", "#FFA726", "#66BB6A", "#42A5F5", "#AB47BC"),
    main = "Composition (pie)")
save.plot("screenshots/06-pie.png", width = 800, height = 800)

cat("Graph gallery written to screenshots/\n")
