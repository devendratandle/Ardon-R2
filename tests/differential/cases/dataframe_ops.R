# data.frame construction, filtering, aggregation, apply family.
df <- data.frame(
  g = c("a", "b", "a", "b", "a", "c"),
  x = c(1.5, 2.5, 3.5, 4.5, 5.5, 6.5),
  y = 1:6
)
cat("df.nrow=", nrow(df), "\n", sep = "")
cat("df.ncol=", ncol(df), "\n", sep = "")
cat("df.colnames=", paste(colnames(df), collapse = ","), "\n", sep = "")
sub <- df[df$x > 3, ]
cat("df.filter.nrow=", nrow(sub), "\n", sep = "")
cat("df.filter.sumy=", sum(sub$y), "\n", sep = "")
ag <- aggregate(x ~ g, data = df, FUN = mean)
cat("df.agg.nrow=", nrow(ag), "\n", sep = "")
cat("df.agg.a=", ag$x[ag$g == "a"], "\n", sep = "")
cat("df.tapply.b=", tapply(df$x, df$g, sum)[["b"]], "\n", sep = "")
cat("df.sapply.sum=", sum(sapply(df[, c("x", "y")], mean)), "\n", sep = "")
m <- matrix(as.numeric(1:6), 2, 3)
cat("apply.rows=", paste(apply(m, 1, sum), collapse = ","), "\n", sep = "")
cat("apply.cols=", paste(apply(m, 2, sum), collapse = ","), "\n", sep = "")
cat("lapply.len=", length(lapply(1:3, function(i) i * i)), "\n", sep = "")
cat("vapply.ok=", sum(vapply(1:4, function(i) i + 0.5, numeric(1))), "\n", sep = "")
cat("df.order.top=", df$y[order(df$x, decreasing = TRUE)][1], "\n", sep = "")
