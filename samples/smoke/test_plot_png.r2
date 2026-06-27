# Test PNG output + clear save-path messages.

x <- c(1, 2, 3, 4, 5)
y <- c(2, 4, 6, 8, 10)

# 1. Auto-saved SVG (this happens automatically; path is now absolute).
plot(x, y, main = "Test scatter")

# 2. Explicit SVG save to a chosen path.
save.plot("my-chart.svg")

# 3. Explicit PNG save — the new feature.
save.plot("my-chart.png", width = 1024, height = 768)

# 4. PNG at a different aspect ratio.
save.plot("wide.png", width = 1600, height = 400)
