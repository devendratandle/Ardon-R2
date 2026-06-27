# Minimal repro for the loop subscript-assignment bug.
x <- c(0, 0, 0, 0, 0)
for (i in 2:5) {
  x[i] <- x[i-1] + 1
}
print(x)   # Expected: 0 1 2 3 4
