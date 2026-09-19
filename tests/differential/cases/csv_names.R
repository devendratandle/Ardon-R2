# read.csv applies make.names to the header by default (check.names = TRUE):
# spaces and punctuation become ".", a leading digit gets "X", duplicates
# get ".1", ".2". check.names = FALSE keeps the raw header.
d <- read.csv("cases/data/spaces.csv")
cat("names=", paste(names(d), collapse="|"), "\n", sep = "")
cat("a.b.sum=", sum(d$a.b), "\n", sep = "")
cat("X1x.sum=", sum(d$X1x), "\n", sep = "")
r <- read.csv("cases/data/spaces.csv", check.names = FALSE)
cat("raw=", paste(names(r), collapse="|"), "\n", sep = "")
m <- make.names(c("a b", "1x", ".1", "if", "a b", "a b", "", "x_y", ".ok"), unique = TRUE)
m0 <- make.names(c("a b", "a b"))
cat("mk0=", paste(m0, collapse="|"), "
", sep = "")
cat("mk=", paste(m, collapse="|"), "\n", sep = "")
