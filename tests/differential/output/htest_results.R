# A hypothesis test RETURNS its result and prints nothing while computing;
# the report belongs to printing the result. Storing a test and taking one
# field out of it shows only that field — exactly as in R. (The report's
# own layout is not compared here: R2's differs from R's print.htest.)
m <- matrix(c(12, 5, 7, 9), 2)
res <- chisq.test(m)
res$p.value
chisq.test(m)$p.value
tt <- t.test(c(1, 2, 3, 4, 5), mu = 2)
tt$p.value
ct <- cor.test(c(1, 2, 3, 4, 5, 6), c(2, 1, 4, 3, 6, 5))
ct$p.value
f <- function() chisq.test(m)
x <- f()
cat(x$parameter, "\n")
