# a factor column in a printed data frame shows its levels (an NA as <NA>)
d <- data.frame(score = c(81.5, 77, 92.25, NA, 68),
                grp = factor(c("a", "bb", "a", NA, "ccc")))
print(d)
d$grp
