# sort / order / unique / rank, string collation, the NA literal, c(),
# element assignment, matrix fill, cat(sep), pmax and sapply simplification.
j <- function(x) paste(x, collapse = ",")

# sort keeps the type, drops NA/NaN, honours decreasing
kv <- function(k, v) cat(k, "=", v, "\n", sep = "")
kv("sort.nan", j(sort(c(3, NaN, 1, NA))))
kv("sort.dec", j(sort(c(3, 1, 2), decreasing = TRUE)))
kv("sort.int.class", class(sort(c(3L, 1L))))
kv("sort.na.last", j(sort(c(3, NA, 1), na.last = TRUE)))
kv("sort.lgl", j(sort(c(TRUE, FALSE, TRUE))))
# collation: case-insensitive first, lowercase first, punctuation < digits < letters
kv("sort.case", j(sort(c("B", "a", "A", "b", "ab", "Aa"))))
kv("sort.punct", j(sort(c("item-2", "item 1", "item_3", "Item4"))))
kv("sort.mixed", j(sort(c("banana", "Apple", "apple", "Banana", "x1", "x10", "x2"))))
kv("order.chr", j(order(c("b", "a", "B"))))
kv("order.keys", j(order(c(2, 1, 2), c(3, 9, 1))))
kv("order.dec.na", j(order(c(2, NA, 1), decreasing = TRUE)))
kv("levels.case", j(levels(factor(c("b", "B", "a", "A")))))
# unique / duplicated: exact, NA kept once, NA and NaN distinct
kv("unique.chr", j(unique(c("b", "a", "b", NA, NA))))
kv("unique.num", j(unique(c(1, NA, NaN, 1, NA))))
kv("dup", j(duplicated(c(NA, 1, NA, 1))))
# rank: ties averaged, NA last
kv("rank.ties", j(rank(c(10, 20, 10, 5))))
kv("rank.na", j(rank(c(10, NA, 5))))
kv("rank.min", j(rank(c(2, 2, 1), ties.method = "min")))

# NA is logical; c() keeps the narrowest type and the names
kv("na.class", class(NA))
kv("c.lgl", paste(class(c(TRUE, FALSE, NA)), j(c(TRUE, FALSE, NA))))
kv("c.int", class(c(1L, NA)))
kv("c.chr", j(c("a", TRUE, 1L, 2.5, 0.1 + 0.2)))
kv("c.chr.na", sum(is.na(c("a", NA))))
kv("c.names", j(names(c(a = 1, b = NA, 3))))
kv("na.typed", paste(class(NA_real_), class(NA_integer_), class(NA_character_)))
kv("rep.na", class(rep(NA, 3)))

# element assignment
x <- c(1, 5, 3); x[x > 2] <- 0; kv("asg.mask", j(x))
x <- c(1, 5, 3); x[-1] <- 9; kv("asg.neg", j(x))
x <- c(a = 1, b = 2); x["c"] <- 7; kv("asg.name", paste(j(x), j(names(x))))
x <- 1:3; x[2] <- 2.5; kv("asg.widen", paste(class(x), j(x)))
x <- c(1, 2); x[2] <- "z"; kv("asg.chr", paste(class(x), j(x)))
s <- c("a", "b"); s[1] <- NA; kv("asg.na.chr", is.na(s[1]))
w <- rep(NA, 3); w[1] <- 2.5; kv("asg.from.na", paste(class(w), j(w)))
x <- c(1, NA, 3); x[is.na(x)] <- 0; kv("asg.isna", j(x))
x <- c(1, 2, 3); x[5] <- 1; kv("asg.grow", j(x))
f <- factor(c("a", "b")); f[2] <- "a"; kv("asg.factor", j(as.character(f)))
df <- data.frame(a = c(1, 2), b = c(3, 4)); df$a[2] <- 10; kv("asg.df.nested", j(df$a))
df[2, "b"] <- 7; kv("asg.df.cell", j(df$b))
df$c <- 0; kv("asg.df.recycle", j(df$c))
l <- list(v = c(1, 2)); l$v[2] <- 9; kv("asg.list.nested", j(l$v))
l[["w"]] <- 5; kv("asg.list.name", j(names(l)))
g <- function() { x[2] <<- 99 }; x <- c(1, 2); g(); kv("asg.super", j(x))
h <- function() { y <<- 5 }; y <- 1; h(); kv("super.plain", y)

# matrix() recycles and keeps NA; is.na keeps cells
kv("matrix.recycle", j(matrix(1, 2, 2)))
kv("matrix.na", sum(is.na(matrix(c(1, NA, 3, 4), 2))))
kv("matrix.na.fill", sum(is.na(matrix(NA, 2, 2))))
m <- matrix(1:4, 2); m[1, 2] <- 0; kv("matrix.asg", j(m))

cat("cat.sep=", c(1, 2, 3), sep = ","); cat("\n")
kv("pmax.na", j(pmax(c(1, NA), 0)))
kv("pmax.narm", j(pmax(c(1, NA), 0, na.rm = TRUE)))
kv("sapply.na", j(sapply(1:3, function(i) if (i == 2) NA else i)))
kv("sapply.int.class", class(sapply(1:3, function(i) i)))
kv("sapply.names", j(names(sapply(c("a", "bb"), nchar))))

# factor labels= / ordered= and stringsAsFactors
f <- factor(c("lo", "hi", "lo"), levels = c("lo", "hi"), labels = c("Low", "High"))
kv("labels", paste(j(levels(f)), j(as.integer(f))))
kv("labels.prefix", j(levels(factor(c(1, 2, 3), labels = "L"))))
h2 <- factor(c("a", "b", "c"), labels = c("x", "x", "y"))
kv("labels.merge", paste(j(levels(h2)), j(as.integer(h2))))
kv("ordered", is.ordered(factor(c("s", "m"), levels = c("s", "m"), ordered = TRUE)))
d <- data.frame(g = c("b", "a"), y = 1:2, stringsAsFactors = TRUE)
kv("saf", paste(class(d$g), j(levels(d$g))))
kv("ttest.names", j(names(t.test(c(1, 2, 3, 4), c(2, 4, 6, 9))$estimate)))
