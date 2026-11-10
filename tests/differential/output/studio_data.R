# What Ardon Studio's Data tab writes as R2 code: every form of `[<-` on a data
# frame, adding rows to one with factor columns, cut(), save()/load() of
# factors, and read.csv's na.strings and dec. Whole output.
d <- data.frame(x = c(1, 2, 3), g = factor(c("a", "b", "a")), s = c("p", "q", "r"))
d[1, 1] <- 10
d[1, 2] <- "b"
d[1, 3] <- "w"
d[2, "s"] <- "v"
d
d[1, ] <- list(5, "b", "z")
d
d[, 1] <- 0
d[2:3, "x"] <- c(7, 8)
d[d$x > 7, "x"] <- 99
d[["s"]][1] <- "k"
d$g[2] <- "a"
d
e <- d
e[4, ] <- list(4, "b", "t")
e
levels(e$g)

# a row with a new level, and two frames with different levels
r <- rbind(d, data.frame(x = 4, g = "c", s = "t"))
r
levels(r$g)
r2 <- rbind(data.frame(g = factor(c("a", "b"))), data.frame(g = factor(c("c", "a"))))
levels(r2$g)
as.integer(r2$g)
rbind(data.frame(b = 1, a = 2), data.frame(a = 3, b = 4))

# cut() gives a factor
cut(c(1, 5, 9), breaks = c(0, 3, 6, 10))
class(cut(c(1, 5, 9), 3))
cut(c(1, 5, 9), 3)
cut(c(1, 5, 9), breaks = c(0, 3, 6, 10), labels = c("lo", "mid", "hi"))
cut(c(0.5, 2.25, 10), breaks = c(0, 1, 5, 10), right = FALSE)
cut(c(0, 5, 10), breaks = c(0, 5, 10), include.lowest = TRUE)
cut(1:10, 4, labels = FALSE)
table(cut(mtcars$mpg, 3))

# save() and load() keep factors and names
f <- tempfile(fileext = ".RData")
dd <- d
save(dd, file = f)
rm(dd)
loaded <- load(f)
loaded
str(dd)
levels(dd$g)
g1 <- function() { load(f); exists("dd") }
g1()

# read.csv: na.strings, dec, sep
csv <- tempfile(fileext = ".csv")
writeLines(c("a;b", "1,5;x", "-;y", "3,25;z"), csv)
read.csv(csv, sep = ";", dec = ",", na.strings = "-")
out <- tempfile(fileext = ".csv")
write.csv(data.frame(a = 1:2), out, row.names = FALSE)
file.exists(out)
