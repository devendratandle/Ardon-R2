# Dates, date-times and time differences through arithmetic, seq and diff;
# names and dim carried by operators; logical matrices. Whole output.
d <- as.Date("2024-03-15")
d + 1
class(d + 1)
1 + d
d - 1
d2 <- as.Date("2024-03-20")
d2 - d
class(d2 - d)
as.numeric(d2 - d)
d + 0:2
d2 > d
diff(c(d, d2))
seq(d, by = "day", length.out = 3)
t1 <- as.POSIXct("2024-03-15 10:00:00", tz = "UTC")
t1 + 3600
t1 + 90 - t1
t1 + 7200 - t1
x <- c(a = 1, b = 2)
x * 2
2 + x
m <- matrix(1:6, 2)
g <- m > 2
g
dim(g)
is.matrix(g)
!g
which(g)
which(g, arr.ind = TRUE)
apply(g, 2, sum)
sum(g)
m == 3
s <- structure(1:6, dim = c(2, 3))
s
dim(s)
ifelse(m > 2, 1, 0)
m * 2
m + m
as.POSIXct("2024-03-15 10:00:00", tz = "UTC")
attributes(as.POSIXct("2024-03-15 10:00:00", tz = "UTC"))
as.POSIXct("2024-03-15", tz = "UTC")
difftime(as.Date("2024-03-20"), as.Date("2024-03-15"))
seq(d, by = "day", length.out = 3)
seq(d, as.Date("2024-04-15"), by = "week")
seq(as.Date("2024-01-31"), by = "month", length.out = 4)
seq(d, by = "2 weeks", length.out = 3)
d + as.difftime(2, units = "days")
c(a = d, b = d2)
format(d + 1)
weekdays(d + 1)
x[c(TRUE,FALSE)] * 3
-m
(-x)
y <- c(1,2); class(y) <- "foo"; class(y * 2)
m2 <- m; m2[m2 > 3] <- 0; m2
weekdays(c(d, d2), abbreviate = TRUE)
months(d)
months(t1)

# a logical matrix by row and column
g[1, ]
g[, 2]
g[2, 3]
g[1:2, 2:3]
m[g]
