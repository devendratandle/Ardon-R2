# `[` keeps names and takes them; lists subset as lists; lists and vectors carry
# classes. Whole output, so the printing of names, <NA> and attr(,"class") counts.
x <- c(a = 1, b = 2, c = 3)
x["b"]
x[c("c", "a")]
x["z"]
x[c("a", "z")]
x[c("a", "a")]
x[2]
x[2:3]
x[c(TRUE, FALSE, TRUE)]
x[-1]
x[c(-1, -3)]
x[c(1, NA)]
x[5]

(1:3)["a"]
i <- c(a = 1L, b = 2L)
i["b"]
i[2]
s <- c(a = "p", b = "q")
s["b"]
s[2]
s[c(FALSE, TRUE)]
g <- c(a = TRUE, b = FALSE)
g["a"]
g[-1]

# assigning by name
x["b"] <- 20
x
x["d"] <- 4
x

# lists
l <- list(a = 1, b = "s", c = 3)
l["b"]
l[2]
l[c(1, 3)]
l[c("c", "a")]
l[c(TRUE, FALSE, TRUE)]
l[-1]
l["z"]
length(l[1:2])
u <- list(1, "s")
u[2]
l[["b"]]
l$c

# the predicates
is.integer(1L)
is.integer(1)
is.integer(1:3)
is.integer("a")

# row and col
m <- matrix(1:6, 2)
row(m)
col(m)
m[row(m) > col(m)]

# classes
k <- list(a = 1, b = 2)
class(k) <- "foo"
class(k)
inherits(k, "foo")
inherits(k, "bar")
is.list(k)
names(k)
length(k)
k$a
k[["b"]]
k
unclass(k)
class(unclass(k))
class(k) <- NULL
class(k)
class(k) <- c("foo", "bar")
class(k)
inherits(k, "bar")
oldClass(k)
print(oldClass(list(1)))
k$c <- 3
class(k)
names(k)

w <- structure(list(a = 1), class = "foo")
class(w)
inherits(w, "foo")
w

v <- c(1, 2)
class(v) <- "myclass"
class(v)
inherits(v, "myclass")
is.numeric(v)
v
unclass(v)
class(1:2)
class(1)
class("a")
class(list())

# attributes on a list
a <- list(1, 2)
attr(a, "note") <- "hello"
attr(a, "note")
attributes(a)
a
y <- c(1, 2)
attr(y, "note") <- "hello"
attr(y, "note")
y

# classed results read as the lists they are
q <- qr(matrix(c(2, 1, 1, 3), 2))
is.list(q)
length(q)
names(q)

# what `[` keeps
v[1]
k1 <- structure(list(a = 1, b = 2), class = "foo")
class(k1[1])
d <- as.Date(c("2024-03-15", "2024-03-16"))
print(d[2])
x[0]
x[c(0, 1)]
(1:3)[0]
(1:3)[c(0, 2)]
c(1, 2, 3)[c(TRUE, NA, FALSE)]
c(1, 2, 3)[TRUE]
c(1, 2, 3, 4)[c(TRUE, FALSE)]
k1[["c"]] <- 3
k1["d"] <- 4
class(k1)
names(k1)
names(k1) <- c("p", "q", "r", "s")
class(k1)
k1["q"] <- NULL
class(k1)
length(k1)
class(v) <- "numeric"
v
class(v)
n <- 1:2
class(n) <- "character"
n

# out of range and zero subscripts, and a matrix read as a vector
c(1, 2)[5]
c(1, 2)[0]
c(1, 2)[c(0, 2)]
m[7]
length(m[0])
m[-1]
m[c(TRUE, FALSE)]
m[m > 2]

# printing lists
list()
list(a = list(b = 1, c = list(d = 2)), e = 3)
list(1, list(2, 3))
list(a = 1, 2)
list(a = NULL)

# `[[` gives the element without its name
x[["a"]]
x[[2]]
s[["b"]]
