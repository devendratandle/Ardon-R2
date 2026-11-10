# S3 dispatch: UseMethod, NextMethod, default methods, methods for builtins
# and operators, and auto-print through print.<class>.
area <- function(s, ...) UseMethod("area")
area.circle <- function(s, ...) pi * s$r^2
area.square <- function(s, ...) s$side^2
area.default <- function(s, ...) stop("unknown shape")
c1 <- structure(list(r = 2), class = "circle")
s1 <- structure(list(side = 3), class = c("square", "shape"))
area(c1)
area(s1)
r <- tryCatch(area(1), error = function(e) conditionMessage(e))
r

# a generic with no default and no method
speak <- function(x) UseMethod("speak")
speak.dog <- function(x) "woof"
speak(structure(list(), class = "dog"))
msg <- tryCatch(speak(1), error = function(e) conditionMessage(e))
msg

# implicit classes
kind <- function(x) UseMethod("kind")
kind.numeric <- function(x) "a number"
kind.character <- function(x) "some text"
kind.function <- function(x) "a function"
kind.default <- function(x) "something else"
kind(1.5)
kind(2L)
kind("a")
kind(sum)
kind(TRUE)
kind(matrix(1:4, 2))

# NextMethod walks the class vector, then the default
describe <- function(x, ...) UseMethod("describe")
describe.shape <- function(x, ...) cat("a shape\n")
describe.square <- function(x, ...) { cat("a square, which is "); NextMethod() }
describe(s1)
greet <- function(x) UseMethod("greet")
greet.default <- function(x) cat("hello from default\n")
greet.child <- function(x) { cat("child, then: "); NextMethod() }
greet(structure(1, class = "child"))

# extra arguments reach the method
scale_by <- function(x, k, ...) UseMethod("scale_by")
scale_by.default <- function(x, k, ...) x * k
scale_by(3, k = 4)
scale_by(3, 5)

# methods for builtins
print.money <- function(x, ...) { cat("$", format(unclass(x), nsmall = 2), "\n"); invisible(x) }
m <- structure(12.5, class = "money")
m
print(m)
format.temp <- function(x, ...) paste0(unclass(x), " degrees")
format(structure(21, class = "temp"))
length.stack <- function(x) length(unclass(x)$items)
st <- structure(list(items = list(1, 2, 3)), class = "stack")
length(st)
summary.stack <- function(object, ...) cat("a stack of", length(object), "\n")
summary(st)
toString.money <- function(x, ...) paste("money:", unclass(x))
toString(m)

# a print method that falls back to the default
print.tagged <- function(x, ...) { cat("<tagged>\n"); print(unclass(x)) }
structure(1:3, class = "tagged")

# operators
"+.money" <- function(e1, e2) structure(unclass(e1) + unclass(e2), class = "money")
m + m
Ops.temp <- function(e1, e2) {
  v <- get(.Generic)(unclass(e1), unclass(e2))
  if (.Generic %in% c("+", "-", "*", "/")) structure(v, class = "temp") else v
}
t1 <- structure(20, class = "temp")
t2 <- structure(25, class = "temp")
unclass(t1 + t2)
t1 < t2
t1 == t2
class(t1 * 2)

# the object being dispatched need not be the first argument
pick <- function(a, b) UseMethod("pick", b)
pick.dog <- function(a, b) "dispatched on b"
pick(1, structure(list(), class = "dog"))

# inherits and class() as before
inherits(s1, "shape")
inherits(s1, c("circle", "shape"), which = FALSE)
class(unclass(s1))
