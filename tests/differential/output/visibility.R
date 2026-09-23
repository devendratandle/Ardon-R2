# What auto-prints at the top level — R's visibility rules. Run with both
# engines and compare the whole output: every line below either prints
# (the comment says what) or prints nothing.
f <- function(v) print(v)
f(1)                                   # once, not twice
g <- function() invisible(7)
g()                                    # nothing
h <- function() { x <- 3 }
h()                                    # nothing: ends in an assignment
(x <- 5)                               # 5: parentheses make it visible
x <- 6                                 # nothing
k <- function() 42
k()                                    # 42
if (FALSE) 1                           # nothing
if (TRUE) 2                            # 2
for (i in 1:2) i                       # nothing
y <- c(1, 2); y                        # 1 2
invisible(3)                           # nothing
print(4)                               # 4, once
m <- function() { print("a"); 10 }
m()                                    # "a" then 10
n <- function() { 10; invisible(11) }
n()                                    # nothing
tryCatch(invisible(1), error = function(e) 0)   # nothing
tryCatch(8, error = function(e) 0)              # 8
switch("a", a = invisible(12), b = 13) # nothing
switch("b", a = 12, b = 13)            # 13
local({ z <- 1 })                      # nothing
local({ 14 })                          # 14
p <- function(v) return(invisible(v))
p(15)                                  # nothing
p2 <- function(v) return(v)
p2(16)                                 # 16
q2 <- function(v) cat("hi\n")
q2(1)                                  # hi
1 + invisible(2)                       # 3: arithmetic is visible
w <- 0; while (w < 2) w <- w + 1       # nothing
sapply(c(1, 2, 3), function(i) invisible(i))  # 1 2 3: sapply's value is visible
