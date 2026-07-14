# Mutable environments: stateful closures, <<-, lexical capture.
make_counter <- function() {
  n <- 0
  function() {
    n <<- n + 1
    n
  }
}
c1 <- make_counter()
c2 <- make_counter()
c1(); c1()
cat("counter.third=", c1(), "\n", sep = "")
cat("counter.independent=", c2(), "\n", sep = "")

acc <- function() {
  total <- 0
  function(x) {
    total <<- total + x
    total
  }
}
add <- acc()
add(5); add(10)
cat("accumulator=", add(2.5), "\n", sep = "")

# <<- from a nested call updates the ENCLOSING function's local.
outer_fn <- function() {
  v <- 100
  bump <- function() v <<- v + 1
  bump(); bump()
  v
}
cat("nested.superassign=", outer_fn(), "\n", sep = "")

# <<- with no enclosing binding lands in the global env.
gcount <- 0
inc_global <- function() gcount <<- gcount + 10
inc_global(); inc_global()
cat("global.superassign=", gcount, "\n", sep = "")

# A factory's environments are independent per call.
make_bank <- function(balance) {
  list(
    deposit = function(x) { balance <<- balance + x; balance },
    balance = function() balance
  )
}
b1 <- make_bank(100)
b2 <- make_bank(1000)
b1$deposit(50)
cat("bank.b1=", b1$balance(), "\n", sep = "")
cat("bank.b2=", b2$balance(), "\n", sep = "")

# Loop-captured state via local frames.
adders <- list()
for (k in 1:3) {
  adders[[k]] <- local({ kk <- k; function(x) x + kk })
}
cat("loop.capture=", adders[[2]](10), "\n", sep = "")
