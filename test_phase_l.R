# ============================================================
#  Phase L — first-class language objects — self-test
#  Run:  r2 test_phase_l.R     (CLI)
#  or in R2Gui:  source("E:/R2_Rust _opus4.6/r2/r2/test_phase_l.R")
# ============================================================

cat("===== L.1: quote / eval / parse / deparse =====\n")
e <- quote(x + y * 2)            # capture code WITHOUT running it
cat("quote prints as source : "); print(e)
cat("typeof(quote(...))      :", typeof(e), "\n")
cat("deparse(quote(x+1))     :", deparse(quote(x + 1)), "\n")
cat("eval(parse(text=1+1))   :", eval(parse(text = "1 + 1")), "\n")
cat("eval(quote(1+2*3))      :", eval(quote(1 + 2 * 3)), "\n")

x <- 100; y <- 5
cat("eval over real vars     :", eval(e), "  (x=100,y=5 -> 110)\n")
cat("multi-statement parse   :", eval(parse(text = "a <- 4; a * a")), "\n")

cat("\n===== L.1: call / as.call (build calls programmatically) =====\n")
cc <- call("sum", 1, 2, 3)
cat("call('sum',1,2,3)       : "); print(cc)
cat("eval(that)              :", eval(cc), "\n")
cat("as.call(list(...))      :", eval(as.call(list("max", 7, 2, 9))), "\n")

cat("\n===== L.2: body / formals / args (introspection) =====\n")
f <- function(a, b = 10) a + b
cat("body(f)                 :", deparse(body(f)), "\n")
cat("formals(f):\n"); print(formals(f))
cat("args(f):\n"); print(args(f))

cat("\n===== L.3: substitute (capture the caller's expression) =====\n")
temperature <- c(20, 25, 30)     # must exist: R2 is eager (see caveat below)
speed <- 1:10
label <- function(v) deparse(substitute(v))
cat("label(temperature*1.8)  :", label(temperature * 1.8 + 32), "\n")
cat("label(speed)            :", label(speed), "\n")

cat("\n===== L.3: match.call / sys.call =====\n")
g <- function(x, y, z) match.call()
cat("match.call g(1,2,3)     :", deparse(g(1, 2, 3)), "\n")
cat("match.call reordered    :", deparse(g(z = 9, x = 1, y = 5)), "\n")
h <- function(p, q) sys.call()
cat("sys.call h(10,20)       :", deparse(h(10, 20)), "\n")

cat("\n===== L.3: bquote (quote, but splice .(value)) =====\n")
co <- 42
cat("bquote(x + .(co))       :", deparse(bquote(x + .(co))), "\n")
n <- 3
cat("eval(bquote(2*.(n)))    :", eval(bquote(2 * .(n))), "\n")

cat("\n===== regression: ordinary code still works =====\n")
sq <- function(n) n * n
cat("normal closure sq(9)    :", sq(9), "\n")
cat("vector mean             :", mean(c(2, 4, 6, 8)), "\n")
v <- 1:5
cat("sapply                  :", sapply(v, function(z) z * z), "\n")

cat("\n===== the documented eager-eval caveat =====\n")
cat("# substitute() captures the EXPRESSION, but R2 still evaluates the\n")
cat("# argument eagerly, so it must be a valid expression. Uncomment the\n")
cat("# next line to see it error (q and w are undefined):\n")
cat("#   label(q + w)   ->  Error: object 'q' not found\n")

cat("\nAll Phase L checks ran.\n")
