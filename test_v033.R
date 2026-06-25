# ============================================================
#  Ardon-R2 v0.3.3 — self-test of this release's fixes
#  Run:  r2 test_v033.R      (or  source(...) in R2Gui)
#  Prints PASS/FAIL per check, then a total. All should PASS.
# ============================================================
results <- logical(0)
chk <- function(label, got, want) {
  good <- isTRUE(all.equal(got, want))
  cat(sprintf("  [%s] %-34s %s\n", if (good) "PASS" else "FAIL", label,
              paste(got, collapse = " ")))
  good
}

cat("===== function/variable namespace (name a var 'c'/'t') =====\n")
c <- c(1, 2, 3)                          # shadow builtin c with a vector
results <- c(results, chk("c() still callable", c(c, 4), c(1, 2, 3, 4)))
tt <- t(matrix(1:4, 2, 2))   # transpose; 't' could be shadowed too
results <- c(results, chk("t() transpose", tt[1, ], c(1, 2)))

cat("\n===== repeat / loops =====\n")
i <- 0; repeat { i <- i + 1; if (i == 5) break }
results <- c(results, chk("repeat + break", i, 5))
s <- 0; for (k in c(10, 20, 30)) { v <- k * 2; s <- s + v }
results <- c(results, chk("loop assigned-and-read", s, 120))
j <- 1; tot <- 0; while (j <= 5) { tot <- tot + j; j <- j + 1 }
results <- c(results, chk("while accumulate", tot, 15))

cat("\n===== negative (exclusion) indexing =====\n")
results <- c(results, chk("vector x[-2]", c(10, 20, 30, 40)[-2], c(10, 30, 40)))
results <- c(results, chk("vector x[-c(1,3)]", c(10, 20, 30, 40)[-c(1, 3)], c(20, 40)))
A <- matrix(1:9, 3, 3, byrow = TRUE)
results <- c(results, chk("matrix A[,-2]", as.numeric(A[, -2]), c(1, 4, 7, 3, 6, 9)))
df0 <- data.frame(a = 1:3, b = 4:6, cc = 7:9)
results <- c(results, chk("data.frame df[,-2]", colnames(df0[, -2]), c("a", "cc")))

cat("\n===== strings =====\n")
results <- c(results, chk("strsplit empty sep", strsplit("153", "")[[1]], c("1", "5", "3")))
results <- c(results, chk("paste0 recycle", paste0("x", 1:3), c("x1", "x2", "x3")))
results <- c(results, chk("substr vectorized", substr(c("apple", "berry"), 1, 3), c("app", "ber")))
results <- c(results, chk("toString", toString(c(1, 2, 3)), "1, 2, 3"))
results <- c(results, chk("as.numeric('42')", as.numeric("42") + 1, 43))
nums <- as.numeric(unlist(regmatches("a12b34", gregexpr("[0-9]+", "a12b34"))))
results <- c(results, chk("regmatches/gregexpr", nums, c(12, 34)))

cat("\n===== factors =====\n")
fac <- factor(c("b", "a", "b", "cx"))
results <- c(results, chk("levels sorted", levels(fac), c("a", "b", "cx")))
results <- c(results, chk("factor == 'b'", sum(fac == "b"), 2))
results <- c(results, chk("as.numeric(factor)", as.numeric(factor(c("b", "a", "b"))), c(2, 1, 2)))
results <- c(results, chk("factor[i] subset", as.character(fac[2:3]), c("a", "b")))

cat("\n===== apply / data manipulation =====\n")
m <- matrix(1:6, 2, 3)
results <- c(results, chk("apply(matrix,1,sum)", apply(m, 1, sum), c(9, 12)))
results <- c(results, chk("apply(matrix,2,sum)", apply(m, 2, sum), c(3, 7, 11)))
results <- c(results, chk("tapply by factor",
  as.numeric(tapply(c(10, 20, 30, 40), c("a", "b", "a", "b"), sum)), c(40, 60)))
results <- c(results, chk("table(numeric)", as.numeric(table(c(1, 1, 2, 3, 3, 3))), c(2, 1, 3)))
results <- c(results, chk("ifelse string", ifelse(c(1, -1, 2) > 0, "pos", "neg"), c("pos", "neg", "pos")))

cat("\n===== replacement functions =====\n")
d2 <- data.frame(x = 1:2, y = 3:4); colnames(d2) <- c("p", "q")
results <- c(results, chk("colnames(df) <-", colnames(d2), c("p", "q")))
vv <- c(1, 2, 3); names(vv) <- c("a", "b", "cc")
results <- c(results, chk("names(v) <-", names(vv), c("a", "b", "cc")))

cat("\n===== sprintf format specs =====\n")
results <- c(results, chk("sprintf %05.2f", sprintf("%05.2f", 3.1), "03.10"))
results <- c(results, chk("sprintf flags/width", sprintf("[%-5s|%03d]", "hi", 7), "[hi   |007]"))

cat("\n===== metaprogramming (Phase L) =====\n")
results <- c(results, chk("quote/deparse", deparse(quote(x + y * 2)), "x + y * 2"))
results <- c(results, chk("eval(parse)", eval(parse(text = "sum(1:100)")), 5050))
lab <- function(z) deparse(substitute(z))
qq <- 1; ww <- 2
results <- c(results, chk("substitute label", lab(qq + ww), "qq + ww"))

cat("\n===== statistics sanity (matches R) =====\n")
results <- c(results, chk("cor", round(cor(mtcars$wt, mtcars$mpg), 4), -0.8677))
fit <- lm(mpg ~ wt, data = mtcars)
results <- c(results, chk("lm slope", round(as.numeric(coef(fit)[2]), 3), -5.344))
results <- c(results, chk("t-test", round(as.numeric(t.test(1:5, 6:10)$statistic), 2), -5))

cat(sprintf("\n==================  %d PASS  /  %d FAIL  ==================\n",
            sum(results), sum(!results)))
if (sum(!results) == 0) cat("ALL GREEN — v0.3.3 verified.\n")
