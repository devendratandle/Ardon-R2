# ============================================================
#  Base R smoke test — vectors, math, strings, apply, sets
# ============================================================
ok <- function(label, got, want) {
    pass <- isTRUE(all.equal(got, want))
    cat(sprintf("  [%s] %-28s got=%s\n", if (pass) "PASS" else "FAIL", label,
                paste(got, collapse=",")))
}

cat("=== vectors & sequences ===\n")
ok("seq",        seq(2, 10, by = 2),      c(2,4,6,8,10))
ok("seq_len",    seq_len(4),              c(1,2,3,4))
ok("rep",        rep(c(1,2), times = 3),  c(1,2,1,2,1,2))
ok("rev",        rev(1:5),                c(5,4,3,2,1))
ok("cumsum",     cumsum(1:5),             c(1,3,6,10,15))
ok("diff",       diff(c(1,4,9,16)),       c(3,5,7))

cat("=== math / reductions ===\n")
ok("sum",        sum(1:100),              5050)
ok("prod",       prod(1:5),               120)
ok("mean",       mean(c(2,4,6,8)),        5)
ok("var",        var(c(2,4,6,8)),         20/3)
ok("round",      round(3.14159, 2),       3.14)
ok("which.max",  which.max(c(3,9,2,9)),   2)
ok("cumprod",    cumprod(1:4),            c(1,2,6,24))

cat("=== sorting / ranking / search ===\n")
ok("sort",       sort(c(3,1,2)),          c(1,2,3))
ok("order",      order(c(30,10,20)),      c(2,3,1))
ok("which",      which(c(F,T,F,T)),       c(2,4))
ok("rank",       rank(c(30,10,20)),       c(3,1,2))
ok("unique",     unique(c(1,1,2,3,3)),    c(1,2,3))

cat("=== sets ===\n")
ok("%in%",       c(1,5) %in% c(1,2,3),    c(TRUE, FALSE))
ok("union",      union(c(1,2), c(2,3)),   c(1,2,3))
ok("intersect",  intersect(c(1,2,3), c(2,3,4)), c(2,3))
ok("setdiff",    setdiff(c(1,2,3), c(2)), c(1,3))

cat("=== strings ===\n")
ok("paste0",     paste0("a", 1:3),        c("a1","a2","a3"))
ok("toupper",    toupper("hi"),           "HI")
ok("nchar",      nchar("hello"),          5)
ok("substr",     substr("abcdef", 2, 4),  "bcd")
ok("gsub",       gsub("a", "X", "banana"),"bXnXnX")
ok("sprintf",    sprintf("%05.2f", 3.1),  "03.10")

cat("=== apply / functional ===\n")
ok("sapply",     sapply(1:4, function(x) x^2), c(1,4,9,16))
ok("Reduce",     Reduce(`+`, 1:5),        15)
ok("Filter",     Filter(function(x) x %% 2 == 0, 1:6), c(2,4,6))
ok("Map",        unlist(Map(`+`, 1:3, 4:6)), c(5,7,9))
ok("do.call",    do.call(sum, list(1,2,3)), 6)

cat("\nBASE TEST DONE.\n")
