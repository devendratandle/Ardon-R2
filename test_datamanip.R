# ============================================================
#  Data-manipulation smoke test — data.frames, factors, joins
# ============================================================
ok <- function(label, got, want) {
    pass <- isTRUE(all.equal(got, want))
    cat(sprintf("  [%s] %-30s got=%s\n", if (pass) "PASS" else "FAIL", label,
                paste(got, collapse=",")))
}

df <- data.frame(
    id    = 1:6,
    grp   = c("a","b","a","b","a","b"),
    score = c(10, 20, 30, 40, 50, 60)
)

cat("=== data.frame basics ===\n")
ok("nrow",        nrow(df),                  6)
ok("ncol",        ncol(df),                  3)
ok("colnames",    colnames(df),              c("id","grp","score"))
ok("$ access",    df$score,                  c(10,20,30,40,50,60))
ok("[[ access",   df[["id"]],                c(1,2,3,4,5,6))

cat("=== row / column subsetting ===\n")
ok("df[rows,]",   nrow(df[df$score > 25, ]), 4)
ok("df[,col]",    df[, "id"],                c(1,2,3,4,5,6))
ok("logical sub", df$score[df$grp == "a"],   c(10,30,50))
ok("col select",  ncol(df[c("id","score")]), 2)

cat("=== factors ===\n")
f <- factor(c("lo","hi","lo","hi","mid"))
ok("levels",      levels(f),                 c("hi","lo","mid"))
ok("nlevels",     nlevels(f),                3)
ok("factor[i]",   as.character(f[2:3]),      c("hi","lo"))
ok("factor==",    sum(f == "lo"),            2)
ok("as.numeric",  as.numeric(factor(c("b","a","b"))), c(2,1,2))

cat("=== group aggregation ===\n")
ok("tapply",      as.numeric(tapply(df$score, df$grp, sum)), c(90,120))
ok("table",       as.numeric(table(df$grp)), c(3,3))

cat("=== bind / merge ===\n")
ok("rbind nrow",  nrow(rbind(df, df)),       12)
ok("cbind ncol",  ncol(cbind(df, extra = 6:1)), 4)
d2 <- data.frame(id = c(1,2,3), label = c("x","y","z"))
m  <- merge(df, d2)
ok("merge nrow",  nrow(m),                   3)

cat("=== NA handling ===\n")
v <- c(1, NA, 3, NA, 5)
ok("is.na",       sum(is.na(v)),             2)
ok("na.omit",     as.numeric(na.omit(v)),    c(1,3,5))
ok("mean na.rm",  mean(v, na.rm = TRUE),     3)
ok("ifelse",      ifelse(is.na(v), 0, v),    c(1,0,3,0,5))

cat("=== transform / order ===\n")
df2 <- df[order(df$score, decreasing = TRUE), ]
ok("order rows",  df2$score[1],              60)
ok("head",        head(df$id, 3),            c(1,2,3))

cat("\nDATA-MANIPULATION TEST DONE.\n")
