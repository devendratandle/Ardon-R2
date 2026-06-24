# ============================================================
#  "Random" realistic base-R program — varied, not just asserts.
#  Mixes simulation, data frames, strings, recursion, matrices.
# ============================================================
chk <- function(label, got, want) {
    pass <- isTRUE(all.equal(got, want))
    cat(sprintf("  [%s] %-34s %s\n", if (pass) "PASS" else "FAIL", label,
                paste(got, collapse = " ")))
}

cat("===== 1. simulation: dice rolls (RNG, structural checks) =====\n")
set.seed(42)
rolls <- sample(1:6, 1000, replace = TRUE)
cat(sprintf("  rolled %d dice; mean=%.3f (expect ~3.5); range=[%d,%d]\n",
            length(rolls), mean(rolls), min(rolls), max(rolls)))
counts <- table(rolls)
cat("  face counts:", as.numeric(counts), "\n")
chk("all faces present", length(counts), 6)
chk("counts sum to n",   sum(as.numeric(counts)), 1000)

cat("\n===== 2. data frame: student scores =====\n")
students <- data.frame(
    name  = c("Ada","Bo","Cy","Di","Eve","Fei"),
    group = c("A","B","A","B","A","B"),
    score = c(88, 72, 95, 61, 79, 84)
)
students$grade <- ifelse(students$score >= 80, "pass", "fail")
cat("  grades:", students$grade, "\n")
chk("n passing", sum(students$grade == "pass"), 3)
gm <- tapply(students$score, students$group, mean)
cat("  group means:", round(as.numeric(gm), 2), "\n")
chk("group A mean", round(as.numeric(gm)[1], 2), round((88+95+79)/3, 2))
chk("top scorer", students$name[which.max(students$score)], "Cy")

cat("\n===== 3. strings =====\n")
words <- c("apple", "Banana", "cherry")
caps  <- toupper(substr(words, 1, 1))
rest  <- substr(words, 2, 100)
titled <- paste0(caps, rest)
cat("  title-cased:", titled, "\n")
chk("title-case",  titled, c("Apple","Banana","Cherry"))
chk("gsub vowels", gsub("[aeiou]", "*", "education"), "*d*c*t**n")
chk("nchar",       nchar(words), c(5,6,6))

cat("\n===== 4. recursion + functional =====\n")
fact <- function(n) if (n <= 1) 1 else n * fact(n - 1)
chk("factorial(6)", fact(6), 720)
chk("Reduce prod", Reduce(`*`, 1:6), 720)
fib <- function(n) { a<-0; b<-1; for (i in seq_len(n)) { t<-a+b; a<-b; b<-t }; a }
chk("fib 1..8", sapply(1:8, fib), c(1,1,2,3,5,8,13,21))
chk("Filter evens", Filter(function(x) x %% 2 == 0, 1:10), c(2,4,6,8,10))

cat("\n===== 5. matrices / linear algebra =====\n")
M <- matrix(c(2,1,1,3), nrow = 2)
chk("matmul diag", diag(M %*% diag(c(1,1))), c(2,3))
b <- c(5, 10)
x <- solve(M, b)
cat("  solve Mx=b ->", round(x, 4), "\n")
chk("solve check", round(as.numeric(M %*% x), 4), b)
chk("t() transpose", t(matrix(1:6, nrow=2))[1,], c(1,2))

cat("\n===== 6. numeric routine: estimate pi by series =====\n")
# Leibniz: pi/4 = 1 - 1/3 + 1/5 - 1/7 + ...
terms <- seq(1, 20000, by = 2)
signs <- rep(c(1, -1), length.out = length(terms))
pi_est <- 4 * sum(signs / terms)
cat(sprintf("  pi ~ %.5f (true 3.14159)\n", pi_est))
chk("pi within 0.001", abs(pi_est - pi) < 0.001, TRUE)

cat("\n===== 7. quantiles / summary =====\n")
x <- c(4, 8, 15, 16, 23, 42)
cat("  quantiles:", round(as.numeric(quantile(x)), 2), "\n")
chk("median", median(x), 15.5)
chk("IQR-ish range", max(x) - min(x), 38)
chk("order desc", x[order(x, decreasing = TRUE)][1], 42)

cat("\nRANDOM-BASE PROGRAM DONE.\n")
