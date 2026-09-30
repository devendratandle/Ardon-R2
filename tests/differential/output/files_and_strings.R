x <- "C:\\Users\\deven"
x
nchar(x)
s <- "say \"hi\""
nchar(s)
cat(s, "\n")
"a\tb\nc"
"\u00e9t\u00e9"
old <- setwd("cases")
basename(getwd())
setwd(old)
identical(getwd(), old)
y <- 1
ls()
file.path("C:/data", "DAS paper", "DASS.csv")
file.path("a", c("x.csv", "y.csv"))
basename(c("C:/Users/deven/DASS.csv", "c.txt", "d/"))
dirname(c("C:/Users/deven/DASS.csv", "c.txt"))
list.files("cases", pattern = "^s")
list.files("output", pattern = "PRINT", ignore.case = TRUE)
list.files(".", pattern = "summary", recursive = TRUE)
grepl("abc", c("xABCx", "abc", "no"), ignore.case = TRUE)
