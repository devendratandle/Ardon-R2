# cSEM comparison — run in R on the SAME data R2 exported.
#   1) run  r2 samples/plssem_demo.r2   (writes samples/plssem_data.csv)
#   2) run  Rscript samples/plssem_compare.R
# Compares cSEM's PLS-SEM estimates + wall-clock against R2's plssem().
library(cSEM)

# R2's write.csv omits R's leading row-names column, so read plain.
d <- read.csv("samples/plssem_data.csv")

model <- "
  Anxiety    ~ Stress
  Depression ~ Stress + Anxiety
  Anxiety    =~ Q2 + Q4 + Q7 + Q9 + Q15 + Q19 + Q20
  Depression =~ Q3 + Q5 + Q10 + Q13 + Q16 + Q17 + Q21
  Stress     =~ Q1 + Q6 + Q8 + Q11 + Q12 + Q14 + Q18
"

cat("Running cSEM (200 bootstrap resamples) ...\n")
tt <- system.time(
  fit <- csem(.data = d, .model = model,
              .resample_method = "bootstrap", .R = 200,
              .handle_inadmissibles = "replace")
)
cat("cSEM elapsed (wall):", round(tt[3], 3), "s\n\n")
print(summarize(fit)$Estimates$Path_estimates)
