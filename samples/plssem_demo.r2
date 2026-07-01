# R2sem demo — PLS-SEM on synthetic Anxiety/Depression/Stress data.
# Structure: Stress -> Anxiety (0.50), Stress -> Depression (0.30),
#            Anxiety -> Depression (0.40). 7 reflective indicators/construct.
set.seed(42)
n <- 1000
stress <- rnorm(n)
anx    <- 0.50 * stress + sqrt(0.75) * rnorm(n)
dep    <- 0.30 * stress + 0.40 * anx + 0.70 * rnorm(n)

load  <- 0.75
noise <- sqrt(1 - load^2)
mk <- function(lat) load * lat + noise * rnorm(n)

df <- data.frame(
  Q1 = mk(stress), Q6 = mk(stress), Q8 = mk(stress), Q11 = mk(stress),
  Q12 = mk(stress), Q14 = mk(stress), Q18 = mk(stress),
  Q2 = mk(anx), Q4 = mk(anx), Q7 = mk(anx), Q9 = mk(anx),
  Q15 = mk(anx), Q19 = mk(anx), Q20 = mk(anx),
  Q3 = mk(dep), Q5 = mk(dep), Q10 = mk(dep), Q13 = mk(dep),
  Q16 = mk(dep), Q17 = mk(dep), Q21 = mk(dep)
)

model <- paste(c(
  "Anxiety    ~ Stress",
  "Depression ~ Stress + Anxiety",
  "Anxiety    =~ Q2 + Q4 + Q7 + Q9 + Q15 + Q19 + Q20",
  "Depression =~ Q3 + Q5 + Q10 + Q13 + Q16 + Q17 + Q21",
  "Stress     =~ Q1 + Q6 + Q8 + Q11 + Q12 + Q14 + Q18"
), collapse = "\n")

# Export the exact data so cSEM can run on identical values (fair compare).
write.csv(df, "samples/plssem_data.csv")

cat("Running plssem (200 bootstrap resamples) ...\n")
# plssem() prints its own timing + structural paths with bootstrap SE/t/CI.
fit <- plssem(df, model, .R = 200)
cat("Path estimates vector:", round(fit$path_coef, 3), "\n")
