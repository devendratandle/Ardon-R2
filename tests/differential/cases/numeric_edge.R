# Numeric semantics: NA/NaN/Inf, integer division, coercion, rounding.
cat("na.sum=", sum(c(1, NA, 3), na.rm = TRUE), "\n", sep = "")
cat("na.isna=", paste(as.integer(is.na(c(1, NA, NaN))), collapse = ","), "\n", sep = "")
cat("nan.isnan=", if (is.nan(0 / 0)) 1 else 0, "\n", sep = "")
cat("inf.pos=", if (is.infinite(1 / 0)) 1 else 0, "\n", sep = "")
cat("intdiv=", 17 %/% 5, "\n", sep = "")
cat("mod=", 17 %% 5, "\n", sep = "")
cat("negmod=", -7 %% 3, "\n", sep = "")   # R: 2 (sign of divisor)
cat("negintdiv=", -7 %/% 3, "\n", sep = "")   # R: -3 (floor)
cat("round.half=", round(2.5), "\n", sep = "")   # banker's: 2
cat("round.half3=", round(3.5), "\n", sep = "")  # banker's: 4
cat("round.dig=", round(3.14159, 2), "\n", sep = "")
cat("signif=", signif(123456, 3), "\n", sep = "")
cat("floor=", floor(-2.5), "\n", sep = "")
cat("ceiling=", ceiling(-2.5), "\n", sep = "")
cat("trunc=", trunc(-2.7), "\n", sep = "")
cat("asint=", as.integer(3.9), "\n", sep = "")
cat("asnum=", as.numeric("2.5e3"), "\n", sep = "")
cat("seq.by=", paste(seq(1, 2, by = 0.25), collapse = ","), "\n", sep = "")
cat("seq.len=", paste(seq_len(4), collapse = ","), "\n", sep = "")
cat("rep.times=", paste(rep(1:2, times = 3), collapse = ","), "\n", sep = "")
cat("rep.each=", paste(rep(1:2, each = 3), collapse = ","), "\n", sep = "")
cat("recycle.sum=", sum(c(1, 2, 3, 4) + c(10, 20)), "\n", sep = "")
cat("pow.chain=", 2 ^ 3 ^ 2, "\n", sep = "")   # right-assoc: 512
cat("unary.minus=", -2 ^ 2, "\n", sep = "")    # -(2^2) = -4
cat("log.base=", log(8, base = 2), "\n", sep = "")
cat("exp1=", exp(1), "\n", sep = "")
cat("sqrt2=", sqrt(2), "\n", sep = "")
