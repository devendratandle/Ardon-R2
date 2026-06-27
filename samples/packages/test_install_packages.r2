# Unified install.packages() smoke test.

cat("\n--- A: install from local directory ---\n")
install.packages("mymath", path = "samples/example-r2-package")
library("mymath")
print(add_one(41))
print(double_it(21))
uninstall("mymath")
cat("Cleaned up.\n")

cat("\n--- Path classification (no-op runs to show dispatch) ---\n")
# Just demonstrate dispatch decisions by printing.

cat("\n--- C: github would resolve like this (not actually run; would need network):\n")
cat("    install.packages('r2.survival', path = 'devendratandle/Ardon-R2-libraries',\n")
cat("                     subdir = 'r2pkg-survival')\n")
