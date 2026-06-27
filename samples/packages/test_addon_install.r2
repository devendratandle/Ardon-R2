# End-to-end smoke test for the R2 addon-package system.
# Run with: cargo run -p r2-cli -- samples/test_addon_install.r2

# 1. Install the example package from a local directory.
install.from.dir("samples/example-r2-package")

# 2. List what's installed.
installed.packages()

# 3. Load it.
library("mymath")

# 4. Call the exported functions.
print(add_one(41))
print(double_it(21))
print(greet("R2"))

# 5. Clean up.
uninstall("mymath")
