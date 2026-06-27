# Example R2 addon package.
# This is the manifest — every R2 package must have one at its root.

package_name        <- "mymath"
package_version     <- "0.1.0"
package_description <- "Tiny helpers for demonstrating the R2 addon system."
package_author      <- "R2 community"
package_license     <- "AGPL-3.0"
package_exports     <- c("add_one", "double_it", "greet")
