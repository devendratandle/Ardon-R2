# Source file for the 'mymath' example addon.
# Every function defined here that appears in package_exports
# becomes available after library("mymath").

add_one <- function(x) {
  x + 1
}

double_it <- function(x) {
  x * 2
}

greet <- function(name) {
  paste0("Hello from mymath, ", name, "!")
}
