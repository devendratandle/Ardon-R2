# Writes every object of R's `datasets` package into datasets.r2o, the
# file r2-base embeds. The values come straight from R, so nothing is
# transcribed by hand. Rerun after an R upgrade:
#
#   Rscript crates/r2-base/datasets/export_datasets.R
#
# Format (little-endian):
#   "R2O1", u32 count, then per dataset: str name, str title, u32 offset,
#   u32 length (offsets into the payload that follows the index).
#   str    = u32 byte length + UTF-8 bytes
#   object = u8 type (0 NULL, 1 logical, 2 integer, 3 double, 4 character,
#            5 list), u32 length, the values, u32 attribute count, then per
#            attribute: str name + object.
#   logical/integer: i32 each (NA = INT_MIN). double: f64 bits as R holds
#   them (R's NA is the NaN with low word 1954). character: i32 byte
#   length (-1 = NA) + bytes. list: the element objects.

out_path <- file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                      "datasets.r2o")

con <- rawConnection(raw(0), "wb")
u8  <- function(x) writeBin(as.integer(x), con, size = 1)
u32 <- function(x) writeBin(as.integer(x), con, size = 4, endian = "little")
str_ <- function(s) { b <- charToRaw(enc2utf8(s)); u32(length(b)); writeBin(b, con) }

obj <- function(x) {
  at <- attributes(x)
  # By storage type (typeof), not is.*(): is.integer() is FALSE for a factor.
  v <- unclass(x); attributes(v) <- NULL
  switch(typeof(x),
    "NULL"    = { u8(0); u32(0) },
    logical   = { u8(1); u32(length(v)); writeBin(as.integer(v), con, size = 4, endian = "little") },
    integer   = { u8(2); u32(length(v)); writeBin(v, con, size = 4, endian = "little") },
    double    = { u8(3); u32(length(v)); writeBin(v, con, size = 8, endian = "little") },
    character = { u8(4); u32(length(v)); for (s in v) if (is.na(s)) u32(-1L) else str_(s) },
    list      = { u8(5); u32(length(v)); for (e in v) obj(e) },
    # groupedData attributes hold formulas: kept as their deparsed text.
    language  = , symbol = { d <- paste(deparse(x), collapse = " "); u8(4); u32(1); str_(d); u32(0); return(invisible()) },
    environment = , closure = { u8(0); u32(0); u32(0); return(invisible()) },
    stop("unsupported type ", typeof(x)))
  u32(length(at))
  for (nm in names(at)) { str_(nm); obj(at[[nm]]) }
}

res <- data(package = "datasets")$results
items <- sub(" .*", "", res[, "Item"])           # "beaver1 (beavers)" -> "beaver1"
titles <- setNames(res[, "Title"], items)
names_ <- sort(ls("package:datasets"))

blobs <- lapply(names_, function(n) {
  con <<- rawConnection(raw(0), "wb")
  obj(get(n, envir = as.environment("package:datasets")))
  b <- rawConnectionValue(con); close(con); b
})

con <- rawConnection(raw(0), "wb")
writeBin(charToRaw("R2O1"), con)
u32(length(names_))
off <- 0L
for (i in seq_along(names_)) {
  str_(names_[i])
  str_(if (is.na(titles[names_[i]])) "" else titles[[names_[i]]])
  u32(off); u32(length(blobs[[i]]))
  off <- off + length(blobs[[i]])
}
for (b in blobs) writeBin(b, con)
bytes <- rawConnectionValue(con); close(con)
writeBin(bytes, out_path)
cat(length(names_), "datasets,", length(bytes), "bytes ->", out_path, "\n")
