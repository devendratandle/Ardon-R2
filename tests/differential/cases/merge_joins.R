# merge(): composite keys, outer joins, suffixes, ordering, NA fill.
#
# Every fact here is emitted as a key=value line so the harness can compare
# it numerically against GNU R. Outer joins are the point: an inner join is
# a *subset* of a left join, so a merge that silently ignores `all.x` still
# returns plausible-looking data and only the ROW COUNT gives it away.
# That is exactly how it went unnoticed — check counts, not just values.

a <- data.frame(k = c(3, 1, 2), v = c(30, 10, 20), s = c("c", "a", "b"))
b <- data.frame(k = c(2, 3, 4), w = c(200, 300, 400))

inner <- merge(a, b, by = "k")
cat("inner.nrow=", nrow(inner), "\n", sep = "")
cat("inner.cols=", paste(names(inner), collapse = ","), "\n", sep = "")
# Sorted by the key, so this is 2 then 3 — not the input order 3,1,2.
cat("inner.k=", paste(inner$k, collapse = ","), "\n", sep = "")
cat("inner.w=", paste(inner$w, collapse = ","), "\n", sep = "")

lx <- merge(a, b, by = "k", all.x = TRUE)
cat("allx.nrow=", nrow(lx), "\n", sep = "")
cat("allx.k=", paste(lx$k, collapse = ","), "\n", sep = "")
cat("allx.na=", sum(is.na(lx$w)), "\n", sep = "")

ly <- merge(a, b, by = "k", all.y = TRUE)
cat("ally.nrow=", nrow(ly), "\n", sep = "")
# The right-only row has no left key, so its key must come from the right.
cat("ally.lastk=", ly$k[nrow(ly)], "\n", sep = "")
cat("ally.na=", sum(is.na(ly$v)), "\n", sep = "")

full <- merge(a, b, by = "k", all = TRUE)
cat("full.nrow=", nrow(full), "\n", sep = "")
cat("full.k=", paste(full$k, collapse = ","), "\n", sep = "")
cat("full.sumw=", sum(full$w, na.rm = TRUE), "\n", sep = "")

# Composite key: joining on k1 alone would give a different row count and
# leave k2 duplicated with a suffix.
m1 <- data.frame(k1 = c(1, 1, 2), k2 = c("x", "y", "x"), v = c(10, 11, 20))
m2 <- data.frame(k1 = c(1, 2, 2), k2 = c("y", "x", "z"), w = c(1, 2, 3))
mk <- merge(m1, m2, by = c("k1", "k2"))
cat("multi.nrow=", nrow(mk), "\n", sep = "")
cat("multi.ncol=", ncol(mk), "\n", sep = "")
cat("multi.cols=", paste(names(mk), collapse = ","), "\n", sep = "")
cat("multi.sumv=", sum(mk$v), "\n", sep = "")

mka <- merge(m1, m2, by = c("k1", "k2"), all = TRUE)
cat("multiall.nrow=", nrow(mka), "\n", sep = "")
cat("multiall.k2=", paste(mka$k2, collapse = ","), "\n", sep = "")

# A non-key name in both frames gets suffixed on BOTH sides, not just one.
c1 <- data.frame(k = c(1, 2), v = c("A", "B"))
c2 <- data.frame(k = c(1, 2), v = c("X", "Y"))
cl <- merge(c1, c2, by = "k")
cat("clash.cols=", paste(names(cl), collapse = ","), "\n", sep = "")
cat("clash.vx=", paste(cl$v.x, collapse = ","), "\n", sep = "")
cat("clash.vy=", paste(cl$v.y, collapse = ","), "\n", sep = "")

# Duplicate keys produce the cartesian product within each key group.
d1 <- data.frame(k = c(1, 1), v = c(10, 11))
d2 <- data.frame(k = c(1, 1), w = c(20, 21))
mm <- merge(d1, d2, by = "k")
cat("m2m.nrow=", nrow(mm), "\n", sep = "")
cat("m2m.sum=", sum(mm$v) + sum(mm$w), "\n", sep = "")

# stringsAsFactors is a construction flag, never a column.
sf <- data.frame(k = c(1, 2), v = c("p", "q"), stringsAsFactors = FALSE)
cat("saf.ncol=", ncol(sf), "\n", sep = "")
cat("saf.cols=", paste(names(sf), collapse = ","), "\n", sep = "")

# A short column is recycled to the frame's row count. Without this the
# frame reports 3 rows while the column holds 1, and every row-wise read
# past the end quietly yields NA.
rc <- data.frame(k = c(1, 2, 3), g = "x")
cat("recycle.nrow=", nrow(rc), "\n", sep = "")
cat("recycle.g=", paste(rc$g, collapse = ","), "\n", sep = "")
cat("recycle.len=", length(rc$g), "\n", sep = "")
rc2 <- data.frame(k = c(1, 2, 3, 4), h = c(1, 2))
cat("recycle.h=", paste(rc2$h, collapse = ","), "\n", sep = "")

# And it must survive a join: the recycled column joins like any other.
rj <- merge(rc, data.frame(k = c(2, 3), w = c(20, 30)), by = "k")
cat("recycle.join.nrow=", nrow(rj), "\n", sep = "")
cat("recycle.join.g=", paste(rj$g, collapse = ","), "\n", sep = "")
