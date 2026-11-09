# try(), conditionMessage() and the condition a tryCatch handler receives.
kv <- function(k, v) cat(k, "=[", paste(v, collapse = "|"), "]\n", sep = "")

r <- try(stop("boom"), silent = TRUE)
kv("try_class", class(r))
kv("try_inherits", inherits(r, "try-error"))
kv("try_message", conditionMessage(attr(r, "condition")))
kv("try_ok", try(1 + 1, silent = TRUE))
e <- tryCatch(stop("bad input"), error = function(e) e)
kv("cond_message", conditionMessage(e))
kv("cond_dollar", e$message)
kv("cond_call_null", is.null(conditionCall(tryCatch(stop("x", call. = FALSE), error = function(e) e))))
kv("handler_paste", tryCatch(stop("late"), error = function(e) paste("caught:", conditionMessage(e))))
f <- function(x) { if (x < 0) stop("negative") ; sqrt(x) }
vals <- sapply(c(4, -1, 9), function(v) { r <- try(f(v), silent = TRUE); if (inherits(r, "try-error")) NA else r })
kv("try_in_sapply", vals)
kv("finally_runs", { out <- "no"; tryCatch(stop("z"), error = function(e) NULL, finally = out <- "yes"); out })
