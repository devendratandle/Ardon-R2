# ── Train a 1-million-parameter language model in Ardon-R2 ───────────
# Pure R2: no Python, no external framework, no downloaded model.

cat("Ardon-R2 — training a 1M-parameter model\n\n")

# 1. DEFINE.  Vocabulary is the 256 byte values, so any text works.
m <- llm.new(dim = 128, layers = 5, heads = 4, kv.heads = 2,
             ffn = 384, ctx = 64, lr = 0.003, seed = 42)

info <- llm.info(m)
cat("parameters :", info$params, "\n")
cat("shape      : dim", info$dim, "| layers", info$layers,
    "| heads", info$heads, "/", info$kv.heads, "kv\n")
cat("context    :", info$ctx, "tokens\n\n")

# 2. TRAIN on a corpus.
corpus <- "ardon r2 is a statistical runtime written in pure rust. "
t0 <- Sys.time()
loss <- llm.train(m, corpus, steps = 120, seq = 16, batch = 6, report = 30)
cat("\nfinal loss :", loss, "\n")
cat("train time :", as.numeric(Sys.time()) - as.numeric(t0), "seconds\n\n")

# 3. GENERATE from the trained model (greedy: temperature = 0).
cat("prompt      : 'ardon r2 is'\n")
cat("continuation:", llm.generate(m, "ardon r2 is", 28), "\n\n")

# 4. SAVE for deployment.
path <- llm.save(m, "mymodel")
cat("saved to   :", path, "\n")

llm.free(m)
cat("done.\n")
