# ── Train a 1-million-parameter language model in Ardon-R2 ───────────
# Pure R2: no Python, no external framework, no downloaded model.

cat("Ardon-R2 — training a 1M-parameter model\n\n")

# 1. DEFINE.  The vocabulary is a BPE learned from the training text on
#    the first llm.train() call, so a token is about four bytes rather
#    than one. The 256 byte values stay INSIDE it as the fallback, so any
#    text still works — an emoji no merge covers decomposes into its UTF-8
#    bytes rather than failing.
#
#    ctx = 64 is therefore ~64 word-pieces, not 64 characters. With the old
#    byte-level default it was 64 BYTES — about eleven words, too little
#    context for the model to learn much from.
#
#    vocab = 256 is still available and gives pure byte-level tokenization.
m <- llm.new(dim = 128, layers = 5, heads = 4, kv.heads = 2,
             ffn = 384, ctx = 64, vocab = 512, lr = 0.003, seed = 42)

info <- llm.info(m)
cat("parameters :", info$params, "\n")
cat("shape      : dim", info$dim, "| layers", info$layers,
    "| heads", info$heads, "/", info$kv.heads, "kv\n")
cat("context    :", info$ctx, "tokens\n\n")

# 2. TRAIN on a corpus.
# Repeated so the BPE trainer has enough text to find merges in.
corpus <- paste(rep("ardon r2 is a statistical runtime written in pure rust. ", 400),
                collapse = "")
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
