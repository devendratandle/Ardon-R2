# ── Deploy a trained Ardon-R2 model ──────────────────────────────────
# Loads a saved model and serves it. No training state, no framework,
# no network — a model directory is model.json + model.safetensors.

cat("Ardon-R2 — model deployment\n\n")

# 1. LOAD.  Serving only: no optimizer state, so ~1/3 the memory.
s <- llm.load("mymodel")

info <- llm.info(s)
cat("loaded     :", info$params, "parameters,", info$mode, "\n")
cat("shape      : dim", info$dim, "| layers", info$layers, "| ctx", info$ctx, "\n\n")

# 2. SERVE.  Greedy decoding is deterministic and reproducible.
prompts <- c("ardon r2 is", "a statistical", "pure rust")
for (p in prompts) {
  cat("prompt:", p, "\n")
  cat("  ->  ", llm.generate(s, p, 24), "\n")
}

# 3. SAMPLING.  temperature/top.k trade determinism for variety;
#    the same seed always replays the same output.
cat("\nsampled (temperature 0.8, top.k 20, seed 7):\n")
cat("  ->  ", llm.generate(s, "ardon r2 is", 24,
                           temperature = 0.8, top.k = 20, seed = 7), "\n")

llm.free(s)
cat("\nserved and released.\n")
