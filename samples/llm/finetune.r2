# ── Fine-tune a model in Ardon-R2 ────────────────────────────────────
# Pure R2. The base model is one R2 trained; adapters teach it new answers
# without changing a single base weight.

work <- "finetune_demo"

# 1. A BASE MODEL, trained and saved in the usual way.
corpus <- paste(rep("the sun rises in the east and sets in the west. rivers run down to the sea. ", 200),
                collapse = "")
base <- llm.new(dim = 96, layers = 3, heads = 4, kv.heads = 2, ffn = 288,
                ctx = 64, vocab = 400, lr = 0.003, seed = 42)
invisible(llm.train(base, corpus, steps = 120, seq = 32, batch = 8, shuffle = TRUE))
invisible(llm.save(base, paste0(work, "/base")))
cat("base model :", llm.info(base)$params, "parameters\n")
invisible(llm.free(base))

# 2. THE EXAMPLES. Only the completions count in the loss; the prompts are
#    context, and batches are padded to their longest example.
prompts <- c("capital of india:", "capital of france:", "capital of japan:",
             "capital of kenya:", "capital of peru:", "capital of norway:")
answers <- c(" new delhi.", " paris.", " tokyo.", " nairobi.", " lima.", " oslo.")

# 3. FINE-TUNE THROUGH ADAPTERS. llm.load(train = TRUE) starts a new run from
#    the saved weights; llm.lora freezes them all and adds low-rank adapters.
m <- llm.load(paste0(work, "/base"), train = TRUE, lr = 0.003)
training <- llm.lora(m, rank = 8, alpha = 16)
cat("training   :", training, "adapter values\n\n")

loss <- llm.finetune(m, prompts, answers,
                     steps = 300, batch = 3, seq = 32,
                     wd = 0.01, clip = 1,             # AdamW and gradient clipping
                     schedule = "cosine", warmup = 20, # warmup, then cosine decay
                     accum = 1,                        # > 1 trades time for memory
                     report = 100,
                     checkpoint = paste0(work, "/run"), every = 100, keep = 2)
cat("\nfinal loss :", loss, "\n\n")

# 4. USE IT.
for (p in prompts[1:3]) cat(p, llm.generate(m, p, n = 6), "\n")

# 5. KEEP IT, two ways: a small adapter that applies to this base, or a whole
#    model with the adapters merged in, which runs like any other.
invisible(llm.save.adapter(m, paste0(work, "/adapter")))
invisible(llm.save(m, paste0(work, "/merged")))

s <- llm.load(paste0(work, "/merged"))
a <- llm.load.adapter(paste0(work, "/base"), paste0(work, "/adapter"))
cat("\nmerged     :", llm.generate(s, prompts[2], n = 6), "\n")
cat("adapter    :", llm.generate(a, prompts[2], n = 6), "\n")

# A run that stopped carries on from its checkpoint, exactly:
#   m <- llm.resume(paste0(work, "/run"), base = paste0(work, "/base"))
cat("done.\n")
