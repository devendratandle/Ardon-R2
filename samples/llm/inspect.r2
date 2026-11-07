# ── See inside a training run in Ardon-R2 ────────────────────────────
# Shape presets, a callback on the training loop, and per-block summaries.
# Pure R2. A run that uses none of these costs exactly what it did before.

# 1. A SHAPE BY NAME. Presets set the shape only, every other argument still
#    works, and a named argument overrides the preset.
#    tiny 1.3M | small 7.2M | medium 40M | large 125M parameters.
m <- llm.new(preset = "tiny", vocab = 400, seed = 42)
info <- llm.info(m)
cat("parameters :", info$params, "| dim", info$dim, "| layers", info$layers, "\n\n")

corpus <- paste(rep("the sun rises in the east and sets in the west. rivers run down to the sea. ", 200),
                collapse = "")

# 2. A CALLBACK, called on the steps report = would print: the first step and
#    every report-th. It takes the place of the printed line, so what is shown
#    and what is kept is up to the script. Its value is ignored; stop() in it
#    ends the run with the weights it has reached.
steps <- c()
losses <- c()
watch <- function(step, loss) {
    steps <<- c(steps, step)
    losses <<- c(losses, loss)
    cat("step", step, " loss", round(loss, 3), "\n")
}
final <- llm.train(m, corpus, steps = 60, seq = 32, batch = 4, shuffle = TRUE,
                   report = 20, callback = watch)
cat("final loss :", final, "\n\n")
print(data.frame(step = steps, loss = losses))

# 3. A LOOK INSIDE, on one batch. One extra forward and backward pass and its
#    memory, per call. It changes no weight, so it is safe inside a callback
#    too. grad.ratio is grad.norm / weight.norm: a block far below the others
#    is hardly learning; nonfinite above 0 is the first sign of a blow-up.
look <- llm.inspect(m, corpus, seq = 32, batch = 4)
cat("\nloss on this batch:", look$loss, "\n")
print(head(look$blocks, 8))
print(look$activations)

# 4. INSIDE A RUN. Watch one block's gradient while it trains.
norms <- c()
invisible(llm.train(m, corpus, steps = 40, seq = 32, batch = 4, shuffle = TRUE, report = 10,
                    callback = function(step, loss) {
                        look <- llm.inspect(m, corpus, seq = 32, batch = 4)
                        norms <<- c(norms, look$blocks$grad.norm[look$blocks$block == "layer.0.wq"])
                    }))
cat("\nlayer.0.wq gradient norm:", round(norms, 4), "\n")

invisible(llm.free(m))
