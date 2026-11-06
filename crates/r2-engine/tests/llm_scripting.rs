//! Seeing into training from R2 script: `callback =`, `llm.inspect`,
//! `llm.new(preset =)`, and that a run using none of them is unchanged.

use r2_engine::Engine;
use r2_parser::Parser;
use r2_tensor::model::Config;
use r2_tensor::tokenizer::Tokenizer;
use r2_train::data::{Batcher, Order};
use r2_train::llm::Trainer;
use r2_types::RVal;

fn run(script: &str) -> RVal {
    let mut e = Engine::new();
    let mut last = RVal::Null;
    for ex in Parser::parse(script).expect("parse ok") {
        last = e.eval(&ex).unwrap_or_else(|err| panic!("eval error: {}", err.msg));
    }
    last
}

fn run_err(script: &str) -> String {
    let mut e = Engine::new();
    for ex in Parser::parse(script).expect("parse ok") {
        if let Err(err) = e.eval(&ex) { return err.msg; }
    }
    panic!("expected an error from {script}");
}

fn nums(v: &RVal) -> Vec<f64> {
    match v {
        RVal::Numeric(d, _) => d.iter().map(|x| x.expect("no NA")).collect(),
        other => panic!("expected numbers, got {:?}", other),
    }
}

fn text(v: &RVal) -> Vec<String> {
    match v {
        RVal::Character(c, _) => c.iter().map(|x| x.as_ref().expect("no NA").to_string()).collect(),
        other => panic!("expected text, got {:?}", other),
    }
}

const SETUP: &str = "m <- llm.new(vocab = 256, dim = 32, layers = 2, heads = 2, ffn = 64, ctx = 32, lr = 0.01, seed = 42)\n\
                     txt <- paste(rep(\"the cat sat on the mat. \", 40), collapse = \"\")\n";
const TRAIN: &str = "llm.train(m, txt, steps = 12, seq = 16, batch = 4";

/// A directory for one test's saved model, with a path the script can quote.
fn out_dir(tag: &str) -> (std::path::PathBuf, String) {
    let dir = std::env::temp_dir().join(format!("r2_llm_scripting_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let quoted = dir.to_string_lossy().replace('\\', "/");
    (dir, quoted)
}

/// The bytes of the weights a script's run ends with.
fn weights_after(tag: &str, train: &str) -> Vec<u8> {
    let (dir, q) = out_dir(tag);
    run(&format!("{SETUP}{train}\ninvisible(llm.save(m, \"{q}\"))\n"));
    let bytes = std::fs::read(dir.join("model.safetensors")).expect("weights written");
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

// ── A run without the new features ───────────────────────────────────

/// The loop `llm.train` was before callbacks existed, written out by hand on
/// the trainer: same batches, same steps, nothing else.
#[test]
fn a_run_without_a_callback_trains_to_the_weights_of_the_plain_loop() {
    let cfg = Config { dim: 32, n_heads: 2, n_kv_heads: 2, n_layers: 2, vocab: 256, ffn_hidden: 64,
                       max_seq: 32, rope_base: 10000.0, eps: 1e-5 };
    let mut t = Trainer::new(cfg, 0.01, 42).unwrap();
    let ids = Tokenizer::byte_level().encode(&"the cat sat on the mat. ".repeat(40)).unwrap();
    let batcher = Batcher { batch: 4, seq: 16, order: Order::Sequential };
    for step in 0..12 { t.train_step(&batcher.at(&ids[..], step)).unwrap(); }
    let (dir, _) = out_dir("plain");
    t.to_model().unwrap().save_dir(dir.to_str().unwrap()).unwrap();
    let plain = std::fs::read(dir.join("model.safetensors")).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(plain, weights_after("nocb", &format!("invisible({TRAIN}))")),
               "the builtin's weights differ from the plain loop's");
}

#[test]
fn a_callback_and_an_inspection_change_no_weight() {
    let without = weights_after("without", &format!("invisible({TRAIN}, report = 3))"));
    let quiet = weights_after("quiet", &format!(
        "invisible({TRAIN}, report = 3, callback = function(step, loss) NULL))"));
    let looking = weights_after("looking", &format!(
        "invisible({TRAIN}, report = 3, callback = function(step, loss) llm.inspect(m, txt, seq = 8, batch = 2)))"));
    assert_eq!(without, quiet, "a callback changed the weights");
    assert_eq!(without, looking, "an inspection inside a callback changed the weights");
}

// ── callback = ───────────────────────────────────────────────────────

#[test]
fn the_callback_fires_where_report_prints_with_the_step_and_loss() {
    let calls = run(&format!(
        "{SETUP}s <- c()\nl <- c()\n\
         final <- {TRAIN}, report = 4, callback = function(step, loss) {{ s <<- c(s, step); l <<- c(l, loss) }})\n\
         list(s = s, l = l, final = final)\n"));
    let RVal::List(items, _) = &calls else { panic!("expected a list") };
    assert_eq!(nums(&items[0].1), [1.0, 4.0, 8.0, 12.0]);
    let losses = nums(&items[1].1);
    assert!(losses.iter().all(|x| x.is_finite() && *x > 0.0));
    assert!(losses[3] < losses[0], "the model did not learn: {losses:?}");
    assert_eq!(losses[3], nums(&items[2].1)[0], "step 12's loss is the returned loss");
}

#[test]
fn the_callback_works_on_finetune() {
    let seen = run(&format!(
        "{SETUP}p <- c(\"india:\", \"france:\")\nq <- c(\" delhi\", \" paris\")\ns <- c()\n\
         invisible(llm.finetune(m, p, q, steps = 6, batch = 2, seq = 16, report = 2,\n\
                                callback = function(step, loss) s <<- c(s, step)))\ns\n"));
    assert_eq!(nums(&seen), [1.0, 2.0, 4.0, 6.0]);
}

#[test]
fn a_callback_needs_a_report_and_a_function() {
    let msg = run_err(&format!("{SETUP}{TRAIN}, callback = function(step, loss) NULL)\n"));
    assert!(msg.contains("report"), "{msg}");
    let msg = run_err(&format!("{SETUP}{TRAIN}, report = 2, callback = \"cat\")\n"));
    assert!(msg.contains("function"), "{msg}");
}

#[test]
fn stop_in_a_callback_ends_the_run_and_keeps_the_weights() {
    // Stopped after step 4, the model is at step 4: eight more steps reach
    // the weights of twelve in one go.
    let stopped = weights_after("stopped", &format!(
        "r <- tryCatch({TRAIN}, report = 4,\n\
                       callback = function(step, loss) if (step == 4) stop(\"enough\")),\n\
                       error = function(e) conditionMessage(e))\n\
         stopifnot(grepl(\"enough\", r))\n\
         invisible(llm.train(m, txt, steps = 8, seq = 16, batch = 4))"));
    let whole = weights_after("whole", &format!("invisible({TRAIN}))"));
    assert_eq!(stopped, whole);
}

/// The registry lock must be free while a callback runs: these all take it.
#[test]
fn a_callback_can_read_the_model_it_is_training() {
    let out = run(&format!(
        "{SETUP}said <- \"\"\nn <- 0\n\
         invisible({TRAIN}, report = 6, callback = function(step, loss) {{\n\
             n <<- llm.info(m)$params\n\
             said <<- llm.generate(m, \"the\", n = 2)\n\
         }}))\n\
         c(said, as.character(n > 0))\n"));
    let out = text(&out);
    assert!(!out[0].is_empty());
    assert_eq!(out[1], "TRUE");
}

#[test]
fn a_callback_cannot_change_the_model_it_is_training() {
    for call in ["llm.free(m)", "llm.freeze(m, \"tok_embed\")", "llm.lora(m)",
                 "llm.train(m, txt, steps = 1, seq = 16, batch = 4)",
                 "llm.finetune(m, \"a\", \"b\", steps = 1)"] {
        let msg = run_err(&format!("{SETUP}{TRAIN}, report = 1, callback = function(step, loss) {call})\n"));
        assert!(msg.contains("still training"), "{call}: {msg}");
    }
}

#[test]
fn the_handle_is_released_after_a_callback_fails() {
    let freed = run(&format!(
        "{SETUP}try({TRAIN}, report = 1, callback = function(step, loss) stop(\"no\")), silent = TRUE)\n\
         llm.free(m)\n"));
    assert!(matches!(&freed, RVal::Logical(x, _) if x.iter().next() == Some(&Some(true))));
}

#[test]
fn a_callback_can_train_another_model() {
    let v = run(&format!(
        "{SETUP}o <- llm.new(vocab = 256, dim = 16, layers = 1, heads = 2, ffn = 32, ctx = 16)\nv <- 0\n\
         invisible({TRAIN}, report = 6, callback = function(step, loss)\n\
             v <<- llm.train(o, txt, steps = 1, seq = 8, batch = 2)))\nv\n"));
    assert!(nums(&v)[0] > 0.0);
}

#[cfg(feature = "gpu")]
#[test]
fn a_gpu_run_shows_its_callback_the_current_weights() {
    let same = run(&format!(
        "{SETUP}before <- llm.generate(m, \"the\", n = 8)\nsaid <- \"\"\n\
         invisible({TRAIN}, device = \"gpu\", report = 12,\n\
                   callback = function(step, loss) said <<- llm.generate(m, \"the\", n = 8)))\n\
         c(said != before, said == llm.generate(m, \"the\", n = 8))\n"));
    let RVal::Logical(x, _) = &same else { panic!("expected logicals, got {same:?}") };
    assert_eq!(x.iter().copied().collect::<Vec<_>>(), [Some(true), Some(true)]);
}

// ── llm.inspect ──────────────────────────────────────────────────────

fn column(frame: &RVal, name: &str) -> RVal {
    let RVal::DataFrame(df) = frame else { panic!("expected a data frame, got {frame:?}") };
    df.get_col(name).unwrap_or_else(|| panic!("no column {name}")).clone()
}

#[test]
fn inspect_gives_a_row_per_block_and_per_stage() {
    let look = run(&format!("{SETUP}llm.inspect(m, txt, seq = 8, batch = 2)\n"));
    let RVal::List(items, _) = &look else { panic!("expected a list") };
    assert!(nums(&items[0].1)[0] > 0.0);

    let blocks = text(&column(&items[1].1, "block"));
    assert_eq!(blocks.len(), 3 + 2 * 9);
    assert_eq!(blocks[..4], ["tok_embed", "final_norm", "output", "layer.0.attn_norm"].map(String::from));
    let params = nums(&column(&items[1].1, "params"));
    assert_eq!(params.iter().sum::<f64>(), nums(&run(&format!("{SETUP}llm.info(m)$params\n")))[0]);
    assert!(nums(&column(&items[1].1, "grad.norm")).iter().all(|g| *g > 0.0));
    assert!(nums(&column(&items[1].1, "grad.ratio")).iter().all(|g| g.is_finite()));

    let stages = text(&column(&items[2].1, "stage"));
    assert_eq!(stages.first().map(String::as_str), Some("embed"));
    assert_eq!(stages.last().map(String::as_str), Some("final"));
    assert_eq!(stages.len(), 2 + 2 * 3);
    assert!(nums(&column(&items[2].1, "nonfinite")).iter().all(|n| *n == 0.0));
}

#[test]
fn inspect_shows_a_frozen_block_as_na() {
    let g = run(&format!(
        "{SETUP}invisible(llm.freeze(m, \"tok_embed\"))\n\
         look <- llm.inspect(m, txt, seq = 8, batch = 2)\n\
         c(is.na(look$blocks$grad.norm[1]), is.na(look$blocks$grad.norm[2]))\n"));
    let RVal::Logical(x, _) = &g else { panic!("expected logicals, got {g:?}") };
    assert_eq!(x.iter().copied().collect::<Vec<_>>(), [Some(true), Some(false)]);
}

#[test]
fn inspect_takes_finetune_examples_and_sees_adapters() {
    let names = run(&format!(
        "{SETUP}invisible(llm.lora(m, rank = 2))\n\
         llm.inspect(m, c(\"india:\", \"france:\"), completions = c(\" delhi\", \" paris\"), seq = 16, batch = 2)$blocks$block\n"));
    let names = text(&names);
    assert!(names.len() > 3 + 2 * 9, "no adapter blocks in {names:?}");
}

#[test]
fn inspect_refuses_a_serving_handle() {
    let (dir, q) = out_dir("serving");
    let msg = run_err(&format!(
        "{SETUP}invisible(llm.save(m, \"{q}\"))\ns <- llm.load(\"{q}\")\nllm.inspect(s, txt)\n"));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(msg.contains("train = TRUE"), "{msg}");
}

// ── llm.new(preset =) ────────────────────────────────────────────────

#[test]
fn a_preset_sets_the_shape_and_named_arguments_override_it() {
    let info = |args: &str, field: &str| nums(&run(&format!("llm.info(llm.new({args}))${field}\n")))[0];
    assert_eq!(info("preset = \"tiny\"", "params"), 1_300_352.0);
    assert_eq!(info("preset = \"tiny\"", "dim"), 128.0);
    assert_eq!(info("preset = \"tiny\", layers = 2", "layers"), 2.0);
    assert_eq!(info("preset = \"tiny\", layers = 2", "dim"), 128.0, "an override moved another field");
    assert_eq!(info("preset = \"tiny\", vocab = 256", "vocab"), 256.0);
    assert_eq!(info("preset = \"tiny\", ctx = 32", "ctx"), 32.0);
}

#[test]
fn a_preset_leaves_the_starting_weights_to_the_seed() {
    let weights = |args: &str| {
        let (dir, q) = out_dir("preset_seed");
        run(&format!("m <- llm.new({args})\ninvisible(llm.save(m, \"{q}\"))\n"));
        let w = std::fs::read(dir.join("model.safetensors")).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        w
    };
    let spelled = "dim = 128, layers = 4, heads = 4, kv.heads = 2, ffn = 384, ctx = 64, vocab = 2000";
    assert_eq!(weights("preset = \"tiny\", seed = 9"), weights(&format!("{spelled}, seed = 9")));
    assert_ne!(weights("preset = \"tiny\", seed = 9"), weights("preset = \"tiny\", seed = 10"));
}

#[test]
fn an_unknown_preset_names_the_ones_there_are() {
    let msg = run_err("llm.new(preset = \"llama\")\n");
    for name in ["tiny", "small", "medium", "large"] { assert!(msg.contains(name), "{msg}"); }
}

#[test]
fn without_a_preset_the_defaults_are_unchanged() {
    let i = run("i <- llm.info(llm.new())\nc(i$dim, i$layers, i$heads, i$kv.heads, i$ffn, i$ctx, i$vocab)\n");
    assert_eq!(nums(&i), [128.0, 4.0, 4.0, 4.0, 384.0, 64.0, 8000.0]);
}

// ── ?help ────────────────────────────────────────────────────────────

#[test]
fn help_describes_the_new_arguments() {
    let help = |topic: &str| r2_types::out::capture(|| run(&format!("help(\"{topic}\")\n"))).1;
    assert!(help("llm.new").contains("preset"), "{}", help("llm.new"));
    assert!(help("llm.train").contains("callback"), "{}", help("llm.train"));
    assert!(help("llm.finetune").contains("callback"), "{}", help("llm.finetune"));
    assert!(help("llm.inspect").contains("grad.norm"), "{}", help("llm.inspect"));
}
