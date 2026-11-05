//! Fine-tuning from R2 script: `llm.finetune`, `llm.freeze` and the settings
//! `llm.train` and `llm.finetune` share.

use r2_engine::Engine;
use r2_parser::Parser;
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

fn scalar(v: &RVal) -> f64 {
    match v {
        RVal::Numeric(d, _) => d.iter().next().and_then(|x| *x).expect("scalar"),
        other => panic!("expected a number, got {:?}", other),
    }
}

fn text(v: &RVal) -> String {
    match v {
        RVal::Character(c, _) => c.iter().next().cloned().flatten().expect("a string").to_string(),
        other => panic!("expected text, got {:?}", other),
    }
}

const MODEL: &str = "m <- llm.new(vocab = 256, dim = 32, layers = 1, heads = 2, ffn = 64, ctx = 32, lr = 0.01, seed = 42)\n";
const EXAMPLES: &str = "p <- c(\"india:\", \"france:\", \"japan:\", \"kenya:\")\n\
                        q <- c(\" delhi\", \" paris\", \" tokyo\", \" nairobi\")\n";

#[test]
fn a_model_learns_what_follows_each_prompt() {
    let out = text(&run(&format!(
        "{MODEL}{EXAMPLES}\
         invisible(llm.finetune(m, p, q, steps = 150, batch = 2, seq = 16))\n\
         llm.generate(m, \"france:\", n = 6)\n")));
    assert_eq!(out, " paris", "the model did not learn the completion");
}

#[test]
fn settings_reach_the_trainer() {
    let base = |extra: &str| scalar(&run(&format!(
        "{MODEL}{EXAMPLES}llm.finetune(m, p, q, steps = 12, batch = 4, seq = 16, shuffle = FALSE{extra})\n")));
    let plain = base("");
    for extra in [", wd = 0.1", ", clip = 0.01", ", schedule = \"cosine\", warmup = 4", ", lr = 0.001"] {
        assert_ne!(plain, base(extra), "{extra} changed nothing");
    }
    assert_eq!(plain, base(""), "and the same settings repeat exactly");
    assert!(run_err(&format!("{MODEL}{EXAMPLES}llm.finetune(m, p, q, schedule = \"step\")")).contains("unknown schedule"));
}

#[test]
fn accumulation_is_the_same_update_in_less_memory() {
    let run_with = |extra: &str| scalar(&run(&format!(
        "{MODEL}{EXAMPLES}llm.finetune(m, p, q, steps = 6, batch = 4, seq = 16, shuffle = FALSE{extra})\n")));
    let (whole, parts) = (run_with(""), run_with(", accum = 2"));
    assert!((whole - parts).abs() < 1e-3, "accum = 2 gave {parts}, the whole batch {whole}");
}

#[test]
fn a_frozen_block_does_not_train() {
    let dir = std::env::temp_dir().join(format!("r2_llm_frozen_{}", std::process::id()));
    let d = dir.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/");
    let count = scalar(&run(&format!(
        "{MODEL}{EXAMPLES}invisible(llm.save(m, \"{d}/before\"))\n\
         n <- llm.freeze(m, c(\"tok_embed\", \"layer.0.wq\"))\n\
         invisible(llm.finetune(m, p, q, steps = 5, batch = 2, seq = 16))\n\
         invisible(llm.save(m, \"{d}/after\"))\nn\n")));
    assert_eq!(count, 2.0);
    let open = |sub: &str| r2_tensor::safetensors::SafeTensors::open(dir.join(sub).join("model.safetensors")).unwrap();
    let (a, b) = (open("before"), open("after"));
    for name in ["tok_embed", "layer.0.wq"] {
        assert_eq!(a.raw(name).unwrap(), b.raw(name).unwrap(), "{name} moved while frozen");
    }
    assert_ne!(a.raw("output").unwrap(), b.raw("output").unwrap(), "the rest must still train");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(run_err(&format!("{MODEL}llm.freeze(m, \"layr.0\")")).contains("layr.0"));
}

#[test]
fn lora_trains_small_adapters_and_they_travel_two_ways() {
    let dir = std::env::temp_dir().join(format!("r2_llm_lora_{}", std::process::id()));
    let d = dir.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/");
    let script = format!(
        "{MODEL}{EXAMPLES}invisible(llm.save(m, \"{d}/base\"))\n\
         a <- llm.load(\"{d}/base\", train = TRUE, lr = 0.01)\n\
         n <- llm.lora(a, rank = 4, alpha = 8)\n\
         invisible(llm.finetune(a, p, q, steps = 250, batch = 2, seq = 16))\n\
         x <- llm.generate(a, \"japan:\", n = 6)\n\
         invisible(llm.save.adapter(a, \"{d}/adapter\"))\n\
         invisible(llm.save(a, \"{d}/merged\"))\n\
         y <- llm.generate(llm.load.adapter(\"{d}/base\", \"{d}/adapter\"), \"japan:\", n = 6)\n\
         z <- llm.generate(llm.load(\"{d}/merged\"), \"japan:\", n = 6)\n\
         paste0(n, \"|\", x, \"|\", y, \"|\", z)\n");
    let out = text(&run(&script));
    let v: Vec<&str> = out.split('|').collect();
    assert!(v[0].parse::<f64>().unwrap() > 0.0, "{out}");
    assert_eq!(v[1], " tokyo", "adapters must teach the completion: {out}");
    assert_eq!(v[1], v[2], "base + adapter file must generate the same");
    assert_eq!(v[1], v[3], "and so must the merged model");

    // the base is untouched, and the adapter file is much smaller than it
    let size = |sub: &str, f: &str| std::fs::metadata(dir.join(sub).join(f)).unwrap().len();
    assert!(size("adapter", "adapter.safetensors") * 3 < size("base", "model.safetensors"));

    // another base is refused by name
    let other = format!("{MODEL}invisible(llm.train(m, \"changing the weights of this model now\", steps = 2, seq = 8, batch = 2))\n\
                         invisible(llm.save(m, \"{d}/other\"))\nllm.load.adapter(\"{d}/other\", \"{d}/adapter\")\n");
    assert!(run_err(&other).contains("different base"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The device runs the same recipe: same losses to within f32 rounding, and
/// the weights come back, so generating and saving work as after a CPU run.
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[test]
fn the_gpu_runs_the_same_recipe() {
    if r2_gpu::adapter_info().contains("no GPU") { eprintln!("no GPU adapter; skipped"); return; }
    let go = |device: &str| text(&run(&format!(
        "{MODEL}{EXAMPLES}\
         a <- llm.finetune(m, p, q, steps = 150, batch = 2, seq = 16, wd = 0.01, clip = 1, \
                           schedule = \"cosine\", warmup = 5, device = \"{device}\")\n\
         paste0(a, \"|\", llm.generate(m, \"france:\", n = 6))\n")));
    let (cpu, gpu) = (go("cpu"), go("gpu"));
    let split = |s: &str| { let (l, t) = s.split_once('|').unwrap(); (l.parse::<f64>().unwrap(), t.to_string()) };
    let ((lc, tc), (lg, tg)) = (split(&cpu), split(&gpu));
    assert!((lc - lg).abs() < 2e-3 * lc.max(1.0), "loss cpu {lc}, gpu {lg}");
    assert_eq!(tc, tg, "cpu and gpu learned different things");
    assert_eq!(tg, " paris");
}

/// Adapters are CPU only so far: an error that says so, not a wrong run.
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[test]
fn lora_on_the_gpu_is_refused_by_name() {
    if r2_gpu::adapter_info().contains("no GPU") { return; }
    let e = run_err(&format!("{MODEL}{EXAMPLES}invisible(llm.lora(m, rank = 2))\n\
                              llm.finetune(m, p, q, steps = 1, device = \"gpu\")"));
    assert!(e.contains("CPU only"), "{e}");
}

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
#[test]
fn a_build_without_gpu_support_says_so() {
    assert!(run_err(&format!("{MODEL}{EXAMPLES}llm.finetune(m, p, q, steps = 1, device = \"gpu\")")).contains("--features gpu"));
}

#[test]
fn an_unknown_device_is_refused() {
    assert!(run_err(&format!("{MODEL}{EXAMPLES}llm.finetune(m, p, q, steps = 1, device = \"tpu\")")).contains("unknown device"));
}

#[test]
fn bad_examples_are_refused_by_name() {
    assert!(run_err(&format!("{MODEL}llm.finetune(m, c(\"a\", \"b\"), \"c\")")).contains("2 prompts but 1"));
    assert!(run_err(&format!("{MODEL}llm.finetune(m, \"\", \"c\")")).contains("example 1: empty prompt"));
    assert!(run_err(&format!("{MODEL}llm.finetune(m, \"abcdefghij\", \"c\", seq = 4)")).contains("seq = 4"));
}

#[test]
fn a_fine_tune_resumes_on_its_base() {
    let dir = std::env::temp_dir().join(format!("r2_llm_tune_resume_{}", std::process::id()));
    let d = dir.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/");
    run(&format!("{MODEL}invisible(llm.train(m, \"some base text for the first model to learn from\", steps = 3, seq = 8, batch = 2))\n\
                  invisible(llm.save(m, \"{d}/base\"))\n"));
    let start = format!("{EXAMPLES}s <- llm.load(\"{d}/base\", train = TRUE, lr = 0.005)\ninvisible(llm.freeze(s, \"tok_embed\"))\n");
    let whole = scalar(&run(&format!("{start}llm.finetune(s, p, q, steps = 8, batch = 2, seq = 16)\n")));
    run(&format!("{start}invisible(llm.finetune(s, p, q, steps = 3, batch = 2, seq = 16, checkpoint = \"{d}/ck\"))\n"));
    assert!(run_err(&format!("llm.resume(\"{d}/ck\")")).contains("base"));
    let resumed = scalar(&run(&format!(
        "{EXAMPLES}s <- llm.resume(\"{d}/ck\", base = \"{d}/base\")\nllm.finetune(s, p, q, steps = 5, batch = 2, seq = 16)\n")));
    assert_eq!(whole, resumed, "3 + 5 steps across a process boundary must be 8 steps");
    let _ = std::fs::remove_dir_all(&dir);
}
