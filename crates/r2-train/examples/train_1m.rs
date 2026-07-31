//! Train a ~1M-parameter language model end to end, entirely in R2:
//! define → train → save → load → generate. No external data, no external
//! model, no external format.
use r2_tensor::infer::{Sampler, SamplerConfig};
use r2_tensor::model::{Config, Model};
use r2_tensor::tokenizer::Tokenizer;
use r2_train::llm::Trainer;

fn main() -> Result<(), String> {
    let tok = Tokenizer::byte_level();
    let cfg = Config {
        dim: 128, n_heads: 4, n_kv_heads: 2, n_layers: 5,
        vocab: tok.vocab_size(), ffn_hidden: 384, max_seq: 64,
        rope_base: 10000.0, eps: 1e-5,
    };
    println!("model: {} parameters ({:.2}M), {} layers, {} q-heads / {} kv-heads",
             cfg.n_params(), cfg.n_params() as f64 / 1e6, cfg.n_layers,
             cfg.n_heads, cfg.n_kv_heads);

    let corpus = "ardon r2 is a statistical runtime. ";
    let ids: Vec<usize> = tok.encode(corpus)?.iter().map(|&i| i as usize).collect();
    let seq = 16usize;
    let mut batch = Vec::new();
    for start in 0..(ids.len() - seq - 1).min(6) {
        batch.push((ids[start..start + seq].to_vec(),
                    ids[start + 1..start + seq + 1].to_vec()));
    }
    println!("corpus: {:?} -> {} training sequences of length {}\n",
             corpus, batch.len(), seq);

    let mut tr = Trainer::new(cfg, 0.003, 42)?;
    assert_eq!(tr.n_params(), cfg.n_params());

    let t0 = std::time::Instant::now();
    let mut first = 0.0f32;
    let steps = 120;
    for s in 1..=steps {
        let loss = tr.train_step(&batch)?;
        if s == 1 { first = loss; }
        if s == 1 || s % 20 == 0 {
            println!("  step {s:>3}  loss {loss:.4}  ({:?} elapsed)", t0.elapsed());
        }
        if s == steps {
            println!("\ntrained {steps} steps in {:?}  |  loss {first:.4} -> {loss:.4} ({:.0}% lower)",
                     t0.elapsed(), (1.0 - loss / first) * 100.0);
        }
    }

    let model = tr.to_model()?;
    let dir = std::env::temp_dir().join("r2-1m-model");
    model.save_dir(&dir)?;
    let size: u64 = std::fs::read_dir(&dir).map_err(|e| e.to_string())?
        .flatten().filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum();
    println!("saved to {} ({:.1} MB)", dir.display(), size as f64 / 1e6);
    let served = Model::load_dir(&dir)?;

    let prompt = "ardon r2 is";
    let p: Vec<usize> = tok.encode(prompt)?.iter().map(|&i| i as usize).collect();
    let mut s = Sampler::new(1, SamplerConfig { temperature: 0.0, ..Default::default() });
    let out = served.generate(&p, 24, &mut s, None)?;
    let text = tok.decode(&out.iter().map(|&i| i as u32).collect::<Vec<_>>());
    println!("\nprompt      : {:?}", prompt);
    println!("continuation: {:?}", text);

    let mut s2 = Sampler::new(1, SamplerConfig { temperature: 0.0, ..Default::default() });
    assert_eq!(out, tr.to_model()?.generate(&p, 24, &mut s2, None)?,
               "saved+loaded model must generate identically");
    println!("\nverified: trained == exported == saved == loaded");
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
