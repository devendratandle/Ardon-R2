//! The complete R2 model lifecycle, with no external model, no external
//! format and no vendor dependency: build → save → inspect → load → serve.
use r2_tensor::infer::{Sampler, SamplerConfig};
use r2_tensor::model::{Config, Model};
use r2_tensor::tokenizer::Tokenizer;

fn main() -> Result<(), String> {
    let tok = Tokenizer::byte_level();
    let c = Config { dim: 64, n_heads: 8, n_kv_heads: 2, n_layers: 4,
                     vocab: tok.vocab_size(), ffn_hidden: 128, max_seq: 128,
                     rope_base: 10000.0, eps: 1e-5 };
    let mut m = Model::zeros(c);
    let f = |n: usize, s: f32| (0..n).map(|i| ((i as f32 * 0.0731 + s).sin()) * 0.25).collect::<Vec<f32>>();
    m.tok_embed = f(c.vocab * c.dim, 0.7);
    m.final_norm = vec![1.0; c.dim];
    m.output = f(c.dim * c.vocab, 1.9);
    for (i, l) in m.layers.iter_mut().enumerate() {
        let s = i as f32 * 3.1;
        l.attn_norm = vec![1.0; c.dim]; l.ffn_norm = vec![1.0; c.dim];
        l.wq = f(c.dim*c.dim, s+0.2); l.wk = f(c.dim*c.kv_dim(), s+0.5);
        l.wv = f(c.dim*c.kv_dim(), s+0.9); l.wo = f(c.dim*c.dim, s+1.3);
        l.w1 = f(c.dim*c.ffn_hidden, s+1.7); l.w2 = f(c.ffn_hidden*c.dim, s+2.1);
        l.w3 = f(c.dim*c.ffn_hidden, s+2.5);
    }

    let dir = std::env::temp_dir().join("r2-demo-model");
    println!("1. SAVE   {} params -> {}", m.cfg.n_params(), dir.display());
    m.save_dir(&dir)?;
    for e in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
        let e = e.map_err(|x| x.to_string())?;
        let sz = e.metadata().map(|x| x.len()).unwrap_or(0);
        println!("          {:<20} {:>8} bytes", e.file_name().to_string_lossy(), sz);
    }

    // Inspecting a model must not require loading its weights.
    let peek = Model::peek_config(&dir)?;
    println!("2. INSPECT (no weights read): dim {} heads {}/{} layers {} ctx {}",
             peek.dim, peek.n_heads, peek.n_kv_heads, peek.n_layers, peek.max_seq);

    println!("3. LOAD");
    let loaded = Model::load_dir(&dir)?;

    println!("4. SERVE");
    let ids: Vec<usize> = tok.encode("R2:")?.iter().map(|&i| i as usize).collect();
    let mut s = Sampler::new(7, SamplerConfig { temperature: 0.8, top_k: 40, ..Default::default() });
    let t0 = std::time::Instant::now();
    let out = loaded.generate(&ids, 32, &mut s, None)?;
    println!("          {} tokens in {:?} ({:.0} tok/s)", out.len(), t0.elapsed(),
             out.len() as f64 / t0.elapsed().as_secs_f64());

    // The proof: the loaded model must behave exactly like the original.
    let mut s2 = Sampler::new(7, SamplerConfig { temperature: 0.8, top_k: 40, ..Default::default() });
    let orig = m.generate(&ids, 32, &mut s2, None)?;
    println!("5. VERIFY loaded == original: {}", if out == orig { "IDENTICAL" } else { "DIFFERENT" });
    assert_eq!(out, orig);

    let _ = std::fs::remove_dir_all(&dir);
    println!("\nEntire lifecycle used only R2 code and R2 formats.");
    Ok(())
}
