//! End-to-end: text -> tokens -> generated tokens -> text, plus a
//! safetensors round-trip. Proves the serving pieces compose.
use r2_tensor::infer::{Sampler, SamplerConfig};
use r2_tensor::model::{Config, Model};
use r2_tensor::safetensors::SafeTensors;
use r2_tensor::tokenizer::Tokenizer;
use std::io::Write;

fn main() -> Result<(), String> {
    // ── 1. Tokenizer (byte-level: anything encodes) ──
    let tok = Tokenizer::byte_level();
    let prompt = "R2 says:";
    let ids: Vec<usize> = tok.encode(prompt)?.iter().map(|&i| i as usize).collect();
    println!("prompt {:?} -> {} tokens {:?}", prompt, ids.len(), &ids);

    // ── 2. Model sized to the tokenizer's vocab ──
    let c = Config { dim: 64, n_heads: 8, n_kv_heads: 4, n_layers: 4, vocab: tok.vocab_size(),
                     ffn_hidden: 128, max_seq: 128, rope_base: 10000.0, eps: 1e-5 };
    let mut m = Model::zeros(c);
    let f = |n: usize, s: f32| (0..n).map(|i| ((i as f32 * 0.0731 + s).sin()) * 0.25).collect::<Vec<f32>>();
    m.tok_embed = f(c.vocab * c.dim, 0.7);
    m.final_norm = vec![1.0; c.dim];
    m.output = f(c.dim * c.vocab, 1.9);
    for (i, l) in m.layers.iter_mut().enumerate() {
        let s = i as f32 * 3.1;
        l.attn_norm = vec![1.0; c.dim]; l.ffn_norm = vec![1.0; c.dim];
        l.wq = f(c.dim*c.dim, s+0.2); l.wk = f(c.dim*c.dim, s+0.5);
        l.wv = f(c.dim*c.dim, s+0.9); l.wo = f(c.dim*c.dim, s+1.3);
        l.w1 = f(c.dim*c.ffn_hidden, s+1.7); l.w2 = f(c.ffn_hidden*c.dim, s+2.1);
        l.w3 = f(c.dim*c.ffn_hidden, s+2.5);
    }
    println!("model: {} params, vocab {}", m.cfg.n_params(), c.vocab);

    // ── 3. Generate, then detokenize ──
    let mut s = Sampler::new(7, SamplerConfig { temperature: 0.8, top_k: 40, ..Default::default() });
    let t0 = std::time::Instant::now();
    let out = m.generate(&ids, 24, &mut s, None)?;
    let dt = t0.elapsed();
    let out32: Vec<u32> = out.iter().map(|&i| i as u32).collect();
    println!("generated {} tokens in {:?} ({:.0} tok/s)", out.len(), dt,
             out.len() as f64 / dt.as_secs_f64());
    println!("decoded: {:?}", tok.decode(&out32));

    // ── 4. safetensors: write a checkpoint by hand, read it back ──
    let path = std::env::temp_dir().join("r2_demo.safetensors");
    {
        let vals: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let mut data = Vec::new();
        for v in &vals { data.extend_from_slice(&v.to_le_bytes()); }
        let header = r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[2,4],"data_offsets":[0,32]}}"#;
        let mut fh = std::fs::File::create(&path).map_err(|e| e.to_string())?;
        fh.write_all(&(header.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;
        fh.write_all(header.as_bytes()).map_err(|e| e.to_string())?;
        fh.write_all(&data).map_err(|e| e.to_string())?;
    }
    let st = SafeTensors::open(&path)?;
    println!("\ncheckpoint: {:?}", st);
    println!("  tensors: {:?}", st.names());
    println!("  w (shape-checked) = {:?}", st.tensor_f32_shaped("w", &[2, 4])?);
    let _ = std::fs::remove_file(&path);
    println!("\nOK — tokenizer, model, generator and loader compose.");
    Ok(())
}
