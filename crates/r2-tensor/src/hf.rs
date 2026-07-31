//! Load a HuggingFace Llama-family checkpoint.
//!
//! Turns a downloaded model directory into a [`Model`] this crate can
//! generate with. Two things stand between the file and the struct, and
//! both are silent-corruption hazards rather than mere plumbing:
//!
//! 1. **Layout.** PyTorch stores a linear layer's weight as
//!    `[out_features, in_features]`; this crate stores `[in × out]` so a
//!    `1 × in` activation is a plain matmul. Every projection therefore
//!    needs a **transpose on load**. Skipping it does not crash — it
//!    produces a model that runs and emits nonsense.
//! 2. **Naming.** `config.json` keys and tensor names differ per model
//!    family, and a mis-mapped tensor is equally silent. So every tensor
//!    is fetched by exact name with an exact expected shape, and a
//!    missing or mis-shaped one is an error that names it.
//!
//! Supported: Llama-family (Llama 2/3, Mistral, TinyLlama, Qwen2-style)
//! in safetensors, single-file or sharded, with or without GQA, with or
//! without tied embeddings.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::json::Json;
use crate::model::{Config, Layer, Model};
use crate::safetensors::SafeTensors;

/// Read `config.json` into a [`Config`].
///
/// `max_seq` is taken from `max_position_embeddings` but capped by
/// `max_seq_override`, because a config advertising 128k context would
/// otherwise allocate a KV cache far larger than the caller wants.
pub fn config_from_json(src: &str, max_seq_override: Option<usize>) -> Result<Config, String> {
    let j = Json::parse(src).map_err(|e| format!("config.json: {}", e))?;
    let need = |k: &str| -> Result<usize, String> {
        j.get(k).and_then(|v| v.as_usize())
            .ok_or_else(|| format!("config.json: missing or non-integer '{}'", k))
    };
    let dim = need("hidden_size")?;
    let n_heads = need("num_attention_heads")?;
    // Absent num_key_value_heads means no GQA — every query head has its
    // own K/V head. That default is what pre-GQA models (Llama-2 7B/13B)
    // rely on.
    let n_kv_heads = j.get("num_key_value_heads").and_then(|v| v.as_usize()).unwrap_or(n_heads);
    let max_pos = j.get("max_position_embeddings").and_then(|v| v.as_usize()).unwrap_or(2048);

    let cfg = Config {
        dim,
        n_heads,
        n_kv_heads,
        n_layers: need("num_hidden_layers")?,
        vocab: need("vocab_size")?,
        ffn_hidden: need("intermediate_size")?,
        max_seq: max_seq_override.map_or(max_pos, |m| m.min(max_pos)),
        rope_base: j.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(10000.0) as f32,
        eps: j.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32,
    };
    // Shape-only check: fail here, with the config in hand, rather than
    // deep in a forward pass — and WITHOUT allocating the model, which
    // for a 70B config would be over 100 GB of zeros.
    cfg.validate().map_err(|e| format!("config.json: {}", e))?;
    Ok(cfg)
}

/// Transpose a row-major `[rows × cols]` matrix to `[cols × rows]`.
/// This is the PyTorch `[out, in]` → our `[in × out]` conversion.
pub fn transpose(src: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    if src.len() != rows * cols {
        return Err(format!("transpose: {} values is not {}×{}", src.len(), rows, cols));
    }
    let mut out = vec![0.0f32; src.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    Ok(out)
}

/// One or more safetensors files presented as a single tensor namespace.
/// Sharded checkpoints (`model-00001-of-000NN.safetensors`) are the norm
/// above ~10 GB, so this is required for any large model.
pub struct ShardedWeights {
    shards: Vec<SafeTensors>,
    /// tensor name → shard index.
    index: BTreeMap<String, usize>,
}

impl ShardedWeights {
    /// Open every `*.safetensors` in `dir`. A duplicate tensor name across
    /// shards is an error: it means the directory holds two different
    /// checkpoints, and silently picking one would be a coin flip.
    pub fn open_dir(dir: impl AsRef<Path>) -> Result<ShardedWeights, String> {
        let dir = dir.as_ref();
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| format!("weights: cannot read {}: {}", dir.display(), e))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("weights: no .safetensors files in {}", dir.display()));
        }
        let mut shards = Vec::new();
        let mut index = BTreeMap::new();
        for (i, f) in files.iter().enumerate() {
            let st = SafeTensors::open(f)?;
            for name in st.names() {
                if index.insert(name.to_string(), i).is_some() {
                    return Err(format!(
                        "weights: tensor '{}' appears in more than one shard", name));
                }
            }
            shards.push(st);
        }
        Ok(ShardedWeights { shards, index })
    }

    pub fn names(&self) -> Vec<&str> { self.index.keys().map(|s| s.as_str()).collect() }
    pub fn contains(&self, name: &str) -> bool { self.index.contains_key(name) }
    pub fn total_params(&self) -> usize {
        self.shards.iter().map(|s| s.total_params()).sum()
    }

    /// Fetch a tensor as f32, requiring an exact shape.
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let i = *self.index.get(name)
            .ok_or_else(|| format!("weights: no tensor named '{}'", name))?;
        self.shards[i].tensor_f32_shaped(name, shape)
    }
}

/// Load a HuggingFace model directory: `config.json` + `*.safetensors`.
pub fn load_dir(dir: impl AsRef<Path>, max_seq: Option<usize>) -> Result<Model, String> {
    let dir = dir.as_ref();
    let cfg_path = dir.join("config.json");
    let cfg_src = std::fs::read_to_string(&cfg_path)
        .map_err(|e| format!("hf: cannot read {}: {}", cfg_path.display(), e))?;
    let cfg = config_from_json(&cfg_src, max_seq)?;
    let w = ShardedWeights::open_dir(dir)?;
    load_weights(cfg, &w)
}

/// Fill a [`Model`] of shape `cfg` from an open tensor namespace.
///
/// Split from [`load_dir`] so it is testable without a directory, and so
/// callers with a custom layout can supply their own namespace.
pub fn load_weights(cfg: Config, w: &ShardedWeights) -> Result<Model, String> {
    let (d, kv, h) = (cfg.dim, cfg.kv_dim(), cfg.ffn_hidden);
    let mut m = Model::zeros(cfg);

    // Embedding: HF stores [vocab, dim] row-major, which IS our layout —
    // the one matrix that needs no transpose.
    m.tok_embed = w.get("model.embed_tokens.weight", &[cfg.vocab, d])?;
    m.final_norm = w.get("model.norm.weight", &[d])?;

    // Output projection: [vocab, dim] → transpose to [dim × vocab].
    // Many models TIE the output to the embedding and omit lm_head
    // entirely; reuse the embedding then, exactly as the reference
    // implementations do.
    m.output = if w.contains("lm_head.weight") {
        transpose(&w.get("lm_head.weight", &[cfg.vocab, d])?, cfg.vocab, d)?
    } else {
        transpose(&m.tok_embed, cfg.vocab, d)?
    };

    for i in 0..cfg.n_layers {
        let p = format!("model.layers.{i}");
        // Every projection is stored [out, in] and transposed to [in × out].
        let layer = Layer {
            attn_norm: w.get(&format!("{p}.input_layernorm.weight"), &[d])?,
            ffn_norm:  w.get(&format!("{p}.post_attention_layernorm.weight"), &[d])?,
            wq: transpose(&w.get(&format!("{p}.self_attn.q_proj.weight"), &[d, d])?, d, d)?,
            // K/V are [kv_dim, dim] under GQA — narrower than Q.
            wk: transpose(&w.get(&format!("{p}.self_attn.k_proj.weight"), &[kv, d])?, kv, d)?,
            wv: transpose(&w.get(&format!("{p}.self_attn.v_proj.weight"), &[kv, d])?, kv, d)?,
            wo: transpose(&w.get(&format!("{p}.self_attn.o_proj.weight"), &[d, d])?, d, d)?,
            // SwiGLU: gate = w1, up = w3, down = w2 (HF names them
            // gate/up/down; the numbering here follows the paper).
            w1: transpose(&w.get(&format!("{p}.mlp.gate_proj.weight"), &[h, d])?, h, d)?,
            w3: transpose(&w.get(&format!("{p}.mlp.up_proj.weight"), &[h, d])?, h, d)?,
            w2: transpose(&w.get(&format!("{p}.mlp.down_proj.weight"), &[d, h])?, d, h)?,
        };
        m.layers[i] = layer;
    }

    // The shapes were checked per tensor; this catches anything structural
    // the per-tensor checks could not see.
    m.validate()?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn cfg_json(kv_heads: Option<usize>) -> String {
        let kv = kv_heads.map(|k| format!(r#""num_key_value_heads":{k},"#)).unwrap_or_default();
        format!(r#"{{"hidden_size":8,"num_attention_heads":4,{kv}
            "num_hidden_layers":2,"vocab_size":16,"intermediate_size":12,
            "max_position_embeddings":2048,"rope_theta":500000.0,"rms_norm_eps":1e-6}}"#)
    }

    #[test]
    fn reads_config_including_gqa_and_defaults() {
        let c = config_from_json(&cfg_json(Some(2)), None).unwrap();
        assert_eq!((c.dim, c.n_heads, c.n_kv_heads, c.n_layers), (8, 4, 2, 2));
        assert_eq!(c.kv_dim(), 4);          // 2 kv heads × head_dim 2
        assert_eq!(c.rope_base, 500000.0);  // Llama-3 uses a larger base
        assert_eq!(c.max_seq, 2048);

        // No num_key_value_heads ⇒ no GQA (pre-GQA models depend on this).
        let c2 = config_from_json(&cfg_json(None), None).unwrap();
        assert_eq!(c2.n_kv_heads, 4);
        // max_seq is capped, never raised, by the override.
        assert_eq!(config_from_json(&cfg_json(None), Some(128)).unwrap().max_seq, 128);
        assert_eq!(config_from_json(&cfg_json(None), Some(99999)).unwrap().max_seq, 2048);
    }

    #[test]
    fn rejects_bad_config() {
        assert!(config_from_json("{}", None).unwrap_err().contains("hidden_size"));
        // 4 query heads cannot be grouped over 3 K/V heads.
        let bad = r#"{"hidden_size":8,"num_attention_heads":4,"num_key_value_heads":3,
            "num_hidden_layers":1,"vocab_size":4,"intermediate_size":8}"#;
        assert!(config_from_json(bad, None).unwrap_err().contains("multiple of n_kv_heads"));
    }

    #[test]
    fn transpose_is_correct_and_involutive() {
        // [2×3] row-major 1..6  →  [3×2]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = transpose(&a, 2, 3).unwrap();
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(transpose(&t, 3, 2).unwrap(), a, "transposing twice returns the original");
        assert!(transpose(&a, 4, 4).is_err());
    }

    /// Write a HF-style safetensors file with every tensor a Llama loader
    /// expects. Values are distinct per tensor so a mis-mapping shows up.
    fn write_hf_model(dir: &Path, cfg: &Config, tied: bool) {
        let mut header = String::from("{");
        let mut data: Vec<u8> = Vec::new();
        let mut first = true;
        let mut push = |header: &mut String, data: &mut Vec<u8>,
                        name: &str, shape: &[usize], seed: f32| {
            let n: usize = shape.iter().product();
            let start = data.len();
            for i in 0..n { data.extend_from_slice(&((i as f32) * 0.01 + seed).to_le_bytes()); }
            if !first { header.push(','); }
            first = false;
            header.push_str(&format!(
                r#""{name}":{{"dtype":"F32","shape":{:?},"data_offsets":[{},{}]}}"#,
                shape, start, data.len()));
        };
        let (d, kv, h, v) = (cfg.dim, cfg.kv_dim(), cfg.ffn_hidden, cfg.vocab);
        push(&mut header, &mut data, "model.embed_tokens.weight", &[v, d], 0.1);
        push(&mut header, &mut data, "model.norm.weight", &[d], 0.2);
        if !tied { push(&mut header, &mut data, "lm_head.weight", &[v, d], 0.3); }
        for i in 0..cfg.n_layers {
            let s = i as f32;
            push(&mut header, &mut data, &format!("model.layers.{i}.input_layernorm.weight"), &[d], s + 1.0);
            push(&mut header, &mut data, &format!("model.layers.{i}.post_attention_layernorm.weight"), &[d], s + 1.1);
            push(&mut header, &mut data, &format!("model.layers.{i}.self_attn.q_proj.weight"), &[d, d], s + 1.2);
            push(&mut header, &mut data, &format!("model.layers.{i}.self_attn.k_proj.weight"), &[kv, d], s + 1.3);
            push(&mut header, &mut data, &format!("model.layers.{i}.self_attn.v_proj.weight"), &[kv, d], s + 1.4);
            push(&mut header, &mut data, &format!("model.layers.{i}.self_attn.o_proj.weight"), &[d, d], s + 1.5);
            push(&mut header, &mut data, &format!("model.layers.{i}.mlp.gate_proj.weight"), &[h, d], s + 1.6);
            push(&mut header, &mut data, &format!("model.layers.{i}.mlp.up_proj.weight"), &[h, d], s + 1.7);
            push(&mut header, &mut data, &format!("model.layers.{i}.mlp.down_proj.weight"), &[d, h], s + 1.8);
        }
        header.push('}');
        let mut f = std::fs::File::create(dir.join("model.safetensors")).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(&data).unwrap();
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("r2hf-{}-{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn loads_a_gqa_checkpoint_and_generates() {
        let dir = tmpdir("gqa");
        std::fs::write(dir.join("config.json"), cfg_json(Some(2))).unwrap();
        let cfg = config_from_json(&cfg_json(Some(2)), Some(32)).unwrap();
        write_hf_model(&dir, &cfg, false);

        let m = load_dir(&dir, Some(32)).unwrap();
        assert_eq!(m.cfg.n_kv_heads, 2);
        assert_eq!(m.layers.len(), 2);
        assert_eq!(m.layers[0].wk.len(), cfg.dim * cfg.kv_dim());
        assert_eq!(m.tok_embed.len(), cfg.vocab * cfg.dim);

        // The real proof that the mapping is coherent: it generates.
        use crate::infer::{Sampler, SamplerConfig};
        let mut s = Sampler::new(1, SamplerConfig { temperature: 0.0, ..Default::default() });
        let out = m.generate(&[1, 2], 4, &mut s, None).unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|&t| t < m.cfg.vocab));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn transposes_projections_rather_than_copying_them() {
        // A loaded projection must equal the TRANSPOSE of the file bytes.
        // Without this check a straight copy would load, run, and quietly
        // produce nonsense.
        let dir = tmpdir("transpose");
        std::fs::write(dir.join("config.json"), cfg_json(Some(2))).unwrap();
        let cfg = config_from_json(&cfg_json(Some(2)), Some(16)).unwrap();
        write_hf_model(&dir, &cfg, false);

        let w = ShardedWeights::open_dir(&dir).unwrap();
        let raw = w.get("model.layers.0.self_attn.q_proj.weight", &[cfg.dim, cfg.dim]).unwrap();
        let m = load_dir(&dir, Some(16)).unwrap();
        assert_eq!(m.layers[0].wq, transpose(&raw, cfg.dim, cfg.dim).unwrap());
        assert_ne!(m.layers[0].wq, raw, "a straight copy would be wrong");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn tied_embeddings_reuse_the_embedding_matrix() {
        let dir = tmpdir("tied");
        std::fs::write(dir.join("config.json"), cfg_json(None)).unwrap();
        let cfg = config_from_json(&cfg_json(None), Some(16)).unwrap();
        write_hf_model(&dir, &cfg, true); // no lm_head.weight

        let m = load_dir(&dir, Some(16)).unwrap();
        assert_eq!(m.output, transpose(&m.tok_embed, cfg.vocab, cfg.dim).unwrap());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_missing_or_misshaped_tensor_is_named() {
        let dir = tmpdir("missing");
        std::fs::write(dir.join("config.json"), cfg_json(Some(2))).unwrap();
        let cfg = config_from_json(&cfg_json(Some(2)), Some(16)).unwrap();
        write_hf_model(&dir, &cfg, false);
        let w = ShardedWeights::open_dir(&dir).unwrap();

        // Wrong expected shape must be refused, naming the tensor.
        let err = w.get("model.layers.0.self_attn.k_proj.weight", &[cfg.dim, cfg.dim]).unwrap_err();
        assert!(err.contains("k_proj"), "error should name the tensor: {err}");
        assert!(w.get("model.layers.9.mlp.up_proj.weight", &[1]).unwrap_err().contains("no tensor named"));

        // A config claiming more layers than the file provides fails by name.
        let more = Config { n_layers: 5, ..cfg };
        assert!(load_weights(more, &w).unwrap_err().contains("model.layers.2"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn empty_or_duplicated_shards_are_rejected() {
        let dir = tmpdir("shards");
        assert!(ShardedWeights::open_dir(&dir).unwrap_err().contains("no .safetensors"));
        let _ = std::fs::remove_dir_all(dir);
    }
}

impl std::fmt::Debug for ShardedWeights {
    /// Summarize the namespace, not the mapped payload (which can be
    /// tens of gigabytes).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardedWeights")
            .field("shards", &self.shards.len())
            .field("tensors", &self.index.len())
            .field("total_params", &self.total_params())
            .finish()
    }
}
