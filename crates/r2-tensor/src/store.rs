//! The R2 model format — save, share, load, serve.
//!
//! An R2 model on disk is a directory:
//!
//! ```text
//!   my-model/
//!     model.json          the Config — plain text, readable, diffable
//!     model.safetensors   the weights, memory-mappable
//! ```
//!
//! Two decisions worth stating, because they are what make a model
//! *portable* rather than merely persisted:
//!
//! * **The config is plain JSON**, so a model's shape can be inspected,
//!   diffed and version-controlled without loading gigabytes of weights —
//!   and a corrupted or mismatched model is diagnosable by reading a file.
//! * **The weights are safetensors**, an open container that executes no
//!   code on load and is memory-mappable, so a large model is served
//!   without being read into RAM. Any tool that reads safetensors can read
//!   an R2 model: "download a model and run it" carries no lock-in, and no
//!   dependency on any vendor's format or licence.
//!
//! Tensor names are R2's own (`layer.0.wq`), describing this crate's
//! layout directly — no foreign naming convention to translate, and no
//! transpose on load, which removes an entire class of silent-corruption
//! bug that only appears as degraded output.

use std::path::Path;

use crate::json::Json;
use crate::model::{Config, Layer, Model};
use crate::safetensors::{self, SafeTensors};

const CONFIG_FILE: &str = "model.json";
const WEIGHTS_FILE: &str = "model.safetensors";

/// Serialize a [`Config`] as JSON.
pub fn config_to_json(c: &Config) -> String {
    format!(
        "{{\n  \"format\": \"r2-model-v1\",\n  \"dim\": {},\n  \"n_heads\": {},\n  \
         \"n_kv_heads\": {},\n  \"n_layers\": {},\n  \"vocab\": {},\n  \
         \"ffn_hidden\": {},\n  \"max_seq\": {},\n  \"rope_base\": {},\n  \
         \"eps\": {},\n  \"n_params\": {}\n}}\n",
        c.dim, c.n_heads, c.n_kv_heads, c.n_layers, c.vocab,
        c.ffn_hidden, c.max_seq, c.rope_base, c.eps, c.n_params())
}

/// Parse a [`Config`] from R2 model JSON. Validated shape-only, so a
/// 70B config can be inspected without allocating anything.
pub fn config_from_json(src: &str) -> Result<Config, String> {
    let j = Json::parse(src).map_err(|e| format!("model.json: {}", e))?;
    let need = |k: &str| -> Result<usize, String> {
        j.get(k).and_then(|v| v.as_usize())
            .ok_or_else(|| format!("model.json: missing or non-integer '{}'", k))
    };
    let n_heads = need("n_heads")?;
    let cfg = Config {
        dim: need("dim")?,
        n_heads,
        // Absent means no grouped-query attention.
        n_kv_heads: j.get("n_kv_heads").and_then(|v| v.as_usize()).unwrap_or(n_heads),
        n_layers: need("n_layers")?,
        vocab: need("vocab")?,
        ffn_hidden: need("ffn_hidden")?,
        max_seq: need("max_seq")?,
        rope_base: j.get("rope_base").and_then(|v| v.as_f64()).unwrap_or(10000.0) as f32,
        eps: j.get("eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32,
    };
    cfg.validate().map_err(|e| format!("model.json: {}", e))?;
    Ok(cfg)
}

/// Tensor name for a layer weight — one place, so writer and reader can
/// never disagree.
fn layer_key(i: usize, part: &str) -> String { format!("layer.{i}.{part}") }

impl Model {
    /// Write this model to `dir` as `model.json` + `model.safetensors`.
    pub fn save_dir(&self, dir: impl AsRef<Path>) -> Result<(), String> {
        self.validate()?;
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("model: cannot create {}: {}", dir.display(), e))?;

        let c = &self.cfg;
        let (d, kv, h) = (c.dim, c.kv_dim(), c.ffn_hidden);
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
            ("tok_embed".into(), vec![c.vocab, d], self.tok_embed.clone()),
            ("final_norm".into(), vec![d], self.final_norm.clone()),
            ("output".into(), vec![d, c.vocab], self.output.clone()),
        ];
        for (i, l) in self.layers.iter().enumerate() {
            tensors.push((layer_key(i, "attn_norm"), vec![d], l.attn_norm.clone()));
            tensors.push((layer_key(i, "ffn_norm"), vec![d], l.ffn_norm.clone()));
            tensors.push((layer_key(i, "wq"), vec![d, d], l.wq.clone()));
            tensors.push((layer_key(i, "wk"), vec![d, kv], l.wk.clone()));
            tensors.push((layer_key(i, "wv"), vec![d, kv], l.wv.clone()));
            tensors.push((layer_key(i, "wo"), vec![d, d], l.wo.clone()));
            tensors.push((layer_key(i, "w1"), vec![d, h], l.w1.clone()));
            tensors.push((layer_key(i, "w2"), vec![h, d], l.w2.clone()));
            tensors.push((layer_key(i, "w3"), vec![d, h], l.w3.clone()));
        }

        safetensors::save(dir.join(WEIGHTS_FILE), &tensors)?;
        // Config last: a directory with a config but no weights would look
        // loadable, so write the weights first and let the config commit
        // the model.
        std::fs::write(dir.join(CONFIG_FILE), config_to_json(c))
            .map_err(|e| format!("model: cannot write {}: {}", CONFIG_FILE, e))
    }

    /// Load a model saved by [`save_dir`].
    ///
    /// Every tensor is fetched by exact name with its exact expected
    /// shape, so a truncated, edited or mismatched file is an error that
    /// names the offending tensor rather than silently degraded output.
    pub fn load_dir(dir: impl AsRef<Path>) -> Result<Model, String> {
        let dir = dir.as_ref();
        let cfg_src = std::fs::read_to_string(dir.join(CONFIG_FILE))
            .map_err(|e| format!("model: cannot read {}/{}: {}", dir.display(), CONFIG_FILE, e))?;
        let cfg = config_from_json(&cfg_src)?;
        let st = SafeTensors::open(dir.join(WEIGHTS_FILE))?;

        let (d, kv, h) = (cfg.dim, cfg.kv_dim(), cfg.ffn_hidden);
        let mut m = Model::zeros(cfg);
        m.tok_embed = st.tensor_f32_shaped("tok_embed", &[cfg.vocab, d])?;
        m.final_norm = st.tensor_f32_shaped("final_norm", &[d])?;
        m.output = st.tensor_f32_shaped("output", &[d, cfg.vocab])?;
        for i in 0..cfg.n_layers {
            m.layers[i] = Layer {
                attn_norm: st.tensor_f32_shaped(&layer_key(i, "attn_norm"), &[d])?,
                ffn_norm:  st.tensor_f32_shaped(&layer_key(i, "ffn_norm"), &[d])?,
                wq: st.tensor_f32_shaped(&layer_key(i, "wq"), &[d, d])?,
                wk: st.tensor_f32_shaped(&layer_key(i, "wk"), &[d, kv])?,
                wv: st.tensor_f32_shaped(&layer_key(i, "wv"), &[d, kv])?,
                wo: st.tensor_f32_shaped(&layer_key(i, "wo"), &[d, d])?,
                w1: st.tensor_f32_shaped(&layer_key(i, "w1"), &[d, h])?,
                w2: st.tensor_f32_shaped(&layer_key(i, "w2"), &[h, d])?,
                w3: st.tensor_f32_shaped(&layer_key(i, "w3"), &[d, h])?,
            };
        }
        m.validate()?;
        Ok(m)
    }

    /// Read only the config of a saved model — the shape, parameter count
    /// and context length — WITHOUT touching the weights. This is what
    /// makes `r2 model info` instant on a 30 GB model.
    pub fn peek_config(dir: impl AsRef<Path>) -> Result<Config, String> {
        let p = dir.as_ref().join(CONFIG_FILE);
        let src = std::fs::read_to_string(&p)
            .map_err(|e| format!("model: cannot read {}: {}", p.display(), e))?;
        config_from_json(&src)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::{Sampler, SamplerConfig};
    use std::path::PathBuf;

    fn cfg() -> Config {
        Config { dim: 16, n_heads: 4, n_kv_heads: 2, n_layers: 3, vocab: 12,
                 ffn_hidden: 24, max_seq: 32, rope_base: 10000.0, eps: 1e-5 }
    }

    fn demo() -> Model {
        let c = cfg();
        let f = |n: usize, s: f32| (0..n).map(|i| ((i as f32 * 0.113 + s).sin()) * 0.35).collect::<Vec<f32>>();
        let mut m = Model::zeros(c);
        m.tok_embed = f(c.vocab * c.dim, 0.7);
        m.final_norm = vec![1.0; c.dim];
        m.output = f(c.dim * c.vocab, 1.9);
        for (i, l) in m.layers.iter_mut().enumerate() {
            let s = i as f32 * 3.1;
            l.attn_norm = vec![1.0; c.dim];
            l.ffn_norm = vec![1.0; c.dim];
            l.wq = f(c.dim * c.dim, s + 0.2);
            l.wk = f(c.dim * c.kv_dim(), s + 0.5);
            l.wv = f(c.dim * c.kv_dim(), s + 0.9);
            l.wo = f(c.dim * c.dim, s + 1.3);
            l.w1 = f(c.dim * c.ffn_hidden, s + 1.7);
            l.w2 = f(c.ffn_hidden * c.dim, s + 2.1);
            l.w3 = f(c.dim * c.ffn_hidden, s + 2.5);
        }
        m
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("r2model-{}-{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn a_saved_model_generates_identically_to_the_original() {
        // THE INVARIANT: persisting a model must not change what it does.
        // Logits are compared, not just bytes — that catches a shape or
        // ordering error a byte comparison could still pass.
        let dir = tmpdir("roundtrip");
        let a = demo();
        a.save_dir(&dir).unwrap();
        let b = Model::load_dir(&dir).unwrap();

        assert_eq!(b.cfg, a.cfg, "config must survive exactly");

        let tokens = [3usize, 7, 1, 0, 9];
        let mut ca = a.new_caches().unwrap();
        let mut cb = b.new_caches().unwrap();
        for (t, &tok) in tokens.iter().enumerate() {
            let la = a.forward_step(tok, t, &mut ca).unwrap();
            let lb = b.forward_step(tok, t, &mut cb).unwrap();
            assert_eq!(la, lb, "step {t}: a loaded model must produce identical logits");
        }

        // And the full serving path agrees too.
        let mut s1 = Sampler::new(4, SamplerConfig { temperature: 0.7, top_k: 5, ..Default::default() });
        let mut s2 = Sampler::new(4, SamplerConfig { temperature: 0.7, top_k: 5, ..Default::default() });
        assert_eq!(a.generate(&[2, 5], 8, &mut s1, None).unwrap(),
                   b.generate(&[2, 5], 8, &mut s2, None).unwrap());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn every_weight_survives_bit_exactly() {
        let dir = tmpdir("exact");
        let a = demo();
        a.save_dir(&dir).unwrap();
        let b = Model::load_dir(&dir).unwrap();
        assert_eq!(b.tok_embed, a.tok_embed);
        assert_eq!(b.output, a.output);
        assert_eq!(b.final_norm, a.final_norm);
        for (x, y) in a.layers.iter().zip(&b.layers) {
            assert_eq!((&x.wq, &x.wk, &x.wv, &x.wo), (&y.wq, &y.wk, &y.wv, &y.wo));
            assert_eq!((&x.w1, &x.w2, &x.w3), (&y.w1, &y.w2, &y.w3));
            assert_eq!((&x.attn_norm, &x.ffn_norm), (&y.attn_norm, &y.ffn_norm));
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn config_round_trips_through_json() {
        let c = cfg();
        let back = config_from_json(&config_to_json(&c)).unwrap();
        assert_eq!(back, c);
        // The config is human-readable and states the model's size.
        let js = config_to_json(&c);
        assert!(js.contains("\"n_kv_heads\": 2") && js.contains("\"n_params\""));
    }

    #[test]
    fn peek_reads_the_shape_without_the_weights() {
        // Instant on a 30 GB model: this must not open the weights file.
        let dir = tmpdir("peek");
        let a = demo();
        a.save_dir(&dir).unwrap();
        std::fs::remove_file(dir.join(WEIGHTS_FILE)).unwrap();
        let c = Model::peek_config(&dir).unwrap();
        assert_eq!(c, a.cfg);
        // ...but a real load then fails, clearly.
        assert!(Model::load_dir(&dir).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_corrupted_or_mismatched_model_errors_by_name() {
        let dir = tmpdir("corrupt");
        let a = demo();
        a.save_dir(&dir).unwrap();

        // Config claiming more layers than the weights provide.
        let mut bad = config_to_json(&a.cfg).replace("\"n_layers\": 3", "\"n_layers\": 5");
        bad = bad.replace("\"n_params\"", "\"ignored\"");
        std::fs::write(dir.join(CONFIG_FILE), bad).unwrap();
        let err = Model::load_dir(&dir).unwrap_err();
        assert!(err.contains("layer.3"), "error should name the missing tensor: {err}");

        // Nonsense config is refused before any weight is read.
        std::fs::write(dir.join(CONFIG_FILE), "{}").unwrap();
        assert!(Model::load_dir(&dir).unwrap_err().contains("missing or non-integer"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_model_directory_is_a_clear_error() {
        let err = Model::load_dir(tmpdir("nope")).unwrap_err();
        assert!(err.contains("cannot read") && err.contains("model.json"));
    }
}
