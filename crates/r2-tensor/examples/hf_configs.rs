//! Parse the config.json of real published models and check the derived
//! parameter count against the published figure.
use r2_tensor::hf::config_from_json;

fn main() {
    // Verbatim shapes from the published configs.
    let cases: &[(&str, &str, f64)] = &[
        ("Llama-2-7B", r#"{"hidden_size":4096,"num_attention_heads":32,"num_hidden_layers":32,
          "vocab_size":32000,"intermediate_size":11008,"max_position_embeddings":4096,
          "rms_norm_eps":1e-5}"#, 6.74),
        ("Llama-3-8B (GQA)", r#"{"hidden_size":4096,"num_attention_heads":32,
          "num_key_value_heads":8,"num_hidden_layers":32,"vocab_size":128256,
          "intermediate_size":14336,"max_position_embeddings":8192,"rope_theta":500000.0,
          "rms_norm_eps":1e-5}"#, 8.03),
        ("Mistral-7B (GQA)", r#"{"hidden_size":4096,"num_attention_heads":32,
          "num_key_value_heads":8,"num_hidden_layers":32,"vocab_size":32000,
          "intermediate_size":14336,"max_position_embeddings":32768,"rope_theta":10000.0,
          "rms_norm_eps":1e-5}"#, 7.24),
        ("Llama-2-70B (GQA)", r#"{"hidden_size":8192,"num_attention_heads":64,
          "num_key_value_heads":8,"num_hidden_layers":80,"vocab_size":32000,
          "intermediate_size":28672,"max_position_embeddings":4096,"rms_norm_eps":1e-5}"#, 68.98),
    ];
    println!("{:<20} {:>8} {:>8} {:>10} {:>10} {:>9}", "model", "q-heads", "kv", "params", "published", "KV/token");
    for (name, src, published) in cases {
        let c = config_from_json(src, Some(2048)).expect(name);
        let p = c.n_params() as f64 / 1e9;
        // KV cache bytes per token (fp16): 2 (K and V) * layers * kv_dim * 2 bytes
        let kv_per_tok = 2.0 * c.n_layers as f64 * c.kv_dim() as f64 * 2.0 / 1024.0;
        println!("{:<20} {:>8} {:>8} {:>9.2}B {:>9.2}B {:>7.0} KB",
                 name, c.n_heads, c.n_kv_heads, p, published, kv_per_tok);
    }
    println!("\n(KV/token is why GQA exists: it is what a long context costs in RAM.)");
}
