use r2_tensor::model::Config;
use r2_tensor::tokenizer::Tokenizer;
use r2_train::llm::Trainer;
fn main() {
    let v = Tokenizer::byte_level().vocab_size();
    let cfg = Config { dim: 384, n_heads: 6, n_kv_heads: 2, n_layers: 6, vocab: v,
                       ffn_hidden: 1024, max_seq: 64, rope_base: 10000.0, eps: 1e-5 };
    let tok = Tokenizer::byte_level();
    let ids: Vec<usize> = tok.encode("ardon r2 is a statistical runtime written in pure rust. ")
        .unwrap().iter().map(|&i| i as usize).collect();
    let (seq, bn) = (16usize, 6usize);
    let mut batch = Vec::new();
    for s in 0..bn { batch.push((ids[s..s+seq].to_vec(), ids[s+1..s+seq+1].to_vec())); }
    let tr = Trainer::new(cfg, 0.003, 1).unwrap();
    let (nodes, elems) = tr.tape_stats(&batch);
    println!("tape nodes per step : {}", nodes);
    println!("buffers allocated   : {} (value + gradient per node)", nodes * 2);
    println!("elements held       : {} ({:.1} MB)", elems, elems as f64 * 8.0 / 1e6);
    println!("avg elements/node   : {:.0}", elems as f64 / nodes as f64);
    // How many are the tiny attention ops?
    let per_layer_attn = bn * cfg.n_heads * 8;
    println!("\nof which attention  : ~{} nodes ({:.0}% of the tape)",
             per_layer_attn * cfg.n_layers,
             (per_layer_attn * cfg.n_layers) as f64 / nodes as f64 * 100.0);
    println!("  = {} sequences x {} heads x ~8 ops x {} layers",
             bn, cfg.n_heads, cfg.n_layers);
}
