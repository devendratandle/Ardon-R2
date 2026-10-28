//! A fingerprint of one training step's numbers: the loss bits and an FNV
//! hash over the bits of every gradient, on the 7M benchmark model and a
//! fixed batch. Two builds that print the same line computed the same step
//! to the last bit — the check for a change meant to be bit-identical.
//!
//!     cargo run --release -p r2-train --example grad_fingerprint

use r2_tensor::model::Config;
use r2_train::llm::Trainer;

fn main() {
    let cfg = Config { dim: 256, n_heads: 4, n_kv_heads: 2, n_layers: 4, vocab: 8000,
                       ffn_hidden: 768, max_seq: 64, rope_base: 10000.0, eps: 1e-5 };
    let tr = Trainer::new(cfg, 3e-3, 42).expect("trainer");
    let mut s = 12345u64;
    let mut tok = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 33) as usize % 8000 };
    let batch: Vec<(Vec<usize>, Vec<usize>)> = (0..32).map(|_| {
        let ids: Vec<usize> = (0..65).map(|_| tok()).collect();
        (ids[..64].to_vec(), ids[1..].to_vec())
    }).collect();
    let (loss, _, grads) = tr.loss_logits_grads(&batch).expect("step");
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for g in grads.iter().flatten() { h = (h ^ g.to_bits() as u64).wrapping_mul(0x100_0000_01b3); }
    println!("loss {loss} ({:#010x})  grads {} values  fnv {h:#018x}", loss.to_bits(), grads.iter().map(|g| g.len()).sum::<usize>());
}
