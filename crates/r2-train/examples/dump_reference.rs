//! Dump a small model, its inputs, and everything one forward+backward
//! produces, so an INDEPENDENT implementation can check R2's maths.
//!
//! R2's own tests check R2 against itself — finite differences, its own
//! explicit forms. Those catch a coding slip but not a shared
//! misunderstanding: if the forward and the backward agree on a wrong
//! definition of RMSNorm, finite differences agree with both. Only an
//! implementation written from the published equations by someone else
//! can catch that, and only in float64, so that f32 rounding is not
//! mistaken for a defect.
//!
//! Writes raw little-endian f32 plus a JSON manifest. `accuracy_check.py`
//! rebuilds the same model in torch.float64 and compares the loss, the
//! logits, and EVERY parameter gradient block by block — block by block
//! because a single "max error" figure says something is wrong without
//! saying where.
//!
//!     cargo run --release -p r2-train --example dump_reference
//!     python benchmarks/llm/accuracy_check.py

use r2_tensor::model::Config;
use r2_train::llm::Trainer;

fn write_f32(name: &str, v: &[f32]) {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v { b.extend_from_slice(&x.to_le_bytes()); }
    std::fs::write(name, b).unwrap_or_else(|e| panic!("writing {name}: {e}"));
}

fn main() {
    // Small on purpose. A differential check wants every operator
    // exercised, not a long run: more layers would not find a defect this
    // cannot, and float64 in Python is slow.
    let cfg = Config {
        dim: 64, n_heads: 4, n_kv_heads: 2, n_layers: 2, vocab: 128,
        ffn_hidden: 192, max_seq: 16, rope_base: 10000.0, eps: 1e-5,
    };
    let (bn, seq) = (3usize, 8usize);
    let tr = Trainer::new(cfg, 3e-4, 7).expect("trainer");

    // Deterministic tokens — the check must be reproducible from the
    // manifest alone, with nothing sampled at run time.
    let batch: Vec<(Vec<usize>, Vec<usize>)> = (0..bn).map(|b| {
        let inp: Vec<usize> = (0..seq).map(|i| (b * 31 + i * 7 + 1) % cfg.vocab).collect();
        let tgt: Vec<usize> = (0..seq).map(|i| (b * 31 + i * 7 + 2) % cfg.vocab).collect();
        (inp, tgt)
    }).collect();

    let (loss, logits, grads) = tr.loss_logits_grads(&batch).expect("forward/backward");

    let names = tr.block_names();
    assert_eq!(names.len(), tr.params.len(), "block_names must cover every block");
    assert_eq!(grads.len(), tr.params.len(), "a gradient per parameter block");

    for (i, p) in tr.params.iter().enumerate() {
        write_f32(&format!("ref_param_{i}.bin"), p);
        write_f32(&format!("ref_grad_{i}.bin"), &grads[i]);
    }
    write_f32("ref_logits.bin", &logits);

    let mut j = String::from("{\n");
    j.push_str(&format!("  \"dim\": {}, \"n_heads\": {}, \"n_kv_heads\": {},\n",
                        cfg.dim, cfg.n_heads, cfg.n_kv_heads));
    j.push_str(&format!("  \"n_layers\": {}, \"vocab\": {}, \"ffn_hidden\": {},\n",
                        cfg.n_layers, cfg.vocab, cfg.ffn_hidden));
    j.push_str(&format!("  \"eps\": {}, \"rope_base\": {},\n", cfg.eps, cfg.rope_base));
    j.push_str(&format!("  \"batch\": {bn}, \"seq\": {seq},\n"));
    j.push_str(&format!("  \"loss\": {:.10},\n", loss));
    j.push_str("  \"tokens\": [");
    for (k, (inp, _)) in batch.iter().enumerate() {
        if k > 0 { j.push(','); }
        j.push_str(&format!("{inp:?}"));
    }
    j.push_str("],\n  \"targets\": [");
    for (k, (_, tgt)) in batch.iter().enumerate() {
        if k > 0 { j.push(','); }
        j.push_str(&format!("{tgt:?}"));
    }
    j.push_str("],\n  \"blocks\": [");
    for (i, n) in names.iter().enumerate() {
        if i > 0 { j.push(','); }
        j.push_str(&format!("\n    {{\"i\": {i}, \"name\": \"{n}\", \"len\": {}}}",
                            tr.params[i].len()));
    }
    j.push_str("\n  ]\n}\n");
    std::fs::write("reference_dump.json", j).expect("writing manifest");

    println!("wrote reference_dump.json");
    println!("  {} params in {} blocks, loss {loss:.8}",
             tr.n_params(), tr.params.len());
    println!("  logits {} = {bn} x {seq} x {}", logits.len(), cfg.vocab);
    println!("now run: python benchmarks/llm/accuracy_check.py");
}
