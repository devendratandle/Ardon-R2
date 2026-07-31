//! Does fusing the batch let the Oracle route training onto the iGPU?
use r2_tensor::model::Config;
use r2_tensor::tokenizer::Tokenizer;
use r2_train::llm::Trainer;

fn run(label: &str, gpu: bool, cfg: Config, steps: usize, bn: usize, seq: usize) {
    r2_oracle::set_gpu_enabled(gpu);
    r2_gpu::set_gpu_enabled(gpu);
    let tok = Tokenizer::byte_level();
    let corpus = "ardon r2 is a statistical runtime built in pure rust. ";
    let ids: Vec<usize> = tok.encode(corpus).unwrap().iter().map(|&i| i as usize).collect();
    let mut batch = Vec::new();
    for s in 0..bn.min(ids.len().saturating_sub(seq + 1)) {
        batch.push((ids[s..s+seq].to_vec(), ids[s+1..s+seq+1].to_vec()));
    }
    let mut tr = Trainer::new(cfg, 0.003, 42).unwrap();
    let t0 = std::time::Instant::now();
    let mut last = 0.0;
    for _ in 0..steps { last = tr.train_step(&batch).unwrap(); }
    let dt = t0.elapsed().as_secs_f64();
    let tokens = (steps * batch.len() * seq) as f64;
    let gf = 6.0 * cfg.n_params() as f64 * tokens / dt / 1e9;
    // Report the widest matmul the fused pass performs, and the Oracle's call.
    let rows = batch.len() * seq;
    let work = rows * cfg.dim * cfg.ffn_hidden;
    let route = r2_oracle::dispatch(r2_oracle::Op::TensorMatMul,
                                    r2_oracle::Shape::nmk(rows, cfg.ffn_hidden, cfg.dim));
    println!("{:<12} gpu={:<5} {:>7.2}s {:>7.2} GFLOP/s  loss {:.3}  widest matmul work {:>10} -> {:?}",
             label, gpu, dt, gf, last, work, route);
}

fn main() {
    let v = Tokenizer::byte_level().vocab_size();
    let c10 = Config { dim: 384, n_heads: 6, n_kv_heads: 2, n_layers: 6, vocab: v,
                       ffn_hidden: 1024, max_seq: 64, rope_base: 10000.0, eps: 1e-5 };
    println!("adapter: {}\n", r2_gpu::adapter_info());
    run("10M-wide", false, c10, 4, 16, 32);
    run("10M-wide", true,  c10, 4, 16, 32);
}
