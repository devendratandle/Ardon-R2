//! GPU attention (forward, backward) against the tape's CPU kernels at the
//! two training shapes: small (32 x 64 tokens, 4/2 heads) and medium
//! (8 x 256, 12/4 heads), hd 64. Operands resident, `reps` launches per
//! timing closed by one fence; median of 5. Milliseconds per call.
//!
//!     cargo run --release -p r2-gpu --features gpu --example attn_bench

use r2_autograd::Tape;
use r2_gpu::attention::{backward, forward, Shape};
use r2_gpu::device::{sync, Tensor};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    println!("adapter: {}", r2_gpu::adapter_info());
    println!("{:<26} {:>9} {:>9} {:>9} {:>9}", "shape", "GPU fwd", "CPU fwd", "GPU bwd", "CPU bwd");
    println!("{}", "-".repeat(66));
    for &(nseq, seq, nh, nkv, hd, reps) in &[(32usize, 64usize, 4usize, 2usize, 64usize, 20usize), (8, 256, 12, 4, 64, 10)] {
        let rows = nseq * seq;
        let mk = |n: usize, ph: f32| -> Vec<f32> { (0..n).map(|i| ((i as f32) * 0.0013 + ph).sin()).collect() };
        let (qv, kv, vv, gv) = (mk(rows * nh * hd, 0.0), mk(rows * nkv * hd, 1.0), mk(rows * nkv * hd, 2.0), mk(rows * nh * hd, 3.0));
        let scale = 1.0 / (hd as f32).sqrt();
        let sh = Shape { nseq, seq, nh, nkv, hd, scale };

        let (Some(tq), Some(tk), Some(tv), Some(tg)) = (Tensor::upload(&qv), Tensor::upload(&kv), Tensor::upload(&vv), Tensor::upload(&gv)) else { println!("no GPU"); return; };
        let to = Tensor::zeros(rows * nh * hd).unwrap();
        let tl = Tensor::zeros(rows * nh).unwrap();
        let td = Tensor::zeros(rows * nh).unwrap();
        let (dq, dk, dv) = (Tensor::zeros(rows * nh * hd).unwrap(), Tensor::zeros(rows * nkv * hd).unwrap(), Tensor::zeros(rows * nkv * hd).unwrap());
        forward(&tq, &tk, &tv, &to, &tl, &sh); backward(&tq, &tk, &tv, &to, &tl, &tg, &td, &dq, &dk, &dv, &sh); sync();
        let gpu_f = median((0..5).map(|_| {
            let s = std::time::Instant::now();
            for _ in 0..reps { forward(&tq, &tk, &tv, &to, &tl, &sh); }
            sync(); s.elapsed().as_secs_f64() * 1e3 / reps as f64
        }).collect());
        let gpu_b = median((0..5).map(|_| {
            let s = std::time::Instant::now();
            for _ in 0..reps { backward(&tq, &tk, &tv, &to, &tl, &tg, &td, &dq, &dk, &dv, &sh); }
            sync(); s.elapsed().as_secs_f64() * 1e3 / reps as f64
        }).collect());

        // CPU: the tape's attention op; backward reported as (fwd+bwd) - fwd,
        // the same subtraction lmo15_attention makes.
        let cpu_f = median((0..5).map(|_| {
            let mut tp = Tape::new();
            let (a, b, c) = (tp.leaf(qv.clone(), true), tp.leaf(kv.clone(), true), tp.leaf(vv.clone(), true));
            let s = std::time::Instant::now();
            for _ in 0..reps { std::hint::black_box(tp.attention(a, b, c, nseq, seq, nh, nkv, hd, scale)); }
            s.elapsed().as_secs_f64() * 1e3 / reps as f64
        }).collect());
        let cpu_fb = median((0..5).map(|_| {
            let s = std::time::Instant::now();
            for _ in 0..reps {
                let mut tp = Tape::new();
                let (a, b, c) = (tp.leaf(qv.clone(), true), tp.leaf(kv.clone(), true), tp.leaf(vv.clone(), true));
                let o = tp.attention(a, b, c, nseq, seq, nh, nkv, hd, scale);
                tp.backward_from(o, &gv);
                std::hint::black_box(tp.grad(a).len());
            }
            s.elapsed().as_secs_f64() * 1e3 / reps as f64
        }).collect());
        println!("{:<26} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
                 format!("{nseq}x{seq} {nh}/{nkv} hd{hd}"), gpu_f, cpu_f, gpu_b, (cpu_fb - cpu_f).max(0.0));
    }
}
