//! The three GEMM cases a training step runs, one shape, kernel to kernel.
//!
//! For a layer `C = A·B` with A `t x k` (activations) and B `k x n` (the
//! weight), the backward is two more calls to the SAME routine with
//! transpose flags — exactly what PyTorch's `MmBackward0` issues as
//! `grad.mm(B.t())` and `A.t().mm(grad)`, both stride-only views into
//! MKL's `sgemm` with `transa`/`transb`:
//!
//! ```text
//!   forward  NN   C (t x n)      = A (t x k)  · B (k x n)
//!   grad_A   NT   dA (t x k)     = g (t x n)  · Bᵀ
//!   grad_B   TN   dB (k x n)     = Aᵀ         · g (t x n)
//! ```
//!
//! This prints microseconds per call for the three, at one shape given on
//! the command line, so `benchmarks/llm/gemm_cases.py` can run PyTorch's
//! three calls and this binary back to back for every shape of a sweep —
//! interleaved per shape, which a two-process "all of R2, then all of
//! torch" sweep is not. Median of 7 timed batches after a warm batch,
//! each batch sized to ~0.25 s.
//!
//!     cargo run --release -p r2-linalg --example gemm_cases -- 2048 768 2304
//!     python benchmarks/llm/gemm_cases.py

use r2_linalg::gemm::{sgemm_into, Trans};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_us(flops: f64, mut f: impl FnMut()) -> f64 {
    f();
    // batches of ~0.25 s at a guessed 100 GFLOP/s, at least 2 calls
    let reps = ((0.25 * 100e9) / flops).ceil().clamp(2.0, 200.0) as usize;
    for _ in 0..reps.min(3) { f(); }
    median((0..7).map(|_| {
        let s = std::time::Instant::now();
        for _ in 0..reps { f(); }
        s.elapsed().as_secs_f64() / reps as f64 * 1e6
    }).collect())
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let (t, k, n) = match args[..] {
        [t, k, n] => (t, k, n),
        _ => { eprintln!("usage: gemm_cases <tokens> <k> <n>"); std::process::exit(2); }
    };
    let fill = |len: usize, ph: f32| -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * 0.0007 + ph).sin()).collect()
    };
    let a = fill(t * k, 0.0);          // activations, t x k
    let b = fill(k * n, 1.0);          // weight, k x n
    let g = fill(t * n, 2.0);          // upstream gradient, t x n
    let mut c = vec![0.0f32; t * n];
    let mut da = vec![0.0f32; t * k];
    let mut db = vec![0.0f32; k * n];
    let flops = 2.0 * t as f64 * k as f64 * n as f64;

    let nn = time_us(flops, || sgemm_into(&a, Trans::No, &b, Trans::No, t, k, n, &mut c, true));
    let nt = time_us(flops, || sgemm_into(&g, Trans::No, &b, Trans::Yes, t, n, k, &mut da, true));
    let tn = time_us(flops, || sgemm_into(&a, Trans::Yes, &g, Trans::No, k, t, n, &mut db, true));
    std::hint::black_box((&c, &da, &db));
    println!("{nn:.1} {nt:.1} {tn:.1}");
}
