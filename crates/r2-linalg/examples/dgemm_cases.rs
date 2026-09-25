//! `level3::dgemm` at one column-major shape, for `benchmarks/dgemm_cases.py`.
//!
//! Prints microseconds per call for `C = A·B` (m x k times k x n, f64,
//! column-major — the BLAS entry point `%*%` and the LAPACK routines use).
//! Median of 7 timed batches after a warm batch, each batch ~0.25 s.
//!
//!     cargo run --release -p r2-linalg --example dgemm_cases -- 1024 1024 1024
//!     python benchmarks/dgemm_cases.py

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let (m, k, n) = match args[..] {
        [m, k, n] => (m, k, n),
        _ => { eprintln!("usage: dgemm_cases <m> <k> <n>"); std::process::exit(2); }
    };
    let fill = |len: usize, ph: f64| -> Vec<f64> {
        (0..len).map(|i| ((i as f64) * 0.0007 + ph).sin()).collect()
    };
    let a = fill(m * k, 0.0);
    let b = fill(k * n, 1.0);
    let mut c = vec![0.0f64; m * n];
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let mut f = || r2_linalg::dgemm(m, n, k, 1.0, &a, &b, 0.0, &mut c).unwrap();
    f();
    // batches of ~0.25 s at a guessed 50 GFLOP/s, at least 2 calls
    let reps = ((0.25 * 50e9) / flops).ceil().clamp(2.0, 2000.0) as usize;
    for _ in 0..reps.min(3) { f(); }
    let us = median((0..7).map(|_| {
        let s = std::time::Instant::now();
        for _ in 0..reps { f(); }
        s.elapsed().as_secs_f64() / reps as f64 * 1e6
    }).collect());
    std::hint::black_box(&c);
    println!("{us:.1}");
}
