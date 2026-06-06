//! GEMM benchmark — serial vs Oracle-parallel. Run release:
//!   cargo run -p r2-linalg --release --example bench_gemm
//! A/B the serial path with:
//!   R2_FORCE_SERIAL=1 cargo run -p r2-linalg --release --example bench_gemm
//! Optional size: `--example bench_gemm -- 2048`

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let (m, k) = (n, n);
    let a: Vec<f64> = (0..m * k).map(|i| (i % 7) as f64 * 0.5 - 1.0).collect();
    let b: Vec<f64> = (0..k * n).map(|i| (i % 5) as f64 * 0.3 - 0.5).collect();
    let mut c = vec![0.0; m * n];

    // Warm-up (allocations, thread pool spin-up).
    r2_linalg::dgemm_dispatch(m, n, k, 1.0, &a, &b, 0.0, &mut c).unwrap();

    let reps = 5;
    let t = std::time::Instant::now();
    for _ in 0..reps {
        r2_linalg::dgemm_dispatch(m, n, k, 1.0, &a, &b, 0.0, &mut c).unwrap();
    }
    let secs = t.elapsed().as_secs_f64() / reps as f64;
    let gflops = 2.0 * (m as f64) * (n as f64) * (k as f64) / secs / 1e9;
    let mode = if std::env::var_os("R2_FORCE_SERIAL").is_some() { "serial" } else { "auto" };
    println!(
        "dgemm {}x{}x{} [{}]: {:.4}s/rep  {:.2} GFLOP/s",
        m, n, k, mode, secs, gflops
    );
}
