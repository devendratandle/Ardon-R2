//! crossprod (AᵀA) benchmark — serial vs Oracle-parallel.
//! cargo run -p r2-linalg --release --example bench_crossprod -- <rows> <cols>
fn main() {
    let m: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let a: Vec<f64> = (0..m * n).map(|i| (i % 7) as f64 * 0.5 - 1.0).collect();
    let mut c = vec![0.0; n * n];
    r2_linalg::dcrossprod(m, n, &a, &mut c).unwrap(); // warm
    let reps = 5;
    let t = std::time::Instant::now();
    for _ in 0..reps { r2_linalg::dcrossprod(m, n, &a, &mut c).unwrap(); }
    let secs = t.elapsed().as_secs_f64() / reps as f64;
    let gflops = (n as f64) * (n as f64) * (m as f64) / secs / 1e9; // ~1 mul+add per (i,j,p), triangular≈half
    let mode = if std::env::var_os("R2_FORCE_SERIAL").is_some() { "serial" } else { "auto" };
    println!("crossprod {}x{} (AᵀA {}x{}) [{}]: {:.4}s/rep  {:.2} G(n²m)/s", m, n, n, n, mode, secs, gflops);
}
