//! Measure matmul throughput directly, so optimization targets a real
//! number rather than a guess.
use r2_tensor::ops::matmul;
fn main() {
    println!("{:>6} {:>6} {:>6} {:>10} {:>10}", "m", "k", "n", "ms", "GFLOP/s");
    for &(m, k, n) in &[(256,256,256), (512,512,512), (1024,1024,1024),
                        (16,768,768), (16,768,3072), (64,768,768)] {
        let a: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.002).cos()).collect();
        // Warm up, then time a few reps.
        let _ = matmul(&a, &b, m, k, n);
        let reps = if m*k*n > 100_000_000 { 2 } else { 10 };
        let t0 = std::time::Instant::now();
        for _ in 0..reps { std::hint::black_box(matmul(&a, &b, m, k, n)); }
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        let gf = 2.0 * m as f64 * k as f64 * n as f64 / dt / 1e9;
        println!("{:>6} {:>6} {:>6} {:>10.2} {:>10.1}", m, k, n, dt*1e3, gf);
    }
    println!("\n(2*m*k*n flops per matmul; last three rows are the SHAPES TRAINING USES)");
}
