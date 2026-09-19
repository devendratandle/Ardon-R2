//! Is the 6-core ceiling the POWER ENVELOPE?
//!
//! `scaling_ceiling` measured 5.53x on a register chain; `bandwidth_sweep`
//! measured 2.9x on L1-resident work; `sgemm` and MKL both get 2.4-4.0x.
//! The chain in `scaling_ceiling` is a DEPENDENT FMA sequence — latency
//! bound, one FMA in flight, the core mostly idle, so it draws little
//! power and keeps its clock. A GEMM keeps both FMA ports full and draws
//! the most power an AVX2 core can. On a 15 W part the all-core clock
//! under that load is roughly half the single-core boost, and "6 cores"
//! is then worth about 3x one — for R2, for MKL, for anyone.
//!
//! This test is THROUGHPUT-bound FMA with no memory traffic at all: 12
//! independent 8-wide accumulators per thread (the micro-kernel's own
//! register budget), the same instruction mix the GEMM issues, and
//! nothing to load or store. If it scales ~3x, the ceiling is power and
//! no loop order, mesh or prefetch can raise it on this machine.
//!
//!     cargo run --release -p r2-tensor --example fma_power

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn burn(iters: u64, seed: f32) -> f32 {
    use std::arch::x86_64::*;
    let a = _mm256_set1_ps(seed);
    let b = _mm256_set1_ps(1.000001);
    let mut acc = [_mm256_set1_ps(0.0); 12];
    for (i, v) in acc.iter_mut().enumerate() { *v = _mm256_set1_ps(i as f32 * 1e-3); }
    for _ in 0..iters {
        for v in acc.iter_mut() { *v = _mm256_fmadd_ps(*v, b, a); }
    }
    let mut s = _mm256_set1_ps(0.0);
    for v in &acc { s = _mm256_add_ps(s, *v); }
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(out.as_mut_ptr(), s);
    out.iter().sum()
}

#[cfg(target_arch = "x86_64")]
fn gflops(threads: usize, iters: u64) -> f64 {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    let flop_per_thread = iters as f64 * 12.0 * 8.0 * 2.0;
    let start = std::time::Instant::now();
    pool.install(|| {
        use rayon::prelude::*;
        (0..threads).into_par_iter().for_each(|t| {
            // SAFETY: AVX2+FMA is required for the GEMM path this models;
            // this example is only meaningful on a machine that has it.
            let r = unsafe { burn(iters, t as f32) };
            std::hint::black_box(r);
        });
    });
    flop_per_thread * threads as f64 / start.elapsed().as_secs_f64() / 1e9
}

#[cfg(target_arch = "x86_64")]
fn main() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        println!("needs AVX2+FMA"); return;
    }
    // ~1.5 s per measurement so the clock settles to the sustained value
    let iters: u64 = std::env::var("R2_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(400_000_000);
    println!("throughput-bound FMA, 12 independent 8-wide accumulators per thread, no memory\n");
    println!("{:>8} {:>10} {:>12} {:>7}", "threads", "GFLOP/s", "per thread", "scale");
    let mut one = 0.0;
    for &t in &[1usize, 2, 3, 4, 6] {
        let g = gflops(t, iters);
        if t == 1 { one = g; }
        println!("{t:>8} {g:>10.1} {:>12.1} {:>6.2}x", g / t as f64, g / one);
    }
    println!("\n  1 thread at 76 GFLOP/s is the 2375 MHz FMA peak (2 ports x 8 lanes x 2);");
    println!("  above it is boost clock. The 6-thread figure divided by 6 is the");
    println!("  all-core rate this part can sustain under full AVX2 load.");
}

/// The question this example answers is about an AVX2 machine; on any
/// other architecture there is nothing to measure.
#[cfg(not(target_arch = "x86_64"))]
fn main() {
    println!("fma_power measures AVX2+FMA throughput; this is not an x86-64 build.");
}
