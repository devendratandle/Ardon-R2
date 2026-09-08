//! What parallel speedup can this machine actually give?
//!
//! `sgemm` scales 1.9-2.8x on six cores, and the obvious reading is that
//! the threading is bad. But a 15 W laptop part drops its clock under
//! all-core load, so the ceiling may not be 6x at all — and building a
//! BLIS-style thread mesh to chase a ceiling that does not exist would be
//! a lot of work for nothing.
//!
//! This measures the ceiling with a workload that has NO memory traffic,
//! no synchronisation beyond one join, and perfectly equal chunks: a
//! register-resident FMA chain per thread. Whatever this reaches is the
//! most any parallel code on this machine can reach, and `sgemm` should be
//! judged against it rather than against 6.00x.
//!
//!     cargo run --release -p r2-tensor --example scaling_ceiling

use rayon::prelude::*;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// A dependency-free FMA chain that stays in registers. Returns a value so
/// nothing is optimised away.
#[inline(never)]
fn burn(iters: u64, seed: f32) -> f32 {
    // Seeded through `black_box` so the chain is not constant-foldable —
    // a first attempt used literals and the whole loop vanished, reporting
    // 0.0000 s and a 0.00x speedup.
    let mut a = [seed, seed + 1.0, seed + 2.0, seed + 3.0,
                 seed + 4.0, seed + 5.0, seed + 6.0, seed + 7.0];
    let m = std::hint::black_box(0.9999999f32);
    let add = std::hint::black_box(1.0e-7f32);
    for _ in 0..iters {
        for x in a.iter_mut() { *x = x.mul_add(m, add); }
    }
    a.iter().sum()
}

fn main() {
    let cores = rayon::current_num_threads();
    let iters = 4_000_000u64;
    println!("rayon threads: {cores}\n");
    println!("{:>10} {:>12} {:>10} {:>9}", "threads", "seconds", "vs 1", "efficiency");
    println!("{}", "-".repeat(46));

    let t1 = median((0..5).map(|_| {
        let s = std::time::Instant::now();
        std::hint::black_box(burn(iters, std::hint::black_box(1.0)));
        s.elapsed().as_secs_f64()
    }).collect());
    println!("{:>10} {:>12.4} {:>10} {:>9}", 1, t1, "1.00x", "100%");

    for n in [2usize, 4, cores] {
        if n > cores { continue; }
        let t = median((0..5).map(|_| {
            let s = std::time::Instant::now();
            let v: f32 = (0..n).into_par_iter()
                .map(|i| burn(iters, std::hint::black_box(1.0 + i as f32))).sum();
            std::hint::black_box(v);
            s.elapsed().as_secs_f64()
        }).collect());
        // n threads each doing the SAME work as the 1-thread case, so a
        // perfect machine finishes in the same wall time: speedup = n.
        let speedup = t1 * n as f64 / t;
        println!("{:>10} {:>12.4} {:>9.2}x {:>8.0}%",
                 n, t, speedup, speedup / n as f64 * 100.0);
    }

    println!("\n  Each thread runs the SAME chain, so a machine that held its");
    println!("  clock would finish {cores} threads in the same wall time as 1.");
    println!("  Whatever the last row shows is the CEILING for any parallel");
    println!("  code here — judge sgemm's scaling against that, not 6.00x.");
}
