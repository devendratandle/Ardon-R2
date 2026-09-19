//! How well does memory bandwidth scale across cores, at each level of the
//! cache hierarchy?
//!
//! `scaling_ceiling` answers two extremes: register-resident work scales
//! 5.4x on six cores, and a stream far larger than L3 scales 1.14x. A GEMM
//! lives between them ON PURPOSE — its packed panels are sized so an A
//! block sits in L2 and a B panel in L3. So the number that actually bounds
//! `sgemm` is the scaling at *its* working-set size, not at either extreme.
//!
//! This sweeps the working set from inside L1 to far past L3 at 1/2/4/6
//! threads and reports GB/s and parallel efficiency for each. Read the row
//! whose size matches the panel you care about.
//!
//! It also repeats the whole sweep, because this machine power-throttles
//! under sustained load and a bandwidth ceiling measured cold is not the
//! one a 5-month training run lives with.
//!
//!     cargo run --release -p r2-tensor --example bandwidth_sweep
//!     R2_PASSES=40 cargo run --release -p r2-tensor --example bandwidth_sweep

use rayon::prelude::*;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// STREAM triad: `a = b + s*c`. Three streams, one of them written.
///
/// Note the write costs a read too (read-for-ownership), so the true
/// traffic is 4 buffers where the reported figure counts 3. That makes the
/// absolute GB/s conservative; the SCALING, which is what this example is
/// for, is unaffected.
#[inline(never)]
fn triad(b: &[f32], c: &[f32], a: &mut [f32], s: f32) {
    for i in 0..a.len() {
        a[i] = b[i] + s * c[i];
    }
}

/// Time one configuration, returning GB/s aggregated over all threads.
fn measure(elems: usize, threads: usize, reps: usize) -> f64 {
    let mk = || -> Vec<f32> { (0..elems).map(|i| (i % 251) as f32).collect() };
    // Every thread owns its own buffers: this is a bandwidth measurement,
    // not a false-sharing one.
    let mut sets: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> =
        (0..threads).map(|_| (mk(), mk(), mk())).collect();
    let s = std::hint::black_box(1.000_001f32);

    // Warm the pages so the first timed pass is not measuring faults.
    sets.par_iter_mut().for_each(|(b, c, a)| triad(b, c, a, s));

    // ONE fork-join per timing, with the repetitions INSIDE each worker.
    //
    // The first version of this looped `par_iter_mut` `reps` times, which
    // at an 8 KB working set meant 4,000 fork-joins around a few
    // microseconds of work each — it reported 116 GB/s on one thread and
    // 24 on six, which is a measurement of rayon's barrier, not of memory.
    let secs = median((0..5).map(|_| {
        let st = std::time::Instant::now();
        sets.par_iter_mut().for_each(|(b, c, a)| {
            for _ in 0..reps { triad(b, c, a, s); }
        });
        st.elapsed().as_secs_f64()
    }).collect());
    std::hint::black_box(sets[0].2[0]);

    let bytes = (elems * 4 * 3 * reps * threads) as f64;
    bytes / secs / 1e9
}

fn main() {
    let cores = rayon::current_num_threads();
    let passes: usize = std::env::var("R2_PASSES").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(3);
    println!("rayon threads: {cores}, passes: {passes}");
    println!("Working set is PER THREAD. `sgemm` packs an A block for L2");
    println!("(~96 KB) and a B panel for L3 (~1 MB) — those are the rows");
    println!("that bound it, not the 48 MB one.\n");

    // (label, elements per buffer, repetitions to keep each timing ~ms)
    let cases: [(&str, usize, usize); 5] = [
        ("8 KB   (L1)", 2 * 1024, 4000),
        ("96 KB  (L2, A block)", 24 * 1024, 400),
        ("1 MB   (L3, B panel)", 256 * 1024, 40),
        ("8 MB   (L3 edge)", 2 * 1024 * 1024, 6),
        ("48 MB  (RAM)", 12 * 1024 * 1024, 1),
    ];

    for pass in 1..=passes {
        let clk = "";
        println!("── pass {pass}{clk} {}", "─".repeat(46));
        println!("{:<24}{:>9}{:>9}{:>9}{:>9}   {:>10}",
                 "working set / thread", "1 thr", "2 thr", "4 thr", "6 thr", "scale @6");
        for (label, elems, reps) in cases {
            let mut gbs = Vec::new();
            for t in [1usize, 2, 4, cores] {
                if t > cores { continue; }
                gbs.push(measure(elems, t, reps));
            }
            let scale = gbs.last().copied().unwrap_or(0.0) / gbs[0];
            print!("{label:<24}");
            for g in &gbs { print!("{g:>9.1}"); }
            println!("   {scale:>9.2}x");
        }
        println!();
    }

    println!("  Aggregate GB/s across all threads, so a flat row means the");
    println!("  memory system is already saturated by one core and adding");
    println!("  cores buys nothing. `sgemm` scales 2.7-3.8x; judge that");
    println!("  against the L2/L3 rows, which is where its panels live.");
}
