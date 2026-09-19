//! Does fusing GEMMs that share a left operand pay?
//!
//! The SwiGLU FFN issues `h·w1` and `h·w3` — two 2048x256x768 GEMMs with
//! the same `h` — and attention issues `h·wq`, `h·wk`, `h·wv` (n = 256,
//! 128, 128). Concatenating the weights turns each group into ONE GEMM
//! with a wider n. PyTorch eager does not do this (it runs the separate
//! calls, exactly as R2 does today), so a win here is a structural lead,
//! not catch-up.
//!
//! The question is purely about `sgemm`: is one call at 2n faster than
//! two calls at n? Measured for the forward (NN) and both gradients (NT,
//! TN), separate and fused INTERLEAVED so a clock that drifts over the
//! run cannot favour one side. Times, not rates, because time is what a
//! step pays.
//!
//! What it cannot measure: the copies a fused layout costs elsewhere
//! (slicing gate/up out of the fused output, or reading them strided).
//! Those are decided at step level, and only if this shows headroom
//! worth paying them for.
//!
//!     cargo run --release -p r2-tensor --example fused_shapes

use r2_linalg::gemm::{sgemm, Trans};

fn mk(n: usize, ph: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.001 + ph).sin()).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// One group: a left operand `m x k` shared by weights of widths `ns`.
struct Group {
    label: &'static str,
    m: usize,
    k: usize,
    ns: Vec<usize>,
}

fn main() {
    let groups = [
        Group { label: "ffn w1|w3", m: 2048, k: 256, ns: vec![768, 768] },
        Group { label: "attn q|k|v", m: 2048, k: 256, ns: vec![256, 128, 128] },
    ];
    let rounds: usize = std::env::var("R2_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(9);
    println!("separate calls vs one fused call, {rounds} interleaved rounds, median\n");
    println!("{:>12} {:>5} {:>14} {:>14} {:>8}", "block", "case", "separate ms", "fused ms", "fused/sep");
    println!("{}", "-".repeat(58));

    for g in &groups {
        let (m, k) = (g.m, g.k);
        let ntot: usize = g.ns.iter().sum();
        let a = mk(m * k, 0.0);
        let bs: Vec<Vec<f32>> = g.ns.iter().enumerate().map(|(i, &n)| mk(k * n, 1.0 + i as f32)).collect();
        let bf = mk(k * ntot, 1.5);
        let gs: Vec<Vec<f32>> = g.ns.iter().enumerate().map(|(i, &n)| mk(m * n, 2.0 + i as f32)).collect();
        let gf = mk(m * ntot, 2.5);

        for case in ["NN", "NT", "TN"] {
            let sep = |_: ()| {
                for (i, &n) in g.ns.iter().enumerate() {
                    match case {
                        "NN" => { std::hint::black_box(sgemm(&a, Trans::No, &bs[i], Trans::No, m, k, n, true)); }
                        "NT" => { std::hint::black_box(sgemm(&gs[i], Trans::No, &bs[i], Trans::Yes, m, n, k, true)); }
                        _    => { std::hint::black_box(sgemm(&a, Trans::Yes, &gs[i], Trans::No, k, m, n, true)); }
                    }
                }
            };
            let fus = |_: ()| {
                match case {
                    "NN" => { std::hint::black_box(sgemm(&a, Trans::No, &bf, Trans::No, m, k, ntot, true)); }
                    "NT" => { std::hint::black_box(sgemm(&gf, Trans::No, &bf, Trans::Yes, m, ntot, k, true)); }
                    _    => { std::hint::black_box(sgemm(&a, Trans::Yes, &gf, Trans::No, k, m, ntot, true)); }
                }
            };
            // warm both
            sep(()); fus(());
            let (mut ts, mut tf) = (Vec::new(), Vec::new());
            for r in 0..rounds {
                // alternate which side goes first each round
                let order: [(&str, &dyn Fn(())); 2] = if r % 2 == 0 { [("s", &sep), ("f", &fus)] } else { [("f", &fus), ("s", &sep)] };
                for (tag, f) in order {
                    let s = std::time::Instant::now();
                    for _ in 0..3 { f(()); }
                    let t = s.elapsed().as_secs_f64() / 3.0 * 1e3;
                    if tag == "s" { ts.push(t) } else { tf.push(t) }
                }
            }
            let (ms, mf) = (median(ts), median(tf));
            println!("{:>12} {:>5} {:>14.2} {:>14.2} {:>8.3}", g.label, case, ms, mf, mf / ms);
        }
        println!();
    }
    println!("  fused/sep < 0.9 is the bar: below it, one wide GEMM beats two");
    println!("  narrow ones by more than this machine's noise and the fusion");
    println!("  can afford its slicing cost. Near 1.0 the premise fails.");
}
