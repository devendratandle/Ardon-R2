//! Why is `grad_B` (TN) the slowest of the three GEMM cases?
//!
//! Parallelism in `sgemm` runs over ROW-BLOCKS of C. The three cases have
//! very different M:
//!
//! ```text
//!   forward  (NN)   M = tokens          2048  ->  many blocks
//!   grad_A   (NT)   M = tokens          2048  ->  many blocks
//!   grad_B   (TN)   M = weight in-dim  256/768 -> few blocks
//! ```
//!
//! If that is the cause, TN should show poor PARALLEL EFFICIENCY rather
//! than a poor serial kernel — the same micro-kernel, just not spread
//! across the cores. This measures serial and threaded separately so the
//! two explanations can be told apart.
//!
//!     cargo run --release -p r2-tensor --example gemm_scaling

use r2_linalg::gemm::{sgemm, Trans};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn t_s(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    median((0..5).map(|_| {
        let s = std::time::Instant::now();
        for _ in 0..reps { f(); }
        s.elapsed().as_secs_f64() / reps as f64
    }).collect())
}

fn mk(n: usize, ph: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.001 + ph).sin()).collect()
}

fn main() {
    let shapes = [
        (2048usize, 256usize, 8000usize, "output head"),
        (2048, 256, 768, "ffn w1/w3"),
        (2048, 768, 256, "ffn w2"),
        (2048, 256, 256, "q/o proj"),
    ];
    println!("cores available to rayon: {}\n", rayon::current_num_threads());
    println!("{:>14} {:>6} {:>10} {:>9} {:>9} {:>7}",
             "block", "case", "M", "1 thread", "6 threads", "scale");
    println!("{}", "-".repeat(62));

    for &(m, k, n, label) in &shapes {
        let a = mk(m * k, 0.0);
        let b = mk(k * n, 1.0);
        let g = mk(m * n, 2.0);
        let flop = 2.0 * m as f64 * k as f64 * n as f64;
        let reps = 2;

        // (case, M of the product, closure factory)
        let cases: [(&str, usize, Box<dyn Fn(bool) -> f64>); 3] = [
            ("NN", m, Box::new(|par| {
                let (a, b) = (mk(m * k, 0.0), mk(k * n, 1.0));
                t_s(reps, || { std::hint::black_box(sgemm(&a, Trans::No, &b, Trans::No, m, k, n, par)); })
            })),
            ("NT", m, Box::new(|par| {
                let (g, b) = (mk(m * n, 2.0), mk(k * n, 1.0));
                t_s(reps, || { std::hint::black_box(sgemm(&g, Trans::No, &b, Trans::Yes, m, n, k, par)); })
            })),
            ("TN", k, Box::new(|par| {
                let (a, g) = (mk(m * k, 0.0), mk(m * n, 2.0));
                t_s(reps, || { std::hint::black_box(sgemm(&a, Trans::Yes, &g, Trans::No, k, m, n, par)); })
            })),
        ];
        for (name, mm, run) in cases {
            let ser = run(false);
            let par = run(true);
            println!("{label:>14} {name:>6} {mm:>10} {:>8.1} GF {:>6.1} GF {:>6.2}x",
                     flop / ser / 1e9, flop / par / 1e9, ser / par);
        }
        let _ = (&a, &b, &g);
        println!();
    }
    println!("  A perfect 6-core scale is 6.00x. Where TN falls short of NN");
    println!("  on the SAME arithmetic, the kernel is not the problem — the");
    println!("  number of row-blocks to hand out is.");
}
