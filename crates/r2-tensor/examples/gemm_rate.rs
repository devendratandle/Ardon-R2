//! What rate does `r2_linalg::gemm::sgemm` reach on the shapes an LLM step
//! actually runs — forward (NN) and both gradient cases (NT, TN)?
//!
//! Bars on this machine (~460 GFLOP/s f32 peak, 6 cores, AVX2+FMA):
//!
//! ```text
//!   the i-k-j loop this replaced    27 - 73 GFLOP/s
//!   JAX / Eigen                    ~170 - 280   (portable C++, no assembly)
//!   PyTorch / MKL                   171 - 286   (hand-written assembly)
//! ```
//!
//! Re-measure the reference bars with `benchmarks/llm/lmo14_grad_ab.py`
//! rather than trusting the numbers above: they were taken in one window
//! on one machine, and this one drifts about 10% over hours.
//!
//!     cargo run --release -p r2-tensor --example gemm_rate

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
    // (m, k, n, calls per step, label) — the shipping config: dim 256,
    // 4 layers, ffn 768, vocab 8,000, 32 x 64 = 2,048 tokens.
    let shapes = [
        (2048usize, 256usize, 8000usize, 1usize, "output head"),
        (2048, 256, 768, 8, "ffn w1/w3"),
        (2048, 768, 256, 4, "ffn w2"),
        (2048, 256, 256, 8, "q/o proj"),
        (2048, 256, 128, 8, "k/v proj"),
    ];

    println!("sgemm on the LLM's shapes — forward (NN) and both gradients\n");
    println!("{:>14} {:>18} {:>6} {:>9} {:>9} {:>9}",
             "block", "m x k x n", "x/step", "NN GF/s", "NT GF/s", "TN GF/s");
    println!("{}", "-".repeat(70));

    let (mut tot_ms, mut tot_flop) = (0.0f64, 0.0f64);
    for &(m, k, n, cnt, label) in &shapes {
        let a = mk(m * k, 0.0);
        let b = mk(k * n, 1.0);
        let g = mk(m * n, 2.0);
        let reps = 3;
        let flop = 2.0 * m as f64 * k as f64 * n as f64;

        // forward: C(m x n) = A(m x k) . B(k x n)
        let nn = t_s(reps, || {
            std::hint::black_box(sgemm(&a, Trans::No, &b, Trans::No, m, k, n, true));
        });
        // grad_A(m x k) = g(m x n) . B^T(n x k); B is stored k x n, which IS
        // the n x k the transposed read wants, so nothing is materialised.
        let nt = t_s(reps, || {
            std::hint::black_box(sgemm(&g, Trans::No, &b, Trans::Yes, m, n, k, true));
        });
        // grad_B(k x n) = A^T(k x m) . g(m x n); likewise for A.
        let tn = t_s(reps, || {
            std::hint::black_box(sgemm(&a, Trans::Yes, &g, Trans::No, k, m, n, true));
        });

        tot_ms += (nn + nt + tn) * cnt as f64 * 1e3;
        tot_flop += flop * 3.0 * cnt as f64;
        println!("{label:>14} {:>6}x{k}x{n:<6} {cnt:>6} {:>9.1} {:>9.1} {:>9.1}",
                 m, flop / nn / 1e9, flop / nt / 1e9, flop / tn / 1e9);
    }
    println!("{}", "-".repeat(70));
    println!("  all three, weighted by calls per step: {:.0} ms, {:.1} GFLOP/s",
             tot_ms, tot_flop / (tot_ms * 1e-3) / 1e9);
    println!("\n  NN is the forward, NT is grad_A, TN is grad_B. TN is the");
    println!("  slowest of the three: its M is the weight's INPUT dim, so it");
    println!("  has the fewest row-blocks to spread across cores.");
}
