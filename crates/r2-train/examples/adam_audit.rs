//! How exact is a weight update? R2's f32 Adam/AdamW step against the same
//! recurrence in f64 with the exact constants (β₁ = 0.9 and β₂ = 0.999 as
//! real numbers, bias corrections in f64), both fed the SAME f32 gradients,
//! whose magnitudes span 1e-6 .. 1.
//!
//!   update err  relative error of the step R2 takes (Δp) against the exact
//!               step from the same weights, over every parameter whose
//!               step is not negligible
//!   bias corr   relative error of R2's f32 `1 - β₂^t`
//!   drift       max |p32 - p64| after t steps, relative to how far the
//!               weights have moved in total
//!
//!     cargo run --release -p r2-train --example adam_audit

use r2_train::optim::Adam;

fn main() {
    let n = 100_000usize;
    let (lr, eps, b1, b2) = (3e-4f64, 1e-8f64, 0.9f64, 0.999f64);
    for decay in [0.0, 0.1] {
        let mut seed = 99u64;
        let mut rnd = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                           ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0 };
        let start: Vec<f32> = (0..n).map(|_| (rnd() * 0.06) as f32).collect();
        let mut opt = Adam::new(n, lr as f32);
        opt.weight_decay = decay as f32;
        let mut p32 = vec![start.clone()];
        let mut p: Vec<f64> = start.iter().map(|&x| x as f64).collect();
        let (mut m, mut v) = (vec![0f64; n], vec![0f64; n]);
        println!("{}", if decay > 0.0 { "AdamW, weight decay 0.1" } else { "Adam" });
        println!("  {:>6}  {:>11}  {:>11}  {:>9}", "step", "update err", "bias corr", "drift");
        for t in 1..=3000u64 {
            let g: Vec<f32> = (0..n).map(|i| (rnd() * 10f64.powf(-6.0 * (i % 97) as f64 / 96.0)) as f32).collect();
            let before = p32[0].clone();
            opt.step_blocks(&mut p32, &[&g[..]], 1.0).unwrap();
            let (bc1, bc2) = (1.0 - b1.powi(t as i32), 1.0 - b2.powi(t as i32));
            let mut worst = 0f64;
            for i in 0..n {
                let gi = g[i] as f64;
                m[i] = b1 * m[i] + (1.0 - b1) * gi;
                v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
                let step = lr * (m[i] / bc1) / ((v[i] / bc2).sqrt() + eps);
                p[i] = p[i] * (1.0 - lr * decay) - step;
                // the exact step from R2's own weights, with the exact moments
                let exact = -(before[i] as f64) * lr * decay - step;
                let got = p32[0][i] as f64 - before[i] as f64;
                if exact.abs() > 1e-3 * lr { worst = worst.max((got - exact).abs() / exact.abs()); }
            }
            if matches!(t, 1 | 10 | 100 | 1000 | 3000) {
                let bc_err = ((1.0 - 0.999f32.powi(t as i32)) as f64 - bc2).abs() / bc2;
                let moved = p.iter().zip(&start).map(|(&a, &b)| (a - b as f64).abs()).fold(0f64, f64::max);
                let drift = p32[0].iter().zip(&p).map(|(&a, &b)| (a as f64 - b).abs()).fold(0f64, f64::max);
                println!("  {t:>6}  {worst:>11.2e}  {bc_err:>11.2e}  {:>9.2e}", drift / moved);
            }
        }
    }
}
