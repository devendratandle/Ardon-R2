//! Is the silu backward slower in R2's arithmetic order than in PyTorch's?
//!
//! Same input, same `exp`, same 8-wide AVX2 loop, three variants:
//!
//!   R2     `gout += g * (s + x*s*(1-s))`      two FMAs, accumulating store
//!   torch  `out   = g * s * (1 + x*(1-s))`    ATen's order, plain store
//!   torch+ `torch` followed by the separate `gout += out` pass autograd
//!          runs when a tensor has more than one consumer — the cost R2
//!          folds into its FMA
//!
//! Shape is the shipping one (2048 x 768, one FFN gate per layer).
//! Interleaved rounds, median, single thread so it is the kernel being
//! measured and not the fork-join. The three outputs are also compared
//! elementwise so any speed difference is not a correctness difference.
//!
//!     cargo run --release -p r2-tensor --example silu_forms

use r2_tensor::ops::{silu_bwd, silu_bwd_torch_form};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let n = 2048 * 768;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0007).sin() * 4.0).collect();
    let g: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0011).cos()).collect();
    let rounds = 21;
    let reps = 20;

    let mut out_r2 = vec![0.0f32; n];
    let mut out_t = vec![0.0f32; n];
    let mut acc_t = vec![0.0f32; n];

    let (mut t_r2, mut t_t, mut t_tp) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..rounds {
        let order = if r % 3 == 0 { [0, 1, 2] } else if r % 3 == 1 { [1, 2, 0] } else { [2, 0, 1] };
        for which in order {
            let s = std::time::Instant::now();
            for _ in 0..reps {
                match which {
                    0 => { silu_bwd(&x, &g, &mut out_r2); }
                    1 => { silu_bwd_torch_form(&x, &g, &mut out_t); }
                    _ => {
                        silu_bwd_torch_form(&x, &g, &mut out_t);
                        for (a, &o) in acc_t.iter_mut().zip(&out_t) { *a += o; }
                    }
                }
                std::hint::black_box(&out_r2);
                std::hint::black_box(&out_t);
            }
            let t = s.elapsed().as_secs_f64() / reps as f64 * 1e3;
            match which { 0 => t_r2.push(t), 1 => t_t.push(t), _ => t_tp.push(t) }
        }
    }

    // correctness: the two orders must agree to f32 rounding
    out_r2.iter_mut().for_each(|v| *v = 0.0);
    silu_bwd(&x, &g, &mut out_r2);
    silu_bwd_torch_form(&x, &g, &mut out_t);
    // Absolute difference in units of the largest output: a relative
    // measure blows up where silu' crosses zero (x ~ -1.28) and the two
    // orders cancel differently on a result that is itself ~0.
    let scale = out_r2.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let mut worst = 0.0f32;
    for i in 0..n {
        let d = (out_r2[i] - out_t[i]).abs() / scale;
        if d > worst { worst = d; }
    }

    println!("silu backward, {} elements (2048 x 768), single thread, {rounds} rounds x {reps} reps, median\n", n);
    println!("{:<44} {:>9} {:>12}", "variant", "ms", "ns/element");
    println!("{}", "-".repeat(68));
    for (label, t) in [("R2:     gout += g*(s + x*s*(1-s))  [2 FMA]", &t_r2),
                       ("torch:  out = g*s*(1 + x*(1-s))     [ATen order]", &t_t),
                       ("torch + separate accumulate pass", &t_tp)] {
        let m = median(t.clone());
        println!("{label:<44} {m:>9.3} {:>12.3}", m * 1e6 / n as f64);
    }
    println!("\nmax relative difference between the two orders: {worst:.2e}  (f32 eps = 1.19e-7)");
}
