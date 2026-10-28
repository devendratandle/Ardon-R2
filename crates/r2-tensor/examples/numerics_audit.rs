//! Accuracy of the LLM numerics against an f64 reference.
//!
//! `exp_r2` and `log_r2` are swept over EVERY float in their range; silu,
//! its derivative, softmax-CE, RMSNorm and the RoPE table are measured on
//! the shapes training uses. Errors are in ULPs of the f32 result unless
//! stated otherwise.
//!
//!   cargo run --release -p r2-tensor --example numerics_audit

use rayon::prelude::*;
use r2_tensor::ops;

/// Distance of `got` from the exact value `want`, in ULPs of `want` as an f32.
fn ulps(got: f32, want: f64) -> f64 {
    let w = want as f32;
    if w == 0.0 { return if got == 0.0 { 0.0 } else { f64::INFINITY }; }
    let a = w.abs();
    let ulp = (f32::from_bits(a.to_bits() + 1) - a) as f64;
    (got as f64 - want).abs() / ulp
}

/// Max ULP error of `f` against `r` over every float with bits in `lo..hi`.
fn sweep(lo: u32, hi: u32, f: impl Fn(f32) -> f32 + Sync, r: impl Fn(f64) -> f64 + Sync) -> (f64, f32, f64) {
    let chunks: Vec<u32> = (lo..hi).step_by(1 << 16).collect();
    let (worst, at, over) = chunks.par_iter().map(|&c| {
        let (mut w, mut a, mut n) = (0.0f64, 0.0f32, 0u64);
        for b in c..(c + (1 << 16)).min(hi) {
            let x = f32::from_bits(b);
            let e = ulps(f(x), r(x as f64));
            if e > 0.5 { n += 1; }
            if e > w { w = e; a = x; }
        }
        (w, a, n)
    }).reduce(|| (0.0, 0.0, 0), |p, q| (if p.0 >= q.0 { p.0 } else { q.0 }, if p.0 >= q.0 { p.1 } else { q.1 }, p.2 + q.2));
    (worst, at, over as f64 / (hi - lo) as f64)
}

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64) / (1u64 << 53) as f64
}
fn normal(seed: &mut u64) -> f64 {
    let (u, v) = (lcg(seed).max(1e-300), lcg(seed));
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

fn main() {
    // ── exp: every float that does not flush or clamp ──
    let (lo_neg, hi_pos) = ((-87.336_54f32).to_bits(), 88.376_26f32.to_bits());
    let neg = sweep(0x8000_0000, lo_neg, ops::exp_r2, f64::exp);
    let pos = sweep(0, hi_pos, ops::exp_r2, f64::exp);
    println!("exp_r2   negative x: max {:.3} ulp at {:e}, {:.2e} of inputs > 0.5 ulp", neg.0, neg.1, neg.2);
    println!("exp_r2   positive x: max {:.3} ulp at {:e}, {:.2e} of inputs > 0.5 ulp", pos.0, pos.1, pos.2);
    let libm_neg = sweep(0x8000_0000, lo_neg, f32::exp, f64::exp);
    println!("libm expf negative x: max {:.3} ulp (reference point)", libm_neg.0);

    // the vector path must give exp_r2's bits
    let xs: Vec<f32> = (0..1 << 20).map(|i| -87.0 + 175.0 * (i as f32) / (1 << 20) as f32).collect();
    let mut v = vec![0.0f32; xs.len()];
    ops::exp_shift_sum(&xs, 0.0, &mut v);
    let diff = xs.iter().zip(&v).filter(|(x, e)| ops::exp_r2(**x).to_bits() != e.to_bits()).count();
    println!("exp8 lanes != exp_r2: {diff} of {}", xs.len());

    // ── log: every positive normal float ──
    let lg = sweep(f32::MIN_POSITIVE.to_bits(), f32::INFINITY.to_bits(), ops::log_r2, f64::ln);
    println!("log_r2   normals:    max {:.3} ulp at {:e}, {:.2e} of inputs > 0.5 ulp", lg.0, lg.1, lg.2);

    // ── silu and its derivative, every float in [-88, 88] ──
    let silu64 = |x: f64| x / (1.0 + (-x).exp());
    let silu_vec = |x: f32| { let mut o = [0.0f32]; ops::silu_into(&[x], &mut o); o[0] };
    let s_neg = sweep(0x8000_0000, (-88.0f32).to_bits(), silu_vec, silu64);
    let s_pos = sweep(0, 88.0f32.to_bits(), silu_vec, silu64);
    println!("silu     [-88,0]:    max {:.3} ulp at {:e}", s_neg.0, s_neg.1);
    println!("silu     [0,88]:     max {:.3} ulp at {:e}", s_pos.0, s_pos.1);
    let mut worst_d = (0.0f64, 0.0f32);
    let xs: Vec<f32> = (0..1 << 22).map(|i| -30.0 + 60.0 * (i as f32) / (1 << 22) as f32).collect();
    let mut g = vec![0.0f32; xs.len()];
    ops::silu_bwd_acc(&xs, &vec![1.0; xs.len()], &mut g, false);
    for (&x, &d) in xs.iter().zip(&g) {
        let s = 1.0 / (1.0 + (-(x as f64)).exp());
        let want = s * (1.0 + x as f64 * (1.0 - s));
        let e = (d as f64 - want).abs();   // absolute: d crosses zero at x = -1.278
        if e > worst_d.0 { worst_d = (e, x); }
    }
    println!("silu'    [-30,30]:   max abs error {:.2e} at {} (|silu'| <= 1.1)", worst_d.0, worst_d.1);

    // ── softmax cross-entropy on the training shape ──
    let (rows, vocab) = (2048usize, 8000usize);
    let mut seed = 7u64;
    for spread in [1.0f64, 4.0, 16.0] {
        let logits: Vec<f32> = (0..rows * vocab).map(|_| (spread * normal(&mut seed)) as f32).collect();
        let targets: Vec<usize> = (0..rows).map(|_| (lcg(&mut seed) * vocab as f64) as usize).collect();
        let (mut l32, mut l64, mut worst_lse) = (0.0f32, 0.0f64, 0.0f64);
        let mut worst_g = 0.0f64;
        let mut grow = vec![0.0f32; vocab];
        for r in 0..rows {
            let row = &logits[r * vocab..(r + 1) * vocab];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = m + ops::exp_shift_sum_only(row, m).ln();
            l32 += lse - row[targets[r]];
            let m64 = m as f64;
            let lse64 = m64 + row.iter().map(|&x| (x as f64 - m64).exp()).sum::<f64>().ln();
            l64 += lse64 - row[targets[r]] as f64;
            worst_lse = worst_lse.max(ulps(lse, lse64));
            if r < 64 {
                ops::softmax_ce_grad_acc(row, lse, targets[r], 1.0, &mut grow, false);
                for (j, &gj) in grow.iter().enumerate() {
                    let want = (row[j] as f64 - lse64).exp() - if j == targets[r] { 1.0 } else { 0.0 };
                    worst_g = worst_g.max((gj as f64 - want).abs());
                }
            }
        }
        let (mean32, mean64) = (l32 / rows as f32, l64 / rows as f64);
        println!("CE spread {spread:>4}: loss {mean64:.7}  f32 rel err {:.2e}  lse max {:.2} ulp  grad max abs {:.2e}",
                 ((mean32 as f64 - mean64) / mean64).abs(), worst_lse, worst_g);
    }

    // ── RMSNorm ──
    for d in [256usize, 768] {
        let x: Vec<f32> = (0..64 * d).map(|_| (3.0 * normal(&mut seed)) as f32).collect();
        let w: Vec<f32> = (0..d).map(|_| (1.0 + 0.1 * normal(&mut seed)) as f32).collect();
        let out = ops::rmsnorm(&x, &w, 1e-5);
        let mut worst = 0.0f64;
        for r in 0..64 {
            let xr = &x[r * d..(r + 1) * d];
            let ms = xr.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / d as f64;
            let sc = 1.0 / (ms + 1e-5f32 as f64).sqrt();
            for j in 0..d { worst = worst.max(ulps(out[r * d + j], xr[j] as f64 * sc * w[j] as f64)); }
        }
        println!("rmsnorm  d={d}:      max {worst:.2} ulp");
    }

    // ── RoPE angles by position ──
    let (hd, base, period) = (64usize, 10_000.0f32, 32_768usize);
    let tab = ops::rope_table(period, hd, base);
    let half = hd / 2;
    let mut cut = [(256usize, 0.0f64), (2048, 0.0), (8192, 0.0), (32_768, 0.0)];
    for pos in 0..period {
        for p in 0..half {
            let freq = 1.0 / (base as f64).powf(2.0 * p as f64 / hd as f64);
            let th = pos as f64 * freq;
            let (c, s) = tab[pos * half + p];
            let e = (c as f64 - th.cos()).abs().max((s as f64 - th.sin()).abs());
            for k in cut.iter_mut() { if pos < k.0 { k.1 = k.1.max(e); } }
        }
    }
    for (n, e) in cut { println!("rope     pos < {n:>6}: max abs error in cos/sin {e:.2e}"); }
}
