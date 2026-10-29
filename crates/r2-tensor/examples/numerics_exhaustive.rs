//! Every f32 bit pattern — all 2^32 of them — through the vector kernels a
//! training step runs, against their scalar forms and an f64 reference.
//!
//! `numerics_audit` sweeps `exp_r2`, `log_r2` and silu one element at a
//! time. A one-element slice never enters an 8-wide loop, so that sweep
//! measures the SCALAR tail; the AVX2 lanes training actually runs were
//! checked only on samples. Here every input goes through a lane: the
//! floats are fed in contiguous slices of 65,536 (a multiple of 8), and
//! each lane's result is compared with
//!
//!   * the scalar form of the same function, bit for bit — the promise in
//!     `ops::exp_r2`'s docs is that a value's bits do not depend on
//!     whether it landed in a lane or a tail (a row's length mod 8);
//!   * the exact value in f64, in ULPs of the f32 result.
//!
//! NaN results compare equal to NaN results whatever their payload.
//!
//!   cargo run --release -p r2-tensor --example numerics_exhaustive

use rayon::prelude::*;
use r2_tensor::ops;

const CHUNK: u64 = 1 << 16;

/// Distance of `got` from the exact value `want`, in ULPs of `want` as an
/// f32 (the same measure as `numerics_audit`).
fn ulps(got: f32, want: f64) -> f64 {
    let w = want as f32;
    if w == 0.0 { return if got == 0.0 { 0.0 } else { f64::INFINITY }; }
    if !w.is_finite() { return if got == w { 0.0 } else { f64::INFINITY }; }
    let a = w.abs();
    let ulp = (f32::from_bits(a.to_bits() + 1) - a) as f64;
    (got as f64 - want).abs() / ulp
}

fn same(a: f32, b: f32) -> bool { a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) }

/// Bit-identity of two forms over a set of inputs.
#[derive(Clone, Copy, Default)]
struct Ident { n: u64, differ: u64, first: Option<(f32, f32, f32)> }
impl Ident {
    fn add(&mut self, x: f32, a: f32, b: f32) {
        self.n += 1;
        if !same(a, b) { self.differ += 1; if self.first.is_none() { self.first = Some((x, a, b)); } }
    }
    fn join(self, o: Ident) -> Ident {
        Ident { n: self.n + o.n, differ: self.differ + o.differ, first: self.first.or(o.first) }
    }
    fn line(&self, what: &str) {
        match self.first {
            None => println!("{what:<44} identical on all {} inputs", self.n),
            Some((x, a, b)) => println!("{what:<44} DIFFER on {} of {} ({:.3e}); first x = {x:e}: {a:e} vs {b:e}",
                                        self.differ, self.n, self.differ as f64 / self.n as f64),
        }
    }
}

/// ULP error against an exact reference.
#[derive(Clone, Copy, Default)]
struct Err { n: u64, worst: f64, at: f32, over_half: u64, over_one: u64 }
impl Err {
    fn add(&mut self, x: f32, got: f32, want: f64) {
        self.n += 1;
        let e = ulps(got, want);
        if e > 0.5 { self.over_half += 1; }
        if e > 1.0 { self.over_one += 1; }
        if e > self.worst { self.worst = e; self.at = x; }
    }
    fn join(self, o: Err) -> Err {
        let (worst, at) = if self.worst >= o.worst { (self.worst, self.at) } else { (o.worst, o.at) };
        Err { n: self.n + o.n, worst, at, over_half: self.over_half + o.over_half, over_one: self.over_one + o.over_one }
    }
    fn line(&self, what: &str) {
        println!("{what:<44} max {:.3} ulp at x = {:e}; > 0.5 ulp {:.3e}, > 1 ulp {:.3e} of {} inputs",
                 self.worst, self.at, self.over_half as f64 / self.n as f64,
                 self.over_one as f64 / self.n as f64, self.n);
    }
}

#[derive(Clone, Copy, Default)]
struct All {
    exp_lane: Ident,
    silu_lane: Ident,
    dsilu_lane: Ident,
    dsilu_acc_lane: Ident,
    silu_err: Err,
    dsilu_err: Err,
    silu_sat: Err,
    dsilu_sat: Err,
    /// silu' crosses zero at x = -1.2785, where any relative measure
    /// explodes; the bound that means something there is absolute.
    dsilu_abs: (f64, f32),
}
impl All {
    fn join(self, o: All) -> All {
        All { exp_lane: self.exp_lane.join(o.exp_lane), silu_lane: self.silu_lane.join(o.silu_lane),
              dsilu_lane: self.dsilu_lane.join(o.dsilu_lane), dsilu_acc_lane: self.dsilu_acc_lane.join(o.dsilu_acc_lane),
              silu_err: self.silu_err.join(o.silu_err), dsilu_err: self.dsilu_err.join(o.dsilu_err),
              silu_sat: self.silu_sat.join(o.silu_sat), dsilu_sat: self.dsilu_sat.join(o.dsilu_sat),
              dsilu_abs: if self.dsilu_abs.0 >= o.dsilu_abs.0 { self.dsilu_abs } else { o.dsilu_abs } }
    }
}

fn silu64(x: f64) -> f64 { x / (1.0 + (-x).exp()) }
fn dsilu64(x: f64) -> f64 { let s = 1.0 / (1.0 + (-x).exp()); s + x * s * (1.0 - s) }

fn chunk(start: u64) -> All {
    let xs: Vec<f32> = (start..start + CHUNK).map(|b| f32::from_bits(b as u32)).collect();
    let n = xs.len();
    let mut a = All::default();

    // exp: the lanes of `exp_shift_sum` (shift 0) against `exp_r2`
    let mut e = vec![0.0f32; n];
    ops::exp_shift_sum(&xs, 0.0, &mut e);
    // silu: lanes against the scalar `silu`, and against f64
    let mut s = vec![0.0f32; n];
    ops::silu_into(&xs, &mut s);
    // silu': lanes (assign form, g = 1) against the same function on
    // one-element slices, which run its scalar tail
    let ones = vec![1.0f32; n];
    let mut d = vec![0.0f32; n];
    ops::silu_bwd_acc(&xs, &ones, &mut d, false);
    // ...and the accumulate form, into a buffer already holding 0.75
    let mut da = vec![0.75f32; n];
    ops::silu_bwd_acc(&xs, &ones, &mut da, true);

    for i in 0..n {
        let x = xs[i];
        a.exp_lane.add(x, e[i], ops::exp_r2(x));
        a.silu_lane.add(x, s[i], ops::silu(x));
        let mut t = [0.0f32];
        ops::silu_bwd_acc(&[x], &[1.0], &mut t, false);
        a.dsilu_lane.add(x, d[i], t[0]);
        let mut ta = [0.75f32];
        ops::silu_bwd_acc(&[x], &[1.0], &mut ta, true);
        a.dsilu_acc_lane.add(x, da[i], ta[0]);
        // Below -87.34 `exp_r2(-x)` saturates at 2.4e38 instead of
        // overflowing (see the special values), so silu there is
        // `x / 2.4e38` rather than ~0: measured apart, so it does not hide
        // the accuracy of the range activations live in.
        if x.is_finite() {
            let (se, de) = if x >= -87.336_54 { (&mut a.silu_err, &mut a.dsilu_err) }
                           else { (&mut a.silu_sat, &mut a.dsilu_sat) };
            se.add(x, s[i], silu64(x as f64));
            let want = dsilu64(x as f64);
            de.add(x, d[i], want);
            if x >= -87.336_54 {
                let e = (d[i] as f64 - want).abs();
                if e > a.dsilu_abs.0 { a.dsilu_abs = (e, x); }
            }
        }
    }
    a
}

fn main() {
    let t0 = std::time::Instant::now();
    let starts: Vec<u64> = (0..(1u64 << 32)).step_by(CHUNK as usize).collect();
    let all = starts.par_iter().map(|&s| chunk(s)).reduce(All::default, All::join);
    println!("all 2^32 f32 bit patterns, AVX2 lanes vs scalar forms ({:.0} s)\n", t0.elapsed().as_secs_f64());
    all.exp_lane.line("exp lanes (exp_shift_sum) vs exp_r2");
    all.silu_lane.line("silu lanes (silu_into) vs silu");
    all.dsilu_lane.line("silu' lanes vs scalar tail (assign)");
    all.dsilu_acc_lane.line("silu' lanes vs scalar tail (accumulate)");
    println!();
    all.silu_err.line("silu lanes vs f64, x >= -87.34");
    all.dsilu_err.line("silu' lanes vs f64, x >= -87.34");
    println!("{:<44} max abs error {:.3e} at x = {:e} (|silu'| <= 1.1)", "silu' lanes vs f64, x >= -87.34",
             all.dsilu_abs.0, all.dsilu_abs.1);
    all.silu_sat.line("silu lanes vs f64, x < -87.34 (exp saturated)");
    all.dsilu_sat.line("silu' lanes vs f64, x < -87.34 (exp saturated)");

    // log_r2 below the normal range: documented as not served; measured
    let t1 = std::time::Instant::now();
    let sub: Vec<u32> = (1..f32::MIN_POSITIVE.to_bits()).step_by(CHUNK as usize).collect();
    let lerr = sub.par_iter().map(|&c| {
        let mut e = Err::default();
        for b in c..(c + CHUNK as u32).min(f32::MIN_POSITIVE.to_bits()) {
            let x = f32::from_bits(b);
            e.add(x, ops::log_r2(x), (x as f64).ln());
        }
        e
    }).reduce(Err::default, Err::join);
    println!();
    lerr.line(&format!("log_r2 subnormals vs f64 ({:.0} s)", t1.elapsed().as_secs_f64()));

    // special values: what each function returns, next to IEEE / f64
    println!("\nspecial values                 exp_r2         libm exp     silu          silu'(g=1)    log_r2        f64 ln");
    let specials: [(&str, f32); 12] = [
        ("+0", 0.0), ("-0", -0.0), ("+inf", f32::INFINITY), ("-inf", f32::NEG_INFINITY), ("NaN", f32::NAN),
        ("min subnormal", f32::from_bits(1)), ("min normal", f32::MIN_POSITIVE), ("-87.33654 (exp lo)", -87.336_54),
        ("-87.33655", -87.336_55), ("88.37626 (exp hi)", 88.376_26), ("88.7228 (f32 exp ovf)", 88.722_84),
        ("max", f32::MAX),
    ];
    for (name, x) in specials {
        let mut dd = [0.0f32];
        ops::silu_bwd_acc(&[x], &[1.0], &mut dd, false);
        println!("{name:<24} {:>14e} {:>14e} {:>13e} {:>13e} {:>13e} {:>13e}",
                 ops::exp_r2(x), x.exp(), ops::silu(x), dd[0], ops::log_r2(x), (x as f64).ln());
    }
}
