//! Formatting a numeric vector the way R prints it: ONE format for the
//! whole vector (R's `formatReal`). Each element needs some number of
//! significant digits (at most `digits`, 7 by default); the vector takes
//! the most decimals any element needs, so `c(1, 2.5)` is `1.0 2.5` and
//! `c(1.123456789, 100)` is `1.123457 100.000000`. Fixed notation is used
//! unless scientific is narrower (`1e+05`, `1e-04`). `cat`, `paste` and
//! `as.character` format each element on its own and do not use this.

use crate::Real;

/// R's `scientific()`: for a finite `x` rounded to `digits` significant
/// digits — whether it is negative, its decimal exponent, how many
/// significant digits it really needs, and whether rounding it to the
/// fixed-notation decimals would widen its integer part.
fn scientific(x: f64, digits: i32) -> (bool, i32, i32, bool) {
    if x == 0.0 { return (false, 0, 1, false); }
    let neg = x < 0.0;
    let r = x.abs();
    let mut kp = r.log10().floor() as i32 - digits + 1;
    // r scaled to exactly `digits` digits before the decimal point
    let mut r_prec = r / 10f64.powi(kp);
    if r_prec < 10f64.powi(digits - 1) { r_prec *= 10.0; kp -= 1; }
    let mut alpha = r_prec.round_ties_even();
    let mut nsig = digits;
    for _ in 0..digits {
        alpha /= 10.0;
        if alpha == alpha.floor() { nsig -= 1; } else { break; }
    }
    if nsig == 0 { nsig = 1; kp += 1; }
    let kpower = kp + digits - 1;
    let rgt = (digits - kpower).clamp(0, 22);
    let fuzz = 0.5 / 10f64.powi(rgt);
    let widens = kpower > 0 && kpower <= 22 && r < 10f64.powi(kpower) - fuzz;
    (neg, kpower, nsig, widens)
}

/// How a whole vector is written: `w` characters wide, `d` digits after
/// the point (fixed) or in the mantissa (scientific, `e` > 0 exponent
/// digits beyond the first).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealFormat { pub w: usize, pub d: usize, pub e: usize }

/// R's `formatReal(x, digits, nsmall)` — the common format of `v`.
pub fn real_format(v: &[Real], digits: usize, nsmall: usize) -> RealFormat {
    let digits = digits.clamp(1, 22) as i32;
    let (mut neg, mut any) = (false, false);
    let (mut mxsl, mut rgt, mut mxl, mut mxe, mut mne, mut mxns) = (i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MAX, i32::MIN);
    let (mut na, mut nan, mut pinf, mut ninf) = (false, false, false, false);
    for x in v {
        match *x {
            None => na = true,
            Some(x) if x.is_nan() => nan = true,
            Some(x) if x.is_infinite() => if x > 0.0 { pinf = true } else { ninf = true },
            Some(x) => {
                any = true;
                let (sgn, kpower, nsig, widens) = scientific(x, digits);
                let left = kpower + 1 - widens as i32;
                let sleft = sgn as i32 + left.max(1);
                if sgn { neg = true; }
                rgt = rgt.max(nsig - left);
                mxl = mxl.max(left);
                mxsl = mxsl.max(sleft);
                mxns = mxns.max(nsig);
                mxe = mxe.max(kpower);
                mne = mne.min(kpower);
            }
        }
    }
    let mut f = RealFormat { w: 0, d: 0, e: 0 };
    if any {
        if mxl < 0 { mxsl = 1 + neg as i32; }
        let mut rgt = rgt.max(0);
        let mut w_fixed = mxsl + rgt + (rgt != 0) as i32;
        let e = if mxe >= 100 || mne <= -99 { 2 } else { 1 };
        let d = mxns - 1;
        let w_sci = neg as i32 + (d > 0) as i32 + d + 4 + e;
        if w_fixed <= w_sci {
            if nsmall as i32 > rgt { rgt = nsmall as i32; w_fixed = mxsl + rgt + (rgt != 0) as i32; }
            f = RealFormat { w: w_fixed as usize, d: rgt as usize, e: 0 };
        } else {
            f = RealFormat { w: w_sci as usize, d: d as usize, e: e as usize };
        }
    }
    if na { f.w = f.w.max(2); }
    if nan || pinf { f.w = f.w.max(3); }
    if ninf { f.w = f.w.max(4); }
    f
}

/// One element in the common format `f`, right-aligned to `f.w`.
pub fn encode_real(x: Real, f: RealFormat) -> String {
    let s = match x {
        None => "NA".to_string(),
        Some(x) if x.is_nan() => "NaN".to_string(),
        Some(x) if x.is_infinite() => (if x > 0.0 { "Inf" } else { "-Inf" }).to_string(),
        Some(x) => {
            let x = if x == 0.0 { 0.0 } else { x };   // never "-0"
            if f.e == 0 {
                format!("{:.*}", f.d, x)
            } else {
                let m = format!("{:.*e}", f.d, x);
                let (mant, exp) = m.split_once('e').unwrap_or((&m, "0"));
                let exp: i32 = exp.parse().unwrap_or(0);
                format!("{}e{}{:0w$}", mant, if exp < 0 { '-' } else { '+' }, exp.abs(), w = f.e + 1)
            }
        }
    };
    format!("{:>w$}", s, w = f.w)
}

/// `v` formatted as R prints it: common format, every string `w` wide.
pub fn format_reals(v: &[Real], digits: usize, nsmall: usize) -> Vec<String> {
    let f = real_format(v, digits, nsmall);
    v.iter().map(|x| encode_real(*x, f)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(v: &[f64]) -> Vec<String> {
        format_reals(&v.iter().map(|x| Some(*x)).collect::<Vec<_>>(), 7, 0).iter().map(|s| s.trim().to_string()).collect()
    }

    /// Every expected string is R 4.5.3's print() of the same vector.
    #[test]
    fn matches_r_print() {
        assert_eq!(fr(&[1.0, 2.5]), ["1.0", "2.5"]);
        assert_eq!(fr(&[1.123456789, 100.0]), ["1.123457", "100.000000"]);
        assert_eq!(fr(&[1e5]), ["1e+05"]);
        assert_eq!(fr(&[123456.0]), ["123456"]);
        assert_eq!(fr(&[0.0001]), ["1e-04"]);
        assert_eq!(fr(&[std::f64::consts::PI]), ["3.141593"]);
        assert_eq!(fr(&[-1.5, 2.0]), ["-1.5", "2.0"]);
        assert_eq!(fr(&[0.1 + 0.2]), ["0.3"]);
        assert_eq!(fr(&[-0.0]), ["0"]);
        assert_eq!(fr(&[1e-20, 1.0]), ["1e-20", "1e+00"]);
        assert_eq!(fr(&[0.5, 100000.0]), ["5e-01", "1e+05"]);
        assert_eq!(fr(&[1e10, 1.0]), ["1e+10", "1e+00"]);
        assert_eq!(fr(&[-1e-5, 3.0]), ["-1e-05", "3e+00"]);
    }

    #[test]
    fn common_width_and_specials() {
        let v = [Some(1.0), None, Some(f64::NAN), Some(10.5)];
        assert_eq!(format_reals(&v, 7, 0), [" 1.0", "  NA", " NaN", "10.5"]);
    }
}
