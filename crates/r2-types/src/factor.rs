//! Building factors the way R does.
//!
//! R's `factor(x)` takes its default levels from `sort(unique(x))` in
//! `x`'s own type, then labels them with `as.character`: numbers sort
//! numerically (2 before 10), logicals FALSE before TRUE, NA is never a
//! level. Everything that groups by a vector (`as.factor`, `split`,
//! `tapply`, `table`, model dummies) goes through here so the group order
//! is the same everywhere.
//!
//! Strings sort by byte (the C locale), the same order `order()` uses. R
//! in a UTF-8 or Windows locale collates instead ("a" "A" "b" "B"), so
//! mixed-case levels still differ from such an R.

use crate::{Factor, RVal};
use std::collections::HashMap;
use std::sync::Arc;

/// A double as `as.character` writes it: 15 significant digits, fixed
/// notation unless scientific is narrower (R's width rule), so 1e5 is
/// "1e+05" but 123456 stays "123456".
pub fn num_label(n: f64) -> String {
    if n.is_nan() { return "NaN".into(); }
    if n.is_infinite() { return if n > 0.0 { "Inf".into() } else { "-Inf".into() }; }
    if n == 0.0 { return "0".into(); }
    let sci = format!("{:.14e}", n);
    let (mant, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let mant = if mant.contains('.') { mant.trim_end_matches('0').trim_end_matches('.') } else { mant };
    let sig = mant.bytes().filter(u8::is_ascii_digit).count() as i32;
    let sci = format!("{}e{}{:02}", mant, if exp < 0 { '-' } else { '+' }, exp.abs());
    let fixed = format!("{:.*}", (sig - 1 - exp).max(0) as usize, n);
    if fixed.len() > sci.len() { sci } else { fixed }
}

/// `as.character(x)` element by element, and the default levels
/// `factor(x)` would give it. A factor's levels come back as they are
/// (unused ones included). `None` for anything that is not an atomic
/// vector or factor.
pub fn factor_labels(x: &RVal) -> Option<(Vec<Option<Arc<str>>>, Vec<Arc<str>>)> {
    let lab = |s: String| Arc::<str>::from(s.as_str());
    Some(match x {
        RVal::Character(v, _) => {
            let mut levels: Vec<Arc<str>> = v.iter().flatten().cloned().collect();
            levels.sort_unstable();
            levels.dedup();
            (v.clone(), levels)
        }
        RVal::Numeric(v, _) => {
            let labels = v.iter().map(|x| x.map(|n| lab(num_label(n)))).collect();
            let mut vals: Vec<f64> = v.iter().flatten().copied().collect();
            // NaN is a value, not NA: it sorts last and is a level of its own.
            vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or_else(|| a.is_nan().cmp(&b.is_nan())));
            // Labelling is monotone, so equal labels are adjacent (0 and -0,
            // or two doubles that agree to 15 digits).
            let mut levels: Vec<Arc<str>> = vals.into_iter().map(|n| lab(num_label(n))).collect();
            levels.dedup();
            (labels, levels)
        }
        RVal::Integer(v, _) => {
            let labels = v.iter().map(|x| x.map(|n| lab(n.to_string()))).collect();
            let mut vals: Vec<i32> = v.iter().flatten().copied().collect();
            vals.sort_unstable();
            vals.dedup();
            (labels, vals.into_iter().map(|n| lab(n.to_string())).collect())
        }
        RVal::Logical(v, _) => {
            let tf = |b: bool| Arc::<str>::from(if b { "TRUE" } else { "FALSE" });
            let labels = v.iter().map(|x| x.map(tf)).collect();
            let levels = [false, true].into_iter().filter(|b| v.iter().any(|x| *x == Some(*b))).map(tf).collect();
            (labels, levels)
        }
        RVal::Factor(f) => (f.codes.iter().map(|c| c.and_then(|i| f.levels.get(i as usize).cloned())).collect(),
                            f.levels.clone()),
        _ => return None,
    })
}

impl Factor {
    /// `as.factor(x)`: a factor stays as it is, anything else gets R's
    /// default (sorted) levels.
    pub fn from_values(x: &RVal) -> Option<Factor> {
        if let RVal::Factor(f) = x { return Some(f.clone()); }
        let (labels, levels) = factor_labels(x)?;
        Some(Factor::with_levels(&labels, levels, false))
    }

    /// Code each label by its position in `levels`; a label that is not a
    /// level, or NA, gets an NA code. `levels` must be distinct.
    pub fn with_levels(labels: &[Option<Arc<str>>], levels: Vec<Arc<str>>, ordered: bool) -> Factor {
        let index: HashMap<&str, u32> = levels.iter().enumerate().map(|(i, l)| (l.as_ref(), i as u32)).collect();
        let codes = labels.iter().map(|x| x.as_ref().and_then(|s| index.get(s.as_ref()).copied())).collect();
        Factor { codes, levels, ordered }
    }

    /// `factor(f)` / `droplevels(f)`: only the levels in use, in their
    /// existing order.
    pub fn drop_unused(&self) -> Factor {
        let mut used = vec![false; self.levels.len()];
        for c in self.codes.iter().flatten() { if let Some(u) = used.get_mut(*c as usize) { *u = true; } }
        let mut remap = vec![None; self.levels.len()];
        let mut levels = Vec::new();
        for (i, l) in self.levels.iter().enumerate() {
            if used[i] { remap[i] = Some(levels.len() as u32); levels.push(l.clone()); }
        }
        let codes = self.codes.iter().map(|c| c.and_then(|i| remap.get(i as usize).copied().flatten())).collect();
        Factor { codes, levels, ordered: self.ordered }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Attrs;

    fn chr(v: &[Option<&str>]) -> RVal { RVal::Character(v.iter().map(|x| x.map(Arc::from)).collect(), Attrs::default()) }
    fn levels(f: &Factor) -> Vec<&str> { f.levels.iter().map(|l| l.as_ref()).collect() }

    #[test]
    fn character_levels_sort_and_skip_na() {
        let f = Factor::from_values(&chr(&[Some("b"), Some("a"), None, Some("b")])).unwrap();
        assert_eq!(levels(&f), ["a", "b"]);
        assert_eq!(f.codes, [Some(1), Some(0), None, Some(1)]);
    }

    #[test]
    fn numeric_levels_sort_numerically() {
        let x = RVal::Numeric(vec![Some(10.0), Some(2.0), None, Some(f64::NAN), Some(-1.5)].into(), Attrs::default());
        let f = Factor::from_values(&x).unwrap();
        assert_eq!(levels(&f), ["-1.5", "2", "10", "NaN"]);
        assert_eq!(f.codes, [Some(2), Some(1), None, Some(3), Some(0)]);
    }

    #[test]
    fn num_label_matches_as_character() {
        assert_eq!(num_label(1e5), "1e+05");
        assert_eq!(num_label(123456.0), "123456");
        assert_eq!(num_label(0.1 + 0.2), "0.3");
        assert_eq!(num_label(1.0 / 3.0), "0.333333333333333");
        assert_eq!(num_label(0.0001), "1e-04");
        assert_eq!(num_label(-2.5), "-2.5");
    }

    #[test]
    fn drop_unused_keeps_order() {
        let f = Factor::with_levels(&[Some(Arc::from("c")), Some(Arc::from("a"))],
                                    vec![Arc::from("c"), Arc::from("b"), Arc::from("a")], false);
        let d = f.drop_unused();
        assert_eq!(levels(&d), ["c", "a"]);
        assert_eq!(d.codes, [Some(0), Some(1)]);
    }
}
