//! `table()` — frequency counts. Phase R.7.
//!
//! Counts occurrences of each unique value. Side-effect: prints the
//! count table to stdout. Returns a named `Integer` vector for
//! Character / Factor inputs, or a scalar count of distinct values for
//! Numeric (matches the engine's pre-migration behaviour).

use r2_types::{Attrs, Character, ErrKind, EvalArg, Integer, R2Err, RVal};
use std::sync::Arc;

#[inline]
fn first_arg(a: &[EvalArg]) -> RVal {
    a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null)
}

pub fn bi_table(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &first_arg(a) {
        RVal::Character(v, _) => {
            let mut counts: Vec<(String, usize)> = Vec::new();
            for x in v {
                if let Some(s) = x {
                    if let Some(entry) = counts.iter_mut().find(|(k, _)| k == s.as_ref()) {
                        entry.1 += 1;
                    } else {
                        counts.push((s.to_string(), 1));
                    }
                }
            }
            counts.sort_by(|a, b| a.0.cmp(&b.0));
            for (k, _) in &counts { sout!("{:>12}", k); }
            soutln!();
            for (_, v) in &counts { sout!("{:>12}", v); }
            soutln!();
            let names: Vec<Character> = counts.iter().map(|(k, _)| Some(Arc::from(k.as_str()))).collect();
            let vals: Vec<Integer> = counts.iter().map(|(_, v)| Some(*v as i32)).collect();
            let mut attrs = Attrs::default();
            attrs.names = Some(names.into_iter().filter_map(|x| x).collect());
            Ok(RVal::Integer(vals.into(), attrs))
        }
        RVal::Numeric(v, _) => {
            let mut counts: Vec<(String, usize)> = Vec::new();
            for x in v {
                let key = match x { Some(n) => format!("{}", n), None => "NA".into() };
                if let Some(entry) = counts.iter_mut().find(|(k, _)| *k == key) {
                    entry.1 += 1;
                } else {
                    counts.push((key, 1));
                }
            }
            counts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for (k, _) in &counts { sout!("{:>8}", k); }
            soutln!();
            for (_, v) in &counts { sout!("{:>8}", v); }
            soutln!();
            // Engine compatibility: numeric path returns scalar count.
            Ok(RVal::Integer(vec![Some(counts.len() as i32)].into(), Attrs::default()))
        }
        RVal::Factor(f) => {
            let mut counts: Vec<(String, usize)> = f.levels.iter().map(|l| (l.to_string(), 0)).collect();
            for code in &f.codes {
                if let Some(idx) = code {
                    if let Some(entry) = counts.get_mut(*idx as usize) { entry.1 += 1; }
                }
            }
            for (k, _) in &counts { sout!("{:>12}", k); }
            soutln!();
            for (_, v) in &counts { sout!("{:>12}", v); }
            soutln!();
            let names: Vec<Character> = counts.iter().map(|(k, _)| Some(Arc::from(k.as_str()))).collect();
            let vals: Vec<Integer> = counts.iter().map(|(_, v)| Some(*v as i32)).collect();
            let mut attrs = Attrs::default();
            attrs.names = Some(names.into_iter().filter_map(|x| x).collect());
            Ok(RVal::Integer(vals.into(), attrs))
        }
        _ => Err(R2Err {
            msg: "table() works with character, numeric, or factor vectors. Try as.factor() first".into(),
            kind: ErrKind::Runtime,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    #[test]
    fn table_counts_character_levels() {
        let xs = RVal::Character(
            vec![Some(Arc::from("a")), Some(Arc::from("b")), Some(Arc::from("a"))],
            Attrs::default(),
        );
        let r = bi_table(&[evarg(xs)]).unwrap();
        match r {
            RVal::Integer(v, attrs) => {
                let counts: Vec<i32> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(counts, vec![2, 1]);
                let names = attrs.names.unwrap();
                assert_eq!(names[0].as_ref(), "a");
                assert_eq!(names[1].as_ref(), "b");
            }
            _ => panic!("table() must return Integer with names"),
        }
    }
}
