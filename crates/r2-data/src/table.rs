//! `table()` — frequency counts. Phase R.7.
//!
//! Counts occurrences of each level of `as.factor(x)`. Side-effect:
//! prints the count table to stdout. Returns a named `Integer` vector.

use r2_types::{Attrs, ErrKind, EvalArg, Factor, Integer, R2Err, RVal};

#[inline]
fn first_arg(a: &[EvalArg]) -> RVal {
    a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null)
}

pub fn bi_table(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x = first_arg(a);
    // One count per level of as.factor(x): sorted like R (numbers
    // numerically), NA not counted, a factor's unused levels counted as 0.
    let Some(f) = Factor::from_values(&x) else {
        return Err(R2Err {
            msg: "table() works with character, numeric, logical, or factor vectors. Try as.factor() first".into(),
            kind: ErrKind::Runtime,
        });
    };
    let mut counts = vec![0usize; f.levels.len()];
    for c in f.codes.iter().flatten() { counts[*c as usize] += 1; }
    let width = if matches!(x, RVal::Numeric(..) | RVal::Integer(..)) { 8 } else { 12 };
    for k in &f.levels { sout!("{:>w$}", k, w = width); }
    soutln!();
    for v in &counts { sout!("{:>w$}", v, w = width); }
    soutln!();
    let vals: Vec<Integer> = counts.iter().map(|v| Some(*v as i32)).collect();
    let mut attrs = Attrs::default();
    attrs.names = Some(f.levels);
    Ok(RVal::Integer(vals.into(), attrs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

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
