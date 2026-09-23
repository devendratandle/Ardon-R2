//! `table()` — frequency counts. Phase R.7.
//!
//! Counts occurrences of each level of `as.factor(x)`, as a named
//! `Integer` vector of class "table" (printed with its header line).

use r2_types::{Attrs, ErrKind, EvalArg, Factor, Integer, R2Err, RVal};
use std::sync::Arc;

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
    let vals: Vec<Integer> = counts.iter().map(|v| Some(*v as i32)).collect();
    let mut attrs = Attrs::default();
    attrs.names = Some(f.levels);
    // A value of class "table": printed (with its header line) only when
    // auto-printed or print()ed, never while computing.
    attrs.class = Some(Arc::from("table"));
    let dnn = a.iter().find(|x| x.name.as_deref() == Some("dnn")).map(|x| x.value.clone());
    if let Some(d) = dnn { attrs.custom.insert(Arc::from("dnn"), d); }
    Ok(RVal::Integer(vals.into(), attrs))
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
