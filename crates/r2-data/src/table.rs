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

/// Arguments that are options, not variables to count.
const OPTIONS: [&str; 4] = ["dnn", "useNA", "exclude", "deparse.level"];

fn levels_of(x: &RVal) -> Result<Factor, R2Err> {
    Factor::from_values(x).ok_or_else(|| R2Err {
        msg: "table() works with character, numeric, logical, or factor vectors. Try as.factor() first".into(),
        kind: ErrKind::Runtime,
    })
}

/// A two-way table: counts with the first variable down the rows, as an
/// integer matrix of class "table" whose dimnames carry the levels and are
/// named after the variables (`table(cyl = ..., gear = ...)`).
fn two_way(a: &EvalArg, b: &EvalArg, dnn: Option<&RVal>) -> Result<RVal, R2Err> {
    let (fa, fb) = (levels_of(&a.value)?, levels_of(&b.value)?);
    if fa.codes.len() != fb.codes.len() {
        return Err(R2Err { msg: "all arguments must have the same length".into(), kind: ErrKind::Runtime });
    }
    let (r, c) = (fa.levels.len(), fb.levels.len());
    let mut counts = vec![0i32; r * c];
    for (x, y) in fa.codes.iter().zip(&fb.codes) {
        if let (Some(i), Some(j)) = (x, y) { counts[*j as usize * r + *i as usize] += 1; }
    }
    let given = |k: usize| -> Arc<str> {
        match dnn {
            Some(RVal::Character(d, _)) => d.get(k).cloned().flatten().unwrap_or_else(|| Arc::from("")),
            _ => [a, b][k].name.clone().unwrap_or_else(|| Arc::from("")),
        }
    };
    let labels = |f: &Factor| RVal::Character(f.levels.iter().cloned().map(Some).collect(), Attrs::default());
    let mut attrs = Attrs { dim: Some(vec![r, c]), class: Some(Arc::from("table")), ..Default::default() };
    attrs.custom.insert(Arc::from("dimnames"), RVal::list(vec![(Some(given(0)), labels(&fa)), (Some(given(1)), labels(&fb))]));
    Ok(RVal::Integer(counts.into_iter().map(Some).collect::<Vec<_>>().into(), attrs))
}

pub fn bi_table(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let vars: Vec<&EvalArg> = a.iter().filter(|x| !x.name.as_deref().is_some_and(|n| OPTIONS.contains(&n))).collect();
    let dnn_arg = a.iter().find(|x| x.name.as_deref() == Some("dnn")).map(|x| &x.value);
    match vars.as_slice() {
        [x, y] => return two_way(x, y, dnn_arg),
        [_, _, _, ..] => return Err(R2Err { msg: "table() counts one or two variables here".into(), kind: ErrKind::Runtime }),
        _ => {}
    }
    let x = vars.first().map(|v| v.value.clone()).unwrap_or_else(|| first_arg(a));
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
    // the header line: dnn =, else the variable's name (`table(x)` prints "x")
    let dnn = dnn_arg.cloned().or_else(|| vars.first().and_then(|v| v.name.clone())
        .map(|n| RVal::Character(vec![Some(n)], Attrs::default())));
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
