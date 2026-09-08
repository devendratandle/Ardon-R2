//! NA handling + merge. Phase R.7.
//!
//! `na.omit`, `complete.cases`, `merge`. `merge` implements R's join:
//! composite keys (`by`, or `by.x`/`by.y`), inner and left/right/full
//! outer joins (`all`, `all.x`, `all.y`), `.x`/`.y` suffixing on
//! colliding non-key names, key-ordered output (`sort`), and
//! type-preserving columns.

use r2_types::{Attrs, DataFrame, ErrKind, EvalArg, Logical, R2Err, RVal};
use std::sync::Arc;

#[inline]
fn first_arg(a: &[EvalArg]) -> RVal { a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn nth_arg(a: &[EvalArg], i: usize) -> RVal { a.get(i).map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn arg_named(a: &[EvalArg], name: &str) -> Option<RVal> {
    a.iter().find(|x| x.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|x| x.value.clone())
}

fn to_string_vec(col: &RVal) -> Vec<String> {
    match col {
        RVal::Numeric(v, _) => v.iter().map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).collect(),
        RVal::Integer(v, _) => v.iter().map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).collect(),
        RVal::Character(v, _) => v.iter().map(|x| match x { Some(s) => s.to_string(), None => "NA".into() }).collect(),
        RVal::Logical(v, _) => v.iter().map(|x| match x {
            Some(true) => "TRUE".into(), Some(false) => "FALSE".into(), None => "NA".into()
        }).collect(),
        _ => Vec::new(),
    }
}

fn filter_col_by_mask(col: &RVal, keep: &[bool]) -> RVal {
    match col {
        RVal::Numeric(v, _) => RVal::Numeric(
            v.iter().zip(keep).filter_map(|(x, k)| if *k { Some(*x) } else { None }).collect(),
            Attrs::default(),
        ),
        RVal::Integer(v, _) => RVal::Integer(
            v.iter().zip(keep).filter_map(|(x, k)| if *k { Some(*x) } else { None }).collect(),
            Attrs::default(),
        ),
        RVal::Character(v, _) => RVal::Character(
            v.iter().zip(keep).filter_map(|(x, k)| if *k { Some(x.clone()) } else { None }).collect(),
            Attrs::default(),
        ),
        RVal::Logical(v, _) => RVal::Logical(
            v.iter().zip(keep).filter_map(|(x, k)| if *k { Some(*x) } else { None }).collect(),
            Attrs::default(),
        ),
        _ => col.clone(),
    }
}

pub fn bi_na_omit(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &first_arg(a) {
        RVal::Numeric(v, _) => Ok(RVal::Numeric(
            v.iter().filter(|x| x.is_some()).cloned().collect(),
            Attrs::default(),
        )),
        RVal::Integer(v, _) => Ok(RVal::Integer(
            v.iter().filter(|x| x.is_some()).cloned().collect(),
            Attrs::default(),
        )),
        RVal::Character(v, _) => Ok(RVal::Character(
            v.iter().filter(|x| x.is_some()).cloned().collect(),
            Attrs::default(),
        )),
        RVal::DataFrame(df) => {
            let nrow = df.nrow();
            let keep: Vec<bool> = (0..nrow).map(|r| {
                df.columns.iter().all(|(_, col)| match col {
                    RVal::Numeric(v, _) => v.get(r).map(|x| x.is_some()).unwrap_or(false),
                    RVal::Integer(v, _) => v.get(r).map(|x| x.is_some()).unwrap_or(false),
                    RVal::Character(v, _) => v.get(r).map(|x| x.is_some()).unwrap_or(false),
                    _ => true,
                })
            }).collect();
            let columns: Vec<(Arc<str>, RVal)> = df.columns.iter().map(|(name, col)| {
                (name.clone(), filter_col_by_mask(col, &keep))
            }).collect();
            let removed = keep.iter().filter(|x| !**x).count();
            if removed > 0 { soutln!("Removed {} rows with NA values", removed); }
            Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
        }
        _ => Ok(first_arg(a)),
    }
}

pub fn bi_complete_cases(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &first_arg(a) {
        RVal::DataFrame(df) => {
            let nrow = df.nrow();
            let result: Vec<Logical> = (0..nrow).map(|r| {
                Some(df.columns.iter().all(|(_, col)| match col {
                    RVal::Numeric(v, _) => v.get(r).map(|x| x.is_some()).unwrap_or(false),
                    RVal::Integer(v, _) => v.get(r).map(|x| x.is_some()).unwrap_or(false),
                    RVal::Character(v, _) => v.get(r).map(|x| x.is_some()).unwrap_or(false),
                    _ => true,
                }))
            }).collect();
            Ok(RVal::Logical(result.into(), Attrs::default()))
        }
        _ => Err(R2Err { msg: "complete.cases needs data.frame".into(), kind: ErrKind::Runtime }),
    }
}

// ── merge ────────────────────────────────────────────────────────────
//
// R's `merge()` is a relational join, and the three things that make it
// one — composite keys, outer joins, and a defined row order — were all
// missing. What stood here matched on ONE column, ignored `all.x`/`all.y`
// entirely (silently returning an inner join, so a caller asking for an
// outer join got fewer rows and no error), and rebuilt every column by
// formatting it to a string and re-parsing, which turned a character
// column of digits into numbers.

/// One key cell, in a form that can be ordered as well as compared.
///
/// Matching still goes through the normalised string (so integer `1` and
/// double `1.0` are the same key, as in R), but ORDERING needs the type
/// back: sorting "10" before "9" is exactly the bug that string keys
/// invite.
#[derive(Clone, Debug)]
enum KeyPart {
    Num(f64),
    Str(Arc<str>),
    Na,
}

impl KeyPart {
    /// Ties the three variants into one order so mixed columns still sort
    /// deterministically. NA sorts last, as in R.
    fn rank(&self) -> u8 {
        match self {
            KeyPart::Num(_) => 0,
            KeyPart::Str(_) => 1,
            KeyPart::Na => 2,
        }
    }
}

fn cmp_key(a: &[KeyPart], b: &[KeyPart]) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    for (x, y) in a.iter().zip(b) {
        let o = match (x, y) {
            (KeyPart::Num(p), KeyPart::Num(q)) => p.partial_cmp(q).unwrap_or(Ordering::Equal),
            (KeyPart::Str(p), KeyPart::Str(q)) => p.as_ref().cmp(q.as_ref()),
            _ => x.rank().cmp(&y.rank()),
        };
        if o != Ordering::Equal {
            return o;
        }
    }
    Ordering::Equal
}

/// A column's key cells, keeping the type that decides their order.
fn key_parts(col: &RVal) -> Vec<KeyPart> {
    match col {
        RVal::Numeric(v, _) => v.iter().map(|x| x.map_or(KeyPart::Na, KeyPart::Num)).collect(),
        RVal::Integer(v, _) => v.iter()
            .map(|x| x.map_or(KeyPart::Na, |n| KeyPart::Num(n as f64))).collect(),
        RVal::Logical(v, _) => v.iter()
            .map(|x| x.map_or(KeyPart::Na, |b| KeyPart::Num(if b { 1.0 } else { 0.0 }))).collect(),
        RVal::Character(v, _) => v.iter()
            .map(|x| x.clone().map_or(KeyPart::Na, KeyPart::Str)).collect(),
        RVal::Factor(f) => f.codes.iter()
            .map(|c| c.and_then(|i| f.levels.get(i as usize).cloned())
                      .map_or(KeyPart::Na, KeyPart::Str)).collect(),
        _ => Vec::new(),
    }
}

/// Take rows by index, keeping the column's type. `None` means "no row on
/// this side of the join" and produces NA — which is the whole point of an
/// outer join, and is why this cannot go through strings.
fn gather_col(col: &RVal, idx: &[Option<usize>]) -> RVal {
    match col {
        RVal::Numeric(v, _) => RVal::Numeric(
            idx.iter().map(|i| i.and_then(|i| v.get(i)).copied().flatten()).collect(), Attrs::default()),
        RVal::Integer(v, _) => RVal::Integer(
            idx.iter().map(|i| i.and_then(|i| v.get(i)).copied().flatten()).collect(), Attrs::default()),
        RVal::Logical(v, _) => RVal::Logical(
            idx.iter().map(|i| i.and_then(|i| v.get(i)).copied().flatten()).collect(), Attrs::default()),
        RVal::Character(v, _) => RVal::Character(
            idx.iter().map(|i| i.and_then(|i| v.get(i)).cloned().flatten()).collect(),
            Attrs::default()),
        // A factor keeps its levels; only the codes are re-indexed, so an
        // unmatched row becomes NA rather than a bogus level.
        RVal::Factor(f) => RVal::Factor(r2_types::Factor {
            codes: idx.iter().map(|i| i.and_then(|i| f.codes.get(i).copied()).flatten()).collect(),
            levels: f.levels.clone(),
            ordered: f.ordered,
        }),
        _ => col.clone(),
    }
}

/// The value a column contributes to key matching. Same normalisation for
/// both frames, so `1` and `1.0` agree.
fn key_strings(col: &RVal) -> Vec<String> {
    match col {
        RVal::Factor(f) => f.codes.iter()
            .map(|c| c.and_then(|i| f.levels.get(i as usize))
                      .map(|s| s.to_string()).unwrap_or_else(|| "NA".into()))
            .collect(),
        _ => to_string_vec(col),
    }
}

/// `by = "k"` or `by = c("k1","k2")` — R takes a character vector, and the
/// old code read only the first element, so a two-key merge silently
/// joined on one key and duplicated the other as `.y`.
fn str_vec(v: &RVal) -> Vec<String> {
    match v {
        RVal::Character(c, _) => c.iter().filter_map(|x| x.as_ref().map(|s| s.to_string())).collect(),
        RVal::Factor(f) => f.codes.iter()
            .filter_map(|c| c.and_then(|i| f.levels.get(i as usize)).map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    }
}

fn logical_arg(a: &[EvalArg], name: &str) -> Option<bool> {
    arg_named(a, name).and_then(|v| match v {
        RVal::Logical(l, _) => l.get(0).copied().flatten(),
        RVal::Numeric(n, _) => n.get(0).copied().flatten().map(|x| x != 0.0),
        RVal::Integer(n, _) => n.get(0).copied().flatten().map(|x| x != 0),
        _ => None,
    })
}

pub fn bi_merge(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let df1 = match &first_arg(a) {
        RVal::DataFrame(df) => df.clone(),
        _ => return Err(R2Err { msg: "merge needs data.frame".into(), kind: ErrKind::Type }),
    };
    let df2 = match &nth_arg(a, 1) {
        RVal::DataFrame(df) => df.clone(),
        _ => return Err(R2Err { msg: "merge needs data.frame".into(), kind: ErrKind::Type }),
    };

    // Key columns. `by.x`/`by.y` let the two frames name the same key
    // differently; plain `by` means both. With none given R uses EVERY
    // shared column name, not just the first one it finds.
    let by = arg_named(a, "by").map(|v| str_vec(&v)).unwrap_or_default();
    let by_x = arg_named(a, "by.x").map(|v| str_vec(&v)).unwrap_or_else(|| by.clone());
    let by_y = arg_named(a, "by.y").map(|v| str_vec(&v)).unwrap_or_else(|| by.clone());
    let (by_x, by_y) = if by_x.is_empty() || by_y.is_empty() {
        let shared: Vec<String> = df1.columns.iter()
            .filter(|(n1, _)| df2.columns.iter().any(|(n2, _)| n1 == n2))
            .map(|(n, _)| n.to_string())
            .collect();
        (shared.clone(), shared)
    } else {
        (by_x, by_y)
    };
    if by_x.is_empty() {
        return Err(R2Err {
            msg: "merge: no common column found, specify by=".into(),
            kind: ErrKind::Runtime,
        });
    }
    if by_x.len() != by_y.len() {
        return Err(R2Err {
            msg: "merge: 'by.x' and 'by.y' must name the same number of columns".into(),
            kind: ErrKind::Runtime,
        });
    }

    let col_of = |df: &DataFrame, name: &str, which: &str| -> Result<RVal, R2Err> {
        df.get_col(name).cloned().ok_or_else(|| R2Err {
            msg: format!("merge: '{name}' not in {which} data.frame"),
            kind: ErrKind::Runtime,
        })
    };
    let kx: Vec<RVal> = by_x.iter().map(|n| col_of(&df1, n, "first")).collect::<Result<_, _>>()?;
    let ky: Vec<RVal> = by_y.iter().map(|n| col_of(&df2, n, "second")).collect::<Result<_, _>>()?;

    let (n1, n2) = (df1.nrow(), df2.nrow());
    let sx: Vec<Vec<String>> = kx.iter().map(|c| key_strings(c)).collect();
    let sy: Vec<Vec<String>> = ky.iter().map(|c| key_strings(c)).collect();
    let px: Vec<Vec<KeyPart>> = kx.iter().map(|c| key_parts(c)).collect();
    let py: Vec<Vec<KeyPart>> = ky.iter().map(|c| key_parts(c)).collect();
    let row_key = |s: &[Vec<String>], r: usize| -> Vec<String> {
        s.iter().map(|c| c.get(r).cloned().unwrap_or_else(|| "NA".into())).collect()
    };
    let row_parts = |p: &[Vec<KeyPart>], r: usize| -> Vec<KeyPart> {
        p.iter().map(|c| c.get(r).cloned().unwrap_or(KeyPart::Na)).collect()
    };

    // Index the right frame once. The old nested scan was O(n*m) and this
    // is O(n+m); at 10k x 10k rows that is 100M comparisons against 20k.
    let mut index: std::collections::HashMap<Vec<String>, Vec<usize>> =
        std::collections::HashMap::with_capacity(n2);
    for j in 0..n2 {
        index.entry(row_key(&sy, j)).or_default().push(j);
    }

    let all = logical_arg(a, "all").unwrap_or(false);
    let all_x = logical_arg(a, "all.x").unwrap_or(all);
    let all_y = logical_arg(a, "all.y").unwrap_or(all);
    let sort = logical_arg(a, "sort").unwrap_or(true);

    // Pairs of (left row, right row); `None` on a side is an unmatched row
    // kept by an outer join.
    let mut pairs: Vec<(Option<usize>, Option<usize>)> = Vec::new();
    let mut matched_right = vec![false; n2];
    for i in 0..n1 {
        match index.get(&row_key(&sx, i)) {
            Some(js) => {
                for &j in js {
                    matched_right[j] = true;
                    pairs.push((Some(i), Some(j)));
                }
            }
            None => {
                if all_x {
                    pairs.push((Some(i), None));
                }
            }
        }
    }
    if all_y {
        for (j, hit) in matched_right.iter().enumerate() {
            if !hit {
                pairs.push((None, Some(j)));
            }
        }
    }

    // R returns rows ordered by the key columns (`sort = TRUE` by default),
    // which is why the right-only rows above can simply be appended.
    if sort {
        pairs.sort_by(|l, r| {
            let kl = match l.0 { Some(i) => row_parts(&px, i), None => row_parts(&py, l.1.unwrap()) };
            let kr = match r.0 { Some(i) => row_parts(&px, i), None => row_parts(&py, r.1.unwrap()) };
            cmp_key(&kl, &kr)
        });
    }
    let li: Vec<Option<usize>> = pairs.iter().map(|(i, _)| *i).collect();
    let ri: Vec<Option<usize>> = pairs.iter().map(|(_, j)| *j).collect();

    let mut columns: Vec<(Arc<str>, RVal)> = Vec::new();

    // Key columns first, named after `by.x`. A row present only on the
    // right has no left key, so the value comes from the right — otherwise
    // every `all.y` row would carry an NA key.
    for (c, (cx, cy)) in kx.iter().zip(ky.iter()).enumerate() {
        columns.push((Arc::from(by_x[c].as_str()), gather_key(cx, &li, cy, &ri)));
    }

    // Non-key columns. R suffixes BOTH sides when a name collides — `v.x`
    // and `v.y` — where this used to leave the left one bare.
    let non_key_1: Vec<&(Arc<str>, RVal)> = df1.columns.iter()
        .filter(|(n, _)| !by_x.iter().any(|b| b == n.as_ref())).collect();
    let non_key_2: Vec<&(Arc<str>, RVal)> = df2.columns.iter()
        .filter(|(n, _)| !by_y.iter().any(|b| b == n.as_ref())).collect();
    for (name, col) in &non_key_1 {
        let clash = non_key_2.iter().any(|(n2, _)| n2 == name);
        let out = if clash { Arc::from(format!("{name}.x").as_str()) } else { name.clone() };
        columns.push((out, gather_col(col, &li)));
    }
    for (name, col) in &non_key_2 {
        let clash = non_key_1.iter().any(|(n1, _)| n1 == name);
        let out = if clash { Arc::from(format!("{name}.y").as_str()) } else { name.clone() };
        columns.push((out, gather_col(col, &ri)));
    }

    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
}

/// Build a key column by taking each row from whichever frame has it.
///
/// A row kept by `all.y` has no left index, so its key must come from the
/// right — otherwise every right-only row would carry an NA key. Doing it
/// in one pass avoids needing a mutable set on the columnar types.
fn gather_key(cx: &RVal, li: &[Option<usize>], cy: &RVal, ri: &[Option<usize>]) -> RVal {
    macro_rules! pick {
        ($x:expr, $y:expr) => {
            li.iter().zip(ri).map(|(l, r)| match l {
                Some(i) => $x.get(*i).copied().flatten(),
                None => r.and_then(|j| $y.get(j)).copied().flatten(),
            }).collect()
        };
    }
    match (cx, cy) {
        (RVal::Numeric(x, _), RVal::Numeric(y, _)) => RVal::Numeric(pick!(x, y), Attrs::default()),
        (RVal::Integer(x, _), RVal::Integer(y, _)) => RVal::Integer(pick!(x, y), Attrs::default()),
        (RVal::Logical(x, _), RVal::Logical(y, _)) => RVal::Logical(pick!(x, y), Attrs::default()),
        (RVal::Character(x, _), RVal::Character(y, _)) => RVal::Character(
            li.iter().zip(ri).map(|(l, r)| match l {
                Some(i) => x.get(*i).cloned().flatten(),
                None => r.and_then(|j| y.get(j)).cloned().flatten(),
            }).collect(),
            Attrs::default()),
        // Mixed or exotic key types (a factor joined against a character
        // column, say) fall back to the same normalised rendering the match
        // itself used, so the output cannot disagree with the join.
        _ => {
            let (sx, sy) = (key_strings(cx), key_strings(cy));
            RVal::Character(
                li.iter().zip(ri).map(|(l, r)| {
                    let v = match l { Some(i) => sx.get(*i), None => r.and_then(|j| sy.get(j)) };
                    match v { Some(v) if v != "NA" => Some(Arc::from(v.as_str())), _ => None }
                }).collect(),
                Attrs::default())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    #[test]
    fn na_omit_drops_na_from_numeric() {
        let v = RVal::Numeric(vec![Some(1.0), None, Some(2.0), None].into(), Attrs::default());
        let r = bi_na_omit(&[evarg(v)]).unwrap();
        match r {
            RVal::Numeric(v, _) => assert_eq!(v.len(), 2),
            _ => panic!(),
        }
    }

    fn num(v: &[f64]) -> RVal {
        RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default())
    }
    fn chr(v: &[&str]) -> RVal {
        RVal::Character(v.iter().map(|s| Some(Arc::from(*s))).collect(), Attrs::default())
    }
    fn df(cols: Vec<(&str, RVal)>) -> RVal {
        RVal::DataFrame(DataFrame {
            columns: cols.into_iter().map(|(n, v)| (Arc::from(n), v)).collect(),
            row_names: None,
        })
    }
    fn named(n: &str, v: RVal) -> EvalArg { EvalArg { name: Some(Arc::from(n)), value: v } }
    fn tru() -> RVal { RVal::Logical(vec![Some(true)].into(), Attrs::default()) }
    fn out(r: &RVal) -> &DataFrame {
        match r { RVal::DataFrame(d) => d, _ => panic!("merge did not return a data.frame") }
    }
    fn col_nums(d: &DataFrame, name: &str) -> Vec<Option<f64>> {
        match d.get_col(name).expect(name) {
            RVal::Numeric(v, _) => v.iter().copied().collect(),
            other => panic!("{name} is not numeric: {other:?}"),
        }
    }

    /// The regression that matters: an inner join is a SUBSET of a left
    /// join, so ignoring `all.x` returns believable data with too few rows
    /// and no error. Assert the count, not just the values.
    #[test]
    fn all_x_keeps_unmatched_left_rows_with_na() {
        let a = df(vec![("k", num(&[3.0, 1.0, 2.0])), ("v", num(&[30.0, 10.0, 20.0]))]);
        let b = df(vec![("k", num(&[2.0, 3.0])), ("w", num(&[200.0, 300.0]))]);
        let inner = bi_merge(&[evarg(a.clone()), evarg(b.clone()), named("by", chr(&["k"]))]).unwrap();
        assert_eq!(out(&inner).nrow(), 2, "inner join keeps only matched rows");

        let left = bi_merge(&[evarg(a), evarg(b), named("by", chr(&["k"])), named("all.x", tru())]).unwrap();
        let d = out(&left);
        assert_eq!(d.nrow(), 3, "all.x must keep the unmatched left row");
        // Sorted by key, so k = 1, 2, 3 and the unmatched one leads.
        assert_eq!(col_nums(d, "k"), vec![Some(1.0), Some(2.0), Some(3.0)]);
        assert_eq!(col_nums(d, "w"), vec![None, Some(200.0), Some(300.0)]);
    }

    /// A right-only row has no left key, so the key has to be taken from
    /// the right frame or the whole row is unusable.
    #[test]
    fn all_y_takes_the_key_from_the_right_frame() {
        let a = df(vec![("k", num(&[1.0])), ("v", num(&[10.0]))]);
        let b = df(vec![("k", num(&[1.0, 4.0])), ("w", num(&[100.0, 400.0]))]);
        let r = bi_merge(&[evarg(a), evarg(b), named("by", chr(&["k"])), named("all.y", tru())]).unwrap();
        let d = out(&r);
        assert_eq!(d.nrow(), 2);
        assert_eq!(col_nums(d, "k"), vec![Some(1.0), Some(4.0)]);
        assert_eq!(col_nums(d, "v"), vec![Some(10.0), None]);
    }

    /// Joining on one of two key columns gives a different answer AND
    /// leaves the second key duplicated with a suffix — which is what the
    /// single-column implementation used to do silently.
    #[test]
    fn composite_key_joins_on_every_by_column() {
        let a = df(vec![("k1", num(&[1.0, 1.0, 2.0])), ("k2", chr(&["x", "y", "x"])),
                        ("v", num(&[10.0, 11.0, 20.0]))]);
        let b = df(vec![("k1", num(&[1.0, 2.0, 2.0])), ("k2", chr(&["y", "x", "z"])),
                        ("w", num(&[1.0, 2.0, 3.0]))]);
        let r = bi_merge(&[evarg(a), evarg(b), named("by", chr(&["k1", "k2"]))]).unwrap();
        let d = out(&r);
        assert_eq!(d.nrow(), 2, "(1,y) and (2,x) match; (1,x) and (2,z) do not");
        assert_eq!(d.ncol(), 4, "k1, k2, v, w — k2 must not be duplicated");
        assert_eq!(col_nums(d, "v"), vec![Some(11.0), Some(20.0)]);
    }

    /// R suffixes BOTH sides of a collision, not just the second.
    #[test]
    fn colliding_non_key_names_get_x_and_y_suffixes() {
        let a = df(vec![("k", num(&[1.0])), ("v", chr(&["A"]))]);
        let b = df(vec![("k", num(&[1.0])), ("v", chr(&["X"]))]);
        let r = bi_merge(&[evarg(a), evarg(b), named("by", chr(&["k"]))]).unwrap();
        let d = out(&r);
        let names: Vec<&str> = d.columns.iter().map(|(n, _)| n.as_ref()).collect();
        assert_eq!(names, vec!["k", "v.x", "v.y"]);
    }

    /// Rebuilding columns via `format!` and re-parsing turned a character
    /// column of digits into numbers. Types must survive the join.
    #[test]
    fn character_column_of_digits_stays_character() {
        let a = df(vec![("k", num(&[1.0])), ("code", chr(&["007"]))]);
        let b = df(vec![("k", num(&[1.0])), ("w", num(&[9.0]))]);
        let r = bi_merge(&[evarg(a), evarg(b), named("by", chr(&["k"]))]).unwrap();
        match out(&r).get_col("code").unwrap() {
            RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("007")),
            other => panic!("code became {other:?} — the join must not retype columns"),
        }
    }

    /// Duplicate keys produce the cartesian product within each key group.
    #[test]
    fn duplicate_keys_produce_every_pairing() {
        let a = df(vec![("k", num(&[1.0, 1.0])), ("v", num(&[10.0, 11.0]))]);
        let b = df(vec![("k", num(&[1.0, 1.0])), ("w", num(&[20.0, 21.0]))]);
        let r = bi_merge(&[evarg(a), evarg(b), named("by", chr(&["k"]))]).unwrap();
        assert_eq!(out(&r).nrow(), 4);
    }

    #[test]
    fn complete_cases_marks_clean_rows() {
        let df = DataFrame {
            columns: vec![
                (Arc::from("x"), RVal::Numeric(vec![Some(1.0), None, Some(3.0)].into(), Attrs::default())),
                (Arc::from("y"), RVal::Numeric(vec![Some(2.0), Some(5.0), None].into(), Attrs::default())).into(),
            ],
            row_names: None,
        };
        let r = bi_complete_cases(&[evarg(RVal::DataFrame(df))]).unwrap();
        match r {
            RVal::Logical(v, _) => {
                let got: Vec<bool> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![true, false, false]);
            }
            _ => panic!(),
        }
    }
}
