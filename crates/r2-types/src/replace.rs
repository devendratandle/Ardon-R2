//! Element replacement, `x[i] <- value`, with R's rules: logical,
//! negative, positive and name subscripts; assigning past the end extends
//! with NA; the result takes the wider of the two types
//! (logical < integer < double < character); a list takes elements (and
//! drops them for `NULL`); a factor takes labels that are its levels.

use crate::{factor_labels, rval_length, Attrs, Character, ErrKind, Factor, Integer, Logical, R2Err, RVal, Real};
use std::sync::Arc;

fn rerr(msg: impl Into<String>) -> R2Err { R2Err { msg: msg.into(), kind: ErrKind::Runtime } }

/// The positions `x[idx] <- ...` writes, 0-based, for an `x` of length
/// `n` with `names`. A position may be `>= n` (the vector grows); a name
/// that `x` lacks gets a new position, returned with the name to give it.
pub fn assign_positions(idx: &RVal, n: usize, names: Option<&[Arc<str>]>)
    -> Result<(Vec<usize>, Vec<(usize, Arc<str>)>), R2Err>
{
    let mut new_names: Vec<(usize, Arc<str>)> = Vec::new();
    let pos = match idx {
        RVal::Null => Vec::new(),
        RVal::Logical(mask, _) => {
            if mask.is_empty() { return Ok((Vec::new(), new_names)); }
            (0..n.max(mask.len())).filter(|i| mask[i % mask.len()] == Some(true)).collect()
        }
        RVal::Character(keys, _) => {
            let mut out = Vec::with_capacity(keys.len());
            for k in keys {
                let Some(k) = k else { continue };
                let found = names.and_then(|ns| ns.iter().position(|nm| nm == k))
                    .or_else(|| new_names.iter().find(|(_, nm)| nm == k).map(|(p, _)| *p));
                out.push(found.unwrap_or_else(|| {
                    let p = n + new_names.len();
                    new_names.push((p, k.clone()));
                    p
                }));
            }
            out
        }
        other => {
            let raw: Vec<i64> = match other {
                RVal::Factor(f) => f.codes.iter().flatten().map(|c| *c as i64 + 1).collect(),
                _ => other.as_reals()?.into_iter().flatten().filter(|p| !p.is_nan()).map(|p| p.trunc() as i64).collect(),
            };
            let neg = raw.iter().any(|&p| p < 0);
            if neg && raw.iter().any(|&p| p > 0) {
                return Err(rerr("can't mix positive and negative subscripts"));
            }
            if neg {
                let drop: std::collections::HashSet<usize> = raw.iter().map(|&p| (-p - 1) as usize).collect();
                (0..n).filter(|i| !drop.contains(i)).collect()
            } else {
                raw.into_iter().filter(|&p| p > 0).map(|p| p as usize - 1).collect()
            }
        }
    };
    Ok((pos, new_names))
}

/// An atomic vector in one of R's four replacement types, ordered by width.
enum Atoms { Lgl(Vec<Logical>), Int(Vec<Integer>), Num(Vec<Real>), Chr(Vec<Character>) }

impl Atoms {
    fn rank(&self) -> u8 { match self { Atoms::Lgl(_) => 0, Atoms::Int(_) => 1, Atoms::Num(_) => 2, Atoms::Chr(_) => 3 } }

    fn of(v: &RVal) -> Option<Atoms> {
        Some(match v {
            RVal::Logical(x, _) => Atoms::Lgl(x.to_vec()),
            RVal::Integer(x, _) => Atoms::Int(x.to_vec()),
            RVal::Numeric(..) | RVal::Single(..) => Atoms::Num(v.as_reals().ok()?),
            RVal::Character(x, _) => Atoms::Chr(x.clone()),
            // A factor value contributes its codes, as R's `[<-` does.
            RVal::Factor(f) => Atoms::Int(f.codes.iter().map(|c| c.map(|c| c as i32 + 1)).collect()),
            _ => return None,
        })
    }

    /// Widen to `rank` (never narrows).
    fn widen(self, rank: u8) -> Atoms {
        if rank <= self.rank() { return self; }
        match (self, rank) {
            (Atoms::Lgl(v), 1) => Atoms::Int(v.into_iter().map(|b| b.map(i32::from)).collect()),
            (Atoms::Lgl(v), 2) => Atoms::Num(v.into_iter().map(|b| b.map(|b| if b { 1.0 } else { 0.0 })).collect()),
            (Atoms::Int(v), 2) => Atoms::Num(v.into_iter().map(|i| i.map(f64::from)).collect()),
            (v, _) => {
                let as_rval = v.into_rval(Attrs::default());
                Atoms::Chr(factor_labels(&as_rval).map(|(l, _)| l).unwrap_or_default())
            }
        }
    }

    fn len(&self) -> usize {
        match self { Atoms::Lgl(v) => v.len(), Atoms::Int(v) => v.len(), Atoms::Num(v) => v.len(), Atoms::Chr(v) => v.len() }
    }

    /// `self[pos[k]] <- val[k %% len(val)]`, growing with NA as needed.
    fn put(&mut self, pos: &[usize], val: &Atoms) {
        let top = pos.iter().map(|p| p + 1).max().unwrap_or(0);
        macro_rules! put { ($dst:expr, $src:expr, $na:expr) => {{
            if $dst.len() < top { $dst.resize(top, $na); }
            for (k, &p) in pos.iter().enumerate() { $dst[p] = $src[k % $src.len()].clone(); }
        }} }
        match (self, val) {
            (Atoms::Lgl(d), Atoms::Lgl(s)) => put!(d, s, None),
            (Atoms::Int(d), Atoms::Int(s)) => put!(d, s, None),
            (Atoms::Num(d), Atoms::Num(s)) => put!(d, s, None),
            (Atoms::Chr(d), Atoms::Chr(s)) => put!(d, s, None),
            _ => unreachable!("put() operands are widened to one type first"),
        }
    }

    fn into_rval(self, attrs: Attrs) -> RVal {
        match self {
            Atoms::Lgl(v) => RVal::Logical(v.into(), attrs),
            Atoms::Int(v) => RVal::Integer(v.into(), attrs),
            Atoms::Num(v) => RVal::Numeric(v.into(), attrs),
            Atoms::Chr(v) => RVal::Character(v, attrs),
        }
    }
}

/// `obj[idx] <- val` for a vector, factor, list or `NULL`; the new value.
pub fn replace_elems(obj: &RVal, idx: &RVal, val: &RVal) -> Result<RVal, R2Err> {
    let n = rval_length(obj);
    let names: Option<Vec<Arc<str>>> = match obj {
        RVal::Numeric(_, at) | RVal::Integer(_, at) | RVal::Logical(_, at) | RVal::Character(_, at) => at.names.clone(),
        RVal::List(items) if items.iter().any(|(k, _)| k.is_some()) =>
            Some(items.iter().map(|(k, _)| k.clone().unwrap_or_else(|| Arc::from(""))).collect()),
        _ => None,
    };
    let (pos, new_names) = assign_positions(idx, n, names.as_deref())?;
    let vlen = rval_length(val);
    if pos.is_empty() { return Ok(obj.clone()); }

    match obj {
        RVal::List(items) => {
            let mut items = items.clone();
            if matches!(val, RVal::Null) {
                // `l[i] <- NULL` removes those elements.
                let drop: std::collections::HashSet<usize> = pos.into_iter().collect();
                return Ok(RVal::List(items.into_iter().enumerate().filter(|(i, _)| !drop.contains(i)).map(|(_, e)| e).collect()));
            }
            if vlen == 0 { return Err(rerr("replacement has length zero")); }
            let top = pos.iter().map(|p| p + 1).max().unwrap_or(0);
            while items.len() < top { items.push((None, RVal::Null)); }
            for (k, &p) in pos.iter().enumerate() {
                items[p].1 = match val {
                    RVal::List(vs) => vs[k % vs.len()].1.clone(),
                    other => crate::take(other, &[k % vlen]),
                };
            }
            for (p, nm) in new_names { items[p].0 = Some(nm); }
            return Ok(RVal::List(items));
        }
        RVal::Factor(f) => {
            if vlen == 0 { return Err(rerr("replacement has length zero")); }
            // Labels that are levels take that level's code; others are NA
            // (R warns "invalid factor level, NA generated").
            let labels = factor_labels(val).map(|(l, _)| l).ok_or_else(|| rerr(format!("cannot assign {} into a factor", val.type_name())))?;
            let coded = Factor::with_levels(&labels, f.levels.clone(), f.ordered);
            let mut codes = f.codes.clone();
            let top = pos.iter().map(|p| p + 1).max().unwrap_or(0);
            if codes.len() < top { codes.resize(top, None); }
            for (k, &p) in pos.iter().enumerate() { codes[p] = coded.codes[k % vlen]; }
            return Ok(RVal::Factor(Factor { codes, levels: f.levels.clone(), ordered: f.ordered }));
        }
        _ => {}
    }

    let empty = Atoms::Lgl(Vec::new());
    let dst = if matches!(obj, RVal::Null) { empty } else {
        Atoms::of(obj).ok_or_else(|| rerr(format!("cannot assign by index to {}", obj.type_name())))?
    };
    let src = Atoms::of(val).ok_or_else(|| rerr(format!("cannot assign {} into a vector", val.type_name())))?;
    if src.len() == 0 { return Err(rerr("replacement has length zero")); }
    let rank = dst.rank().max(src.rank());
    let widened = rank != dst.rank();
    let (mut dst, src) = (dst.widen(rank), src.widen(rank));
    dst.put(&pos, &src);

    // Keep the object's attributes (a Date stays a Date), except a class
    // that no longer fits the new type; grow the names with "".
    let mut attrs = match obj {
        RVal::Numeric(_, at) | RVal::Integer(_, at) | RVal::Logical(_, at) | RVal::Character(_, at) => at.clone(),
        _ => Attrs::default(),
    };
    if widened { attrs.class = None; }
    if names.is_some() || !new_names.is_empty() {
        let mut ns = names.unwrap_or_default();
        ns.resize(dst.len(), Arc::from(""));
        for (p, nm) in new_names { ns[p] = nm; }
        attrs.names = Some(ns);
    }
    Ok(dst.into_rval(attrs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num(v: &[Option<f64>]) -> RVal { RVal::Numeric(v.to_vec().into(), Attrs::default()) }
    fn lgl(v: &[Option<bool>]) -> RVal { RVal::Logical(v.to_vec().into(), Attrs::default()) }
    fn chr(v: &[&str]) -> RVal { RVal::Character(v.iter().map(|s| Some(Arc::from(*s))).collect(), Attrs::default()) }

    #[test]
    fn logical_mask_negative_and_growth() {
        let x = num(&[Some(1.0), Some(5.0), Some(3.0)]);
        let r = replace_elems(&x, &lgl(&[Some(false), Some(true), Some(true)]), &num(&[Some(0.0)])).unwrap();
        assert_eq!(r.as_reals().unwrap(), [Some(1.0), Some(0.0), Some(0.0)]);
        let r = replace_elems(&x, &num(&[Some(-1.0)]), &num(&[Some(9.0)])).unwrap();
        assert_eq!(r.as_reals().unwrap(), [Some(1.0), Some(9.0), Some(9.0)]);
        let r = replace_elems(&x, &num(&[Some(5.0)]), &num(&[Some(7.0)])).unwrap();
        assert_eq!(r.as_reals().unwrap(), [Some(1.0), Some(5.0), Some(3.0), None, Some(7.0)]);
    }

    #[test]
    fn na_into_character_is_na_and_types_widen() {
        let r = replace_elems(&chr(&["a", "b"]), &num(&[Some(1.0)]), &lgl(&[None])).unwrap();
        assert!(matches!(r, RVal::Character(ref v, _) if v[0].is_none()));
        let r = replace_elems(&lgl(&[None, None]), &num(&[Some(1.0)]), &num(&[Some(2.5)])).unwrap();
        assert!(matches!(r, RVal::Numeric(..)));
        let r = replace_elems(&num(&[Some(1.0)]), &num(&[Some(2.0)]), &chr(&["z"])).unwrap();
        assert!(matches!(r, RVal::Character(ref v, _) if v[0].as_deref() == Some("1")));
    }

    #[test]
    fn names_select_and_extend() {
        let mut at = Attrs::default();
        at.names = Some(vec![Arc::from("a"), Arc::from("b")]);
        let x = RVal::Numeric(vec![Some(1.0), Some(2.0)].into(), at);
        let r = replace_elems(&x, &chr(&["b", "c"]), &num(&[Some(7.0), Some(8.0)])).unwrap();
        match r {
            RVal::Numeric(v, at) => {
                assert_eq!(&v[..], [Some(1.0), Some(7.0), Some(8.0)]);
                assert_eq!(at.names.unwrap().iter().map(|s| s.as_ref()).collect::<Vec<_>>(), ["a", "b", "c"]);
            }
            _ => panic!(),
        }
    }
}
