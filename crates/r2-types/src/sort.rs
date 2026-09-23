//! Orderings as R defines them: string collation, the element comparison
//! behind `sort` / `order` / `rank`, and the exact-equality key behind
//! `unique` / `duplicated`. Every builtin that orders or de-duplicates a
//! vector goes through here, so they agree with each other.

use crate::{Attrs, Character, Factor, Integer, Logical, RVal, Real};
use std::cmp::Ordering;
use std::sync::Arc;

/// Punctuation in the order R's `sort()` puts it on Windows (and in
/// English locales generally): all of it before digits, digits before
/// letters.
const SYMBOL_ORDER: &str = "'- !\"#$%&()*,./:;?@[\\]^_`{|}~+<=>";

/// A lowercase letter's base letter(s) and its accent class (0 = none):
/// `é` is `e` + acute, `œ` is `oe`, `ß` is `ss`.
fn base_letter(c: char) -> (&'static str, u8) {
    match c {
        'à' => ("a", 1), 'á' => ("a", 2), 'â' => ("a", 3), 'ã' => ("a", 4), 'ä' => ("a", 5), 'å' => ("a", 6),
        'æ' => ("ae", 9), 'ç' => ("c", 7),
        'è' => ("e", 1), 'é' => ("e", 2), 'ê' => ("e", 3), 'ë' => ("e", 5),
        'ì' => ("i", 1), 'í' => ("i", 2), 'î' => ("i", 3), 'ï' => ("i", 5),
        'ñ' => ("n", 4),
        'ò' => ("o", 1), 'ó' => ("o", 2), 'ô' => ("o", 3), 'õ' => ("o", 4), 'ö' => ("o", 5), 'ø' => ("o", 8),
        'œ' => ("oe", 9), 'ß' => ("ss", 9),
        'ù' => ("u", 1), 'ú' => ("u", 2), 'û' => ("u", 3), 'ü' => ("u", 5),
        'ý' => ("y", 2), 'ÿ' => ("y", 5),
        _ => ("", 0),
    }
}

/// A string's sort key under `str_collate`: compared field by field —
/// base characters, then accents, then case, then the raw text — so
/// sorting by the key is sorting by `str_collate`. Build it once per
/// element when sorting many strings.
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct CollationKey {
    base: Vec<u32>,
    accent: Vec<u8>,
    upper: Vec<bool>,
    raw: Box<str>,
}

pub fn collation_key(s: &str) -> CollationKey {
    let (mut base, mut accent, mut upper) = (Vec::new(), Vec::new(), Vec::new());
    for c in s.chars() {
        let up = c.is_uppercase();
        let lc = c.to_lowercase().next().unwrap_or(c);
        let (letters, acc) = base_letter(lc);
        let mut push = |b: char| {
            base.push(match SYMBOL_ORDER.find(b) {
                Some(i) => 1 + i as u32,
                None if b.is_ascii_digit() => 100 + b as u32,
                None if b.is_ascii_control() => 0,
                None => 1000 + b as u32,
            });
            accent.push(acc);
            upper.push(up);
        };
        if letters.is_empty() { push(lc) } else { letters.chars().for_each(&mut push) }
    }
    CollationKey { base, accent, upper, raw: s.into() }
}

/// R's string order in the locale R runs in on Windows and in English
/// UTF-8 locales (`English_*.utf8`, `en_US.UTF-8`) — not the C locale's
/// byte order. Compared level by level: first the base characters
/// (punctuation < digits < letters, letters case- and accent-blind), then
/// accents (`e` before `é`), then case (lowercase first):
/// `"a" "A" "b" "B"`, `"naive" "Naive" "naïve"`, `"a-c" "ab"`. Identical
/// strings are the only ones that compare equal.
pub fn str_collate(a: &str, b: &str) -> Ordering {
    if a == b { return Ordering::Equal; }
    collation_key(a).cmp(&collation_key(b))
}

/// One atomic vector seen as sortable elements. NA — and for doubles NaN
/// — is "missing": R's sort drops it and order puts it last.
pub enum SortCol<'a> {
    Num(&'a [Real]),
    Int(&'a [Integer]),
    Lgl(&'a [Logical]),
    /// Strings carry their collation keys, built once.
    Str(&'a [Character], Vec<Option<CollationKey>>),
    /// A factor sorts by its codes, i.e. in level order.
    Fac(&'a [Option<u32>]),
}

impl<'a> SortCol<'a> {
    pub fn new(v: &'a RVal) -> Option<SortCol<'a>> {
        Some(match v {
            RVal::Numeric(x, _) => SortCol::Num(x),
            RVal::Integer(x, _) => SortCol::Int(x),
            RVal::Logical(x, _) => SortCol::Lgl(x),
            RVal::Character(x, _) => SortCol::Str(x, x.iter().map(|e| e.as_deref().map(collation_key)).collect()),
            RVal::Factor(f) => SortCol::Fac(&f.codes),
            _ => return None,
        })
    }

    pub fn len(&self) -> usize {
        match self {
            SortCol::Num(x) => x.len(), SortCol::Int(x) => x.len(), SortCol::Lgl(x) => x.len(),
            SortCol::Str(x, _) => x.len(), SortCol::Fac(x) => x.len(),
        }
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn is_na(&self, i: usize) -> bool {
        match self {
            SortCol::Num(x) => x[i].map_or(true, f64::is_nan),
            SortCol::Int(x) => x[i].is_none(),
            SortCol::Lgl(x) => x[i].is_none(),
            SortCol::Str(x, _) => x[i].is_none(),
            SortCol::Fac(x) => x[i].is_none(),
        }
    }

    /// Compare two non-missing elements.
    pub fn cmp(&self, i: usize, j: usize) -> Ordering {
        match self {
            SortCol::Num(x) => x[i].partial_cmp(&x[j]).unwrap_or(Ordering::Equal),
            SortCol::Int(x) => x[i].cmp(&x[j]),
            SortCol::Lgl(x) => x[i].cmp(&x[j]),
            SortCol::Str(_, keys) => keys[i].cmp(&keys[j]),
            SortCol::Fac(x) => x[i].cmp(&x[j]),
        }
    }
}

/// Where missing values go: R's `na.last = TRUE / FALSE / NA`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum NaLast { Last, First, Remove }

/// `order(k1, k2, ..., decreasing, na.last)` as 0-based indices: a stable
/// sort by the first key, ties broken by the next. Missing values are
/// placed by `na_last` whatever `decreasing` says, as in R.
pub fn order_indices(keys: &[SortCol], decreasing: bool, na_last: NaLast) -> Vec<usize> {
    let n = keys.first().map_or(0, SortCol::len);
    let mut idx: Vec<usize> = (0..n).collect();
    if na_last == NaLast::Remove {
        idx.retain(|&i| !keys.iter().any(|k| k.is_na(i)));
    }
    idx.sort_by(|&i, &j| {
        for k in keys {
            let ord = match (k.is_na(i), k.is_na(j)) {
                (true, true) => Ordering::Equal,
                (true, false) => if na_last == NaLast::First { Ordering::Less } else { Ordering::Greater },
                (false, true) => if na_last == NaLast::First { Ordering::Greater } else { Ordering::Less },
                (false, false) => if decreasing { k.cmp(j, i) } else { k.cmp(i, j) },
            };
            if ord != Ordering::Equal { return ord; }
        }
        Ordering::Equal
    });
    idx
}

/// `x[idx]` for an atomic vector or factor, keeping its type and
/// reordering its names. Other attributes are dropped, as R's `sort`
/// and `unique` do.
pub fn take(v: &RVal, idx: &[usize]) -> RVal {
    let names = |at: &Attrs| {
        let mut out = Attrs::default();
        out.names = at.names.as_ref().map(|ns| idx.iter().map(|&i| ns[i].clone()).collect());
        out
    };
    match v {
        RVal::Numeric(x, at) => RVal::Numeric(idx.iter().map(|&i| x[i]).collect::<Vec<Real>>().into(), names(at)),
        RVal::Integer(x, at) => RVal::Integer(idx.iter().map(|&i| x[i]).collect::<Vec<Integer>>().into(), names(at)),
        RVal::Logical(x, at) => RVal::Logical(idx.iter().map(|&i| x[i]).collect::<Vec<Logical>>().into(), names(at)),
        RVal::Character(x, at) => RVal::Character(idx.iter().map(|&i| x[i].clone()).collect(), names(at)),
        RVal::Factor(f) => RVal::Factor(Factor {
            codes: idx.iter().map(|&i| f.codes[i]).collect(), levels: f.levels.clone(), ordered: f.ordered,
        }),
        other => other.clone(),
    }
}

/// Exact identity of one element for `unique` / `duplicated` / `match`:
/// NA and NaN are distinct values, `0 == -0`, and doubles compare by value
/// (R uses no tolerance).
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum ElemKey {
    Na,
    NaN,
    Num(u64),
    Int(i32),
    Lgl(bool),
    Str(Arc<str>),
}

/// The `ElemKey` of every element, or `None` for a non-atomic value.
pub fn elem_keys(v: &RVal) -> Option<Vec<ElemKey>> {
    Some(match v {
        RVal::Numeric(x, _) => x.iter().map(|e| match e {
            None => ElemKey::Na,
            Some(d) if d.is_nan() => ElemKey::NaN,
            Some(d) => ElemKey::Num(if *d == 0.0 { 0 } else { d.to_bits() }),
        }).collect(),
        RVal::Integer(x, _) => x.iter().map(|e| e.map_or(ElemKey::Na, ElemKey::Int)).collect(),
        RVal::Logical(x, _) => x.iter().map(|e| e.map_or(ElemKey::Na, ElemKey::Lgl)).collect(),
        RVal::Character(x, _) => x.iter().map(|e| e.clone().map_or(ElemKey::Na, ElemKey::Str)).collect(),
        RVal::Factor(f) => f.codes.iter().map(|c| c.map_or(ElemKey::Na, |c| ElemKey::Int(c as i32))).collect(),
        _ => return None,
    })
}

/// `duplicated(x)`: whether each element equals an earlier one.
pub fn duplicated_mask(keys: &[ElemKey]) -> Vec<bool> {
    let mut seen = std::collections::HashSet::with_capacity(keys.len());
    keys.iter().map(|k| !seen.insert(k)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collation_matches_r_on_windows() {
        // Each expected order is R 4.5.3's sort() on Windows.
        let sorted = |v: &[&'static str]| { let mut v = v.to_vec(); v.sort_by(|a, b| str_collate(a, b)); v };
        assert_eq!(sorted(&["B", "a", "A", "b", "ab", "Aa"]), ["a", "A", "Aa", "ab", "b", "B"]);
        assert_eq!(sorted(&["item_3", "Item4", "item 1", "item-2"]), ["item-2", "item 1", "item_3", "Item4"]);
        assert_eq!(sorted(&["a-c", "ab", "a-b", "A-b", "aB", "Ab"]), ["a-b", "A-b", "a-c", "ab", "aB", "Ab"]);
        assert_eq!(sorted(&["naïve", "Naive", "nb", "ñ", "n", "naive"]), ["n", "ñ", "naive", "Naive", "naïve", "nb"]);
        assert_eq!(sorted(&["œ", "oe", "ø", "o", "st", "ß", "ss"]), ["o", "ø", "oe", "œ", "ss", "ß", "st"]);
    }

    #[test]
    fn order_puts_missing_last_even_when_decreasing() {
        let x = RVal::Numeric(vec![Some(2.0), None, Some(f64::NAN), Some(3.0), Some(1.0)].into(), Attrs::default());
        let k = [SortCol::new(&x).unwrap()];
        assert_eq!(order_indices(&k, false, NaLast::Last), [4, 0, 3, 1, 2]);
        assert_eq!(order_indices(&k, true, NaLast::Last), [3, 0, 4, 1, 2]);
        assert_eq!(order_indices(&k, false, NaLast::Remove), [4, 0, 3]);
    }

    #[test]
    fn keys_separate_na_from_nan_and_merge_signed_zero() {
        let x = RVal::Numeric(vec![None, Some(f64::NAN), Some(0.0), Some(-0.0), None].into(), Attrs::default());
        assert_eq!(duplicated_mask(&elem_keys(&x).unwrap()), [false, false, false, true, true]);
    }
}
