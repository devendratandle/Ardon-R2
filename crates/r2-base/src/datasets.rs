//! R's `datasets` package: all 108 objects, embedded in the binary.
//!
//! `datasets/datasets.r2o` is written by R itself
//! (`datasets/export_datasets.R`), so every value is R's own — nothing is
//! transcribed. The file holds R's objects with their attributes; this
//! module decodes one on request and maps it to the nearest R2 value:
//!
//! - a data frame → `DataFrame` (character row names kept);
//! - a factor → `Factor` (`ordered` from its class);
//! - a numeric 2-d array (matrix, 2-d table, multivariate `ts`) → `Matrix`
//!   with its dimnames, NA as NaN (R2's matrix has no NA);
//! - any other vector → the vector, with `names`, `dim` and the full class
//!   vector set, and every other attribute (`tsp`, `dimnames`, a `dist`
//!   object's `Size`/`Labels`, …) kept as a custom attribute, so
//!   `attr(AirPassengers, "tsp")` answers as in R;
//! - a list → a named list.

use r2_types::*;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

static BYTES: &[u8] = include_bytes!("../datasets/datasets.r2o");

/// One dataset in the index: its name, R's one-line title, and where its
/// object lies in the file.
pub struct Entry {
    pub name: &'static str,
    pub title: &'static str,
    start: usize,
    end: usize,
}

/// Every dataset, in R's (alphabetical) order. The index is read once.
pub fn index() -> &'static [Entry] {
    static IDX: OnceLock<Vec<Entry>> = OnceLock::new();
    IDX.get_or_init(|| read_index(BYTES).expect("datasets.r2o: corrupt index"))
}

/// Decode the dataset `name`, or `None` if R has no such dataset.
pub fn get(name: &str) -> Option<RVal> {
    let e = index().iter().find(|e| e.name == name)?;
    let mut c = Cur { b: BYTES, p: e.start };
    let obj = read_obj(&mut c).expect("datasets.r2o: corrupt object");
    debug_assert_eq!(c.p, e.end);
    Some(to_rval(&obj))
}

/// Bind every dataset into `env` (the base environment).
pub fn register_all(env: &mut HashMap<Arc<str>, RVal>) {
    for e in index() {
        if let Some(v) = get(e.name) { env.insert(Arc::from(e.name), v); }
    }
}

// ── the file ─────────────────────────────────────────────────────────

struct Cur { b: &'static [u8], p: usize }

impl Cur {
    fn take(&mut self, n: usize) -> Option<&'static [u8]> {
        let s = self.b.get(self.p..self.p + n)?;
        self.p += n;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> { Some(self.take(1)?[0]) }
    fn u32(&mut self) -> Option<u32> { Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?)) }
    fn i32(&mut self) -> Option<i32> { Some(i32::from_le_bytes(self.take(4)?.try_into().ok()?)) }
    fn f64(&mut self) -> Option<f64> { Some(f64::from_le_bytes(self.take(8)?.try_into().ok()?)) }
    fn str(&mut self) -> Option<&'static str> {
        let n = self.u32()? as usize;
        std::str::from_utf8(self.take(n)?).ok()
    }
}

fn read_index(b: &'static [u8]) -> Option<Vec<Entry>> {
    let mut c = Cur { b, p: 0 };
    if c.take(4)? != b"R2O1" { return None; }
    let n = c.u32()? as usize;
    let mut raw = Vec::with_capacity(n);
    for _ in 0..n {
        let (name, title) = (c.str()?, c.str()?);
        let (off, len) = (c.u32()? as usize, c.u32()? as usize);
        raw.push((name, title, off, len));
    }
    let base = c.p;
    Some(raw.into_iter()
        .map(|(name, title, off, len)| Entry { name, title, start: base + off, end: base + off + len })
        .collect())
}

/// An R object as stored: its values and its attributes.
enum Vals {
    Null,
    Logical(Vec<Option<bool>>),
    Integer(Vec<Option<i32>>),
    Double(Vec<Option<f64>>),
    Character(Vec<Option<&'static str>>),
    List(Vec<Obj>),
}

struct Obj { vals: Vals, attrs: Vec<(&'static str, Obj)> }

impl Obj {
    fn attr(&self, name: &str) -> Option<&Obj> {
        self.attrs.iter().find(|(n, _)| *n == name).map(|(_, o)| o)
    }
    fn strings(&self) -> Vec<Option<&'static str>> {
        match &self.vals { Vals::Character(v) => v.clone(), _ => Vec::new() }
    }
    fn classes(&self) -> Vec<&'static str> {
        self.attr("class").map(|c| c.strings().into_iter().flatten().collect()).unwrap_or_default()
    }
}

/// R's `NA_real_` is the NaN whose low word is 1954; any other NaN is NaN.
fn r_na(x: f64) -> bool { x.is_nan() && x.to_bits() & 0xFFFF_FFFF == 1954 }

fn read_obj(c: &mut Cur) -> Option<Obj> {
    let ty = c.u8()?;
    let n = c.u32()? as usize;
    let vals = match ty {
        0 => Vals::Null,
        1 => Vals::Logical((0..n).map(|_| c.i32().map(|v| (v != i32::MIN).then_some(v != 0))).collect::<Option<_>>()?),
        2 => Vals::Integer((0..n).map(|_| c.i32().map(|v| (v != i32::MIN).then_some(v))).collect::<Option<_>>()?),
        3 => Vals::Double((0..n).map(|_| c.f64().map(|v| (!r_na(v)).then_some(v))).collect::<Option<_>>()?),
        4 => Vals::Character((0..n).map(|_| {
            let len = c.i32()?;
            if len < 0 { return Some(None); }
            Some(Some(std::str::from_utf8(c.take(len as usize)?).ok()?))
        }).collect::<Option<_>>()?),
        5 => Vals::List((0..n).map(|_| read_obj(c)).collect::<Option<_>>()?),
        _ => return None,
    };
    let na = c.u32()? as usize;
    let attrs = (0..na).map(|_| Some((c.str()?, read_obj(c)?))).collect::<Option<_>>()?;
    Some(Obj { vals, attrs })
}

// ── R object → R2 value ──────────────────────────────────────────────

fn arc(s: &str) -> Arc<str> { Arc::from(s) }

/// Names as R2 holds them: an NA name reads "NA", as R prints it.
fn names_of(o: &Obj) -> Vec<Arc<str>> {
    o.strings().into_iter().map(|s| arc(s.unwrap_or("NA"))).collect()
}

fn to_rval(o: &Obj) -> RVal {
    let classes = o.classes();
    if classes.contains(&"data.frame") { return data_frame(o); }
    if classes.contains(&"factor") { return factor(o, classes.contains(&"ordered")); }
    if let Some(m) = matrix(o) { return m; }
    let mut at = Attrs::default();
    for (name, a) in &o.attrs {
        match *name {
            "names" => at.names = Some(names_of(a)),
            "class" => at.set_classes(classes.iter().map(|c| arc(c)).collect()),
            "dim" => at.dim = match &a.vals {
                Vals::Integer(d) => Some(d.iter().map(|x| x.unwrap_or(0) as usize).collect()),
                _ => None,
            },
            other => { at.custom.insert(arc(other), to_rval(a)); }
        }
    }
    match &o.vals {
        Vals::Null => RVal::Null,
        Vals::Logical(v) => RVal::Logical(v.clone().into(), at),
        Vals::Integer(v) => RVal::Integer(v.clone().into(), at),
        Vals::Double(v) => RVal::Numeric(v.clone().into(), at),
        Vals::Character(v) => RVal::Character(v.iter().map(|s| s.map(arc)).collect(), at),
        Vals::List(items) => {
            let names = o.attr("names").map(names_of);
            let items = items.iter().enumerate()
                .map(|(i, e)| (names.as_ref().and_then(|n| n.get(i).cloned()), to_rval(e)))
                .collect();
            at.names = None;
            RVal::List(items, at)
        }
    }
}

fn data_frame(o: &Obj) -> RVal {
    let names = o.attr("names").map(names_of).unwrap_or_default();
    let cols = match &o.vals { Vals::List(c) => c, _ => return RVal::Null };
    let columns = cols.iter().zip(names).map(|(c, n)| (n, to_rval(c))).collect();
    // Automatic row names are stored compactly as integers; only
    // character ones (mtcars' car names) are real names.
    let row_names = o.attr("row.names").and_then(|r| match &r.vals {
        Vals::Character(v) => Some(v.iter().map(|s| arc(s.unwrap_or("NA"))).collect()),
        _ => None,
    });
    RVal::DataFrame(DataFrame { columns, row_names })
}

fn factor(o: &Obj, ordered: bool) -> RVal {
    let codes = match &o.vals {
        Vals::Integer(v) => v.iter().map(|c| c.map(|k| (k - 1) as u32)).collect(),
        _ => Vec::new(),
    };
    let levels = o.attr("levels").map(names_of).unwrap_or_default();
    RVal::Factor(Factor { codes, levels, ordered })
}

/// A numeric array with exactly two dimensions becomes a `Matrix`.
fn matrix(o: &Obj) -> Option<RVal> {
    let dim = match &o.attr("dim")?.vals { Vals::Integer(d) if d.len() == 2 => d.clone(), _ => return None };
    let data: Vec<f64> = match &o.vals {
        Vals::Double(v) => v.iter().map(|x| x.unwrap_or(f64::NAN)).collect(),
        Vals::Integer(v) => v.iter().map(|x| x.map_or(f64::NAN, f64::from)).collect(),
        _ => return None,
    };
    let mut m = Matrix::new(data, dim[0]? as usize, dim[1]? as usize);
    if let Some(Obj { vals: Vals::List(dn), .. }) = o.attr("dimnames") {
        let side = |i: usize| dn.get(i).filter(|d| !matches!(d.vals, Vals::Null)).map(names_of);
        m.row_names = side(0);
        m.col_names = side(1);
    }
    Some(RVal::Matrix(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every dataset in the index decodes, and its object ends exactly
    /// where the index says it does.
    #[test]
    fn every_dataset_decodes_to_its_recorded_extent() {
        assert!(index().len() >= 100);
        for e in index() {
            let mut c = Cur { b: BYTES, p: e.start };
            assert!(read_obj(&mut c).is_some(), "{} does not decode", e.name);
            assert_eq!(c.p, e.end, "{} overruns its extent", e.name);
            assert!(get(e.name).is_some());
        }
    }

    /// A data frame's columns all have its row count, and every factor
    /// code points at one of its levels.
    #[test]
    fn frames_are_rectangular_and_factor_codes_are_in_range() {
        fn check_factor(name: &str, f: &Factor) {
            assert!(f.codes.iter().flatten().all(|&k| (k as usize) < f.levels.len()), "{name}: factor code out of range");
        }
        for e in index() {
            match get(e.name).unwrap() {
                RVal::DataFrame(df) => {
                    let n = df.nrow();
                    for (cn, col) in &df.columns {
                        let len = match col {
                            RVal::Factor(f) => { check_factor(e.name, f); f.codes.len() }
                            other => rval_length(other),
                        };
                        assert_eq!(len, n, "{}${} has {} rows, frame has {}", e.name, cn, len, n);
                    }
                    if let Some(rn) = &df.row_names { assert_eq!(rn.len(), n, "{} row names", e.name); }
                }
                RVal::Factor(f) => check_factor(e.name, &f),
                RVal::Matrix(m) => assert_eq!(m.data.len(), m.nrow * m.ncol, "{} matrix shape", e.name),
                _ => {}
            }
        }
    }
}
