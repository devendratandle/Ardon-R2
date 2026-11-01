//! R2 values as JSON for the IDE protocol (docs/IDE_PROTOCOL.md §4).
//!
//! Numbers keep full precision (Rust's shortest round-trip form). JSON
//! has no NA/NaN/Inf, so: NA → `null`, NaN → `"NaN"`, ±Inf → `"Inf"` /
//! `"-Inf"`. Long vectors are cut at [`MAX_ELEMS`] with `"truncated":
//! true` and the full `"length"`, so a request can never produce a
//! gigabyte reply; the IDE asks for pages (`view`) or summaries instead.

use r2_types::RVal;

/// Most elements sent for one vector (and in total for one data frame).
pub const MAX_ELEMS: usize = 100_000;

/// JSON string literal, quotes included.
pub fn string(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"), '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn real(x: Option<f64>) -> String {
    match x {
        None => "null".into(),
        Some(v) if v.is_nan() => "\"NaN\"".into(),
        Some(v) if v.is_infinite() => if v > 0.0 { "\"Inf\"".into() } else { "\"-Inf\"".into() },
        Some(v) => format!("{v:?}"),
    }
}

fn opt_str(x: Option<&str>) -> String { x.map(string).unwrap_or_else(|| "null".into()) }

/// `[a,b,…]` of the first MAX_ELEMS items, and whether it was cut.
fn array<T>(items: &[T], f: impl Fn(&T) -> String) -> (String, bool) {
    let body: Vec<String> = items.iter().take(MAX_ELEMS).map(f).collect();
    (format!("[{}]", body.join(",")), items.len() > MAX_ELEMS)
}

fn vector(ty: &str, n: usize, values: (String, bool), names: Option<&Vec<std::sync::Arc<str>>>) -> String {
    let mut o = format!("{{\"type\":\"{ty}\",\"length\":{n},\"values\":{}", values.0);
    if values.1 { o.push_str(",\"truncated\":true"); }
    if let Some(nm) = names {
        o.push_str(&format!(",\"names\":{}", array(nm, |s| string(s)).0));
    }
    o.push('}');
    o
}

/// One value as a JSON object with a `"type"` field.
pub fn value(v: &RVal) -> String {
    match v {
        RVal::Null => "{\"type\":\"NULL\"}".into(),
        RVal::Numeric(_, a) | RVal::Single(_, a) | RVal::Integer(_, a) => {
            let x = v.as_reals().unwrap_or_default();
            let ty = if matches!(v, RVal::Integer(..)) { "integer" } else { "numeric" };
            vector(ty, x.len(), array(&x, |e| real(*e)), a.names.as_ref())
        }
        RVal::Logical(..) => {
            let l = v.as_logicals().unwrap_or_default();
            vector("logical", l.len(), array(&l, |e| match e { Some(true) => "true".into(), Some(false) => "false".into(), None => "null".into() }), None)
        }
        RVal::Character(s, a) => vector("character", s.len(), array(s, |e| opt_str(e.as_deref())), a.names.as_ref()),
        RVal::Factor(f) => {
            let (codes, cut) = array(&f.codes, |c| c.map(|i| (i + 1).to_string()).unwrap_or_else(|| "null".into()));
            format!("{{\"type\":\"factor\",\"length\":{},\"levels\":{},\"codes\":{codes}{}}}", f.codes.len(),
                    array(&f.levels, |l| string(l)).0, if cut { ",\"truncated\":true" } else { "" })
        }
        RVal::List(items) => {
            let body: Vec<String> = items.iter().take(MAX_ELEMS)
                .map(|(n, x)| format!("{{\"name\":{},\"value\":{}}}", opt_str(n.as_deref()), value(x))).collect();
            format!("{{\"type\":\"list\",\"length\":{},\"items\":[{}]}}", items.len(), body.join(","))
        }
        RVal::TypeInstance(t) => {
            let mut keys: Vec<_> = t.fields.keys().collect();
            keys.sort();
            let body: Vec<String> = keys.iter().map(|k| format!("{}:{}", string(k), value(&t.fields[*k]))).collect();
            format!("{{\"type\":\"object\",\"class\":{},\"fields\":{{{}}}}}", string(&t.type_name), body.join(","))
        }
        RVal::DataFrame(df) => {
            // whole columns only while the frame is small; else names only
            let small = df.nrow() * df.ncol().max(1) <= MAX_ELEMS;
            let cols: Vec<String> = df.columns.iter()
                .map(|(n, c)| format!("{{\"name\":{},\"value\":{}}}", string(n),
                                      if small { value(c) } else { format!("{{\"type\":\"{}\"}}", c.type_name()) }))
                .collect();
            format!("{{\"type\":\"data.frame\",\"nrow\":{},\"ncol\":{},\"columns\":[{}]{}}}", df.nrow(), df.ncol(),
                    cols.join(","), if small { "" } else { ",\"truncated\":true" })
        }
        RVal::Matrix(m) => {
            let (data, cut) = array(&m.data, |x| real(Some(*x)));
            format!("{{\"type\":\"matrix\",\"nrow\":{},\"ncol\":{},\"data\":{data}{}}}", m.nrow, m.ncol,
                    if cut { ",\"truncated\":true" } else { "" })
        }
        other => format!("{{\"type\":{},\"text\":{}}}", string(other.type_name()), string(&format!("{other}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use r2_types::Attrs;

    #[test]
    fn numbers_keep_precision_and_mark_missing() {
        let v = RVal::Numeric(vec![Some(64.62), None, Some(f64::NAN), Some(f64::INFINITY), Some(0.1 + 0.2)].into(),
                              Attrs::default());
        assert_eq!(value(&v), "{\"type\":\"numeric\",\"length\":5,\"values\":[64.62,null,\"NaN\",\"Inf\",0.30000000000000004]}");
    }

    #[test]
    fn strings_are_escaped_and_lists_nest() {
        let s = RVal::Character(vec![Some("a\"b\n".into()), None], Attrs::default());
        assert_eq!(value(&s), "{\"type\":\"character\",\"length\":2,\"values\":[\"a\\\"b\\n\",null]}");
        let l = RVal::List(vec![(Some("x".into()), RVal::Null), (None, RVal::Null)]);
        assert_eq!(value(&l), "{\"type\":\"list\",\"length\":2,\"items\":[{\"name\":\"x\",\"value\":{\"type\":\"NULL\"}},{\"name\":null,\"value\":{\"type\":\"NULL\"}}]}");
    }

    #[test]
    fn long_vectors_are_cut() {
        let v = RVal::Numeric(vec![Some(1.0); MAX_ELEMS + 5].into(), Attrs::default());
        let j = value(&v);
        assert!(j.contains(&format!("\"length\":{}", MAX_ELEMS + 5)) && j.contains("\"truncated\":true"));
    }
}
