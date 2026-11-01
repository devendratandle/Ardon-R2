//! The IDE's Environment pane rows (`env` request, docs/IDE_PROTOCOL.md):
//! one per variable in the global environment, grouped as RStudio groups
//! them (Data, Values, Functions), with a one-line summary in the spirit
//! of `str()`. R2 formats its own values; the IDE only displays them:
//!
//! ```text
//!   df    6 obs. of 2 variables
//!   x     num [1:6] 4 8 15 16 23 42
//!   f     function (v)
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use r2_types::{format_reals, RVal};

/// Longest summary kept, in characters.
const SUMMARY_MAX: usize = 70;
/// How many leading elements a vector summary shows.
const HEAD: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Group { Data, Values, Functions }

impl Group {
    pub fn label(self) -> &'static str {
        match self { Group::Data => "Data", Group::Values => "Values", Group::Functions => "Functions" }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnvRow {
    pub name: String,
    pub group: Group,
    pub summary: String,
    /// The data viewer can show it (data frames, matrices, vectors).
    pub viewable: bool,
}

/// Rows for every visible binding, grouped, then sorted by name as `ls()`
/// sorts them. Names starting with `.` are hidden, as in R.
pub fn env_rows(bindings: &HashMap<Arc<str>, RVal>) -> Vec<EnvRow> {
    let mut rows: Vec<EnvRow> = bindings.iter()
        .filter(|(k, _)| !k.starts_with('.'))
        .map(|(k, v)| row(k, v))
        .collect();
    rows.sort_by(|a, b| a.group.cmp(&b.group)
        .then_with(|| r2_types::collation_key(&a.name).cmp(&r2_types::collation_key(&b.name))));
    rows
}

fn row(name: &str, v: &RVal) -> EnvRow {
    let (group, summary, viewable) = match v {
        RVal::DataFrame(df) => (Group::Data,
            format!("{} obs. of {} variable{}", df.nrow(), df.ncol(), if df.ncol() == 1 { "" } else { "s" }), true),
        RVal::Matrix(m) => (Group::Data,
            format!("num [1:{}, 1:{}] {}", m.nrow, m.ncol, nums(&m.data.iter().map(|x| Some(*x)).collect::<Vec<_>>())), true),
        RVal::Closure(c) => (Group::Functions,
            format!("function ({})", c.params.iter().map(|p| if p.dots { "...".to_string() } else { p.name.to_string() })
                .collect::<Vec<_>>().join(", ")), false),
        RVal::BuiltinFn(n) => (Group::Functions, format!("built-in {n}"), false),
        RVal::Factor(f) => (Group::Values, format!("Factor w/ {} level{} {}: {}", f.levels.len(),
            if f.levels.len() == 1 { "" } else { "s" },
            f.levels.iter().take(3).map(|l| format!("\"{l}\"")).collect::<Vec<_>>().join(","),
            f.codes.iter().take(HEAD).map(|c| c.map(|c| (c + 1).to_string()).unwrap_or("NA".into()))
                .collect::<Vec<_>>().join(" ")), true),
        RVal::List(items) => (Group::Values, format!("List of {}", items.len()), false),
        RVal::Character(s, _) => (Group::Values, vector("chr", s.len(), &s.iter().take(HEAD)
            .map(|x| x.as_ref().map(|s| format!("\"{s}\"")).unwrap_or("NA".into())).collect::<Vec<_>>()), true),
        RVal::Logical(..) => {
            let l = v.as_logicals().unwrap_or_default();
            (Group::Values, vector("logi", l.len(), &l.iter().take(HEAD)
                .map(|x| match x { Some(true) => "TRUE", Some(false) => "FALSE", None => "NA" }.to_string())
                .collect::<Vec<_>>()), true)
        }
        RVal::Integer(..) | RVal::Numeric(..) | RVal::Single(..) => {
            let x = v.as_reals().unwrap_or_default();
            let tag = if matches!(v, RVal::Integer(..)) { "int" } else { "num" };
            if x.len() == 1 { (Group::Values, nums(&x), true) }
            else { (Group::Values, format!("{tag} [1:{}] {}", x.len(), nums(&x)), true) }
        }
        RVal::Null => (Group::Values, "NULL".into(), false),
        other => (Group::Values, other.type_name().to_string(), false),
    };
    EnvRow { name: name.to_string(), group, summary: clip(summary), viewable }
}

/// The first few values, each formatted on its own as R prints a scalar.
fn nums(x: &[Option<f64>]) -> String {
    let mut s: Vec<String> = x.iter().take(HEAD).map(|v| format_reals(&[*v], 7, 0)[0].trim().to_string()).collect();
    if x.len() > HEAD { s.push("...".into()); }
    s.join(" ")
}

fn vector(tag: &str, n: usize, head: &[String]) -> String {
    let more = if n > head.len() { " ..." } else { "" };
    if n == 1 { head.join(" ") } else { format!("{tag} [1:{n}] {}{more}", head.join(" ")) }
}

fn clip(s: String) -> String {
    if s.chars().count() <= SUMMARY_MAX { return s; }
    let mut t: String = s.chars().take(SUMMARY_MAX - 3).collect();
    t.push_str("...");
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use r2_types::{Attrs, DataFrame};

    fn num(v: &[f64]) -> RVal { RVal::Numeric(v.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(), Attrs::default()) }

    #[test]
    fn summaries_read_like_str() {
        let mut b: HashMap<Arc<str>, RVal> = HashMap::new();
        b.insert("x".into(), num(&[4.0, 8.0, 15.5]));
        b.insert("m".into(), num(&[18.0]));
        b.insert(".hidden".into(), num(&[1.0]));
        b.insert("s".into(), RVal::Character(vec![Some("a".into()), None], Attrs::default()));
        b.insert("d".into(), RVal::DataFrame(DataFrame { columns: vec![("a".into(), num(&[1.0, 2.0]))], row_names: None }));
        let rows = env_rows(&b);
        let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["d", "m", "s", "x"], "Data first, then values by name; .hidden left out");
        assert_eq!(rows[0].summary, "2 obs. of 1 variable");
        assert_eq!(rows[0].group, Group::Data);
        assert_eq!(rows[1].summary, "18");
        assert_eq!(rows[2].summary, "chr [1:2] \"a\" NA");
        assert_eq!(rows[3].summary, "num [1:3] 4 8 15.5");
    }

    #[test]
    fn long_vectors_are_cut_short() {
        let mut b: HashMap<Arc<str>, RVal> = HashMap::new();
        b.insert("big".into(), num(&(0..1000).map(|i| i as f64 * 1.25).collect::<Vec<_>>()));
        let r = &env_rows(&b)[0];
        assert!(r.summary.starts_with("num [1:1000] 0 1.25 2.5"));
        assert!(r.summary.chars().count() <= SUMMARY_MAX);
    }
}
