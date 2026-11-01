//! The IDE data viewer's cells (`view` request, docs/IDE_PROTOCOL.md): a
//! data frame, matrix or vector as a grid of cells formatted the way
//! `print()` formats them. R2 formats; the IDE only displays the page.
//!
//! Each numeric column gets ONE common format from the whole column
//! (`real_format`, R's `formatReal`), so its decimals line up as in a
//! printed data frame. Cells are then rendered on demand, a screenful at
//! a time: opening a million-row frame does one pass per column, not a
//! million string allocations.

use r2_types::{encode_real, real_format, RVal, RealFormat};

/// Display width a column may take, in characters.
const MIN_W: usize = 3;
const MAX_W: usize = 40;
/// Rows inspected when sizing a non-numeric column.
const SIZE_SAMPLE: usize = 2000;

enum Col {
    Num(Vec<Option<f64>>, RealFormat),
    Str(Vec<Option<String>>),
}

pub struct DataGrid {
    headers: Vec<String>,
    /// `num`, `int`, `chr`, `logi`, `factor` — shown under each header.
    pub types: Vec<&'static str>,
    cols: Vec<Col>,
    row_names: Option<Vec<String>>,
    nrow: usize,
    widths: Vec<usize>,
}

impl DataGrid {
    /// The grid for a value the viewer can show; `None` for anything else
    /// (functions, environments, ...).
    pub fn from_value(v: &RVal) -> Option<DataGrid> {
        let mut g = DataGrid { headers: vec![], types: vec![], cols: vec![], row_names: None, nrow: 0, widths: vec![] };
        match v {
            RVal::DataFrame(df) => {
                for (name, col) in &df.columns { g.push(name, col)?; }
                g.row_names = df.row_names.as_ref().map(|r| r.iter().map(|s| s.to_string()).collect());
            }
            RVal::Matrix(m) => {
                for c in 0..m.ncol {
                    let name = m.col_names.as_ref().and_then(|n| n.get(c)).map(|s| s.to_string())
                        .unwrap_or_else(|| format!("V{}", c + 1));
                    let data: Vec<Option<f64>> = (0..m.nrow).map(|r| Some(m.get(r, c))).collect();
                    let f = real_format(&data, 7, 0);
                    g.add(name, "num", Col::Num(data, f));
                }
                g.row_names = m.row_names.as_ref().map(|r| r.iter().map(|s| s.to_string()).collect());
            }
            RVal::Numeric(..) | RVal::Integer(..) | RVal::Single(..) | RVal::Character(..)
            | RVal::Logical(..) | RVal::Factor(..) => g.push("V1", v)?,
            _ => return None,
        }
        g.nrow = g.cols.iter().map(|c| match c { Col::Num(v, _) => v.len(), Col::Str(v) => v.len() }).max().unwrap_or(0);
        g.widths = (0..g.cols.len()).map(|c| g.measure(c)).collect();
        Some(g)
    }

    fn push(&mut self, name: &str, col: &RVal) -> Option<()> {
        let (ty, c) = match col {
            RVal::Numeric(..) | RVal::Single(..) | RVal::Integer(..) => {
                let v = col.as_reals().ok()?;
                let f = real_format(&v, 7, 0);
                (if matches!(col, RVal::Integer(..)) { "int" } else { "num" }, Col::Num(v, f))
            }
            RVal::Character(s, _) => ("chr", Col::Str(s.iter().map(|x| x.as_ref().map(|s| s.to_string())).collect())),
            RVal::Logical(..) => ("logi", Col::Str(col.as_logicals().ok()?.iter()
                .map(|x| x.map(|b| if b { "TRUE" } else { "FALSE" }.to_string())).collect())),
            RVal::Factor(f) => ("factor", Col::Str(f.codes.iter()
                .map(|c| c.and_then(|i| f.levels.get(i as usize)).map(|s| s.to_string())).collect())),
            _ => return None,
        };
        self.add(name.to_string(), ty, c);
        Some(())
    }

    fn add(&mut self, name: String, ty: &'static str, c: Col) {
        self.headers.push(name);
        self.types.push(ty);
        self.cols.push(c);
    }

    fn measure(&self, c: usize) -> usize {
        let body = match &self.cols[c] {
            Col::Num(_, f) => f.w,
            Col::Str(v) => v.iter().take(SIZE_SAMPLE).map(|s| s.as_deref().map_or(2, |s| s.chars().count())).max().unwrap_or(0),
        };
        body.max(self.headers[c].chars().count()).max(self.types[c].len()).clamp(MIN_W, MAX_W)
    }

    pub fn nrow(&self) -> usize { self.nrow }
    pub fn ncol(&self) -> usize { self.cols.len() }
    pub fn header(&self, c: usize) -> &str { &self.headers[c] }
    /// Column width in characters (header, type and values included).
    pub fn width(&self, c: usize) -> usize { self.widths[c] }
    /// Numbers right-align, text left-aligns, as `print` lays them out.
    pub fn right_aligned(&self, c: usize) -> bool { matches!(self.cols[c], Col::Num(..)) }

    /// The row label: the row name, or the 1-based row number.
    pub fn row_label(&self, r: usize) -> String {
        self.row_names.as_ref().and_then(|n| n.get(r).cloned()).unwrap_or_else(|| (r + 1).to_string())
    }

    /// One cell as text, cut to the column width (`NA` for missing).
    pub fn cell(&self, r: usize, c: usize) -> String {
        let s = match &self.cols[c] {
            Col::Num(v, f) => v.get(r).map(|x| encode_real(*x, *f).trim().to_string()).unwrap_or_default(),
            Col::Str(v) => v.get(r).map(|x| x.clone().unwrap_or_else(|| "NA".into())).unwrap_or_default(),
        };
        let w = self.widths[c];
        if s.chars().count() > w { s.chars().take(w.saturating_sub(1)).chain(['…']).collect() } else { s }
    }

    /// "6 rows × 2 columns" — the viewer's status line.
    pub fn shape(&self) -> String {
        format!("{} row{} × {} column{}", self.nrow, if self.nrow == 1 { "" } else { "s" },
                self.ncol(), if self.ncol() == 1 { "" } else { "s" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use r2_types::{Attrs, DataFrame, Factor, Matrix};

    fn num(v: &[f64]) -> RVal { RVal::Numeric(v.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(), Attrs::default()) }

    #[test]
    fn a_data_frame_formats_like_print() {
        let df = RVal::DataFrame(DataFrame { columns: vec![
            ("x".into(), RVal::Numeric(vec![Some(1.5), Some(10.0), None].into(), Attrs::default())),
            ("g".into(), RVal::Factor(Factor { codes: vec![Some(0), Some(1), None], levels: vec!["lo".into(), "hi".into()], ordered: false })),
            ("s".into(), RVal::Character(vec![Some("a fairly long label".into()), None, Some("b".into())], Attrs::default())),
        ], row_names: None });
        let g = DataGrid::from_value(&df).unwrap();
        assert_eq!((g.nrow(), g.ncol()), (3, 3));
        assert_eq!(g.types, vec!["num", "factor", "chr"]);
        // one common format for the column: 1.5 and 10.0 print with one decimal
        assert_eq!((g.cell(0, 0), g.cell(1, 0), g.cell(2, 0)), ("1.5".into(), "10.0".into(), "NA".into()));
        assert_eq!((g.cell(0, 1), g.cell(2, 1)), ("lo".into(), "NA".into()));
        assert_eq!(g.width(2), 19);
        assert!(g.right_aligned(0) && !g.right_aligned(2));
        assert_eq!(g.row_label(2), "3");
        assert_eq!(g.shape(), "3 rows × 3 columns");
    }

    #[test]
    fn matrices_vectors_and_unviewable_values() {
        let m = RVal::Matrix(Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let g = DataGrid::from_value(&m).unwrap();
        assert_eq!((g.header(1), g.cell(1, 1)), ("V2", "4".to_string()));
        let v = DataGrid::from_value(&num(&[0.25, 1.0])).unwrap();
        assert_eq!((v.ncol(), v.cell(1, 0)), (1, "1.00".to_string()));
        assert!(DataGrid::from_value(&RVal::Null).is_none());
    }

    #[test]
    fn wide_text_is_cut_to_the_column_limit() {
        let long = "x".repeat(100);
        let g = DataGrid::from_value(&RVal::Character(vec![Some(long.as_str().into())], Attrs::default())).unwrap();
        assert_eq!(g.width(0), MAX_W);
        assert_eq!(g.cell(0, 0).chars().count(), MAX_W);
        assert!(g.cell(0, 0).ends_with('…'));
    }
}
