//! ML FOUNDATION + DATA HANDLING — extracted from lib.rs
//! (engine-split, opus-4.8 session, content-anchored).
//!
//! Covers: svd() adapters, the CART decision-tree engine
//! (build_tree + gini + mse_impurity), read.csv parsing
//! (bi_read_csv_v2 + parse_csv_line), and the dplyr-style data
//! verbs (filter/select/mutate/arrange/regex helpers).
//!
//! Tree + CSV helpers are module-private. `r2_ml::tree::TreeNode`
//! is imported inline where used.

#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::all)]
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use r2_types::*;

use crate::{gv, gn, val_to_str, Engine};
use crate::err;

// ── svd() — Singular Value Decomposition ─────────────────────────────

// Phase R.4: bi_svd moved to r2-linalg::ops. Returns full thin SVD
// (`$d`, `$u`, `$v`) via `dgesvd_full` (shipped v0.1.0).
pub(crate) fn bi_svd(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_base::linalg_ops::bi_svd(a)
}

// ── eigen() — Eigenvalue decomposition ───────────────────────────────

// Phase R.4: bi_eigen moved to r2-linalg::ops.
pub(crate) fn bi_eigen(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_base::linalg_ops::bi_eigen(a)
}

// ── det() — matrix determinant via LU (partial pivoting) ─────────────

pub(crate) fn bi_det(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let m = match &gv(a, 0) {
        RVal::Matrix(m) => m.clone(),
        _ => return err!(Runtime, "det: argument must be a matrix"),
    };
    if m.nrow != m.ncol { return err!(Runtime, "det: 'a' must be a square matrix"); }
    let d = r2_linalg::ddet(m.nrow, &m.data)
        .map_err(|e| R2Err { msg: format!("det: {}", e), kind: ErrKind::Runtime })?;
    Ok(RVal::Numeric(vec![Some(d)].into(), Attrs::default()))
}

// ── solve() — matrix inverse, or solve a·x = b ───────────────────────

pub(crate) fn bi_solve(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let m = match &gv(a, 0) {
        RVal::Matrix(m) => m.clone(),
        _ => return err!(Runtime, "solve: 'a' must be a matrix"),
    };
    if m.nrow != m.ncol { return err!(Runtime, "solve: 'a' must be a square matrix"); }
    match a.get(1).map(|x| &x.value) {
        // solve(a): inverse.
        None | Some(RVal::Null) => {
            let inv = r2_linalg::dgetri(m.nrow, &m.data)
                .map_err(|e| R2Err { msg: format!("solve: {}", e), kind: ErrKind::Runtime })?;
            Ok(RVal::Matrix(Matrix::new(inv, m.nrow, m.ncol)))
        }
        // solve(a, B): solve for each column of B.
        Some(RVal::Matrix(b)) => {
            if b.nrow != m.nrow { return err!(Runtime, "solve: 'b' must have the same number of rows as 'a'"); }
            let mut out = vec![0.0; b.nrow * b.ncol];
            for col in 0..b.ncol {
                let bcol: Vec<f64> = (0..b.nrow).map(|r| b.data[col * b.nrow + r]).collect();
                let x = m.solve(&bcol)
                    .map_err(|e| R2Err { msg: format!("solve: {}", e), kind: ErrKind::Runtime })?;
                for r in 0..b.nrow { out[col * b.nrow + r] = x[r]; }
            }
            Ok(RVal::Matrix(Matrix::new(out, b.nrow, b.ncol)))
        }
        // solve(a, b): b a vector.
        Some(other) => {
            let bvec: Vec<f64> = e.as_reals(other)?.into_iter().flatten().collect();
            if bvec.len() != m.nrow { return err!(Runtime, "solve: length of 'b' must match the matrix dimension"); }
            let x = m.solve(&bvec)
                .map_err(|e| R2Err { msg: format!("solve: {}", e), kind: ErrKind::Runtime })?;
            Ok(RVal::Numeric(x.iter().map(|v| Some(*v)).collect::<Vec<_>>().into(), Attrs::default()))
        }
    }
}

// ── backsolve()/forwardsolve() — triangular solve ────────────────────

/// Read a named logical argument (`TRUE`/`FALSE`, or numeric ≠ 0) if present.
fn named_bool(a: &[EvalArg], name: &str) -> Option<bool> {
    match gn(a, name)? {
        RVal::Logical(b, _) => b.first().copied().flatten(),
        RVal::Numeric(v, _) => v.first().copied().flatten().map(|x| x != 0.0),
        RVal::Integer(v, _) => v.first().copied().flatten().map(|x| x != 0),
        _ => None,
    }
}

/// `backsolve(r, x, k, upper.tri = TRUE, transpose = FALSE)` and
/// `forwardsolve(l, x, k, upper.tri = FALSE, transpose = FALSE)` — solve the
/// triangular system `op(r)·y = x` by substitution. `x` may be a vector or a
/// matrix (each column solved independently). `k` selects the leading k×k
/// block. Built on `r2_linalg::dtrsv_gen`.
fn triangular_solve(e: &mut Engine, a: &[EvalArg], default_upper: bool, who: &str) -> Result<RVal, R2Err> {
    let r = match &gv(a, 0) {
        RVal::Matrix(m) => m.clone(),
        _ => return err!(Runtime, "{}: first argument must be a matrix", who),
    };
    if r.nrow != r.ncol { return err!(Runtime, "{}: matrix must be square", who); }
    let n = r.nrow;
    let upper = named_bool(a, "upper.tri").unwrap_or(default_upper);
    let transpose = named_bool(a, "transpose").unwrap_or(false);
    // k: named, else the third positional arg, else n.
    let k = match gn(a, "k").or_else(|| a.get(2).filter(|x| x.name.is_none()).map(|x| x.value.clone())) {
        Some(v) => e.as_reals(&v)?.first().copied().flatten().map(|x| x as usize).unwrap_or(n),
        None => n,
    };
    if k == 0 || k > n { return err!(Runtime, "{}: 'k' out of range", who); }

    // Leading k×k block of r (column-major).
    let rsub: Vec<f64> = if k == n { r.data.clone() } else {
        let mut s = vec![0.0; k * k];
        for c in 0..k { for row in 0..k { s[c * k + row] = r.data[c * n + row]; } }
        s
    };

    // x: the second positional argument.
    let xval = a.get(1).map(|x| x.value.clone()).unwrap_or(RVal::Null);
    let solve_col = |b: &mut Vec<f64>| -> Result<(), R2Err> {
        r2_linalg::dtrsv_gen(k, &rsub, b, upper, transpose)
            .map_err(|err| R2Err { msg: format!("{}: {}", who, err), kind: ErrKind::Runtime })
    };

    match xval {
        RVal::Matrix(x) => {
            if x.nrow < k { return err!(Runtime, "{}: 'x' has fewer rows than 'k'", who); }
            let mut out = vec![0.0; k * x.ncol];
            for c in 0..x.ncol {
                let mut b: Vec<f64> = (0..k).map(|row| x.data[c * x.nrow + row]).collect();
                solve_col(&mut b)?;
                for row in 0..k { out[c * k + row] = b[row]; }
            }
            Ok(RVal::Matrix(Matrix::new(out, k, x.ncol)))
        }
        other => {
            let xv: Vec<f64> = e.as_reals(&other)?.into_iter().flatten().collect();
            if xv.len() < k { return err!(Runtime, "{}: length of 'x' is less than 'k'", who); }
            let mut b = xv[..k].to_vec();
            solve_col(&mut b)?;
            Ok(RVal::Numeric(b.iter().map(|v| Some(*v)).collect::<Vec<_>>().into(), Attrs::default()))
        }
    }
}

pub(crate) fn bi_backsolve(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    triangular_solve(e, a, true, "backsolve")
}
pub(crate) fn bi_forwardsolve(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    triangular_solve(e, a, false, "forwardsolve")
}

// ── rcond()/kappa() — condition number ───────────────────────────────

/// `rcond(x)` — reciprocal of the 1-norm condition number,
/// `1 / (‖A‖₁ · ‖A⁻¹‖₁)`, computed exactly via the LU inverse. Returns 0 for a
/// singular matrix. (LAPACK estimates this; we compute it exactly — accurate
/// and fine for the moderate n where R2 is used.)
pub(crate) fn bi_rcond(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let m = match &gv(a, 0) {
        RVal::Matrix(m) => m.clone(),
        _ => return err!(Runtime, "rcond: argument must be a matrix"),
    };
    if m.nrow != m.ncol { return err!(Runtime, "rcond: matrix must be square"); }
    let n = m.nrow;
    let norm1 = |data: &[f64]| (0..n).map(|c| (0..n).map(|r| data[c * n + r].abs()).sum::<f64>())
        .fold(0.0f64, f64::max);
    let a_norm = norm1(&m.data);
    let rc = match r2_linalg::dgetri(n, &m.data) {
        Ok(inv) => {
            let inv_norm = norm1(&inv);
            if a_norm == 0.0 || inv_norm == 0.0 { 0.0 } else { 1.0 / (a_norm * inv_norm) }
        }
        Err(_) => 0.0, // singular → condition number ∞ → rcond 0
    };
    Ok(RVal::Numeric(vec![Some(rc)].into(), Attrs::default()))
}

/// `kappa(z)` — 2-norm condition number `σ_max / σ_min` (R's
/// `kappa(z, exact = TRUE)`; R's default is a cheaper estimate). The singular
/// values are `√λ` of `AᵀA` via the accurate symmetric eigensolver (`dsyev`) —
/// this avoids `dgesvd`'s current precision loss on the singular values. The
/// normal-equations form squares the condition number internally, so for
/// severely ill-conditioned `A` the estimate is optimistic; fine for the
/// conditioning checks kappa is used for. `Inf` for a singular matrix.
pub(crate) fn bi_kappa(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let m = match &gv(a, 0) {
        RVal::Matrix(m) => m.clone(),
        _ => return err!(Runtime, "kappa: argument must be a matrix"),
    };
    let (rows, cols) = (m.nrow, m.ncol);
    // G = AᵀA (cols×cols, symmetric PSD), column-major.
    let mut g = vec![0.0f64; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut s = 0.0;
            for k in 0..rows { s += m.data[i * rows + k] * m.data[j * rows + k]; }
            g[j * cols + i] = s;
        }
    }
    let eig = r2_linalg::dsyev(cols, &g)
        .map_err(|e| R2Err { msg: format!("kappa: {}", e), kind: ErrKind::Runtime })?;
    let lmax = eig.iter().cloned().fold(0.0f64, f64::max);
    let lmin = eig.iter().cloned().fold(f64::INFINITY, f64::min).max(0.0);
    let k = if lmin <= 0.0 { f64::INFINITY } else { (lmax / lmin).sqrt() };
    Ok(RVal::Numeric(vec![Some(k)].into(), Attrs::default()))
}

// ── Out-of-core columnar (memory-mapped) ────────────────────────────
// mmap.write(x, path) writes a numeric vector as a packed-f64 file;
// mmap.col(path) opens it as a handle whose reductions (sum/mean/min/max/
// length) STREAM over the memory map — so files LARGER THAN RAM work with
// bounded memory (the OS demand-pages). Verified on an 8 GB > RAM file.

fn rstr_(s: &str) -> RVal { RVal::Character(vec![Some(Arc::from(s))], Attrs::default()) }

pub(crate) fn bi_mmap_write(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let x: Vec<f64> = e.as_reals(&gv(a, 0))?.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let path = val_to_str(&gv(a, 1));
    if path.is_empty() { return err!(Runtime, "mmap.write(x, path): a file path is required"); }
    r2_arrow::write_packed_f64(&path, &x)
        .map_err(|m| R2Err { msg: format!("mmap.write: {}", m), kind: ErrKind::Runtime })?;
    Ok(rstr_(&path))
}

/// `mmap.map(path_in, FUN, path_out)` — out-of-core scalar transform.
/// Streams a named unary transform over the input column and writes the
/// result to a new packed-f64 file, never holding either column fully in
/// RAM (>RAM in → >RAM out, bounded RSS). Returns an `mmapcol` handle to
/// the output, so it composes directly with `sum`/`mean`/`mmap.map`/…
pub(crate) fn bi_mmap_map(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path_in = val_to_str(&gv(a, 0));
    let func = val_to_str(&gv(a, 1)).to_lowercase();
    let path_out = val_to_str(&gv(a, 2));
    if path_in.is_empty() || path_out.is_empty() {
        return err!(Runtime, "mmap.map(path_in, FUN, path_out): input and output paths are required");
    }
    let col = r2_arrow::MmapColumnar::open(&path_in)
        .map_err(|m| R2Err { msg: format!("mmap.map: {}", m), kind: ErrKind::Runtime })?;
    // Native scalar-transform allowlist, applied streaming/out-of-core.
    let f: fn(f64) -> f64 = match func.as_str() {
        "log" | "ln"     => f64::ln,
        "log2"           => f64::log2,
        "log10"          => f64::log10,
        "exp"            => f64::exp,
        "sqrt"           => f64::sqrt,
        "abs"            => f64::abs,
        "square" | "sq"  => |x| x * x,
        "neg"            => |x| -x,
        other => return Err(R2Err {
            msg: format!("mmap.map: unsupported transform '{}' (try log/log2/log10/exp/sqrt/abs/square/neg)", other),
            kind: ErrKind::Runtime }),
    };
    let n = col.map_to(&path_out, f)
        .map_err(|m| R2Err { msg: format!("mmap.map: {}", m), kind: ErrKind::Runtime })?;
    let mut fields = HashMap::new();
    fields.insert(Arc::from("path"), rstr_(&path_out));
    fields.insert(Arc::from("length"), RVal::Numeric(vec![Some(n as f64)].into(), Attrs::default()));
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("mmapcol"), fields }))
}

/// `read.parquet(file)` — import a Parquet file as a data.frame.
/// Pure-Rust (parquet/arrow crates), reads row-group by row-group.
/// Numeric Parquet types → numeric, boolean → logical, strings → character,
/// other types (dates/timestamps/decimals) → character via cast.
pub(crate) fn bi_read_parquet(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path = val_to_str(&gv(a, 0));
    if path.is_empty() { return err!(Runtime, "read.parquet(file): a file path is required"); }
    let table = r2_arrow::read_parquet(&path)
        .map_err(|m| R2Err { msg: m, kind: ErrKind::Runtime })?;
    let mut columns: Vec<(Arc<str>, RVal)> = Vec::with_capacity(table.columns.len());
    for (name, col) in table.names.into_iter().zip(table.columns.into_iter()) {
        let rv = match col {
            r2_arrow::ParquetCol::F64(v)  => RVal::Numeric(v.into(), Attrs::default()),
            r2_arrow::ParquetCol::Bool(v) => RVal::Logical(Logicals::new(v), Attrs::default()),
            r2_arrow::ParquetCol::Utf8(v) => RVal::Character(
                v.into_iter().map(|o| o.map(|s| Arc::from(s.as_str()))).collect(),
                Attrs::default()),
        };
        columns.push((Arc::from(name.as_str()), rv));
    }
    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
}

/// Append one parsed CSV row's numeric fields to the per-column writers.
/// Non-numeric / missing cells in a numeric column become NaN (R's NA for
/// f64). Columns with `None` writers (detected non-numeric) are skipped.
fn csv_write_row(
    writers: &mut [Option<(String, r2_arrow::MmapWriter)>],
    fields: &[String],
) -> Result<(), String> {
    for (c, w) in writers.iter_mut().enumerate() {
        if let Some((_, ww)) = w.as_mut() {
            let v = fields.get(c).and_then(|s| s.parse::<f64>().ok()).unwrap_or(f64::NAN);
            ww.append(&[v])?;
        }
    }
    Ok(())
}

/// `mmap.csv(file, sep=",")` — out-of-core CSV import. Streams the file
/// line by line (never holding it in RAM), writes each NUMERIC column to
/// its own packed-f64 sidecar (`<stem>__<col>.f64`) via the streaming
/// `MmapWriter`, and returns a named list of `mmap.col` handles. The
/// out-of-core reductions (`sum`/`mean`/`sd`/…) then run on each column,
/// so a larger-than-RAM CSV becomes analyzable with bounded memory.
/// Numeric columns are detected from the first data row; non-numeric
/// columns are skipped in this version (string columns await the Utf8
/// columnar dtype). Missing/garbled numeric cells import as NA (NaN).
pub(crate) fn bi_mmap_csv(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    use std::io::BufRead;
    let path = val_to_str(&gv(a, 0));
    if path.is_empty() { return err!(Runtime, "mmap.csv(file): a file path is required"); }
    let sep = gn(a, "sep").map(|v| val_to_str(&v)).filter(|s| !s.is_empty()).unwrap_or_else(|| ",".to_string());

    let f = std::fs::File::open(&path)
        .map_err(|e| R2Err { msg: format!("mmap.csv: cannot open '{}': {}", path, e), kind: ErrKind::Runtime })?;
    let mut reader = std::io::BufReader::new(f);
    let rderr = |e: std::io::Error| R2Err { msg: format!("mmap.csv: {}", e), kind: ErrKind::Runtime };

    // Header → column names.
    let mut header = String::new();
    if reader.read_line(&mut header).map_err(rderr)? == 0 {
        return err!(Runtime, "mmap.csv: empty file");
    }
    let names = parse_csv_line(header.trim_end_matches(['\r', '\n']), &sep);
    let ncol = names.len();

    // First data row → detect numeric columns.
    let mut first = String::new();
    if reader.read_line(&mut first).map_err(rderr)? == 0 {
        return err!(Runtime, "mmap.csv: file has a header but no data rows");
    }
    let first_fields = parse_csv_line(first.trim_end_matches(['\r', '\n']), &sep);
    let is_numeric: Vec<bool> = (0..ncol)
        .map(|c| first_fields.get(c).map(|s| s.parse::<f64>().is_ok()).unwrap_or(false))
        .collect();

    // One streaming writer per numeric column, sidecar next to the CSV.
    let dir = std::path::Path::new(&path).parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let stem = std::path::Path::new(&path).file_stem()
        .map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "data".to_string());
    let mut writers: Vec<Option<(String, r2_arrow::MmapWriter)>> = Vec::with_capacity(ncol);
    for c in 0..ncol {
        if is_numeric[c] {
            let safe: String = names[c].chars().map(|ch| if ch.is_alphanumeric() { ch } else { '_' }).collect();
            let sidecar = dir.join(format!("{}__{}.f64", stem, if safe.is_empty() { format!("col{}", c) } else { safe }));
            let pstr = sidecar.to_string_lossy().to_string();
            let w = r2_arrow::MmapWriter::create(&pstr)
                .map_err(|m| R2Err { msg: format!("mmap.csv: {}", m), kind: ErrKind::Runtime })?;
            writers.push(Some((pstr, w)));
        } else {
            writers.push(None);
        }
    }

    let to_err = |m: String| R2Err { msg: m, kind: ErrKind::Runtime };
    csv_write_row(&mut writers, &first_fields).map_err(to_err)?;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).map_err(rderr)? == 0 { break; }
        let t = line.trim_end_matches(['\r', '\n']);
        if t.is_empty() { continue; }
        let fields = parse_csv_line(t, &sep);
        csv_write_row(&mut writers, &fields).map_err(to_err)?;
    }

    // Finish writers → mmap.col handles in a named list.
    let mut items: Vec<(Option<Arc<str>>, RVal)> = Vec::new();
    for c in 0..ncol {
        if let Some((pstr, w)) = writers[c].take() {
            let len = w.finish().map_err(to_err)?;
            let mut fields = HashMap::new();
            fields.insert(Arc::from("path"), rstr_(&pstr));
            fields.insert(Arc::from("length"), RVal::Numeric(vec![Some(len as f64)].into(), Attrs::default()));
            items.push((
                Some(Arc::from(names[c].as_str())),
                RVal::TypeInstance(TypeInstance { type_name: Arc::from("mmapcol"), fields }),
            ));
        }
    }
    if items.is_empty() { return err!(Runtime, "mmap.csv: no numeric columns detected in the first data row"); }
    Ok(RVal::List(items))
}

/// Look up an `mmap.col` handle by name in an `mmap.csv` result list and
/// return its packed-f64 file path.
fn ooc_col_path(list: &RVal, name: &str) -> Option<String> {
    if let RVal::List(items) = list {
        for (n, v) in items {
            if n.as_deref() == Some(name) {
                if let RVal::TypeInstance(i) = v {
                    if i.type_name.as_ref() == "mmapcol" {
                        return i.fields.get("path").map(val_to_str);
                    }
                }
            }
        }
    }
    None
}

/// `mmap.lm(data, response, predictors)` — out-of-core ordinary least
/// squares. `data` is an `mmap.csv` list; `response` and `predictors`
/// (a character vector) name its columns. Accumulates XᵀX (p×p) and Xᵀy
/// in a SINGLE streaming pass over the memory-mapped columns — peak RAM is
/// p² + the OS page cache, independent of row count — then solves the
/// small normal-equations system. Returns named coefficients
/// (`(Intercept)`, predictors…).
///
/// Note: normal equations (not QR), so very ill-conditioned designs lose
/// precision; NA (NaN) rows poison the fit (no na.omit yet). Fine for the
/// well-conditioned large-n case this targets.
pub(crate) fn bi_mmap_lm(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let data = gv(a, 0);
    let resp = val_to_str(&gv(a, 1));
    let preds: Vec<String> = match &gv(a, 2) {
        RVal::Character(v, _) => v.iter().filter_map(|o| o.as_ref().map(|s| s.to_string())).collect(),
        other => { let s = val_to_str(other); if s.is_empty() { vec![] } else { vec![s] } }
    };
    if resp.is_empty() || preds.is_empty() {
        return err!(Runtime, "mmap.lm(data, response, predictors): a response and at least one predictor are required");
    }

    let y_path = ooc_col_path(&data, &resp)
        .ok_or_else(|| R2Err { msg: format!("mmap.lm: response column '{}' not found in data", resp), kind: ErrKind::Runtime })?;
    let y_col = r2_arrow::MmapColumnar::open(&y_path)
        .map_err(|m| R2Err { msg: format!("mmap.lm: {}", m), kind: ErrKind::Runtime })?;
    let y = y_col.as_slice();
    let n = y.len();

    let mut pred_cols = Vec::with_capacity(preds.len());
    for pn in &preds {
        let pp = ooc_col_path(&data, pn)
            .ok_or_else(|| R2Err { msg: format!("mmap.lm: predictor column '{}' not found in data", pn), kind: ErrKind::Runtime })?;
        let c = r2_arrow::MmapColumnar::open(&pp)
            .map_err(|m| R2Err { msg: format!("mmap.lm: {}", m), kind: ErrKind::Runtime })?;
        if c.len() != n {
            return err!(Runtime, "mmap.lm: column '{}' has length {} but response has {}", pn, c.len(), n);
        }
        pred_cols.push(c);
    }
    let pred_slices: Vec<&[f64]> = pred_cols.iter().map(|c| c.as_slice()).collect();

    let npred = preds.len();
    let p = 1 + npred; // intercept + predictors
    let mut xtx = vec![0.0f64; p * p];
    let mut xty = vec![0.0f64; p];
    let mut xr = vec![0.0f64; p];
    for r in 0..n {
        xr[0] = 1.0;
        for j in 0..npred { xr[1 + j] = pred_slices[j][r]; }
        let yr = y[r];
        for i in 0..p {
            xty[i] += xr[i] * yr;
            for j in i..p { xtx[i * p + j] += xr[i] * xr[j]; }
        }
    }
    // XᵀX is symmetric — mirror the lower triangle (also makes it valid
    // column-major for Matrix, which is identical for a symmetric matrix).
    for i in 0..p {
        for j in 0..i { xtx[i * p + j] = xtx[j * p + i]; }
    }

    let beta = Matrix::new(xtx, p, p).solve(&xty)
        .map_err(|e| R2Err { msg: format!("mmap.lm: normal-equations solve failed (collinear predictors?): {}", e), kind: ErrKind::Runtime })?;

    let mut names: Vec<Arc<str>> = Vec::with_capacity(p);
    names.push(Arc::from("(Intercept)"));
    for pn in &preds { names.push(Arc::from(pn.as_str())); }
    let mut attrs = Attrs::default();
    attrs.names = Some(names);
    Ok(RVal::Numeric(beta.iter().map(|b| Some(*b)).collect::<Vec<_>>().into(), attrs))
}

pub(crate) fn bi_mmap_col(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path = val_to_str(&gv(a, 0));
    let col = r2_arrow::MmapColumnar::open(&path)
        .map_err(|m| R2Err { msg: format!("mmap.col: {}", m), kind: ErrKind::Runtime })?;
    let len = col.len();
    let mut fields = HashMap::new();
    fields.insert(Arc::from("path"), rstr_(&path));
    fields.insert(Arc::from("length"), RVal::Numeric(vec![Some(len as f64)].into(), Attrs::default()));
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("mmapcol"), fields }))
}

/// If `v` is an `mmap.col` handle, stream-reduce it over the memory map
/// (`op` = sum/mean/min/max/length); otherwise `None` so the caller falls
/// back to the normal in-memory reduction.
/// If `v` is an `mmap.col` handle, compute approximate quantiles at
/// `probs` over the memory map (two-pass histogram, bounded RAM);
/// otherwise `None` so the caller falls back to the in-memory path.
pub(crate) fn mmap_quantile(v: &RVal, probs: &[f64]) -> Option<Result<Vec<f64>, R2Err>> {
    let inst = match v {
        RVal::TypeInstance(i) if i.type_name.as_ref() == "mmapcol" => i,
        _ => return None,
    };
    let path = inst.fields.get("path").map(val_to_str).unwrap_or_default();
    let probs = probs.to_vec();
    Some((|| {
        let col = r2_arrow::MmapColumnar::open(&path)
            .map_err(|m| R2Err { msg: format!("quantile(mmapcol): {}", m), kind: ErrKind::Runtime })?;
        Ok(col.quantile_hist(&probs, 1024))
    })())
}

pub(crate) fn mmap_reduce(v: &RVal, op: &str) -> Option<Result<RVal, R2Err>> {
    let inst = match v {
        RVal::TypeInstance(i) if i.type_name.as_ref() == "mmapcol" => i,
        _ => return None,
    };
    let path = inst.fields.get("path").map(val_to_str).unwrap_or_default();
    Some((|| {
        let col = r2_arrow::MmapColumnar::open(&path)
            .map_err(|m| R2Err { msg: format!("{}(mmapcol): {}", op, m), kind: ErrKind::Runtime })?;
        // `range` yields a length-2 vector; every other op is scalar.
        if op == "range" {
            let (mn, mx) = col.range();
            return Ok(RVal::Numeric(vec![Some(mn), Some(mx)].into(), Attrs::default()));
        }
        let r = match op {
            "sum" => col.sum(),
            "mean" => col.mean(),
            "min" => col.min(),
            "max" => col.max(),
            "length" => col.len() as f64,
            "prod" => col.prod(),
            "sd" => col.sd(1),   // R's sample sd (ddof = 1)
            "var" => col.var(1), // R's sample variance (ddof = 1)
            _ => return Err(R2Err { msg: format!("mmapcol: unsupported reduction '{}'", op), kind: ErrKind::Runtime }),
        };
        Ok(RVal::Numeric(vec![Some(r)].into(), Attrs::default()))
    })())
}

// ── prcomp() — Principal Component Analysis ──────────────────────────

// Phase R.1 step 4: bi_prcomp moved to r2-ml::dispatch.

// ── kmeans() — K-means clustering ────────────────────────────────────

// Phase R.1 step 4: bi_kmeans moved to r2-ml::dispatch. Per-point
// centroid assignment uses kernel::par_for(Op::PerPointDistance, ...).

// ── knn() — K-nearest neighbors classification ──────────────────────

// Phase R.1 step 4: bi_knn moved to r2-ml::dispatch.

// ── naive.bayes() — Naive Bayes classifier ──────────────────────────

// Phase R.1 step 4: bi_naive_bayes moved to r2-ml::dispatch.

// ── scale() — center and scale matrix columns ───────────────────────

pub(crate) fn bi_scale(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // R promotes a plain numeric vector to an n x 1 matrix before scaling.
    let mat = match &gv(a,0) {
        RVal::Matrix(m) => m.clone(),
        v @ (RVal::Numeric(..) | RVal::Integer(..)) => {
            let d: Vec<f64> = e.as_reals(v)?.into_iter().map(|o| o.unwrap_or(f64::NAN)).collect();
            let n = d.len();
            Matrix::new(d, n, 1)
        }
        _ => return err!(Runtime, "scale() needs a numeric vector or matrix"),
    };
    let center = gn(a,"center").and_then(|v| e.as_logicals(&v).ok()).map(|v| v[0] == Some(true)).unwrap_or(true);
    let do_scale = gn(a,"scale").and_then(|v| e.as_logicals(&v).ok()).map(|v| v[0] == Some(true)).unwrap_or(true);
    let (m, n) = (mat.nrow, mat.ncol);
    let mut x = mat.data.clone();
    let means = mat.col_means();
    for c in 0..n {
        let col_start = c * m;
        let mean = if center { means[c] } else { 0.0 };
        let mut ss = 0.0;
        for r in 0..m { ss += (x[col_start + r] - mean).powi(2); }
        let sd = if do_scale { (ss / (m - 1).max(1) as f64).sqrt().max(1e-15) } else { 1.0 };
        for r in 0..m {
            if center { x[col_start + r] -= mean; }
            if do_scale { x[col_start + r] /= sd; }
        }
    }
    Ok(RVal::Matrix(Matrix::new(x, m, n)))
}

// ═══════════════════════════════════════════════════════════════════════
// Decision Tree (CART — Classification and Regression Tree)
// ═══════════════════════════════════════════════════════════════════════

// Phase R.1 step 1: TreeNode struct extracted to r2-ml::tree. The engine
// keeps wrapper definitions of `build_tree` / `tree_predict_one` /
// `count_splits` / `serialize_tree` that delegate to the r2-ml versions —
// this preserves callsite signatures while the actual algorithms live in
// the domain crate.
use r2_ml::tree::TreeNode;

fn build_tree(x: &[f64], y: &[f64], m: usize, n: usize, row_mask: &[bool],
    max_depth: usize, min_samples: usize, depth: usize, is_classification: bool) -> TreeNode
{ r2_ml::tree::build_tree(x, y, m, n, row_mask, max_depth, min_samples, depth, is_classification) }

#[allow(dead_code)]
fn __build_tree_old(x: &[f64], y: &[f64], m: usize, n: usize, row_mask: &[bool],
    max_depth: usize, min_samples: usize, depth: usize, is_classification: bool) -> TreeNode
{
    let active: Vec<usize> = row_mask.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i).collect();
    let count = active.len();

    // Compute prediction: mean for regression, majority vote for classification
    let prediction = if is_classification {
        let mut votes: HashMap<i64, usize> = HashMap::new();
        for &i in &active { *votes.entry(y[i] as i64).or_insert(0) += 1; }
        votes.into_iter().max_by_key(|(_, c)| *c).map(|(k, _)| k as f64).unwrap_or(0.0)
    } else {
        active.iter().map(|&i| y[i]).sum::<f64>() / count.max(1) as f64
    };

    // Leaf conditions
    if count <= min_samples || depth >= max_depth {
        return TreeNode { is_leaf: true, prediction, feature: 0, threshold: 0.0, left: None, right: None, n_samples: count, impurity: 0.0 };
    }

    // Check if all y values are same
    let all_same = active.windows(2).all(|w| (y[w[0]] - y[w[1]]).abs() < 1e-10);
    if all_same {
        return TreeNode { is_leaf: true, prediction, feature: 0, threshold: 0.0, left: None, right: None, n_samples: count, impurity: 0.0 };
    }

    // Find best split
    let mut best_gain = 0.0f64;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;

    let parent_impurity = if is_classification { gini(&active, y) } else { mse_impurity(&active, y) };

    for feat in 0..n {
        // Get sorted indices for this feature
        let mut indexed: Vec<(f64, usize)> = active.iter().map(|&i| (x[feat * m + i], i)).collect();
        indexed.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        if is_classification {
            // Incremental gini: scan sorted data, maintain left/right class counts
            // Find unique classes (small integers)
            let mut max_class = 0i64;
            for &(_, idx) in &indexed { max_class = max_class.max(y[idx] as i64); }
            let nc = (max_class + 1) as usize;
            if nc > 1000 { continue; } // safety: too many classes

            let mut right_counts = vec![0usize; nc];
            for &(_, idx) in &indexed {
                let c = y[idx] as usize;
                if c < nc { right_counts[c] += 1; }
            }
            let mut left_counts = vec![0usize; nc];
            let mut left_n = 0usize;
            let mut right_n = count;

            // Limit candidate splits to ~32 evenly spaced
            let step = (indexed.len() / 32).max(1);
            let mut last_split = 0;

            for i in 0..indexed.len() - 1 {
                let c = y[indexed[i].1] as usize;
                if c < nc { left_counts[c] += 1; right_counts[c] -= 1; }
                left_n += 1;
                right_n -= 1;

                // Only evaluate at step boundaries or when value changes
                if i - last_split < step && i + 1 < indexed.len() - 1 { continue; }
                if (indexed[i].0 - indexed[i + 1].0).abs() < 1e-10 { continue; }

                last_split = i;
                let threshold = (indexed[i].0 + indexed[i + 1].0) / 2.0;

                // Compute gini from counts directly (no allocation)
                let left_gini = 1.0 - left_counts.iter().map(|&c| { let p = c as f64 / left_n as f64; p * p }).sum::<f64>();
                let right_gini = 1.0 - right_counts.iter().map(|&c| { let p = c as f64 / right_n as f64; p * p }).sum::<f64>();
                let weighted = (left_n as f64 * left_gini + right_n as f64 * right_gini) / count as f64;
                let gain = parent_impurity - weighted;

                if gain > best_gain { best_gain = gain; best_feature = feat; best_threshold = threshold; }
            }
        } else {
            // Regression: incremental MSE using running sums
            let mut left_sum = 0.0;
            let mut left_sq = 0.0;
            let total_sum: f64 = indexed.iter().map(|&(_, idx)| y[idx]).sum();
            let total_sq: f64 = indexed.iter().map(|&(_, idx)| y[idx] * y[idx]).sum();
            let mut left_n = 0usize;

            let step = (indexed.len() / 32).max(1);
            let mut last_split = 0;

            for i in 0..indexed.len() - 1 {
                let yi = y[indexed[i].1];
                left_sum += yi;
                left_sq += yi * yi;
                left_n += 1;
                let right_n = count - left_n;

                if i - last_split < step && i + 1 < indexed.len() - 1 { continue; }
                if (indexed[i].0 - indexed[i + 1].0).abs() < 1e-10 { continue; }
                last_split = i;

                let threshold = (indexed[i].0 + indexed[i + 1].0) / 2.0;
                let right_sum = total_sum - left_sum;

                let left_mse = left_sq / left_n as f64 - (left_sum / left_n as f64).powi(2);
                let right_mse = (total_sq - left_sq) / right_n as f64 - (right_sum / right_n as f64).powi(2);
                let weighted = (left_n as f64 * left_mse + right_n as f64 * right_mse) / count as f64;
                let gain = parent_impurity - weighted;

                if gain > best_gain { best_gain = gain; best_feature = feat; best_threshold = threshold; }
            }
        }
    }

    if best_gain <= 0.0 {
        return TreeNode { is_leaf: true, prediction, feature: 0, threshold: 0.0, left: None, right: None, n_samples: count, impurity: parent_impurity };
    }

    // Split
    let mut left_mask = vec![false; m];
    let mut right_mask = vec![false; m];
    for &i in &active {
        if x[best_feature * m + i] <= best_threshold { left_mask[i] = true; }
        else { right_mask[i] = true; }
    }

    let left = build_tree(x, y, m, n, &left_mask, max_depth, min_samples, depth + 1, is_classification);
    let right = build_tree(x, y, m, n, &right_mask, max_depth, min_samples, depth + 1, is_classification);

    TreeNode {
        is_leaf: false, prediction, feature: best_feature, threshold: best_threshold,
        left: Some(Box::new(left)), right: Some(Box::new(right)),
        n_samples: count, impurity: parent_impurity,
    }
}

fn gini(indices: &[usize], y: &[f64]) -> f64 {
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for &i in indices { *counts.entry(y[i] as i64).or_insert(0) += 1; }
    let n = indices.len() as f64;
    1.0 - counts.values().map(|&c| (c as f64 / n).powi(2)).sum::<f64>()
}

fn mse_impurity(indices: &[usize], y: &[f64]) -> f64 {
    let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len().max(1) as f64;
    indices.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>() / indices.len().max(1) as f64
}

// ── rpart() — Decision tree interface ────────────────────────────────

// Phase R.1 step 4: bi_rpart moved to r2-ml::dispatch. The 1-line adapter
// here exists only to satisfy r2-engine's `BuiltinFn` signature, which
// carries `&mut Engine` and `&EnvRef` for stateful builtins. Pure ML
// builtins ignore those — the adapter is FFI glue, not bloat.

// ── rf() — Random Forest ─────────────────────────────────────────────

// Phase R.1 step 4: bi_rf moved to r2-ml::dispatch. Uses kernel::par_for
// instead of par_iter — Rayon stays below the kernel layer (§4.9).

// ═══════════════════════════════════════════════════════════════════════
// PHASE: DATA HANDLING — filter, select, mutate, arrange, regex, etc.
// ═══════════════════════════════════════════════════════════════════════

// ── sub() / regexpr basics ───────────────────────────────────────────




// ── duplicated() / distinct values ───────────────────────────────────


// ── order() — return indices that would sort the vector ──────────────


// ── rank() — ranks of values ─────────────────────────────────────────


// ── cummax, cummin ───────────────────────────────────────────────────

pub(crate) fn bi_cummax(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::summary::bi_cummax(a) }

pub(crate) fn bi_cummin(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::summary::bi_cummin(a) }

// ── which() improvements — named results ────────────────────────────

// (which already exists, but let's add which.min/max for data.frame columns)

// ── Improved read.csv — handles quotes, various delimiters, type inference ──

pub(crate) fn bi_read_csv_v2(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path = match &gv(a,0) {
        RVal::Character(v,_) => v[0].as_ref().map(|s| s.to_string()).ok_or(R2Err{msg:"NA path".into(),kind:ErrKind::Runtime})?,
        _ => return err!(Runtime, "read.csv needs path"),
    };
    let header = gn(a,"header").and_then(|v| e.as_logicals(&v).ok()).map(|v| v[0] == Some(true)).unwrap_or(true);
    let sep = gn(a,"sep").and_then(|v| match v { RVal::Character(s,_) => s[0].as_ref().map(|s| s.to_string()), _ => None }).unwrap_or(",".into());
    let na_strings = vec!["NA", "na", "N/A", "n/a", "", ".", "NULL", "null", "None", "none"];

    let content = std::fs::read_to_string(&path).map_err(|e| R2Err{msg:format!("cannot read '{}': {}", path, e),kind:ErrKind::Runtime})?;
    let mut lines = content.lines();

    // Parse header
    let col_names: Vec<String> = if header {
        lines.next().map(|l| parse_csv_line(l, &sep)).unwrap_or_default()
    } else { Vec::new() };

    // Read all rows
    let mut raw_rows: Vec<Vec<String>> = Vec::new();
    for line in lines {
        if line.trim().is_empty() { continue; }
        raw_rows.push(parse_csv_line(line, &sep));
    }

    if raw_rows.is_empty() { return err!(Runtime, "empty CSV file"); }

    let ncol = col_names.len().max(raw_rows.iter().map(|r| r.len()).max().unwrap_or(0));
    let nrow = raw_rows.len();

    // Build columns with type inference
    let mut columns = Vec::new();
    for c in 0..ncol {
        let name = if c < col_names.len() { Arc::from(col_names[c].as_str()) } else { Arc::from(format!("V{}", c+1).as_str()) };

        let col_vals: Vec<String> = raw_rows.iter().map(|r| r.get(c).cloned().unwrap_or_default()).collect();

        // Type inference: try integer → numeric → character
        let all_int = col_vals.iter().all(|s| na_strings.contains(&s.as_str()) || s.parse::<i32>().is_ok());
        let all_num = col_vals.iter().all(|s| na_strings.contains(&s.as_str()) || s.parse::<f64>().is_ok());
        let has_num = col_vals.iter().any(|s| s.parse::<f64>().is_ok());

        if all_int && has_num {
            let vals: Vec<Integer> = col_vals.iter().map(|s| {
                if na_strings.contains(&s.as_str()) { None } else { s.parse().ok() }
            }).collect();
            columns.push((name, RVal::Integer(vals.into(), Attrs::default())));
        } else if all_num && has_num {
            let vals: Vec<Real> = col_vals.iter().map(|s| {
                if na_strings.contains(&s.as_str()) { None } else { s.parse().ok() }
            }).collect();
            columns.push((name, RVal::Numeric(vals.into(), Attrs::default())));
        } else {
            let vals: Vec<Character> = col_vals.iter().map(|s| {
                if na_strings.contains(&s.as_str()) { None } else { Some(Arc::from(s.as_str())) }
            }).collect();
            columns.push((name, RVal::Character(vals, Attrs::default())));
        }
    }

    soutln!("Read {} rows × {} columns from '{}'", nrow, ncol, path);
    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
}

/// Parse a CSV line handling quoted fields
fn parse_csv_line(line: &str, sep: &str) -> Vec<String> {
    let sep_char = sep.chars().next().unwrap_or(',');
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    current.push('"'); // escaped quote
                    chars.next();
                } else {
                    in_quotes = false; // end quote
                }
            } else {
                current.push(c);
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c == sep_char {
            fields.push(current.trim().to_string());
            current = String::new();
        } else {
            current.push(c);
        }
    }
    fields.push(current.trim().to_string());
    fields
}

// ── DataFrame pipe-friendly operations: filter, select, mutate, arrange ──

// Phase R.2: bi_filter moved to r2-data::dplyr.
pub(crate) fn bi_filter(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    return r2_data::dplyr::bi_filter(a);
    #[allow(unreachable_code)]
    {
    let df = match &gv(a,0) { RVal::DataFrame(df) => df.clone(), _ => return err!(Runtime, "filter() needs data.frame") };
    let e = _e;
    let mask = e.as_logicals(&gv(a,1))?;
    let keep: Vec<usize> = mask.iter().enumerate().filter(|(_, m)| **m == Some(true)).map(|(i, _)| i).collect();
    let nrow = df.nrow();

    let columns: Vec<(Arc<str>, RVal)> = df.columns.iter().map(|(name, col)| {
        let new_col = match col {
            RVal::Numeric(v, _) => RVal::Numeric(keep.iter().map(|&r| if r < v.len() { v[r] } else { None }).collect(), Attrs::default()),
            RVal::Integer(v, _) => RVal::Integer(keep.iter().map(|&r| if r < v.len() { v[r] } else { None }).collect(), Attrs::default()),
            RVal::Character(v, _) => RVal::Character(keep.iter().map(|&r| if r < v.len() { v[r].clone() } else { None }).collect(), Attrs::default()),
            RVal::Logical(v, _) => RVal::Logical(keep.iter().map(|&r| if r < v.len() { v[r] } else { None }).collect(), Attrs::default()),
            _ => col.clone(),
        };
        (name.clone(), new_col)
    }).collect();

    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
    } // end of #[allow(unreachable_code)] block (Phase R.2)
}

// Phase R.2: bi_select moved to r2-data::dplyr.
pub(crate) fn bi_select(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    return r2_data::dplyr::bi_select(a);
    #[allow(unreachable_code)]
    {
    let df = match &gv(a,0) { RVal::DataFrame(df) => df.clone(), _ => return err!(Runtime, "select() needs data.frame") };

    // Collect column names from remaining args
    let mut col_names: Vec<String> = Vec::new();
    for i in 1..10 {
        match &gv(a, i) {
            RVal::Character(v, _) => {
                for c in v { if let Some(s) = c { col_names.push(s.to_string()); } }
            }
            RVal::Null => break,
            _ => break,
        }
    }

    if col_names.is_empty() { return Ok(RVal::DataFrame(df)); }

    let columns: Vec<(Arc<str>, RVal)> = col_names.iter().filter_map(|name| {
        df.columns.iter().find(|(n, _)| n.as_ref() == name.as_str()).cloned()
    }).collect();

    if columns.is_empty() { return err!(Runtime, "select: no matching columns found"); }
    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
    } // end of #[allow(unreachable_code)] block (Phase R.2)
}

// Phase R.2: bi_arrange moved to r2-data::dplyr.
pub(crate) fn bi_arrange(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    return r2_data::dplyr::bi_arrange(a);
    #[allow(unreachable_code)]
    {
    let df = match &gv(a,0) { RVal::DataFrame(df) => df.clone(), _ => return err!(Runtime, "arrange() needs data.frame") };
    let e = _e;
    let sort_vals = e.as_reals(&gv(a,1))?;
    let decreasing = gn(a,"decreasing").and_then(|v| e.as_logicals(&v).ok()).map(|v| v[0] == Some(true)).unwrap_or(false);

    let nrow = df.nrow();
    let mut indices: Vec<usize> = (0..nrow).collect();
    indices.sort_by(|&a, &b| {
        let va = sort_vals.get(a).and_then(|x| *x).unwrap_or(f64::NAN);
        let vb = sort_vals.get(b).and_then(|x| *x).unwrap_or(f64::NAN);
        if decreasing { vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal) }
        else { va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal) }
    });

    let columns: Vec<(Arc<str>, RVal)> = df.columns.iter().map(|(name, col)| {
        let new_col = match col {
            RVal::Numeric(v, _) => RVal::Numeric(indices.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Integer(v, _) => RVal::Integer(indices.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Character(v, _) => RVal::Character(indices.iter().map(|&r| v.get(r).cloned().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Logical(v, _) => RVal::Logical(indices.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            _ => col.clone(),
        };
        (name.clone(), new_col)
    }).collect();

    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
    } // end of #[allow(unreachable_code)] block (Phase R.2)
}

// ── Sys.getenv() — read environment variable ─────────────────────────

pub(crate) fn bi_sys_getenv(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = val_to_str(&gv(a,0));
    let val = std::env::var(&name).unwrap_or_default();
    Ok(rstr(&val))
}

// ── file.exists() — check if file exists ─────────────────────────────


// ── list.files() — list files in directory ───────────────────────────


// end of file

// ── Memory-budget management (Pillar 2) ──────────────────────────────
//
// mem.budget("8G") / mem.budget()  — set/query the soft in-RAM budget.
// mem.status()                     — live bytes, budget, spill count.
// mem.spill(x, [dir])              — write a numeric vector to a packed
//                                    mmap file; returns an `mmapcol` handle
//                                    (composes with sum/mean/… unchanged).
// mem.restore(h)                   — read an mmapcol back into a dense
//                                    numeric vector.

pub(crate) fn bi_mem_budget(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    if a.is_empty() {
        return Ok(RVal::Numeric(vec![Some(crate::membudget::budget() as f64)].into(), Attrs::default()));
    }
    let v = gv(a, 0);
    let bytes = match &v {
        RVal::Character(cs, _) => cs.first().and_then(|x| x.as_ref())
            .and_then(|s| crate::membudget::parse_budget(s))
            .ok_or_else(|| R2Err { msg: "mem.budget: could not parse size (try \"8G\", \"512M\", or a byte count)".into(), kind: ErrKind::Runtime })?,
        _ => e.scalar_f64(&v)?.unwrap_or(0.0).max(0.0) as u64,
    };
    crate::membudget::set_budget(bytes);
    Ok(RVal::Numeric(vec![Some(bytes as f64)].into(), Attrs::default()))
}

pub(crate) fn bi_mem_status(_e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let mk = |n: &str, v: f64| (Some(Arc::from(n)), RVal::Numeric(vec![Some(v)].into(), Attrs::default()));
    Ok(RVal::List(vec![
        mk("live.bytes",  crate::membudget::live_bytes() as f64),
        mk("budget.bytes", crate::membudget::budget() as f64),
        mk("spills",       crate::membudget::spill_count() as f64),
        (Some(Arc::from("over.budget")), rbool(crate::membudget::over_budget())),
    ]))
}

pub(crate) fn bi_mem_spill(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let data: Vec<f64> = e.as_reals(&gv(a, 0))?.into_iter().map(|o| o.unwrap_or(f64::NAN)).collect();
    if data.is_empty() { return err!(Runtime, "mem.spill: needs a non-empty numeric vector"); }
    // Spill target: given dir arg, else a unique temp file.
    let dir = if a.len() > 1 { val_to_str(&gv(a, 1)) } else { std::env::temp_dir().to_string_lossy().into_owned() };
    let fname = format!("r2spill_{}_{}.f64",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    let path = std::path::Path::new(&dir).join(fname);
    let path_s = path.to_string_lossy().into_owned();

    // Write BEFORE dropping the RAM copy — spill never loses data.
    let mut w = r2_arrow::MmapWriter::create(&path_s)
        .map_err(|m| R2Err { msg: format!("mem.spill: {}", m), kind: ErrKind::Runtime })?;
    w.append(&data).map_err(|m| R2Err { msg: format!("mem.spill: {}", m), kind: ErrKind::Runtime })?;
    let n = w.finish().map_err(|m| R2Err { msg: format!("mem.spill: {}", m), kind: ErrKind::Runtime })?;

    // Account: the RAM copy is now spillable; note the transfer.
    crate::membudget::sub((n as u64) * 8);
    crate::membudget::note_spill();

    let mut fields = HashMap::new();
    fields.insert(Arc::from("path"), rstr_(&path_s));
    fields.insert(Arc::from("length"), RVal::Numeric(vec![Some(n as f64)].into(), Attrs::default()));
    fields.insert(Arc::from("spilled"), rbool(true));
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("mmapcol"), fields }))
}

pub(crate) fn bi_mem_restore(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path = match &gv(a, 0) {
        RVal::TypeInstance(i) if i.type_name.as_ref() == "mmapcol" =>
            i.fields.get("path").map(val_to_str).unwrap_or_default(),
        other => val_to_str(other),
    };
    if path.is_empty() { return err!(Runtime, "mem.restore: needs an mmapcol handle (from mem.spill)"); }
    let col = r2_arrow::MmapColumnar::open(&path)
        .map_err(|m| R2Err { msg: format!("mem.restore: {}", m), kind: ErrKind::Runtime })?;
    let data: Vec<f64> = col.as_slice().to_vec();
    crate::membudget::add((data.len() as u64) * 8); // back in RAM
    Ok(RVal::Numeric(Reals::from_dense_f64(data), Attrs::default()))
}

// ── GPU dispatcher surface (Pillar 1) ────────────────────────────────
//
// options(r2.gpu = TRUE/FALSE) → gpu.enable(); gpu.status() reports the
// routing decision; gpu.map("relu", x) runs an f32-safe element map via
// the dispatcher (GPU when enabled+eligible+available, else the CPU
// reference — never a wrong answer). Statistics are never GPU-routed.

fn gpu_op_by_name(name: &str, a: &[EvalArg], e: &mut Engine) -> Result<r2_gpu::Op, R2Err> {
    Ok(match name.to_lowercase().as_str() {
        "identity" | "id" => r2_gpu::Op::Identity,
        "relu"            => r2_gpu::Op::Relu,
        "sigmoid"         => r2_gpu::Op::Sigmoid,
        "tanh"            => r2_gpu::Op::Tanh,
        "exp"             => r2_gpu::Op::Exp,
        "scale"           => {
            let a2 = e.scalar_f64(&gv(a, 2))?.unwrap_or(1.0) as f32;
            r2_gpu::Op::Scale(a2)
        }
        other => return err!(Runtime,
            "gpu.map: unsupported op '{}' (identity/relu/sigmoid/tanh/exp/scale). Statistics are never GPU-routed.", other),
    })
}

pub(crate) fn bi_gpu_enable(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let on = match &gv(a, 0) {
        RVal::Logical(b, _) => b.first().copied().flatten().unwrap_or(false),
        RVal::Null => true,
        other => e.scalar_f64(other)?.unwrap_or(0.0) != 0.0,
    };
    r2_gpu::set_gpu_enabled(on);
    Ok(rbool(r2_gpu::gpu_enabled()))
}

pub(crate) fn bi_gpu_status(_e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let mk = |n: &str, v: RVal| (Some(Arc::from(n)), v);
    Ok(RVal::List(vec![
        mk("enabled", rbool(r2_gpu::gpu_enabled())),
        mk("min.elems", RVal::Numeric(vec![Some(r2_gpu::GPU_MIN_ELEMS as f64)].into(), Attrs::default())),
        mk("report", rstr_(&r2_gpu::backend_report(r2_gpu::GPU_MIN_ELEMS, r2_gpu::Op::Relu))),
    ]))
}

pub(crate) fn bi_gpu_map(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = val_to_str(&gv(a, 0));
    let op = gpu_op_by_name(&name, a, e)?;
    // f32 in/out at the GPU boundary; the result is promoted back to f64
    // for the numeric vector (the CPU reference and GPU agree within f32).
    let xs: Vec<f32> = e.as_reals(&gv(a, 1))?.into_iter().map(|o| o.unwrap_or(f64::NAN) as f32).collect();
    if xs.is_empty() { return err!(Runtime, "gpu.map: needs a non-empty numeric vector"); }
    let out = r2_gpu::dispatch(op, &xs);
    Ok(RVal::Numeric(Reals::from_dense_f64(out.into_iter().map(|v| v as f64).collect()), Attrs::default()))
}
