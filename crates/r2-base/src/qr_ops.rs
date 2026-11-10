//! `qr()` and its accessors, with R's arguments, errors and result shapes. The
//! numerics are `r2_linalg::linpack`; a `y` keeps its shape and names through
//! `qr.qy`, `qr.qty`, `qr.resid` and `qr.fitted`, as it does in R.

use r2_linalg::linpack;
use r2_types::{rint, rints, rnums, Attrs, ErrKind, EvalArg, Matrix, R2Err, RVal, TypeInstance};
use std::collections::HashMap;
use std::sync::Arc;

const DEFAULT_TOL: f64 = 1e-7;
/// n·p² at which the blocked QR starts to pay for its extra copy and check.
/// Measured 2026-11-09 on the AMD laptop, other work running: 300 x 60 was
/// slower blocked (1.68 ms against ~1.3), 1000 x 200 about 20% faster (40 ms),
/// 2000 x 500 about 2x (164 ms against 345).
const BLOCKED_FROM: usize = 1 << 24;
const NOT_QR: &str = "argument is not a QR decomposition";

fn fail<T>(msg: impl Into<String>) -> Result<T, R2Err> {
    Err(R2Err { msg: msg.into(), kind: ErrKind::Runtime })
}

/// R's argument matching: exact names, then unique prefixes, then position.
fn matched<'a>(a: &'a [EvalArg], formals: &[&str], dots: bool) -> Result<Vec<Option<&'a RVal>>, R2Err> {
    let mut out: Vec<Option<&RVal>> = vec![None; formals.len()];
    let mut rest = Vec::new();
    for arg in a {
        match &arg.name {
            Some(name) => match formals.iter().position(|f| *f == name.as_ref()) {
                Some(i) => out[i] = Some(&arg.value),
                None => rest.push(arg),
            },
            None => rest.push(arg),
        }
    }
    let mut positional = Vec::new();
    for arg in rest {
        let Some(name) = &arg.name else { positional.push(&arg.value); continue };
        let hits: Vec<usize> = (0..formals.len())
            .filter(|&i| out[i].is_none() && formals[i].starts_with(name.as_ref()))
            .collect();
        match hits.as_slice() {
            [i] => out[*i] = Some(&arg.value),
            _ if dots => {}
            _ => return fail(format!("unused argument ({name})")),
        }
    }
    let mut next = positional.into_iter();
    for slot in out.iter_mut().filter(|s| s.is_none()) {
        match next.next() { Some(v) => *slot = Some(v), None => break }
    }
    if !dots && next.next().is_some() { return fail("unused argument"); }
    Ok(out)
}

fn flag(v: Option<&RVal>) -> bool {
    v.and_then(|x| x.scalar_f64().ok().flatten()).is_some_and(|x| x != 0.0)
}

fn reals(v: &RVal) -> Result<Vec<f64>, R2Err> {
    Ok(v.as_reals()?.iter().map(|x| x.unwrap_or(f64::NAN)).collect())
}

/// A matrix, or a vector as a one-column matrix with its names as row names.
fn to_matrix(v: &RVal) -> Result<Option<Matrix>, R2Err> {
    Ok(match v {
        RVal::Matrix(m) => Some(m.clone()),
        RVal::Numeric(_, at) | RVal::Integer(_, at) | RVal::Logical(_, at) => {
            let data = reals(v)?;
            let n = data.len();
            let mut m = Matrix::new(data, n, 1);
            m.row_names = at.names.clone();
            Some(m)
        }
        _ => None,
    })
}

/// The fields of a `qr` object.
struct Qr<'a> { m: &'a Matrix, rank: usize, qraux: Vec<f64>, pivot: Vec<usize> }

fn view<'a>(v: Option<&'a RVal>, not_qr: &str) -> Result<Qr<'a>, R2Err> {
    let inst = match v {
        Some(RVal::TypeInstance(i)) if i.type_name.as_ref() == "qr" => i,
        _ => return fail(not_qr),
    };
    let (Some(RVal::Matrix(m)), Some(rank), Some(qraux), Some(pivot)) =
        (inst.fields.get("qr"), inst.fields.get("rank"), inst.fields.get("qraux"), inst.fields.get("pivot"))
    else { return fail(not_qr) };
    Ok(Qr {
        m,
        rank: rank.scalar_f64()?.unwrap_or(0.0) as usize,
        qraux: reals(qraux)?,
        pivot: reals(pivot)?.iter().map(|&x| x as usize).collect(),
    })
}

/// A right-hand side: its numbers, and what it must look like coming back.
struct Rhs<'a> { data: Vec<f64>, nrow: usize, ncol: usize, source: &'a RVal }

fn rhs<'a>(y: Option<&'a RVal>, n: usize) -> Result<Rhs<'a>, R2Err> {
    let Some(source) = y else { return fail("argument \"y\" is missing, with no default") };
    let (data, nrow, ncol) = match source {
        RVal::Matrix(m) => (m.data.clone(), m.nrow, m.ncol),
        RVal::Numeric(..) | RVal::Integer(..) | RVal::Logical(..) => {
            let d = reals(source)?;
            let len = d.len();
            (d, len, 1)
        }
        _ => return fail("'y' must be numeric"),
    };
    if nrow != n { return fail("'qr' and 'y' must have the same number of rows"); }
    Ok(Rhs { data, nrow, ncol, source })
}

impl Rhs<'_> {
    /// Fortran refuses these unless told otherwise, and only `qr.coef` tells it.
    fn require_finite(&self) -> Result<(), R2Err> {
        if self.data.iter().all(|x| x.is_finite()) { return Ok(()); }
        fail("NA/NaN/Inf in foreign function call (arg 5)")
    }

    fn each_column(&mut self, mut f: impl FnMut(&mut [f64])) {
        if self.nrow > 0 { self.data.chunks_mut(self.nrow).for_each(|c| f(c)); }
    }

    fn finish(self) -> RVal {
        match self.source {
            RVal::Matrix(m) => RVal::Matrix(Matrix {
                data: self.data, nrow: m.nrow, ncol: m.ncol,
                col_names: m.col_names.clone(), row_names: m.row_names.clone(),
            }),
            RVal::Numeric(_, at) | RVal::Integer(_, at) | RVal::Logical(_, at) => RVal::Numeric(
                r2_types::Reals::from_dense_f64(self.data),
                Attrs { names: at.names.clone(), ..Default::default() },
            ),
            _ => rnums(&self.data),
        }
    }
}

/// `qr(x, tol = 1e-07, LAPACK = FALSE, ...)`. `LAPACK` chooses an
/// implementation in R; R2 chooses its own (see `decompose`), so the argument
/// is accepted for scripts that pass it and has no effect.
pub fn bi_qr(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let args = matched(a, &["x", "tol", "LAPACK"], true)?;
    let tol = match args[1] { Some(t) => t.scalar_f64()?.unwrap_or(DEFAULT_TOL), None => DEFAULT_TOL };
    let Some(x) = args[0] else { return fail("argument \"x\" is missing, with no default") };
    let Some(m) = to_matrix(x)? else { return fail("qr: 'x' must be a numeric matrix") };
    decompose(m, tol)
}

fn decompose(mut m: Matrix, tol: f64) -> Result<RVal, R2Err> {
    if !m.data.iter().all(|x| x.is_finite()) {
        return fail("NA/NaN/Inf in foreign function call (arg 1)");
    }
    // A large tall matrix goes to the blocked, parallel QR, which returns the
    // same object when no column needs pivoting and declines otherwise.
    let large = m.nrow * m.ncol * m.ncol >= BLOCKED_FROM;
    let d = match large.then(|| linpack::dqrdc2_blocked(&mut m.data, m.nrow, m.ncol, tol)).flatten() {
        Some(d) => d,
        None => linpack::dqrdc2(&mut m.data, m.nrow, m.ncol, tol),
    };
    if let Some(cn) = &m.col_names {
        m.col_names = Some(d.pivot.iter().map(|&j| cn[j as usize - 1].clone()).collect());
    }
    let fields: HashMap<Arc<str>, RVal> = [
        (Arc::from("qr"), RVal::Matrix(m)),
        (Arc::from("rank"), rint(d.rank as i32)),
        (Arc::from("qraux"), rnums(&d.qraux)),
        (Arc::from("pivot"), rints(&d.pivot)),
    ].into_iter().collect();
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("qr"), fields }))
}

pub fn bi_is_qr(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let is = matches!(a.first().map(|x| &x.value), Some(RVal::TypeInstance(i)) if i.type_name.as_ref() == "qr");
    Ok(RVal::Logical(vec![Some(is)].into(), Attrs::default()))
}

fn apply(
    a: &[EvalArg],
    op: fn(&[f64], usize, usize, &[f64], &mut [f64]),
) -> Result<RVal, R2Err> {
    let args = matched(a, &["qr", "y"], false)?;
    let q = view(args[0], NOT_QR)?;
    let mut y = rhs(args[1], q.m.nrow)?;
    y.require_finite()?;
    y.each_column(|c| op(&q.m.data, q.m.nrow, q.rank, &q.qraux, c));
    Ok(y.finish())
}

pub fn bi_qr_qy(a: &[EvalArg]) -> Result<RVal, R2Err> { apply(a, linpack::qy) }
pub fn bi_qr_qty(a: &[EvalArg]) -> Result<RVal, R2Err> { apply(a, linpack::qty) }

pub fn bi_qr_resid(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let args = matched(a, &["qr", "y"], false)?;
    let q = view(args[0], NOT_QR)?;
    if q.rank == 0 {
        return match args[1] { Some(y) => Ok(y.clone()), None => fail("argument \"y\" is missing, with no default") };
    }
    apply(a, linpack::resid)
}

pub fn bi_qr_fitted(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let args = matched(a, &["qr", "y", "k"], false)?;
    let q = view(args[0], NOT_QR)?;
    let k = match args[2] { Some(k) => k.scalar_f64()?.unwrap_or(0.0).max(0.0) as usize, None => q.rank };
    if k > q.rank { return fail("'k' is too large"); }
    let mut y = rhs(args[1], q.m.nrow)?;
    y.require_finite()?;
    y.each_column(|c| linpack::fitted(&q.m.data, q.m.nrow, k, &q.qraux, c));
    Ok(y.finish())
}

/// The coefficients in the original column order, NA for the columns left out.
fn coef(q: &Qr<'_>, y: &Rhs<'_>) -> Result<RVal, R2Err> {
    let (n, p, k) = (q.m.nrow, q.m.ncol, q.rank);
    let mut out = vec![f64::NAN; p * y.ncol];
    if k > 0 {
        for c in 0..y.ncol {
            let b = linpack::coef(&q.m.data, n, k, &q.qraux, &y.data[c * n..(c + 1) * n])
                .or_else(|_| fail("exact singularity in 'qr.coef'"))?;
            for (i, v) in b.into_iter().enumerate() { out[c * p + q.pivot[i] - 1] = v; }
        }
    }
    let names: Option<Vec<Arc<str>>> = q.m.col_names.as_ref().map(|nam| {
        let mut sorted = nam.clone();
        for (i, name) in nam.iter().enumerate() { sorted[q.pivot[i] - 1] = name.clone(); }
        sorted
    });
    Ok(match y.source {
        RVal::Matrix(m) => RVal::Matrix(Matrix {
            data: out, nrow: p, ncol: y.ncol, col_names: m.col_names.clone(), row_names: names,
        }),
        _ => RVal::Numeric(
            out.iter().map(|x| if x.is_nan() { None } else { Some(*x) }).collect(),
            Attrs { names, ..Default::default() },
        ),
    })
}

pub fn bi_qr_coef(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let args = matched(a, &["qr", "y"], false)?;
    let q = view(args[0], "first argument must be a QR decomposition")?;
    coef(&q, &rhs(args[1], q.m.nrow)?)
}

/// `qr.Q(qr, complete = FALSE, Dvec)`.
pub fn bi_qr_q(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let args = matched(a, &["qr", "complete", "Dvec"], false)?;
    let q = view(args[0], NOT_QR)?;
    let n = q.m.nrow;
    let cols = if flag(args[1]) { n } else { n.min(q.m.ncol) };
    let dvec = match args[2] { Some(d) => reals(d)?, None => vec![1.0] };
    if dvec.is_empty() { return fail("'x' must have positive length"); }
    let mut d = vec![0.0; n * cols];
    for j in 0..cols { d[j * n + j] = dvec[j % dvec.len()]; }
    if n > 0 {
        for c in d.chunks_mut(n) { linpack::qy(&q.m.data, n, q.rank, &q.qraux, c); }
    }
    Ok(RVal::Matrix(Matrix::new(d, n, cols)))
}

fn r_factor(m: &Matrix, rows: usize) -> Matrix {
    let mut data = vec![0.0; rows * m.ncol];
    for j in 0..m.ncol {
        for i in 0..rows.min(j + 1) { data[j * rows + i] = m.data[j * m.nrow + i]; }
    }
    Matrix {
        data, nrow: rows, ncol: m.ncol, col_names: m.col_names.clone(),
        row_names: m.row_names.as_ref().map(|r| r[..rows].to_vec()),
    }
}

/// `qr.R(qr, complete = FALSE)`.
pub fn bi_qr_r(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let args = matched(a, &["qr", "complete"], false)?;
    let q = view(args[0], NOT_QR)?;
    let rows = if flag(args[1]) { q.m.nrow } else { q.m.nrow.min(q.m.ncol) };
    Ok(RVal::Matrix(r_factor(q.m, rows)))
}

/// `qr.X(qr, complete = FALSE, ncol)`: the matrix back, in its original order.
pub fn bi_qr_x(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let args = matched(a, &["qr", "complete", "ncol"], false)?;
    let q = view(args[0], NOT_QR)?;
    let (n, p) = (q.m.nrow, q.m.ncol);
    let cols = match args[2] {
        Some(c) => c.scalar_f64()?.unwrap_or(0.0).max(0.0) as usize,
        None if flag(args[1]) => n,
        None => n.min(p),
    };
    let pivoted = q.pivot.iter().enumerate().any(|(i, &j)| j != i + 1);
    if pivoted && cols < q.pivot.len() { return fail("need larger value of 'ncol' as pivoting occurred"); }
    let r = r_factor(q.m, n);
    let mut res = if cols < p {
        Matrix {
            data: r.data[..cols * n].to_vec(), nrow: n, ncol: cols,
            col_names: r.col_names.as_ref().map(|c| c[..cols].to_vec()), row_names: r.row_names.clone(),
        }
    } else if cols > p {
        let mut data = vec![0.0; n * cols];
        data[..n * p].copy_from_slice(&r.data);
        for j in p..cols.min(n) { data[j * n + j] = 1.0; }
        Matrix::new(data, n, cols)
    } else {
        r
    };
    if n > 0 {
        for c in res.data.chunks_mut(n) { linpack::qy(&q.m.data, n, q.rank, &q.qraux, c); }
    }
    if pivoted {
        let old = res.data.clone();
        for (i, &j) in q.pivot.iter().enumerate() {
            res.data[(j - 1) * n..j * n].copy_from_slice(&old[i * n..(i + 1) * n]);
        }
        if let Some(cn) = res.col_names.clone() {
            let names = res.col_names.as_mut().expect("just read");
            for (i, &j) in q.pivot.iter().enumerate() { names[j - 1] = cn[i].clone(); }
        }
    }
    Ok(RVal::Matrix(res))
}

/// `qr.solve` and `solve.qr`: coefficients with the left-out columns at 0.
fn solve(q: &Qr<'_>, b: Option<&RVal>, singular: &str) -> Result<RVal, R2Err> {
    let (nr, nc) = (q.m.nrow, q.m.ncol);
    if q.rank != nr.min(nc) { return fail(singular); }
    let identity;
    let b = match b {
        Some(b) => b,
        None => {
            if nc != nr { return fail("only square matrices can be inverted"); }
            let mut d = vec![0.0; nc * nc];
            for j in 0..nc { d[j * nc + j] = 1.0; }
            identity = RVal::Matrix(Matrix::new(d, nc, nc));
            &identity
        }
    };
    Ok(match coef(q, &rhs(Some(b), nr)?)? {
        RVal::Matrix(mut m) => {
            for x in m.data.iter_mut().filter(|x| x.is_nan()) { *x = 0.0; }
            RVal::Matrix(m)
        }
        RVal::Numeric(v, at) => RVal::Numeric(v.iter().map(|x| Some(x.unwrap_or(0.0))).collect(), at),
        other => other,
    })
}

/// `qr.solve(a, b, tol = 1e-07)`.
pub fn bi_qr_solve(a: &[EvalArg]) -> Result<RVal, R2Err> {
    const SINGULAR: &str = "singular matrix 'a' in solve";
    let args = matched(a, &["a", "b", "tol"], false)?;
    let Some(first) = args[0] else { return fail("argument \"a\" is missing, with no default") };
    if matches!(first, RVal::TypeInstance(_)) {
        return solve(&view(args[0], NOT_QR)?, args[1], SINGULAR);
    }
    let tol = match args[2] { Some(t) => t.scalar_f64()?.unwrap_or(DEFAULT_TOL), None => DEFAULT_TOL };
    let Some(m) = to_matrix(first)? else { return fail("qr.solve: 'a' must be a numeric matrix") };
    let made = decompose(m, tol)?;
    solve(&view(Some(&made), NOT_QR)?, args[1], SINGULAR)
}

/// `solve(a, b)` where `a` is a `qr` object.
pub fn solve_qr(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let args = matched(a, &["a", "b"], false)?;
    solve(&view(args[0], NOT_QR)?, args[1], "singular matrix 'a' in 'solve'")
}
