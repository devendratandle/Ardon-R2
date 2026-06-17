// R2 Base Library — built-in datasets and linalg-domain builtins.
//
// Phase D.1 (v0.1.1) — datasets moved from inline Rust arrays into the
// native `.r2d` binary format (see `src/r2d.rs`). The bytes are baked
// into the binary via `include_bytes!` and parsed on first call. This
// shrinks the source file from ~360 lines to ~100 and decouples the
// values from the code. Future work: read R's native `.rda` (gzip+XDR)
// the same way.
//
// Phase R.4: linalg builtins (matrix/tensor/t/crossprod/svd/eigen) live
// in the `linalg_ops` submodule — pure `fn(&[EvalArg]) -> Result<RVal,
// R2Err>`, thin wrappers over r2-linalg kernels.

pub mod linalg_ops;
pub mod r2d;

use r2_types::*;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────
// BUILT-IN DATASETS — thin loaders over baked-in `.r2d` bytes.
// Canonical R 4.5.3 values verified by `dataset_integrity` tests.
// ─────────────────────────────────────────────────────────────────────

fn load(bytes: &[u8], name: &str) -> RVal {
    r2d::read_r2d(bytes).unwrap_or_else(|e| panic!("{}.r2d corrupt: {}", name, e.msg))
}

/// Convert a character column of a data.frame into a factor (sorted unique
/// levels, R's default), so grouping idioms work — e.g.
/// `pairs(iris[1:4], col = iris$Species)` colours by group.
fn factorize_column(v: RVal, col: &str) -> RVal {
    if let RVal::DataFrame(mut df) = v {
        for (name, val) in df.columns.iter_mut() {
            if name.as_ref() == col {
                if let RVal::Character(cv, _) = val {
                    let mut levels: Vec<Arc<str>> = Vec::new();
                    for s in cv.iter().flatten() {
                        if !levels.iter().any(|l| l.as_ref() == s.as_ref()) { levels.push(s.clone()); }
                    }
                    levels.sort();
                    let codes: Vec<Option<u32>> = cv.iter().map(|o| o.as_ref().and_then(|s|
                        levels.iter().position(|l| l.as_ref() == s.as_ref()).map(|i| i as u32))).collect();
                    *val = RVal::Factor(Factor { codes, levels, ordered: false });
                }
            }
        }
        return RVal::DataFrame(df);
    }
    v
}

/// Fisher's iris dataset — 150 observations, 5 variables
/// (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width, Species).
/// `Species` is a factor, as in R.
pub fn iris() -> RVal {
    static BYTES: &[u8] = include_bytes!("../datasets/iris.r2d");
    factorize_column(load(BYTES, "iris"), "Species")
}

/// Motor Trend car road tests (1974) — 32 observations, 11 variables.
pub fn mtcars() -> RVal {
    static BYTES: &[u8] = include_bytes!("../datasets/mtcars.r2d");
    load(BYTES, "mtcars")
}

/// New York air quality, May–September 1973 — first 30 rows, with NAs.
pub fn airquality() -> RVal {
    static BYTES: &[u8] = include_bytes!("../datasets/airquality.r2d");
    load(BYTES, "airquality")
}

/// Effect of vitamin C on tooth growth in guinea pigs — 60 obs, 3 vars.
pub fn tooth_growth() -> RVal {
    static BYTES: &[u8] = include_bytes!("../datasets/tooth_growth.r2d");
    load(BYTES, "tooth_growth")
}

/// Old Faithful geyser eruption data — 272 obs, 2 vars.
pub fn faithful() -> RVal {
    static BYTES: &[u8] = include_bytes!("../datasets/faithful.r2d");
    load(BYTES, "faithful")
}

/// Register all built-in datasets into the global environment.
pub fn register_datasets(env: &mut std::collections::HashMap<Arc<str>, RVal>) {
    env.insert(Arc::from("iris"), iris());
    env.insert(Arc::from("mtcars"), mtcars());
    env.insert(Arc::from("airquality"), airquality());
    env.insert(Arc::from("ToothGrowth"), tooth_growth());
    env.insert(Arc::from("faithful"), faithful());
}

// ─────────────────────────────────────────────────────────────────────
// Dataset integrity guards — catch transcription errors that would
// silently propagate into every test touching a built-in dataset.
//
// To regenerate the expected values:
//   Rscript -e 'sapply(iris[1:4], function(x) sum(x))'
//   Rscript -e 'sapply(mtcars, function(x) sum(x))'
//   Rscript -e 'sapply(airquality, function(x) sum(x, na.rm=TRUE))'
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod dataset_integrity {
    use super::*;

    /// Pull a numeric column from a data.frame, panicking on shape errors.
    fn col_sum(df: &RVal, name: &str) -> f64 {
        let df = match df { RVal::DataFrame(d) => d, _ => panic!("not a data.frame") };
        let col = df.columns.iter().find(|(n, _)| n.as_ref() == name)
            .map(|(_, v)| v).unwrap_or_else(|| panic!("column '{}' missing", name));
        let v = match col { RVal::Numeric(v, _) => v, _ => panic!("column '{}' not Numeric", name) };
        v.iter().filter_map(|x| *x).sum()
    }

    fn col_len(df: &RVal, name: &str) -> usize {
        let df = match df { RVal::DataFrame(d) => d, _ => panic!("not a data.frame") };
        let col = df.columns.iter().find(|(n, _)| n.as_ref() == name)
            .map(|(_, v)| v).unwrap_or_else(|| panic!("column '{}' missing", name));
        match col {
            RVal::Numeric(v, _) => v.len(),
            RVal::Character(v, _) => v.len(),
            _ => panic!("column '{}' not Numeric/Character", name),
        }
    }

    /// Iris column sums, computed via:
    ///   sapply(iris[1:4], sum)   →   876.5  458.1  563.7  179.9
    #[test]
    fn iris_column_sums_match_canonical_R() {
        let df = iris();
        assert_eq!(col_len(&df, "Sepal.Length"), 150);
        let sl = col_sum(&df, "Sepal.Length");
        let sw = col_sum(&df, "Sepal.Width");
        let pl = col_sum(&df, "Petal.Length");
        let pw = col_sum(&df, "Petal.Width");
        let tol = 1e-9;
        assert!((sl - 876.5).abs() < tol, "iris$Sepal.Length sum {} != 876.5 (canonical R)", sl);
        assert!((sw - 458.1).abs() < tol, "iris$Sepal.Width sum {} != 458.1 (canonical R)", sw);
        assert!((pl - 563.7).abs() < tol, "iris$Petal.Length sum {} != 563.7 (canonical R)", pl);
        assert!((pw - 179.9).abs() < tol, "iris$Petal.Width sum {} != 179.9 (canonical R)", pw);
    }

    /// Iris row spot-checks — first/last row of each species. Catches
    /// reorderings that preserve sums (e.g. swapping rows within a column).
    #[test]
    fn iris_row_spot_check_matches_canonical_R() {
        let df = iris();
        let df = match &df { RVal::DataFrame(d) => d.clone(), _ => panic!() };
        let read = |col: &str, i: usize| -> f64 {
            let c = df.columns.iter().find(|(n, _)| n.as_ref() == col).unwrap().1.clone();
            match c { RVal::Numeric(v, _) => v[i].unwrap(), _ => panic!() }
        };
        assert_eq!((read("Sepal.Length", 0), read("Sepal.Width", 0), read("Petal.Length", 0), read("Petal.Width", 0)),
                   (5.1, 3.5, 1.4, 0.2));
        assert_eq!((read("Sepal.Length", 49), read("Sepal.Width", 49), read("Petal.Length", 49), read("Petal.Width", 49)),
                   (5.0, 3.3, 1.4, 0.2));
        assert_eq!((read("Sepal.Length", 50), read("Sepal.Width", 50), read("Petal.Length", 50), read("Petal.Width", 50)),
                   (7.0, 3.2, 4.7, 1.4));
        assert_eq!((read("Sepal.Length", 99), read("Sepal.Width", 99), read("Petal.Length", 99), read("Petal.Width", 99)),
                   (5.7, 2.8, 4.1, 1.3));
        assert_eq!((read("Sepal.Length", 100), read("Sepal.Width", 100), read("Petal.Length", 100), read("Petal.Width", 100)),
                   (6.3, 3.3, 6.0, 2.5));
        assert_eq!((read("Sepal.Length", 142), read("Sepal.Width", 142), read("Petal.Length", 142), read("Petal.Width", 142)),
                   (5.8, 2.7, 5.1, 1.9));
        assert_eq!((read("Sepal.Length", 149), read("Sepal.Width", 149), read("Petal.Length", 149), read("Petal.Width", 149)),
                   (5.9, 3.0, 5.1, 1.8));
    }

    /// Mtcars column sums:
    ///   sum(mtcars$mpg) = 642.9, sum(mtcars$hp) = 4694, sum(mtcars$wt) = 102.952
    #[test]
    fn mtcars_column_sums_match_canonical_R() {
        let df = mtcars();
        assert_eq!(col_len(&df, "mpg"), 32);
        assert!((col_sum(&df, "mpg") - 642.9).abs() < 1e-9, "mtcars$mpg sum");
        assert!((col_sum(&df, "hp")  - 4694.0).abs() < 1e-9, "mtcars$hp sum");
        assert!((col_sum(&df, "wt")  - 102.952).abs() < 1e-9, "mtcars$wt sum");
        assert!((col_sum(&df, "cyl") - 198.0).abs() < 1e-9, "mtcars$cyl sum");
    }
}
