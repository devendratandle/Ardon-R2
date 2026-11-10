// Regenerates the `.r2d` binary dataset files from the current
// (inline or already-loaded) dataset functions in r2-base.
//
// Run with:
//   cargo test -p r2-base --release --test gen_datasets -- --ignored
//
// The test is `#[ignore]` so it does not run by default; it is a
// one-shot generator, not part of the regular test suite.

use r2_base::{iris, mtcars, airquality, tooth_growth, faithful};
use r2_base::r2d::write_r2d;
use r2_types::RVal;

fn dump(name: &str, v: RVal) {
    let df = match v {
        RVal::DataFrame(d) => d,
        _ => panic!("{}: not a data.frame", name),
    };
    let bytes = write_r2d(&df);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join(format!("{}.r2d", name));
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    eprintln!("wrote {} ({} bytes)", path.display(), bytes.len());
}

/// One dataset from R's own data written as CSV (`write.csv(x, f, row.names =
/// FALSE)`): `ints` columns integer, `texts` columns text, the rest numeric,
/// as R types them.
fn from_csv(name: &str, path: &std::path::Path, rows_expected: usize, ints: &[&str], texts: &[&str]) {
    use r2_types::{Attrs, DataFrame};
    use std::sync::Arc;
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {}", path.display(), e));
    let mut lines = text.lines();
    let names: Vec<String> = lines.next().unwrap().split(',').map(|s| s.trim_matches('"').to_string()).collect();
    let rows: Vec<Vec<&str>> = lines.filter(|l| !l.trim().is_empty()).map(|l| l.split(',').collect()).collect();
    assert_eq!(rows.len(), rows_expected, "R's {} has {} rows", name, rows_expected);
    let columns = names.iter().enumerate().map(|(j, col_name)| {
        let cells = rows.iter().map(|r| r[j].trim());
        let col = if ints.contains(&col_name.as_str()) {
            RVal::Integer(cells.map(|c| c.parse::<i32>().ok()).collect::<Vec<_>>().into(), Attrs::default())
        } else if texts.contains(&col_name.as_str()) {
            RVal::Character(cells.map(|c| Some(Arc::from(c.trim_matches('"')))).collect(), Attrs::default())
        } else {
            RVal::Numeric(cells.map(|c| c.parse::<f64>().ok()).collect::<Vec<_>>().into(), Attrs::default())
        };
        (Arc::from(col_name.as_str()), col)
    }).collect();
    dump(name, RVal::DataFrame(DataFrame { columns, row_names: None }));
}

/// Rebuilds airquality, faithful and iris from R's own data:
///   Rscript -e "for (d in c('airquality','faithful','iris')) write.csv(get(d), file.path('dir', paste0(d, '.csv')), row.names = FALSE)"
///   R2_DATASETS_CSV=dir cargo test -p r2-base --release --test gen_datasets -- --ignored datasets_from_r
#[test]
#[ignore]
fn datasets_from_r() {
    let dir = std::path::PathBuf::from(std::env::var("R2_DATASETS_CSV").expect("set R2_DATASETS_CSV to a folder of R's datasets as CSV"));
    from_csv("airquality", &dir.join("airquality.csv"), 153, &["Ozone", "Solar.R", "Temp", "Month", "Day"], &[]);
    from_csv("faithful", &dir.join("faithful.csv"), 272, &[], &[]);
    from_csv("iris", &dir.join("iris.csv"), 150, &[], &["Species"]);
}

#[test]
#[ignore]
fn generate_all() {
    dump("iris", iris());
    dump("mtcars", mtcars());
    dump("airquality", airquality());
    dump("tooth_growth", tooth_growth());
    dump("faithful", faithful());
}
