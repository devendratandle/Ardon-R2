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

/// Rebuilds `airquality.r2d`, all 153 rows, from R's own data written as CSV:
///   Rscript -e "write.csv(airquality, 'aq.csv', row.names = FALSE)"
///   R2_AIRQUALITY_CSV=aq.csv cargo test -p r2-base --release --test gen_datasets -- --ignored airquality_from_csv
/// Wind is numeric and the other columns integer, as in R.
#[test]
#[ignore]
fn airquality_from_csv() {
    use r2_types::{Attrs, DataFrame};
    use std::sync::Arc;
    let path = std::env::var("R2_AIRQUALITY_CSV").expect("set R2_AIRQUALITY_CSV to R's airquality as CSV");
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let names: Vec<String> = lines.next().unwrap().split(',').map(|s| s.trim_matches('"').to_string()).collect();
    let rows: Vec<Vec<&str>> = lines.filter(|l| !l.trim().is_empty()).map(|l| l.split(',').collect()).collect();
    assert_eq!(rows.len(), 153, "R's airquality has 153 rows");
    let columns = names.iter().enumerate().map(|(j, name)| {
        let cells = rows.iter().map(|r| r[j].trim());
        let col = if name == "Wind" {
            RVal::Numeric(cells.map(|c| c.parse::<f64>().ok()).collect::<Vec<_>>().into(), Attrs::default())
        } else {
            RVal::Integer(cells.map(|c| c.parse::<i32>().ok()).collect::<Vec<_>>().into(), Attrs::default())
        };
        (Arc::from(name.as_str()), col)
    }).collect();
    dump("airquality", RVal::DataFrame(DataFrame { columns, row_names: None }));
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
