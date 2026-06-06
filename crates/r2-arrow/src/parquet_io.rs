//! Parquet reader — Phase F.6 interop.
//!
//! Pure-Rust, via the `parquet` + `arrow` crates (no C / FFI). Reads a
//! Parquet file **row-group by row-group** and materialises each column
//! into a neutral [`ParquetCol`] (f64 / bool / utf8) that the engine
//! turns into an `RVal` data-frame column. Row-group streaming means peak
//! decode memory is one row group, not the whole file — so very large
//! files import with bounded RAM (the values still accumulate into the
//! returned vectors for the in-RAM data-frame path; a future variant can
//! stream each column straight to an mmap file for true out-of-core).
//!
//! Type mapping (Parquet/Arrow → R):
//!   * any integer / float  → `f64`   (R's numeric)
//!   * boolean              → logical
//!   * utf8 / large-utf8    → character
//!   * anything else (dates, timestamps, decimals, …) → cast to utf8
//!     so the column still imports as readable strings rather than failing.

use std::path::Path;

use arrow::array::{Array, BooleanArray, Float64Array, LargeStringArray, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// One imported column in a neutral, R-friendly representation.
pub enum ParquetCol {
    /// Numeric column (every integer/float Parquet type maps here).
    F64(Vec<Option<f64>>),
    /// Logical column (Parquet `BOOLEAN`).
    Bool(Vec<Option<bool>>),
    /// Character column (utf8 / large-utf8, plus any type cast to string).
    Utf8(Vec<Option<String>>),
}

/// A whole Parquet file decoded into columns + names.
pub struct ParquetTable {
    /// Column names, in file order.
    pub names: Vec<String>,
    /// Column data, parallel to `names`.
    pub columns: Vec<ParquetCol>,
    /// Total row count across all row groups.
    pub nrows: usize,
}

fn is_numeric(dt: &DataType) -> bool {
    use DataType::*;
    matches!(
        dt,
        Int8 | Int16 | Int32 | Int64
            | UInt8 | UInt16 | UInt32 | UInt64
            | Float16 | Float32 | Float64
    )
}

/// Read a Parquet file into a [`ParquetTable`].
pub fn read_parquet<P: AsRef<Path>>(path: P) -> Result<ParquetTable, String> {
    let file = std::fs::File::open(&path).map_err(|e| {
        format!("read.parquet: cannot open '{}': {}", path.as_ref().display(), e)
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("read.parquet: {}", e))?;
    let schema = builder.schema().clone();
    let ncols = schema.fields().len();
    let names: Vec<String> =
        schema.fields().iter().map(|f| f.name().to_string()).collect();

    // One growing accumulator per column, typed from the schema.
    enum Acc {
        F64(Vec<Option<f64>>),
        Bool(Vec<Option<bool>>),
        Utf8(Vec<Option<String>>),
    }
    let mut accs: Vec<Acc> = schema
        .fields()
        .iter()
        .map(|f| match f.data_type() {
            DataType::Boolean => Acc::Bool(Vec::new()),
            DataType::Utf8 | DataType::LargeUtf8 => Acc::Utf8(Vec::new()),
            dt if is_numeric(dt) => Acc::F64(Vec::new()),
            _ => Acc::Utf8(Vec::new()), // dates / timestamps / decimals → string
        })
        .collect();

    let reader = builder.build().map_err(|e| format!("read.parquet: {}", e))?;
    let mut nrows = 0usize;
    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("read.parquet: {}", e))?;
        nrows += batch.num_rows();
        for ci in 0..ncols {
            let array = batch.column(ci);
            match &mut accs[ci] {
                Acc::Bool(v) => {
                    let a = array
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .ok_or_else(|| {
                            format!("read.parquet: column '{}' is not boolean", names[ci])
                        })?;
                    for i in 0..a.len() {
                        v.push(if a.is_null(i) { None } else { Some(a.value(i)) });
                    }
                }
                Acc::F64(v) => {
                    let casted = cast(array, &DataType::Float64).map_err(|e| {
                        format!("read.parquet: column '{}': {}", names[ci], e)
                    })?;
                    let a = casted.as_any().downcast_ref::<Float64Array>().unwrap();
                    for i in 0..a.len() {
                        v.push(if a.is_null(i) { None } else { Some(a.value(i)) });
                    }
                }
                Acc::Utf8(v) => {
                    if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
                        for i in 0..a.len() {
                            v.push(if a.is_null(i) { None } else { Some(a.value(i).to_string()) });
                        }
                    } else if let Some(a) = array.as_any().downcast_ref::<LargeStringArray>() {
                        for i in 0..a.len() {
                            v.push(if a.is_null(i) { None } else { Some(a.value(i).to_string()) });
                        }
                    } else {
                        let casted = cast(array, &DataType::Utf8).map_err(|e| {
                            format!("read.parquet: column '{}': {}", names[ci], e)
                        })?;
                        let a = casted.as_any().downcast_ref::<StringArray>().unwrap();
                        for i in 0..a.len() {
                            v.push(if a.is_null(i) { None } else { Some(a.value(i).to_string()) });
                        }
                    }
                }
            }
        }
    }

    let columns = accs
        .into_iter()
        .map(|a| match a {
            Acc::F64(v) => ParquetCol::F64(v),
            Acc::Bool(v) => ParquetCol::Bool(v),
            Acc::Utf8(v) => ParquetCol::Utf8(v),
        })
        .collect();

    Ok(ParquetTable { names, columns, nrows })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BooleanArray, Int64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    #[test]
    fn parquet_roundtrip_mixed_types_with_nulls() {
        let path = std::env::temp_dir()
            .join(format!("r2arrow_pq_{}.parquet", std::process::id()));
        let _ = std::fs::remove_file(&path);

        // f64 with a null, int64 (→ f64), bool with a null, utf8 with a null.
        let fcol = Float64Array::from(vec![Some(1.5), None, Some(3.5)]);
        let icol = Int64Array::from(vec![Some(10), Some(20), Some(30)]);
        let bcol = BooleanArray::from(vec![Some(true), Some(false), None]);
        let scol = StringArray::from(vec![Some("a"), Some("b"), None]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("f", DataType::Float64, true),
            Field::new("i", DataType::Int64, true),
            Field::new("b", DataType::Boolean, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(fcol), Arc::new(icol), Arc::new(bcol), Arc::new(scol)],
        )
        .unwrap();
        {
            let file = std::fs::File::create(&path).unwrap();
            let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
            w.write(&batch).unwrap();
            w.close().unwrap();
        }

        let t = read_parquet(&path).unwrap();
        assert_eq!(t.nrows, 3);
        assert_eq!(t.names, vec!["f", "i", "b", "s"]);
        match &t.columns[0] {
            ParquetCol::F64(v) => assert_eq!(v, &vec![Some(1.5), None, Some(3.5)]),
            _ => panic!("f not f64"),
        }
        match &t.columns[1] {
            // int64 maps to numeric (f64)
            ParquetCol::F64(v) => assert_eq!(v, &vec![Some(10.0), Some(20.0), Some(30.0)]),
            _ => panic!("i not f64"),
        }
        match &t.columns[2] {
            ParquetCol::Bool(v) => assert_eq!(v, &vec![Some(true), Some(false), None]),
            _ => panic!("b not bool"),
        }
        match &t.columns[3] {
            ParquetCol::Utf8(v) => {
                assert_eq!(v, &vec![Some("a".to_string()), Some("b".to_string()), None])
            }
            _ => panic!("s not utf8"),
        }
        std::fs::remove_file(&path).ok();
    }
}
