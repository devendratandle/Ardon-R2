//! Tiny helper to write a sample Parquet file for end-to-end testing of
//! `read.parquet`. Run: `cargo run -p r2-arrow --features parquet
//! --example make_parquet -- sample.parquet`

#[cfg(feature = "parquet")]
fn main() {
    use arrow::array::{BooleanArray, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let path = std::env::args().nth(1).unwrap_or_else(|| "sample.parquet".to_string());
    let id = Int64Array::from(vec![1i64, 2, 3, 4, 5]);
    let x = Float64Array::from(vec![Some(1.1), Some(2.2), None, Some(4.4), Some(5.5)]);
    let grp = StringArray::from(vec!["a", "b", "a", "b", "a"]);
    let flag = BooleanArray::from(vec![true, false, true, true, false]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("x", DataType::Float64, true),
        Field::new("grp", DataType::Utf8, false),
        Field::new("flag", DataType::Boolean, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(id), Arc::new(x), Arc::new(grp), Arc::new(flag)],
    )
    .unwrap();
    let file = std::fs::File::create(&path).unwrap();
    let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    println!("wrote {} (5 rows: id i64, x f64+NA, grp utf8, flag bool)", path);
}

#[cfg(not(feature = "parquet"))]
fn main() {
    eprintln!("rebuild with --features parquet");
}
