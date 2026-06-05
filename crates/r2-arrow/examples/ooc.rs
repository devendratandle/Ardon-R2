//! Out-of-core demo: reduce a packed-f64 file LARGER THAN RAM via
//! MmapColumnar. Writes ~8 GB (1e9 f64 = 1.0), memory-maps it, and sums.
//! If RAM stays far below 8 GB, the OS is demand-paging — proving the
//! mmap path scales beyond physical memory for streaming reductions.

use r2_arrow::MmapColumnar;
use std::io::Write;
use std::time::Instant;

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000_000);
    let path = std::env::temp_dir().join("r2_ooc_demo.bin");

    let t0 = Instant::now();
    {
        let f = std::fs::File::create(&path).expect("create file");
        let mut w = std::io::BufWriter::with_capacity(16 << 20, f);
        let chunk = vec![1.0f64; 1_000_000]; // 1e6 ones; we never hold the whole array
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(chunk.as_ptr() as *const u8, chunk.len() * 8) };
        for _ in 0..(n / chunk.len()) {
            w.write_all(bytes).expect("write");
        }
        w.flush().expect("flush");
    }
    println!("wrote {} GB in {:.1?}", n * 8 / 1_000_000_000, t0.elapsed());

    let t1 = Instant::now();
    let col = MmapColumnar::open(&path).expect("mmap open");
    let s = col.sum();
    let m = col.mean();
    println!(
        "mmap-reduced {} values ({} GB): sum={}, mean={} in {:.1?}",
        col.len(),
        col.len() * 8 / 1_000_000_000,
        s,
        m,
        t1.elapsed()
    );
    println!("expected sum = {} (correct = {})", n, (s - n as f64).abs() < 1.0);

    std::fs::remove_file(&path).ok();
}
