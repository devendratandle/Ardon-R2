//! One LAPACK-level routine at one size, for `benchmarks/lapack_cases.py`.
//!
//!     cargo run --release -p r2-linalg --example lapack_cases -- getrf 1000
//!
//! Routines: getrf (LU), potrf (Cholesky), geqrf (QR), syev (symmetric
//! eigen, with vectors), gesvd (singular values). Prints milliseconds per
//! call: median of timed calls after one warm call, at least 3 and ~1 s.

use r2_linalg::*;

fn mat(m: usize, n: usize, seed: u64) -> Vec<f64> {
    let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
    (0..m * n).map(|_| {
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (routine, n) = match &args[..] {
        [r, n] => (r.clone(), n.parse::<usize>().expect("n")),
        _ => { eprintln!("usage: lapack_cases <getrf|potrf|geqrf|syev|gesvd> <n>"); std::process::exit(2); }
    };
    let g = mat(n, n, 1);
    let general: Vec<f64> = { let mut a = g.clone(); for i in 0..n { a[i * n + i] += n as f64 * 0.5; } a };
    let symmetric: Vec<f64> = { let mut s = vec![0.0; n * n];
        for j in 0..n { for i in 0..n { s[j * n + i] = 0.5 * (g[j * n + i] + g[i * n + j]); } } s };
    let spd: Vec<f64> = { let mut a = vec![0.0; n * n];
        dgemm_t(n, &g, &mut a); for i in 0..n { a[i * n + i] += n as f64; } a };

    let mut run: Box<dyn FnMut()> = match routine.as_str() {
        "getrf" => Box::new(|| { let mut a = general.clone(); std::hint::black_box(dgetrf(n, &mut a).unwrap()); }),
        "potrf" => Box::new(|| { let mut a = spd.clone(); dpotrf(n, &mut a).unwrap(); std::hint::black_box(&a); }),
        "geqrf" => Box::new(|| { let mut a = general.clone(); std::hint::black_box(dgeqrf(n, n, &mut a).unwrap()); }),
        "syev"  => Box::new(|| { std::hint::black_box(dsyev_full(n, &symmetric).unwrap()); }),
        "gesvd" => Box::new(|| { std::hint::black_box(dgesvd(n, n, &general).unwrap()); }),
        r => { eprintln!("unknown routine {r}"); std::process::exit(2); }
    };
    run();
    let mut times = Vec::new();
    let start = std::time::Instant::now();
    while times.len() < 3 || (start.elapsed().as_secs_f64() < 1.0 && times.len() < 50) {
        let s = std::time::Instant::now();
        run();
        times.push(s.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{:.3}", times[times.len() / 2]);
}

/// a = gᵀ·g (n x n, column-major) with the library's own dgemm.
fn dgemm_t(n: usize, g: &[f64], a: &mut [f64]) {
    let mut gt = vec![0.0; n * n];
    for j in 0..n { for i in 0..n { gt[i * n + j] = g[j * n + i]; } }
    dgemm(n, n, n, 1.0, &gt, g, 0.0, a).unwrap();
}
