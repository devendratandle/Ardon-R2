//! The GPU `sgemm` against the CPU one, on the shapes a training step
//! runs, on this machine's adapter.
//!
//! Operands are resident on the device before the clock starts — the way
//! a training loop holds them — and each timing covers `reps` back-to-back
//! launches closed by one fence, so launch overhead is amortised the way a
//! step amortises it. The CPU side is `r2_linalg::gemm::sgemm`, the kernel
//! that beats MKL on these shapes. GFLOP/s per case, median of 5.
//!
//!     cargo run --release -p r2-gpu --features gpu --example gemm_bench

use r2_gpu::device::{sync, Tensor};
use r2_gpu::gemm::gemm;
use r2_linalg::gemm::{sgemm_into, Trans};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    println!("adapter: {}", r2_gpu::adapter_info());
    println!("{:>18} {:>5} {:>10} {:>10} {:>8}", "t x k x n", "case", "GPU GF/s", "CPU GF/s", "GPU/CPU");
    println!("{}", "-".repeat(56));
    let shapes = [(2048usize, 256usize, 768usize), (2048, 768, 768), (2048, 768, 2304),
                  (2048, 2304, 768), (2048, 768, 8000), (512, 768, 2304), (2048, 2048, 2048)];
    for &(t, k, n) in &shapes {
        let fill = |len: usize, ph: f32| -> Vec<f32> { (0..len).map(|i| ((i as f32) * 0.0007 + ph).sin()).collect() };
        let a = fill(t * k, 0.0);          // activations t x k
        let b = fill(k * n, 1.0);          // weight k x n
        let g = fill(t * n, 2.0);          // upstream gradient t x n
        let flops = 2.0 * t as f64 * k as f64 * n as f64;
        let reps = ((0.5 * 200e9) / flops).ceil().clamp(2.0, 100.0) as usize;
        // (a, ta, b, tb, m, k, n): forward NN, grad_A NT, grad_B TN
        let cases: [(&str, &Vec<f32>, bool, &Vec<f32>, bool, usize, usize, usize); 3] = [
            ("NN", &a, false, &b, false, t, k, n),
            ("NT", &g, false, &b, true, t, n, k),
            ("TN", &a, true, &g, false, k, t, n),
        ];
        for (name, x, tx, y, ty, m, kk, nn) in cases {
            let (Some(dx), Some(dy), Some(dc)) = (Tensor::upload(x), Tensor::upload(y), Tensor::zeros(m * nn)) else {
                println!("no GPU"); return;
            };
            gemm(&dx, tx, &dy, ty, m, kk, nn, &dc, false); sync();
            let gpu_s = median((0..5).map(|_| {
                let s = std::time::Instant::now();
                for _ in 0..reps { gemm(&dx, tx, &dy, ty, m, kk, nn, &dc, false); }
                sync();
                s.elapsed().as_secs_f64() / reps as f64
            }).collect());
            let mut c = vec![0.0f32; m * nn];
            let (ta, tb) = (if tx { Trans::Yes } else { Trans::No }, if ty { Trans::Yes } else { Trans::No });
            sgemm_into(x, ta, y, tb, m, kk, nn, &mut c, true);
            let cpu_s = median((0..5).map(|_| {
                let s = std::time::Instant::now();
                for _ in 0..reps.min(10) { sgemm_into(x, ta, y, tb, m, kk, nn, &mut c, true); }
                s.elapsed().as_secs_f64() / reps.min(10) as f64
            }).collect());
            println!("{:>18} {:>5} {:>10.1} {:>10.1} {:>8.2}",
                     if name == "NN" { format!("{t}x{k}x{n}") } else { String::new() },
                     name, flops / gpu_s / 1e9, flops / cpu_s / 1e9, cpu_s / gpu_s);
        }
    }
}
