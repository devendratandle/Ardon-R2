//! The CUDA kernels against the CPU ones, labelled with what really ran, so a
//! CPU fallback can't be read as a GPU figure: with no device this says so and
//! prints no GPU number. Each row's device side is checked by a fence that
//! reports faults. Operands are resident before the clock starts, each timing
//! covers `reps` launches closed by one fence, median of 5, and the two GEMM
//! paths alternate within one window.
//!
//!     cargo run --release -p r2-gpu --features cuda --example cuda_bench

use r2_autograd::Tape;
use r2_gpu::cuda::attention::{backward, force_fused, forward, profile_start, profile_take, Shape};
use r2_gpu::cuda::gemm::{force_path, gemm_cublas, gemm_tiled, last_path, Path};
use r2_gpu::cuda::optim::adam_step;
use r2_gpu::cuda::{self, ffi, Tensor};
use r2_gpu::{cpu, dispatch, last_backend, set_gpu_enabled, Op};
use r2_linalg::gemm::{sgemm_into, Trans};
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn fill(len: usize, ph: f32) -> Vec<f32> { (0..len).map(|i| ((i as f32) * 0.0007 + ph).sin()).collect() }

/// Seconds per call of `f`, which queues work: `reps` calls and one fence,
/// median of 5. `None` if a call failed or the fence reported a fault.
fn timed(reps: usize, mut f: impl FnMut() -> bool) -> Option<f64> {
    let mut ok = f() && cuda::sync();
    let v: Vec<f64> = (0..5).map(|_| {
        let s = Instant::now();
        for _ in 0..reps { ok &= f(); }
        ok &= cuda::sync();
        s.elapsed().as_secs_f64() / reps as f64
    }).collect();
    ok.then(|| median(v))
}

fn show(gflops: Option<f64>) -> String {
    gflops.map_or("failed".to_string(), |g| format!("{g:.1}"))
}

fn elementwise() {
    println!("\n-- dispatch (elementwise), 16M elements, ms per call --");
    println!("{:<14} {:>8} {:>10} {:>8}", "op", "backend", "ran ms", "CPU ms");
    set_gpu_enabled(true);
    let xs = fill(1 << 24, 0.0);
    for op in [Op::Relu, Op::Sigmoid, Op::Tanh, Op::Exp, Op::Scale(0.7)] {
        let time = |f: &dyn Fn() -> Vec<f32>| median((0..5).map(|_| {
            let s = Instant::now();
            std::hint::black_box(f());
            s.elapsed().as_secs_f64() * 1e3
        }).collect());
        let ran = time(&|| dispatch(op, &xs));
        let backend = last_backend();
        let c = time(&|| cpu::map(op, &xs));
        println!("{:<14} {:>8} {:>10.1} {:>8.1}", format!("{op:?}"), backend.to_string(), ran, c);
    }
    set_gpu_enabled(false);
}

fn gemms() {
    println!("\n-- GEMM, GFLOP/s, each column names its own path; repeat = bits equal across two runs --");
    println!("{:>18} {:>5} {:>10} {:>10} {:>10} {:>14}", "t x k x n", "case", "CUDA tiled", "cuBLAS", "CPU", "cuBLAS repeat");
    let shapes = [(2048usize, 256usize, 256usize), (2048, 256, 128), (2048, 256, 768), (2048, 768, 256), (2048, 256, 8000),
                  (2048, 768, 768), (2048, 768, 2304), (2048, 2304, 768), (2048, 768, 8000), (512, 768, 2304), (2048, 2048, 2048)];
    for &(t, k, n) in &shapes {
        let (a, b, g) = (fill(t * k, 0.0), fill(k * n, 1.0), fill(t * n, 2.0));
        let flops = 2.0 * t as f64 * k as f64 * n as f64;
        let reps = ((0.5 * 200e9) / flops).ceil().clamp(2.0, 100.0) as usize;
        // forward NN, grad_A NT, grad_B TN
        let cases: [(&str, &Vec<f32>, bool, &Vec<f32>, bool, usize, usize, usize); 3] =
            [("NN", &a, false, &b, false, t, k, n), ("NT", &g, false, &b, true, t, n, k), ("TN", &a, true, &g, false, k, t, n)];
        for (name, x, tx, y, ty, m, kk, nn) in cases {
            let (Some(dx), Some(dy), Some(dc)) = (Tensor::upload(x), Tensor::upload(y), Tensor::zeros(m * nn)) else {
                println!("upload failed"); return;
            };
            let (mut tiled, mut blas) = (Vec::new(), Vec::new());
            for _ in 0..3 {
                tiled.push(timed(reps, || gemm_tiled(&dx, tx, &dy, ty, m, kk, nn, &dc, false)));
                blas.push(timed(reps, || gemm_cublas(&dx, tx, &dy, ty, m, kk, nn, &dc, false)));
            }
            let rate = |v: Vec<Option<f64>>| -> Option<f64> { Some(flops / median(v.into_iter().collect::<Option<Vec<f64>>>()?) / 1e9) };
            let run = || { gemm_cublas(&dx, tx, &dy, ty, m, kk, nn, &dc, false); dc.download() };
            let repeat = if run() == run() { "same" } else { "DIFFERS" };
            let mut c = vec![0.0f32; m * nn];
            let (ta, tb) = (if tx { Trans::Yes } else { Trans::No }, if ty { Trans::Yes } else { Trans::No });
            let cpu_s = median((0..5).map(|_| {
                let s = Instant::now();
                sgemm_into(x, ta, y, tb, m, kk, nn, &mut c, true);
                s.elapsed().as_secs_f64()
            }).collect());
            println!("{:>18} {:>5} {:>10} {:>10} {:>10.1} {:>14}", if name == "NN" { format!("{t}x{k}x{n}") } else { String::new() },
                     name, show(rate(tiled)), show(rate(blas)), flops / cpu_s / 1e9, repeat);
        }
    }
}

fn attention() {
    println!("\n-- attention, ms per call, CUDA kernels vs the tape on the CPU --");
    println!("{:<26} {:>9} {:>9}", "shape", "fwd", "bwd");
    for &(nseq, seq, nh, nkv, hd, reps) in &[(32usize, 64usize, 4usize, 2usize, 64usize, 20usize), (8, 256, 12, 4, 64, 10)] {
        let rows = nseq * seq;
        let (qv, kv, vv, gv) = (fill(rows * nh * hd, 0.0), fill(rows * nkv * hd, 1.0), fill(rows * nkv * hd, 2.0), fill(rows * nh * hd, 3.0));
        let scale = 1.0 / (hd as f32).sqrt();
        let sh = Shape { nseq, seq, nh, nkv, hd, scale };
        let up = |x: &[f32]| Tensor::upload(x).unwrap();
        let z = |x: usize| Tensor::zeros(x).unwrap();
        let (tq, tk, tv, tg) = (up(&qv), up(&kv), up(&vv), up(&gv));
        let (to, tl, td) = (z(rows * nh * hd), z(rows * nh), z(rows * nh));
        let (dq, dk, dv) = (z(rows * nh * hd), z(rows * nkv * hd), z(rows * nkv * hd));
        let name = format!("{nseq}x{seq} {nh}/{nkv} hd{hd}");
        let ms = |x: Option<f64>| x.map_or("failed".to_string(), |s| format!("{:.2}", s * 1e3));
        // interleaved, a path's two rows apart: the split below drains after every
        // launch, so its sum is above the row's total and only its shares mean anything
        for (path, fused) in [(Path::Tiled, false), (Path::Tiled, true), (Path::Cublas, false), (Path::Tiled, false), (Path::Tiled, true), (Path::Cublas, false)] {
            force_path(Some(path));
            force_fused(fused);
            let fwd = timed(reps, || forward(&tq, &tk, &tv, &to, &tl, &sh));
            let bwd = timed(reps, || backward(&tq, &tk, &tv, &to, &tl, &tg, &td, &dq, &dk, &dv, &sh));
            profile_start();
            forward(&tq, &tk, &tv, &to, &tl, &sh);
            let f = profile_take().unwrap_or_default();
            profile_start();
            backward(&tq, &tk, &tv, &to, &tl, &tg, &td, &dq, &dk, &dv, &sh);
            let b = profile_take().unwrap_or_default();
            println!("{:<26} {:>9} {:>9} {:?} fused={fused} (ran {:?}) products/rows fwd {:.2}/{:.2} bwd {:.2}/{:.2}",
                     name, ms(fwd), ms(bwd), path, last_path(), f[0], f[1], b[0], b[1]);
            force_path(None);
            force_fused(false);
        }
        let leafs = |tp: &mut Tape| (tp.leaf(qv.clone(), true), tp.leaf(kv.clone(), true), tp.leaf(vv.clone(), true));
        let cpu_f = median((0..5).map(|_| {
            let mut tp = Tape::new();
            let (a, b, c) = leafs(&mut tp);
            let s = Instant::now();
            std::hint::black_box(tp.attention(a, b, c, nseq, seq, nh, nkv, hd, scale));
            s.elapsed().as_secs_f64() * 1e3
        }).collect());
        let cpu_fb = median((0..5).map(|_| {
            let mut tp = Tape::new();
            let (a, b, c) = leafs(&mut tp);
            let s = Instant::now();
            let o = tp.attention(a, b, c, nseq, seq, nh, nkv, hd, scale);
            tp.backward_from(o, &gv);
            s.elapsed().as_secs_f64() * 1e3
        }).collect());
        println!("{:<26} {:>9.2} {:>9.2} tape on the CPU", name, cpu_f, (cpu_fb - cpu_f).max(0.0));
    }
}

/// One product on the 125M model's real shapes, with the activations stored as
/// now (a row per token) and as cuBLAS's native column-major would have them.
fn layouts() {
    println!("\n-- layout, ms per call: activations by row (now) vs by column, cuBLAS, interleaved --");
    let Ok(blas) = ffi::blas() else { println!("no cuBLAS"); return };
    for &(t, k, n, name) in &[(4096usize, 768usize, 3072usize, "w1 forward"), (4096, 3072, 768, "w2 forward"),
                              (4096, 768, 768, "wq forward"), (2048, 768, 2304, "medium w1"), (2048, 2304, 768, "medium w2")] {
        let (a, b) = (Tensor::upload(&fill(t * k, 0.0)).unwrap(), Tensor::upload(&fill(k * n, 1.0)).unwrap());
        let c = Tensor::zeros(t * n).unwrap();
        let flops = 2.0 * (t * k * n) as f64;
        let reps = 20;
        let (mut by_row, mut by_col) = (Vec::new(), Vec::new());
        for _ in 0..5 {
            by_row.extend(timed(reps, || gemm_cublas(&a, false, &b, false, t, k, n, &c, false)));
            // the same numbers, read as column-major: a k x t, b n x k, c t x n
            by_col.extend(timed(reps, || blas.sgemm(true, true, t, n, k, &a, k, &b, n, 0.0, &c, t).is_ok()));
        }
        let (r, cc) = (median(by_row), median(by_col));
        println!("{name:<12} {t}x{k}x{n}: by row {:.2} ms ({:.0} GFLOP/s), by column {:.2} ms ({:.0} GFLOP/s)",
                 r * 1e3, flops / r / 1e9, cc * 1e3, flops / cc / 1e9);
    }
}

/// Host slices to host slices, copies included, against the CPU: where routing
/// to the device starts to pay.
fn crossovers() {
    println!("\n-- crossover, ms per call, copies included: device vs CPU --");
    let time = |f: &mut dyn FnMut()| median((0..5).map(|_| {
        let s = Instant::now();
        f();
        s.elapsed().as_secs_f64() * 1e3
    }).collect());
    set_gpu_enabled(true);
    for len in [1usize << 16, 1 << 18, 1 << 20, 1 << 22] {
        let xs = fill(len, 0.0);
        for op in [Op::Relu, Op::Scale(0.7), Op::Sigmoid, Op::Tanh, Op::Exp] {
            let d = time(&mut || { std::hint::black_box(dispatch(op, &xs)); });
            let ran = last_backend();
            let c = time(&mut || { std::hint::black_box(cpu::map(op, &xs)); });
            println!("map {:>8} {:<12} {:>6} {:>9.3} {:>9.3}", len, format!("{op:?}"), ran.to_string(), d, c);
        }
    }
    for n in [300usize, 400, 512, 640, 800, 1200] {
        let (a, b) = (fill(n * n, 0.0), fill(n * n, 1.0));
        let d = time(&mut || { std::hint::black_box(r2_gpu::matmul(&a, &b, n, n, n)); });
        let ran = last_backend();
        let mut out = vec![0.0f32; n * n];
        let c = time(&mut || sgemm_into(&a, Trans::No, &b, Trans::No, n, n, n, &mut out, true));
        println!("matmul {n}^3 ({:>5}M work) {:>6} {:>9.3} {:>9.3}", n * n * n / 1_000_000, ran.to_string(), d, c);
    }
    set_gpu_enabled(false);
}

fn adam() {
    println!("\n-- Adam, 40M parameters, ms per step --");
    let n = 40_000_000usize;
    let (tw, tg) = (Tensor::upload(&fill(n, 0.0)).unwrap(), Tensor::upload(&fill(n, 1.0)).unwrap());
    let (tm, tv) = (Tensor::zeros(n).unwrap(), Tensor::zeros(n).unwrap());
    let mut t = 0u32;
    let s = timed(5, || { t += 1; adam_step(&tw, &tg, &tm, &tv, n, 3e-4, 0.9, 0.999, 1e-8, t, 1.0, 0.0) });
    println!("CUDA, closed by the fence: {}", s.map_or("failed".to_string(), |s| format!("{:.2} ms", s * 1e3)));
    // The same steps closed by reading the weights back, which cannot return
    // before the kernels finish: a check that the fence really waits.
    let read = median((0..5).map(|_| { let s = Instant::now(); std::hint::black_box(tw.download()); s.elapsed().as_secs_f64() }).collect());
    let closed = median((0..5).map(|_| {
        let s = Instant::now();
        for _ in 0..20 { t += 1; adam_step(&tw, &tg, &tm, &tv, n, 3e-4, 0.9, 0.999, 1e-8, t, 1.0, 0.0); }
        std::hint::black_box(tw.download());
        s.elapsed().as_secs_f64()
    }).collect());
    println!("CUDA, closed by a download: {:.2} ms per step ({:.0} ms for 20 steps + read, read alone {:.0} ms)",
             (closed - read) / 20.0 * 1e3, closed * 1e3, read * 1e3);
    let w = tw.download();
    println!("weights after all steps: {} of {n} finite", w.iter().filter(|x| x.is_finite()).count());
}

fn main() {
    println!("backend: {}", cuda::device_line());
    if ffi::device().is_err() {
        println!("no GPU figures: this machine would run these on the CPU.");
        return;
    }
    println!("cuBLAS: {}", ffi::blas().map_or_else(|e| format!("not available ({e})"), |_| "loaded".to_string()));
    if std::env::args().nth(1).as_deref() == Some("layout") { layouts(); return; }
    elementwise();
    crossovers();
    gemms();
    attention();
    adam();
}
