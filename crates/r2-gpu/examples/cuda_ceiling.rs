//! The card's own limits, with none of R2's work in the way. Each operation of a
//! step is held against these in `benchmarks/llm/llmoptreport.md`.
//!
//!     CEILING <what> <value> <unit>
//!
//!     cargo run --release -p r2-gpu --features cuda --example cuda_ceiling

use r2_gpu::cuda::ffi::{self, copy_both_ways, HostBuffer};
use r2_gpu::cuda::{ceiling, sync, Tensor};
use std::time::Instant;

/// Seconds per call: enough calls for 0.2 s closed by one fence, median of 5.
fn timed(mut f: impl FnMut() -> bool) -> f64 {
    assert!(f() && sync(), "kernel failed");
    let s = Instant::now();
    assert!(f() && sync(), "kernel failed");
    let reps = (0.2 / s.elapsed().as_secs_f64().max(1e-6)).ceil().clamp(1.0, 500.0) as usize;
    let mut v: Vec<f64> = (0..5).map(|_| {
        let s = Instant::now();
        for _ in 0..reps { assert!(f(), "kernel failed"); }
        assert!(sync(), "fence failed");
        s.elapsed().as_secs_f64() / reps as f64
    }).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[2]
}

fn say(what: &str, value: f64, unit: &str) { println!("CEILING {what} {value:.1} {unit}"); }

/// The rate of one kind of work held for 30 s.
fn sustain(kind: &str) {
    let out = Tensor::zeros(1 << 24).unwrap();
    let n = 1usize << 28;
    let (a, y) = (Tensor::upload(&vec![1.0f32; n]).unwrap(), Tensor::zeros(n).unwrap());
    let (t, k, m) = (4096usize, 768usize, 3072usize);
    // Constant operands toggle few bits, so they cost less energy than real ones.
    let operand = |n: usize, seed: u32| -> Vec<f32> {
        if kind != "cublas_random" { return vec![0.5; n]; }
        let mut x = seed;
        (0..n).map(|_| { x = x.wrapping_mul(1664525).wrapping_add(1013904223); (x >> 8) as f32 / (1u32 << 24) as f32 - 0.5 }).collect()
    };
    let (ga, gb, gc) = (Tensor::upload(&operand(t * k, 1)).unwrap(), Tensor::upload(&operand(k * m, 2)).unwrap(), Tensor::zeros(t * m).unwrap());
    let start = Instant::now();
    let mut units = 0.0f64;
    let unit = match kind { "fma" | "cublas" | "cublas_random" => "GFLOP/s", _ => "GB/s" };
    while start.elapsed().as_secs_f64() < 30.0 {
        for _ in 0..20 {
            let ok = match kind {
                "fma" => { units += 256.0 * 4096.0 * 1024.0 * 16.0; ceiling::fma(&out, 4096, 1024) }
                "cublas" | "cublas_random" => { units += 2.0 * (t * k * m) as f64; r2_gpu::cuda::gemm::gemm_cublas(&ga, false, &gb, false, t, k, m, &gc, false) }
                "read" => { units += (n * 4) as f64; ceiling::read(&a, &out, n, 768) }
                _ => { units += (n * 8) as f64; ceiling::copy(&a, &y, n, 768) }
            };
            assert!(ok, "kernel failed");
        }
        assert!(sync(), "fence failed");
    }
    say(&format!("sustained_{kind}"), units / start.elapsed().as_secs_f64() / 1e9, unit);
}

fn main() {
    println!("device: {}", r2_gpu::cuda::device_line());
    if ffi::device().is_err() { println!("no device"); return; }

    // `sustain <kind>`: 30 s of one kind of work, for a sampler to read power and
    // clock under it. The T4 has a 70 W cap, so a burst overstates what a step gets.
    if std::env::args().nth(1).as_deref() == Some("sustain") {
        let kind = std::env::args().nth(2).unwrap_or_default();
        sustain(&kind);
        return;
    }

    // registers: 2 FLOPs a multiply-add
    let out = Tensor::zeros(1 << 24).unwrap();
    for (blocks, iters) in [(256u32, 4096u32), (4096, 1024), (16384, 256)] {
        let s = timed(|| ceiling::fma(&out, blocks, iters));
        let flops = 256.0 * blocks as f64 * iters as f64 * 8.0 * 2.0;
        say(&format!("fma_registers_blocks_{blocks}"), flops / s / 1e9, "GFLOP/s");
    }

    // memory: sizes inside and past the caches, a warp walking runs of 768 elements
    let big = 1usize << 28;
    let (a, y) = (Tensor::upload(&vec![1.0f32; big]).unwrap(), Tensor::zeros(big).unwrap());
    for mb in [1usize, 16, 256, 1024] {
        let n = (mb << 20) / 4;
        let gb = |s: f64, bytes: usize| bytes as f64 / s / 1e9;
        say(&format!("read_{mb}MB"), gb(timed(|| ceiling::read(&a, &out, n, 768)), n * 4), "GB/s");
        say(&format!("write_{mb}MB"), gb(timed(|| ceiling::write(&y, n, 768)), n * 4), "GB/s");
        say(&format!("copy_{mb}MB"), gb(timed(|| ceiling::copy(&a, &y, n, 768)), n * 8), "GB/s");
    }

    // a tile's loads, on the 125M model's activations, stored by row and by column
    for (m, k) in [(4096usize, 768usize), (4096, 3072)] {
        let bytes = (m * k * 4) as f64;
        let across = timed(|| ceiling::tile(&a, &out, m, k, k, 1, false));
        let along = timed(|| ceiling::tile(&a, &out, m, k, k, 1, true));
        let by_col = timed(|| ceiling::tile(&a, &out, m, k, 1, m, false));
        say(&format!("tile_loads_{m}x{k}_by_row_lanes_across"), bytes / across / 1e9, "GB/s");
        say(&format!("tile_loads_{m}x{k}_by_row_lanes_along"), bytes / along / 1e9, "GB/s");
        say(&format!("tile_loads_{m}x{k}_by_column"), bytes / by_col / 1e9, "GB/s");
    }

    // the link, pinned host memory, each way and both at once
    let n = 1usize << 27;
    let (mut down, up) = (HostBuffer::new(n).unwrap(), HostBuffer::new(n).unwrap());
    let bytes = (n * 4) as f64;
    let once = |f: &mut dyn FnMut()| { let mut v: Vec<f64> = (0..5).map(|_| { let s = Instant::now(); f(); s.elapsed().as_secs_f64() }).collect(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[2] };
    say("pcie_down_512MB", bytes / once(&mut || a.copy_to(down.as_mut_slice()).unwrap()) / 1e9, "GB/s");
    say("pcie_up_512MB", bytes / once(&mut || y.copy_from(0, up.as_slice()).unwrap()) / 1e9, "GB/s");
    let both = once(&mut || {
        copy_both_ways(|d, u| {
            d.download(&a, 0, down.as_mut_slice())?;
            u.upload(&y, 0, up.as_slice())
        }).unwrap();
    });
    say("pcie_both_512MB_each", 2.0 * bytes / both / 1e9, "GB/s total");
}
