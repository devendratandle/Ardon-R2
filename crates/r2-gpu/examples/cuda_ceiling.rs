//! The card's own limits, with none of R2's work in the way. Each operation of a
//! step is held against these in `benchmarks/llm/llmoptreport.md`.
//!
//!     CEILING <what> <value> <unit>
//!
//!     cargo run --release -p r2-gpu --features cuda --example cuda_ceiling

use r2_gpu::cuda::ffi::{self, copy_both_ways, GemmForm, HostBuffer};
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

/// IEEE f16 from f32, rounded to nearest, for values well inside its range.
fn to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let abs = x.abs();
    if abs < 6.1e-5 { return sign | (abs * 16_777_216.0 + 0.5) as u16; }   // subnormal
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    let r = ((e as u32) << 10) + (m >> 13) + ((m >> 12) & 1);
    sign | r as u16
}

fn from_f16(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 31) as i32, (h & 1023) as f32);
    let v = if e == 0 { m * 2f32.powi(-24) } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) };
    if s == 1 { -v } else { v }
}

/// f16 values in f32 slots, two to a slot, as they lie in device memory.
fn pack_f16(x: &[f32]) -> Vec<f32> {
    x.chunks(2).map(|c| {
        let (lo, hi) = (to_f16(c[0]) as u32, c.get(1).map_or(0, |y| to_f16(*y)) as u32);
        f32::from_bits(lo | (hi << 16))
    }).collect()
}

fn unpack_f16(x: &[f32], n: usize) -> Vec<f32> {
    x.iter().flat_map(|s| { let b = s.to_bits(); [from_f16(b as u16), from_f16((b >> 16) as u16)] }).take(n).collect()
}

/// What tensor-core arithmetic costs in accuracy on one random product: each form's
/// error against f32, relative to the product's own size.
fn gemm_ex_accuracy() {
    let (m, k, n) = (4096usize, 768usize, 3072usize);
    let random = |len: usize, seed: u32| -> Vec<f32> {
        let mut x = seed;
        (0..len).map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        }).collect()
    };
    let (a, b) = (random(m * k, 1), random(k * n, 2));
    let Ok(blas) = ffi::blas() else { println!("no cuBLAS"); return };
    let (ta, tb) = (Tensor::upload(&a).unwrap(), Tensor::upload(&b).unwrap());
    let reference = Tensor::zeros(m * n).unwrap();
    blas.gemm_ex(GemmForm::F32, m, n, k, &ta, &tb, &reference).expect("f32");
    let exact = reference.download();
    let report = |name: &str, got: &[f32]| {
        let (mut e2, mut r2, mut worst) = (0.0f64, 0.0f64, 0.0f64);
        for (g, r) in got.iter().zip(&exact) {
            let d = (*g - *r) as f64;
            e2 += d * d;
            r2 += (*r as f64) * (*r as f64);
            worst = worst.max(d.abs());
        }
        println!("CEILING accuracy_{name} rms {:.3e} worst {:.3e} of rms value {:.3e}", (e2 / r2).sqrt(), worst, (r2 / exact.len() as f64).sqrt());
    };
    let fast = Tensor::zeros(m * n).unwrap();
    match blas.gemm_ex(GemmForm::F32Fast16, m, n, k, &ta, &tb, &fast) {
        Ok(()) => report("fast16", &fast.download()),
        Err(e) => println!("fast16 refused: {e}"),
    }
    let (ha, hb, hc) = (Tensor::upload(&pack_f16(&a)).unwrap(), Tensor::upload(&pack_f16(&b)).unwrap(), Tensor::zeros((m * n).div_ceil(2)).unwrap());
    match blas.gemm_ex(GemmForm::F16, m, n, k, &ha, &hb, &hc) {
        Ok(()) => report("f16", &unpack_f16(&hc.download(), m * n)),
        Err(e) => println!("f16 refused: {e}"),
    }
    match blas.gemm_ex(GemmForm::F16Sum32, m, n, k, &ha, &hb, &hc) {
        Ok(()) => report("f16_sum32", &unpack_f16(&hc.download(), m * n)),
        Err(e) => println!("f16_sum32 refused: {e}"),
    }
}

/// What a launch costs, without the kernel's work: a nearly empty block, queued
/// without a fence, and with one per launch, the round trip a read-back pays.
fn launch_cost() {
    let out = Tensor::zeros(1 << 16).unwrap();
    let queued = timed(|| ceiling::fma(&out, 1, 1));
    say("launch_queued_us", queued * 1e6, "us");
    let mut v: Vec<f64> = (0..200).map(|_| {
        let s = Instant::now();
        assert!(ceiling::fma(&out, 1, 1) && sync(), "kernel failed");
        s.elapsed().as_secs_f64()
    }).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    say("launch_with_fence_us", v[100] * 1e6, "us");
}

/// The rate of one kind of work held for 30 s.
fn sustain(kind: &str) {
    let out = Tensor::zeros(1 << 24).unwrap();
    let n = 1usize << 28;
    let (a, y) = (Tensor::upload(&vec![1.0f32; n]).unwrap(), Tensor::zeros(n).unwrap());
    let (t, k, m) = (4096usize, 768usize, 3072usize);
    // Constant operands toggle few bits, so they cost less energy than real ones.
    // `cublas_bits_N` keeps N significant bits of each random operand.
    let form = match kind { "gemm_ex_f32" => Some(GemmForm::F32), "gemm_ex_fast16" => Some(GemmForm::F32Fast16), "gemm_ex_f16" => Some(GemmForm::F16), "gemm_ex_f16_sum32" => Some(GemmForm::F16Sum32), _ => None };
    let bits = if kind == "cublas_random" || form.is_some() { Some(24) } else { kind.strip_prefix("cublas_bits_").and_then(|b| b.parse::<u32>().ok()) };
    let operand = |n: usize, seed: u32| -> Vec<f32> {
        let Some(bits) = bits else { return vec![0.5; n] };
        let mask = !0u32 << (24 - bits);
        let mut x = seed;
        (0..n).map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            f32::from_bits(((x >> 8) as f32 / (1u32 << 24) as f32 - 0.5).to_bits() & mask)
        }).collect()
    };
    let stored = |x: Vec<f32>| if matches!(form, Some(GemmForm::F16 | GemmForm::F16Sum32)) { pack_f16(&x) } else { x };
    let (ga, gb, gc) = (Tensor::upload(&stored(operand(t * k, 1))).unwrap(), Tensor::upload(&stored(operand(k * m, 2))).unwrap(), Tensor::zeros(t * m).unwrap());
    let start = Instant::now();
    let mut units = 0.0f64;
    let unit = if kind == "fma" || kind.starts_with("cublas") || form.is_some() { "GFLOP/s" } else { "GB/s" };
    while start.elapsed().as_secs_f64() < 30.0 {
        for _ in 0..20 {
            let ok = match kind {
                "fma" => { units += 256.0 * 4096.0 * 1024.0 * 16.0; ceiling::fma(&out, 4096, 1024) }
                s if s.starts_with("gemm_ex") => {
                    units += 2.0 * (t * k * m) as f64;
                    if let Err(e) = ffi::blas().expect("cuBLAS").gemm_ex(form.unwrap(), t, m, k, &ga, &gb, &gc) {
                        println!("{kind} refused: {e}");
                        return;
                    }
                    true
                }
                s if s.starts_with("cublas") => { units += 2.0 * (t * k * m) as f64; r2_gpu::cuda::gemm::gemm_cublas(&ga, false, &gb, false, t, k, m, &gc, false) }
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

    // `accuracy` and `launch`: one figure each, no sampler needed.
    match std::env::args().nth(1).as_deref() {
        Some("accuracy") => { gemm_ex_accuracy(); return; }
        Some("launch") => { launch_cost(); return; }
        _ => {}
    }

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
