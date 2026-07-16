//! `cargo run -p r2-gpu --release --features gpu --example flops`
//! FLOPS/throughput: CPU reference vs GPU dispatch across small/medium/
//! large element-map workloads. Emits `key=value` lines (elems/sec).
use std::time::Instant;

fn time_best<F: Fn()>(reps: u32, f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        let dt = t.elapsed().as_secs_f64();
        if dt < best { best = dt; }
    }
    best
}

fn main() {
    println!("gpu.adapter={}", r2_gpu::adapter_info());
    let op = r2_gpu::Op::Sigmoid; // representative activation (exp inside)
    for &(name, n) in &[("small", 4_096usize), ("medium", 262_144), ("large", 4_194_304)] {
        let xs: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-4 - 20.0).collect();

        // CPU reference (always the truth).
        let cpu = time_best(5, || { let _ = r2_gpu::cpu::map(op, &xs); });
        println!("cpu.{}.secs={}", name, cpu);
        println!("cpu.{}.elems_per_sec={}", name, (n as f64) / cpu);

        // GPU dispatch (forced on; below threshold small still falls to CPU
        // by policy — that IS the correct behavior to report).
        r2_gpu::set_gpu_enabled(true);
        let gpu = time_best(5, || { let _ = r2_gpu::dispatch(op, &xs); });
        r2_gpu::set_gpu_enabled(false);
        let routed = n >= r2_gpu::GPU_MIN_ELEMS;
        println!("gpu.{}.secs={}", name, gpu);
        println!("gpu.{}.elems_per_sec={}", name, (n as f64) / gpu);
        println!("gpu.{}.routed_to_gpu={}", name, routed);

        // Accuracy check at this size.
        r2_gpu::set_gpu_enabled(true);
        let g = r2_gpu::dispatch(op, &xs);
        r2_gpu::set_gpu_enabled(false);
        let c = r2_gpu::cpu::map(op, &xs);
        let maxerr = g.iter().zip(&c).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("acc.{}.gpu_vs_cpu_maxerr={:.3e}", name, maxerr);
    }
}
