//! `cargo run -p r2-gpu --features gpu --example probe`
//! Reports the adapter wgpu selects (integrated GPUs count) and checks a
//! kernel against the CPU reference. Proves offload works with no discrete
//! card required.
fn main() {
    println!("adapter: {}", r2_gpu::adapter_info());
    r2_gpu::set_gpu_enabled(true);
    let n = r2_gpu::GPU_MIN_ELEMS;
    let xs: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-3 - 30.0).collect();
    for op in [r2_gpu::Op::Relu, r2_gpu::Op::Sigmoid, r2_gpu::Op::Tanh] {
        let g = r2_gpu::dispatch(op, &xs);
        let c = r2_gpu::cpu::map(op, &xs);
        let maxerr = g.iter().zip(&c).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("{:?}: n={} maxerr(GPU vs CPU)={:.2e}", op, n, maxerr);
    }
}
