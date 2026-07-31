//! iGPU vs CPU matmul: correctness first, then throughput.
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m { for p in 0..k {
        let aik = a[i*k+p];
        let (br, cr) = (&b[p*n..p*n+n], &mut c[i*n..i*n+n]);
        for j in 0..n { cr[j] += aik * br[j]; }
    }}
    c
}
fn main() {
    r2_gpu::set_gpu_enabled(true);
    println!("adapter: {}\n", r2_gpu::adapter_info());
    println!("{:>6} {:>6} {:>6} {:>10} {:>10} {:>9} {:>9}",
             "m","k","n","cpu ms","gpu ms","cpu GF/s","gpu GF/s");
    for &(m,k,n) in &[(256,256,256),(512,512,512),(1024,1024,1024),(2048,2048,2048)] {
        let a: Vec<f32> = (0..m*k).map(|i| ((i%97) as f32 - 48.0) * 0.01).collect();
        let b: Vec<f32> = (0..k*n).map(|i| ((i%89) as f32 - 44.0) * 0.01).collect();
        let t0 = std::time::Instant::now();
        let c_cpu = cpu_matmul(&a,&b,m,k,n);
        let dt_c = t0.elapsed().as_secs_f64();

        let _ = r2_gpu::matmul(&a,&b,m,k,n);            // warm up (device init)
        let t1 = std::time::Instant::now();
        let g = r2_gpu::matmul(&a,&b,m,k,n);
        let dt_g = t1.elapsed().as_secs_f64();

        match g {
            Some(c_gpu) => {
                // Correctness gate: the GPU must match the CPU reference.
                let mut worst = 0.0f32;
                for (x,y) in c_cpu.iter().zip(&c_gpu) {
                    worst = worst.max((x-y).abs() / x.abs().max(1.0));
                }
                assert!(worst < 1e-4, "GPU disagrees with CPU: rel err {worst:e}");
                let f = 2.0*m as f64*k as f64*n as f64;
                println!("{:>6} {:>6} {:>6} {:>10.1} {:>10.1} {:>9.1} {:>9.1}",
                         m,k,n, dt_c*1e3, dt_g*1e3, f/dt_c/1e9, f/dt_g/1e9);
            }
            None => println!("{:>6} {:>6} {:>6} {:>10.1} {:>10} {:>9.1} {:>9}",
                             m,k,n, dt_c*1e3, "n/a", 2.0*m as f64*k as f64*n as f64/dt_c/1e9, "-"),
        }
    }
    println!("\n(GPU results verified against the CPU reference to <1e-4 relative)");
}
