//! Does keeping weights on the device change the verdict?
//! Shapes are the ones fused training actually produces.
fn cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m*n];
    for i in 0..m { for p in 0..k { let x=a[i*k+p];
        let (br,cr)=(&b[p*n..p*n+n], &mut c[i*n..i*n+n]);
        for j in 0..n { cr[j]+=x*br[j]; } } }
    c
}
fn main() {
    r2_gpu::set_gpu_enabled(true);
    println!("adapter: {}\n", r2_gpu::adapter_info());
    println!("{:>16} {:>9} {:>11} {:>11} {:>9}", "shape", "cpu GF/s", "gpu GF/s", "resident", "gain");
    // (m,k,n): fused training shapes + a large reference.
    for &(m,k,n) in &[(512,384,1024),(512,1024,384),(512,384,384),(2048,2048,2048)] {
        let a: Vec<f32> = (0..m*k).map(|i| ((i%97) as f32-48.0)*0.01).collect();
        let b: Vec<f32> = (0..k*n).map(|i| ((i%89) as f32-44.0)*0.01).collect();
        let f = 2.0*m as f64*k as f64*n as f64;
        let reps = 20;

        let t=std::time::Instant::now();
        for _ in 0..reps { std::hint::black_box(cpu(&a,&b,m,k,n)); }
        let gcpu = f*reps as f64/t.elapsed().as_secs_f64()/1e9;

        let _ = r2_gpu::matmul(&a,&b,m,k,n);           // warm
        let t=std::time::Instant::now();
        for _ in 0..reps { std::hint::black_box(r2_gpu::matmul(&a,&b,m,k,n)); }
        let gfresh = f*reps as f64/t.elapsed().as_secs_f64()/1e9;

        // Weight uploaded ONCE, as a training loop would.
        let res = r2_gpu::upload(&b,k,n).expect("upload");
        let _ = r2_gpu::matmul_resident(&a,&res,m);
        let t=std::time::Instant::now();
        for _ in 0..reps { std::hint::black_box(r2_gpu::matmul_resident(&a,&res,m)); }
        let gres = f*reps as f64/t.elapsed().as_secs_f64()/1e9;

        // Correctness of the resident path against CPU.
        let want = cpu(&a,&b,m,k,n);
        let got = r2_gpu::matmul_resident(&a,&res,m).unwrap();
        let worst = want.iter().zip(&got).map(|(x,y)| (x-y).abs()/x.abs().max(1.0)).fold(0.0f32,f32::max);
        assert!(worst < 1e-4, "resident matmul wrong: {worst:e}");

        println!("{:>4}x{:<4}x{:<6} {:>9.1} {:>11.1} {:>11.1} {:>8.2}x",
                 m,k,n, gcpu, gfresh, gres, gres/gfresh);
    }
    println!("\n(resident = weight uploaded once, as a training loop does)");
}
