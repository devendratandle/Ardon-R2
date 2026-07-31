//! Show the Oracle deciding the backend for each matmul shape.
use r2_oracle::{dispatch, Op, Shape, Backend};
fn main() {
    for enabled in [false, true] {
        r2_oracle::set_gpu_enabled(enabled);
        println!("\ngpu_enabled = {enabled}");
        for &(m,k,n) in &[(16,768,768),(64,768,768),(256,256,256),(512,512,512),(1024,1024,1024)] {
            let b = dispatch(Op::MatMul, Shape::nmk(m,n,k));
            println!("  {:>5}x{:<5}x{:<5} work {:>12} -> {:?}", m,k,n, m*k*n, b);
        }
    }
    println!("\n(One component decides; kernels just obey.)");
}
