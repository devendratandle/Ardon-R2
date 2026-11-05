//! What the CUDA backend finds on this machine: the device name, whether
//! each embedded kernel module loads, whether cuBLAS is there, and which
//! backend `dispatch` really used. Prints a clear line and exits 0 on a
//! machine with no CUDA.
//!
//!     cargo run --release -p r2-gpu --features cuda --example cuda_probe

use r2_gpu::cuda::{self, ffi};
use r2_gpu::{dispatch, last_backend, set_gpu_enabled, Op, GPU_MIN_ELEMS};

fn main() {
    println!("device: {}", cuda::device_line());
    for (name, status) in cuda::module_status() {
        match status {
            Ok(()) => println!("module {name}: loaded"),
            Err(why) => println!("module {name}: FAILED: {why}"),
        }
    }
    match ffi::blas() {
        Ok(_) => println!("cuBLAS: loaded"),
        Err(why) => println!("cuBLAS: not available ({why})"),
    }
    set_gpu_enabled(true);
    let xs = vec![0.5f32; GPU_MIN_ELEMS];
    let out = dispatch(Op::Scale(2.0), &xs);
    let right = out.iter().all(|&x| x == 1.0);
    println!("dispatch(Scale(2), {} elements): backend ran = {}, correct = {right}", xs.len(), last_backend());
}
