//! What this machine's adapter is and the limits the kernels are sized
//! against: workgroup storage, invocations per workgroup, dispatch dims.
//!
//!     cargo run --release -p r2-gpu --features gpu --example probe

fn main() {
    println!("adapter: {}", r2_gpu::adapter_info());
    #[cfg(feature = "gpu")]
    if let Some(g) = r2_gpu::device::gpu() {
        let l = g.device.limits();
        println!("max_compute_workgroup_storage_size   {} bytes", l.max_compute_workgroup_storage_size);
        println!("max_compute_invocations_per_workgroup {}", l.max_compute_invocations_per_workgroup);
        println!("max_compute_workgroup_size            {} x {} x {}", l.max_compute_workgroup_size_x, l.max_compute_workgroup_size_y, l.max_compute_workgroup_size_z);
        println!("max_compute_workgroups_per_dimension  {}", l.max_compute_workgroups_per_dimension);
        println!("max_storage_buffer_binding_size       {} MB", l.max_storage_buffer_binding_size / (1 << 20));
        println!("max_buffer_size                       {} MB", l.max_buffer_size / (1 << 20));
        println!("features: subgroup {}", g.device.features().contains(wgpu::Features::SUBGROUP));
    }
}
