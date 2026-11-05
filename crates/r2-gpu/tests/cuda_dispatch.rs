//! `dispatch` and `matmul` through the public API, in a process of their own
//! because the opt-in flag is global.
//!
//! With a device they must really run on it. Without one they must give the
//! CPU's answer and say so. `R2_REQUIRE_CUDA` is set on the Kaggle GPU run, and
//! the same test runs there with `CUDA_VISIBLE_DEVICES` empty to cover the
//! fallback on Linux, where the library loads and finds no device.
#![cfg(feature = "cuda")]

use r2_gpu::cuda::ffi::device;
use r2_gpu::{cpu, dispatch, last_backend, matmul, set_gpu_enabled, Backend, Op, GPU_MIN_ELEMS};

#[test]
fn dispatch_and_matmul_report_where_they_ran() {
    let have = device().is_ok();
    if std::env::var_os("R2_REQUIRE_CUDA").is_some() {
        assert!(have, "R2_REQUIRE_CUDA is set but there is no device: {:?}", device().err());
    }
    let xs: Vec<f32> = (0..GPU_MIN_ELEMS).map(|i| i as f32 * 1e-3 - 30.0).collect();
    let want = cpu::map(Op::Scale(0.5), &xs);

    // off by default: never the device, whatever is present
    assert_eq!(dispatch(Op::Scale(0.5), &xs), want);
    assert_eq!(last_backend(), Backend::Cpu);
    assert!(matmul(&[1.0], &[1.0], 1, 1, 1).is_none());
    assert_eq!(last_backend(), Backend::Cpu);

    set_gpu_enabled(true);
    // below the threshold stays on the CPU even when enabled
    assert_eq!(dispatch(Op::Scale(0.5), &xs[..100]), want[..100]);
    assert_eq!(last_backend(), Backend::Cpu);

    let expect = if have { Backend::Cuda } else { Backend::Cpu };
    assert_eq!(dispatch(Op::Scale(0.5), &xs), want);
    assert_eq!(last_backend(), expect);

    let got = matmul(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2);
    assert_eq!(last_backend(), expect);
    assert_eq!(got, have.then(|| vec![19.0, 22.0, 43.0, 50.0]));
    set_gpu_enabled(false);
}
