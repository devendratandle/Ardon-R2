//! The CUDA backend reached from R2 script: `gpu.map` runs on the device and
//! `gpu.status()` says so. `llm_finetune.rs` covers `device = "gpu"` training.
#![cfg(feature = "cuda")]

use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::RVal;

fn run(script: &str) -> RVal {
    let mut e = Engine::new();
    let mut last = RVal::Null;
    for ex in Parser::parse(script).expect("parse ok") {
        last = e.eval(&ex).unwrap_or_else(|err| panic!("eval error: {}", err.msg));
    }
    last
}

fn text(v: &RVal) -> String {
    match v {
        RVal::Character(s, _) => s[0].as_ref().expect("a string").to_string(),
        other => panic!("expected a string, got {other:?}"),
    }
}

#[test]
fn gpu_map_runs_on_the_device_and_status_reports_it() {
    let have = r2_gpu::cuda::ffi::device().is_ok();
    if std::env::var_os("R2_REQUIRE_CUDA").is_some() { assert!(have, "R2_REQUIRE_CUDA is set but there is no device"); }
    let backend = text(&run("gpu.enable(TRUE)\nx <- gpu.map(\"scale\", rep(1.5, 70000), 2)\n\
                             stopifnot(all(x == 3))\ngpu.status()$last.backend"));
    assert_eq!(backend, if have { "CUDA" } else { "CPU" });
    let device = text(&run("gpu.status()$device"));
    assert_eq!(device.contains("no GPU"), !have, "{device}");

    // the tape's f32 matmul: gpu.enable reaches the Oracle, so a big enough one
    // goes to the device, and gives the CPU kernel's answer either way
    let (m, k, n) = (401usize, 402usize, 403usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.031).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.017).cos()).collect();
    let got = r2_tensor::ops::matmul(&a, &b, m, k, n);
    assert_eq!(r2_gpu::last_backend().to_string(), if have { "CUDA" } else { "CPU" });
    run("gpu.enable(FALSE)");
    let cpu = r2_tensor::ops::matmul(&a, &b, m, k, n);
    assert!(!r2_oracle::gpu_enabled(), "gpu.enable(FALSE) must route back");
    let scale = cpu.iter().fold(1.0f32, |s, x| s.max(x.abs()));
    assert!(got.iter().zip(&cpu).all(|(x, y)| (x - y).abs() <= 1e-5 * scale));
}
