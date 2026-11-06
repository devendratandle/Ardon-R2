//! The CUDA backend (`--features cuda`), the same kernels as the wgpu path
//! with the same contract: opt-in, checked against the CPU reference, and
//! any failure falls back to the CPU rather than erroring.
//!
//! No crate and no link-time dependency: [`ffi`] loads `libcuda.so.1` at run
//! time and holds all the `unsafe`. Kernels are `.cu` files in `kernels/`,
//! compiled to PTX for `sm_75` by `kernels/build.sh`; the generated `.ptx` is
//! committed and embedded with `include_str!`, so building this crate needs
//! no `nvcc`, and the driver compiles the PTX for the GPU it finds.
//!
//! Storage is f32 only: the wgpu path's f16 storage exists for the bandwidth of
//! an integrated GPU and is not ported, so `R2_GPU_PREC=mixed` is refused here.
//! Of two or more GPUs, the first is used.

#![deny(unsafe_code)]

#[allow(unsafe_code)]
pub mod ffi;
pub mod attention;
pub mod cluster;
pub mod elementwise;
pub mod gemm;
pub mod optim;

use crate::Op;
use ffi::{device, Arg, DeviceBuffer, Kernel};
use std::sync::OnceLock;

/// A buffer of f32 on the device, as `device::Tensor` is for wgpu.
pub type Tensor = DeviceBuffer<f32>;

/// Storage types, as `device::Dtype`. Only f32 is served here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype { F32, F16 }

impl DeviceBuffer<f32> {
    /// Upload host data. `None` without a device or on allocation failure.
    pub fn upload(data: &[f32]) -> Option<Tensor> {
        Tensor::from_slice(device().ok()?, data).ok()
    }

    /// `len` zeroed elements.
    pub fn zeros(len: usize) -> Option<Tensor> {
        Tensor::zeroed(device().ok()?, len).ok()
    }

    /// `len` zeroed elements of `dtype`. `None` for f16, which this backend
    /// does not store: that is how a request for mixed precision is refused.
    pub fn zeros_as(len: usize, dtype: Dtype) -> Option<Tensor> {
        if dtype == Dtype::F32 { Tensor::zeros(len) } else { None }
    }

    /// Overwrite the buffer's start from the host; `false` if it failed.
    pub fn write(&self, data: &[f32]) -> bool { self.write_at(0, data) }

    /// Overwrite `data.len()` elements from element `at`; `false` if it failed.
    pub fn write_at(&self, at: usize, data: &[f32]) -> bool { self.copy_from(at, data).is_ok() }

    /// Read the buffer back, waiting for every kernel launched before this.
    /// Empty on any failure, as with `device::Tensor::download`.
    pub fn download(&self) -> Vec<f32> { self.to_vec().unwrap_or_default() }
}

/// Block until the device is idle: the fence a benchmark closes on. `false`
/// when there is no device or a kernel faulted, so a timing can be thrown out.
pub fn sync() -> bool {
    device().is_ok_and(|d| d.sync().is_ok())
}

/// One line about the device, or why there is none.
pub fn device_line() -> String {
    match device() {
        Ok(d) => format!("{} (CUDA, driver API {}.{}, {} MB, device 0 of {})",
                         d.name, d.driver / 1000, d.driver % 1000 / 10, d.memory >> 20, d.count),
        Err(why) => format!("no GPU: no CUDA device ({why}), CPU fallback active"),
    }
}

/// The kernels of one PTX module, loaded once on first use. A failure sticks
/// and callers take the CPU path; [`module_status`] reports the reason.
type Loaded<T> = OnceLock<Result<T, String>>;

/// The contents of `cell`, built from the PTX module `ptx` by `make`.
fn loaded<T>(cell: &'static Loaded<T>, ptx: &str, make: impl FnOnce(&ffi::Module) -> Result<T, String>) -> Result<&'static T, &'static str> {
    cell.get_or_init(|| {
        let module = device().map_err(str::to_string)?.load_module(ptx)?;
        make(&module)
    }).as_ref().map_err(|e| e.as_str())
}

const MAP_PTX: &str = include_str!("kernels/map.ptx");
const MAP_ARITY: usize = 4;
const MAP_THREADS: u32 = 256;

static MAP: Loaded<Kernel> = OnceLock::new();

fn map_kernel() -> Result<&'static Kernel, &'static str> {
    loaded(&MAP, MAP_PTX, |m| m.kernel("map_op", MAP_ARITY))
}

/// Whether each PTX module loads on this device, by name, with the reason if
/// not: what `cuda_probe` prints.
pub fn module_status() -> Vec<(&'static str, Result<(), String>)> {
    let st = |r: Result<(), &str>| r.map_err(str::to_string);
    vec![("map", st(map_kernel().map(|_| ()))),
         ("gemm", st(gemm::loads())),
         ("elementwise", st(elementwise::loads())),
         ("attention", st(attention::loads())),
         ("optim", st(optim::loads()))]
}

fn op_code(op: Op) -> (u32, f32) {
    match op {
        Op::Identity => (0, 0.0),
        Op::Relu => (1, 0.0),
        Op::Sigmoid => (2, 0.0),
        Op::Tanh => (3, 0.0),
        Op::Exp => (4, 0.0),
        Op::Scale(a) => (5, a),
    }
}

/// Whether `op` is worth the two copies at all. On a Tesla T4 the device took
/// about 2x the CPU's time for these at every size from 64K to 4M elements, while
/// Sigmoid, Exp and Tanh beat it from 64K up: they are cheap enough that the copies
/// are the whole cost. Measured by `cuda_bench`; `try_map` itself still serves them.
pub fn pays_for_copies(op: Op) -> bool { !matches!(op, Op::Identity | Op::Relu | Op::Scale(_)) }

/// Run `op` over `xs` on the device. `None` on ANY failure, so that
/// `dispatch` falls back to the CPU: a result only comes back after the
/// download has confirmed the kernel did not fault.
pub fn try_map(op: Op, xs: &[f32]) -> Option<Vec<f32>> {
    let n = u32::try_from(xs.len()).ok()?;
    if n == 0 { return Some(Vec::new()); }
    let k = map_kernel().ok()?;
    let t = Tensor::upload(xs)?;
    let (code, a) = op_code(op);
    k.launch([n.div_ceil(MAP_THREADS), 1, 1], [MAP_THREADS, 1, 1],
             &[Arg::F(&t), Arg::Int(n), Arg::Int(code), Arg::Real(a)]).ok()?;
    t.to_vec().ok()
}

/// For tests: a device is here, or the run did not ask for one. With
/// `R2_REQUIRE_CUDA` set (the Kaggle run sets it) a missing device is a
/// failure, so a CPU fallback can never pass for a GPU test run.
#[cfg(test)]
pub(crate) fn have_device() -> bool {
    match device() {
        Ok(_) => true,
        Err(why) => {
            assert!(std::env::var_os("R2_REQUIRE_CUDA").is_none(), "R2_REQUIRE_CUDA is set but: {why}");
            eprintln!("no CUDA device ({why}); skipped");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cpu, GPU_MIN_ELEMS};

    /// Distance in ULPs between two f32s; two NaNs are equal.
    fn ulps(a: f32, b: f32) -> u64 {
        if a.is_nan() && b.is_nan() { return 0; }
        if a.is_nan() || b.is_nan() { return u64::MAX; }
        let ord = |x: f32| -> i64 {
            let i = x.to_bits() as i32;
            (if i < 0 { i32::MIN.wrapping_sub(i) } else { i }) as i64
        };
        (ord(a) - ord(b)).unsigned_abs()
    }

    /// The grid the wgpu test uses, reaching the exp overflow-free range, and
    /// the values that special-case: zeros, tiny, huge, inf, NaN.
    fn inputs() -> Vec<f32> {
        let mut xs: Vec<f32> = (0..GPU_MIN_ELEMS + 7).map(|i| (i as f32) * 1e-3 - 30.0).collect();
        xs.extend_from_slice(&[0.0, -0.0, 1e-30, -1e-30, 88.0, -87.0, 1e30, -1e30, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
        xs
    }

    /// Every op against `cpu::map`, each in the terms it can promise: exact
    /// for the ones that do no rounding beyond one IEEE operation, a few ULP
    /// for the ones that go through the two sides' own libm.
    #[test]
    fn every_op_matches_the_cpu_reference() {
        if !have_device() { return; }
        let xs = inputs();
        let cases: [(Op, u64); 7] = [(Op::Identity, 0), (Op::Relu, 0), (Op::Scale(0.7), 0), (Op::Scale(-3.0), 0),
                                     (Op::Sigmoid, 4), (Op::Tanh, 4), (Op::Exp, 4)];
        for (op, max_ulp) in cases {
            let got = try_map(op, &xs).expect("the device ran the kernel");
            let want = cpu::map(op, &xs);
            assert_eq!(got.len(), want.len());
            let worst = got.iter().zip(&want).map(|(g, w)| ulps(*g, *w)).max().unwrap();
            let at = got.iter().zip(&want).position(|(g, w)| ulps(*g, *w) == worst).unwrap();
            println!("{op:?}: worst {worst} ULP at x = {}", xs[at]);
            assert!(worst <= max_ulp, "{op:?}: {worst} ULP at x = {}: device {} vs cpu {}", xs[at], got[at], want[at]);
        }
    }

    #[test]
    fn empty_and_one_element_inputs() {
        if !have_device() { return; }
        assert_eq!(try_map(Op::Relu, &[]), Some(vec![]));
        assert_eq!(try_map(Op::Scale(2.0), &[1.5]), Some(vec![3.0]));
    }

    /// A buffer's writes are bounds-checked, partial writes land where asked,
    /// and dropping many buffers leaves the device usable.
    #[test]
    fn device_buffer_round_trips_and_checks_bounds() {
        if !have_device() { return; }
        let t = Tensor::zeros(8).unwrap();
        assert_eq!(t.download(), vec![0.0; 8]);
        assert!(t.write_at(6, &[1.0, 2.0]));
        assert!(!t.write_at(7, &[1.0, 2.0]), "a write past the end must be refused");
        assert_eq!(t.download(), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
        for _ in 0..1000 { drop(Tensor::zeros(1 << 16).unwrap()); }
        assert_eq!(Tensor::upload(&[4.0]).unwrap().download(), vec![4.0]);
    }

    /// The module declares the arity the host passes, and a wrong count of
    /// arguments is refused on the host side, before the driver reads them.
    #[test]
    fn a_wrong_argument_count_is_refused() {
        if !have_device() { return; }
        let k = map_kernel().unwrap();
        assert!(k.launch([1, 1, 1], [MAP_THREADS, 1, 1], &[Arg::Int(0)]).is_err());
        let m = device().unwrap().load_module(MAP_PTX).unwrap();
        assert!(m.kernel("map_op", MAP_ARITY + 1).is_err(), "more parameters than the PTX declares");
        assert!(m.kernel("map_op", MAP_ARITY - 1).is_err(), "fewer parameters than the PTX declares");
        assert!(m.kernel("no_such_kernel", 0).is_err());
    }
}
