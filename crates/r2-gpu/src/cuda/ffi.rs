//! The only `unsafe` in the CUDA backend.
//!
//! `libcuda.so.1` and `libcublas.so.13` are found at run time with `dlopen`
//! and `dlsym`, declared here by hand with no crate, so `r2-gpu` builds on a
//! machine with no CUDA and such a machine simply has no device. Everything
//! above this file sees safe types: [`Device`], [`DeviceBuffer`] (frees on
//! drop), [`Kernel`] and [`Blas`].
//!
//! # Why this surface is sound
//!
//! Host memory is touched only by the copies, which are bounds-checked here
//! against both the slice and the buffer. A kernel cannot reach host memory:
//! a bad index in device code faults the GPU context, the next call returns
//! an error, and callers turn that into the CPU path. A launch reads one host
//! value per parameter the kernel declares, so [`Kernel::launch`] refuses any
//! other argument count, and every argument has its own 8-byte slot, so a
//! mis-sized parameter cannot read past it.
//!
//! All work goes on the context's default stream, which runs in order and is
//! ordered with the blocking copies: no stream objects and no events. Targets
//! other than Unix have no loader here and report "no device".

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::marker::PhantomData;
use std::sync::{Mutex, OnceLock};

type CuResult = c_int;
type DevPtr = u64;
type Handle = *mut c_void;

#[cfg(unix)]
mod dl {
    use super::*;
    #[cfg_attr(target_os = "linux", link(name = "dl"))]
    extern "C" {
        pub fn dlopen(name: *const c_char, flag: c_int) -> *mut c_void;
        pub fn dlsym(lib: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    pub const RTLD_NOW: c_int = 2;
}

/// The library called `name`, or `None`. It stays loaded for the process.
fn open_lib(name: &str) -> Option<Handle> {
    #[cfg(unix)]
    {
        let c = CString::new(name).ok()?;
        // SAFETY: `c` is a valid C string; dlopen has no other precondition.
        let h = unsafe { dl::dlopen(c.as_ptr(), dl::RTLD_NOW) };
        if h.is_null() { None } else { Some(h) }
    }
    #[cfg(not(unix))]
    { let _ = name; None }
}

/// The address of a symbol, or `None`.
fn raw_sym(lib: Handle, name: &str) -> Option<*mut c_void> {
    #[cfg(unix)]
    {
        let c = CString::new(name).ok()?;
        // SAFETY: `lib` came from dlopen and is never closed; `c` is valid.
        let p = unsafe { dl::dlsym(lib, c.as_ptr()) };
        if p.is_null() { None } else { Some(p) }
    }
    #[cfg(not(unix))]
    { let _ = (lib, name); None }
}

/// A library's function table: one typed pointer per symbol, all resolved up
/// front, so a missing symbol is one load error and not a later crash.
macro_rules! table {
    ($name:ident { $($f:ident = $sym:literal : fn($($a:ty),*) -> $r:ty;)* }) => {
        struct $name { $($f: unsafe extern "C" fn($($a),*) -> $r,)* }
        impl $name {
            fn load(lib: Handle) -> Result<Self, String> {
                Ok(Self { $($f: {
                    let p = raw_sym(lib, $sym).ok_or_else(|| format!("missing symbol {}", $sym))?;
                    // SAFETY: the symbol's C signature is the one declared in
                    // this table, taken from the CUDA headers.
                    unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($a),*) -> $r>(p) }
                },)* })
            }
        }
    };
}

table!(Cu {
    init = "cuInit": fn(c_uint) -> CuResult;
    device_count = "cuDeviceGetCount": fn(*mut c_int) -> CuResult;
    device_get = "cuDeviceGet": fn(*mut c_int, c_int) -> CuResult;
    device_name = "cuDeviceGetName": fn(*mut c_char, c_int, c_int) -> CuResult;
    device_mem = "cuDeviceTotalMem_v2": fn(*mut usize, c_int) -> CuResult;
    driver_version = "cuDriverGetVersion": fn(*mut c_int) -> CuResult;
    ctx_create = "cuCtxCreate_v2": fn(*mut Handle, c_uint, c_int) -> CuResult;
    ctx_set = "cuCtxSetCurrent": fn(Handle) -> CuResult;
    ctx_sync = "cuCtxSynchronize": fn() -> CuResult;
    alloc = "cuMemAlloc_v2": fn(*mut DevPtr, usize) -> CuResult;
    free = "cuMemFree_v2": fn(DevPtr) -> CuResult;
    h_to_d = "cuMemcpyHtoD_v2": fn(DevPtr, *const c_void, usize) -> CuResult;
    d_to_h = "cuMemcpyDtoH_v2": fn(*mut c_void, DevPtr, usize) -> CuResult;
    set32 = "cuMemsetD32_v2": fn(DevPtr, c_uint, usize) -> CuResult;
    module_load = "cuModuleLoadDataEx": fn(*mut Handle, *const c_void, c_uint, *mut c_int, *mut *mut c_void) -> CuResult;
    module_fn = "cuModuleGetFunction": fn(*mut Handle, Handle, *const c_char) -> CuResult;
    param_info = "cuFuncGetParamInfo": fn(Handle, usize, *mut usize, *mut usize) -> CuResult;
    launch = "cuLaunchKernel": fn(Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle, *mut *mut c_void, *mut *mut c_void) -> CuResult;
    err_str = "cuGetErrorString": fn(CuResult, *mut *const c_char) -> CuResult;
});

table!(Cb {
    create = "cublasCreate_v2": fn(*mut Handle) -> c_int;
    sgemm = "cublasSgemm_v2": fn(Handle, c_int, c_int, c_int, c_int, c_int, *const f32, DevPtr, c_int, DevPtr, c_int, *const f32, DevPtr, c_int) -> c_int;
    sgemm_strided = "cublasSgemmStridedBatched": fn(Handle, c_int, c_int, c_int, c_int, c_int, *const f32, DevPtr, c_int, i64, DevPtr, c_int, i64, *const f32, DevPtr, c_int, i64, c_int) -> c_int;
});

/// One GPU with its context: the process-wide CUDA device.
pub struct Device {
    cu: Cu,
    ctx: Handle,
    /// `cuDeviceGetName`.
    pub name: String,
    /// GPUs CUDA can see. This backend uses the first.
    pub count: usize,
    /// Device memory in bytes.
    pub memory: usize,
    /// The driver's CUDA API version, `major * 1000 + minor * 10`.
    pub driver: i32,
}

// SAFETY: the context handle is an opaque id that CUDA lets any thread make
// current, which `bind` does before every call; the rest is plain data.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    fn err(&self, what: &str, code: CuResult) -> String {
        let mut p: *const c_char = std::ptr::null();
        // SAFETY: `p` is a valid out pointer; a returned string is static.
        let text = unsafe {
            if (self.cu.err_str)(code, &mut p) == 0 && !p.is_null() {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            } else { String::new() }
        };
        format!("{what}: error {code} {text}")
    }

    /// Make the context current on this thread. "Current" is per thread in
    /// CUDA, and any thread may call in (a test harness certainly does).
    fn bind(&self) -> Result<(), String> {
        // SAFETY: `ctx` is live for the whole process.
        match unsafe { (self.cu.ctx_set)(self.ctx) } { 0 => Ok(()), c => Err(self.err("cuCtxSetCurrent", c)) }
    }

    /// Wait for everything launched so far, and report a fault it hit.
    pub fn sync(&self) -> Result<(), String> {
        self.bind()?;
        // SAFETY: no arguments; the context is current.
        match unsafe { (self.cu.ctx_sync)() } { 0 => Ok(()), c => Err(self.err("cuCtxSynchronize", c)) }
    }

    /// Load PTX text. The driver JIT-compiles it and keeps a disk cache.
    pub fn load_module(&'static self, ptx: &str) -> Result<Module, String> {
        self.bind()?;
        let src = CString::new(ptx).map_err(|_| "PTX contains a NUL".to_string())?;
        let mut log = vec![0u8; 8192];
        // CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES; the
        // size travels as an integer in a pointer, by CUDA's own rule
        let mut opts: [c_int; 2] = [5, 6];
        let mut vals: [*mut c_void; 2] = [log.as_mut_ptr().cast(), log.len() as *mut c_void];
        let mut m: Handle = std::ptr::null_mut();
        // SAFETY: `src` is NUL-terminated PTX; `opts` and `vals` hold the two
        // entries stated; the log buffer outlives the call and its size is given.
        let code = unsafe { (self.cu.module_load)(&mut m, src.as_ptr().cast(), 2, opts.as_mut_ptr(), vals.as_mut_ptr()) };
        if code != 0 {
            let end = log.iter().position(|&b| b == 0).unwrap_or(log.len());
            return Err(format!("{}: {}", self.err("cuModuleLoadDataEx", code), String::from_utf8_lossy(&log[..end])));
        }
        Ok(Module { dev: self, raw: m })
    }
}

/// A loaded PTX module. It lives for the process, as do its kernels.
pub struct Module { dev: &'static Device, raw: Handle }
// SAFETY: an opaque id, used only with the context bound.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Module {
    /// The entry point `name`, checked to declare exactly `arity` parameters.
    pub fn kernel(&self, name: &str, arity: usize) -> Result<Kernel, String> {
        self.dev.bind()?;
        let c = CString::new(name).map_err(|e| e.to_string())?;
        let mut f: Handle = std::ptr::null_mut();
        // SAFETY: `f` is a valid out pointer, the module is live, `c` is valid.
        let code = unsafe { (self.dev.cu.module_fn)(&mut f, self.raw, c.as_ptr()) };
        if code != 0 { return Err(self.dev.err(name, code)); }
        let (mut off, mut size) = (0usize, 0usize);
        // SAFETY: `f` is the function just found; the out pointers are valid.
        // Asking about a parameter past the end returns an error and writes nothing.
        let has = |i: usize, off: &mut usize, size: &mut usize| unsafe { (self.dev.cu.param_info)(f, i, off, size) == 0 };
        let last_ok = arity == 0 || has(arity - 1, &mut off, &mut size);
        if !last_ok || has(arity, &mut off, &mut size) {
            return Err(format!("{name}: the PTX does not declare {arity} parameters"));
        }
        Ok(Kernel { dev: self.dev, raw: f, arity })
    }
}

/// One launch argument: a device pointer or a 32-bit scalar.
#[derive(Clone, Copy)]
pub enum Arg<'a> {
    F(&'a DeviceBuffer<f32>),
    U(&'a DeviceBuffer<u32>),
    Int(u32),
    Real(f32),
}

/// A kernel entry point.
pub struct Kernel { dev: &'static Device, raw: Handle, arity: usize }
// SAFETY: an opaque id, used only with the context bound.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    /// Queue `grid` blocks of `block` threads each. Returns once queued; a
    /// fault inside the kernel surfaces at the next sync or download.
    pub fn launch(&self, grid: [u32; 3], block: [u32; 3], args: &[Arg]) -> Result<(), String> {
        if args.len() != self.arity {
            return Err(format!("kernel takes {} arguments, got {}", self.arity, args.len()));
        }
        self.dev.bind()?;
        // every value in its own 8-byte slot; the driver copies them out
        // before cuLaunchKernel returns
        let mut slots: Vec<u64> = args.iter().map(|a| match *a {
            Arg::F(b) => b.ptr,
            Arg::U(b) => b.ptr,
            Arg::Int(v) => v as u64,
            Arg::Real(v) => v.to_bits() as u64,
        }).collect();
        let mut ptrs: Vec<*mut c_void> = slots.iter_mut().map(|s| (s as *mut u64).cast()).collect();
        // SAFETY: one pointer per declared parameter (`arity` is checked
        // against the PTX in `Module::kernel` and against `args` above), each
        // to a live 8-byte slot. Stream and extra are null: the default stream.
        let code = unsafe {
            (self.dev.cu.launch)(self.raw, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0,
                                 std::ptr::null_mut(), ptrs.as_mut_ptr(), std::ptr::null_mut())
        };
        if code == 0 { Ok(()) } else { Err(self.dev.err("cuLaunchKernel", code)) }
    }
}

/// Element types a [`DeviceBuffer`] may hold: plain 4-byte values with every
/// bit pattern valid, which is what makes reading device memory back safe.
pub trait Pod: Copy + Default + private::Sealed {}
impl Pod for f32 {}
impl Pod for u32 {}
mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
}

/// `len` elements on the device, freed on drop.
pub struct DeviceBuffer<T: Pod> {
    dev: &'static Device,
    ptr: DevPtr,
    len: usize,
    _t: PhantomData<T>,
}
// SAFETY: the buffer is an integer address on the device; every operation
// binds the context first, and CUDA's copies and launches are thread-safe.
unsafe impl<T: Pod> Send for DeviceBuffer<T> {}
unsafe impl<T: Pod> Sync for DeviceBuffer<T> {}

impl<T: Pod> DeviceBuffer<T> {
    /// Elements in the buffer.
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// `len` elements with undefined device contents. At least one is
    /// allocated, as CUDA refuses a zero-byte allocation.
    fn alloc(dev: &'static Device, len: usize) -> Result<Self, String> {
        dev.bind()?;
        let bytes = len.max(1).checked_mul(4).ok_or("buffer size overflows")?;
        let mut ptr: DevPtr = 0;
        // SAFETY: `ptr` is a valid out pointer.
        match unsafe { (dev.cu.alloc)(&mut ptr, bytes) } {
            0 => Ok(Self { dev, ptr, len, _t: PhantomData }),
            c => Err(dev.err("cuMemAlloc", c)),
        }
    }

    /// A buffer holding a copy of `data`.
    pub fn from_slice(dev: &'static Device, data: &[T]) -> Result<Self, String> {
        let b = Self::alloc(dev, data.len())?;
        b.copy_from(0, data)?;
        Ok(b)
    }

    /// `len` zeroed elements.
    pub fn zeroed(dev: &'static Device, len: usize) -> Result<Self, String> {
        let b = Self::alloc(dev, len)?;
        // SAFETY: the allocation holds `len.max(1)` 4-byte elements.
        match unsafe { (dev.cu.set32)(b.ptr, 0, len.max(1)) } {
            0 => Ok(b),
            c => Err(dev.err("cuMemsetD32", c)),
        }
    }

    /// Overwrite `data.len()` elements from element `at`, ordered after every
    /// kernel already launched.
    pub fn copy_from(&self, at: usize, data: &[T]) -> Result<(), String> {
        let end = at.checked_add(data.len()).ok_or("range overflows")?;
        if end > self.len { return Err(format!("write up to {end} in {} elements", self.len)); }
        if data.is_empty() { return Ok(()); }
        self.dev.bind()?;
        // SAFETY: the destination range was checked against the allocation;
        // the source is a live slice of exactly that many 4-byte elements.
        let code = unsafe { (self.dev.cu.h_to_d)(self.ptr + at as u64 * 4, data.as_ptr().cast(), data.len() * 4) };
        if code == 0 { Ok(()) } else { Err(self.dev.err("cuMemcpyHtoD", code)) }
    }

    /// Read the whole buffer. Waits for every kernel launched before it and
    /// reports a fault in any of them.
    pub fn to_vec(&self) -> Result<Vec<T>, String> {
        let mut out = vec![T::default(); self.len];
        if self.len == 0 { return Ok(out); }
        self.dev.bind()?;
        // SAFETY: `out` and the allocation both hold `len` 4-byte elements,
        // and `Pod` makes any device bit pattern a valid `T`.
        let code = unsafe { (self.dev.cu.d_to_h)(out.as_mut_ptr().cast(), self.ptr, self.len * 4) };
        if code == 0 { Ok(out) } else { Err(self.dev.err("cuMemcpyDtoH", code)) }
    }
}

impl<T: Pod> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        // A kernel still queued may be using this memory, so wait for it. An
        // error here has nowhere to go, and leaking is the safe outcome.
        if self.dev.sync().is_ok() {
            // SAFETY: `ptr` came from cuMemAlloc and is freed only here, and
            // nothing queued can reference it after the sync above.
            let _ = unsafe { (self.dev.cu.free)(self.ptr) };
        }
    }
}

fn open_device() -> Result<Device, String> {
    let lib = open_lib("libcuda.so.1").ok_or("libcuda.so.1 not found")?;
    let cu = Cu::load(lib)?;
    let mut d = Device { cu, ctx: std::ptr::null_mut(), name: String::new(), count: 0, memory: 0, driver: 0 };
    // SAFETY: every out pointer below is a valid local; lengths are stated.
    unsafe {
        let code = (d.cu.init)(0);
        if code != 0 { return Err(d.err("cuInit", code)); }
        let mut n: c_int = 0;
        let code = (d.cu.device_count)(&mut n);
        if code != 0 { return Err(d.err("cuDeviceGetCount", code)); }
        if n < 1 { return Err("no CUDA device".into()); }
        d.count = n as usize;
        let mut dev: c_int = 0;
        let code = (d.cu.device_get)(&mut dev, 0);
        if code != 0 { return Err(d.err("cuDeviceGet", code)); }
        let mut buf = [0 as c_char; 256];
        let code = (d.cu.device_name)(buf.as_mut_ptr(), buf.len() as c_int, dev);
        if code != 0 { return Err(d.err("cuDeviceGetName", code)); }
        d.name = CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned();
        let code = (d.cu.device_mem)(&mut d.memory, dev);
        if code != 0 { return Err(d.err("cuDeviceTotalMem", code)); }
        let code = (d.cu.driver_version)(&mut d.driver);
        if code != 0 { return Err(d.err("cuDriverGetVersion", code)); }
        let mut ctx: Handle = std::ptr::null_mut();
        let code = (d.cu.ctx_create)(&mut ctx, 0, dev);
        if code != 0 { return Err(d.err("cuCtxCreate", code)); }
        d.ctx = ctx;
    }
    Ok(d)
}

static DEVICE: OnceLock<Result<Device, String>> = OnceLock::new();

/// The process-wide device, opened on first use. `Err` says why there is
/// none, which is what a machine without CUDA gets.
pub fn device() -> Result<&'static Device, &'static str> {
    DEVICE.get_or_init(open_device).as_ref().map_err(|e| e.as_str())
}

/// cuBLAS, found at run time. It is optional: the tiled GEMM is always there.
pub struct Blas { dev: &'static Device, cb: Cb, handle: Handle, lock: Mutex<()> }
// SAFETY: an opaque handle; calls hold `lock`, as a cuBLAS handle is not
// promised to be usable from two threads at once.
unsafe impl Send for Blas {}
unsafe impl Sync for Blas {}

impl Blas {
    /// `C = op(A) op(B) + beta C` in cuBLAS's column-major terms, `alpha` = 1,
    /// with `ta` / `tb` true for a transposed operand. Extents are checked
    /// against the buffers, as cuBLAS would otherwise fault the context.
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm(&self, ta: bool, tb: bool, m: usize, n: usize, k: usize,
                 a: &DeviceBuffer<f32>, lda: usize, b: &DeviceBuffer<f32>, ldb: usize,
                 beta: f32, c: &DeviceBuffer<f32>, ldc: usize) -> Result<(), String> {
        if m == 0 || n == 0 || k == 0 { return Err("empty cuBLAS operand".into()); }
        let (arows, acols) = if ta { (k, m) } else { (m, k) };
        let (brows, bcols) = if tb { (n, k) } else { (k, n) };
        let need = |cols: usize, ld: usize, rows: usize| (cols - 1).saturating_mul(ld).saturating_add(rows);
        if lda < arows || ldb < brows || ldc < m
            || a.len() < need(acols, lda, arows) || b.len() < need(bcols, ldb, brows) || c.len() < need(n, ldc, m) {
            return Err("cuBLAS operand too small".into());
        }
        let dim = |x: usize| c_int::try_from(x).map_err(|_| "cuBLAS dimension too large".to_string());
        let (mi, ni, ki, ldai, ldbi, ldci) = (dim(m)?, dim(n)?, dim(k)?, dim(lda)?, dim(ldb)?, dim(ldc)?);
        const NOTRANS: c_int = 0; // CUBLAS_OP_N
        const TRANS: c_int = 1;   // CUBLAS_OP_T
        let alpha = 1.0f32;
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        self.dev.bind()?;
        // SAFETY: the handle is live; the extents were checked above; alpha
        // and beta are read through host pointers (cuBLAS's default mode)
        // before this returns.
        let code = unsafe {
            (self.cb.sgemm)(self.handle, if ta { TRANS } else { NOTRANS }, if tb { TRANS } else { NOTRANS },
                            mi, ni, ki, &alpha, a.ptr, ldai, b.ptr, ldbi, &beta, c.ptr, ldci)
        };
        if code == 0 { Ok(()) } else { Err(format!("cublasSgemm_v2: status {code}")) }
    }

    /// `op(A_b) op(B_b) alpha -> C_b` for every `b` in `0..batch`, the matrices
    /// at `offset + b * stride` in cuBLAS's column-major terms. Each operand's
    /// extent is checked against its buffer, as in [`Blas::sgemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_strided(&self, m: usize, n: usize, k: usize, alpha: f32, a: Strided<'_>, b: Strided<'_>, c: Strided<'_>,
                         batch: usize) -> Result<(), String> {
        if m == 0 || n == 0 || k == 0 || batch == 0 { return Err("empty cuBLAS operand".into()); }
        if c.trans { return Err("cuBLAS cannot transpose its output".into()); }
        let (ab, bb, cb) = (a.extent(if a.trans { (k, m) } else { (m, k) }, batch), b.extent(if b.trans { (n, k) } else { (k, n) }, batch), c.extent((m, n), batch));
        if !(ab && bb && cb) { return Err("cuBLAS operand too small".into()); }
        let dim = |x: usize| c_int::try_from(x).map_err(|_| "cuBLAS dimension too large".to_string());
        let stride = |x: usize| i64::try_from(x).map_err(|_| "cuBLAS stride too large".to_string());
        let op = |t: bool| if t { 1 } else { 0 }; // CUBLAS_OP_T, CUBLAS_OP_N
        let beta = 0.0f32;
        let (ma, na, ka, la, lb, lc, ba) = (dim(m)?, dim(n)?, dim(k)?, dim(a.ld)?, dim(b.ld)?, dim(c.ld)?, dim(batch)?);
        let (sa, sb, sc) = (stride(a.stride)?, stride(b.stride)?, stride(c.stride)?);
        let at = |s: &Strided<'_>| s.buf.ptr + (s.offset * std::mem::size_of::<f32>()) as DevPtr;
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        self.dev.bind()?;
        // SAFETY: the handle is live; every matrix of the batch was checked to
        // lie inside its buffer; alpha and beta are read through host pointers
        // before this returns.
        let code = unsafe {
            (self.cb.sgemm_strided)(self.handle, op(a.trans), op(b.trans), ma, na, ka, &alpha,
                                    at(&a), la, sa, at(&b), lb, sb, &beta, at(&c), lc, sc, ba)
        };
        if code == 0 { Ok(()) } else { Err(format!("cublasSgemmStridedBatched: status {code}")) }
    }
}

/// One operand of a strided batch: matrix `b` starts `offset + b * stride`
/// elements into `buf`, its columns `ld` apart. `trans` asks for `op` = transpose.
#[derive(Clone, Copy)]
pub struct Strided<'a> { pub buf: &'a DeviceBuffer<f32>, pub offset: usize, pub trans: bool, pub ld: usize, pub stride: usize }

impl Strided<'_> {
    /// Whether all `batch` matrices of `(rows, cols)` lie inside the buffer.
    fn extent(&self, (rows, cols): (usize, usize), batch: usize) -> bool {
        let end = (batch - 1).checked_mul(self.stride)
            .and_then(|e| e.checked_add((cols - 1).checked_mul(self.ld)?))
            .and_then(|e| e.checked_add(rows)).and_then(|e| e.checked_add(self.offset));
        self.ld >= rows && end.is_some_and(|e| e <= self.buf.len())
    }
}

static BLAS: OnceLock<Result<Blas, String>> = OnceLock::new();

fn open_blas(lib: &str) -> Result<Blas, String> {
    let dev = device().map_err(str::to_string)?;
    let h = open_lib(lib).ok_or_else(|| format!("{lib} not found"))?;
    let cb = Cb::load(h)?;
    dev.bind()?;
    let mut handle: Handle = std::ptr::null_mut();
    // SAFETY: `handle` is a valid out pointer; the context is current.
    let code = unsafe { (cb.create)(&mut handle) };
    if code != 0 { return Err(format!("cublasCreate_v2: status {code}")); }
    Ok(Blas { dev, cb, handle, lock: Mutex::new(()) })
}

/// cuBLAS, opened on first use. `Err` says why it is not available.
pub fn blas() -> Result<&'static Blas, &'static str> {
    BLAS.get_or_init(|| open_blas("libcublas.so.13")).as_ref().map_err(|e| e.as_str())
}
