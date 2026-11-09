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
//! ordered with the blocking copies; only [`CopyQueue`] has streams, so a copy
//! each way can overlap, and it holds what it copies until they are done. Targets
//! other than Unix have no loader here and report "no device".

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::marker::PhantomData;
use std::sync::{Mutex, OnceLock};

type CuResult = c_int;
type DevPtr = u64;
/// `CUipcMemHandle`: 64 opaque bytes, passed by value.
#[repr(C)]
#[derive(Clone, Copy)]
struct IpcHandle([u8; 64]);
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
    alloc_host = "cuMemAllocHost_v2": fn(*mut *mut c_void, usize) -> CuResult;
    free_host = "cuMemFreeHost": fn(*mut c_void) -> CuResult;
    h_to_d = "cuMemcpyHtoD_v2": fn(DevPtr, *const c_void, usize) -> CuResult;
    d_to_d = "cuMemcpyDtoD_v2": fn(DevPtr, DevPtr, usize) -> CuResult;
    host_register = "cuMemHostRegister_v2": fn(*mut c_void, usize, c_uint) -> CuResult;
    host_unregister = "cuMemHostUnregister": fn(*mut c_void) -> CuResult;
    can_access = "cuDeviceCanAccessPeer": fn(*mut c_int, c_int, c_int) -> CuResult;
    ipc_get = "cuIpcGetMemHandle": fn(*mut IpcHandle, DevPtr) -> CuResult;
    ipc_open = "cuIpcOpenMemHandle_v2": fn(*mut DevPtr, IpcHandle, c_uint) -> CuResult;
    ipc_close = "cuIpcCloseMemHandle": fn(DevPtr) -> CuResult;
    addr_range = "cuMemGetAddressRange_v2": fn(*mut DevPtr, *mut usize, DevPtr) -> CuResult;
    d_to_h = "cuMemcpyDtoH_v2": fn(*mut c_void, DevPtr, usize) -> CuResult;
    stream_create = "cuStreamCreate": fn(*mut Handle, c_uint) -> CuResult;
    stream_sync = "cuStreamSynchronize": fn(Handle) -> CuResult;
    stream_destroy = "cuStreamDestroy_v2": fn(Handle) -> CuResult;
    d_to_h_async = "cuMemcpyDtoHAsync_v2": fn(*mut c_void, DevPtr, usize, Handle) -> CuResult;
    h_to_d_async = "cuMemcpyHtoDAsync_v2": fn(DevPtr, *const c_void, usize, Handle) -> CuResult;
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
    gemm_ex = "cublasGemmEx": fn(Handle, c_int, c_int, c_int, c_int, c_int, *const c_void, DevPtr, c_int, c_int, DevPtr, c_int, c_int, *const c_void, DevPtr, c_int, c_int, c_int, c_int) -> c_int;
});

/// One GPU with its context: the process-wide CUDA device.
pub struct Device {
    pub ordinal: usize,
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

    /// What another process passes to [`PeerBuffer::open`] to reach this buffer.
    pub fn ipc_handle(&self) -> Result<[u8; 64], String> {
        self.dev.bind()?;
        let mut h = IpcHandle([0; 64]);
        // SAFETY: `h` is a valid out pointer; `ptr` came from cuMemAlloc.
        let code = unsafe { (self.dev.cu.ipc_get)(&mut h, self.ptr) };
        if code == 0 { Ok(h.0) } else { Err(self.dev.err("cuIpcGetMemHandle", code)) }
    }

    /// Read the first `out.len()` elements into `out`, waiting as [`Self::to_vec`]
    /// does. Pinned memory, a [`HostBuffer`], makes this a direct transfer.
    pub fn copy_to(&self, out: &mut [T]) -> Result<(), String> {
        if out.len() > self.len { return Err(format!("read of {} in {} elements", out.len(), self.len)); }
        if out.is_empty() { return Ok(()); }
        self.dev.bind()?;
        // SAFETY: `out` is a live slice, no longer than the allocation, of
        // 4-byte elements, and `Pod` makes any device bit pattern a valid `T`.
        let code = unsafe { (self.dev.cu.d_to_h)(out.as_mut_ptr().cast(), self.ptr, out.len() * 4) };
        if code == 0 { Ok(()) } else { Err(self.dev.err("cuMemcpyDtoH", code)) }
    }
}

/// One direction of copies that overlap the other's: a stream of its own, not
/// ordered with the default stream, so the caller drains that first. Everything
/// handed to it stays borrowed for `'a`, until [`copy_both_ways`] has waited.
pub struct CopyQueue<'a> { dev: &'static Device, stream: Handle, _held: PhantomData<&'a mut [u8]> }

impl CopyQueue<'_> {
    fn new(dev: &'static Device) -> Result<Self, String> {
        dev.bind()?;
        let mut stream: Handle = std::ptr::null_mut();
        // SAFETY: the context is current and `stream` is a valid out pointer.
        // Flag 1 is CU_STREAM_NON_BLOCKING.
        match unsafe { (dev.cu.stream_create)(&mut stream, 1) } {
            0 => Ok(CopyQueue { dev, stream, _held: PhantomData }),
            c => Err(dev.err("cuStreamCreate", c)),
        }
    }

    /// Wait for what this queue was given.
    pub fn finish(&self) -> Result<(), String> {
        self.dev.bind()?;
        // SAFETY: `stream` is live until drop.
        match unsafe { (self.dev.cu.stream_sync)(self.stream) } { 0 => Ok(()), c => Err(self.dev.err("cuStreamSynchronize", c)) }
    }
}

impl<'a> CopyQueue<'a> {
    /// Queues `src[at..at + out.len()]` into `out`.
    pub fn download<T: Pod>(&self, src: &'a DeviceBuffer<T>, at: usize, out: &'a mut [T]) -> Result<(), String> {
        if at.checked_add(out.len()).is_none_or(|e| e > src.len) { return Err("download past the buffer".into()); }
        if out.is_empty() { return Ok(()); }
        self.dev.bind()?;
        // SAFETY: bounds checked above; `out` and `src` are borrowed for 'a, which
        // outlives the wait in `copy_both_ways`; `Pod` makes any bit pattern valid.
        let code = unsafe { (self.dev.cu.d_to_h_async)(out.as_mut_ptr().cast(), src.ptr + (at * 4) as DevPtr, out.len() * 4, self.stream) };
        if code == 0 { Ok(()) } else { Err(self.dev.err("cuMemcpyDtoHAsync", code)) }
    }

    /// Queues `data` into `dst[at..]`.
    pub fn upload<T: Pod>(&self, dst: &'a DeviceBuffer<T>, at: usize, data: &'a [T]) -> Result<(), String> {
        if at.checked_add(data.len()).is_none_or(|e| e > dst.len) { return Err("upload past the buffer".into()); }
        if data.is_empty() { return Ok(()); }
        self.dev.bind()?;
        // SAFETY: bounds checked above; both are borrowed for 'a, as in `download`.
        let code = unsafe { (self.dev.cu.h_to_d_async)(dst.ptr + (at * 4) as DevPtr, data.as_ptr().cast(), data.len() * 4, self.stream) };
        if code == 0 { Ok(()) } else { Err(self.dev.err("cuMemcpyHtoDAsync", code)) }
    }
}

impl Drop for CopyQueue<'_> {
    fn drop(&mut self) {
        // SAFETY: the stream came from cuStreamCreate and is destroyed only here;
        // CUDA finishes what it holds first. An error here has nowhere to go.
        let _ = self.dev.bind().map(|_| unsafe { (self.dev.cu.stream_destroy)(self.stream) });
    }
}

/// Runs `f` with a queue for each direction, and returns once both have finished,
/// even when `f` fails, so nothing they were given is released under a copy.
pub fn copy_both_ways<'a, R>(f: impl FnOnce(&CopyQueue<'a>, &CopyQueue<'a>) -> Result<R, String>) -> Result<R, String> {
    let dev = device()?;
    let (down, up) = (CopyQueue::new(dev)?, CopyQueue::new(dev)?);
    let result = f(&down, &up);
    let waited = down.finish().and(up.finish());
    let r = result?;
    waited?;
    Ok(r)
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

/// Another process's device buffer mapped into this device's address space, for
/// direct copies between the two cards. Unmapped on drop.
pub struct PeerBuffer { dev: &'static Device, ptr: DevPtr, len: usize }
// SAFETY: an integer address on the device, as `DeviceBuffer`.
unsafe impl Send for PeerBuffer {}

impl PeerBuffer {
    /// Maps what [`DeviceBuffer::ipc_handle`] gave in the other process, which must
    /// keep its buffer alive for as long as this lives. Its size is the driver's.
    pub fn open(handle: [u8; 64]) -> Result<PeerBuffer, String> {
        let dev = device().map_err(str::to_string)?;
        dev.bind()?;
        const PEER_MAPPING: c_uint = 1; // CUDA_IPC_HANDLE_PEER_MAPPING
        let mut ptr: DevPtr = 0;
        // SAFETY: `ptr` is a valid out pointer; the driver validates the handle.
        let code = unsafe { (dev.cu.ipc_open)(&mut ptr, IpcHandle(handle), PEER_MAPPING) };
        if code != 0 { return Err(dev.err("cuIpcOpenMemHandle", code)); }
        let (mut base, mut bytes): (DevPtr, usize) = (0, 0);
        // SAFETY: valid out pointers, and `ptr` was just mapped.
        let code = unsafe { (dev.cu.addr_range)(&mut base, &mut bytes, ptr) };
        if code != 0 { return Err(dev.err("cuMemGetAddressRange", code)); }
        Ok(PeerBuffer { dev, ptr, len: bytes.saturating_sub((ptr - base) as usize) / 4 })
    }
    /// f32 elements the mapping holds.
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    fn checked(&self, at: usize, n: usize, mine: &DeviceBuffer<f32>, mine_at: usize) -> Result<(), String> {
        
        let fits = |at: usize, len: usize| at.checked_add(n).is_some_and(|e| e <= len);
        if fits(at, self.len) && fits(mine_at, mine.len) { Ok(()) } else { Err("peer copy out of range".into()) }
    }
    /// `n` elements from `src` at `src_at` to here at `at`. Returns when done.
    pub fn write(&self, at: usize, src: &DeviceBuffer<f32>, src_at: usize, n: usize) -> Result<(), String> {
        self.checked(at, n, src, src_at)?;
        self.dev.bind()?;
        // SAFETY: both ranges were checked against their allocations.
        let code = unsafe { (self.dev.cu.d_to_d)(self.ptr + at as u64 * 4, src.ptr + src_at as u64 * 4, n * 4) };
        if code == 0 { Ok(()) } else { Err(self.dev.err("cuMemcpyDtoD", code)) }
    }
    /// `n` elements from here at `at` to `dst` at `dst_at`. Returns when done.
    pub fn read(&self, at: usize, dst: &DeviceBuffer<f32>, dst_at: usize, n: usize) -> Result<(), String> {
        self.checked(at, n, dst, dst_at)?;
        self.dev.bind()?;
        // SAFETY: both ranges were checked against their allocations.
        let code = unsafe { (self.dev.cu.d_to_d)(dst.ptr + dst_at as u64 * 4, self.ptr + at as u64 * 4, n * 4) };
        if code == 0 { Ok(()) } else { Err(self.dev.err("cuMemcpyDtoD", code)) }
    }
}

impl Drop for PeerBuffer {
    fn drop(&mut self) {
        if self.dev.sync().is_ok() {
            // SAFETY: mapped by cuIpcOpenMemHandle, nothing queued can use it now.
            let _ = unsafe { (self.dev.cu.ipc_close)(self.ptr) };
        }
    }
}

/// Whether this device can reach device `other` directly, and this device's ordinal.
pub fn peer_access(other: usize) -> Result<(bool, usize), String> {
    let dev = device().map_err(str::to_string)?;
    let (mut a, mut b, mut can): (c_int, c_int, c_int) = (0, 0, 0);
    // SAFETY: valid out pointers; the ordinals are checked by the driver.
    unsafe {
        let code = (dev.cu.device_get)(&mut a, dev.ordinal as c_int);
        if code != 0 { return Err(dev.err("cuDeviceGet", code)); }
        let code = (dev.cu.device_get)(&mut b, other as c_int);
        if code != 0 { return Err(dev.err("cuDeviceGet", code)); }
        let code = (dev.cu.can_access)(&mut can, a, b);
        if code != 0 { return Err(dev.err("cuDeviceCanAccessPeer", code)); }
    }
    Ok((can != 0, dev.ordinal))
}

table!(Libc {
    mmap = "mmap": fn(*mut c_void, usize, c_int, c_int, c_int, i64) -> *mut c_void;
    munmap = "munmap": fn(*mut c_void, usize) -> c_int;
});

/// `len` f32 in a file under `/dev/shm`, mapped into this process and pinned for
/// the device, so two processes on one machine can hand each other data with no
/// copy between them. The ranks take turns by a message on their control link; a
/// rank must not read a region while the other may be writing it.
pub struct SharedMem { dev: &'static Device, libc: Libc, ptr: *mut f32, len: usize }
// SAFETY: this owns its mapping, and `&mut` access goes through `&mut self`.
unsafe impl Send for SharedMem {}

impl SharedMem {
    #[cfg(not(unix))]
    pub fn map(path: &str, _len: usize, _create: bool) -> Result<SharedMem, String> {
        Err(format!("{path}: shared memory needs unix"))
    }

    /// Maps `path`, creating it with this length when `create`. Readable and
    /// writable only by this user.
    #[cfg(unix)]
    pub fn map(path: &str, len: usize, create: bool) -> Result<SharedMem, String> {
        let dev = device().map_err(str::to_string)?;
        let lib = open_lib("libc.so.6").ok_or("libc.so.6 not found")?;
        let libc = Libc::load(lib)?;
        let bytes = len.checked_mul(4).filter(|&b| b > 0).ok_or("shared memory size")?;
        let file = {
            use std::os::unix::fs::OpenOptionsExt;
            let mut o = std::fs::OpenOptions::new();
            o.read(true).write(true).mode(0o600);
            if create { o.create(true).truncate(true); }
            o.open(path).map_err(|e| format!("{path}: {e}"))?
        };
        if create { file.set_len(bytes as u64).map_err(|e| format!("{path}: {e}"))?; }
        if file.metadata().map_err(|e| e.to_string())?.len() != bytes as u64 { return Err(format!("{path}: wrong size")); }
        let fd = std::os::fd::AsRawFd::as_raw_fd(&file);
        const PROT_READ_WRITE: c_int = 3;
        const MAP_SHARED: c_int = 1;
        // SAFETY: a fresh mapping at an address of the kernel's choosing, of a file
        // just checked to hold `bytes`; it outlives the descriptor by POSIX rule.
        let p = unsafe { (libc.mmap)(std::ptr::null_mut(), bytes, PROT_READ_WRITE, MAP_SHARED, fd, 0) };
        if p.is_null() || p as isize == -1 { return Err(format!("{path}: mmap failed")); }
        dev.bind()?;
        // SAFETY: `p` is page-aligned and `bytes` long, and stays mapped until drop,
        // which unregisters it first.
        let code = unsafe { (dev.cu.host_register)(p, bytes, 0) };
        if code != 0 {
            // SAFETY: mapped just above, and nothing else refers to it.
            unsafe { (libc.munmap)(p, bytes); }
            return Err(dev.err("cuMemHostRegister", code));
        }
        Ok(SharedMem { dev, libc, ptr: p.cast(), len })
    }
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `len` f32 mapped for this value's life; any bits are a valid f32,
        // and the turn-taking in the type's contract keeps the peer from writing now.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: as `as_slice`, and `&mut self` makes it exclusive in this process.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for SharedMem {
    fn drop(&mut self) {
        // SAFETY: registered and mapped by `map`; the copies using it are synchronous.
        unsafe {
            let _ = (self.dev.cu.host_unregister)(self.ptr.cast());
            (self.libc.munmap)(self.ptr.cast(), self.len * 4);
        }
    }
}

/// `len` f32 of page-locked host memory, zeroed, freed on drop: what the device
/// copies to and from directly, and what a gradient exchange sends.
pub struct HostBuffer { dev: &'static Device, ptr: *mut f32, len: usize }
// SAFETY: this owns its allocation, and `&mut` access goes through `&mut self`.
unsafe impl Send for HostBuffer {}

impl HostBuffer {
    pub fn new(len: usize) -> Result<HostBuffer, String> {
        let dev = device().map_err(str::to_string)?;
        let bytes = len.checked_mul(4).filter(|&b| b > 0).ok_or("host buffer size")?;
        dev.bind()?;
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: `p` is a valid out pointer; the context is current.
        match unsafe { (dev.cu.alloc_host)(&mut p, bytes) } {
            0 => {
                // SAFETY: the allocation is `bytes` long and exclusively ours.
                unsafe { std::ptr::write_bytes(p.cast::<u8>(), 0, bytes) };
                Ok(HostBuffer { dev, ptr: p.cast(), len })
            }
            c => Err(dev.err("cuMemAllocHost", c)),
        }
    }
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `len` initialized f32 owned by self, for the borrow's life.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: as `as_slice`, and `&mut self` makes the borrow exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    /// The same memory as bytes, for a socket. Any byte pattern is a valid f32.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: as `as_slice`, over the same `len * 4` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.cast(), self.len * 4) }
    }
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        // SAFETY: as `as_mut_slice`; any bytes written are a valid f32.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast(), self.len * 4) }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from cuMemAllocHost; the copies using it are
        // synchronous, so nothing is still reading it.
        let _ = unsafe { (self.dev.cu.free_host)(self.ptr.cast()) };
    }
}

fn open_device() -> Result<Device, String> {
    let lib = open_lib("libcuda.so.1").ok_or("libcuda.so.1 not found")?;
    let cu = Cu::load(lib)?;
    let ordinal = std::env::var("R2_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0usize);
    let mut d = Device { cu, ordinal, ctx: std::ptr::null_mut(), name: String::new(), count: 0, memory: 0, driver: 0 };
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
        let code = (d.cu.device_get)(&mut dev, ordinal as c_int);
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

    /// `C = A B`, no transposes, in cuBLAS's column-major terms, through
    /// `cublasGemmEx`: storage and arithmetic types chosen per call. Only
    /// `examples/cuda_ceiling.rs` uses this, to measure what tensor cores give.
    pub fn gemm_ex(&self, form: GemmForm, m: usize, n: usize, k: usize, a: &DeviceBuffer<f32>, b: &DeviceBuffer<f32>,
                   c: &DeviceBuffer<f32>) -> Result<(), String> {
        const R_32F: c_int = 0;              // CUDA_R_32F
        const R_16F: c_int = 6;              // CUDA_R_16F
        const COMPUTE_32F: c_int = 0;        // CUBLAS_COMPUTE_32F
        const COMPUTE_16F: c_int = 64;       // CUBLAS_COMPUTE_16F
        const COMPUTE_32F_FAST_16F: c_int = 74;
        const GEMM_DEFAULT: c_int = -1;
        let (store, compute, algo) = match form {
            GemmForm::F32 => (R_32F, COMPUTE_32F, GEMM_DEFAULT),
            // the compute type alone asks for tensor cores; algorithm 160 was refused (run 33)
            GemmForm::F16 => (R_16F, COMPUTE_16F, GEMM_DEFAULT),
            GemmForm::F32Fast16 => (R_32F, COMPUTE_32F_FAST_16F, GEMM_DEFAULT),
        };
        // a buffer holds f32 slots, and two f16 fit in each
        let slots = |elems: usize| if store == R_16F { elems.div_ceil(2) } else { elems };
        if a.len() < slots(m * k) || b.len() < slots(k * n) || c.len() < slots(m * n) {
            return Err("cuBLAS operand too small".into());
        }
        let dim = |x: usize| c_int::try_from(x).map_err(|_| "cuBLAS dimension too large".to_string());
        let (mi, ni, ki) = (dim(m)?, dim(n)?, dim(k)?);
        // alpha and beta have the arithmetic's type: 16-bit one and zero, or f32
        let (a16, z16) = (0x3C00u16, 0u16);
        let (a32, z32) = (1.0f32, 0.0f32);
        let (alpha, beta): (*const c_void, *const c_void) = if compute == COMPUTE_16F {
            (&a16 as *const u16 as *const c_void, &z16 as *const u16 as *const c_void)
        } else {
            (&a32 as *const f32 as *const c_void, &z32 as *const f32 as *const c_void)
        };
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        self.dev.bind()?;
        // SAFETY: the handle is live; the extents were checked above in units of the
        // storage type; alpha and beta are read through host pointers before this returns.
        let code = unsafe {
            (self.cb.gemm_ex)(self.handle, 0, 0, mi, ni, ki, alpha, a.ptr, store, mi, b.ptr, store, ki,
                              beta, c.ptr, store, mi, compute, algo)
        };
        if code == 0 { Ok(()) } else { Err(format!("cublasGemmEx: status {code}")) }
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

/// What [`Blas::gemm_ex`] stores and computes in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GemmForm {
    /// f32 throughout, the control.
    F32,
    /// f16 storage and tensor-core arithmetic.
    F16,
    /// f32 storage, with cuBLAS free to do the arithmetic in f16.
    F32Fast16,
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
