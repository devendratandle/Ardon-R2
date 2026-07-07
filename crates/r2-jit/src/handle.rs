//! Compiled-function handle: the native code pointer + ABI shims.

use cranelift_jit::JITModule;

// ── Compiled function handle ─────────────────────────────────────────

pub struct CompiledFn {
    pub ptr: *const u8,
    pub arity: usize,
    /// Kind of specialization — Scalar (Phase C.2) or Vector1ToScalar (Phase C.3).
    pub kind: r2_types::JitKind,
    pub(crate) _module: JITModule,
}

// SAFETY (Phase P): `ptr` points at finalized, immutable, reentrant native
// code; `_module` owns that executable memory read-only after
// `finalize_definitions` and is never mutated afterwards. Calling the compiled
// function from multiple threads is data-race-free (pure arithmetic, no shared
// mutable state). We only ever *call* through a shared `&self`, never mutate.
unsafe impl Send for CompiledFn {}
unsafe impl Sync for CompiledFn {}

impl std::fmt::Debug for CompiledFn {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "CompiledFn {{ kind: {:?}, arity: {}, ptr: {:p} }}", self.kind, self.arity, self.ptr)
    }
}

impl r2_types::JitHandle for CompiledFn {
    fn kind(&self) -> r2_types::JitKind { self.kind }
    fn arity(&self) -> usize { self.arity }

    fn try_call_real(&self, args: &[f64]) -> Option<f64> {
        if self.kind != r2_types::JitKind::Scalar { return None; }
        if args.len() != self.arity { return None; }
        unsafe {
            Some(match self.arity {
                0 => self.call0(),
                1 => self.call1(args[0]),
                2 => self.call2(args[0], args[1]),
                _ => return None,
            })
        }
    }

    unsafe fn try_call_vec1(&self, ptr: *const f64, len: i64) -> Option<f64> {
        if self.kind != r2_types::JitKind::Vector1ToScalar { return None; }
        let f: extern "C" fn(*const f64, i64) -> f64 = std::mem::transmute(self.ptr);
        Some(f(ptr, len))
    }

    unsafe fn try_call_vec2(&self, a_ptr: *const f64, b_ptr: *const f64, len: i64) -> Option<f64> {
        if self.kind != r2_types::JitKind::Vector2ToScalar { return None; }
        let f: extern "C" fn(*const f64, *const f64, i64) -> f64 = std::mem::transmute(self.ptr);
        Some(f(a_ptr, b_ptr, len))
    }

    unsafe fn try_call_vec_map(&self, in_ptr: *const f64, out_ptr: *mut f64, len: i64) -> bool {
        if self.kind != r2_types::JitKind::VectorMap { return false; }
        let f: extern "C" fn(*const f64, *mut f64, i64) = std::mem::transmute(self.ptr);
        f(in_ptr, out_ptr, len);
        true
    }

    unsafe fn try_call_vec_binary(&self, a_ptr: *const f64, b_ptr: *const f64, out_ptr: *mut f64, len: i64) -> bool {
        if self.kind != r2_types::JitKind::VectorBinaryMap { return false; }
        let f: extern "C" fn(*const f64, *const f64, *mut f64, i64) = std::mem::transmute(self.ptr);
        f(a_ptr, b_ptr, out_ptr, len);
        true
    }

    unsafe fn try_call_vec_ternary(
        &self,
        a_ptr: *const f64,
        b_ptr: *const f64,
        c_ptr: *const f64,
        out_ptr: *mut f64,
        len: i64,
    ) -> bool {
        if self.kind != r2_types::JitKind::VectorTernaryMap { return false; }
        let f: extern "C" fn(*const f64, *const f64, *const f64, *mut f64, i64) =
            std::mem::transmute(self.ptr);
        f(a_ptr, b_ptr, c_ptr, out_ptr, len);
        true
    }

    // Phase J.3 indexed-store maps. The native code returns a dummy f64 (the
    // loop's NULL value) declared in the fn type and ignored; the result is
    // written through `out_ptr`.
    unsafe fn try_call_ixstore1(&self, in_ptr: *const f64, out_ptr: *mut f64, len: i64) -> bool {
        if self.kind != r2_types::JitKind::IndexedStoreMap1 { return false; }
        let f: extern "C" fn(*const f64, *mut f64, i64) -> f64 = std::mem::transmute(self.ptr);
        f(in_ptr, out_ptr, len);
        true
    }
    unsafe fn try_call_ixstore2(&self, a_ptr: *const f64, b_ptr: *const f64, out_ptr: *mut f64, len: i64) -> bool {
        if self.kind != r2_types::JitKind::IndexedStoreMap2 { return false; }
        let f: extern "C" fn(*const f64, *const f64, *mut f64, i64) -> f64 = std::mem::transmute(self.ptr);
        f(a_ptr, b_ptr, out_ptr, len);
        true
    }
}

impl CompiledFn {
    /// SAFETY: only call when arity == 0 and the function returns f64.
    pub unsafe fn call0(&self) -> f64 {
        debug_assert_eq!(self.arity, 0);
        let f: extern "C" fn() -> f64 = std::mem::transmute(self.ptr);
        f()
    }
    pub unsafe fn call1(&self, a: f64) -> f64 {
        debug_assert_eq!(self.arity, 1);
        let f: extern "C" fn(f64) -> f64 = std::mem::transmute(self.ptr);
        f(a)
    }
    pub unsafe fn call2(&self, a: f64, b: f64) -> f64 {
        debug_assert_eq!(self.arity, 2);
        let f: extern "C" fn(f64, f64) -> f64 = std::mem::transmute(self.ptr);
        f(a, b)
    }
}
