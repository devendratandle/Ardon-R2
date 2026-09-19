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
        match self.arity {
            0 => self.call0(),
            1 => self.call1(args[0]),
            2 => self.call2(args[0], args[1]),
            _ => None,
        }
    }

    fn try_call_vec1(&self, x: &[f64]) -> Option<f64> {
        if self.kind != r2_types::JitKind::Vector1ToScalar { return None; }
        // SAFETY: `x` is a live slice, so its pointer is valid for
        // `x.len()` reads; the code at `self.ptr` was compiled from IR of
        // this kind, whose ABI is (ptr, len) -> f64.
        let f: extern "C" fn(*const f64, i64) -> f64 = unsafe { std::mem::transmute(self.ptr) };
        Some(f(x.as_ptr(), x.len() as i64))
    }

    fn try_call_vec2(&self, a: &[f64], b: &[f64]) -> Option<f64> {
        if self.kind != r2_types::JitKind::Vector2ToScalar || a.len() != b.len() { return None; }
        // SAFETY: as try_call_vec1; both slices checked equal in length.
        let f: extern "C" fn(*const f64, *const f64, i64) -> f64 = unsafe { std::mem::transmute(self.ptr) };
        Some(f(a.as_ptr(), b.as_ptr(), a.len() as i64))
    }

    fn try_call_vec_map(&self, x: &[f64], out: &mut [f64]) -> bool {
        if self.kind != r2_types::JitKind::VectorMap || x.len() != out.len() { return false; }
        // SAFETY: as try_call_vec1; `out` is a live mutable slice of the
        // same length, so every write the code makes is in bounds.
        let f: extern "C" fn(*const f64, *mut f64, i64) = unsafe { std::mem::transmute(self.ptr) };
        f(x.as_ptr(), out.as_mut_ptr(), x.len() as i64);
        true
    }

    fn try_call_vec_binary(&self, a: &[f64], b: &[f64], out: &mut [f64]) -> bool {
        if self.kind != r2_types::JitKind::VectorBinaryMap || a.len() != b.len() || a.len() != out.len() { return false; }
        // SAFETY: as try_call_vec_map.
        let f: extern "C" fn(*const f64, *const f64, *mut f64, i64) = unsafe { std::mem::transmute(self.ptr) };
        f(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len() as i64);
        true
    }

    fn try_call_vec_ternary(&self, a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) -> bool {
        if self.kind != r2_types::JitKind::VectorTernaryMap
            || a.len() != b.len() || a.len() != c.len() || a.len() != out.len() { return false; }
        // SAFETY: as try_call_vec_map.
        let f: extern "C" fn(*const f64, *const f64, *const f64, *mut f64, i64) =
            unsafe { std::mem::transmute(self.ptr) };
        f(a.as_ptr(), b.as_ptr(), c.as_ptr(), out.as_mut_ptr(), a.len() as i64);
        true
    }

    // Phase J.3 indexed-store maps. The native code returns a dummy f64 (the
    // loop's NULL value) declared in the fn type and ignored; the result is
    // written through `out`.
    fn try_call_ixstore1(&self, x: &[f64], out: &mut [f64]) -> bool {
        if self.kind != r2_types::JitKind::IndexedStoreMap1 || x.len() != out.len() { return false; }
        // SAFETY: as try_call_vec_map.
        let f: extern "C" fn(*const f64, *mut f64, i64) -> f64 = unsafe { std::mem::transmute(self.ptr) };
        f(x.as_ptr(), out.as_mut_ptr(), x.len() as i64);
        true
    }
    fn try_call_ixstore2(&self, a: &[f64], b: &[f64], out: &mut [f64]) -> bool {
        if self.kind != r2_types::JitKind::IndexedStoreMap2 || a.len() != b.len() || a.len() != out.len() { return false; }
        // SAFETY: as try_call_vec_map.
        let f: extern "C" fn(*const f64, *const f64, *mut f64, i64) -> f64 = unsafe { std::mem::transmute(self.ptr) };
        f(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len() as i64);
        true
    }

    // Phase J.4 matrix state — (matrix, vector) → p-vector out.
    fn try_call_matvec(&self, m: &[f64], nrow: usize, ncol: usize, v: &[f64], out: &mut [f64]) -> bool {
        if self.kind != r2_types::JitKind::MatVecIterOut
            || m.len() != nrow * ncol || v.len() != nrow || out.len() != ncol { return false; }
        // SAFETY: as try_call_vec_map; the three extents are checked
        // against the dimensions the code will index with.
        let f: extern "C" fn(*const f64, i64, i64, *const f64, *mut f64) = unsafe { std::mem::transmute(self.ptr) };
        f(m.as_ptr(), nrow as i64, ncol as i64, v.as_ptr(), out.as_mut_ptr());
        true
    }
}

impl CompiledFn {
    /// Scalar entry points. Each checks the arity it is about to assume,
    /// so a caller cannot pass two arguments to code compiled for three;
    /// with that check in place the `transmute` is the only unverifiable
    /// step and it lives here, once per ABI shape.
    pub fn call0(&self) -> Option<f64> {
        if self.kind != r2_types::JitKind::Scalar || self.arity != 0 { return None; }
        // SAFETY: `ptr` is finalized native code compiled from IR with
        // zero f64 parameters and an f64 result (checked just above).
        let f: extern "C" fn() -> f64 = unsafe { std::mem::transmute(self.ptr) };
        Some(f())
    }
    pub fn call1(&self, a: f64) -> Option<f64> {
        if self.kind != r2_types::JitKind::Scalar || self.arity != 1 { return None; }
        // SAFETY: as call0, for one f64 parameter.
        let f: extern "C" fn(f64) -> f64 = unsafe { std::mem::transmute(self.ptr) };
        Some(f(a))
    }
    pub fn call2(&self, a: f64, b: f64) -> Option<f64> {
        if self.kind != r2_types::JitKind::Scalar || self.arity != 2 { return None; }
        // SAFETY: as call0, for two f64 parameters.
        let f: extern "C" fn(f64, f64) -> f64 = unsafe { std::mem::transmute(self.ptr) };
        Some(f(a, b))
    }
}
