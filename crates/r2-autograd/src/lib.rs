//! r2-autograd — reverse-mode automatic differentiation over the
//! r2-tensor op set (Layer 4 of the trillion-scale architecture;
//! docs/LLM_TRILLION_ARCHITECTURE.md).
//!
//! A `Tape` records the forward computation as a DAG of ops; `backward()`
//! walks it in reverse, accumulating gradients into every leaf marked
//! `requires_grad`. The op set is the transformer-critical one (matmul,
//! elementwise add/mul, SiLU, RMSNorm, softmax-cross-entropy, MSE).
//!
//! THE ACCURACY DISCIPLINE (mirrors the differential harness for stats,
//! and the GPU-vs-CPU contract): every backward is checked against
//! FINITE DIFFERENCES in the tests. Analytic gradient must match the
//! numeric gradient to a tight tolerance, or it doesn't ship. A wrong
//! gradient trains a wrong model silently — so gradients are gated, not
//! trusted.
//!
//! Shard-awareness (a grad on a `TensorParallel` tensor triggering the
//! right collective) is the integration point with r2-mesh — wired by
//! r2-train / Opus; the local tape here is the correctness reference.
//!
//! Files: this one holds the tape and its buffers; `forward.rs` the ops;
//! `backward.rs` their adjoints; `kernels.rs` the attention and dot-product
//! kernels; `tests.rs` the finite-difference gates.

mod forward;
mod backward;
mod kernels;
#[cfg(test)]
mod tests;

/// Index of a value node on the tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(pub usize);

/// One recorded operation. Holds the input `Var`s and any constants the
/// backward pass needs. Backward math lives in `Tape::backward`.
enum Op {
    Leaf,
    Add(Var, Var),
    Mul(Var, Var),
    MatMul { a: Var, b: Var, m: usize, k: usize, n: usize },
    /// Token embedding: row `tokens[i]` of a `vocab x d` table becomes row
    /// `i` of the output. A GATHER forward, a SCATTER-ADD backward — the
    /// same decomposition PyTorch uses (`aten::embedding` dispatches to
    /// `index_select`; its backward is `embedding_dense_backward`).
    ///
    /// Replaces `matmul(onehot, table, t, vocab, d)`, which computes the
    /// identical result by doing `vocab` times the necessary arithmetic and
    /// materialising a `t x vocab` buffer every forward pass. Measured at
    /// 2,048 tokens and dim 256, forward plus backward:
    ///
    /// ```text
    /// vocab      one-hot matmul     gather (PyTorch)
    ///   256           24,283 us              413 us
    /// 8,000          722,659 us            1,594 us
    /// 32,000       3,416,475 us            5,173 us
    /// ```
    ///
    /// A gather's cost is FLAT in the vocabulary, as it must be: it copies
    /// `t*d` elements whatever the table's height. The one-hot form's grows
    /// linearly, which was tolerable at vocab 256 and is not at 8,000.
    ///
    /// The tokens are stored so the backward can scatter into the rows the
    /// forward touched.
    Embed { table: Var, tokens: Vec<usize>, d: usize },
    Silu(Var),
    /// RMSNorm over the last dim `d`, with weight and eps. Rows = len/d.
    Rmsnorm { x: Var, w: Var, d: usize, eps: f32 },
    /// Transpose a rows×cols matrix → cols×rows.
    Transpose { x: Var, rows: usize, cols: usize },
    /// Rotary position embedding over a `rows × (n_heads*head_dim)`
    /// activation, row `r` rotated for position `r`. RoPE is an
    /// ORTHOGONAL transform, so its backward is simply the rotation by
    /// the negated angle — no Jacobian to store.
    /// `period` is the sequence length in a fused batch: B sequences of T
    /// tokens stacked as B·T rows rotate with position = row % T, so every
    /// sequence sees positions 0..T exactly as it would alone.
    Rope { x: Var, rows: usize, period: usize, n_heads: usize, head_dim: usize, base: f32 },
    /// Contiguous row range of a `rows × cols` matrix. Rows are contiguous
    /// in memory, so this is how one sequence is cut out of a fused batch
    /// for attention (which must never see across sequence boundaries).
    SliceRows { x: Var, cols: usize, start: usize, len: usize },
    /// Stack matrices with the same column count vertically — the inverse
    /// of SliceRows, reassembling per-sequence attention outputs into the
    /// fused activation.
    ConcatRows { xs: Vec<Var> },
    /// Take a contiguous column range from each row — how one attention
    /// head is separated out of a packed multi-head activation.
    SliceCols { x: Var, rows: usize, total: usize, start: usize, len: usize },
    /// Concatenate equal-width column blocks — how heads are packed back
    /// together before the output projection.
    ConcatCols { xs: Vec<Var>, rows: usize, each: usize },
    /// Scale attention scores and apply the causal mask in one step: a
    /// masked position contributes nothing and receives no gradient,
    /// which is what stops a token from learning to read its future.
    ScaleMaskCausal { x: Var, t: usize, scale: f32 },
    /// Softmax over each `d`-wide row (a differentiable op, distinct from
    /// the fused SoftmaxCE loss — this one is used INSIDE attention).
    SoftmaxRows { x: Var, d: usize },
    /// Grouped-query causal attention over a whole batch, FUSED.
    ///
    /// Replaces this, which is what `llm.rs` used to build per sequence
    /// per head — `nseq * n_heads` times, 128 at the shipping shape:
    ///
    /// ```text
    /// slice_rows x3 -> slice_cols x3 -> transpose -> matmul
    ///   -> scale_mask_causal -> softmax_rows -> matmul -> concat_cols
    /// ```
    ///
    /// That decomposition is correct and was ~10x slower than PyTorch's
    /// EXPLICIT form — not because of the arithmetic. Measured at batch 32
    /// x seq 64 (`--example lmo15_breakdown`), forward:
    ///
    /// ```text
    /// slice rows          11,132 us   47%
    /// softmax rows         4,982       21%
    /// transpose k          2,979       13%
    /// slice cols           2,558       11%
    /// scale & mask           972        4%
    /// QK matmul              385      1.6%   <- the arithmetic
    /// AV matmul + concat     662      2.8%   <- the arithmetic
    /// ```
    ///
    /// The two matmuls are 4.4% of the block. The rest is chopping the
    /// data into 128 fragments of 64x64 so those matmuls can run on it,
    /// and putting 1,156 nodes — 2,312 Vec allocations — on the tape to
    /// hold the pieces. This op is ONE node: no slices, no transposes, no
    /// materialised score matrix, and the inner loops walk contiguous
    /// memory.
    ///
    /// Layout matches the projections that feed it: `q` is
    /// `(nseq*seq) x (n_heads*head_dim)`, `k` and `v` are
    /// `(nseq*seq) x (n_kv*head_dim)`, row `s*seq + i` is position `i` of
    /// sequence `s`. Query head `qh` reads kv head `qh / (n_heads/n_kv)`.
    Attention { q: Var, k: Var, v: Var, nseq: usize, seq: usize,
                nh: usize, nkv: usize, hd: usize, scale: f32 },
    /// Σ of all elements → scalar.
    SumAll(Var),
    /// Mean squared error against a constant target (target has no grad).
    Mse { pred: Var, target: Vec<f32> },
    /// Softmax over `d`-wide rows then cross-entropy against per-row class
    /// indices. Scalar loss; the classic fused backward (softmax − onehot).
    ///
    /// `lse` is the per-row log-sum-exp, computed once in the forward and
    /// kept so the backward needs neither the probability matrix nor a
    /// second reduction pass over it. At vocab 8,000 and 2,048 rows that
    /// matrix is 16.4M floats — 64 MB — and the previous implementation
    /// built it TWICE per step: once in the forward, to read 2,048 values
    /// out of it, and again in the backward.
    SoftmaxCE { logits: Var, d: usize, targets: Vec<usize>, lse: Vec<f32> },
}

/// The autograd tape: values + grads + the op that produced each node.
pub struct Tape {
    vals: Vec<Vec<f32>>,
    grads: Vec<Vec<f32>>,
    ops: Vec<Op>,
    requires: Vec<bool>,
    /// Value buffers recycled from the previous step — see [`BufPool`].
    pool: BufPool,
    /// How many of this tape's gradient buffers came back from the pool
    /// holding last step's gradients (contents stale).
    recycled_grads: usize,
    /// Forward census (`R2_TAPE_STATS=1`): the instant of the last push.
    /// The forward is sequential, so the time between two pushes is the
    /// cost of the op pushed second — recorded in `push`, no timer in any
    /// builder.
    last_push: Option<std::time::Instant>,
    /// Per node: has a backward arm written this node's gradient yet?
    /// The FIRST writer assigns, every later one accumulates — so no
    /// gradient buffer is ever zeroed, recycled or not. See
    /// `backward_from`.
    gwritten: Vec<bool>,
    /// Has a backward pass already run on this tape?
    ///
    /// Gradient buffers are allocated zero (`push`), so on the FIRST
    /// backward there is nothing to clear. Only a second pass over the same
    /// tape needs the blanket reset, and this is what tells the two apart.
    differentiated: bool,
}

/// Element count below which an op stays SERIAL.
///
/// A rayon fork-join on this machine costs tens of microseconds and, worse,
/// returns only when its slowest worker does — one preempted worker stalls
/// the whole join (see `perf-measurement-law`: isolated parallel kernel
/// timings scatter up to 21x for exactly this reason). Paying that on a
/// 4 KB copy is a loss. 32,768 elements is where the copy is large enough
/// to absorb it.
const PAR_MIN: usize = 1 << 15;

/// Value buffers kept alive between steps so the next tape can reuse them.
///
/// A training step builds ~3,000 nodes whose value buffers total ~143 MB
/// at the shipping shape, then drops them all; the next step allocates the
/// same sizes again. Freeing measured 38 ms/step (with the gradient half)
/// and re-faulting the pages ~19 ms — neither is arithmetic. Every forward
/// op writes its whole output, so a value buffer from the last step can be
/// handed straight back without zeroing; the pool keeps them by exact
/// length, and a step draws from it through [`Tape::alloc`].
///
/// GRADIENT buffers are pooled too, and they are never zeroed. A calloc'd
/// buffer is "free" to zero — the kernel hands out zero pages — but it is
/// NOT free to touch: the first write to each page takes a fault and a
/// kernel-side memset, single-threaded, on the writing thread. Measured on
/// the 8 MB table gradient of the embedding: the scatter-add into a warm
/// buffer takes 413 us and into a fresh calloc'd one 2,126 us. A step's
/// ~143 MB of gradients paid that every step, inside `backward`, where no
/// census could see it. Zeroing recycled buffers in one parallel pass
/// measured about the same (~22 ms: this laptop's DRAM write bandwidth
/// either way). So instead the backward tracks, per node, whether its
/// gradient has been written yet: the FIRST consumer to contribute
/// ASSIGNS, later ones accumulate. Nothing is zeroed, nothing is faulted.
#[derive(Default)]
pub struct BufPool {
    free: std::collections::HashMap<usize, Vec<Vec<f32>>>,
}

impl BufPool {
    pub fn new() -> Self { Self::default() }

    /// A buffer of exactly `n` elements. Its contents are whatever the
    /// previous user left — the caller MUST write every element.
    pub fn take(&mut self, n: usize) -> Vec<f32> {
        self.take_tagged(n).0
    }

    /// As [`BufPool::take`], also saying whether the buffer was recycled
    /// (`true`: contents are stale) or freshly calloc'd (`false`: zero).
    pub fn take_tagged(&mut self, n: usize) -> (Vec<f32>, bool) {
        if let Some(list) = self.free.get_mut(&n) {
            if let Some(v) = list.pop() { debug_assert_eq!(v.len(), n); return (v, true); }
        }
        (vec![0.0f32; n], false)
    }

    /// Return a buffer for reuse. Empty buffers are dropped.
    pub fn give(&mut self, v: Vec<f32>) {
        if v.is_empty() { return; }
        self.free.entry(v.len()).or_default().push(v);
    }

    /// Buffers held, and their total elements.
    pub fn stats(&self) -> (usize, usize) {
        let mut n = 0; let mut e = 0;
        for (len, l) in &self.free { n += l.len(); e += len * l.len(); }
        (n, e)
    }
}

impl Op {
    /// Short kind name, for the in-situ backward census.
    fn kind(&self) -> &'static str {
        match self {
            Op::Leaf => "leaf", Op::Add(..) => "add", Op::Mul(..) => "mul",
            Op::Embed { .. } => "embed", Op::MatMul { .. } => "matmul", Op::Silu(..) => "silu",
            Op::Rmsnorm { .. } => "rmsnorm", Op::Transpose { .. } => "transpose", Op::Rope { .. } => "rope",
            Op::SliceRows { .. } | Op::ConcatRows { .. } | Op::SliceCols { .. } | Op::ConcatCols { .. } => "slice/concat",
            Op::ScaleMaskCausal { .. } => "scale_mask", Op::SoftmaxRows { .. } => "softmax_rows",
            Op::Attention { .. } => "attention", Op::SumAll(..) => "sum", Op::Mse { .. } => "mse",
            Op::SoftmaxCE { .. } => "softmax_ce",
        }
    }
}

/// In-situ census: time per op kind across the forward (between pushes)
/// and across `backward_from` arms. Off unless `R2_TAPE_STATS=1`. Read
/// with [`take_forward_stats`] / [`take_backward_stats`]. The op-level
/// census (`step_census`) times each op in isolation with hot inputs;
/// this is what a step actually pays, at whatever shape is being trained.
fn tape_stats_on() -> bool {
    use std::sync::OnceLock;
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("R2_TAPE_STATS").map(|v| v == "1").unwrap_or(false))
}
static BWD_STATS: std::sync::Mutex<Vec<(&'static str, f32)>> = std::sync::Mutex::new(Vec::new());
static FWD_STATS: std::sync::Mutex<Vec<(&'static str, f32)>> = std::sync::Mutex::new(Vec::new());

/// Drain the forward census: `(op kind, milliseconds)` per node pushed.
pub fn take_forward_stats() -> Vec<(&'static str, f32)> {
    std::mem::take(&mut *FWD_STATS.lock().unwrap())
}

/// Drain the backward census: `(op kind, milliseconds)` per arm executed.
pub fn take_backward_stats() -> Vec<(&'static str, f32)> {
    std::mem::take(&mut *BWD_STATS.lock().unwrap())
}

impl Tape {
    pub fn new() -> Self {
        Self::with_pool(BufPool::new())
    }

    /// A tape that draws its value buffers from `pool` — the previous
    /// step's buffers, handed on by [`Tape::into_pool`].
    pub fn with_pool(pool: BufPool) -> Self {
        Tape { vals: Vec::new(), grads: Vec::new(), ops: Vec::new(),
               requires: Vec::new(), differentiated: false, pool,
               recycled_grads: 0, last_push: None, gwritten: Vec::new() }
    }

    /// Dismantle the tape: every value buffer goes into the pool for the
    /// next step, and the gradient buffers are freed on a background
    /// thread so the training thread never waits on the allocator.
    /// Parameter values taken back with [`Tape::take_value`] are already
    /// gone from `vals` and are not affected.
    pub fn into_pool(mut self) -> BufPool {
        let mut pool = std::mem::take(&mut self.pool);
        for v in self.vals.drain(..) { pool.give(v); }
        for g in self.grads.drain(..) { pool.give(g); }
        pool
    }

    /// Buffers currently held by this tape's pool (count, elements).
    pub fn pool_stats(&self) -> (usize, usize) { self.pool.stats() }

    /// An output buffer of `n` elements for a forward op. Recycled when the
    /// pool has one of that size; contents are undefined and the op must
    /// write every element.
    fn alloc(&mut self, n: usize) -> Vec<f32> { self.pool.take(n) }

    fn push(&mut self, val: Vec<f32>, op: Op, requires: bool) -> Var {
        if tape_stats_on() {
            let now = std::time::Instant::now();
            if let Some(t) = self.last_push {
                FWD_STATS.lock().unwrap().push((op.kind(), (now - t).as_secs_f32() * 1e3));
            }
            self.last_push = Some(now);
        }
        let idx = self.vals.len();
        let (g, recycled) = self.pool.take_tagged(val.len());
        if recycled { self.recycled_grads += 1; }
        self.grads.push(g);
        self.vals.push(val);
        self.ops.push(op);
        self.requires.push(requires);
        self.gwritten.push(false);
        Var(idx)
    }

    /// A leaf parameter (or input). `requires_grad` marks it for gradient
    /// accumulation (weights = true; fixed inputs = false).
    pub fn leaf(&mut self, val: Vec<f32>, requires_grad: bool) -> Var {
        self.push(val, Op::Leaf, requires_grad)
    }

    /// Number of nodes recorded. Each node owns a value buffer AND a
    /// gradient buffer, so this is also an allocation count — the figure
    /// that matters when small ops dominate a step.
    pub fn len(&self) -> usize { self.vals.len() }
    pub fn is_empty(&self) -> bool { self.vals.is_empty() }
    /// Total elements held across all node buffers.
    pub fn elements(&self) -> usize { self.vals.iter().map(|v| v.len()).sum() }

    pub fn value(&self, v: Var) -> &[f32] { &self.vals[v.0] }
    pub fn grad(&self, v: Var) -> &[f32] { &self.grads[v.0] }

    /// Take a node's value buffer back out, leaving the node empty.
    ///
    /// For the one caller that OWNS what it put on the tape: a training
    /// step pushes the model's weights as leaves and, once backward has
    /// run, wants them back. Cloning them in and copying them out again
    /// costs 29 MB each way at the shipping shape — measured, the copy in
    /// alone is **11.4 ms, 2.0% of a step**, for buffers a forward pass
    /// only ever reads. PyTorch does not copy parameters into its graph
    /// either; it references them.
    ///
    /// The node's GRADIENT is untouched, so `grad()` still works after
    /// this — which is the whole point, since the caller needs both.
    ///
    /// Only sound after the value is no longer needed: a later backward
    /// over this node would read an empty buffer. Training calls it after
    /// `backward()` and then drops the tape.
    pub fn take_value(&mut self, v: Var) -> Vec<f32> {
        std::mem::take(&mut self.vals[v.0])
    }
}

impl Default for Tape { fn default() -> Self { Tape::new() } }

/// Finite-difference gradient of a scalar function of `params` — the
/// numeric reference the analytic backward is checked against. Central
/// difference (O(h²)), h chosen for f32. Rebuilds the graph each eval via
/// the caller's closure.
pub fn finite_diff<F: Fn(&[f32]) -> f32>(params: &[f32], f: F) -> Vec<f32> {
    let h = 1e-3f32;
    let mut g = vec![0.0f32; params.len()];
    let mut p = params.to_vec();
    for i in 0..params.len() {
        let orig = p[i];
        p[i] = orig + h; let fp = f(&p);
        p[i] = orig - h; let fm = f(&p);
        p[i] = orig;
        g[i] = (fp - fm) / (2.0 * h);
    }
    g
}
