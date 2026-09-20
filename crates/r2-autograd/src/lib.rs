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

/// In-situ backward census: time per op kind across `backward_from`
/// arms, on unless... off unless `R2_TAPE_STATS=1`. Read with
/// [`take_backward_stats`]. The op-level census (`step_census`) times
/// each op in isolation with hot inputs; this is what the backward
/// actually pays, inside a step, at whatever shape is being trained.
fn tape_stats_on() -> bool {
    use std::sync::OnceLock;
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("R2_TAPE_STATS").map(|v| v == "1").unwrap_or(false))
}
static BWD_STATS: std::sync::Mutex<Vec<(&'static str, f32)>> = std::sync::Mutex::new(Vec::new());

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
               recycled_grads: 0, gwritten: Vec::new() }
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

    /// As [`Tape::alloc`], zero-filled — for the few ops that accumulate
    /// into their output rather than assigning it.
    fn alloc_zeroed(&mut self, n: usize) -> Vec<f32> {
        let mut v = self.pool.take(n);
        for x in v.iter_mut() { *x = 0.0; }
        v
    }

    fn push(&mut self, val: Vec<f32>, op: Op, requires: bool) -> Var {
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

    // ── forward ops (each records enough for backward) ─────────────────

    // ── elementwise forwards ───────────────────────────────────────────
    //
    // These were serial iterator chains over millions of elements. At the
    // shipping shape `mul` runs over 1.57M elements four times a step and
    // `add` over 524k eight times, all on one core of six.
    //
    // An elementwise map parallelises BIT-IDENTICALLY: every output depends
    // only on the input at the same index, so splitting the range changes
    // no arithmetic and no ordering. That is not true of a reduction, where
    // splitting changes the summation order — which is why those need the
    // fixed-chain treatment in `r2_tensor::ops` instead.

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let n = self.vals[a.0].len();
        let mut val = self.alloc(n);
        {
            let (va, vb) = (&self.vals[a.0], &self.vals[b.0]);
            if n >= PAR_MIN {
                use rayon::prelude::*;
                const C: usize = 1 << 14;
                val.par_chunks_mut(C).zip(va.par_chunks(C)).zip(vb.par_chunks(C))
                    .for_each(|((d, x), y)| for i in 0..d.len() { d[i] = x[i] + y[i]; });
            } else {
                for i in 0..n { val[i] = va[i] + vb[i]; }
            }
        }
        let req = self.requires[a.0] || self.requires[b.0];
        self.push(val, Op::Add(a, b), req)
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        let n = self.vals[a.0].len();
        let mut val = self.alloc(n);
        {
            let (va, vb) = (&self.vals[a.0], &self.vals[b.0]);
            if n >= PAR_MIN {
                use rayon::prelude::*;
                const C: usize = 1 << 14;
                val.par_chunks_mut(C).zip(va.par_chunks(C)).zip(vb.par_chunks(C))
                    .for_each(|((d, x), y)| for i in 0..d.len() { d[i] = x[i] * y[i]; });
            } else {
                for i in 0..n { val[i] = va[i] * vb[i]; }
            }
        }
        let req = self.requires[a.0] || self.requires[b.0];
        self.push(val, Op::Mul(a, b), req)
    }

    /// Gather rows `tokens` out of the `vocab x d` `table`.
    ///
    /// Panics on an out-of-range token rather than reading a neighbouring
    /// row. The one-hot form silently produced a row of zeros for such a
    /// token, which trains a subtly wrong model instead of failing.
    pub fn embed(&mut self, table: Var, tokens: &[usize], d: usize) -> Var {
        let table_len = self.vals[table.0].len();
        assert_eq!(table_len % d, 0,
                   "embed: table length {table_len} is not a multiple of d={d}");
        let vocab = table_len / d;
        if let Some(&bad) = tokens.iter().find(|&&t| t >= vocab) {
            panic!("embed: token {bad} out of range for vocab {vocab}");
        }
        let mut val = self.alloc(tokens.len() * d);
        {
            let vt = &self.vals[table.0];
            // Rows are independent, so the gather splits cleanly across
            // cores — and it needs to. Measured at 2,048 tokens x dim 256,
            // vocab 8,000: 686 us serial, 62 us on six threads. PyTorch's
            // `index_select` runs under `at::parallel_for` and costs 49 us
            // here; forcing torch to one thread puts it at 316, which is
            // the whole of the forward gap. Below the threshold the
            // fork-join costs more than the copy saves.
            if tokens.len() * d >= PAR_MIN {
                use rayon::prelude::*;
                val.par_chunks_exact_mut(d).zip(tokens.par_iter())
                    .for_each(|(dst, &tok)| {
                        dst.copy_from_slice(&vt[tok * d..tok * d + d]);
                    });
            } else {
                for (dst, &tok) in val.chunks_exact_mut(d).zip(tokens) {
                    dst.copy_from_slice(&vt[tok * d..tok * d + d]);
                }
            }
        }
        let req = self.requires[table.0];
        self.push(val, Op::Embed { table, tokens: tokens.to_vec(), d }, req)
    }

    pub fn matmul(&mut self, a: Var, b: Var, m: usize, k: usize, n: usize) -> Var {
        let mut val = self.alloc(m * n);
        r2_tensor::ops::matmul_into(&self.vals[a.0], &self.vals[b.0], m, k, n, &mut val);
        let req = self.requires[a.0] || self.requires[b.0];
        self.push(val, Op::MatMul { a, b, m, k, n }, req)
    }

    /// SiLU, `x * sigmoid(x)`.
    ///
    /// One `exp` per element is irreducible here — unlike RoPE, whose
    /// angles were redundant, every element's sigmoid is genuinely
    /// different. The backward could skip its own `exp` by storing the
    /// sigmoid computed here, but that is 6.3 MB per node and ~25 MB a
    /// step, which is the wrong trade on a machine already bound by memory
    /// traffic. What was actually wrong was that both directions ran on
    /// one core.
    pub fn silu(&mut self, x: Var) -> Var {
        let mut val = self.alloc(self.vals[x.0].len());
        let vx = &self.vals[x.0];
        // `silu_into` carries a vectorised `exp`; the scalar `f32::exp` is
        // a libm call and cannot vectorise at all.
        if vx.len() >= PAR_MIN {
            use rayon::prelude::*;
            const C: usize = 1 << 14;
            val.par_chunks_mut(C).zip(vx.par_chunks(C))
                .for_each(|(d, s)| r2_tensor::ops::silu_into(s, d));
        } else {
            r2_tensor::ops::silu_into(vx, &mut val);
        }
        let req = self.requires[x.0];
        self.push(val, Op::Silu(x), req)
    }

    pub fn rmsnorm(&mut self, x: Var, w: Var, d: usize, eps: f32) -> Var {
        let mut val = self.alloc(self.vals[x.0].len());
        r2_tensor::ops::rmsnorm_into(&self.vals[x.0], &self.vals[w.0], eps, &mut val);
        let req = self.requires[x.0] || self.requires[w.0];
        self.push(val, Op::Rmsnorm { x, w, d, eps }, req)
    }

    pub fn transpose(&mut self, x: Var, rows: usize, cols: usize) -> Var {
        let vx = &self.vals[x.0];
        let mut val = vec![0.0f32; rows * cols];
        for i in 0..rows { for j in 0..cols { val[j * rows + i] = vx[i * cols + j]; } }
        let req = self.requires[x.0];
        self.push(val, Op::Transpose { x, rows, cols }, req)
    }

    /// Apply RoPE to each head of each row, using the row index as the
    /// position. Matches r2_tensor::ops::rope_inplace exactly, so a model
    /// trained here and served by r2-tensor share one definition.
    pub fn rope(&mut self, x: Var, rows: usize, n_heads: usize, head_dim: usize, base: f32) -> Var {
        self.rope_seq(x, rows, rows, n_heads, head_dim, base)
    }

    /// RoPE for a FUSED batch: B sequences of length `period` stacked as
    /// `rows = B * period` rows. Position restarts at each sequence
    /// boundary (row % period), so every sequence is rotated exactly as it
    /// would be alone — which is what makes fused-batch logits equal
    /// per-sequence logits.
    pub fn rope_seq(&mut self, x: Var, rows: usize, period: usize,
                    n_heads: usize, head_dim: usize, base: f32) -> Var {
        let mut val = self.alloc(self.vals[x.0].len());
        val.copy_from_slice(&self.vals[x.0]);
        // The angles depend only on (position, pair), so they are built
        // ONCE and then reused by every row and every head. Calling
        // `rope_inplace` per row per head recomputed a `powf` and a
        // `sin_cos` for each element: 262,144 transcendental pairs where
        // 2,048 are distinct, and 138.9 ms of a training step.
        //
        // Rows are independent, so they also split across cores.
        let half = head_dim / 2;
        let per = period.max(1);
        let tab = r2_tensor::ops::rope_table(per, head_dim, base);
        {
            let row = |r: usize, vrow: &mut [f32]| {
                let trow = &tab[(r % per) * half..(r % per) * half + half];
                for h in 0..n_heads {
                    let off = h * head_dim;
                    for p in 0..half {
                        let (c, s) = trow[p];
                        let (a, b) = (vrow[off + 2 * p], vrow[off + 2 * p + 1]);
                        vrow[off + 2 * p] = a * c - b * s;
                        vrow[off + 2 * p + 1] = a * s + b * c;
                    }
                }
            };
            let w = n_heads * head_dim;
            if rows * w >= PAR_MIN {
                use rayon::prelude::*;
                val.par_chunks_mut(w).enumerate().for_each(|(r, vrow)| row(r, vrow));
            } else {
                val.chunks_mut(w).enumerate().for_each(|(r, vrow)| row(r, vrow));
            }
        }
        let req = self.requires[x.0];
        self.push(val, Op::Rope { x, rows, period, n_heads, head_dim, base }, req)
    }

    /// Cut rows `[start, start+len)` out of a `? × cols` matrix. Rows are
    /// contiguous, so the slice is one memcpy.
    pub fn slice_rows(&mut self, x: Var, cols: usize, start: usize, len: usize) -> Var {
        let val = self.vals[x.0][start * cols..(start + len) * cols].to_vec();
        let req = self.requires[x.0];
        self.push(val, Op::SliceRows { x, cols, start, len }, req)
    }

    /// Stack matrices with equal column counts vertically.
    pub fn concat_rows(&mut self, xs: &[Var]) -> Var {
        let mut val = Vec::new();
        for v in xs { val.extend_from_slice(&self.vals[v.0]); }
        let req = xs.iter().any(|v| self.requires[v.0]);
        self.push(val, Op::ConcatRows { xs: xs.to_vec() }, req)
    }

    /// Extract columns `[start, start+len)` from a `rows × total` matrix.
    pub fn slice_cols(&mut self, x: Var, rows: usize, total: usize, start: usize, len: usize) -> Var {
        let src = &self.vals[x.0];
        let mut val = Vec::with_capacity(rows * len);
        for r in 0..rows { val.extend_from_slice(&src[r * total + start..r * total + start + len]); }
        let req = self.requires[x.0];
        self.push(val, Op::SliceCols { x, rows, total, start, len }, req)
    }

    /// Concatenate equal-width blocks side by side.
    pub fn concat_cols(&mut self, xs: &[Var], rows: usize, each: usize) -> Var {
        let n = xs.len();
        let mut val = vec![0.0f32; rows * n * each];
        for (i, v) in xs.iter().enumerate() {
            let src = &self.vals[v.0];
            for r in 0..rows {
                val[r * n * each + i * each..r * n * each + (i + 1) * each]
                    .copy_from_slice(&src[r * each..(r + 1) * each]);
            }
        }
        let req = xs.iter().any(|v| self.requires[v.0]);
        self.push(val, Op::ConcatCols { xs: xs.to_vec(), rows, each }, req)
    }

    /// Scale scores by `scale` and mask out future positions.
    pub fn scale_mask_causal(&mut self, x: Var, t: usize, scale: f32) -> Var {
        let src = &self.vals[x.0];
        let mut val = vec![0.0f32; t * t];
        for i in 0..t {
            for j in 0..t {
                val[i * t + j] = if j <= i { src[i * t + j] * scale } else { f32::NEG_INFINITY };
            }
        }
        let req = self.requires[x.0];
        self.push(val, Op::ScaleMaskCausal { x, t, scale }, req)
    }

    pub fn softmax_rows(&mut self, x: Var, d: usize) -> Var {
        let val = r2_tensor::ops::softmax(&self.vals[x.0], d);
        let req = self.requires[x.0];
        self.push(val, Op::SoftmaxRows { x, d }, req)
    }

    /// Fused grouped-query causal attention. See [`Op::Attention`].
    ///
    /// `scale` is applied to the scores before the mask, matching
    /// `scale_mask_causal` — for standard attention pass
    /// `1.0 / (head_dim as f32).sqrt()`.
    ///
    /// Softmax is taken over `j <= i` only. That is exactly equivalent to
    /// masking with `-inf` and softmaxing the full row, since
    /// `exp(-inf - max) == 0`, but it never writes the masked half: the
    /// score row is `i+1` long, not `seq` long, so the block does half the
    /// score work the decomposition did and materialises none of it.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(&mut self, q: Var, k: Var, v: Var, nseq: usize, seq: usize,
                     nh: usize, nkv: usize, hd: usize, scale: f32) -> Var {
        assert!(nh % nkv == 0, "attention: {nh} query heads is not a multiple of {nkv} kv heads");
        let rows = nseq * seq;
        assert_eq!(self.vals[q.0].len(), rows * nh * hd, "attention: q has the wrong length");
        assert_eq!(self.vals[k.0].len(), rows * nkv * hd, "attention: k has the wrong length");
        assert_eq!(self.vals[v.0].len(), rows * nkv * hd, "attention: v has the wrong length");

        let mut out = self.alloc(rows * nh * hd);
        {
            let (vq, vk, vv) = (&self.vals[q.0], &self.vals[k.0], &self.vals[v.0]);
            // Split by SEQUENCE. Sequence `s` reads and writes only its own
            // `seq` rows of q/k/v/out, so the workers are disjoint without
            // any coordination — and a token still cannot see another
            // example's tokens, which is the property the whole block
            // exists to preserve.
            let work = |s: usize, oblk: &mut [f32]| {
                attn_forward_seq(vq, vk, vv, oblk, s, seq, nh, nkv, hd, scale);
            };
            if nseq > 1 && rows * nh * hd >= PAR_MIN {
                use rayon::prelude::*;
                out.par_chunks_mut(seq * nh * hd).enumerate()
                    .for_each(|(s, oblk)| work(s, oblk));
            } else {
                out.chunks_mut(seq * nh * hd).enumerate()
                    .for_each(|(s, oblk)| work(s, oblk));
            }
        }
        let req = self.requires[q.0] || self.requires[k.0] || self.requires[v.0];
        self.push(out, Op::Attention { q, k, v, nseq, seq, nh, nkv, hd, scale }, req)
    }

    pub fn sum_all(&mut self, x: Var) -> Var {
        let s: f32 = self.vals[x.0].iter().sum();
        let req = self.requires[x.0];
        self.push(vec![s], Op::SumAll(x), req)
    }

    pub fn mse(&mut self, pred: Var, target: Vec<f32>) -> Var {
        let n = target.len() as f32;
        let s: f32 = self.vals[pred.0].iter().zip(&target).map(|(p, t)| (p - t) * (p - t)).sum();
        let req = self.requires[pred.0];
        self.push(vec![s / n], Op::Mse { pred, target }, req)
    }

    /// Fused softmax + cross-entropy, via the log-sum-exp.
    ///
    /// `-ln(softmax(x)[t])` is `lse(x) - x[t]` exactly, so the loss needs
    /// only a per-row max and a per-row sum — never the probabilities
    /// themselves. The previous form materialised the whole `rows x d`
    /// probability matrix to read one value per row out of it: 64 MB
    /// written and 8 KB used, at vocab 8,000.
    ///
    /// It is also better conditioned. The old form computed
    /// `-ln(p.max(1e-30))`, which silently clamps a confidently-wrong
    /// prediction to a loss of 69 instead of reporting it; `lse - x[t]`
    /// has no such floor and no division.
    pub fn softmax_ce(&mut self, logits: Var, d: usize, targets: Vec<usize>) -> Var {
        let x = &self.vals[logits.0];
        let rows = targets.len();
        assert_eq!(x.len(), rows * d, "softmax_ce: logits are not rows x d");
        let mut lse = vec![0.0f32; rows];
        let mut loss = 0.0f32;
        for (r, &t) in targets.iter().enumerate() {
            let row = &x[r * d..r * d + d];
            debug_assert!(t < d, "softmax_ce: target {t} out of range for d={d}");
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let l = m + r2_tensor::ops::exp_shift_sum_only(row, m).ln();
            lse[r] = l;
            loss += l - row[t];
        }
        loss /= rows as f32;
        let req = self.requires[logits.0];
        self.push(vec![loss], Op::SoftmaxCE { logits, d, targets, lse }, req)
    }

    // ── backward: reverse-mode gradient accumulation ───────────────────

    /// Seed the given scalar output with grad 1 and propagate to all
    /// `requires_grad` leaves. `loss` must be a length-1 node.
    pub fn backward(&mut self, loss: Var) {
        assert_eq!(self.vals[loss.0].len(), 1, "backward() expects a scalar loss");
        self.backward_from(loss, &[1.0]);
    }

    /// Census hook: just the blanket zeroing `backward()` opens with, so it
    /// can be timed on its own. Not part of the differentiation API.
    #[doc(hidden)]
    pub fn zero_grads_census(&mut self) {
        for gb in self.grads.iter_mut() { gb.fill(0.0); }
    }

    /// Seed an output of ANY shape with a supplied gradient and propagate.
    ///
    /// This is PyTorch's `y.backward(g)`. Training uses [`Tape::backward`] —
    /// a scalar loss seeded with 1 — but a BENCHMARK of one op must not
    /// have to invent a scalar to get a gradient flowing. LMO-1 used to
    /// build `mul(x, g)` then `sum_all` for that, which put an extra
    /// elementwise multiply, an extra leaf, a reduction and their backwards
    /// on R2's side of a comparison whose other side was a bare
    /// `.backward(g)`. That tail measured ~4 ms of a ~6.6 ms reading: the
    /// benchmark was reporting R2's harness, not R2's embedding.
    ///
    /// `seed` must match `v`'s length.
    pub fn backward_from(&mut self, v: Var, seed: &[f32]) {
        assert_eq!(self.vals[v.0].len(), seed.len(),
                   "backward_from: seed is {} long, node is {}",
                   seed.len(), self.vals[v.0].len());
        // Clear only what a PREVIOUS backward dirtied.
        //
        // `push` allocates every gradient buffer with `vec![0.0; n]`, so on
        // a tape's first backward they are already zero and this memset
        // writes zeros over zeros. Training builds a fresh `Tape` per step
        // (`Trainer::train_step`), which makes that every step: measured by
        // `--example step_census`, 71.86M elements — 287 MB of memset —
        // for **27.7 ms, 4.8% of a step**, achieving nothing.
        //
        // Worse than the write itself: `fill` TOUCHES every page, forcing
        // resident what the allocator had handed out as untouched zero
        // pages, including the gradient buffers of `requires = false` nodes
        // that no backward will ever write.
        //
        // Threading it was tried and measured WORSE (187.5 -> 191.5 s on
        // the 30-step BPE arm): a fork-join per buffer costs more than the
        // memset. That was the right measurement of the wrong fix — the
        // memset should not happen at all. PyTorch has no equivalent step
        // either; its gradients are freshly-allocated outputs and
        // `w.grad = None` lets AccumulateGrad take ownership.
        //
        // A second backward on the SAME tape does need the reset, because
        // the first one left gradients in those buffers.
        // No zeroing pass, ever. Each arm below asks `first_write(node)`
        // before touching a gradient: the first writer assigns the whole
        // buffer (or zeroes it and accumulates, for the arms that write
        // only part of it), later writers accumulate. A second backward on
        // the same tape simply starts the bookkeeping again.
        for w in self.gwritten.iter_mut() { *w = false; }
        self.differentiated = true;
        self.grads[v.0].copy_from_slice(seed);
        self.gwritten[v.0] = true;

        // Nodes were pushed in topological order → reverse index order is
        // a valid reverse-topological walk.
        let census = tape_stats_on();
        for i in (0..self.ops.len()).rev() {
            let t_arm = if census { Some(std::time::Instant::now()) } else { None };
            // MOVE this node's accumulated gradient out rather than cloning
            // it. A tape has thousands of nodes and every one was being
            // deep-copied here — allocation, not arithmetic, dominated the
            // backward pass. Nothing writes to node i during its own arm
            // (a node's inputs always have lower indices in a DAG), so the
            // buffer is safe to borrow away and hand back.
            let g = std::mem::take(&mut self.grads[i]);
            match &self.ops[i] {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    let (a, b) = (a.0, b.0);
                    const C: usize = 1 << 14;
                    let par = g.len() >= PAR_MIN;
                    for side in [a, b] {
                        if !self.requires[side] { continue; }
                        let assign = first_write(&mut self.gwritten, side);
                        if par {
                            use rayon::prelude::*;
                            self.grads[side].par_chunks_mut(C).zip(g.par_chunks(C))
                                .for_each(|(gd, gi)| {
                                    if assign { gd.copy_from_slice(gi); }
                                    else { for (d, s) in gd.iter_mut().zip(gi) { *d += s; } }
                                });
                        } else if assign {
                            self.grads[side].copy_from_slice(&g);
                        } else {
                            for (gd, gi) in self.grads[side].iter_mut().zip(&g) { *gd += gi; }
                        }
                    }
                }
                Op::Mul(a, b) => {
                    let (a, b) = (a.0, b.0);
                    // `vals` and `grads` are DISJOINT fields, so borrow them
                    // as such instead of cloning both operands — the same
                    // trick the MatMul arm already uses. Those two clones
                    // were a full copy of each input on every backward
                    // (2 MB each at 2,048 tokens x dim 256).
                    //
                    // The `requires` guards skip a side whose subtree holds
                    // no parameter at all. A constant multiplier — an
                    // attention mask, an upstream gradient fed in as data —
                    // was getting a full gradient computed into a buffer
                    // nothing would ever read.
                    let Tape { vals, grads, requires, gwritten, .. } = self;
                    const C: usize = 1 << 14;
                    let par = g.len() >= PAR_MIN;
                    // grad of `a` reads the value of `b`, and vice versa.
                    for (dst, src) in [(a, b), (b, a)] {
                        if !requires[dst] { continue; }
                        let assign = first_write(gwritten, dst);
                        let other = &vals[src];
                        if par {
                            use rayon::prelude::*;
                            grads[dst].par_chunks_mut(C).zip(g.par_chunks(C))
                                .zip(other.par_chunks(C))
                                .for_each(|((gd, gi), o)| {
                                    if assign { for ((d, s), o) in gd.iter_mut().zip(gi).zip(o) { *d = s * o; } }
                                    else { for ((d, s), o) in gd.iter_mut().zip(gi).zip(o) { *d += s * o; } }
                                });
                        } else if assign {
                            for ((d, s), o) in grads[dst].iter_mut().zip(&g).zip(other) { *d = s * o; }
                        } else {
                            for ((d, s), o) in grads[dst].iter_mut().zip(&g).zip(other) { *d += s * o; }
                        }
                    }
                }
                Op::Embed { table, tokens, d } => {
                    // The adjoint of a gather is a SCATTER-ADD. `+=`, never
                    // `=`: a token appearing twice must accumulate both
                    // contributions, and dropping one is a silent error on
                    // exactly the commonest tokens.
                    let (ti, d) = (table.0, *d);
                    if self.requires[ti] {
                        // A scatter touches only the rows of tokens present,
                        // so the first write must zero the table gradient
                        // (8 MB at vocab 8,000) before accumulating. Done in
                        // parallel: warm pages, six threads.
                        if first_write(&mut self.gwritten, ti) { zero_par(&mut self.grads[ti]); }
                        let gt = &mut self.grads[ti];
                        // Threading a scatter-add needs care: two tokens can
                        // hit the SAME row, so splitting the tokens across
                        // workers would race. Split the TABLE instead — each
                        // worker owns a disjoint block of rows and scans the
                        // token list for the ones landing in it. No locks, no
                        // atomics, and because each row's contributions are
                        // still applied in ascending token order the result is
                        // BIT-IDENTICAL to the serial loop; float addition is
                        // not associative, so anything less would make the
                        // gradient depend on the thread count.
                        //
                        // The redundant scan is `threads * tokens` integer
                        // compares — 12k at 2,048 tokens — against a copy of
                        // `tokens * d` floats. Measured at vocab 8,000:
                        // 1,293 us serial, 519 us on six threads, versus
                        // PyTorch's `embedding_dense_backward` at 336 us
                        // threaded and 1,541 us on one thread.
                        let nthreads = rayon::current_num_threads();
                        if tokens.len() * d >= PAR_MIN && nthreads > 1 {
                            use rayon::prelude::*;
                            let vocab = gt.len() / d;
                            let rows_per = vocab.div_ceil(nthreads);
                            gt.par_chunks_mut(rows_per * d).enumerate()
                                .for_each(|(w, blk)| {
                                    let lo = w * rows_per;
                                    let hi = lo + blk.len() / d;
                                    for (i, &tok) in tokens.iter().enumerate() {
                                        if tok >= lo && tok < hi {
                                            let off = (tok - lo) * d;
                                            for (o, s) in blk[off..off + d].iter_mut()
                                                .zip(&g[i * d..i * d + d]) { *o += s; }
                                        }
                                    }
                                });
                        } else {
                            for (i, &tok) in tokens.iter().enumerate() {
                                let dst = &mut gt[tok * d..tok * d + d];
                                for (o, s) in dst.iter_mut().zip(&g[i * d..i * d + d]) {
                                    *o += s;
                                }
                            }
                        }
                    }
                }
                Op::MatMul { a, b, m, k, n } => {
                    let (ai, bi, m, k, n) = (a.0, b.0, *m, *k, *n);
                    // Neither gradient is worth computing into a subtree
                    // that holds no parameter. This is not a micro-saving:
                    // the one-hot embedding form is `matmul(onehot, table)`
                    // with `requires(onehot) == false`, and grad_A there is
                    // `g . tableT` — a t x vocab matrix costing
                    // 2*t*vocab*d = 8.4 GFLOP at 2,048 tokens and vocab
                    // 8,000, roughly HALF that path's 658 ms, written into a
                    // buffer nothing reads. Every frozen input — embedded
                    // constants, a masked score matrix, a distillation
                    // teacher's activations — gets the same relief.
                    // NB: no early `continue` here — the end of this loop
                    // body hands `g` back to `self.grads[i]`, and skipping
                    // that leaves the node's gradient an empty Vec. The two
                    // guards below skip the work without skipping the
                    // hand-back.
                    let (need_a, need_b) = (self.requires[ai], self.requires[bi]);

                    // Backward is ~2/3 of training FLOPs. When the Oracle
                    // routes this shape to the GPU, express both gradients
                    // as matmuls so they use the SAME accelerated kernel as
                    // the forward pass — otherwise the GPU only ever sees a
                    // third of the work and cannot pay for itself.
                    //
                    // grad_A = g·Bᵀ and grad_B = Aᵀ·g. The transposes are
                    // O(size) against an O(m·k·n) multiply, so they are
                    // cheap at exactly the sizes this branch is taken.
                    if matches!(
                        r2_oracle::dispatch(r2_oracle::Op::TensorMatMul,
                                            r2_oracle::Shape::nmk(m, n, k)),
                        r2_oracle::Backend::Gpu)
                    {
                        if need_a {
                            let bt = transpose_of(&self.vals[bi], k, n);   // n×k
                            let ga = r2_tensor::ops::matmul(&g, &bt, m, n, k);
                            if first_write(&mut self.gwritten, ai) { self.grads[ai].copy_from_slice(&ga); }
                            else { for (dst, v) in self.grads[ai].iter_mut().zip(&ga) { *dst += v; } }
                        }
                        if need_b {
                            let at = transpose_of(&self.vals[ai], m, k);   // k×m
                            let gb = r2_tensor::ops::matmul(&at, &g, k, m, n);
                            if first_write(&mut self.gwritten, bi) { self.grads[bi].copy_from_slice(&gb); }
                            else { for (dst, v) in self.grads[bi].iter_mut().zip(&gb) { *dst += v; } }
                        }
                        // Hand the buffer back before skipping the rest —
                        // `continue` used to jump over the assignment at the
                        // bottom of the loop, leaving this node's gradient an
                        // EMPTY Vec. A second `backward()` on the same tape
                        // then read zero gradient out of it, and `grad()`
                        // returned an empty slice, both silently.
                        self.grads[i] = g;
                        continue;
                    }
                    // grad_A(m×k) = g(m×n) · Bᵀ(n×k) and
                    // grad_B(k×n) = Aᵀ(k×m) · g(m×n).
                    //
                    // These are the NT and TN cases of one GEMM, and they
                    // are now CALLS to it rather than two hand-written
                    // loop nests. What was here before was ~200 lines of
                    // blocking-free triple loops, and it showed: measured
                    // against the best of PyTorch and JAX on the shapes
                    // this model runs, grad_A was 12.6-25.4x behind and
                    // grad_B 3.0-14.6x. On the output-head shape alone,
                    // grad_A took 676 ms and grad_B 465 ms, against 77 ms
                    // and 110 ms for the same arithmetic through `gemm`.
                    //
                    // Neither transpose is materialised. `gemm`'s packing
                    // pass already moves every element, so it reads the
                    // operand transposed for free — which is why
                    // `REPORT.md` records materialising them as a
                    // REJECTED attempt at 393 ms against 326.
                    use r2_linalg::gemm::{sgemm_assign_into, sgemm_into, Trans};
                    let par = m * k * n >= PAR_MIN;
                    if need_a {
                        // (M, K, N) = (m, n, k); B is stored k×n, which IS
                        // the N×K the transposed read wants.
                        let Tape { vals, grads, gwritten, .. } = self;
                        let f = if first_write(gwritten, ai) { sgemm_assign_into } else { sgemm_into };
                        f(&g, Trans::No, &vals[bi], Trans::Yes, m, n, k, &mut grads[ai], par);
                    }
                    if need_b {
                        // (M, K, N) = (k, m, n); A is stored m×k, which IS
                        // the K×M the transposed read wants.
                        let Tape { vals, grads, gwritten, .. } = self;
                        let f = if first_write(gwritten, bi) { sgemm_assign_into } else { sgemm_into };
                        f(&vals[ai], Trans::Yes, &g, Trans::No, k, m, n, &mut grads[bi], par);
                    }
                }
                Op::Silu(x) => {
                    let xi = x.0;
                    if self.requires[xi] {
                        // `vals` and `grads` are DISJOINT fields, so borrow
                        // them as such. This used to `clone()` the whole
                        // input first — 6.3 MB per call at the shipping
                        // shape, four times a step, to read it once.
                        let Tape { vals, grads, gwritten, .. } = self;
                        let acc = !first_write(gwritten, xi);
                        let vx = &vals[xi];
                        // d/dv [v*s] = s + v*s*(1-s), with the sigmoid's
                        // `exp` vectorised — same reason as the forward.
                        let work = |gx: &mut [f32], gi: &[f32], v: &[f32]| {
                            r2_tensor::ops::silu_bwd_acc(v, gi, gx, acc);
                        };
                        if g.len() >= PAR_MIN {
                            use rayon::prelude::*;
                            const C: usize = 1 << 14;
                            grads[xi].par_chunks_mut(C).zip(g.par_chunks(C))
                                .zip(vx.par_chunks(C))
                                .for_each(|((gx, gi), v)| work(gx, gi, v));
                        } else {
                            work(&mut grads[xi], &g, vx);
                        }
                    }
                }
                Op::Rmsnorm { x, w, d, eps } => {
                    let (xi, wi, d, eps) = (x.0, w.0, *d, *eps);
                    let assign_x = first_write(&mut self.gwritten, xi);
                    if first_write(&mut self.gwritten, wi) { self.grads[wi].fill(0.0); }
                    // `vals` and `grads` are disjoint fields: borrow, do not
                    // clone (this used to copy the 6 MB input per call).
                    let Tape { vals, grads, .. } = self;
                    let (vx, vw) = (&vals[xi], &vals[wi]);
                    let rows = vx.len() / d;
                    // Rows are independent, so dL/dx is row-parallel and
                    // bit-identical to the serial loop; each row's 1/rms is
                    // kept so the weight gradient below — a sum over rows,
                    // whose order must not change — reads it once.
                    let mut rinv = vec![0.0f32; rows];
                    {
                        let row = |r: usize, gx: &mut [f32], rinv_r: &mut f32| {
                            let xr = &vx[r * d..r * d + d];
                            let gr = &g[r * d..r * d + d];
                            let ms = r2_tensor::ops::sum_sq4(xr) / d as f32;
                            let ri = 1.0 / (ms + eps).sqrt();
                            *rinv_r = ri;
                            // s = Σ_j g_j w_j x_j
                            let s = r2_tensor::ops::dot3_4(gr, vw, xr);
                            let coef = ri * ri * ri / d as f32;
                            for j in 0..d {
                                // dL/dx_i = g_i w_i r  -  r³ x_i/d * s
                                let dx = gr[j] * vw[j] * ri - coef * xr[j] * s;
                                if assign_x { gx[j] = dx; } else { gx[j] += dx; }
                            }
                        };
                        if rows * d >= PAR_MIN {
                            use rayon::prelude::*;
                            grads[xi].par_chunks_mut(d).zip(rinv.par_iter_mut()).enumerate()
                                .for_each(|(r, (gx, ri))| row(r, gx, ri));
                        } else {
                            for (r, (gx, ri)) in grads[xi].chunks_mut(d).zip(rinv.iter_mut()).enumerate() { row(r, gx, ri); }
                        }
                    }
                    // dL/dw_j = Σ_r g_rj x_rj r_r — in row order, as before.
                    let gw = &mut grads[wi];
                    for r in 0..rows {
                        let xr = &vx[r * d..r * d + d];
                        let gr = &g[r * d..r * d + d];
                        let ri = rinv[r];
                        for j in 0..d { gw[j] += gr[j] * xr[j] * ri; }
                    }
                }
                Op::Transpose { x, rows, cols } => {
                    let (xi, rows, cols) = (x.0, *rows, *cols);
                    zero_if_first(&mut self.gwritten, &mut self.grads, xi);
                    // grad_x[i,j] += g[j,i]
                    for i in 0..rows { for j in 0..cols {
                        self.grads[xi][i * cols + j] += g[j * rows + i];
                    }}
                }
                Op::Rope { x, rows, period, n_heads, head_dim, base } => {
                    // Rotation is orthogonal: the adjoint is the inverse
                    // rotation, i.e. the same op at angle -theta.
                    let (xi, rows, period, nh, hd, base) =
                        (x.0, *rows, *period, *n_heads, *head_dim, *base);
                    // Same table as the forward, and for the same reason:
                    // the angle is a function of (position, pair) alone.
                    let (half, per) = (hd / 2, period.max(1));
                    let tab = r2_tensor::ops::rope_table(per, hd, base);
                    let w = nh * hd;
                    let assign = first_write(&mut self.gwritten, xi);
                    let row = |r: usize, grow: &mut [f32]| {
                        // Same position mapping as the forward pass: in a
                        // fused batch the angle restarts each sequence.
                        let trow = &tab[(r % per) * half..(r % per) * half + half];
                        let gsrc = &g[r * w..r * w + w];
                        for h in 0..nh {
                            let off = h * hd;
                            for p in 0..half {
                                let (c, s) = trow[p];
                                let (ga, gb) = (gsrc[off + 2 * p], gsrc[off + 2 * p + 1]);
                                // Inverse of [c -s; s c] is [c s; -s c].
                                let (ra, rb) = (ga * c + gb * s, -ga * s + gb * c);
                                if assign { grow[off + 2 * p] = ra; grow[off + 2 * p + 1] = rb; }
                                else { grow[off + 2 * p] += ra; grow[off + 2 * p + 1] += rb; }
                            }
                        }
                    };
                    if rows * w >= PAR_MIN {
                        use rayon::prelude::*;
                        self.grads[xi].par_chunks_mut(w).enumerate()
                            .for_each(|(r, grow)| row(r, grow));
                    } else {
                        self.grads[xi].chunks_mut(w).enumerate()
                            .for_each(|(r, grow)| row(r, grow));
                    }
                }
                Op::SliceRows { x, cols, start, len } => {
                    // Rows are contiguous: the adjoint scatters the
                    // incoming gradient back into its row range.
                    let (xi, cols, start, len) = (x.0, *cols, *start, *len);
                    zero_if_first(&mut self.gwritten, &mut self.grads, xi);
                    let base = start * cols;
                    for i in 0..len * cols { self.grads[xi][base + i] += g[i]; }
                }
                Op::ConcatRows { xs } => {
                    // Each input owns a contiguous slab of the output.
                    let mut off = 0usize;
                    for v in xs {
                        zero_if_first(&mut self.gwritten, &mut self.grads, v.0);
                        let n = self.grads[v.0].len();
                        for i in 0..n { self.grads[v.0][i] += g[off + i]; }
                        off += n;
                    }
                }
                Op::SliceCols { x, rows, total, start, len } => {
                    let (xi, rows, total, start, len) = (x.0, *rows, *total, *start, *len);
                    zero_if_first(&mut self.gwritten, &mut self.grads, xi);
                    for r in 0..rows { for j in 0..len {
                        self.grads[xi][r * total + start + j] += g[r * len + j];
                    }}
                }
                Op::ConcatCols { xs, rows, each } => {
                    let (rows, each, n) = (*rows, *each, xs.len());
                    for (i, v) in xs.iter().enumerate() {
                        zero_if_first(&mut self.gwritten, &mut self.grads, v.0);
                        for r in 0..rows { for j in 0..each {
                            self.grads[v.0][r * each + j] += g[r * n * each + i * each + j];
                        }}
                    }
                }
                Op::ScaleMaskCausal { x, t, scale } => {
                    let (xi, t, scale) = (x.0, *t, *scale);
                    zero_if_first(&mut self.gwritten, &mut self.grads, xi);
                    // Masked entries are constants (-inf), so they pass no
                    // gradient back — a token cannot learn from its future.
                    for i in 0..t { for j in 0..=i {
                        self.grads[xi][i * t + j] += g[i * t + j] * scale;
                    }}
                }
                Op::SoftmaxRows { x, d } => {
                    let (xi, d) = (x.0, *d);
                    zero_if_first(&mut self.gwritten, &mut self.grads, xi);
                    let y = self.vals[i].clone(); // this node's value = softmax
                    let rows = y.len() / d;
                    for r in 0..rows {
                        let yr = &y[r * d..r * d + d];
                        let gr = &g[r * d..r * d + d];
                        // dot = Σ_j g_j y_j ; dL/dx_i = y_i (g_i − dot)
                        let dot = dot4(gr, yr);
                        for j in 0..d {
                            self.grads[xi][r * d + j] += yr[j] * (gr[j] - dot);
                        }
                    }
                }
                Op::Attention { q, k, v, nseq, seq, nh, nkv, hd, scale } => {
                    let (qi, ki, vi) = (q.0, k.0, v.0);
                    let (nseq, seq, nh, nkv, hd, scale) =
                        (*nseq, *seq, *nh, *nkv, *hd, *scale);
                    let rows = nseq * seq;
                    let (need_q, need_k, need_v) =
                        (self.requires[qi], self.requires[ki], self.requires[vi]);
                    if need_q || need_k || need_v {
                        // The probabilities are RECOMPUTED rather than
                        // stored. Storing them costs nseq*nh*seq*seq floats
                        // — 2 MB at the shipping shape and quadratic in the
                        // context — for one saved QK pass. Recompute is
                        // what flash-attention does and for the same
                        // reason: the memory is worth more than the flops.
                        //
                        // Accumulate into local buffers, then add into the
                        // tape's gradients. Three entries of `self.grads`
                        // cannot be borrowed mutably at once, and the adds
                        // are O(size) against an O(nseq*nh*seq^2*hd)
                        // backward.
                        // Scratch from the pool (warm pages), zeroed in
                        // parallel: the sequence kernel accumulates into it.
                        let mut gq = self.pool.take(rows * nh * hd);  zero_par(&mut gq);
                        let mut gk = self.pool.take(rows * nkv * hd); zero_par(&mut gk);
                        let mut gv = self.pool.take(rows * nkv * hd); zero_par(&mut gv);
                        {
                            let (vq, vk, vv) = (&self.vals[qi], &self.vals[ki], &self.vals[vi]);
                            // Same disjoint-by-sequence split as the
                            // forward. Within a worker the query heads run
                            // in order, so several heads sharing one kv
                            // head accumulate into it serially — grouped
                            // query attention needs that and it is free
                            // here.
                            let work = |s: usize, gqb: &mut [f32], gkb: &mut [f32], gvb: &mut [f32]| {
                                attn_backward_seq(vq, vk, vv, &g, gqb, gkb, gvb,
                                                  s, seq, nh, nkv, hd, scale);
                            };
                            if nseq > 1 && rows * nh * hd >= PAR_MIN {
                                use rayon::prelude::*;
                                gq.par_chunks_mut(seq * nh * hd)
                                    .zip(gk.par_chunks_mut(seq * nkv * hd))
                                    .zip(gv.par_chunks_mut(seq * nkv * hd))
                                    .enumerate()
                                    .for_each(|(s, ((gqb, gkb), gvb))| work(s, gqb, gkb, gvb));
                            } else {
                                for s in 0..nseq {
                                    let (a, b, c) = (seq * nh * hd, seq * nkv * hd, seq * nkv * hd);
                                    work(s, &mut gq[s * a..(s + 1) * a],
                                         &mut gk[s * b..(s + 1) * b],
                                         &mut gv[s * c..(s + 1) * c]);
                                }
                            }
                        }
                        for (need, node, src) in [(need_q, qi, &gq), (need_k, ki, &gk), (need_v, vi, &gv)] {
                            if !need { continue; }
                            if first_write(&mut self.gwritten, node) { self.grads[node].copy_from_slice(src); }
                            else { for (d, x) in self.grads[node].iter_mut().zip(src) { *d += x; } }
                        }
                        self.pool.give(gq); self.pool.give(gk); self.pool.give(gv);
                    }
                }
                Op::SumAll(x) => {
                    let xi = x.0;
                    zero_if_first(&mut self.gwritten, &mut self.grads, xi);
                    for gx in self.grads[xi].iter_mut() { *gx += g[0]; }
                }
                Op::Mse { pred, target } => {
                    let pi = pred.0;
                    zero_if_first(&mut self.gwritten, &mut self.grads, pi);
                    let n = target.len() as f32;
                    let vp = self.vals[pi].clone();
                    let target = target.clone();
                    for (gp, (p, t)) in self.grads[pi].iter_mut().zip(vp.iter().zip(&target)) {
                        *gp += g[0] * 2.0 * (p - t) / n;
                    }
                }
                Op::SoftmaxCE { logits, d, targets, lse } => {
                    let (li, d) = (logits.0, *d);
                    let inv = g[0] / targets.len() as f32;
                    // grad = (softmax − onehot) / batch, scaled by upstream g.
                    //
                    // `softmax(x)[j]` is `exp(x[j] - lse)`, and `lse` was
                    // computed in the forward — so this needs no probability
                    // matrix and no second reduction pass. It used to call
                    // `ops::softmax` again here, allocating and filling a
                    // second 64 MB buffer at vocab 8,000.
                    //
                    // Rows are independent and write disjoint slices of the
                    // gradient, so they split across cores with no
                    // coordination.
                    let (targets, lse) = (targets.clone(), lse.clone());
                    let Tape { vals, grads, gwritten, .. } = self;
                    let acc = !first_write(gwritten, li);
                    let xs = &vals[li];
                    let work = |r: usize, grow: &mut [f32]| {
                        r2_tensor::ops::softmax_ce_grad_acc(
                            &xs[r * d..r * d + d], lse[r], targets[r], inv, grow, acc);
                    };
                    if targets.len() * d >= PAR_MIN {
                        use rayon::prelude::*;
                        grads[li].par_chunks_mut(d).enumerate()
                            .for_each(|(r, grow)| work(r, grow));
                    } else {
                        grads[li].chunks_mut(d).enumerate()
                            .for_each(|(r, grow)| work(r, grow));
                    }
                }
            }
            // Return the buffer so grad() still reports this node.
            self.grads[i] = g;
            if let Some(t) = t_arm {
                BWD_STATS.lock().unwrap().push((self.ops[i].kind(), t.elapsed().as_secs_f32() * 1e3));
            }
        }
        // A node no arm wrote (an unused leaf, a branch not reaching the
        // seed) must still report a zero gradient. On a fresh tape its
        // buffer already is zero and is left untouched — no fault. Only a
        // recycled buffer can hold stale values.
        if self.recycled_grads > 0 {
            for i in 0..self.grads.len() {
                if !self.gwritten[i] && self.requires[i] { zero_par(&mut self.grads[i]); }
            }
        }
    }
}

/// `true` exactly once per node per backward: the caller is the FIRST
/// writer of that node's gradient and must ASSIGN (or zero, then add).
#[inline]
fn first_write(gwritten: &mut [bool], node: usize) -> bool {
    let first = !gwritten[node];
    gwritten[node] = true;
    first
}

/// For arms that write only PART of a gradient (a slice, a masked
/// triangle, a scatter): zero the whole buffer on the first write so the
/// untouched part reads as zero, then accumulate as before.
#[inline]
fn zero_if_first(gwritten: &mut [bool], grads: &mut [Vec<f32>], node: usize) {
    if first_write(gwritten, node) { zero_par(&mut grads[node]); }
}

/// Zero a buffer, in parallel when it is large enough to matter.
fn zero_par(v: &mut [f32]) {
    if v.len() >= PAR_MIN {
        use rayon::prelude::*;
        v.par_chunks_mut(1 << 16).for_each(|c| c.fill(0.0));
    } else {
        v.fill(0.0);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A small transformer-shaped graph run three times through ONE pool:
    /// step 1 has fresh buffers, steps 2 and 3 get recycled ones full of
    /// stale values and stale gradients. Every value and every gradient
    /// must be bit-identical to a tape with no pool at all.
    #[test]
    fn pooled_tapes_reproduce_fresh_tapes_bit_for_bit() {
        let (t, d, ffn, vocab) = (16usize, 8usize, 24usize, 40usize);
        let table: Vec<f32> = (0..vocab * d).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();
        let w1: Vec<f32> = (0..d * ffn).map(|i| ((i as f32) * 0.11).cos() * 0.3).collect();
        let w3: Vec<f32> = (0..d * ffn).map(|i| ((i as f32) * 0.23).sin() * 0.3).collect();
        let w2: Vec<f32> = (0..ffn * d).map(|i| ((i as f32) * 0.17).cos() * 0.3).collect();
        let nw: Vec<f32> = vec![1.0; d];
        let tokens: Vec<usize> = (0..t).map(|i| (i * 7 + 3) % vocab).collect();
        let targets: Vec<usize> = (0..t).map(|i| (i * 11 + 1) % vocab).collect();

        let run = |tape: &mut Tape| -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
            let lt = tape.leaf(table.clone(), true);
            let l1 = tape.leaf(w1.clone(), true);
            let l3 = tape.leaf(w3.clone(), true);
            let l2 = tape.leaf(w2.clone(), true);
            let ln = tape.leaf(nw.clone(), true);
            let x = tape.embed(lt, &tokens, d);
            let h = tape.rmsnorm(x, ln, d, 1e-5);
            let gate = tape.matmul(h, l1, t, d, ffn);
            let up = tape.matmul(h, l3, t, d, ffn);
            let act = tape.silu(gate);
            let gated = tape.mul(act, up);
            let down = tape.matmul(gated, l2, t, ffn, d);
            let y = tape.add(x, down);
            // logits via the table as an output head: t x vocab
            let tt = tape.transpose(lt, vocab, d);
            let logits = tape.matmul(y, tt, t, d, vocab);
            let loss = tape.softmax_ce(logits, vocab, targets.clone());
            tape.backward(loss);
            let vals = [x, h, gate, up, act, gated, down, y, logits, loss].iter().map(|v| tape.value(*v).to_vec()).collect();
            let grads = [lt, l1, l3, l2, ln, x, h, gate, y].iter().map(|v| tape.grad(*v).to_vec()).collect();
            (vals, grads)
        };

        let mut fresh = Tape::new();
        let want = run(&mut fresh);

        let mut pool = BufPool::new();
        for step in 0..3 {
            let mut tape = Tape::with_pool(std::mem::take(&mut pool));
            let got = run(&mut tape);
            assert_eq!(got.0, want.0, "values differ on pooled step {step}");
            assert_eq!(got.1, want.1, "gradients differ on pooled step {step}");
            let (n, _) = tape.pool_stats();
            pool = tape.into_pool();
            assert!(pool.stats().0 > n, "step {step}: into_pool did not return the buffers");
        }
    }

    /// Build attention the OLD way — slice, transpose, matmul, mask,
    /// softmax, matmul, concat — exactly as `llm.rs::forward_fused` did
    /// before `Op::Attention` existed. The fused op has to agree with this
    /// or it is not a refactor, it is a different model.
    #[allow(clippy::too_many_arguments)]
    fn attention_decomposed(tape: &mut Tape, q: Var, k: Var, v: Var, nseq: usize,
                            seq: usize, nh: usize, nkv: usize, hd: usize) -> Var {
        let group = nh / nkv;
        let mut seq_ctx: Vec<Var> = Vec::with_capacity(nseq);
        for s in 0..nseq {
            let (q_s, k_s, v_s) = if nseq == 1 {
                (q, k, v)
            } else {
                (tape.slice_rows(q, nh * hd, s * seq, seq),
                 tape.slice_rows(k, nkv * hd, s * seq, seq),
                 tape.slice_rows(v, nkv * hd, s * seq, seq))
            };
            let mut heads: Vec<Var> = Vec::with_capacity(nh);
            for qh in 0..nh {
                let kvh = qh / group;
                let qs = tape.slice_cols(q_s, seq, nh * hd, qh * hd, hd);
                let ks = tape.slice_cols(k_s, seq, nkv * hd, kvh * hd, hd);
                let vs = tape.slice_cols(v_s, seq, nkv * hd, kvh * hd, hd);
                let kt = tape.transpose(ks, seq, hd);
                let sc = tape.matmul(qs, kt, seq, hd, seq);
                let sc = tape.scale_mask_causal(sc, seq, 1.0 / (hd as f32).sqrt());
                let at = tape.softmax_rows(sc, seq);
                heads.push(tape.matmul(at, vs, seq, seq, hd));
            }
            seq_ctx.push(tape.concat_cols(&heads, seq, hd));
        }
        if nseq == 1 { seq_ctx[0] } else { tape.concat_rows(&seq_ctx) }
    }

    /// The fused op must compute the SAME function as the decomposition it
    /// replaces — value and all three gradients — or the 1,156 tape nodes
    /// it deletes were carrying meaning.
    ///
    /// Tolerance is f32 rounding, not equality: the fused form sums over
    /// `j <= i` while the decomposition softmaxes a full row containing
    /// `-inf`, so the two do the same arithmetic in a different order.
    #[test]
    fn attention_matches_the_decomposition() {
        for &(nseq, seq, nh, nkv, hd) in &[
            (1usize, 4usize, 2usize, 1usize, 4usize),   // MQA, one sequence
            (3, 5, 4, 2, 6),                            // GQA, ragged-ish
            (2, 6, 3, 3, 4),                            // MHA, no grouping
        ] {
            let rows = nseq * seq;
            let mk = |n: usize, ph: f32| -> Vec<f32> {
                (0..n).map(|i| ((i as f32) * 0.37 + ph).sin() * 0.8).collect()
            };
            let (qv, kv_, vv) = (mk(rows * nh * hd, 0.0), mk(rows * nkv * hd, 1.3),
                                 mk(rows * nkv * hd, 2.7));
            let g = mk(rows * nh * hd, 0.9);
            let scale = 1.0 / (hd as f32).sqrt();

            let mut ta = Tape::new();
            let (qa, ka, va) = (ta.leaf(qv.clone(), true), ta.leaf(kv_.clone(), true),
                                ta.leaf(vv.clone(), true));
            let oa = ta.attention(qa, ka, va, nseq, seq, nh, nkv, hd, scale);
            ta.backward_from(oa, &g);

            let mut tb = Tape::new();
            let (qb, kb, vb) = (tb.leaf(qv.clone(), true), tb.leaf(kv_.clone(), true),
                                tb.leaf(vv.clone(), true));
            let ob = attention_decomposed(&mut tb, qb, kb, vb, nseq, seq, nh, nkv, hd);
            tb.backward_from(ob, &g);

            let close = |a: &[f32], b: &[f32], what: &str| {
                assert_eq!(a.len(), b.len(), "{what}: length differs");
                for (i, (x, y)) in a.iter().zip(b).enumerate() {
                    assert!((x - y).abs() <= 2e-5 * (1.0 + y.abs()),
                            "{what}[{i}] fused {x} vs decomposed {y} \
                             (nseq {nseq} seq {seq} nh {nh} nkv {nkv} hd {hd})");
                }
            };
            close(ta.value(oa), tb.value(ob), "value");
            close(ta.grad(qa), tb.grad(qb), "grad_q");
            close(ta.grad(ka), tb.grad(kb), "grad_k");
            close(ta.grad(va), tb.grad(vb), "grad_v");
        }
    }

    /// And it must pass the same finite-difference gate as every other op,
    /// independently of the decomposition — if both were wrong the test
    /// above would still pass.
    #[test]
    fn attention_gradient_matches_finite_difference() {
        let (nseq, seq, nh, nkv, hd) = (2usize, 4usize, 2usize, 1usize, 3usize);
        let rows = nseq * seq;
        let scale = 1.0 / (hd as f32).sqrt();
        let mk = |n: usize, ph: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32) * 0.41 + ph).sin() * 0.7).collect()
        };
        let kv_ = mk(rows * nkv * hd, 1.1);
        let vv = mk(rows * nkv * hd, 2.2);
        // Differentiate w.r.t. q, with k and v fixed: a scalar loss so
        // finite differences apply.
        let qv = mk(rows * nh * hd, 0.0);
        check_grad(&qv, |t: &mut Tape, p: &[f32]| {
            let q = t.leaf(p.to_vec(), true);
            let k = t.leaf(kv_.clone(), false);
            let v = t.leaf(vv.clone(), false);
            let o = t.attention(q, k, v, nseq, seq, nh, nkv, hd, scale);
            let l = t.sum_all(o);
            (q, l)
        });
        // And w.r.t. v, which reaches the loss by a different path.
        check_grad(&vv, |t: &mut Tape, p: &[f32]| {
            let q = t.leaf(qv.clone(), false);
            let k = t.leaf(kv_.clone(), false);
            let v = t.leaf(p.to_vec(), true);
            let o = t.attention(q, k, v, nseq, seq, nh, nkv, hd, scale);
            let l = t.sum_all(o);
            (v, l)
        });
    }

    /// A token must not see its future. Perturbing position `i` of k or v
    /// may change outputs at positions >= i and must leave every earlier
    /// position bit-identical — the causal mask is the one property whose
    /// failure trains a model that cheats and still looks healthy.
    #[test]
    fn attention_is_causal() {
        let (nseq, seq, nh, nkv, hd) = (1usize, 6usize, 2usize, 1usize, 4usize);
        let rows = nseq * seq;
        let scale = 1.0 / (hd as f32).sqrt();
        let mk = |n: usize, ph: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32) * 0.29 + ph).sin()).collect()
        };
        let qv = mk(rows * nh * hd, 0.0);
        let kv_ = mk(rows * nkv * hd, 1.0);
        let vv = mk(rows * nkv * hd, 2.0);
        let run = |k: &[f32], v: &[f32]| -> Vec<f32> {
            let mut t = Tape::new();
            let (a, b, c) = (t.leaf(qv.clone(), false), t.leaf(k.to_vec(), false),
                             t.leaf(v.to_vec(), false));
            let o = t.attention(a, b, c, nseq, seq, nh, nkv, hd, scale);
            t.value(o).to_vec()
        };
        let base = run(&kv_, &vv);
        for pos in 1..seq {
            let mut k2 = kv_.clone();
            let mut v2 = vv.clone();
            for c in 0..nkv * hd {
                k2[pos * nkv * hd + c] += 3.0;
                v2[pos * nkv * hd + c] += 3.0;
            }
            let got = run(&k2, &v2);
            for i in 0..pos {
                for c in 0..nh * hd {
                    let (a, b) = (base[i * nh * hd + c], got[i * nh * hd + c]);
                    assert_eq!(a, b,
                        "changing position {pos} changed output at EARLIER position {i} \
                         (col {c}): {a} -> {b}. Attention is not causal.");
                }
            }
        }
    }

    /// Assert analytic grad (from a fresh tape built by `build`) matches
    /// the finite-difference grad of the same scalar function.
    fn check_grad<B>(params: &[f32], build: B)
    where B: Fn(&mut Tape, &[f32]) -> (Var, Var) {
        // Analytic: build tape, backward, read leaf grad.
        let mut t = Tape::new();
        let (leaf, loss) = build(&mut t, params);
        t.backward(loss);
        let analytic = t.grad(leaf).to_vec();
        // Numeric: scalar loss as a function of the leaf's params.
        let numeric = finite_diff(params, |p| {
            let mut t = Tape::new();
            let (_, loss) = build(&mut t, p);
            t.value(loss)[0]
        });
        let maxerr = analytic.iter().zip(&numeric)
            .map(|(a, n)| (a - n).abs()).fold(0.0f32, f32::max);
        assert!(maxerr < 2e-2, "grad mismatch: analytic {:?} numeric {:?}", analytic, numeric);
    }

    #[test]
    fn add_mul_chain_grad() {
        check_grad(&[1.5, -2.0, 0.5], |t, p| {
            let x = t.leaf(p.to_vec(), true);
            let c = t.leaf(vec![2.0, 3.0, -1.0], false);
            let y = t.mul(x, c);       // x*c
            let z = t.add(y, x);       // x*c + x
            let loss = t.sum_all(z);
            (x, loss)
        });
    }

    #[test]
    fn matmul_grad() {
        check_grad(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], |t, p| {
            let a = t.leaf(p.to_vec(), true);          // 2×3
            let b = t.leaf(vec![1.0, 0.5, -1.0, 2.0, 0.0, 1.5], false); // 3×2
            let c = t.matmul(a, b, 2, 3, 2);           // 2×2
            let loss = t.sum_all(c);
            (a, loss)
        });
    }

    #[test]
    fn silu_grad() {
        check_grad(&[-1.0, 0.3, 2.0, -0.5], |t, p| {
            let x = t.leaf(p.to_vec(), true);
            let y = t.silu(x);
            let loss = t.sum_all(y);
            (x, loss)
        });
    }

    #[test]
    fn rmsnorm_grad_wrt_x() {
        check_grad(&[0.5, -1.5, 2.0, 0.25], |t, p| {
            let x = t.leaf(p.to_vec(), true);
            let w = t.leaf(vec![1.0, 0.5, 1.5, 2.0], false);
            let y = t.rmsnorm(x, w, 4, 1e-5);
            let loss = t.sum_all(y);
            (x, loss)
        });
    }

    #[test]
    fn rmsnorm_grad_wrt_w() {
        check_grad(&[1.0, 0.5, 1.5, 2.0], |t, p| {
            let x = t.leaf(vec![0.5, -1.5, 2.0, 0.25], false);
            let w = t.leaf(p.to_vec(), true);
            let y = t.rmsnorm(x, w, 4, 1e-5);
            let loss = t.sum_all(y);
            (w, loss)
        });
    }

    #[test]
    fn mse_grad() {
        check_grad(&[0.2, 0.8, -0.4], |t, p| {
            let pred = t.leaf(p.to_vec(), true);
            let loss = t.mse(pred, vec![1.0, 0.0, -1.0]);
            (pred, loss)
        });
    }

    #[test]
    fn transpose_grad() {
        check_grad(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], |t, p| {
            let x = t.leaf(p.to_vec(), true);       // 2×3
            let xt = t.transpose(x, 2, 3);          // 3×2
            let c = t.leaf(vec![1.0, -2.0, 0.5, 1.5, -1.0, 2.0], false);
            let prod = t.mul(xt, c);
            let loss = t.sum_all(prod);
            (x, loss)
        });
    }

    #[test]
    fn softmax_rows_grad() {
        check_grad(&[1.0, 2.0, 0.5, -1.0, 0.3, 2.0], |t, p| {
            let x = t.leaf(p.to_vec(), true);       // 2 rows × 3
            let sm = t.softmax_rows(x, 3);
            let c = t.leaf(vec![1.0, 0.0, -1.0, 2.0, 1.0, 0.5], false);
            let prod = t.mul(sm, c);
            let loss = t.sum_all(prod);
            (x, loss)
        });
    }

    #[test]
    fn softmax_cross_entropy_grad() {
        // 2 rows × 3 classes; targets [0, 2].
        check_grad(&[2.0, 1.0, 0.1, -1.0, 0.5, 3.0], |t, p| {
            let logits = t.leaf(p.to_vec(), true);
            let loss = t.softmax_ce(logits, 3, vec![0, 2]);
            (logits, loss)
        });
    }

    #[test]
    fn tiny_mlp_trains() {
        // A 1-layer net loss must DECREASE under gradient steps — proves
        // forward+backward compose into real learning.
        let mut w = vec![0.1f32; 6]; // 3→2
        let x = vec![1.0, 2.0, -1.0]; // 1×3
        let target = vec![1.0, -1.0];
        let mut prev = f32::INFINITY;
        for _ in 0..50 {
            let mut t = Tape::new();
            let wv = t.leaf(w.clone(), true);
            let xv = t.leaf(x.clone(), false);
            let y = t.matmul(xv, wv, 1, 3, 2);
            let a = t.silu(y);
            let loss = t.mse(a, target.clone());
            t.backward(loss);
            let g = t.grad(wv).to_vec();
            for (wi, gi) in w.iter_mut().zip(&g) { *wi -= 0.1 * gi; }
            let l = t.value(loss)[0];
            assert!(l <= prev + 1e-5, "loss went up: {} -> {}", prev, l);
            prev = l;
        }
        assert!(prev < 1.0, "final loss {}", prev);
    }
}

#[cfg(test)]
mod rope_tests {
    use super::*;

    /// RoPE must pass the same finite-difference gate as every other op:
    /// the analytic backward has to match a numeric derivative, or a model
    /// trains toward the wrong thing while still appearing to converge.
    #[test]
    fn rope_gradient_matches_finite_difference() {
        let (rows, nh, hd, base) = (4usize, 2usize, 4usize, 10000.0f32);
        let n = rows * nh * hd;
        let x0: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin()).collect();
        // Scalar objective: sum of squares after rotation.
        let loss_of = |p: &[f32]| -> f32 {
            let mut t = Tape::new();
            let x = t.leaf(p.to_vec(), true);
            let r = t.rope(x, rows, nh, hd, base);
            let sq = t.mul(r, r);
            let l = t.sum_all(sq);
            t.vals[l.0][0]
        };
        let mut t = Tape::new();
        let x = t.leaf(x0.clone(), true);
        let r = t.rope(x, rows, nh, hd, base);
        let sq = t.mul(r, r);
        let l = t.sum_all(sq);
        t.backward(l);
        let analytic = t.grad(x).to_vec();
        let numeric = finite_diff(&x0, loss_of);
        for (i, (a, b)) in analytic.iter().zip(&numeric).enumerate() {
            assert!((a - b).abs() < 2e-2, "elem {i}: analytic {a} vs numeric {b}");
        }
    }

    /// A rotation preserves length — the property that lets RoPE encode
    /// position without changing the scale of what flows through it.
    #[test]
    fn rope_preserves_norm_and_matches_the_inference_kernel() {
        let (rows, nh, hd) = (3usize, 2usize, 4usize);
        let n = rows * nh * hd;
        let x0: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.21).cos()).collect();
        let mut t = Tape::new();
        let x = t.leaf(x0.clone(), false);
        let r = t.rope(x, rows, nh, hd, 10000.0);
        let out = t.vals[r.0].clone();

        let norm = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>().sqrt();
        assert!((norm(&out) - norm(&x0)).abs() < 1e-4, "RoPE must preserve norm");

        // And it must agree bit-for-bit with the serving kernel, so a
        // trained model behaves identically when served by r2-tensor.
        let mut want = x0.clone();
        for r_i in 0..rows {
            for h in 0..nh {
                let off = r_i * nh * hd + h * hd;
                r2_tensor::ops::rope_inplace(&mut want[off..off + hd], r_i, 10000.0);
            }
        }
        assert_eq!(out, want, "training RoPE must equal the inference RoPE exactly");
    }
}

/// Dot product with four independent accumulators — see the note beside
/// the same helpers in `r2_tensor::ops`, which is where the shared copies
/// live.
///
/// This one is deliberately LOCAL rather than imported from there. It is
/// called from inside `#[target_feature(enable = "avx2")]` kernels, and a
/// function defined in another crate is not reliably inlined across that
/// boundary — when it is not, the hottest loop in attention silently loses
/// the wide codegen it was given. Measured: importing it cost the training
/// step 31.05 -> 33.29 s, with no other change.
#[inline(always)]
fn dot4(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    let mut a = [0.0f32; 4];
    let full = n - n % 4;
    let mut i = 0;
    while i < full {
        a[0] += x[i] * y[i];
        a[1] += x[i + 1] * y[i + 1];
        a[2] += x[i + 2] * y[i + 2];
        a[3] += x[i + 3] * y[i + 3];
        i += 4;
    }
    let mut t = 0.0f32;
    while i < n { t += x[i] * y[i]; i += 1; }
    (a[0] + a[1]) + (a[2] + a[3]) + t
}

/// Is the wide kernel usable here? Resolved once per process.
///
/// The workspace sets no `target-cpu`, so this crate compiles for baseline
/// x86-64 — SSE2, and no FMA. `r2_linalg::gemm` already dispatches an
/// AVX2 micro-kernel at runtime for exactly this reason and gained 4x from
/// it; the attention kernels had been left on the baseline path.
#[inline]
fn have_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static OK: OnceLock<bool> = OnceLock::new();
        *OK.get_or_init(|| {
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// One sequence of fused causal attention, forward — flash-style.
///
/// `oblk` is that sequence's `seq x (nh*hd)` slice of the output; `vq`,
/// `vk`, `vv` are the WHOLE tensors and `s` selects the rows. Reading the
/// full tensors rather than slices of them is the point of the op: a slice
/// would be a copy.
///
/// # Why it is blocked and why the softmax is online
///
/// The straightforward form walks one query at a time: compute its whole
/// score row, find the max, exponentiate, normalise, then accumulate the
/// output. That is three passes over the row, and — the expensive part —
/// EVERY query re-reads all of K and all of V. Per head-pass that is
/// `seq^2 * hd` element reads; at seq 64, hd 64 it is 1 MB per head-pass
/// and 134 MB for one attention block.
///
/// Blocking the queries fixes the traffic: a `BC`-wide block of K and V is
/// loaded once and used by `BR` queries, so K/V traffic falls by `BR`.
/// What makes that legal is the ONLINE softmax (Milakov & Gimelshein; the
/// same identity flash-attention is built on) — a running max `m` and
/// running denominator `l` per query, with the accumulator rescaled by
/// `exp(m_old - m_new)` whenever a block raises the max:
///
/// ```text
///   m' = max(m, max(block))
///   acc = acc * exp(m - m')  +  Σ_j exp(s_j - m') · v_j
///   l   = l   * exp(m - m')  +  Σ_j exp(s_j - m')
/// ```
///
/// so the result is the ordinary softmax, computed in one pass over the
/// keys with nothing quadratic ever materialised.
///
/// The causal mask is structural, not a `-inf` fill: key blocks entirely
/// above the diagonal are never visited, and only the diagonal block needs
/// a per-element check. A token cannot read its future because the loop
/// bound stops it, not because a large negative number was added.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn attn_forward_seq_impl(vq: &[f32], vk: &[f32], vv: &[f32], oblk: &mut [f32],
                    s: usize, seq: usize, nh: usize, nkv: usize, hd: usize, scale: f32) {
    // ── why this is tiled the way a GEMM is ──
    //
    // The previous kernel computed every score as `dot4(q_row, k_row)`:
    // eight FMAs and then a HORIZONTAL REDUCTION per (query, key) pair.
    // The reduction is the cost — a dozen shuffles and adds per eight
    // FMAs — and it is exactly what `scaled_dot_product_attention` does
    // not pay: it forms a TILE of scores lane-parallel, broadcasting one
    // query element against a row of keys, so every FMA lane holds a
    // different key and no reduction ever happens. That needs K
    // transposed (a row of keys per head-dim), which is packed ONCE per
    // sequence and kv-head here — 16 KB at seq 64, L2-resident at 2,048
    // — and shared by every query head in the GQA group.
    //
    // With the whole score row of a query block available at once, the
    // softmax is the plain two-pass form: row max, `exp_shift_sum`,
    // normalise. The online (flash) rescaling the previous kernel carried
    // measured nothing here (K/V re-reads were already L1-served; see
    // REPORT.md), and the score row is 4 x seq floats — 32 KB at 2,048 —
    // so nothing quadratic is materialised.
    //
    // The causal mask is still structural: keys beyond the last query of
    // a block are never computed, and only the diagonal tile is checked
    // per element.
    /// Queries per block.
    const BR: usize = 4;
    /// Keys per register tile.
    const BC: usize = 16;

    let group = nh / nkv;
    let (qw, kw) = (nh * hd, nkv * hd);
    let base = s * seq;
    // Kᵀ, padded to a multiple of the tile width so the tile loop needs
    // no ragged edge: kt[d * seqp + j] = K[j][d].
    let seqp = seq.next_multiple_of(BC);
    let mut kt = vec![0.0f32; hd * seqp];
    // Score rows for one query block, then their exponentials.
    let mut sc = vec![0.0f32; BR * seqp];
    let mut ex = vec![0.0f32; seqp];
    let mut acc = vec![0.0f32; hd];

    for kvh in 0..nkv {
        // ── pack Kᵀ for this kv-head ──
        for j in 0..seq {
            let koff = (base + j) * kw + kvh * hd;
            let krow = &vk[koff..koff + hd];
            for d in 0..hd { kt[d * seqp + j] = krow[d]; }
        }
        for qh in kvh * group..(kvh + 1) * group {
            for i0 in (0..seq).step_by(BR) {
                let br = BR.min(seq - i0);
                let jmax = i0 + br - 1;           // furthest key any query here can see
                let tiles = (jmax / BC) + 1;      // key tiles that reach it
                // ── scores: BR x (tiles*BC), lane-parallel over keys ──
                for t in 0..tiles {
                    let j0 = t * BC;
                    let mut tile = [[0.0f32; BC]; BR];
                    for d in 0..hd {
                        let kr = &kt[d * seqp + j0..d * seqp + j0 + BC];
                        for ii in 0..br {
                            let q = vq[(base + i0 + ii) * qw + qh * hd + d];
                            let row = &mut tile[ii];
                            for jj in 0..BC { row[jj] += q * kr[jj]; }
                        }
                    }
                    for ii in 0..br {
                        let dst = &mut sc[ii * seqp + j0..ii * seqp + j0 + BC];
                        for jj in 0..BC { dst[jj] = tile[ii][jj] * scale; }
                    }
                }
                // ── softmax + PV per query ──
                for ii in 0..br {
                    let qi = i0 + ii;
                    let row = &mut sc[ii * seqp..ii * seqp + qi + 1];  // causal: keys 0..=qi
                    let mut m = f32::NEG_INFINITY;
                    for &x in row.iter() { if x > m { m = x; } }
                    let l = r2_tensor::ops::exp_shift_sum(row, m, &mut ex[..qi + 1]);
                    for x in acc.iter_mut() { *x = 0.0; }
                    for j in 0..=qi {
                        let e = ex[j];
                        if e == 0.0 { continue; }
                        let voff = (base + j) * kw + kvh * hd;
                        let vrow = &vv[voff..voff + hd];
                        for c in 0..hd { acc[c] += e * vrow[c]; }
                    }
                    let inv = 1.0 / l;
                    // `oblk` is this sequence's own slice: local row index.
                    let dst = &mut oblk[qi * qw + qh * hd..qi * qw + qh * hd + hd];
                    for (d, v) in dst.iter_mut().zip(&acc) { *d = v * inv; }
                }
            }
        }
    }
}

/// One sequence of fused causal attention, backward.
///
/// `d out[i] = Σ_j p[i,j] v[j]` gives, with `s[i,j] = scale · q[i]·k[j]`:
///
/// ```text
/// grad_v[j] += Σ_i p[i,j] g[i]
/// grad_p[i,j] = g[i]·v[j]
/// grad_s[i,j] = p[i,j] (grad_p[i,j] − Σ_l p[i,l] grad_p[i,l])   (softmax)
/// grad_q[i] += scale Σ_j grad_s[i,j] k[j]
/// grad_k[j] += scale Σ_i grad_s[i,j] q[i]
/// ```
///
/// Masked positions (`j > i`) never enter any sum, which is what stops a
/// token receiving gradient from its future.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn attn_backward_seq_impl(vq: &[f32], vk: &[f32], vv: &[f32], g: &[f32],
                     gqb: &mut [f32], gkb: &mut [f32], gvb: &mut [f32],
                     s: usize, seq: usize, nh: usize, nkv: usize, hd: usize, scale: f32) {
    // Same tiling as the forward, for the same reason. Two products here
    // were per-pair dot products with a horizontal reduction each — the
    // scores `Q·Kᵀ` (recomputed rather than stored, see the forward) and
    // `dP = dO·Vᵀ`. Both are now register tiles over a packed Kᵀ and Vᵀ,
    // lane-parallel across keys. The three accumulations — dQ, dK, dV —
    // were already in broadcast form (one scalar against a contiguous row)
    // and stay as they were.
    //
    // Per query i, with P its softmax row and dP = dO_i·Vᵀ:
    //   D_i  = Σ_j P_ij dP_ij
    //   dS_j = P_ij (dP_ij − D_i) · scale
    //   dQ_i += Σ_j dS_j K_j      dK_j += dS_j Q_i      dV_j += P_ij dO_i
    const BR: usize = 4;
    const BC: usize = 16;

    let group = nh / nkv;
    let (qw, kw) = (nh * hd, nkv * hd);
    let base = s * seq;
    let seqp = seq.next_multiple_of(BC);
    let mut kt = vec![0.0f32; hd * seqp];
    let mut vt = vec![0.0f32; hd * seqp];
    let mut sc = vec![0.0f32; BR * seqp];   // scores, then probabilities
    let mut dp = vec![0.0f32; BR * seqp];   // dO·Vᵀ
    let mut ex = vec![0.0f32; seqp];

    for kvh in 0..nkv {
        for j in 0..seq {
            let off = (base + j) * kw + kvh * hd;
            let (krow, vrow) = (&vk[off..off + hd], &vv[off..off + hd]);
            for d in 0..hd { kt[d * seqp + j] = krow[d]; vt[d * seqp + j] = vrow[d]; }
        }
        for qh in kvh * group..(kvh + 1) * group {
            for i0 in (0..seq).step_by(BR) {
                let br = BR.min(seq - i0);
                let jmax = i0 + br - 1;
                let tiles = (jmax / BC) + 1;
                // ── scores and dP, both lane-parallel over keys ──
                for t in 0..tiles {
                    let j0 = t * BC;
                    let mut ts = [[0.0f32; BC]; BR];
                    let mut td = [[0.0f32; BC]; BR];
                    for d in 0..hd {
                        let kr = &kt[d * seqp + j0..d * seqp + j0 + BC];
                        let vr = &vt[d * seqp + j0..d * seqp + j0 + BC];
                        for ii in 0..br {
                            let off = (base + i0 + ii) * qw + qh * hd + d;
                            let (q, go) = (vq[off], g[off]);
                            let (rs, rd) = (&mut ts[ii], &mut td[ii]);
                            for jj in 0..BC { rs[jj] += q * kr[jj]; rd[jj] += go * vr[jj]; }
                        }
                    }
                    for ii in 0..br {
                        let (ds, dd) = (&mut sc[ii * seqp + j0..ii * seqp + j0 + BC],
                                        &mut dp[ii * seqp + j0..ii * seqp + j0 + BC]);
                        for jj in 0..BC { ds[jj] = ts[ii][jj] * scale; dd[jj] = td[ii][jj]; }
                    }
                }
                // ── per query: softmax P, D, and dS (written over dp) ──
                for ii in 0..br {
                    let qi = i0 + ii;
                    let n = qi + 1;                    // causal: keys 0..=qi
                    let prow = &mut sc[ii * seqp..ii * seqp + n];
                    let mut m = f32::NEG_INFINITY;
                    for &x in prow.iter() { if x > m { m = x; } }
                    let sum = r2_tensor::ops::exp_shift_sum(prow, m, &mut ex[..n]);
                    let inv = 1.0 / sum;
                    for j in 0..n { prow[j] = ex[j] * inv; }
                    let dprow = &mut dp[ii * seqp..ii * seqp + n];
                    let mut dot = 0.0f32;
                    for j in 0..n { dot += prow[j] * dprow[j]; }
                    for j in 0..n { dprow[j] = prow[j] * (dprow[j] - dot) * scale; }
                    // keys this query cannot see contribute nothing
                    for j in n..jmax + 1 { sc[ii * seqp + j] = 0.0; dp[ii * seqp + j] = 0.0; }
                }
                // ── accumulate, KEY-OUTER: each dK_j / dV_j row is read and
                // written once per query block, not once per query. The
                // previous order did that read-modify-write of two 256-byte
                // rows for every (query, key) pair — the backward's cost was
                // that traffic, not the arithmetic.
                for j in 0..=jmax {
                    let gvo = j * kw + kvh * hd;
                    let koff = (base + j) * kw + kvh * hd;
                    let krow = &vk[koff..koff + hd];
                    for ii in 0..br {
                        let qi = i0 + ii;
                        if j > qi { continue; }
                        let pj = sc[ii * seqp + j];
                        let d = dp[ii * seqp + j];
                        let qoff = (base + qi) * qw + qh * hd;
                        let qrow = &vq[qoff..qoff + hd];
                        let grow = &g[qoff..qoff + hd];
                        let gqo = qi * qw + qh * hd;
                        for c in 0..hd {
                            gvb[gvo + c] += pj * grow[c];
                            gkb[gvo + c] += d * qrow[c];
                            gqb[gqo + c] += d * krow[c];
                        }
                    }
                }
            }
        }
    }
}

/// AVX2+FMA build of the forward kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
fn attn_forward_seq_avx2(vq: &[f32], vk: &[f32], vv: &[f32], oblk: &mut [f32],
                                s: usize, seq: usize, nh: usize, nkv: usize,
                                hd: usize, scale: f32) {
    attn_forward_seq_impl(vq, vk, vv, oblk, s, seq, nh, nkv, hd, scale)
}

/// AVX2+FMA build of the backward kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
fn attn_backward_seq_avx2(vq: &[f32], vk: &[f32], vv: &[f32], g: &[f32],
                                 gqb: &mut [f32], gkb: &mut [f32], gvb: &mut [f32],
                                 s: usize, seq: usize, nh: usize, nkv: usize,
                                 hd: usize, scale: f32) {
    attn_backward_seq_impl(vq, vk, vv, g, gqb, gkb, gvb, s, seq, nh, nkv, hd, scale)
}

/// Dispatch. The branch is per SEQUENCE, not per element.
#[allow(clippy::too_many_arguments)]
fn attn_forward_seq(vq: &[f32], vk: &[f32], vv: &[f32], oblk: &mut [f32],
                    s: usize, seq: usize, nh: usize, nkv: usize, hd: usize, scale: f32) {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        // SAFETY: guarded by the runtime feature check; the callee's body
        // is the same safe code, compiled with wider instructions.
        unsafe { attn_forward_seq_avx2(vq, vk, vv, oblk, s, seq, nh, nkv, hd, scale) };
        return;
    }
    attn_forward_seq_impl(vq, vk, vv, oblk, s, seq, nh, nkv, hd, scale)
}

#[allow(clippy::too_many_arguments)]
fn attn_backward_seq(vq: &[f32], vk: &[f32], vv: &[f32], g: &[f32],
                     gqb: &mut [f32], gkb: &mut [f32], gvb: &mut [f32],
                     s: usize, seq: usize, nh: usize, nkv: usize, hd: usize, scale: f32) {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        // SAFETY: as above.
        unsafe {
            attn_backward_seq_avx2(vq, vk, vv, g, gqb, gkb, gvb,
                                   s, seq, nh, nkv, hd, scale)
        };
        return;
    }
    attn_backward_seq_impl(vq, vk, vv, g, gqb, gkb, gvb, s, seq, nh, nkv, hd, scale)
}

/// Transpose a row-major `rows × cols` matrix. Used by the GPU backward
/// path to express both gradients as plain matmuls.
fn transpose_of(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols { out[j * rows + i] = x[i * cols + j]; }
    }
    out
}
