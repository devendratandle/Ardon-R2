//! Reverse-mode gradient accumulation. `backward_from` walks the tape from
//! the seeded output back to the leaves; each op's adjoint is its own
//! method below. No gradient buffer is ever blanket-zeroed: the FIRST
//! writer of a node's gradient assigns, later writers accumulate
//! (`first_write`).

use super::*;
use crate::kernels::{attn_backward_group, dot4, head_pieces, transpose_of};

impl Tape {
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
        // No zeroing pass, ever. Each adjoint asks `first_write(node)`
        // before touching a gradient: the first writer assigns the whole
        // buffer (or zeroes it and accumulates, for the adjoints that write
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
            // backward pass. Nothing writes to node i during its own adjoint
            // (a node's inputs always have lower indices in a DAG), so the
            // buffer is safe to borrow away and hand back. The op is moved
            // out the same way, so its adjoint can take `&mut self`.
            let g = std::mem::take(&mut self.grads[i]);
            let op = std::mem::replace(&mut self.ops[i], Op::Leaf);
            self.adjoint(i, &op, &g);
            self.ops[i] = op;
            // Return the buffer so grad() still reports this node.
            self.grads[i] = g;
            if let Some(t) = t_arm {
                BWD_STATS.lock().unwrap().push((self.ops[i].kind(), t.elapsed().as_secs_f32() * 1e3));
            }
        }
        // A node no adjoint wrote (an unused leaf, a branch not reaching the
        // seed) must still report a zero gradient. On a fresh tape its
        // buffer already is zero and is left untouched — no fault. Only a
        // recycled buffer can hold stale values.
        if self.recycled_grads > 0 {
            for i in 0..self.grads.len() {
                if !self.gwritten[i] && self.requires[i] { zero_par(&mut self.grads[i]); }
            }
        }
    }

    /// Accumulate node `i`'s gradient `g` into the gradients of its inputs.
    fn adjoint(&mut self, i: usize, op: &Op, g: &[f32]) {
        match op {
            Op::Leaf => {}
            Op::Add(a, b) => self.bwd_add(a.0, b.0, g),
            Op::Mul(a, b) => self.bwd_mul(a.0, b.0, g),
            Op::Embed { table, tokens, d } => self.bwd_embed(table.0, tokens, *d, g),
            Op::MatMul { a, b, m, k, n } => self.bwd_matmul(a.0, b.0, *m, *k, *n, g),
            Op::Silu(x) => self.bwd_silu(x.0, g),
            Op::Rmsnorm { x, w, d, eps } => self.bwd_rmsnorm(x.0, w.0, *d, *eps, g),
            Op::Transpose { x, rows, cols } => {
                let (xi, rows, cols) = (x.0, *rows, *cols);
                zero_if_first(&mut self.gwritten, &mut self.grads, xi);
                // grad_x[r,c] += g[c,r]
                for r in 0..rows { for c in 0..cols {
                    self.grads[xi][r * cols + c] += g[c * rows + r];
                }}
            }
            Op::Rope { x, rows, period, n_heads, head_dim, base } =>
                self.bwd_rope(x.0, *rows, *period, *n_heads, *head_dim, *base, g),
            Op::SliceRows { x, cols, start, len } => {
                // Rows are contiguous: the adjoint scatters the incoming
                // gradient back into its row range.
                let (xi, cols, start, len) = (x.0, *cols, *start, *len);
                zero_if_first(&mut self.gwritten, &mut self.grads, xi);
                let base = start * cols;
                for j in 0..len * cols { self.grads[xi][base + j] += g[j]; }
            }
            Op::ConcatRows { xs } => {
                // Each input owns a contiguous slab of the output.
                let mut off = 0usize;
                for v in xs {
                    zero_if_first(&mut self.gwritten, &mut self.grads, v.0);
                    let n = self.grads[v.0].len();
                    for j in 0..n { self.grads[v.0][j] += g[off + j]; }
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
                for (c, v) in xs.iter().enumerate() {
                    zero_if_first(&mut self.gwritten, &mut self.grads, v.0);
                    for r in 0..rows { for j in 0..each {
                        self.grads[v.0][r * each + j] += g[r * n * each + c * each + j];
                    }}
                }
            }
            Op::ScaleMaskCausal { x, t, scale } => {
                let (xi, t, scale) = (x.0, *t, *scale);
                zero_if_first(&mut self.gwritten, &mut self.grads, xi);
                // Masked entries are constants (-inf), so they pass no
                // gradient back — a token cannot learn from its future.
                for r in 0..t { for j in 0..=r {
                    self.grads[xi][r * t + j] += g[r * t + j] * scale;
                }}
            }
            Op::SoftmaxRows { x, d } => self.bwd_softmax_rows(i, x.0, *d, g),
            Op::Attention { q, k, v, nseq, seq, nh, nkv, hd, scale } =>
                self.bwd_attention(q.0, k.0, v.0, *nseq, *seq, *nh, *nkv, *hd, *scale, g),
            Op::SumAll(x) => {
                let xi = x.0;
                zero_if_first(&mut self.gwritten, &mut self.grads, xi);
                for gx in self.grads[xi].iter_mut() { *gx += g[0]; }
            }
            Op::Mse { pred, target } => {
                let pi = pred.0;
                zero_if_first(&mut self.gwritten, &mut self.grads, pi);
                let n = target.len() as f32;
                let Tape { vals, grads, .. } = self;
                for (gp, (p, t)) in grads[pi].iter_mut().zip(vals[pi].iter().zip(target)) {
                    *gp += g[0] * 2.0 * (p - t) / n;
                }
            }
            Op::SoftmaxCE { logits, d, targets, lse } => self.bwd_softmax_ce(logits.0, *d, targets, lse, g),
        }
    }

    fn bwd_add(&mut self, a: usize, b: usize, g: &[f32]) {
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
                self.grads[side].copy_from_slice(g);
            } else {
                for (gd, gi) in self.grads[side].iter_mut().zip(g) { *gd += gi; }
            }
        }
    }

    fn bwd_mul(&mut self, a: usize, b: usize, g: &[f32]) {
        // `vals` and `grads` are DISJOINT fields, so borrow them as such
        // instead of cloning both operands — the same trick the MatMul
        // adjoint uses. Those two clones were a full copy of each input on
        // every backward (2 MB each at 2,048 tokens x dim 256).
        //
        // The `requires` guards skip a side whose subtree holds no
        // parameter at all. A constant multiplier — an attention mask, an
        // upstream gradient fed in as data — was getting a full gradient
        // computed into a buffer nothing would ever read.
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
                for ((d, s), o) in grads[dst].iter_mut().zip(g).zip(other) { *d = s * o; }
            } else {
                for ((d, s), o) in grads[dst].iter_mut().zip(g).zip(other) { *d += s * o; }
            }
        }
    }

    /// The adjoint of a gather is a SCATTER-ADD. `+=`, never `=`: a token
    /// appearing twice must accumulate both contributions, and dropping one
    /// is a silent error on exactly the commonest tokens.
    fn bwd_embed(&mut self, ti: usize, tokens: &[usize], d: usize, g: &[f32]) {
        if !self.requires[ti] { return; }
        // A scatter touches only the rows of tokens present, so the first
        // write must zero the table gradient (8 MB at vocab 8,000) before
        // accumulating. Done in parallel: warm pages, six threads.
        if first_write(&mut self.gwritten, ti) { zero_par(&mut self.grads[ti]); }
        let gt = &mut self.grads[ti];
        // Threading a scatter-add needs care: two tokens can hit the SAME
        // row, so splitting the tokens across workers would race. Split the
        // TABLE instead — each worker owns a disjoint block of rows and
        // scans the token list for the ones landing in it. No locks, no
        // atomics, and because each row's contributions are still applied
        // in ascending token order the result is BIT-IDENTICAL to the
        // serial loop; float addition is not associative, so anything less
        // would make the gradient depend on the thread count.
        //
        // The redundant scan is `threads * tokens` integer compares — 12k at
        // 2,048 tokens — against a copy of `tokens * d` floats. Measured at
        // vocab 8,000: 1,293 us serial, 519 us on six threads, versus
        // PyTorch's `embedding_dense_backward` at 336 us threaded and
        // 1,541 us on one thread.
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

    fn bwd_matmul(&mut self, ai: usize, bi: usize, m: usize, k: usize, n: usize, g: &[f32]) {
        // Neither gradient is worth computing into a subtree that holds no
        // parameter. This is not a micro-saving: the one-hot embedding form
        // is `matmul(onehot, table)` with `requires(onehot) == false`, and
        // grad_A there is `g . tableT` — a t x vocab matrix costing
        // 2*t*vocab*d = 8.4 GFLOP at 2,048 tokens and vocab 8,000, roughly
        // HALF that path's 658 ms, written into a buffer nothing reads.
        // Every frozen input — embedded constants, a masked score matrix, a
        // distillation teacher's activations — gets the same relief.
        let (need_a, need_b) = (self.requires[ai], self.requires[bi]);

        // Backward is ~2/3 of training FLOPs. When the Oracle routes this
        // shape to the GPU, express both gradients as matmuls so they use
        // the SAME accelerated kernel as the forward pass — otherwise the
        // GPU only ever sees a third of the work and cannot pay for itself.
        //
        // grad_A = g·Bᵀ and grad_B = Aᵀ·g. The transposes are O(size)
        // against an O(m·k·n) multiply, so they are cheap at exactly the
        // sizes this branch is taken.
        if matches!(
            r2_oracle::dispatch(r2_oracle::Op::TensorMatMul, r2_oracle::Shape::nmk(m, n, k)),
            r2_oracle::Backend::Gpu)
        {
            if need_a {
                let bt = transpose_of(&self.vals[bi], k, n);   // n×k
                let ga = r2_tensor::ops::matmul(g, &bt, m, n, k);
                if first_write(&mut self.gwritten, ai) { self.grads[ai].copy_from_slice(&ga); }
                else { for (dst, v) in self.grads[ai].iter_mut().zip(&ga) { *dst += v; } }
            }
            if need_b {
                let at = transpose_of(&self.vals[ai], m, k);   // k×m
                let gb = r2_tensor::ops::matmul(&at, g, k, m, n);
                if first_write(&mut self.gwritten, bi) { self.grads[bi].copy_from_slice(&gb); }
                else { for (dst, v) in self.grads[bi].iter_mut().zip(&gb) { *dst += v; } }
            }
            return;
        }
        // grad_A(m×k) = g(m×n) · Bᵀ(n×k) and grad_B(k×n) = Aᵀ(k×m) · g(m×n).
        //
        // These are the NT and TN cases of one GEMM, and they are CALLS to
        // it rather than two hand-written loop nests. What was here before
        // was ~200 lines of blocking-free triple loops, and it showed:
        // measured against the best of PyTorch and JAX on the shapes this
        // model runs, grad_A was 12.6-25.4x behind and grad_B 3.0-14.6x. On
        // the output-head shape alone, grad_A took 676 ms and grad_B 465 ms,
        // against 77 ms and 110 ms for the same arithmetic through `gemm`.
        //
        // Neither transpose is materialised. `gemm`'s packing pass already
        // moves every element, so it reads the operand transposed for free —
        // which is why `REPORT.md` records materialising them as a REJECTED
        // attempt at 393 ms against 326.
        use r2_linalg::gemm::{sgemm_assign_into, sgemm_into, Trans};
        let par = m * k * n >= PAR_MIN;
        let Tape { vals, grads, gwritten, .. } = self;
        if need_a {
            // (M, K, N) = (m, n, k); B is stored k×n, which IS the N×K the
            // transposed read wants.
            let f = if first_write(gwritten, ai) { sgemm_assign_into } else { sgemm_into };
            f(g, Trans::No, &vals[bi], Trans::Yes, m, n, k, &mut grads[ai], par);
        }
        if need_b {
            // (M, K, N) = (k, m, n); A is stored m×k, which IS the K×M the
            // transposed read wants.
            let f = if first_write(gwritten, bi) { sgemm_assign_into } else { sgemm_into };
            f(&vals[ai], Trans::Yes, g, Trans::No, k, m, n, &mut grads[bi], par);
        }
    }

    fn bwd_silu(&mut self, xi: usize, g: &[f32]) {
        if !self.requires[xi] { return; }
        // `vals` and `grads` are DISJOINT fields, so borrow them as such.
        // This used to `clone()` the whole input first — 6.3 MB per call at
        // the shipping shape, four times a step, to read it once.
        let Tape { vals, grads, gwritten, .. } = self;
        let acc = !first_write(gwritten, xi);
        let vx = &vals[xi];
        // d/dv [v*s] = s + v*s*(1-s), with the sigmoid's `exp` vectorised —
        // same reason as the forward.
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
            work(&mut grads[xi], g, vx);
        }
    }

    fn bwd_rmsnorm(&mut self, xi: usize, wi: usize, d: usize, eps: f32, g: &[f32]) {
        let assign_x = first_write(&mut self.gwritten, xi);
        if first_write(&mut self.gwritten, wi) { self.grads[wi].fill(0.0); }
        // `vals` and `grads` are disjoint fields: borrow, do not clone (this
        // used to copy the 6 MB input per call).
        let Tape { vals, grads, .. } = self;
        let (vx, vw) = (&vals[xi], &vals[wi]);
        let rows = vx.len() / d;
        // Rows are independent, so dL/dx is row-parallel and bit-identical
        // to the serial loop; each row's 1/rms is kept so the weight
        // gradient below — a sum over rows, whose order must not change —
        // reads it once.
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

    /// Rotation is orthogonal: the adjoint is the inverse rotation, i.e.
    /// the same op at angle -theta.
    #[allow(clippy::too_many_arguments)]
    fn bwd_rope(&mut self, xi: usize, rows: usize, period: usize, nh: usize, hd: usize,
                base: f32, g: &[f32]) {
        // Same table as the forward, and for the same reason: the angle is
        // a function of (position, pair) alone.
        let (half, per) = (hd / 2, period.max(1));
        let tab = r2_tensor::ops::rope_table(per, hd, base);
        let w = nh * hd;
        let assign = first_write(&mut self.gwritten, xi);
        let row = |r: usize, grow: &mut [f32]| {
            // Same position mapping as the forward pass: in a fused batch
            // the angle restarts each sequence.
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

    /// `i` is the softmax node itself: its value is the softmax.
    fn bwd_softmax_rows(&mut self, i: usize, xi: usize, d: usize, g: &[f32]) {
        zero_if_first(&mut self.gwritten, &mut self.grads, xi);
        let Tape { vals, grads, .. } = self;
        let y = &vals[i];
        let rows = y.len() / d;
        for r in 0..rows {
            let yr = &y[r * d..r * d + d];
            let gr = &g[r * d..r * d + d];
            // dot = Σ_j g_j y_j ; dL/dx_i = y_i (g_i − dot)
            let dot = dot4(gr, yr);
            for j in 0..d {
                grads[xi][r * d + j] += yr[j] * (gr[j] - dot);
            }
        }
    }

    /// The probabilities are RECOMPUTED rather than stored. Storing them
    /// costs nseq*nh*seq*seq floats — 2 MB at the shipping shape and
    /// quadratic in the context — for one saved QK pass. Recompute is what
    /// flash-attention does and for the same reason: the memory is worth
    /// more than the flops.
    #[allow(clippy::too_many_arguments)]
    fn bwd_attention(&mut self, qi: usize, ki: usize, vi: usize, nseq: usize, seq: usize,
                     nh: usize, nkv: usize, hd: usize, scale: f32, g: &[f32]) {
        let rows = nseq * seq;
        let (need_q, need_k, need_v) = (self.requires[qi], self.requires[ki], self.requires[vi]);
        if !(need_q || need_k || need_v) { return; }
        // Accumulate into local buffers, then add into the tape's gradients.
        // Three entries of `self.grads` cannot be borrowed mutably at once,
        // and the adds are O(size) against an O(nseq*nh*seq^2*hd) backward.
        // Scratch from the pool (warm pages). Not zeroed: every element is
        // assigned by exactly one unit.
        let mut gq = self.pool.take(rows * nh * hd);
        let mut gk = self.pool.take(rows * nkv * hd);
        let mut gv = self.pool.take(rows * nkv * hd);
        {
            let (vq, vk, vv) = (&self.vals[qi], &self.vals[ki], &self.vals[vi]);
            // One work unit per (sequence, kv head): the unit runs every
            // query head of the GQA group, so the kv head's dK/dV are
            // complete inside it and no two units share a row. Same reason
            // as the forward: units were sequences, and the batch is not
            // always larger than the core count. `head_pieces` hands each
            // unit its strided pieces of the three buffers.
            let group = nh / nkv;
            let mut uq = head_pieces(&mut gq, seq, nkv, group * hd);
            let mut uk = head_pieces(&mut gk, seq, nkv, hd);
            let mut uv = head_pieces(&mut gv, seq, nkv, hd);
            let work = |u: usize, dq: &mut Vec<&mut [f32]>, dk: &mut Vec<&mut [f32]>, dv: &mut Vec<&mut [f32]>| {
                attn_backward_group(vq, vk, vv, g, dq, dk, dv,
                                    u / nkv, u % nkv, seq, nh, nkv, hd, scale);
            };
            if rows * nh * hd >= PAR_MIN {
                use rayon::prelude::*;
                uq.par_iter_mut().zip(uk.par_iter_mut()).zip(uv.par_iter_mut())
                    .enumerate()
                    .for_each(|(u, ((dq, dk), dv))| work(u, dq, dk, dv));
            } else {
                for (u, ((dq, dk), dv)) in uq.iter_mut().zip(uk.iter_mut()).zip(uv.iter_mut()).enumerate() {
                    work(u, dq, dk, dv);
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

    /// grad = (softmax − onehot) / batch, scaled by upstream g.
    ///
    /// `softmax(x)[j]` is `exp(x[j] - lse)`, and `lse` was computed in the
    /// forward — so this needs no probability matrix and no second
    /// reduction pass. It used to call `ops::softmax` again here, allocating
    /// and filling a second 64 MB buffer at vocab 8,000.
    ///
    /// Rows are independent and write disjoint slices of the gradient, so
    /// they split across cores with no coordination.
    fn bwd_softmax_ce(&mut self, li: usize, d: usize, targets: &[usize], lse: &[f32], g: &[f32]) {
        let inv = g[0] / targets.len() as f32;
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

/// `true` exactly once per node per backward: the caller is the FIRST
/// writer of that node's gradient and must ASSIGN (or zero, then add).
#[inline]
fn first_write(gwritten: &mut [bool], node: usize) -> bool {
    let first = !gwritten[node];
    gwritten[node] = true;
    first
}

/// For adjoints that write only PART of a gradient (a slice, a masked
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
