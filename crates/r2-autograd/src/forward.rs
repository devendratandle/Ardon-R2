//! The forward ops: each computes its value and records on the tape what
//! its backward (`backward.rs`) will need.

use super::*;
use crate::kernels::{attn_forward_head, head_pieces};

impl Tape {
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
            // One work unit per (sequence, query head). Units were whole
            // sequences until the medium model showed why that is wrong:
            // the batch there is 8 sequences on 6 cores — two rounds, the
            // second with four cores idle — and at batch 1 it is serial.
            // A head's columns are strided through `out`, so the disjoint
            // pieces each unit writes are collected by `head_pieces`
            // (reborrows, no copy). A token still cannot see another
            // example's tokens: the kernel reads only its own sequence's
            // rows.
            let mut units = head_pieces(&mut out, seq, nh, hd);
            let work = |u: usize, orows: &mut Vec<&mut [f32]>| {
                attn_forward_head(vq, vk, vv, orows, u / nh, u % nh, seq, nh, nkv, hd, scale);
            };
            if rows * nh * hd >= PAR_MIN {
                use rayon::prelude::*;
                units.par_iter_mut().enumerate().for_each(|(u, orows)| work(u, orows));
            } else {
                units.iter_mut().enumerate().for_each(|(u, orows)| work(u, orows));
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
}
