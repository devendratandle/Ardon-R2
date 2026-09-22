//! The training step on the GPU: weights, activations, gradients and
//! optimizer state resident on the device for the whole run — tokens go
//! in, a loss comes out, nothing else crosses the bus.
//!
//! This is the same model `Trainer::forward_fused` builds on the tape,
//! written as a static graph: the forward is the sequence of kernels
//! below, the backward is the same sequence reversed with every gradient
//! written by exactly one kernel launch (assign) or accumulated by the
//! second writer, following the tape's first-writer rule. There is no
//! tape, no allocator in the loop and no per-op dispatch: one launch per
//! op, ~25 forward and ~35 backward per layer, and one Adam launch per
//! parameter block. The layout of `params` is `Trainer`'s, so a
//! `GpuTrainer` is built from a `Trainer` and hands its weights and
//! optimizer state back with `download_into`, which is how the same
//! held-out evaluation and checkpoint code serves both.
//!
//! Numerics: every kernel is the CPU kernel's arithmetic (checked in
//! `r2-gpu`'s tests to f32 rounding); the device reproduces its own
//! results bit for bit.
//!
//! # Mixed precision
//!
//! `precision = Mixed` is the standard recipe: f32 master weights and
//! Adam state; an f16 copy of every GEMM weight cast once per step (the
//! norm weights stay f32); f16 activations and activation gradients; the
//! residual stream, the logits, the row statistics and every WEIGHT
//! gradient f32; every kernel accumulates in f32 whatever it reads. The
//! loss gradient is multiplied by a scale so f16 activation gradients
//! do not underflow, Adam divides it back out, and the scale is dynamic:
//! an overflow skips the step and halves it, a thousand clean steps
//! double it (to 65,536). Overflow is tested on every f16 GRADIENT
//! buffer as it is produced — inf, nan, or saturated at 65,504, because
//! an adapter may clamp an overflowing f16 store instead of producing
//! inf (this one does) — and on the f32 weight gradients for inf/nan.
//! The flag rides in the loss-rows buffer, so a step still has one
//! readback.
//!
//! Every kernel still sums in a fixed order, so mixed precision is as
//! reproducible as f32: the same run gives the same bits.

use crate::llm::Trainer;
use r2_gpu::attention::{self, Shape};
use r2_gpu::device::{Dtype, Tensor};
use r2_gpu::elementwise as ew;
use r2_gpu::gemm::gemm;
use r2_gpu::optim::adam_step;
use r2_tensor::model::Config;

/// A layer's activations past its input: what the backward needs, or
/// recomputes from `x` under checkpointing.
struct LayerActs {
    h: Tensor,      // rmsnorm(x)               t x d
    q: Tensor,      // h·Wq                     t x d
    k: Tensor,      // h·Wk                     t x kv
    v: Tensor,      // h·Wv                     t x kv
    qr: Tensor,     // rope(q)
    kr: Tensor,     // rope(k)
    ctx: Tensor,    // attention(qr, kr, v)     t x d
    lse: Tensor,    // per (row, head)          t x nh
    x1: Tensor,     // x + ctx·Wo               t x d
    h2: Tensor,     // rmsnorm(x1)
    gate: Tensor,   // h2·W1                    t x ffn
    up: Tensor,     // h2·W3                    t x ffn
    sg: Tensor,     // silu(gate)
    act: Tensor,    // sg * up
}

/// Activations and gradient scratch for one token count.
struct Acts {
    t: usize,
    seq: usize,
    x: Vec<Tensor>,         // each layer's input residual stream, t x d
    /// One work set per layer, or ONE for all of them under
    /// checkpointing, refilled from `x[l]` before layer l's backward.
    work: Vec<LayerActs>,
    x_out: Tensor,      // residual stream after the last layer
    xn: Tensor,         // final rmsnorm
    logits: Tensor,     // t x vocab
    lse_ce: Tensor,     // t
    loss_rows: Tensor,  // t
    // gradient scratch
    gres: Tensor,       // the residual stream's gradient, carried down the layers
    gh: Tensor,
    gq: Tensor, gk: Tensor, gv: Tensor, gqr: Tensor, gkr: Tensor, gctx: Tensor,
    gact: Tensor, gup: Tensor, gsg: Tensor,
    glogits: Tensor,
    delta: Tensor,      // t x nh, attention backward scratch
    rinv: Tensor,       // rmsnorm backward scratch (rinv + dW partials)
}

/// The storage precision of a step.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Precision { F32, Mixed }

/// Loss scale bounds and growth interval of the dynamic scaling.
const SCALE_INIT: f32 = 1024.0;
const SCALE_MAX: f32 = 65536.0;
const SCALE_GROW_EVERY: u32 = 1000;

pub struct GpuTrainer {
    pub cfg: Config,
    w: Vec<Tensor>,
    /// f16 copies of the GEMM weights under `Mixed` (`None` for the norm
    /// weights, and for every block under `F32`); refreshed each step.
    w16: Vec<Option<Tensor>>,
    g: Vec<Tensor>,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    sizes: Vec<usize>,
    lr: f32, beta1: f32, beta2: f32, eps: f32,
    t: u64,
    pub step: u64,
    /// Activation checkpointing: keep only each layer's input, recompute
    /// the rest in the backward. One layer's work set instead of one per
    /// layer — at dim d and t tokens, 14·t·d + t·nh floats per layer
    /// become t·d — for a quarter more forward arithmetic.
    pub checkpoint: bool,
    pub precision: Precision,
    /// The loss scale (1 under `F32`), the clean steps since it last
    /// changed, and how many steps overflowed and were skipped.
    pub loss_scale: f32,
    clean_steps: u32,
    pub overflows: u64,
    acts: Option<Acts>,
}

fn tz(n: usize) -> Result<Tensor, String> { Tensor::zeros(n).ok_or_else(|| "no GPU device".to_string()) }

impl GpuTrainer {
    /// Upload a CPU trainer's weights and optimizer state. The CPU
    /// trainer keeps its buffers; `download_into` brings the device's
    /// state back.
    pub fn from_trainer(tr: &Trainer) -> Result<GpuTrainer, String> {
        let up = |d: &[f32]| Tensor::upload(d).ok_or_else(|| "no GPU device".to_string());
        let mut w = Vec::new(); let mut g = Vec::new(); let mut m = Vec::new(); let mut v = Vec::new();
        let mut sizes = Vec::new();
        let mut off = 0;
        for p in &tr.params {
            let n = p.len();
            w.push(up(p)?);
            g.push(tz(n)?);
            m.push(up(&tr.opt.m[off..off + n])?);
            v.push(up(&tr.opt.v[off..off + n])?);
            sizes.push(n);
            off += n;
        }
        let w16 = (0..w.len()).map(|_| None).collect();
        Ok(GpuTrainer { cfg: tr.cfg, w, w16, g, m, v, sizes, lr: tr.opt.lr, beta1: tr.opt.beta1,
                        beta2: tr.opt.beta2, eps: tr.opt.eps, t: tr.opt.t, step: tr.step,
                        checkpoint: false, precision: Precision::F32, loss_scale: 1.0,
                        clean_steps: 0, overflows: 0, acts: None })
    }

    /// Switch the storage precision. `Mixed` needs an adapter with
    /// `shader-f16`; the f16 weight copies are allocated here and cast
    /// each step.
    pub fn set_precision(&mut self, p: Precision) -> Result<(), String> {
        if p == self.precision { return Ok(()); }
        self.precision = p;
        self.acts = None;                              // activations change type
        if p == Precision::Mixed {
            let l = self.cfg.n_layers;
            let is_gemm = |i: usize| i == 0 || i == 2 || (i >= 3 && !matches!((i - 3) % 9, 0 | 5));
            self.w16 = (0..self.w.len()).map(|i| {
                if is_gemm(i) { Tensor::zeros_as(self.sizes[i], Dtype::F16).map(Some) } else { Some(None) }
            }).collect::<Option<Vec<_>>>().ok_or("mixed precision needs an adapter with shader-f16")?;
            debug_assert_eq!(self.w16.len(), 3 + 9 * l);
            self.loss_scale = SCALE_INIT;
        } else {
            self.w16 = (0..self.w.len()).map(|_| None).collect();
            self.loss_scale = 1.0;
        }
        self.clean_steps = 0;
        Ok(())
    }

    /// Weights and optimizer state back to the CPU trainer.
    pub fn download_into(&self, tr: &mut Trainer) {
        let mut off = 0;
        for (i, n) in self.sizes.iter().enumerate() {
            tr.params[i] = self.w[i].download();
            tr.opt.m[off..off + n].copy_from_slice(&self.m[i].download());
            tr.opt.v[off..off + n].copy_from_slice(&self.v[i].download());
            off += n;
        }
        tr.opt.t = self.t;
        tr.step = self.step;
    }

    fn acts(&mut self, t: usize, seq: usize) -> Result<&Acts, String> {
        let sets = if self.checkpoint { 1 } else { self.cfg.n_layers };
        if self.acts.as_ref().map(|a| a.t == t && a.seq == seq && a.work.len() == sets).unwrap_or(false) {
            return Ok(self.acts.as_ref().unwrap());
        }
        let c = &self.cfg;
        let (d, kv, ffn, nh) = (c.dim, c.kv_dim(), c.ffn_hidden, c.n_heads);
        // activations and their gradients in the step's storage type; the
        // residual stream, logits, row statistics and scratch stay f32
        let adt = if self.precision == Precision::Mixed { Dtype::F16 } else { Dtype::F32 };
        let ta = |n: usize| Tensor::zeros_as(n, adt).ok_or_else(|| "no GPU device (or no f16)".to_string());
        let mut x = Vec::with_capacity(c.n_layers);
        for _ in 0..c.n_layers { x.push(tz(t * d)?); }
        let mut work = Vec::with_capacity(sets);
        for _ in 0..sets {
            work.push(LayerActs {
                h: ta(t * d)?, q: ta(t * d)?, k: ta(t * kv)?, v: ta(t * kv)?,
                qr: ta(t * d)?, kr: ta(t * kv)?, ctx: ta(t * d)?, lse: tz(t * nh)?, x1: tz(t * d)?,
                h2: ta(t * d)?, gate: ta(t * ffn)?, up: ta(t * ffn)?, sg: ta(t * ffn)?, act: ta(t * ffn)?,
            });
        }
        self.acts = Some(Acts {
            t, seq, x, work,
            // loss_rows has one slot past the t losses: the overflow flag
            x_out: tz(t * d)?, xn: ta(t * d)?, logits: tz(t * c.vocab)?, lse_ce: tz(t)?, loss_rows: tz(t + 1)?,
            gres: tz(t * d)?, gh: ta(t * d)?, gq: ta(t * d)?, gk: ta(t * kv)?, gv: ta(t * kv)?,
            gqr: ta(t * d)?, gkr: ta(t * kv)?, gctx: ta(t * d)?,
            gact: ta(t * ffn)?, gup: ta(t * ffn)?, gsg: ta(t * ffn)?,
            glogits: ta(t * c.vocab)?, delta: tz(t * nh)?, rinv: tz(ew::rmsnorm_scratch_len(t, d))?,
        });
        Ok(self.acts.as_ref().unwrap())
    }

    /// One optimizer step over a batch of equal-length sequences. Returns
    /// the mean cross-entropy over every token, as `Trainer::train_step`.
    pub fn train_step(&mut self, batch: &[(Vec<usize>, Vec<usize>)]) -> Result<f32, String> {
        let (loss, overflow) = self.forward_backward(batch)?;
        if overflow {
            // dynamic loss scaling: the gradients are unusable, the step
            // is skipped and the scale halved
            self.overflows += 1;
            self.loss_scale = (self.loss_scale / 2.0).max(1.0);
            self.clean_steps = 0;
            self.step += 1;
            return Ok(loss);
        }
        // ── Adam, one launch per block; the loss scale divided back out ──
        self.t += 1;
        for i in 0..self.w.len() {
            if !adam_step(&self.w[i], &self.g[i], &self.m[i], &self.v[i], self.sizes[i],
                          self.lr, self.beta1, self.beta2, self.eps, self.t as u32, self.loss_scale) {
                return Err("GPU kernel failed: adam".into());
            }
        }
        if self.precision == Precision::Mixed {
            self.clean_steps += 1;
            if self.clean_steps >= SCALE_GROW_EVERY && self.loss_scale < SCALE_MAX {
                self.loss_scale *= 2.0;
                self.clean_steps = 0;
            }
        }
        self.step += 1;
        Ok(loss)
    }

    /// The loss and the raw gradients of every block, without the
    /// optimizer step — `Trainer::loss_logits_grads`'s counterpart, which
    /// is how a mismatch is localised to the kernel that produced it.
    pub fn loss_and_grads(&mut self, batch: &[(Vec<usize>, Vec<usize>)]) -> Result<(f32, Vec<Vec<f32>>), String> {
        let (loss, _) = self.forward_backward(batch)?;
        let inv = 1.0 / self.loss_scale;               // unscaled, as the tape's
        Ok((loss, self.g.iter().map(|g| g.download().iter().map(|x| x * inv).collect()).collect()))
    }

    /// Forward and backward: the gradients are left in `self.g` (times
    /// the loss scale), every block written by exactly one kernel
    /// (assign) or accumulated by the second writer. Returns the mean
    /// loss and whether any weight gradient overflowed — the only
    /// read-back.
    fn forward_backward(&mut self, batch: &[(Vec<usize>, Vec<usize>)]) -> Result<(f32, bool), String> {
        if batch.is_empty() { return Err("train_step: empty batch".into()); }
        let seq = batch[0].0.len();
        if !batch.iter().all(|(i, t)| i.len() == seq && t.len() == seq) {
            return Err("GpuTrainer::train_step: sequences must have equal length".into());
        }
        let nseq = batch.len();
        let t = nseq * seq;
        let mut toks = Vec::with_capacity(t);
        let mut tgts = Vec::with_capacity(t);
        for (i, tg) in batch { toks.extend_from_slice(i); tgts.extend_from_slice(tg); }
        let c = self.cfg;
        let (d, kv, ffn, nh, nkv, hd, vocab) = (c.dim, c.kv_dim(), c.ffn_hidden, c.n_heads, c.n_kv_heads, c.head_dim(), c.vocab);
        let scale = 1.0 / (hd as f32).sqrt();
        let shape = Shape { nseq, seq, nh, nkv, hd, scale };
        let ids = ew::upload_ids(&toks).ok_or("no GPU device")?;
        let tg_ids = ew::upload_ids(&tgts).ok_or("no GPU device")?;
        let index = ew::TokenIndex::build(&toks, vocab).ok_or("no GPU device")?;
        self.acts(t, seq)?;
        let a = self.acts.as_ref().unwrap();
        let mixed = self.precision == Precision::Mixed;
        // `w[i]` is what the norms read (f32); `wg(i)` is what the GEMMs and
        // the embedding read — the f16 copy under Mixed
        let w = &self.w;
        let wg = |i: usize| -> &Tensor { self.w16[i].as_ref().unwrap_or(&self.w[i]) };
        // the overflow flag slot, cleared before the step writes it
        a.loss_rows.write_at(t, &[0.0]);
        // `R2_GPU_STATS=<step>` drains the queue after every launch of that
        // step and prints where the time went, label by label. The drain
        // serialises what normally overlaps, so the total overstates the
        // step; the DISTRIBUTION is the information.
        let census = std::env::var("R2_GPU_STATS").ok().and_then(|v| v.parse::<u64>().ok())
            .map(|s| s == self.step + 1).unwrap_or(false);
        let table: std::cell::RefCell<Vec<(&'static str, u32, f64)>> = Default::default();
        let mark = std::cell::Cell::new(std::time::Instant::now());
        let flag = |x: &Tensor, n: usize| -> bool { !mixed || ew::nonfinite_flag(x, &a.loss_rows, t, n) };
        let ok = |b: bool, what: &'static str| -> Result<(), String> {
            if !b { return Err(format!("GPU kernel failed: {what}")); }
            if census {
                r2_gpu::device::sync();
                let dt = mark.get().elapsed().as_secs_f64() * 1e3;
                let mut tb = table.borrow_mut();
                match tb.iter_mut().find(|r| r.0 == what) {
                    Some(r) => { r.1 += 1; r.2 += dt; }
                    None => tb.push((what, 1, dt)),
                }
                mark.set(std::time::Instant::now());
            }
            Ok(())
        };
        let blk = |l: usize, p: usize| 3 + l * 9 + p;

        // which work set layer l's activations live in
        let ck = self.checkpoint;
        let ws = |l: usize| -> &LayerActs { &a.work[if ck { 0 } else { l }] };

        // one layer's forward from its input `x`, into its work set; the
        // output residual stream (x of the next layer) only when `out`
        // is given — the recompute in the backward already has it
        let layer_fwd = |l: usize, out: Option<&Tensor>| -> Result<(), String> {
            let (x, la) = (&a.x[l], ws(l));
            ok(ew::rmsnorm_fwd(x, &w[blk(l, 0)], &la.h, t, d, c.eps), "rmsnorm")?;
            ok(gemm(&la.h, false, wg(blk(l, 1)), false, t, d, d, &la.q, false), "wq")?;
            ok(gemm(&la.h, false, wg(blk(l, 2)), false, t, d, kv, &la.k, false), "wk")?;
            ok(gemm(&la.h, false, wg(blk(l, 3)), false, t, d, kv, &la.v, false), "wv")?;
            ok(ew::rope_fwd(&la.q, &la.qr, t, seq, nh, hd, c.rope_base), "rope q")?;
            ok(ew::rope_fwd(&la.k, &la.kr, t, seq, nkv, hd, c.rope_base), "rope k")?;
            ok(attention::forward(&la.qr, &la.kr, &la.v, &la.ctx, &la.lse, &shape), "attention")?;
            // x1 = ctx·Wo, then += x (the residual, in place)
            ok(gemm(&la.ctx, false, wg(blk(l, 4)), false, t, d, d, &la.x1, false), "wo")?;
            ok(ew::flat_fwd(ew::Flat::AddInto, x, x, &la.x1, t * d), "residual")?;
            ok(ew::rmsnorm_fwd(&la.x1, &w[blk(l, 5)], &la.h2, t, d, c.eps), "ffn rmsnorm")?;
            ok(gemm(&la.h2, false, wg(blk(l, 6)), false, t, d, ffn, &la.gate, false), "w1")?;
            ok(gemm(&la.h2, false, wg(blk(l, 8)), false, t, d, ffn, &la.up, false), "w3")?;
            ok(ew::flat_fwd(ew::Flat::Silu, &la.gate, &la.gate, &la.sg, t * ffn), "silu")?;
            ok(ew::flat_fwd(ew::Flat::Mul, &la.sg, &la.up, &la.act, t * ffn), "gate*up")?;
            if let Some(xnext) = out {
                // next residual stream = act·W2 + x1
                ok(gemm(&la.act, false, wg(blk(l, 7)), false, t, ffn, d, xnext, false), "w2")?;
                ok(ew::flat_fwd(ew::Flat::AddInto, &la.x1, &la.x1, xnext, t * d), "residual")?;
            }
            Ok(())
        };

        // ── the f16 weight copies for this step ──
        if mixed {
            for i in 0..w.len() {
                if let Some(c16) = &self.w16[i] {
                    ok(ew::flat_fwd(ew::Flat::Copy, &w[i], &w[i], c16, self.sizes[i]), "cast f16")?;
                }
            }
        }

        // ── forward ──
        ok(ew::embed_fwd(wg(0), &ids, &a.x[0], t, d), "embed")?;
        for l in 0..c.n_layers {
            let xnext = if l + 1 < c.n_layers { &a.x[l + 1] } else { &a.x_out };
            layer_fwd(l, Some(xnext))?;
        }
        ok(ew::rmsnorm_fwd(&a.x_out, &w[1], &a.xn, t, d, c.eps), "final rmsnorm")?;
        ok(gemm(&a.xn, false, wg(2), false, t, d, vocab, &a.logits, false), "head")?;
        ok(ew::softmax_ce_fwd(&a.logits, &tg_ids, &a.lse_ce, &a.loss_rows, t, vocab), "softmax_ce")?;

        // ── backward: the same graph reversed; the residual stream's
        //    gradient lives in `gres` and is accumulated down the layers ──
        let g = &self.g;
        let inv = self.loss_scale / t as f32;         // the mean over tokens, times the loss scale
        ok(ew::softmax_ce_bwd(&a.logits, &tg_ids, &a.lse_ce, &a.glogits, t, vocab, inv, true), "softmax_ce bwd")?;
        ok(flag(&a.glogits, t * vocab), "overflow check")?;
        ok(gemm(&a.xn, true, &a.glogits, false, d, t, vocab, &g[2], false), "head grad_B")?;
        ok(gemm(&a.glogits, false, wg(2), true, t, vocab, d, &a.gh, false), "head grad_A")?;
        ok(flag(&a.gh, t * d), "overflow check")?;
        ok(ew::rmsnorm_bwd(&a.x_out, &w[1], &a.gh, &a.gres, &g[1], &a.rinv, t, d, c.eps, true, true), "final rmsnorm bwd")?;
        for l in (0..c.n_layers).rev() {
            // under checkpointing the layer's activations are rebuilt
            // from its input first (the shared work set holds the layer
            // above's until now)
            if ck { layer_fwd(l, None)?; }
            let (x, la) = (&a.x[l], ws(l));
            // x2 = x1 + act·W2
            ok(gemm(&la.act, true, &a.gres, false, ffn, t, d, &g[blk(l, 7)], false), "w2 grad_B")?;
            ok(gemm(&a.gres, false, wg(blk(l, 7)), true, t, d, ffn, &a.gact, false), "w2 grad_A")?;
            ok(flag(&a.gact, t * ffn), "overflow check")?;
            // act = sg * up
            ok(ew::flat_bwd(ew::Flat::Mul, &la.sg, &la.up, &a.gact, Some((&a.gsg, true)), Some((&a.gup, true)), t * ffn), "gate*up bwd")?;
            ok(flag(&a.gsg, t * ffn) && flag(&a.gup, t * ffn), "overflow check")?;
            // sg = silu(gate): d gate written over gact (free now)
            ok(ew::flat_bwd(ew::Flat::Silu, &la.gate, &la.gate, &a.gsg, Some((&a.gact, true)), None, t * ffn), "silu bwd")?;
            ok(flag(&a.gact, t * ffn), "overflow check")?;
            // gate = h2·W1, up = h2·W3
            ok(gemm(&la.h2, true, &a.gact, false, d, t, ffn, &g[blk(l, 6)], false), "w1 grad_B")?;
            ok(gemm(&la.h2, true, &a.gup, false, d, t, ffn, &g[blk(l, 8)], false), "w3 grad_B")?;
            ok(gemm(&a.gact, false, wg(blk(l, 6)), true, t, ffn, d, &a.gh, false), "w1 grad_A")?;
            ok(gemm(&a.gup, false, wg(blk(l, 8)), true, t, ffn, d, &a.gh, true), "w3 grad_A")?;
            ok(flag(&a.gh, t * d), "overflow check")?;
            // h2 = rmsnorm(x1): dx1 accumulates onto gres (x1 also feeds x2)
            ok(ew::rmsnorm_bwd(&la.x1, &w[blk(l, 5)], &a.gh, &a.gres, &g[blk(l, 5)], &a.rinv, t, d, c.eps, false, true), "ffn rmsnorm bwd")?;
            // x1 = x + ctx·Wo
            ok(gemm(&la.ctx, true, &a.gres, false, d, t, d, &g[blk(l, 4)], false), "wo grad_B")?;
            ok(gemm(&a.gres, false, wg(blk(l, 4)), true, t, d, d, &a.gctx, false), "wo grad_A")?;
            ok(flag(&a.gctx, t * d), "overflow check")?;
            ok(attention::backward(&la.qr, &la.kr, &la.v, &la.ctx, &la.lse, &a.gctx, &a.delta, &a.gqr, &a.gkr, &a.gv, &shape), "attention bwd")?;
            ok(ew::rope_bwd(&a.gqr, &a.gq, t, seq, nh, hd, c.rope_base, true), "rope q bwd")?;
            ok(ew::rope_bwd(&a.gkr, &a.gk, t, seq, nkv, hd, c.rope_base, true), "rope k bwd")?;
            ok(flag(&a.gq, t * d) && flag(&a.gk, t * kv) && flag(&a.gv, t * kv), "overflow check")?;
            // q = h·Wq, k = h·Wk, v = h·Wv
            ok(gemm(&la.h, true, &a.gq, false, d, t, d, &g[blk(l, 1)], false), "wq grad_B")?;
            ok(gemm(&la.h, true, &a.gk, false, d, t, kv, &g[blk(l, 2)], false), "wk grad_B")?;
            ok(gemm(&la.h, true, &a.gv, false, d, t, kv, &g[blk(l, 3)], false), "wv grad_B")?;
            ok(gemm(&a.gq, false, wg(blk(l, 1)), true, t, d, d, &a.gh, false), "wq grad_A")?;
            ok(gemm(&a.gk, false, wg(blk(l, 2)), true, t, kv, d, &a.gh, true), "wk grad_A")?;
            ok(gemm(&a.gv, false, wg(blk(l, 3)), true, t, kv, d, &a.gh, true), "wv grad_A")?;
            ok(flag(&a.gh, t * d), "overflow check")?;
            // h = rmsnorm(x): dx accumulates onto gres (x also feeds x1)
            ok(ew::rmsnorm_bwd(x, &w[blk(l, 0)], &a.gh, &a.gres, &g[blk(l, 0)], &a.rinv, t, d, c.eps, false, true), "rmsnorm bwd")?;
        }
        ok(ew::embed_bwd(&a.gres, &index, &g[0], vocab, d, true), "embed bwd")?;
        if mixed {
            for i in 0..g.len() {
                ok(ew::nonfinite_flag(&g[i], &a.loss_rows, t, self.sizes[i]), "overflow check")?;
            }
        }

        // t per-token losses and the overflow flag come back; nothing else does
        let rows = a.loss_rows.download();
        if census {
            let mut tb = table.into_inner();
            tb.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap());
            let total: f64 = tb.iter().map(|r| r.2).sum();
            eprintln!("GPU step census (step {}, queue drained after every launch; {t} tokens)", self.step + 1);
            eprintln!("  {:<20} {:>6} {:>10} {:>7}", "kernel", "calls", "ms", "%");
            for (what, n, ms) in &tb {
                eprintln!("  {what:<20} {n:>6} {ms:>10.2} {:>6.1}%", ms / total * 100.0);
            }
            eprintln!("  {:<20} {:>6} {total:>10.2}", "total", tb.iter().map(|r| r.1).sum::<u32>());
        }
        Ok((rows[..t].iter().sum::<f32>() / t as f32, rows[t] != 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use r2_gpu::device::gpu;

    fn cfg() -> Config {
        Config { dim: 32, n_heads: 4, n_kv_heads: 2, n_layers: 2, vocab: 40,
                 ffn_hidden: 64, max_seq: 16, rope_base: 10000.0, eps: 1e-5 }
    }

    fn batch(nseq: usize, seq: usize, vocab: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 33) as usize % vocab };
        (0..nseq).map(|_| {
            let toks: Vec<usize> = (0..seq + 1).map(|_| next()).collect();
            (toks[..seq].to_vec(), toks[1..].to_vec())
        }).collect()
    }

    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs() / scale))
    }

    /// The device's loss and every block's gradient against the tape's,
    /// from the same weights and tokens: f32 rounding apart, block by
    /// block, so a wrong kernel names itself.
    #[test]
    fn gpu_loss_and_gradients_match_the_tape_block_by_block() {
        if gpu().is_none() { eprintln!("no GPU adapter; skipped"); return; }
        let c = cfg();
        let tr = Trainer::new(c, 1e-3, 7).unwrap();
        let b = batch(3, 12, c.vocab);
        let (loss_cpu, _, g_cpu) = tr.loss_logits_grads(&b).unwrap();
        let mut gt = GpuTrainer::from_trainer(&tr).unwrap();
        let (loss_gpu, g_gpu) = gt.loss_and_grads(&b).unwrap();
        assert!((loss_cpu - loss_gpu).abs() < 1e-4 * loss_cpu.abs().max(1.0),
                "loss: cpu {loss_cpu} gpu {loss_gpu}");
        let names = |i: usize| -> String {
            match i { 0 => "embed".into(), 1 => "final_norm".into(), 2 => "head".into(),
                      _ => format!("layer {} {}", (i - 3) / 9,
                                   ["attn_norm","wq","wk","wv","wo","ffn_norm","w1","w2","w3"][(i - 3) % 9]) }
        };
        for i in 0..g_cpu.len() {
            let e = max_rel(&g_gpu[i], &g_cpu[i]);
            eprintln!("block {:2} {:<20} max rel err {e:.2e}", i, names(i));
            assert!(e < 1e-5, "block {} ({}): max |gpu - cpu| / max|cpu| = {e}", i, names(i));
        }
        // and the device agrees with itself to the bit
        let (loss2, g2) = gt.loss_and_grads(&b).unwrap();
        assert_eq!(loss2.to_bits(), loss_gpu.to_bits());
        assert!(g2 == g_gpu, "the device did not reproduce its own gradients");
        // checkpointing recomputes the same activations: the same bits
        gt.checkpoint = true;
        let (loss3, g3) = gt.loss_and_grads(&b).unwrap();
        assert_eq!(loss3.to_bits(), loss_gpu.to_bits(), "checkpointed loss");
        assert!(g3 == g_gpu, "checkpointing changed the gradients");
    }

    /// Mixed precision against the f32 step from the same state: the
    /// loss and every gradient block agree to f16 accuracy (a few parts
    /// in a thousand — activations and their gradients are rounded to
    /// 11 bits at every kernel edge), the result is bit-reproducible,
    /// and three steps track the f32 trainer's losses.
    #[test]
    fn mixed_precision_tracks_f32_and_reproduces_itself() {
        if gpu().map(|g| g.f16) != Some(true) { eprintln!("no f16 on this adapter; skipped"); return; }
        let c = cfg();
        let tr = Trainer::new(c, 1e-3, 7).unwrap();
        let b = batch(3, 12, c.vocab);
        let mut f32t = GpuTrainer::from_trainer(&tr).unwrap();
        let (loss32, g32) = f32t.loss_and_grads(&b).unwrap();
        let mut mx = GpuTrainer::from_trainer(&tr).unwrap();
        mx.set_precision(Precision::Mixed).unwrap();
        assert_eq!(mx.loss_scale, SCALE_INIT);
        let (loss16, g16) = mx.loss_and_grads(&b).unwrap();
        eprintln!("loss f32 {loss32} mixed {loss16}");
        assert!((loss16 - loss32).abs() < 5e-3 * loss32.abs().max(1.0), "loss: f32 {loss32} mixed {loss16}");
        for i in 0..g32.len() {
            let e = max_rel(&g16[i], &g32[i]);
            eprintln!("block {i:2} mixed-vs-f32 max rel err {e:.2e}");
            assert!(e < 2e-2, "block {i}: mixed vs f32 gradient differ by {e}");
        }
        let (loss16b, g16b) = mx.loss_and_grads(&b).unwrap();
        assert_eq!(loss16b.to_bits(), loss16.to_bits());
        assert!(g16b == g16, "mixed precision did not reproduce its own gradients");

        // three steps: losses track the f32 trainer, no overflow at scale 1024
        let mut cpu = Trainer::new(c, 1e-3, 11).unwrap();
        let mut mx = GpuTrainer::from_trainer(&cpu).unwrap();
        mx.set_precision(Precision::Mixed).unwrap();
        let b = batch(2, 16, c.vocab);
        for step in 0..3 {
            let lc = cpu.train_step(&b).unwrap();
            let lm = mx.train_step(&b).unwrap();
            eprintln!("step {step}: loss f32 {lc} mixed {lm}");
            assert!((lc - lm).abs() < 1e-2 * lc.abs().max(1.0), "step {step}: loss f32 {lc} mixed {lm}");
        }
        assert_eq!(mx.overflows, 0);

        // an absurd loss scale overflows the f16 loss gradient: the step is
        // skipped, the scale halves, the weights are untouched
        let before = mx.w[2].download();
        mx.loss_scale = 1.0e30;
        let t_before = mx.t;
        mx.train_step(&b).unwrap();
        assert_eq!(mx.overflows, 1, "the overflow must be detected");
        assert_eq!(mx.loss_scale, 0.5e30);
        assert_eq!(mx.t, t_before, "an overflowed step must not advance Adam");
        assert_eq!(mx.w[2].download(), before, "an overflowed step must not touch the weights");
    }

    /// Three optimizer steps on each side from the same state: the losses
    /// track and the weights that come back are the tape's to f32
    /// rounding — the round trip `from_trainer` / `download_into` is what
    /// evaluation and checkpoints go through.
    #[test]
    fn gpu_steps_track_the_cpu_trainer_and_round_trip_the_state() {
        if gpu().is_none() { eprintln!("no GPU adapter; skipped"); return; }
        let c = cfg();
        let mut cpu = Trainer::new(c, 1e-3, 11).unwrap();
        let mut back = Trainer::new(c, 1e-3, 11).unwrap();
        let mut gt = GpuTrainer::from_trainer(&cpu).unwrap();
        let b = batch(2, 16, c.vocab);
        for step in 0..3 {
            let lc = cpu.train_step(&b).unwrap();
            let lg = gt.train_step(&b).unwrap();
            eprintln!("step {step}: loss cpu {lc} gpu {lg}");
            assert!((lc - lg).abs() < 1e-3 * lc.abs().max(1.0), "step {step}: loss cpu {lc} gpu {lg}");
        }
        gt.download_into(&mut back);
        assert_eq!(back.step, cpu.step);
        assert_eq!(back.opt.t, cpu.opt.t);
        for i in 0..cpu.params.len() {
            let e = max_rel(&back.params[i], &cpu.params[i]);
            eprintln!("block {i:2} weights max rel err {e:.2e}");
            assert!(e < 1e-4, "block {i}: weights differ by {e} after three steps");
        }
        // the losses of the round-tripped model equal the CPU trainer's
        let (l1, l2) = (cpu.eval_loss(&b).unwrap(), back.eval_loss(&b).unwrap());
        assert!((l1 - l2).abs() < 1e-3, "eval after round trip: {l1} vs {l2}");
    }
}
