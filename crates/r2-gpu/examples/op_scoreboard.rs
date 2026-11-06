//! One line per operation of a training step, the kernel the step really calls on
//! the shapes it really calls it with, for `benchmarks/llm/scoreboard.py` to hold
//! against the same lines from `benchmarks/llm/op_scoreboard.py`:
//!
//!     SCORE r2 <layer> <size> <op> <phase> <count per step> <ms per call> <check>
//!
//! `check` sums the absolute output on data both sides generate alike, so a row
//! whose two sides did different work shows up as that, not as a speed.
//!
//!     SCORE_LAYER=cpu cargo run --release -p r2-gpu --example op_scoreboard
//!     SCORE_LAYER=vega cargo run --release -p r2-gpu --features gpu --example op_scoreboard
//!     SCORE_LAYER=t4 cargo run --release -p r2-gpu --features cuda --example op_scoreboard
//!
//! The model comes from the usual R2_DIM .. R2_BATCH, its name from SCORE_SIZE.

use r2_autograd::{BufPool, Tape, Var};
use std::time::Instant;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// `SCORE_STEP` over 0.1 makes neighbours unalike, as training data is.
fn fill(len: usize, ph: f32) -> Vec<f32> {
    let step: f32 = env("SCORE_STEP", 0.0007);
    (0..len).map(|i| ((i as f32) * step + ph).sin()).collect()
}

fn abs_sum(x: &[f32]) -> f64 { x.iter().map(|v| v.abs() as f64).sum() }

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[derive(Clone, Copy)]
struct Model { d: usize, kv: usize, ffn: usize, vocab: usize, nh: usize, nkv: usize, hd: usize, nl: usize, nseq: usize, seq: usize, t: usize }

impl Model {
    fn from_env() -> Model {
        let (d, nh, nkv) = (env("R2_DIM", 256usize), env("R2_HEADS", 4usize), env("R2_KV", 2usize));
        let (nseq, seq) = (env("R2_BATCH", 32usize), env("R2_SEQ", 64usize));
        let hd = d / nh;
        Model { d, kv: nkv * hd, ffn: env("R2_FFN", 768), vocab: env("R2_VOCAB", 8000), nh, nkv, hd,
                nl: env("R2_LAYERS", 4), nseq, seq, t: nseq * seq }
    }
    /// The parameter blocks, as the trainer holds them.
    fn blocks(&self) -> Vec<usize> {
        let (d, kv, f) = (self.d, self.kv, self.ffn);
        let mut v = vec![self.vocab * d, d, d * self.vocab];
        for _ in 0..self.nl { v.extend([d, d * d, d * kv, d * kv, d * d, d, d * f, f * d, d * f]); }
        v
    }
    fn tokens(&self, shift: usize) -> Vec<usize> {
        (0..self.nseq).flat_map(|b| (0..self.seq).map(move |i| (b, i)))
            .map(|(b, i)| (b * 131 + i * 7919 + shift) % self.vocab).collect()
    }
    /// (name, k, n, products of this shape in a step)
    fn gemms(&self) -> [(&'static str, usize, usize, usize); 5] {
        let (d, kv, f, l) = (self.d, self.kv, self.ffn, self.nl);
        [("gemm_wq", d, d, 2 * l), ("gemm_wk", d, kv, 2 * l), ("gemm_w1", d, f, 2 * l), ("gemm_w2", f, d, l), ("gemm_head", d, self.vocab, 1)]
    }
}

struct Out { layer: String, size: String }

impl Out {
    fn line(&self, op: &str, phase: &str, count: usize, ms: Option<f64>, check: f64) {
        let ms = ms.map_or("failed".to_string(), |m| format!("{m:.4}"));
        println!("SCORE r2 {} {} {op} {phase} {count} {ms} {check:.6e}", self.layer, self.size);
    }
}

// ── the CPU: the tape, with its pool, as the trainer drives it ──────────────
fn tape_row(o: &Out, pool: &mut BufPool, op: &str, count: usize, leaves: &[&[f32]], seed: &[f32],
            build: &dyn Fn(&mut Tape, &[Var]) -> Var) {
    let (mut fwd, mut bwd) = (Vec::new(), Vec::new());
    let (mut cf, mut cb) = (0.0, 0.0);
    for round in 0..8 {
        let mut tp = Tape::with_pool(std::mem::take(pool));
        let vars: Vec<Var> = leaves.iter().map(|l| tp.leaf(l.to_vec(), true)).collect();
        let s = Instant::now();
        let y = build(&mut tp, &vars);
        let f = s.elapsed().as_secs_f64() * 1e3;
        let s = Instant::now();
        tp.backward_from(y, seed);
        let b = s.elapsed().as_secs_f64() * 1e3;
        if round >= 3 { fwd.push(f); bwd.push(b); }
        cf = abs_sum(tp.value(y));
        cb = vars.iter().map(|&v| abs_sum(tp.grad(v))).sum();
        *pool = tp.into_pool();
    }
    o.line(op, "fwd", count, Some(median(fwd)), cf);
    o.line(op, "bwd", count, Some(median(bwd)), cb);
}

fn cpu_layer(o: &Out, m: Model) {
    let (t, d, l) = (m.t, m.d, m.nl);
    let scale = 1.0 / (m.hd as f32).sqrt();
    let mut pool = BufPool::new();
    let p = &mut pool;
    let toks = m.tokens(1);
    let tgts = m.tokens(2);
    tape_row(o, p, "embed", 1, &[&fill(m.vocab * d, 3.0)], &fill(t * d, 2.0), &|tp, v| tp.embed(v[0], &toks, d));
    tape_row(o, p, "rmsnorm", 2 * l + 1, &[&fill(t * d, 0.0), &fill(d, 1.0)], &fill(t * d, 2.0), &|tp, v| tp.rmsnorm(v[0], v[1], d, 1e-5));
    for (name, k, n, c) in m.gemms() {
        tape_row(o, p, name, c, &[&fill(t * k, 0.0), &fill(k * n, 1.0)], &fill(t * n, 2.0), &|tp, v| tp.matmul(v[0], v[1], t, k, n));
    }
    let rope = |tp: &mut Tape, v: &[Var], heads: usize| tp.rope_seq(v[0], t, m.seq, heads, m.hd, 10000.0);
    tape_row(o, p, "rope_q", l, &[&fill(t * d, 0.0)], &fill(t * d, 2.0), &|tp, v| rope(tp, v, m.nh));
    tape_row(o, p, "rope_k", l, &[&fill(t * m.kv, 0.0)], &fill(t * m.kv, 2.0), &|tp, v| rope(tp, v, m.nkv));
    tape_row(o, p, "attention", l, &[&fill(t * d, 0.0), &fill(t * m.kv, 1.0), &fill(t * m.kv, 2.0)], &fill(t * d, 3.0),
             &|tp, v| tp.attention(v[0], v[1], v[2], m.nseq, m.seq, m.nh, m.nkv, m.hd, scale));
    tape_row(o, p, "add", 2 * l, &[&fill(t * d, 0.0), &fill(t * d, 1.0)], &fill(t * d, 2.0), &|tp, v| tp.add(v[0], v[1]));
    tape_row(o, p, "silu", l, &[&fill(t * m.ffn, 0.0)], &fill(t * m.ffn, 2.0), &|tp, v| tp.silu(v[0]));
    tape_row(o, p, "mul", l, &[&fill(t * m.ffn, 0.0), &fill(t * m.ffn, 1.0)], &fill(t * m.ffn, 2.0), &|tp, v| tp.mul(v[0], v[1]));
    tape_row(o, p, "softmax_ce", 1, &[&fill(t * m.vocab, 0.0)], &[1.0], &|tp, v| tp.softmax_ce(v[0], m.vocab, tgts.clone()));

    let sizes = m.blocks();
    let total: usize = sizes.iter().sum();
    let mut params: Vec<Vec<f32>> = sizes.iter().map(|&n| fill(n, 0.0)).collect();
    let grads: Vec<Vec<f32>> = sizes.iter().map(|&n| fill(n, 1.0)).collect();
    let refs: Vec<&[f32]> = grads.iter().map(|g| &g[..]).collect();
    let mut opt = r2_train::optim::Adam::new(total, 3e-4);
    let ms: Vec<f64> = (0..8).map(|_| {
        let s = Instant::now();
        opt.step_blocks(&mut params, &refs, 1.0).unwrap();
        s.elapsed().as_secs_f64() * 1e3
    }).collect();
    o.line("adam", "step", 1, Some(median(ms[3..].to_vec())), f64::NAN);
}

// ── a device: the kernels the GPU step launches, with its flags ─────────────
#[cfg(feature = "cuda")]
mod dev {
    pub use r2_gpu::cuda::{attention, elementwise as ew, gemm::gemm, optim, Tensor};
    pub fn fence() -> bool { r2_gpu::cuda::sync() }
    pub fn line() -> String { r2_gpu::cuda::device_line() }
}
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
mod dev {
    pub use r2_gpu::{attention, device::Tensor, elementwise as ew, gemm::gemm, optim};
    pub fn fence() -> bool { r2_gpu::device::sync(); true }
    pub fn line() -> String { r2_gpu::device::adapter_line() }
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
mod device_layer {
    use super::dev::{attention, ew, fence, gemm, optim, Tensor};
    use super::*;

    /// Milliseconds per call: enough calls for `SCORE_SECS` closed by one fence, median of 5.
    fn timed(mut f: impl FnMut() -> bool) -> Option<f64> {
        let budget: f64 = env("SCORE_SECS", 0.15);
        let mut ok = f() && fence();
        let s = Instant::now();
        ok &= f() && fence();
        let reps = (budget / s.elapsed().as_secs_f64().max(1e-6)).ceil().clamp(1.0, 200.0) as usize;
        let v: Vec<f64> = (0..5).map(|_| {
            let s = Instant::now();
            for _ in 0..reps { ok &= f(); }
            ok &= fence();
            s.elapsed().as_secs_f64() / reps as f64 * 1e3
        }).collect();
        ok.then(|| median(v))
    }

    fn up(x: &[f32]) -> Tensor { Tensor::upload(x).expect("upload") }
    fn zeros(n: usize) -> Tensor { Tensor::zeros(n).expect("zeros") }
    fn sums(ts: &[&Tensor]) -> f64 { ts.iter().map(|t| abs_sum(&t.download())).sum() }

    pub fn run(o: &Out, m: Model) {
        let (t, d, kv, ffn, l) = (m.t, m.d, m.kv, m.ffn, m.nl);
        let eps = 1e-5f32;

        // embed
        let (toks, tgts) = (m.tokens(1), m.tokens(2));
        let (table, ids, y) = (up(&fill(m.vocab * d, 3.0)), ew::upload_ids(&toks).unwrap(), zeros(t * d));
        o.line("embed", "fwd", 1, timed(|| ew::embed_fwd(&table, &ids, &y, t, d)), sums(&[&y]));
        let (g, gt) = (up(&fill(t * d, 2.0)), zeros(m.vocab * d));
        let index = ew::TokenIndex::build(&toks, m.vocab).unwrap();
        let ms = timed(|| ew::embed_bwd(&g, &index, &gt, m.vocab, d, true));
        o.line("embed", "bwd", 1, ms, sums(&[&gt]));

        // rmsnorm: the step's flags, dX accumulating; the check starts from zero
        let (x, w, gw) = (up(&fill(t * d, 0.0)), up(&fill(d, 1.0)), zeros(d));
        let (gx, sc) = (zeros(t * d), zeros(ew::rmsnorm_scratch_len(t, d)));
        o.line("rmsnorm", "fwd", 2 * l + 1, timed(|| ew::rmsnorm_fwd(&x, &w, &y, t, d, eps)), sums(&[&y]));
        let ms = timed(|| ew::rmsnorm_bwd(&x, &w, &g, &gx, &gw, &sc, t, d, eps, false, true));
        let gx0 = zeros(t * d);
        ew::rmsnorm_bwd(&x, &w, &g, &gx0, &gw, &sc, t, d, eps, false, true);
        o.line("rmsnorm", "bwd", 2 * l + 1, ms, sums(&[&gx0, &gw]));

        // the products; a gradient is both of a product's two
        for (name, k, n, c) in m.gemms() {
            let (a, b, g, cc) = (up(&fill(t * k, 0.0)), up(&fill(k * n, 1.0)), up(&fill(t * n, 2.0)), zeros(t * n));
            o.line(name, "fwd", c, timed(|| gemm(&a, false, &b, false, t, k, n, &cc, false)), sums(&[&cc]));
            let (ga, gb) = (zeros(t * k), zeros(k * n));
            let ms = timed(|| gemm(&a, true, &g, false, k, t, n, &gb, false) && gemm(&g, false, &b, true, t, n, k, &ga, false));
            o.line(name, "bwd", c, ms, sums(&[&ga, &gb]));
        }

        // RoPE
        for (name, heads, width) in [("rope_q", m.nh, d), ("rope_k", m.nkv, kv)] {
            let (a, out, g, ga) = (up(&fill(t * width, 0.0)), zeros(t * width), up(&fill(t * width, 2.0)), zeros(t * width));
            o.line(name, "fwd", l, timed(|| ew::rope_fwd(&a, &out, t, m.seq, heads, m.hd, 10000.0)), sums(&[&out]));
            let ms = timed(|| ew::rope_bwd(&g, &ga, t, m.seq, heads, m.hd, 10000.0, true));
            o.line(name, "bwd", l, ms, sums(&[&ga]));
        }

        // attention
        let sh = attention::Shape { nseq: m.nseq, seq: m.seq, nh: m.nh, nkv: m.nkv, hd: m.hd, scale: 1.0 / (m.hd as f32).sqrt() };
        // SCORE_AMP over 1 peaks the softmax, as trained activations can
        let amp: f32 = env("SCORE_AMP", 1.0);
        let scaled = |n: usize, ph: f32| -> Vec<f32> { fill(n, ph).iter().map(|x| x * amp).collect() };
        let (q, k, v) = (up(&scaled(t * d, 0.0)), up(&scaled(t * kv, 1.0)), up(&fill(t * kv, 2.0)));
        let (ctx, lse, ga, de) = (zeros(t * d), zeros(t * m.nh), up(&fill(t * d, 3.0)), zeros(t * m.nh));
        let (dq, dk, dv) = (zeros(t * d), zeros(t * kv), zeros(t * kv));
        o.line("attention", "fwd", l, timed(|| attention::forward(&q, &k, &v, &ctx, &lse, &sh)), sums(&[&ctx]));
        let ms = timed(|| attention::backward(&q, &k, &v, &ctx, &lse, &ga, &de, &dq, &dk, &dv, &sh));
        o.line("attention", "bwd", l, ms, sums(&[&dq, &dk, &dv]));

        // the flat ones
        let (a, b) = (up(&fill(t * d, 0.0)), up(&fill(t * d, 1.0)));
        let z = zeros(t * d);
        ew::flat_fwd(ew::Flat::Add, &a, &b, &z, t * d);
        fence();
        let check = sums(&[&z]);
        o.line("add", "fwd", 2 * l, timed(|| ew::flat_fwd(ew::Flat::AddInto, &a, &a, &b, t * d)), check);
        let n = t * ffn;
        let (a, b, g, r, g1, g2) = (up(&fill(n, 0.0)), up(&fill(n, 1.0)), up(&fill(n, 2.0)), zeros(n), zeros(n), zeros(n));
        o.line("silu", "fwd", l, timed(|| ew::flat_fwd(ew::Flat::Silu, &a, &a, &r, n)), sums(&[&r]));
        let ms = timed(|| ew::flat_bwd(ew::Flat::Silu, &a, &a, &g, Some((&g1, true)), None, n));
        o.line("silu", "bwd", l, ms, sums(&[&g1]));
        o.line("mul", "fwd", l, timed(|| ew::flat_fwd(ew::Flat::Mul, &a, &b, &r, n)), sums(&[&r]));
        let ms = timed(|| ew::flat_bwd(ew::Flat::Mul, &a, &b, &g, Some((&g1, true)), Some((&g2, true)), n));
        o.line("mul", "bwd", l, ms, sums(&[&g1, &g2]));

        // loss
        let (lg, tg, ls, rows, gl) = (up(&fill(t * m.vocab, 0.0)), ew::upload_ids(&tgts).unwrap(), zeros(t), zeros(t + 1), zeros(t * m.vocab));
        o.line("softmax_ce", "fwd", 1, timed(|| ew::softmax_ce_fwd(&lg, &tg, &ls, &rows, t, m.vocab)), sums(&[&rows]) / t as f64);
        let ms = timed(|| ew::softmax_ce_bwd(&lg, &tg, &ls, &gl, t, m.vocab, 1.0 / t as f32, true));
        o.line("softmax_ce", "bwd", 1, ms, sums(&[&gl]));

        // what surrounds the kernels each step
        let s = Instant::now();
        std::hint::black_box(rows.download());
        o.line("read_loss", "step", 1, Some(s.elapsed().as_secs_f64() * 1e3), f64::NAN);
        let ms = timed(|| {
            let (a, b) = (ew::upload_ids(&toks), ew::upload_ids(&tgts));
            let c = ew::TokenIndex::build(&toks, m.vocab);
            a.is_some() && b.is_some() && c.is_some()
        });
        o.line("batch_setup", "step", 1, ms, f64::NAN);

        // Adam, block by block
        let sizes = m.blocks();
        let (ws, gs): (Vec<Tensor>, Vec<Tensor>) = (sizes.iter().map(|&n| up(&fill(n, 0.0))).collect(), sizes.iter().map(|&n| up(&fill(n, 1.0))).collect());
        let (ms1, vs): (Vec<Tensor>, Vec<Tensor>) = (sizes.iter().map(|&n| zeros(n)).collect(), sizes.iter().map(|&n| zeros(n)).collect());
        let mut step = 0u32;
        let ms = timed(|| {
            step += 1;
            (0..sizes.len()).all(|i| optim::adam_step(&ws[i], &gs[i], &ms1[i], &vs[i], sizes[i], 3e-4, 0.9, 0.999, 1e-8, step, 1.0, 0.0))
        });
        o.line("adam", "step", 1, ms, f64::NAN);

        // not a share of the step: what one launch and a copy cost
        let (a, b, c) = (up(&[1.0]), up(&[2.0]), zeros(1));
        o.line("launch_one", "info", 0, timed(|| ew::flat_fwd(ew::Flat::Add, &a, &b, &c, 1)), f64::NAN);
        let big = fill(16 << 20, 0.0);
        let dst = zeros(big.len());
        let copy = |f: &mut dyn FnMut()| median((0..5).map(|_| { let s = Instant::now(); f(); s.elapsed().as_secs_f64() * 1e3 }).collect());
        o.line("upload_64MB", "info", 0, Some(copy(&mut || { dst.write(&big); fence(); })), f64::NAN);
        o.line("download_64MB", "info", 0, Some(copy(&mut || { std::hint::black_box(dst.download()); })), f64::NAN);
    }
}

/// Two processes: one model's gradients averaged, as `GpuTrainer` does each step.
#[cfg(feature = "cuda")]
fn exchange(o: &Out, m: Model) {
    use r2_gpu::cuda::{cluster::Cluster, sync, Tensor};
    let Ok(Some(mut c)) = Cluster::from_env() else { panic!("exchange needs R2_WORLD, R2_RANK") };
    let sizes = m.blocks();
    let gs: Vec<Tensor> = sizes.iter().map(|&n| Tensor::upload(&fill(n, 1.0)).unwrap()).collect();
    let refs: Vec<&Tensor> = gs.iter().collect();
    c.average(&refs, &sizes).expect("average");
    let ms: Vec<f64> = (0..8).map(|_| {
        let s = Instant::now();
        c.average(&refs, &sizes).expect("average");
        sync();
        s.elapsed().as_secs_f64() * 1e3
    }).collect();
    o.line("exchange", "step", 1, Some(median(ms[3..].to_vec())), f64::NAN);
}

fn main() {
    let m = Model::from_env();
    let o = Out { layer: env("SCORE_LAYER", "cpu".to_string()), size: env("SCORE_SIZE", "small".to_string()) };
    #[cfg(feature = "cuda")]
    if std::env::var("R2_WORLD").is_ok() { exchange(&o, m); return; }
    if o.layer == "cpu" { cpu_layer(&o, m); return; }
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        println!("device: {}", dev::line());
        device_layer::run(&o, m);
        return;
    }
    #[allow(unreachable_code)]
    { panic!("layer {} needs --features gpu or cuda", o.layer); }
}
