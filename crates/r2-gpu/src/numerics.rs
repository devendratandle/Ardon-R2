//! What this adapter's arithmetic actually does, operation by operation,
//! against the correctly rounded answer.
//!
//! Cross-vendor bit identity needs every float operation to give the same
//! bits on every device. The WGSL and Vulkan precision rules do not
//! require that: `x / y` may be off by 2.5 ULP, `sqrt` inherits the
//! looseness of `inversesqrt`, `exp`/`log`/`sin` are the vendor's own, and
//! a driver may fuse `a * b + c` into one rounding or not. Which of these
//! a given device exercises is an empirical question, so this module asks
//! it: the same inputs through each operation on the device, compared
//! with the IEEE result computed on the host (division, sqrt and the
//! unfused / fused multiply-add are exact in Rust; the transcendentals
//! are computed in f64 and rounded, which is correctly rounded but for
//! vanishingly rare double-rounding cases).
//!
//! `--example precision_probe` prints the table; run it on every adapter
//! the determinism guarantee is meant to cover.

use crate::device::{gpu, Tensor};
use wgpu::util::DeviceExt;

/// The operations probed, in output order. The last two are R2's own
/// correctly rounded division and square root ([`CR_WGSL`]).
pub const OPS: [&str; 11] = ["a / b", "1 / b", "sqrt(a)", "a * b + c", "fma(a, b, c)", "exp(x)", "log(a)", "sin(x)",
                             "div_cr(a, b)", "sqrt_cr(a)", "exp_r2(x)"];
const NOPS: usize = OPS.len();

/// Correctly rounded division and square root, for kernels that promise
/// the same bits on every device.
///
/// The builtins are allowed to be off (2.5 ULP for `/`), and on this
/// adapter 29% of divisions and 15% of square roots are. Each function
/// here takes the builtin's answer and applies one correction step whose
/// remainder is computed with `fma` — exact, because `fma` rounds once
/// and the remainder of a near-correct quotient is representable — so
/// the correction lands on the correctly rounded result. That relies on
/// `fma` itself being exact, which `probe` checks per device.
///
/// The correction is skipped when the builtin's answer is zero or not
/// finite: those are already exact (`sqrt(0) = 0`, `a / ±inf = ±0`,
/// `a / 0 = ±inf`), and correcting them would compute `0 * inf = NaN` —
/// which it did, in Adam, for every weight whose gradient had only ever
/// been zero (`v = 0`, so `sqrt_cr(0)`), and a real training run went to
/// NaN while every unit test passed. Non-finiteness is read from the
/// exponent bits, which no fast-math rewrite can touch.
pub const CR_WGSL: &str = r#"
fn cr_special(x: f32) -> bool {
    return x == 0.0 || (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u;
}
fn div_cr(a0: f32, b0: f32) -> f32 {
    // This adapter divides as a * (1 / b), and 1 / b goes subnormal — and
    // is flushed to zero — once |b| > 2^126: f32::MAX / f32::MAX came back
    // 0. Scaling both operands by 1/4 is exact (a power of two), leaves
    // the quotient unchanged, and keeps the reciprocal normal.
    var a = a0;
    var b = b0;
    if (abs(b) >= bitcast<f32>(0x7e800000u)) { a = a * 0.25; b = b * 0.25; }
    let q = a / b;
    if (cr_special(q)) { return q; }
    let r = fma(-q, b, a);
    return fma(r, 1.0 / b, q);
}
fn sqrt_cr(a: f32) -> f32 {
    let s = sqrt(a);
    if (cr_special(s)) { return s; }
    let r = fma(-s, s, a);
    return fma(r, 0.5 / s, s);
}
"#;

/// R2's `exp` in WGSL: `r2_tensor::ops::exp_r2` — and so the CPU's AVX2
/// `exp8` lane — instruction for instruction. Same clamp, ties-to-even
/// `round`, split-ln2 reduction and Horner chain in `fma`, same `2^n`
/// assembly in the exponent bits. Every step is exactly rounded (the
/// probe checks `fma` per device), so the GPU's `exp` gives the CPU's
/// bits. The constants are written as the bit patterns of the CPU's own
/// literals: no decimal re-parsing stands between the two.
pub fn exp_wgsl() -> String {
    let b = |x: f32| format!("bitcast<f32>({:#010x}u)", x.to_bits());
    format!(r#"
fn exp_r2(x0: f32) -> f32 {{
    if (x0 < {lo}) {{ return 0.0; }}
    let x = min(x0, {hi});
    let n = round(x * {log2e});
    var r = fma(-n, {c1}, x);
    r = fma(-n, {c2}, r);
    var y = {p0};
    y = fma(y, r, {p1});
    y = fma(y, r, {p2});
    y = fma(y, r, {p3});
    y = fma(y, r, {p4});
    y = fma(y, r, {p5});
    y = fma(y, r * r, r);
    y = y + 1.0;
    return y * bitcast<f32>(u32(i32(n) + 127) << 23u);
}}
"#, lo = b(-87.336_54), hi = b(88.376_26), log2e = b(std::f32::consts::LOG2_E),
        c1 = b(0.693_359_38), c2 = b(-2.121_944_4e-4),
        p0 = b(1.987_569_1e-4), p1 = b(1.398_199_9e-3), p2 = b(8.333_452e-3),
        p3 = b(4.166_579_6e-2), p4 = b(1.666_666_5e-1), p5 = b(5.000_000_1e-1))
}

/// Everything a kernel needs to compute the same bits on every device:
/// correctly rounded division and square root, and R2's `exp`. Prepended
/// to every kernel after its `enable` directives.
pub fn prelude() -> String { format!("{CR_WGSL}{}", exp_wgsl()) }

/// How one operation compared with the correctly rounded result.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpStats {
    pub n: usize,
    /// Bit-identical to the correctly rounded result.
    pub exact: usize,
    /// The largest distance, in units in the last place.
    pub max_ulp: u64,
    /// For `a * b + c` only: elements that match the FUSED result (one
    /// rounding) and not the unfused one — evidence the driver contracted.
    pub fused: usize,
}

const SRC: &str = r#"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> C: array<f32>;
@group(0) @binding(3) var<storage, read> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@group(0) @binding(5) var<uniform> n: vec4<u32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= n.x) { return; }
    let a = A[i];
    let b = B[i];
    let c = C[i];
    let x = X[i];
    let o = i * 11u;
    O[o + 0u] = a / b;
    O[o + 1u] = 1.0 / b;
    O[o + 2u] = sqrt(a);
    O[o + 3u] = a * b + c;
    O[o + 4u] = fma(a, b, c);
    O[o + 5u] = exp(x);
    O[o + 6u] = log(a);
    O[o + 7u] = sin(x);
    O[o + 8u] = div_cr(a, b);
    O[o + 9u] = sqrt_cr(a);
    O[o + 10u] = exp_r2(x);
}
"#;

/// Distance in ULPs between two finite f32s (0 when bit-identical).
pub fn ulps(a: f32, b: f32) -> u64 {
    if a.to_bits() == b.to_bits() { return 0; }
    if a.is_nan() || b.is_nan() { return u64::MAX; }
    let ord = |x: f32| -> i64 {
        let i = x.to_bits() as i32;
        (if i < 0 { i32::MIN.wrapping_sub(i) } else { i }) as i64
    };
    (ord(a) - ord(b)).unsigned_abs()
}

/// The inputs: `a` positive (for sqrt and log) spread over 2^-20..2^20,
/// `b` of either sign and similar spread, `c` of mixed sign near `a * b`
/// so that the rounding of the product decides the sum, and `x` in
/// [-10, 10] for exp and sin. A fixed xorshift sequence: the probe is
/// itself reproducible.
fn inputs(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut u = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 };
    let mut a = Vec::with_capacity(n); let mut b = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n); let mut x = Vec::with_capacity(n);
    for _ in 0..n {
        let av = ((1.0 + u()) * 2f64.powi((u() * 40.0) as i32 - 20)) as f32;
        let bv = ((1.0 + u()) * 2f64.powi((u() * 40.0) as i32 - 20) * if u() < 0.5 { -1.0 } else { 1.0 }) as f32;
        let cv = (-(av as f64 * bv as f64) * (1.0 + (u() - 0.5) * 1e-3)) as f32;
        a.push(av); b.push(bv); c.push(cv); x.push(((u() - 0.5) * 20.0) as f32);
    }
    (a, b, c, x)
}

/// Run every operation over `n` inputs on the device and compare. `None`
/// when there is no device.
pub fn probe(n: usize) -> Option<Vec<OpStats>> {
    let g = gpu()?;
    let (a, b, c, x) = inputs(n);
    let (ta, tb, tc, tx) = (Tensor::upload(&a)?, Tensor::upload(&b)?, Tensor::upload(&c)?, Tensor::upload(&x)?);
    let out = Tensor::zeros(n * NOPS)?;
    let module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("r2gpu-numerics"), source: wgpu::ShaderSource::Wgsl(format!("{}{SRC}", prelude()).into()),
    });
    let pipe = g.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("r2gpu-numerics"), layout: None, module: &module,
        entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });
    let dims = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&[n as u32, 0, 0, 0]), usage: wgpu::BufferUsages::UNIFORM,
    });
    let bufs = [&ta.buf, &tb.buf, &tc.buf, &tx.buf, &out.buf, &dims];
    let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
    let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipe.get_bind_group_layout(0), entries: &entries,
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
    }
    g.queue.submit(Some(enc.finish()));
    let got = out.download();

    let mut st = [OpStats { n, ..Default::default() }; NOPS];
    for i in 0..n {
        let (a, b, c, x) = (a[i], b[i], c[i], x[i]);
        let unfused = a * b + c;                  // Rust never contracts: two roundings
        let fused = a.mul_add(b, c);              // one rounding
        let want = [a / b, 1.0 / b, a.sqrt(), unfused, fused,
                    (x as f64).exp() as f32, (a as f64).ln() as f32, (x as f64).sin() as f32,
                    a / b, a.sqrt(), (x as f64).exp() as f32];
        for (k, w) in want.iter().enumerate() {
            let d = ulps(got[i * NOPS + k], *w);
            let s = &mut st[k];
            if d == 0 { s.exact += 1; }
            s.max_ulp = s.max_ulp.max(d);
        }
        if unfused.to_bits() != fused.to_bits() && got[i * NOPS + 3].to_bits() == fused.to_bits() {
            st[3].fused += 1;
        }
    }
    Some(st.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulps_counts_units_in_the_last_place() {
        assert_eq!(ulps(1.0, 1.0), 0);
        assert_eq!(ulps(1.0, f32::from_bits(1.0f32.to_bits() + 1)), 1);
        assert_eq!(ulps(-0.0, 0.0), 0);
        assert_eq!(ulps(f32::from_bits(1), -f32::from_bits(1)), 2);
    }

    /// `y[i] = f(x[i])` on the device, for a WGSL expression in `x`.
    fn gpu_map(expr: &str, xs: &[f32]) -> Vec<f32> {
        let g = gpu().unwrap();
        let src = format!("{}
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&X)) {{ return; }}
    let x = X[i];
    Y[i] = {expr};
}}", prelude());
        let (tx, ty) = (Tensor::upload(xs).unwrap(), Tensor::zeros(xs.len()).unwrap());
        let module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None, source: wgpu::ShaderSource::Wgsl(src.into()) });
        let pipe = g.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None });
        let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &pipe.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: tx.buf.as_entire_binding() },
                       wgpu::BindGroupEntry { binding: 1, resource: ty.buf.as_entire_binding() }] });
        let mut enc = g.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((xs.len() as u32).div_ceil(256), 1, 1);
        }
        g.queue.submit(Some(enc.finish()));
        ty.download()
    }

    /// THE point of `exp_r2`: the GPU computes the CPU's bits. Every value
    /// of a fine grid over the whole range, plus the edges.
    #[test]
    fn gpu_exp_r2_is_bit_identical_to_the_cpu() {
        if gpu().is_none() { eprintln!("no GPU adapter; skipped"); return; }
        let n = 1 << 20;
        let mut xs: Vec<f32> = (0..n).map(|i| -90.0 + 180.0 * i as f32 / n as f32).collect();
        xs.extend_from_slice(&[0.0, -0.0, -87.336_54, -87.336_55, 88.376_26, 88.4, f32::MIN_POSITIVE, -1e-30]);
        let got = gpu_map("exp_r2(x)", &xs);
        let off: Vec<usize> = (0..xs.len()).filter(|&i| got[i].to_bits() != r2_tensor::ops::exp_r2(xs[i]).to_bits()).collect();
        assert!(off.is_empty(), "{} of {} differ; first x = {:?}", off.len(), xs.len(),
                off.first().map(|&i| (xs[i], got[i], r2_tensor::ops::exp_r2(xs[i]))));
    }

    /// The special values the probe's random inputs never contain — and
    /// that a real training run hits at once: zero, infinity, the edges of
    /// the normal range. (Subnormals are left out on purpose: whether a
    /// device keeps them is exactly what SPIR-V's Float Controls pins, and
    /// WGSL cannot ask for it.)
    #[test]
    fn cr_ops_handle_zeros_and_infinities_like_ieee() {
        if gpu().is_none() { eprintln!("no GPU adapter; skipped"); return; }
        let xs = [0.0f32, -0.0, f32::MIN_POSITIVE, 1.0, 2.0, 1e-30, 1e30, f32::MAX, f32::INFINITY, -1.0, -f32::INFINITY];
        let cases: [(&str, fn(f32) -> f32); 4] = [
            ("sqrt_cr(abs(x))", |x| x.abs().sqrt()),
            ("div_cr(1.0, x)", |x| 1.0 / x),
            ("div_cr(x, 3.0)", |x| x / 3.0),
            ("div_cr(x, x + 1.0)", |x| x / (x + 1.0)),
        ];
        for (expr, cpu) in cases {
            let got = gpu_map(expr, &xs);
            for (&x, &g) in xs.iter().zip(&got) {
                let w = cpu(x);
                // this adapter flushes a SUBNORMAL result to zero (1 / f32::MAX
                // gives 0, not 2.9e-39): the denormal rule, which only SPIR-V
                // Float Controls can pin — not a div_cr question
                if w != 0.0 && w.abs() < f32::MIN_POSITIVE { continue; }
                let same = g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan());
                assert!(same, "{expr} at x = {x}: gpu {g} vs cpu {w}");
            }
        }
    }

    /// And so a whole elementwise op: silu on the GPU, written the way the
    /// CPU writes it, gives the CPU's `silu_into` bits.
    #[test]
    fn gpu_silu_is_bit_identical_to_the_cpu() {
        if gpu().is_none() { eprintln!("no GPU adapter; skipped"); return; }
        let n = 1 << 20;
        let xs: Vec<f32> = (0..n).map(|i| -40.0 + 80.0 * i as f32 / n as f32).collect();
        let got = gpu_map("div_cr(x, 1.0 + exp_r2(-x))", &xs);
        let mut want = vec![0.0f32; n];
        r2_tensor::ops::silu_into(&xs, &mut want);
        let off = (0..n).filter(|&i| got[i].to_bits() != want[i].to_bits()).count();
        assert_eq!(off, 0, "{off} of {n} silu values differ from the CPU");
    }

    /// The probe runs and is itself deterministic; what it reports is
    /// a property of the adapter, so it is printed, not asserted.
    #[test]
    fn probe_runs_and_reproduces() {
        if gpu().is_none() { eprintln!("no GPU adapter; skipped"); return; }
        let (a, b) = (probe(4096).unwrap(), probe(4096).unwrap());
        for (k, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!((x.exact, x.max_ulp), (y.exact, y.max_ulp), "{} changed between runs", OPS[k]);
        }
    }
}
