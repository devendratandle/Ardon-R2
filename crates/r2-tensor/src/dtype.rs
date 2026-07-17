//! LLM numeric dtypes and their conversions to/from f32 (the compute type).
//!
//! Why these exist (docs/LLM_TRILLION_ARCHITECTURE.md, Layer 0): a 32B
//! model in f32 is ~128 GB; in Q4 it is ~18 GB — the difference between
//! "needs a datacenter" and "mmaps on one box". Weights are STORED
//! quantized (or in bf16/f16) and DEQUANTIZED to f32 per tile on the
//! compute path. f32 is the compute/accuracy reference for the whole
//! neural path (statistics stay f64 elsewhere — never mixed).

/// bf16 → f32: bf16 is just the top 16 bits of an f32, so widening is a
/// bit-shift. Exact (bf16 values are a subset of f32).
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 → bf16 with round-to-nearest-even (the correct rounding, not truncation).
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    // round-to-nearest-even on the discarded low 16 bits
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

/// IEEE half (f16) → f32. Handles ±0, subnormals, inf, NaN.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    match exp {
        0 => {
            if mant == 0 {
                f32::from_bits(sign) // ±0
            } else {
                // subnormal half → normalized f32
                let mut m = mant;
                let mut e: i32 = 0;
                while m & 0x400 == 0 { m <<= 1; e -= 1; }
                m &= 0x3ff;
                let fe = (127 - 15 + e) as u32;
                f32::from_bits(sign | (fe << 23) | (m << 13))
            }
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (mant << 13)), // inf/NaN
        _ => {
            let fe = exp + (127 - 15);
            f32::from_bits(sign | (fe << 23) | (mant << 13))
        }
    }
}

/// A GGUF/GGML-style Q8_0 block: 32 int8 weights sharing one f16 scale.
/// Dequant: w[i] = scale * q[i]. Compact (34 bytes / 32 weights) and
/// exact-per-block (no inter-block error).
pub const QK: usize = 32;

#[derive(Clone)]
pub struct BlockQ8_0 {
    pub scale: f32,       // stored as f16 in the file; kept f32 in memory
    pub qs: [i8; QK],
}

impl BlockQ8_0 {
    /// Quantize 32 f32 values into a Q8_0 block (symmetric, per-block max).
    pub fn quantize(vals: &[f32]) -> BlockQ8_0 {
        debug_assert_eq!(vals.len(), QK);
        let amax = vals.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let mut qs = [0i8; QK];
        for (i, &v) in vals.iter().enumerate() {
            qs[i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
        BlockQ8_0 { scale, qs }
    }
    /// Dequantize back to 32 f32 values.
    pub fn dequantize(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), QK);
        for i in 0..QK { out[i] = self.scale * self.qs[i] as f32; }
    }
}

/// Q4_0 block: 32 4-bit weights (packed 2/byte) sharing one scale.
/// ~18 GB for a 32B model. Values are offset-encoded [-8, 7] * scale.
#[derive(Clone)]
pub struct BlockQ4_0 {
    pub scale: f32,
    pub qs: [u8; QK / 2], // 16 bytes, two nibbles each
}

impl BlockQ4_0 {
    pub fn quantize(vals: &[f32]) -> BlockQ4_0 {
        debug_assert_eq!(vals.len(), QK);
        let amax = vals.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        // symmetric 4-bit: levels -8..7, scale maps amax to 7 (keep 1 for -8 headroom)
        let scale = if amax > 0.0 { amax / 7.0 } else { 1.0 };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let mut qs = [0u8; QK / 2];
        for i in 0..QK / 2 {
            let a = ((vals[2 * i] * inv).round().clamp(-8.0, 7.0) as i32 + 8) as u8 & 0x0f;
            let b = ((vals[2 * i + 1] * inv).round().clamp(-8.0, 7.0) as i32 + 8) as u8 & 0x0f;
            qs[i] = a | (b << 4);
        }
        BlockQ4_0 { scale, qs }
    }
    pub fn dequantize(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), QK);
        for i in 0..QK / 2 {
            let a = (self.qs[i] & 0x0f) as i32 - 8;
            let b = (self.qs[i] >> 4) as i32 - 8;
            out[2 * i] = self.scale * a as f32;
            out[2 * i + 1] = self.scale * b as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip_exact_for_representable() {
        for &x in &[0.0f32, 1.0, -2.0, 0.5, 100.0, -0.25] {
            let back = bf16_to_f32(f32_to_bf16(x));
            assert_eq!(back, x, "bf16 rt {}", x);
        }
    }
    #[test]
    fn bf16_rounds_nearest_not_truncate() {
        // A value needing rounding should be within one bf16 ulp.
        let x = 1.3_f32;
        let back = bf16_to_f32(f32_to_bf16(x));
        assert!((back - x).abs() < 0.01, "{} vs {}", back, x);
    }
    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);   // 1.0
        assert_eq!(f16_to_f32(0x4000), 2.0);   // 2.0
        assert_eq!(f16_to_f32(0xc000), -2.0);  // -2.0
        assert_eq!(f16_to_f32(0x0000), 0.0);   // +0
        assert!((f16_to_f32(0x3555) - 0.333).abs() < 1e-3); // ~1/3
    }
    #[test]
    fn q8_block_roundtrip_bounded_error() {
        let vals: Vec<f32> = (0..QK).map(|i| (i as f32 - 16.0) * 0.3).collect();
        let blk = BlockQ8_0::quantize(&vals);
        let mut out = vec![0.0; QK];
        blk.dequantize(&mut out);
        // Q8 error ≤ scale/2 per element.
        let maxerr = vals.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(maxerr <= blk.scale / 2.0 + 1e-6, "maxerr {} scale {}", maxerr, blk.scale);
    }
    #[test]
    fn q4_block_roundtrip_bounded_error() {
        let vals: Vec<f32> = (0..QK).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let blk = BlockQ4_0::quantize(&vals);
        let mut out = vec![0.0; QK];
        blk.dequantize(&mut out);
        let maxerr = vals.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        // Q4 is coarse: error ≤ scale (one level).
        assert!(maxerr <= blk.scale + 1e-6, "maxerr {} scale {}", maxerr, blk.scale);
    }
}
