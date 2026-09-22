//! IEEE binary16 on the host: the conversions an f16 tensor needs to be
//! uploaded from and read back into f32. Round-to-nearest-even, the
//! rounding every device applies, so a value that round-trips through a
//! tensor is the value the device would have produced from the same f32.

/// f32 -> f16 bits, round to nearest even; overflow to infinity,
/// underflow to a subnormal or zero.
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        // inf or nan: keep a nan a nan
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;                     // rebiased exponent
    if e >= 0x1f {
        return sign | 0x7c00;                   // overflow -> inf
    }
    if e <= 0 {
        // subnormal half (or zero): shift the full 24-bit significand
        if e < -10 { return sign; }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1u32 << (shift - 1);
        let rounded = if rem > mid || (rem == mid && (half & 1) == 1) { half + 1 } else { half };
        return sign | rounded as u16;
    }
    let half = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    let rounded = if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) { half + 1 } else { half };
    sign | rounded as u16                       // a carry into the exponent is the right answer
}

/// f16 bits -> f32, exact.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // subnormal: normalise
            let mut m = mant;
            let mut e: i32 = 1;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            let m = m & 0x3ff;
            sign | (((e - 15 + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp - 15 + 127) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_rounds_to_nearest_even() {
        for &(x, bits) in &[(0.0f32, 0x0000u16), (1.0, 0x3c00), (-2.0, 0xc000), (65504.0, 0x7bff),
                            (0.333251953125, 0x3555), (6.103515625e-5, 0x0400), (5.960464477539063e-8, 0x0001)] {
            assert_eq!(f32_to_f16(x), bits, "{x}");
            assert_eq!(f16_to_f32(bits), x, "{bits:#x}");
        }
        assert_eq!(f32_to_f16(1e6), 0x7c00, "overflow is +inf");
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        // ties to even: 1 + 2^-11 is exactly between 1.0 and the next half
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        // every half round-trips exactly
        for h in 0..=0xffffu16 {
            let x = f16_to_f32(h);
            if x.is_nan() { continue; }
            assert_eq!(f32_to_f16(x), h, "{h:#x}");
        }
    }
}
