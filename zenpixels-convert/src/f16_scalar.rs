//! Local IEEE 754 binary16 (half-precision) conversion with scalar
//! fallback and F16C SIMD dispatch.
//!
//! The two **scalar** functions (`f16_bits_to_f32`, `f32_to_f16_bits`)
//! cover every platform and are the source of truth for correctness. The
//! **slice** wrappers (`f16_bits_to_f32_slice`, `f32_to_f16_bits_slice`)
//! dispatch to F16C intrinsics on x86-64 when the CPU has them (via
//! archmage's `X64V3Token`, which carries `f16c`), processing 8 lanes
//! per instruction; they fall back to the scalar loop otherwise.
//!
//! Binary16 format:
//! ```text
//!   sign     exp       mantissa
//!    1    |  5   |       10
//! ```
//!
//! - Bias: 15 (vs f32's 127). Unbiased range: [-14, +15] for normals.
//! - Max normal: (2 - 2^-10) × 2^15 ≈ 65504.
//! - Smallest normal: 2^-14 ≈ 6.10e-5.
//! - Smallest subnormal: 2^-24 ≈ 5.96e-8.
//!
//! Round-to-nearest-even (the IEEE 754 default) is used for f32→f16
//! rounding. f16→f32 is exact (f32 can represent every finite f16 value
//! without loss). The F16C instruction set uses the same semantics, so
//! SIMD and scalar paths produce bit-identical output for non-NaN inputs.
//!
//! NEON FP16 intrinsics (ARMv8.2-A FEAT_FP16) are deferred — they
//! stabilized in Rust 1.94, but our MSRV is 1.89.

/// Convert raw f16 bits to an f32 value. Lossless.
#[inline]
pub(crate) fn f16_bits_to_f32(h: u16) -> f32 {
    // f32 layout:  sign(1) | exp(8) | mantissa(23)
    // f16 layout:  sign(1) | exp(5) | mantissa(10)
    //
    // Sign goes straight through (shifted by 16 to hit bit 31).
    // Mantissa goes to bits 22..13 of f32 (so: shift left 13).
    // Exponent: f16-biased → f32-biased = f16_exp + (127 - 15) = f16_exp + 112.

    let sign = ((h as u32) & 0x8000) << 16;
    let exp_mant = (h as u32) & 0x7fff;

    if exp_mant == 0 {
        // ±0
        return f32::from_bits(sign);
    }

    let exp = (exp_mant >> 10) & 0x1f;
    let mant = exp_mant & 0x3ff;

    if exp == 0x1f {
        // Infinity or NaN: f32 exp becomes 0xff, mantissa shifted into f32 position.
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }

    if exp == 0 {
        // f16 subnormal: value = mant × 2^-24. f32 can represent this as a normal.
        //
        // Let p = position of MSB in mant (0-indexed from LSB), p ∈ [0, 9].
        // Then value = 2^(p-24) × (1 + mant'/2^p) where mant' = mant − 2^p.
        // In f32: biased_exp = (p - 24) + 127 = p + 103,
        //         stored_mant = mant' shifted to fit 23-bit mantissa = (mant << (23 - p)) & 0x7fffff
        //                     = ((mant << (23 - p)) with the implicit bit masked off).
        let p = 31 - mant.leading_zeros(); // p ∈ [0, 9]
        let biased_exp_f32 = p + 103;
        let stored_mant_f32 = (mant << (23 - p)) & 0x7f_ffff;
        return f32::from_bits(sign | (biased_exp_f32 << 23) | stored_mant_f32);
    }

    // Normal
    let biased_exp_f32 = exp + 112;
    f32::from_bits(sign | (biased_exp_f32 << 23) | (mant << 13))
}

/// Convert f32 to raw f16 bits. Round-to-nearest-even.
///
/// - Values beyond ±65504 saturate to ±infinity.
/// - Values below the smallest f16 subnormal (2^-24, with ties toward zero)
///   underflow to ±0.
/// - NaN is preserved as NaN; payload is packed into the high mantissa bits
///   with a guaranteed non-zero stored mantissa (quiet NaN).
#[inline]
pub(crate) fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_f32 = ((bits >> 23) & 0xff) as i32;
    let frac_f32 = bits & 0x7f_ffff;

    // Inf or NaN: exp_f32 == 255
    if exp_f32 == 0xff {
        if frac_f32 == 0 {
            return sign | 0x7c00; // ±inf
        }
        // NaN. Preserve high-order payload bits and force a non-zero stored
        // mantissa so the result is a valid NaN (not accidentally infinity).
        // Quiet bit (bit 9 of the f16 mantissa) always set.
        let payload = (frac_f32 >> 13) as u16;
        return sign | 0x7c00 | (payload | 0x0200);
    }

    // f32 zero or f32 subnormal: both underflow to f16 ±0 (f32 subnormals are
    // already below the smallest f16 subnormal 2^-24).
    if exp_f32 == 0 {
        return sign;
    }

    // Compute target f16 biased exponent.
    let target_exp = exp_f32 - 127 + 15;

    // Overflow to ±infinity.
    if target_exp >= 0x1f {
        return sign | 0x7c00;
    }

    if target_exp > 0 {
        // Normal → Normal: shift mantissa right by 13, round-to-nearest-even.
        //
        // 23-bit f32 mantissa → 10-bit f16 mantissa, so 13 bits drop.
        // Round bit = bit 12 of f32 mantissa; sticky = any bit below.
        let truncated = (frac_f32 >> 13) as u16;
        let round_bit = (frac_f32 >> 12) & 1;
        let sticky = (frac_f32 & 0xfff) != 0;
        let round_up = round_bit != 0 && (sticky || (truncated & 1) != 0);

        if !round_up {
            return sign | ((target_exp as u16) << 10) | truncated;
        }

        let new_mant = truncated + 1;
        if new_mant < 0x400 {
            return sign | ((target_exp as u16) << 10) | new_mant;
        }
        // Mantissa overflowed (was 0x3ff, became 0x400): carry into exponent,
        // drop mantissa to 0. Check for infinity again after the carry.
        let new_exp = target_exp + 1;
        if new_exp >= 0x1f {
            return sign | 0x7c00;
        }
        return sign | ((new_exp as u16) << 10);
    }

    // target_exp ≤ 0 → f16 subnormal candidate, or underflow.
    //
    // Full f32 significand including the implicit 1 bit occupies 24 bits
    // (bits 0..23). We need to shift it right by enough to align with the
    // f16 subnormal representation (10-bit mantissa, value = mant × 2^-24).
    //
    // For target_exp == 0 (boundary): result is subnormal form of smallest
    // f16 normal. shift = 14 bits.
    // For target_exp == -10: shift = 24 bits, result is 1-bit mantissa.
    // For target_exp < -10: shift > 24, result underflows.

    if target_exp < -10 {
        // Underflow (beyond even smallest subnormal's rounding range).
        // The rounded-to-nearest-even result is ±0 for all f32 values
        // smaller than (1/2 × smallest subnormal) = 2^-25.
        //
        // For values exactly equal to 2^-25 (tie), ties-to-even rounds to 0.
        // For values in (2^-25, 2^-24), we'd round up to the smallest
        // subnormal. exp_f32 for those: 2^-25 has exp_f32 = 102.
        if target_exp == -11 {
            // Value is in [2^-25, 2^-24). Check sticky for round-up.
            //
            // Full significand = (1 | frac_f32), 24 bits. Shifting right by 25
            // puts the implicit 1 at bit 0 of the round position. Round bit = 1.
            // Sticky = frac_f32 != 0. Truncated = 0.
            // Round up if (sticky || truncated is odd = false). So round up
            // iff sticky (frac_f32 != 0), giving smallest subnormal (mant=1).
            // If frac_f32 == 0 (value exactly 2^-25), round to even: truncated
            // is 0 (even), so stays at 0.
            if frac_f32 != 0 {
                return sign | 1;
            }
        }
        return sign;
    }

    // target_exp ∈ [-10, 0]: subnormal path with 14..24-bit shift.
    let shift = (14 - target_exp) as u32; // shift ∈ [14, 24]
    let full_mant = (1u32 << 23) | frac_f32; // 24-bit significand

    let round_bit = (full_mant >> (shift - 1)) & 1;
    let sticky_mask = (1u32 << (shift - 1)) - 1;
    let sticky = (full_mant & sticky_mask) != 0;
    let truncated = full_mant >> shift;
    let round_up = round_bit != 0 && (sticky || (truncated & 1) != 0);

    let result = if round_up { truncated + 1 } else { truncated };

    // Rounding can carry a subnormal all the way up to the smallest normal
    // (mantissa 0x400 == 1024). The bit pattern `sign | 0x400` is actually
    // the correct encoding of the smallest normal f16 (exp=1, mant=0), so no
    // special handling is required.
    sign | (result as u16)
}

// ---------------------------------------------------------------------------
// Slice-level f16 ↔ f32 conversion with archmage dispatch.
//
// Pattern: each tier implementation uses `#[archmage::arcane]` to mark
// itself as safely callable via its CPU-capability token. The outer
// dispatcher uses `archmage::incant!` to pick the best tier at runtime.
// No `unsafe` in the source — `#[forbid(unsafe_code)]` compatible; the
// macro-generated unsafe wrapping is internal to archmage.
//
// Tier layout (archmage suffix convention):
//   `{prefix}_scalar`  — always available
//   `{prefix}_v3`      — x86-64 V3 (Haswell+, carries F16C + AVX2)
// ---------------------------------------------------------------------------

/// Convert a slice of f16 bits into a slice of f32 values. Lossless. Uses
/// 8-lane F16C (`vcvtph2ps`) on x86-64 CPUs that have it; scalar otherwise.
pub(crate) fn f16_bits_to_f32_slice(src: &[u16], dst: &mut [f32]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "f16_bits_to_f32_slice length mismatch"
    );
    // Explicit tier list: only _v3 (x86-64 F16C) and _scalar are defined.
    // The default incant! list also includes _neon and _wasm128, which we
    // haven't implemented — NEON FP16 intrinsics require Rust 1.94 (>MSRV
    // 1.89), so on aarch64 we fall back to scalar.
    archmage::incant!(cvt_f16_to_f32(src, dst), [v3, scalar])
}

/// Convert a slice of f32 values into a slice of f16 bits with
/// round-to-nearest-even. Uses 8-lane F16C (`vcvtps2ph`) on x86-64 CPUs
/// that have it; scalar otherwise.
pub(crate) fn f32_to_f16_bits_slice(src: &[f32], dst: &mut [u16]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "f32_to_f16_bits_slice length mismatch"
    );
    archmage::incant!(cvt_f32_to_f16(src, dst), [v3, scalar])
}

// -- Scalar tier -------------------------------------------------------------

fn cvt_f16_to_f32_scalar(_tok: archmage::ScalarToken, src: &[u16], dst: &mut [f32]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f16_bits_to_f32(*s);
    }
}

fn cvt_f32_to_f16_scalar(_tok: archmage::ScalarToken, src: &[f32], dst: &mut [u16]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f32_to_f16_bits(*s);
    }
}

// -- x86-64 V3 tier (F16C, 8 lanes per conversion) ---------------------------

#[cfg(target_arch = "x86_64")]
#[archmage::arcane(import_intrinsics)]
fn cvt_f16_to_f32_v3(_tok: archmage::X64V3Token, src: &[u16], dst: &mut [f32]) {
    let n = src.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let s_chunk: &[u16; 8] = (&src[i * 8..i * 8 + 8]).try_into().unwrap();
        let d_chunk: &mut [f32; 8] = (&mut dst[i * 8..i * 8 + 8]).try_into().unwrap();
        // safe_unaligned_simd accepts any `&T: Is128BitsUnaligned` or
        // `Is256BitsUnaligned` — [u16; 8] and [f32; 8] both qualify.
        let packed = _mm_loadu_si128(s_chunk);
        let lanes = _mm256_cvtph_ps(packed);
        _mm256_storeu_ps(d_chunk, lanes);
    }
    let tail_start = chunks * 8;
    for i in tail_start..n {
        dst[i] = f16_bits_to_f32(src[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane(import_intrinsics)]
fn cvt_f32_to_f16_v3(_tok: archmage::X64V3Token, src: &[f32], dst: &mut [u16]) {
    let n = src.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let s_chunk: &[f32; 8] = (&src[i * 8..i * 8 + 8]).try_into().unwrap();
        let d_chunk: &mut [u16; 8] = (&mut dst[i * 8..i * 8 + 8]).try_into().unwrap();
        let lanes = _mm256_loadu_ps(s_chunk);
        // imm8 is a 3-bit rounding mode per Intel docs for VCVTPS2PH.
        // _MM_FROUND_TO_NEAREST_INT = 0 → IEEE 754 round-to-nearest-even.
        let packed = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(lanes);
        _mm_storeu_si128(d_chunk, packed);
    }
    let tail_start = chunks * 8;
    for i in tail_start..n {
        dst[i] = f32_to_f16_bits(src[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exhaustive: all 65536 f16 bit patterns round-trip through f32 and back.
    /// Non-NaN values must be bit-exact; NaN must stay NaN (though the bit
    /// pattern may differ as payload bits flow through f32 canonicalization).
    #[test]
    fn exhaustive_f16_f32_f16_roundtrip() {
        for bits in 0u16..=0xffff {
            let f = f16_bits_to_f32(bits);
            let back = f32_to_f16_bits(f);

            // NaN → NaN (bit pattern may canonicalize)
            let is_f16_nan = (bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0;
            if is_f16_nan {
                assert!(
                    f.is_nan(),
                    "bits {:#06x} should produce f32 NaN, got {}",
                    bits,
                    f
                );
                let back_is_nan = (back & 0x7c00) == 0x7c00 && (back & 0x03ff) != 0;
                assert!(
                    back_is_nan,
                    "bits {:#06x} round-tripped to non-NaN {:#06x}",
                    bits, back
                );
                continue;
            }

            assert_eq!(
                bits, back,
                "bit-exact roundtrip failed for {:#06x} → {} → {:#06x}",
                bits, f, back
            );
        }
    }

    /// Cross-validate f16_bits_to_f32 against the half crate for all 65536
    /// bit patterns. Any deviation (for non-NaN inputs) is a bug in ours.
    #[test]
    fn f16_bits_to_f32_matches_half_crate() {
        for bits in 0u16..=0xffff {
            let ours = f16_bits_to_f32(bits);
            let theirs = half::f16::from_bits(bits).to_f32();

            if ours.is_nan() && theirs.is_nan() {
                continue;
            }

            assert_eq!(
                ours.to_bits(),
                theirs.to_bits(),
                "bits {:#06x}: ours={} ({:#010x}), theirs={} ({:#010x})",
                bits,
                ours,
                ours.to_bits(),
                theirs,
                theirs.to_bits()
            );
        }
    }

    /// Cross-validate f32_to_f16_bits against the half crate across a large
    /// sample spanning normals, subnormals, boundaries, and infinities.
    #[test]
    fn f32_to_f16_bits_matches_half_crate_sampled() {
        // Sample every f16 value reinterpreted as f32 (exact), plus some
        // offsets that exercise rounding boundaries.
        for bits in 0u16..=0xffff {
            let center = f16_bits_to_f32(bits);
            if !center.is_finite() {
                continue;
            }
            // Generate nearby f32 values: center, center + smallest f32 ULP,
            // center + half of the next f16 step up, etc.
            let next = f16_bits_to_f32(bits.wrapping_add(1));
            let midpoint = center + (next - center) * 0.5;

            for v in [center, midpoint, center.next_up(), center.next_down()] {
                if v.is_nan() {
                    continue;
                }
                let ours = f32_to_f16_bits(v);
                let theirs = half::f16::from_f32(v).to_bits();
                assert_eq!(
                    ours,
                    theirs,
                    "f32 {} ({:#010x}): ours={:#06x}, theirs={:#06x}",
                    v,
                    v.to_bits(),
                    ours,
                    theirs
                );
            }
        }
    }

    /// Exhaustive cross-check: for all 65536 f16 bit patterns, the scalar
    /// and SIMD paths (when available) produce identical f32 output.
    /// Exercises F16C when the CPU has it.
    #[test]
    fn slice_f16_to_f32_simd_matches_scalar_exhaustive() {
        let bits: Vec<u16> = (0u16..=0xffff).collect();
        let mut via_slice = vec![0.0f32; bits.len()];
        let mut via_scalar = vec![0.0f32; bits.len()];

        f16_bits_to_f32_slice(&bits, &mut via_slice);
        for (i, &b) in bits.iter().enumerate() {
            via_scalar[i] = f16_bits_to_f32(b);
        }

        for i in 0..bits.len() {
            let a = via_slice[i];
            let b = via_scalar[i];
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "f16 bits {:#06x}: slice={} (bits {:#010x}), scalar={} (bits {:#010x})",
                bits[i],
                a,
                a.to_bits(),
                b,
                b.to_bits()
            );
        }
    }

    /// Cross-check f32 → f16 across sampled f32 values: the slice (SIMD)
    /// path and scalar path must agree bit-exactly (modulo NaN canonicalization).
    #[test]
    fn slice_f32_to_f16_simd_matches_scalar_sampled() {
        // Build a diverse sample: every f16 value as its f32, nearby f32
        // neighbors, and the rounding midpoints.
        let mut samples: Vec<f32> = Vec::new();
        for b in 0u16..=0xffff {
            let c = f16_bits_to_f32(b);
            samples.push(c);
            if c.is_finite() {
                samples.push(c.next_up());
                samples.push(c.next_down());
                let next = f16_bits_to_f32(b.wrapping_add(1));
                if next.is_finite() {
                    samples.push(c + (next - c) * 0.5);
                }
            }
        }
        let mut via_slice = vec![0u16; samples.len()];
        let mut via_scalar = vec![0u16; samples.len()];

        f32_to_f16_bits_slice(&samples, &mut via_slice);
        for (i, &v) in samples.iter().enumerate() {
            via_scalar[i] = f32_to_f16_bits(v);
        }

        for i in 0..samples.len() {
            if samples[i].is_nan() {
                // Both paths should produce NaN. Bit patterns may differ.
                let a_nan = (via_slice[i] & 0x7c00) == 0x7c00 && (via_slice[i] & 0x03ff) != 0;
                let b_nan = (via_scalar[i] & 0x7c00) == 0x7c00 && (via_scalar[i] & 0x03ff) != 0;
                assert!(a_nan && b_nan, "NaN input lost NaN-ness");
                continue;
            }
            assert_eq!(
                via_slice[i],
                via_scalar[i],
                "f32 {} ({:#010x}): slice={:#06x}, scalar={:#06x}",
                samples[i],
                samples[i].to_bits(),
                via_slice[i],
                via_scalar[i],
            );
        }
    }

    /// Specific boundary cases worth documenting in the test output.
    #[test]
    fn f32_to_f16_boundary_cases() {
        // ±0
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);

        // ±1.0
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-1.0), 0xbc00);

        // Smallest f16 normal: 2^-14
        assert_eq!(f32_to_f16_bits(2.0f32.powi(-14)), 0x0400);

        // Smallest f16 subnormal: 2^-24
        assert_eq!(f32_to_f16_bits(2.0f32.powi(-24)), 0x0001);

        // Largest f16: 65504
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);

        // Overflow to infinity: 65520 rounds to inf (IEEE 754 rule: ties to
        // nearest — 65520 is exactly halfway between 65504 and infinity, and
        // "infinity" is chosen because it's effectively the even value
        // at that magnitude per the rounding rule).
        assert_eq!(f32_to_f16_bits(65520.0), 0x7c00);

        // Way beyond range
        assert_eq!(f32_to_f16_bits(1e9), 0x7c00);
        assert_eq!(f32_to_f16_bits(-1e9), 0xfc00);

        // Infinity / -infinity
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);

        // NaN → NaN
        assert!((f32_to_f16_bits(f32::NAN) & 0x7c00) == 0x7c00);
        assert!((f32_to_f16_bits(f32::NAN) & 0x03ff) != 0);
    }

    #[test]
    fn f16_bits_to_f32_boundary_cases() {
        // ±0
        assert_eq!(f16_bits_to_f32(0x0000).to_bits(), 0x0000_0000);
        assert_eq!(f16_bits_to_f32(0x8000).to_bits(), 0x8000_0000);

        // ±1.0
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0xbc00), -1.0);

        // Smallest subnormal f16 = 2^-24
        let v = f16_bits_to_f32(0x0001);
        assert_eq!(v, 2.0f32.powi(-24));

        // Largest subnormal f16 = (2^10 - 1) × 2^-24
        let v = f16_bits_to_f32(0x03ff);
        let expected = 1023.0 * 2.0f32.powi(-24);
        assert!((v - expected).abs() < 1e-30);

        // Smallest normal f16 = 2^-14
        let v = f16_bits_to_f32(0x0400);
        assert_eq!(v, 2.0f32.powi(-14));

        // Max normal f16 = 65504
        assert_eq!(f16_bits_to_f32(0x7bff), 65504.0);

        // ±infinity
        assert!(f16_bits_to_f32(0x7c00).is_infinite());
        assert!(f16_bits_to_f32(0x7c00).is_sign_positive());
        assert!(f16_bits_to_f32(0xfc00).is_infinite());
        assert!(f16_bits_to_f32(0xfc00).is_sign_negative());

        // NaN
        assert!(f16_bits_to_f32(0x7e00).is_nan());
        assert!(f16_bits_to_f32(0xffff).is_nan());
    }
}
