//! Fast Display P3 ↔ sRGB conversion.
//!
//! Both color spaces share the same transfer function (sRGB TRC) and white
//! point (D65), so conversion is: linearize → 3×3 matrix → re-encode.
//!
//! This module hardcodes the exact conversion matrices and uses `linear-srgb`'s
//! rational polynomial TRC (no lookup tables). Compared to a full CMS path:
//!
//! - **No profile parsing** — matrices are compile-time constants
//! - **No table allocation** — polynomial TRC uses ~0 bytes vs ~200KB for LUTs
//! - **Higher accuracy** — continuous polynomial vs 65536-entry table quantization
//! - **Extended range** — negative and >1.0 values preserved (sign-preserving TRC)
//!
//! # Supported conversions
//!
//! - [`p3_to_srgb_f32`] — f32 RGB, extended range preserved
//! - [`srgb_to_p3_f32`] — f32 RGB, extended range preserved

/// Display P3 linear RGB → sRGB linear RGB.
///
/// Derived from the chromaticity coordinates of Display P3 (DCI-P3 primaries
/// with D65 white point) and sRGB (BT.709 primaries, D65 white point):
/// `inv(M_srgb_to_xyz) × M_p3_to_xyz`
///
/// Both spaces share D65 so no chromatic adaptation is needed.
const P3_TO_SRGB: [[f32; 3]; 3] = [
    [1.2249401763, -0.2249401763, 0.0],
    [-0.0420569547, 1.0420569547, 0.0],
    [-0.0196375546, -0.0786360456, 1.0982736001],
];

/// sRGB linear RGB → Display P3 linear RGB.
///
/// Inverse of [`P3_TO_SRGB`].
const SRGB_TO_P3: [[f32; 3]; 3] = [
    [0.8224619687, 0.1775380313, 0.0],
    [0.0331941989, 0.9668058011, 0.0],
    [0.0170826307, 0.0723974407, 0.9105199286],
];

/// Linearize an sRGB/P3 encoded value using the sRGB transfer function.
///
/// Sign-preserving for extended range: `sign(v) × linearize(|v|)`.
#[inline(always)]
fn linearize(v: f32) -> f32 {
    if v >= 0.0 {
        linear_srgb::precise::srgb_to_linear_extended(v)
    } else {
        -linear_srgb::precise::srgb_to_linear_extended(-v)
    }
}

/// Encode a linear value using the sRGB transfer function.
///
/// Sign-preserving for extended range: `sign(v) × encode(|v|)`.
#[inline(always)]
fn encode(v: f32) -> f32 {
    if v >= 0.0 {
        linear_srgb::precise::linear_to_srgb_extended(v)
    } else {
        -linear_srgb::precise::linear_to_srgb_extended(-v)
    }
}

/// Apply a 3×3 matrix to an RGB triple.
#[inline(always)]
fn mat3x3(m: &[[f32; 3]; 3], r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        m[0][0].mul_add(r, m[0][1].mul_add(g, m[0][2] * b)),
        m[1][0].mul_add(r, m[1][1].mul_add(g, m[1][2] * b)),
        m[2][0].mul_add(r, m[2][1].mul_add(g, m[2][2] * b)),
    )
}

// ---------------------------------------------------------------------------
// Fused SIMD kernel: linearize + matrix + encode in one pass
// ---------------------------------------------------------------------------

use archmage::prelude::*;
use linear_srgb::tokens::x8 as trc_x8;
use magetypes::simd::f32x8 as mt_f32x8;

/// Fused linearize → matrix → encode for 8 RGB pixels (24 f32s).
///
/// Processes 8 pixels at a time using AVX2+FMA SIMD for both the rational
/// polynomial TRC and the matrix multiply. Single pass over data.
#[rite]
fn fused_8px_rgb(token: X64V3Token, m: &[[f32; 3]; 3], data: &mut [f32]) {
    // Deinterleave 8 stride-3 pixels into per-channel [f32; 8]
    let mut r_arr = [0.0f32; 8];
    let mut g_arr = [0.0f32; 8];
    let mut b_arr = [0.0f32; 8];
    for i in 0..8 {
        r_arr[i] = data[i * 3];
        g_arr[i] = data[i * 3 + 1];
        b_arr[i] = data[i * 3 + 2];
    }

    // Linearize (SIMD rational polynomial, 8 values at once)
    let r_lin = trc_x8::srgb_to_linear_v3(token, r_arr);
    let g_lin = trc_x8::srgb_to_linear_v3(token, g_arr);
    let b_lin = trc_x8::srgb_to_linear_v3(token, b_arr);

    // Matrix multiply (SIMD FMA, 8 pixels at once per channel)
    let rl = mt_f32x8::from_array(token, r_lin);
    let gl = mt_f32x8::from_array(token, g_lin);
    let bl = mt_f32x8::from_array(token, b_lin);

    let or = mt_f32x8::splat(token, m[0][0]).mul_add(
        rl,
        mt_f32x8::splat(token, m[0][1]).mul_add(gl, mt_f32x8::splat(token, m[0][2]) * bl),
    );
    let og = mt_f32x8::splat(token, m[1][0]).mul_add(
        rl,
        mt_f32x8::splat(token, m[1][1]).mul_add(gl, mt_f32x8::splat(token, m[1][2]) * bl),
    );
    let ob = mt_f32x8::splat(token, m[2][0]).mul_add(
        rl,
        mt_f32x8::splat(token, m[2][1]).mul_add(gl, mt_f32x8::splat(token, m[2][2]) * bl),
    );

    // Encode (SIMD rational polynomial, 8 values at once)
    let r_out = trc_x8::linear_to_srgb_v3(token, or.to_array());
    let g_out = trc_x8::linear_to_srgb_v3(token, og.to_array());
    let b_out = trc_x8::linear_to_srgb_v3(token, ob.to_array());

    // Interleave back to stride-3
    for i in 0..8 {
        data[i * 3] = r_out[i];
        data[i * 3 + 1] = g_out[i];
        data[i * 3 + 2] = b_out[i];
    }
}

/// Fused linearize → matrix → encode for 8 RGBA pixels (32 f32s).
#[rite]
fn fused_8px_rgba(token: X64V3Token, m: &[[f32; 3]; 3], data: &mut [f32]) {
    let mut r_arr = [0.0f32; 8];
    let mut g_arr = [0.0f32; 8];
    let mut b_arr = [0.0f32; 8];
    for i in 0..8 {
        r_arr[i] = data[i * 4];
        g_arr[i] = data[i * 4 + 1];
        b_arr[i] = data[i * 4 + 2];
    }

    let r_lin = trc_x8::srgb_to_linear_v3(token, r_arr);
    let g_lin = trc_x8::srgb_to_linear_v3(token, g_arr);
    let b_lin = trc_x8::srgb_to_linear_v3(token, b_arr);

    let rl = mt_f32x8::from_array(token, r_lin);
    let gl = mt_f32x8::from_array(token, g_lin);
    let bl = mt_f32x8::from_array(token, b_lin);

    let or = mt_f32x8::splat(token, m[0][0]).mul_add(
        rl,
        mt_f32x8::splat(token, m[0][1]).mul_add(gl, mt_f32x8::splat(token, m[0][2]) * bl),
    );
    let og = mt_f32x8::splat(token, m[1][0]).mul_add(
        rl,
        mt_f32x8::splat(token, m[1][1]).mul_add(gl, mt_f32x8::splat(token, m[1][2]) * bl),
    );
    let ob = mt_f32x8::splat(token, m[2][0]).mul_add(
        rl,
        mt_f32x8::splat(token, m[2][1]).mul_add(gl, mt_f32x8::splat(token, m[2][2]) * bl),
    );

    let r_out = trc_x8::linear_to_srgb_v3(token, or.to_array());
    let g_out = trc_x8::linear_to_srgb_v3(token, og.to_array());
    let b_out = trc_x8::linear_to_srgb_v3(token, ob.to_array());

    for i in 0..8 {
        data[i * 4] = r_out[i];
        data[i * 4 + 1] = g_out[i];
        data[i * 4 + 2] = b_out[i];
        // alpha unchanged
    }
}

/// Fused SIMD dispatch: 8-pixel SIMD for bulk, scalar remainder.
#[arcane]
fn convert_rgb_fused_v3(token: X64V3Token, m: &[[f32; 3]; 3], data: &mut [f32]) {
    let chunks = data.len() / 24;
    let bulk = chunks * 24;

    for chunk_start in (0..bulk).step_by(24) {
        fused_8px_rgb(token, m, &mut data[chunk_start..chunk_start + 24]);
    }

    for pixel in data[bulk..].chunks_exact_mut(3) {
        let r = linear_srgb::tf::srgb_to_linear(pixel[0]);
        let g = linear_srgb::tf::srgb_to_linear(pixel[1]);
        let b = linear_srgb::tf::srgb_to_linear(pixel[2]);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        pixel[0] = linear_srgb::tf::linear_to_srgb(nr);
        pixel[1] = linear_srgb::tf::linear_to_srgb(ng);
        pixel[2] = linear_srgb::tf::linear_to_srgb(nb);
    }
}

#[arcane]
fn convert_rgba_fused_v3(token: X64V3Token, m: &[[f32; 3]; 3], data: &mut [f32]) {
    let chunks = data.len() / 32;
    let bulk = chunks * 32;

    for chunk_start in (0..bulk).step_by(32) {
        fused_8px_rgba(token, m, &mut data[chunk_start..chunk_start + 32]);
    }

    for pixel in data[bulk..].chunks_exact_mut(4) {
        let r = linear_srgb::tf::srgb_to_linear(pixel[0]);
        let g = linear_srgb::tf::srgb_to_linear(pixel[1]);
        let b = linear_srgb::tf::srgb_to_linear(pixel[2]);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        pixel[0] = linear_srgb::tf::linear_to_srgb(nr);
        pixel[1] = linear_srgb::tf::linear_to_srgb(ng);
        pixel[2] = linear_srgb::tf::linear_to_srgb(nb);
    }
}

// ---------------------------------------------------------------------------
// Scalar fallbacks (for platforms without AVX2+FMA)
// ---------------------------------------------------------------------------

fn convert_rgb_fused_scalar(_token: ScalarToken, m: &[[f32; 3]; 3], data: &mut [f32]) {
    for pixel in data.chunks_exact_mut(3) {
        let r = linear_srgb::tf::srgb_to_linear(pixel[0]);
        let g = linear_srgb::tf::srgb_to_linear(pixel[1]);
        let b = linear_srgb::tf::srgb_to_linear(pixel[2]);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        pixel[0] = linear_srgb::tf::linear_to_srgb(nr);
        pixel[1] = linear_srgb::tf::linear_to_srgb(ng);
        pixel[2] = linear_srgb::tf::linear_to_srgb(nb);
    }
}

fn convert_rgba_fused_scalar(_token: ScalarToken, m: &[[f32; 3]; 3], data: &mut [f32]) {
    for pixel in data.chunks_exact_mut(4) {
        let r = linear_srgb::tf::srgb_to_linear(pixel[0]);
        let g = linear_srgb::tf::srgb_to_linear(pixel[1]);
        let b = linear_srgb::tf::srgb_to_linear(pixel[2]);
        let (nr, ng, nb) = mat3x3(m, r, g, b);
        pixel[0] = linear_srgb::tf::linear_to_srgb(nr);
        pixel[1] = linear_srgb::tf::linear_to_srgb(ng);
        pixel[2] = linear_srgb::tf::linear_to_srgb(nb);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Convert Display P3 → sRGB, f32 RGB pixels in-place.
///
/// Fused single-pass SIMD: linearize + matrix + encode for 8 pixels at a time.
/// Uses rational polynomial TRC (~5e-7 max error vs f64, no LUT quantization).
/// Values are clamped to [0,1].
///
/// `data` is a slice of `[R, G, B, R, G, B, ...]` f32 values.
/// Length must be a multiple of 3.
pub fn p3_to_srgb_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_fused(&P3_TO_SRGB, data));
}

/// Convert sRGB → Display P3, f32 RGB pixels in-place.
///
/// Fused single-pass SIMD with rational polynomial TRC.
/// Values are clamped to [0,1].
pub fn srgb_to_p3_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    incant!(convert_rgb_fused(&SRGB_TO_P3, data));
}

/// Convert Display P3 → sRGB, f32 RGBA pixels in-place.
///
/// Alpha channel is passed through unchanged.
pub fn p3_to_srgb_f32_rgba(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 4, 0);
    incant!(convert_rgba_fused(&P3_TO_SRGB, data));
}

/// Convert sRGB → Display P3, f32 RGBA pixels in-place.
///
/// Alpha channel is passed through unchanged.
pub fn srgb_to_p3_f32_rgba(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 4, 0);
    incant!(convert_rgba_fused(&SRGB_TO_P3, data));
}

/// Convert Display P3 → sRGB, f32 RGB pixels in-place (extended range).
///
/// Uses sign-preserving scalar TRC — values outside [0,1] are preserved.
/// Slower than the clamped variant due to scalar `powf` per channel.
pub fn p3_to_srgb_f32_extended(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    for pixel in data.chunks_exact_mut(3) {
        let r = linearize(pixel[0]);
        let g = linearize(pixel[1]);
        let b = linearize(pixel[2]);
        let (sr, sg, sb) = mat3x3(&P3_TO_SRGB, r, g, b);
        pixel[0] = encode(sr);
        pixel[1] = encode(sg);
        pixel[2] = encode(sb);
    }
}

/// Convert sRGB → Display P3, f32 RGB pixels in-place (extended range).
///
/// Uses sign-preserving scalar TRC — values outside [0,1] are preserved.
pub fn srgb_to_p3_f32_extended(data: &mut [f32]) {
    debug_assert_eq!(data.len() % 3, 0);
    for pixel in data.chunks_exact_mut(3) {
        let r = linearize(pixel[0]);
        let g = linearize(pixel[1]);
        let b = linearize(pixel[2]);
        let (pr, pg, pb) = mat3x3(&SRGB_TO_P3, r, g, b);
        pixel[0] = encode(pr);
        pixel[1] = encode(pg);
        pixel[2] = encode(pb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_roundtrips() {
        let mut px = [1.0_f32, 1.0, 1.0];
        p3_to_srgb_f32(&mut px);
        // White in P3 is white in sRGB (same white point)
        for c in &px {
            assert!((c - 1.0).abs() < 1e-5, "white should map to white: {px:?}");
        }

        let mut px = [1.0_f32, 1.0, 1.0];
        srgb_to_p3_f32(&mut px);
        for c in &px {
            assert!((c - 1.0).abs() < 1e-5, "white should map to white: {px:?}");
        }
    }

    #[test]
    fn black_roundtrips() {
        let mut px = [0.0_f32, 0.0, 0.0];
        p3_to_srgb_f32(&mut px);
        for c in &px {
            assert!(c.abs() < 1e-7, "black should map to black: {px:?}");
        }
    }

    #[test]
    fn roundtrip_p3_srgb_p3() {
        // A mid-gray P3 pixel should survive roundtrip with minimal error.
        let original = [0.5_f32, 0.5, 0.5];
        let mut px = original;
        p3_to_srgb_f32(&mut px);
        srgb_to_p3_f32(&mut px);
        for (i, (a, b)) in original.iter().zip(px.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "channel {i}: roundtrip error {a} → {b}"
            );
        }
    }

    #[test]
    fn saturated_p3_green_goes_negative_in_srgb_extended() {
        // Pure P3 green (0, 1, 0) is outside sRGB gamut.
        // The sRGB red channel should go negative (extended range).
        let mut px = [0.0_f32, 1.0, 0.0];
        p3_to_srgb_f32_extended(&mut px);
        assert!(
            px[0] < 0.0,
            "P3 green should have negative sRGB red: {px:?}"
        );
        assert!(px[1] > 1.0, "P3 green should have >1.0 sRGB green: {px:?}");
    }

    #[test]
    fn saturated_p3_green_clamped_in_srgb() {
        // Clamped variant: out-of-gamut values clamp to [0,1]
        let mut px = [0.0_f32, 1.0, 0.0];
        p3_to_srgb_f32(&mut px);
        assert!(px[0] >= 0.0, "clamped should not go negative: {px:?}");
        assert!(px[1] <= 1.0, "clamped should not exceed 1.0: {px:?}");
    }

    #[test]
    fn alpha_passthrough() {
        let mut px = [0.5_f32, 0.5, 0.5, 0.7];
        p3_to_srgb_f32_rgba(&mut px);
        assert!(
            (px[3] - 0.7).abs() < f32::EPSILON,
            "alpha should be unchanged"
        );
    }

    #[test]
    fn srgb_subset_of_p3() {
        // Any valid sRGB color converted to P3 should stay in [0, 1].
        for r in (0..=255).step_by(17) {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(17) {
                    let mut px = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0];
                    srgb_to_p3_f32(&mut px);
                    for (i, c) in px.iter().enumerate() {
                        assert!(
                            *c >= -1e-5 && *c <= 1.0 + 1e-5,
                            "sRGB ({r},{g},{b}) channel {i} out of P3 gamut: {c}"
                        );
                    }
                }
            }
        }
    }
}
