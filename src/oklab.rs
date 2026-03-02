#![allow(clippy::excessive_precision)]
//! Oklab color space conversion constants and scalar reference functions.
//!
//! Provides the building blocks for planar Oklab conversion:
//! - Const matrices for XYZ↔LMS and LMS^(1/3)↔Oklab (from Björn Ottosson)
//! - Combined RGB↔LMS matrix computation for any supported primaries
//! - `fast_cbrt`: fast cube root (~22 bits precision)
//! - Scalar `rgb_to_oklab` / `oklab_to_rgb` reference implementations

use crate::ColorPrimaries;
use crate::gamut::{GamutMatrix, mat3_mul};

// ---------------------------------------------------------------------------
// Oklab matrices (Björn Ottosson, 2020)
// ---------------------------------------------------------------------------

/// M1 step 1: CIE XYZ → LMS (Hunt-Pointer-Estevez variant, Ottosson 2020).
pub const LMS_FROM_XYZ: GamutMatrix = [
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.6338517070],
];

/// Inverse of M1: LMS → CIE XYZ.
pub const XYZ_FROM_LMS: GamutMatrix = [
    [1.2270138511, -0.5577999807, 0.2812561490],
    [-0.0405801784, 1.1122568696, -0.0716766787],
    [-0.0763812845, -0.4214819784, 1.5861632204],
];

/// M2: LMS^(1/3) → Oklab [L, a, b]. Universal, primaries-independent.
pub const OKLAB_FROM_LMS_CBRT: GamutMatrix = [
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
];

/// Inverse of M2: Oklab [L, a, b] → LMS^(1/3).
pub const LMS_CBRT_FROM_OKLAB: GamutMatrix = [
    [1.0, 0.3963377774, 0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
];

// ---------------------------------------------------------------------------
// Combined matrices
// ---------------------------------------------------------------------------

/// Compute the combined linear RGB → LMS matrix for a given primaries set.
///
/// This is `LMS_FROM_XYZ × RGB_TO_XYZ`, so you can go straight from
/// linear RGB to LMS without an intermediate XYZ buffer.
pub fn rgb_to_lms_matrix(primaries: ColorPrimaries) -> Option<GamutMatrix> {
    let to_xyz = primaries.to_xyz_matrix()?;
    Some(mat3_mul(&LMS_FROM_XYZ, to_xyz))
}

/// Compute the combined LMS → linear RGB matrix for a given primaries set.
///
/// This is `XYZ_TO_RGB × XYZ_FROM_LMS`, so you can go straight from
/// LMS to linear RGB without an intermediate XYZ buffer.
pub fn lms_to_rgb_matrix(primaries: ColorPrimaries) -> Option<GamutMatrix> {
    let from_xyz = primaries.from_xyz_matrix()?;
    Some(mat3_mul(from_xyz, &XYZ_FROM_LMS))
}

// ---------------------------------------------------------------------------
// Fast cube root
// ---------------------------------------------------------------------------

/// Fast approximate cube root with ~22 bits of precision.
///
/// Uses bit-manipulation for an initial estimate followed by two
/// Newton-Raphson iterations. Handles negative inputs and zero correctly.
pub fn fast_cbrt(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    let sign = x.signum();
    let x = x.abs();

    // Initial estimate via IEEE 754 bit trick: cbrt(x) ≈ 2^(log2(x)/3)
    let bits = x.to_bits();
    let estimate = f32::from_bits((bits / 3) + (0x2a51_7d48)); // magic constant

    // Two Newton-Raphson iterations for cbrt: y = y - (y³ - x) / (3y²)
    // Rearranged: y = (2y + x/y²) / 3
    let mut y = estimate;
    y = (2.0 * y + x / (y * y)) / 3.0;
    y = (2.0 * y + x / (y * y)) / 3.0;

    sign * y
}

// ---------------------------------------------------------------------------
// Scalar reference conversions
// ---------------------------------------------------------------------------

/// Scalar reference: linear RGB → Oklab \[L, a, b\].
///
/// `m1` is the combined RGB→LMS matrix from [`rgb_to_lms_matrix`].
/// All inputs should be in linear light (apply EOTF first).
pub fn rgb_to_oklab(r: f32, g: f32, b: f32, m1: &GamutMatrix) -> [f32; 3] {
    // Step 1: linear RGB → LMS
    let l = m1[0][0] * r + m1[0][1] * g + m1[0][2] * b;
    let m = m1[1][0] * r + m1[1][1] * g + m1[1][2] * b;
    let s = m1[2][0] * r + m1[2][1] * g + m1[2][2] * b;

    // Step 2: cube root (non-linearity)
    let l_ = fast_cbrt(l);
    let m_ = fast_cbrt(m);
    let s_ = fast_cbrt(s);

    // Step 3: LMS^(1/3) → Oklab via M2
    let ok_l = OKLAB_FROM_LMS_CBRT[0][0] * l_
        + OKLAB_FROM_LMS_CBRT[0][1] * m_
        + OKLAB_FROM_LMS_CBRT[0][2] * s_;
    let ok_a = OKLAB_FROM_LMS_CBRT[1][0] * l_
        + OKLAB_FROM_LMS_CBRT[1][1] * m_
        + OKLAB_FROM_LMS_CBRT[1][2] * s_;
    let ok_b = OKLAB_FROM_LMS_CBRT[2][0] * l_
        + OKLAB_FROM_LMS_CBRT[2][1] * m_
        + OKLAB_FROM_LMS_CBRT[2][2] * s_;

    [ok_l, ok_a, ok_b]
}

/// Scalar reference: Oklab \[L, a, b\] → linear RGB.
///
/// `m1_inv` is the combined LMS→RGB matrix from [`lms_to_rgb_matrix`].
pub fn oklab_to_rgb(l: f32, a: f32, b: f32, m1_inv: &GamutMatrix) -> [f32; 3] {
    // Step 1: Oklab → LMS^(1/3) via inverse M2
    let l_ = LMS_CBRT_FROM_OKLAB[0][0] * l
        + LMS_CBRT_FROM_OKLAB[0][1] * a
        + LMS_CBRT_FROM_OKLAB[0][2] * b;
    let m_ = LMS_CBRT_FROM_OKLAB[1][0] * l
        + LMS_CBRT_FROM_OKLAB[1][1] * a
        + LMS_CBRT_FROM_OKLAB[1][2] * b;
    let s_ = LMS_CBRT_FROM_OKLAB[2][0] * l
        + LMS_CBRT_FROM_OKLAB[2][1] * a
        + LMS_CBRT_FROM_OKLAB[2][2] * b;

    // Step 2: cube (inverse of cube root)
    let lms_l = l_ * l_ * l_;
    let lms_m = m_ * m_ * m_;
    let lms_s = s_ * s_ * s_;

    // Step 3: LMS → linear RGB via inverse M1
    let r = m1_inv[0][0] * lms_l + m1_inv[0][1] * lms_m + m1_inv[0][2] * lms_s;
    let g = m1_inv[1][0] * lms_l + m1_inv[1][1] * lms_m + m1_inv[1][2] * lms_s;
    let b = m1_inv[2][0] * lms_l + m1_inv[2][1] * lms_m + m1_inv[2][2] * lms_s;

    [r, g, b]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_cbrt_accuracy() {
        let test_values: [f32; 10] = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 8.0, 27.0, 100.0];
        for &x in &test_values {
            let expected = x.cbrt();
            let got = fast_cbrt(x);
            let err = (got - expected).abs();
            assert!(
                err < 1e-5 || err / expected.max(1e-10) < 1e-5,
                "fast_cbrt({x}) = {got}, expected {expected}, err = {err}"
            );
        }
    }

    #[test]
    fn fast_cbrt_negative() {
        let got = fast_cbrt(-8.0);
        assert!((got - (-2.0)).abs() < 1e-5, "fast_cbrt(-8) = {got}");
    }

    #[test]
    fn oklab_roundtrip_bt709() {
        let m1 = rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();
        let m1_inv = lms_to_rgb_matrix(ColorPrimaries::Bt709).unwrap();

        let test_colors = [
            [0.5, 0.3, 0.8],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.18, 0.18, 0.18], // 18% gray
        ];

        for rgb in &test_colors {
            let [l, a, b] = rgb_to_oklab(rgb[0], rgb[1], rgb[2], &m1);
            let [r2, g2, b2] = oklab_to_rgb(l, a, b, &m1_inv);
            for c in 0..3 {
                let err = (rgb[c] - [r2, g2, b2][c]).abs();
                assert!(err < 1e-4, "roundtrip error for {rgb:?} channel {c}: {err}");
            }
        }
    }

    #[test]
    fn oklab_white_point() {
        let m1 = rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();
        let [l, a, b] = rgb_to_oklab(1.0, 1.0, 1.0, &m1);
        // Tolerance is 5e-4 because the combined matrix (mat3_mul of two f32 matrices)
        // accumulates rounding beyond 1e-4. The reference Oklab uses f64 matrices.
        assert!((l - 1.0).abs() < 5e-4, "white L should be ~1.0, got {l}");
        assert!(a.abs() < 5e-4, "white a should be ~0.0, got {a}");
        assert!(b.abs() < 5e-4, "white b should be ~0.0, got {b}");
    }

    #[test]
    fn oklab_black_point() {
        let m1 = rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();
        let [l, a, b] = rgb_to_oklab(0.0, 0.0, 0.0, &m1);
        assert!(l.abs() < 1e-6, "black L should be ~0.0, got {l}");
        assert!(a.abs() < 1e-6, "black a should be ~0.0, got {a}");
        assert!(b.abs() < 1e-6, "black b should be ~0.0, got {b}");
    }

    #[test]
    fn oklab_roundtrip_bt2020() {
        let m1 = rgb_to_lms_matrix(ColorPrimaries::Bt2020).unwrap();
        let m1_inv = lms_to_rgb_matrix(ColorPrimaries::Bt2020).unwrap();

        let rgb = [0.4, 0.6, 0.2];
        let [l, a, b] = rgb_to_oklab(rgb[0], rgb[1], rgb[2], &m1);
        let [r2, g2, b2] = oklab_to_rgb(l, a, b, &m1_inv);
        for c in 0..3 {
            let err = (rgb[c] - [r2, g2, b2][c]).abs();
            assert!(err < 1e-4, "BT.2020 roundtrip error channel {c}: {err}");
        }
    }

    #[test]
    fn combined_matrices_available() {
        assert!(rgb_to_lms_matrix(ColorPrimaries::Bt709).is_some());
        assert!(rgb_to_lms_matrix(ColorPrimaries::DisplayP3).is_some());
        assert!(rgb_to_lms_matrix(ColorPrimaries::Bt2020).is_some());
        assert!(rgb_to_lms_matrix(ColorPrimaries::Unknown).is_none());

        assert!(lms_to_rgb_matrix(ColorPrimaries::Bt709).is_some());
        assert!(lms_to_rgb_matrix(ColorPrimaries::Unknown).is_none());
    }
}
