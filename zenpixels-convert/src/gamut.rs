#![allow(clippy::excessive_precision)]
//! Color gamut conversion matrices for BT.709, BT.2020, and Display P3.
//!
//! All matrices operate in **linear light** — apply EOTF first, then gamut
//! matrix, then OETF. The matrices are computed from the CIE 1931 xy
//! chromaticity coordinates of each primary set using D65 white point.
//!
//! These are exact, standards-derived values (not fitted or approximated).

use crate::ColorPrimaries;

/// A 3×3 row-major matrix for linear RGB ↔ linear RGB gamut conversion.
pub type GamutMatrix = [[f32; 3]; 3];

// ---------------------------------------------------------------------------
// RGB ↔ CIE XYZ (D65 white point) matrices
// ---------------------------------------------------------------------------

/// BT.709/sRGB → CIE XYZ (D65). Derived from CIE 1931 xy chromaticities.
pub(crate) const BT709_TO_XYZ: GamutMatrix = [
    [0.4123907993, 0.3575843394, 0.1804807884],
    [0.2126390059, 0.7151686788, 0.0721923154],
    [0.0193308187, 0.1191947798, 0.9505321522],
];

/// CIE XYZ (D65) → BT.709/sRGB.
pub(crate) const XYZ_TO_BT709: GamutMatrix = [
    [3.2409699419, -1.5373831776, -0.4986107603],
    [-0.9692436363, 1.8759675015, 0.0415550574],
    [0.0556300797, -0.2039769589, 1.0569715142],
];

/// Display P3 → CIE XYZ (D65). Same white point as sRGB, wider primaries.
pub(crate) const DISPLAY_P3_TO_XYZ: GamutMatrix = [
    [0.4865709486, 0.2656676932, 0.1982172852],
    [0.2289745641, 0.6917385218, 0.0792869141],
    [0.0000000000, 0.0451133819, 1.0439443689],
];

/// CIE XYZ (D65) → Display P3.
pub(crate) const XYZ_TO_DISPLAY_P3: GamutMatrix = [
    [2.4934969119, -0.9313836179, -0.4027107845],
    [-0.8294889696, 1.7626640603, 0.0236246858],
    [0.0358458302, -0.0761723893, 0.9568845240],
];

/// BT.2020 → CIE XYZ (D65).
pub(crate) const BT2020_TO_XYZ: GamutMatrix = [
    [0.6369580484, 0.1446169036, 0.1688809752],
    [0.2627002120, 0.6779980715, 0.0593017165],
    [0.0000000000, 0.0280726930, 1.0609850578],
];

/// CIE XYZ (D65) → BT.2020.
pub(crate) const XYZ_TO_BT2020: GamutMatrix = [
    [1.7166511880, -0.3556707838, -0.2533662814],
    [-0.6666843518, 1.6164812366, 0.0157685458],
    [0.0176398574, -0.0427706133, 0.9421031212],
];

/// Multiply two 3×3 row-major matrices: `C = A × B`.
pub fn mat3_mul(a: &GamutMatrix, b: &GamutMatrix) -> GamutMatrix {
    let mut c = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

/// Apply a 3×3 gamut matrix to a single linear RGB pixel (in-place).
#[inline]
pub fn apply_matrix_f32(rgb: &mut [f32; 3], m: &GamutMatrix) {
    let [r, g, b] = *rgb;
    rgb[0] = m[0][0] * r + m[0][1] * g + m[0][2] * b;
    rgb[1] = m[1][0] * r + m[1][1] * g + m[1][2] * b;
    rgb[2] = m[2][0] * r + m[2][1] * g + m[2][2] * b;
}

/// Apply a gamut matrix to a row of linear F32 RGB pixels.
///
/// `data` is `&mut [f32]` with `width * 3` elements (RGB, no alpha).
/// For RGBA data, call [`apply_matrix_row_rgba_f32`] instead.
pub fn apply_matrix_row_f32(data: &mut [f32], width: usize, m: &GamutMatrix) {
    for i in 0..width {
        let base = i * 3;
        let r = data[base];
        let g = data[base + 1];
        let b = data[base + 2];
        data[base] = m[0][0] * r + m[0][1] * g + m[0][2] * b;
        data[base + 1] = m[1][0] * r + m[1][1] * g + m[1][2] * b;
        data[base + 2] = m[2][0] * r + m[2][1] * g + m[2][2] * b;
    }
}

/// Apply a gamut matrix to a row of linear F32 RGBA pixels.
///
/// Alpha channel is preserved unchanged.
pub fn apply_matrix_row_rgba_f32(data: &mut [f32], width: usize, m: &GamutMatrix) {
    for i in 0..width {
        let base = i * 4;
        let r = data[base];
        let g = data[base + 1];
        let b = data[base + 2];
        data[base] = m[0][0] * r + m[0][1] * g + m[0][2] * b;
        data[base + 1] = m[1][0] * r + m[1][1] * g + m[1][2] * b;
        data[base + 2] = m[2][0] * r + m[2][1] * g + m[2][2] * b;
        // data[base + 3] (alpha) unchanged.
    }
}

/// Look up the gamut conversion matrix between two color primary sets.
///
/// Delegates to [`ColorPrimaries::gamut_matrix_to`], which computes the matrix
/// from chromaticity coordinates with Bradford chromatic adaptation when needed.
/// Returns `None` if the primaries are the same or if either is `Unknown`.
pub fn conversion_matrix(from: ColorPrimaries, to: ColorPrimaries) -> Option<GamutMatrix> {
    if from as u8 == to as u8 {
        return None;
    }
    from.gamut_matrix_to(to)
}

/// Outcome of a fits-in-gamut check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GamutFit {
    /// Every pixel transformed cleanly into the target gamut: no negative
    /// linear components and no values exceeding `1.0 + epsilon`. Safe to
    /// re-tag the buffer with the target primaries after applying the matrix.
    AllInside,
    /// At least one pixel produced a transformed component outside
    /// `[-epsilon, 1.0 + epsilon]`. The source genuinely uses the wider
    /// gamut — narrowing would clip color and is not lossless.
    OutOfGamut,
}

/// Default epsilon for floating-point bounds tests. Loose enough to absorb
/// matrix roundoff (typical sRGB primaries roundtrip stays within 1e-5),
/// tight enough that visible out-of-gamut content fails the test.
pub const DEFAULT_GAMUT_EPSILON: f32 = 5e-4;

/// Check whether every linear-RGB pixel in `data` would land inside
/// `[-epsilon, 1.0 + epsilon]` after applying matrix `m`. Buffer is **not**
/// modified — this is a read-only scan with early-exit on the first
/// out-of-gamut pixel.
///
/// Inputs must be linear-light (apply EOTF before calling, OETF after).
/// Use [`apply_matrix_row_f32`] to actually transform the buffer once
/// the fit is confirmed.
///
/// `data.len()` must be a multiple of 3 (one f32 per channel, RGB layout).
/// Returns [`GamutFit::AllInside`] for empty inputs.
pub fn check_fits_in_gamut_linear_f32_rgb(data: &[f32], m: &GamutMatrix, epsilon: f32) -> GamutFit {
    debug_assert_eq!(data.len() % 3, 0, "rgb buffer must be multiple of 3 floats");
    let lo = -epsilon;
    let hi = 1.0 + epsilon;
    for px in data.chunks_exact(3) {
        let r = px[0];
        let g = px[1];
        let b = px[2];
        let nr = m[0][0] * r + m[0][1] * g + m[0][2] * b;
        let ng = m[1][0] * r + m[1][1] * g + m[1][2] * b;
        let nb = m[2][0] * r + m[2][1] * g + m[2][2] * b;
        if nr < lo || nr > hi || ng < lo || ng > hi || nb < lo || nb > hi {
            return GamutFit::OutOfGamut;
        }
    }
    GamutFit::AllInside
}

/// Same as [`check_fits_in_gamut_linear_f32_rgb`] but for RGBA layouts;
/// alpha is ignored.
///
/// `data.len()` must be a multiple of 4. Returns [`GamutFit::AllInside`]
/// for empty inputs.
pub fn check_fits_in_gamut_linear_f32_rgba(
    data: &[f32],
    m: &GamutMatrix,
    epsilon: f32,
) -> GamutFit {
    debug_assert_eq!(
        data.len() % 4,
        0,
        "rgba buffer must be multiple of 4 floats"
    );
    let lo = -epsilon;
    let hi = 1.0 + epsilon;
    for px in data.chunks_exact(4) {
        let r = px[0];
        let g = px[1];
        let b = px[2];
        let nr = m[0][0] * r + m[0][1] * g + m[0][2] * b;
        let ng = m[1][0] * r + m[1][1] * g + m[1][2] * b;
        let nb = m[2][0] * r + m[2][1] * g + m[2][2] * b;
        if nr < lo || nr > hi || ng < lo || ng > hi || nb < lo || nb > hi {
            return GamutFit::OutOfGamut;
        }
    }
    GamutFit::AllInside
}

/// Combined check + transform. If every pixel fits in the target gamut,
/// applies the matrix in-place and returns [`GamutFit::AllInside`]. If
/// any pixel is out of gamut, returns [`GamutFit::OutOfGamut`] **without
/// modifying** `data` — the caller can then keep the original wide-gamut
/// representation.
///
/// Two passes through the buffer when the check succeeds (one read-only
/// scan + one transform pass), one early-exiting pass when it fails.
/// Combine into a single pass with a scratch buffer if memory bandwidth
/// matters more than the worst-case rejection cost.
pub fn fit_and_transform_linear_f32_rgb(
    data: &mut [f32],
    m: &GamutMatrix,
    epsilon: f32,
) -> GamutFit {
    match check_fits_in_gamut_linear_f32_rgb(data, m, epsilon) {
        GamutFit::AllInside => {
            let width = data.len() / 3;
            apply_matrix_row_f32(data, width, m);
            GamutFit::AllInside
        }
        GamutFit::OutOfGamut => GamutFit::OutOfGamut,
    }
}

/// Combined check + transform for RGBA buffers (alpha preserved).
pub fn fit_and_transform_linear_f32_rgba(
    data: &mut [f32],
    m: &GamutMatrix,
    epsilon: f32,
) -> GamutFit {
    match check_fits_in_gamut_linear_f32_rgba(data, m, epsilon) {
        GamutFit::AllInside => {
            let width = data.len() / 4;
            apply_matrix_row_rgba_f32(data, width, m);
            GamutFit::AllInside
        }
        GamutFit::OutOfGamut => GamutFit::OutOfGamut,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ColorPrimaries;

    // ── Gamut-fit-check tests ──────────────────────────────────────────

    fn p3_to_srgb_matrix() -> GamutMatrix {
        ColorPrimaries::DisplayP3
            .gamut_matrix_to(ColorPrimaries::Bt709)
            .expect("p3 → bt709 matrix")
    }

    #[test]
    fn empty_buffer_fits() {
        let m = p3_to_srgb_matrix();
        assert_eq!(
            check_fits_in_gamut_linear_f32_rgb(&[], &m, DEFAULT_GAMUT_EPSILON),
            GamutFit::AllInside
        );
        assert_eq!(
            check_fits_in_gamut_linear_f32_rgba(&[], &m, DEFAULT_GAMUT_EPSILON),
            GamutFit::AllInside
        );
    }

    #[test]
    fn neutral_gray_fits_p3_to_srgb() {
        // Mid gray in P3 maps to mid gray in sRGB (white points match) —
        // every channel stays well inside [0, 1].
        let mut data = [0.5_f32; 3 * 4]; // 4 RGB pixels
        let m = p3_to_srgb_matrix();
        assert_eq!(
            check_fits_in_gamut_linear_f32_rgb(&data, &m, DEFAULT_GAMUT_EPSILON),
            GamutFit::AllInside
        );
        // Combined fit-and-transform should also succeed and modify in place.
        let original = data;
        let outcome = fit_and_transform_linear_f32_rgb(&mut data, &m, DEFAULT_GAMUT_EPSILON);
        assert_eq!(outcome, GamutFit::AllInside);
        // Gray stays gray after a chromatic-adaptation-free white-point-aligned matrix.
        for i in 0..data.len() {
            assert!(
                (data[i] - 0.5).abs() < 1e-4,
                "gray drift: data[{i}]={} original={}",
                data[i],
                original[i]
            );
        }
    }

    #[test]
    fn saturated_p3_red_overflows_srgb() {
        // (1, 0, 0) in P3 is brighter than sRGB red — the matrix produces
        // a value > 1.0 in the red channel. Lossy if narrowed.
        let data = [1.0_f32, 0.0, 0.0];
        let m = p3_to_srgb_matrix();
        assert_eq!(
            check_fits_in_gamut_linear_f32_rgb(&data, &m, DEFAULT_GAMUT_EPSILON),
            GamutFit::OutOfGamut
        );
    }

    #[test]
    fn p3_red_with_screenshot_levels_fits() {
        // Lighter P3 red — common screenshot territory: not maxed out.
        let data = [0.4_f32, 0.1, 0.1];
        let m = p3_to_srgb_matrix();
        assert_eq!(
            check_fits_in_gamut_linear_f32_rgb(&data, &m, DEFAULT_GAMUT_EPSILON),
            GamutFit::AllInside
        );
    }

    #[test]
    fn out_of_gamut_buffer_left_unmodified() {
        let mut data = [0.5_f32, 0.5, 0.5, 1.0, 0.0, 0.0]; // pixel 1 is saturated red
        let original = data;
        let m = p3_to_srgb_matrix();
        let outcome = fit_and_transform_linear_f32_rgb(&mut data, &m, DEFAULT_GAMUT_EPSILON);
        assert_eq!(outcome, GamutFit::OutOfGamut);
        assert_eq!(data, original, "buffer must not be touched on rejection");
    }

    #[test]
    fn rgba_alpha_preserved_after_fit_and_transform() {
        // 2 RGBA pixels, both gray, both partially transparent.
        let mut data = [0.5_f32, 0.5, 0.5, 0.25, 0.7, 0.7, 0.7, 0.9];
        let m = p3_to_srgb_matrix();
        let outcome = fit_and_transform_linear_f32_rgba(&mut data, &m, DEFAULT_GAMUT_EPSILON);
        assert_eq!(outcome, GamutFit::AllInside);
        assert!((data[3] - 0.25).abs() < 1e-7, "alpha pixel 0 changed");
        assert!((data[7] - 0.9).abs() < 1e-7, "alpha pixel 1 changed");
    }

    #[test]
    fn negative_component_treated_as_out_of_gamut() {
        // Hand-crafted near-out-of-gamut: pure saturated cyan in P3 has
        // a slightly-negative red component when expressed in sRGB.
        let data = [0.0_f32, 1.0, 1.0];
        let m = p3_to_srgb_matrix();
        assert_eq!(
            check_fits_in_gamut_linear_f32_rgb(&data, &m, DEFAULT_GAMUT_EPSILON),
            GamutFit::OutOfGamut
        );
    }

    #[test]
    fn epsilon_tolerates_tiny_overshoot() {
        // Construct a value that lands at exactly 1.0 + 1e-5 after the matrix.
        // Should fit with default epsilon (5e-4) but fail with strict zero.
        // Use the identity-ish case: white in BT.709 → BT.709 (no-op).
        // Slightly above-1.0 input — simulates roundoff.
        let data = [1.0001_f32, 1.0001, 1.0001];
        // sRGB → sRGB matrix is identity
        let m: GamutMatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(
            check_fits_in_gamut_linear_f32_rgb(&data, &m, DEFAULT_GAMUT_EPSILON),
            GamutFit::AllInside
        );
        assert_eq!(
            check_fits_in_gamut_linear_f32_rgb(&data, &m, 0.0),
            GamutFit::OutOfGamut
        );
    }

    #[test]
    fn early_exit_on_first_out_of_gamut_pixel() {
        // Buffer of mostly-fine pixels with one out-of-gamut at index 5.
        // Verifies we don't depend on later pixels.
        let mut data = vec![0.4_f32, 0.4, 0.4]; // pixel 0
        for _ in 0..4 {
            data.extend_from_slice(&[0.5, 0.5, 0.5]);
        }
        data.extend_from_slice(&[1.0, 0.0, 0.0]); // pixel 5 out of gamut
        for _ in 0..100 {
            data.extend_from_slice(&[0.5, 0.5, 0.5]);
        }
        let m = p3_to_srgb_matrix();
        assert_eq!(
            check_fits_in_gamut_linear_f32_rgb(&data, &m, DEFAULT_GAMUT_EPSILON),
            GamutFit::OutOfGamut
        );
    }

    /// Verify that forward × inverse ≈ identity for BT.709 ↔ BT.2020.
    #[test]
    fn bt709_bt2020_roundtrip() {
        let fwd = ColorPrimaries::Bt709
            .gamut_matrix_to(ColorPrimaries::Bt2020)
            .unwrap();
        let inv = ColorPrimaries::Bt2020
            .gamut_matrix_to(ColorPrimaries::Bt709)
            .unwrap();
        let test_rgb = [0.5f32, 0.3, 0.8];
        let mut rgb = test_rgb;
        apply_matrix_f32(&mut rgb, &fwd);
        apply_matrix_f32(&mut rgb, &inv);
        for c in 0..3 {
            assert!(
                (rgb[c] - test_rgb[c]).abs() < 1e-4,
                "BT.709→BT.2020→BT.709 roundtrip error in ch{c}: {:.6} vs {:.6}",
                rgb[c],
                test_rgb[c]
            );
        }
    }

    /// Verify that forward × inverse ≈ identity for BT.709 ↔ Display P3.
    #[test]
    fn bt709_displayp3_roundtrip() {
        let fwd = ColorPrimaries::Bt709
            .gamut_matrix_to(ColorPrimaries::DisplayP3)
            .unwrap();
        let inv = ColorPrimaries::DisplayP3
            .gamut_matrix_to(ColorPrimaries::Bt709)
            .unwrap();
        let test_rgb = [0.5f32, 0.3, 0.8];
        let mut rgb = test_rgb;
        apply_matrix_f32(&mut rgb, &fwd);
        apply_matrix_f32(&mut rgb, &inv);
        for c in 0..3 {
            assert!(
                (rgb[c] - test_rgb[c]).abs() < 1e-4,
                "BT.709→P3→BT.709 roundtrip error in ch{c}: {:.6} vs {:.6}",
                rgb[c],
                test_rgb[c]
            );
        }
    }

    /// White point preservation: [1,1,1] in BT.709 → BT.2020 should remain ~[1,1,1].
    #[test]
    fn white_point_preservation() {
        let m = ColorPrimaries::Bt709
            .gamut_matrix_to(ColorPrimaries::Bt2020)
            .unwrap();
        let mut rgb = [1.0f32, 1.0, 1.0];
        apply_matrix_f32(&mut rgb, &m);
        for (c, &val) in rgb.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-4,
                "White point not preserved in ch{c}: {val:.6}",
            );
        }
    }

    /// RGBA row preserves alpha.
    #[test]
    fn rgba_alpha_preserved() {
        let m = ColorPrimaries::Bt709
            .gamut_matrix_to(ColorPrimaries::Bt2020)
            .unwrap();
        let mut row = [0.5f32, 0.3, 0.8, 0.42, 0.1, 0.9, 0.2, 0.99];
        apply_matrix_row_rgba_f32(&mut row, 2, &m);
        assert_eq!(row[3], 0.42);
        assert_eq!(row[7], 0.99);
    }

    /// Verify XYZ→RGB × RGB→XYZ ≈ identity for BT.709.
    #[test]
    fn xyz_bt709_roundtrip() {
        let rgb = [0.5f32, 0.3, 0.8];
        let mut v = rgb;
        apply_matrix_f32(&mut v, &BT709_TO_XYZ);
        apply_matrix_f32(&mut v, &XYZ_TO_BT709);
        for c in 0..3 {
            assert!(
                (v[c] - rgb[c]).abs() < 1e-4,
                "XYZ BT.709 roundtrip ch{c}: {:.6} vs {:.6}",
                v[c],
                rgb[c]
            );
        }
    }

    /// Verify XYZ→RGB × RGB→XYZ ≈ identity for Display P3.
    #[test]
    fn xyz_displayp3_roundtrip() {
        let rgb = [0.5f32, 0.3, 0.8];
        let mut v = rgb;
        apply_matrix_f32(&mut v, &DISPLAY_P3_TO_XYZ);
        apply_matrix_f32(&mut v, &XYZ_TO_DISPLAY_P3);
        for c in 0..3 {
            assert!(
                (v[c] - rgb[c]).abs() < 1e-4,
                "XYZ P3 roundtrip ch{c}: {:.6} vs {:.6}",
                v[c],
                rgb[c]
            );
        }
    }

    /// Verify XYZ→RGB × RGB→XYZ ≈ identity for BT.2020.
    #[test]
    fn xyz_bt2020_roundtrip() {
        let rgb = [0.5f32, 0.3, 0.8];
        let mut v = rgb;
        apply_matrix_f32(&mut v, &BT2020_TO_XYZ);
        apply_matrix_f32(&mut v, &XYZ_TO_BT2020);
        for c in 0..3 {
            assert!(
                (v[c] - rgb[c]).abs() < 1e-4,
                "XYZ BT.2020 roundtrip ch{c}: {:.6} vs {:.6}",
                v[c],
                rgb[c]
            );
        }
    }

    /// XYZ white point preservation: [1,1,1] → XYZ → RGB should remain ~[1,1,1].
    #[test]
    fn xyz_white_point() {
        for (name, to, from) in [
            ("BT.709", &BT709_TO_XYZ, &XYZ_TO_BT709),
            ("P3", &DISPLAY_P3_TO_XYZ, &XYZ_TO_DISPLAY_P3),
            ("BT.2020", &BT2020_TO_XYZ, &XYZ_TO_BT2020),
        ] {
            let mut rgb = [1.0f32; 3];
            apply_matrix_f32(&mut rgb, to);
            apply_matrix_f32(&mut rgb, from);
            for (c, &val) in rgb.iter().enumerate() {
                assert!(
                    (val - 1.0).abs() < 1e-3,
                    "{name} XYZ white point ch{c}: {val:.6}",
                );
            }
        }
    }

    /// mat3_mul: verify A × A_inv ≈ identity.
    #[test]
    fn mat3_mul_inverse() {
        let identity = mat3_mul(&BT709_TO_XYZ, &XYZ_TO_BT709);
        for (i, row) in identity.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-4,
                    "mat3_mul identity [{i}][{j}] = {val:.6}, expected {expected:.1}",
                );
            }
        }
    }

    /// Verify cross-gamut via XYZ: BT.709→XYZ→BT.2020 ≈ direct BT.709→BT.2020.
    #[test]
    fn xyz_cross_gamut_consistency() {
        let via_xyz = mat3_mul(&XYZ_TO_BT2020, &BT709_TO_XYZ);
        let direct = ColorPrimaries::Bt709
            .gamut_matrix_to(ColorPrimaries::Bt2020)
            .unwrap();
        let rgb = [0.5f32, 0.3, 0.8];
        let mut v1 = rgb;
        apply_matrix_f32(&mut v1, &via_xyz);
        let mut v2 = rgb;
        apply_matrix_f32(&mut v2, &direct);
        for c in 0..3 {
            assert!(
                (v1[c] - v2[c]).abs() < 1e-3,
                "cross-gamut ch{c}: via_xyz={:.6} vs direct={:.6}",
                v1[c],
                v2[c]
            );
        }
    }
}
