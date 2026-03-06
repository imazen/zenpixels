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

/// BT.709 → BT.2020 (row-major).
///
/// Converts linear BT.709/sRGB RGB to linear BT.2020 RGB.
pub(crate) const BT709_TO_BT2020: GamutMatrix = [
    [0.6274_0389, 0.3292_8303, 0.0433_1307],
    [0.0690_9729, 0.9195_4040, 0.0113_6232],
    [0.0163_9170, 0.0880_1327, 0.8955_9503],
];

/// BT.2020 → BT.709 (row-major).
///
/// Converts linear BT.2020 RGB to linear BT.709/sRGB RGB.
/// Values outside \[0,1\] indicate out-of-gamut colors.
pub(crate) const BT2020_TO_BT709: GamutMatrix = [
    [1.6604_9100, -0.5876_5614, -0.0728_3486],
    [-0.1245_5047, 1.1328_9990, -0.0083_4942],
    [-0.0181_5076, -0.1005_7890, 1.1187_2966],
];

/// BT.709 → Display P3 (row-major).
pub(crate) const BT709_TO_DISPLAY_P3: GamutMatrix = [
    [0.8224_5811, 0.1775_4189, 0.0000_0000],
    [0.0331_9419, 0.9668_0581, 0.0000_0000],
    [0.0170_8263, 0.0723_9744, 0.9105_3993],
];

/// Display P3 → BT.709 (row-major).
pub(crate) const DISPLAY_P3_TO_BT709: GamutMatrix = [
    [1.2249_4018, -0.2249_4018, 0.0000_0000],
    [-0.0420_4986, 1.0420_4986, 0.0000_0000],
    [-0.0196_4113, -0.0786_4905, 1.0982_5018],
];

/// BT.2020 → Display P3 (row-major).
pub(crate) const BT2020_TO_DISPLAY_P3: GamutMatrix = [
    [1.3434_6376, -0.2826_7869, -0.0607_8507],
    [-0.0652_8279, 1.0764_0361, -0.0111_2082],
    [-0.0028_8423, -0.0193_4633, 1.0222_3056],
];

/// Display P3 → BT.2020 (row-major).
pub(crate) const DISPLAY_P3_TO_BT2020: GamutMatrix = [
    [0.7536_7740, 0.1985_4087, 0.0477_8174],
    [0.0457_0150, 0.9417_7793, 0.0125_2057],
    [0.0011_7409, 0.0176_4065, 0.9811_8526],
];

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
/// Returns `None` if the primaries are the same or if no matrix is available.
pub fn conversion_matrix(from: ColorPrimaries, to: ColorPrimaries) -> Option<&'static GamutMatrix> {
    match (from, to) {
        (ColorPrimaries::Bt709, ColorPrimaries::Bt2020) => Some(&BT709_TO_BT2020),
        (ColorPrimaries::Bt2020, ColorPrimaries::Bt709) => Some(&BT2020_TO_BT709),
        (ColorPrimaries::Bt709, ColorPrimaries::DisplayP3) => Some(&BT709_TO_DISPLAY_P3),
        (ColorPrimaries::DisplayP3, ColorPrimaries::Bt709) => Some(&DISPLAY_P3_TO_BT709),
        (ColorPrimaries::Bt2020, ColorPrimaries::DisplayP3) => Some(&BT2020_TO_DISPLAY_P3),
        (ColorPrimaries::DisplayP3, ColorPrimaries::Bt2020) => Some(&DISPLAY_P3_TO_BT2020),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that forward × inverse ≈ identity for BT.709 ↔ BT.2020.
    #[test]
    fn bt709_bt2020_roundtrip() {
        let test_rgb = [0.5f32, 0.3, 0.8];
        let mut rgb = test_rgb;
        apply_matrix_f32(&mut rgb, &BT709_TO_BT2020);
        apply_matrix_f32(&mut rgb, &BT2020_TO_BT709);
        for c in 0..3 {
            assert!(
                (rgb[c] - test_rgb[c]).abs() < 1e-5,
                "BT.709→BT.2020→BT.709 roundtrip error in ch{c}: {:.6} vs {:.6}",
                rgb[c],
                test_rgb[c]
            );
        }
    }

    /// Verify that forward × inverse ≈ identity for BT.709 ↔ Display P3.
    #[test]
    fn bt709_displayp3_roundtrip() {
        let test_rgb = [0.5f32, 0.3, 0.8];
        let mut rgb = test_rgb;
        apply_matrix_f32(&mut rgb, &BT709_TO_DISPLAY_P3);
        apply_matrix_f32(&mut rgb, &DISPLAY_P3_TO_BT709);
        for c in 0..3 {
            assert!(
                (rgb[c] - test_rgb[c]).abs() < 1e-5,
                "BT.709→P3→BT.709 roundtrip error in ch{c}: {:.6} vs {:.6}",
                rgb[c],
                test_rgb[c]
            );
        }
    }

    /// White point preservation: [1,1,1] in BT.709 → BT.2020 should remain ~[1,1,1].
    #[test]
    fn white_point_preservation() {
        let mut rgb = [1.0f32, 1.0, 1.0];
        apply_matrix_f32(&mut rgb, &BT709_TO_BT2020);
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
        let mut row = [0.5f32, 0.3, 0.8, 0.42, 0.1, 0.9, 0.2, 0.99];
        apply_matrix_row_rgba_f32(&mut row, 2, &BT709_TO_BT2020);
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
        let rgb = [0.5f32, 0.3, 0.8];
        let mut v1 = rgb;
        apply_matrix_f32(&mut v1, &via_xyz);
        let mut v2 = rgb;
        apply_matrix_f32(&mut v2, &BT709_TO_BT2020);
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
