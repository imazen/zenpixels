#![allow(clippy::excessive_precision)]
//! Color gamut conversion matrices for BT.709, BT.2020, and Display P3.
//!
//! All matrices operate in **linear light** — apply EOTF first, then gamut
//! matrix, then OETF. The matrices are computed from the CIE 1931 xy
//! chromaticity coordinates of each primary set using D65 white point.
//!
//! These are exact, standards-derived values (not fitted or approximated).

use crate::ColorPrimaries;

/// A 3×3 row-major matrix between linear-light tristimulus spaces, in `f32`.
///
/// Dereferences to `[[f32; 3]; 3]`, so `m[0][2]` reads and SIMD kernels taking
/// the plain array accept `&m`.
///
/// ```
/// use zenpixels_convert::{ColorPrimaries, gamut::GamutMatrix};
///
/// let to_srgb = GamutMatrix::between(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709).unwrap();
/// let mut row = [[1.0_f32, 1.0, 1.0], [0.5, 0.5, 0.5]];
/// to_srgb.apply_row_rgb(&mut row);
/// assert!((row[1][0] - 0.5).abs() < 1e-4, "gray stays gray");
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct GamutMatrix([[f32; 3]; 3]);

impl core::ops::Deref for GamutMatrix {
    type Target = [[f32; 3]; 3];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<[[f32; 3]; 3]> for GamutMatrix {
    #[inline]
    fn from(rows: [[f32; 3]; 3]) -> Self {
        Self(rows)
    }
}

impl From<GamutMatrix> for [[f32; 3]; 3] {
    #[inline]
    fn from(m: GamutMatrix) -> Self {
        m.0
    }
}

impl GamutMatrix {
    /// A matrix from its rows.
    #[inline]
    #[must_use]
    pub const fn from_rows(rows: [[f32; 3]; 3]) -> Self {
        Self(rows)
    }

    /// The matrix taking linear RGB in `from` to linear RGB in `to`.
    ///
    /// Equal primaries give the identity. `None` when either is
    /// [`Unknown`](ColorPrimaries::Unknown). Computed from chromaticities, with
    /// Bradford chromatic adaptation between different white points.
    #[must_use]
    pub const fn between(from: ColorPrimaries, to: ColorPrimaries) -> Option<Self> {
        match from.gamut_matrix_to(to) {
            Some(rows) => Some(Self(rows)),
            None => None,
        }
    }

    /// The matrix that applies `self`, then `next`: `next × self`.
    #[must_use]
    pub fn then(&self, next: &Self) -> Self {
        let (a, b) = (&next.0, &self.0);
        Self(core::array::from_fn(|i| {
            core::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
        }))
    }

    /// The inverse, or `None` for a singular matrix.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let [[a, b, c], [d, e, f], [g, h, i]] = self.0;
        let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        if det.abs() < 1e-30 {
            return None;
        }
        let inv = 1.0 / det;
        Some(Self([
            [
                (e * i - f * h) * inv,
                -(b * i - c * h) * inv,
                (b * f - c * e) * inv,
            ],
            [
                -(d * i - f * g) * inv,
                (a * i - c * g) * inv,
                -(a * f - c * d) * inv,
            ],
            [
                (d * h - e * g) * inv,
                -(a * h - b * g) * inv,
                (a * e - b * d) * inv,
            ],
        ]))
    }

    /// This matrix applied to one color.
    #[inline]
    #[must_use]
    pub fn apply(&self, [r, g, b]: [f32; 3]) -> [f32; 3] {
        let m = &self.0;
        [
            m[0][0] * r + m[0][1] * g + m[0][2] * b,
            m[1][0] * r + m[1][1] * g + m[1][2] * b,
            m[2][0] * r + m[2][1] * g + m[2][2] * b,
        ]
    }

    /// Apply in place to a row of RGB pixels in linear light.
    ///
    /// For conversions that also change transfer or depth, or that aren't in
    /// place, use [`RowConverter`](crate::RowConverter), whose fused kernels are
    /// faster than this followed by a separate transfer step.
    #[inline]
    pub fn apply_row_rgb(&self, row: &mut [[f32; 3]]) {
        crate::fast_gamut::convert_linear_rgb(&self.0, row.as_flattened_mut());
    }

    /// Apply in place to a row of RGBA pixels in linear light. Alpha is unchanged.
    #[inline]
    pub fn apply_row_rgba(&self, row: &mut [[f32; 4]]) {
        crate::fast_gamut::convert_linear_rgba(&self.0, row.as_flattened_mut());
    }
}

// ---------------------------------------------------------------------------
// RGB ↔ CIE XYZ (D65 white point) matrices
// ---------------------------------------------------------------------------

/// BT.709/sRGB → CIE XYZ (D65). Derived from CIE 1931 xy chromaticities.
pub(crate) const BT709_TO_XYZ: GamutMatrix = GamutMatrix([
    [0.4123907993, 0.3575843394, 0.1804807884],
    [0.2126390059, 0.7151686788, 0.0721923154],
    [0.0193308187, 0.1191947798, 0.9505321522],
]);

/// CIE XYZ (D65) → BT.709/sRGB.
pub(crate) const XYZ_TO_BT709: GamutMatrix = GamutMatrix([
    [3.2409699419, -1.5373831776, -0.4986107603],
    [-0.9692436363, 1.8759675015, 0.0415550574],
    [0.0556300797, -0.2039769589, 1.0569715142],
]);

/// Display P3 → CIE XYZ (D65). Same white point as sRGB, wider primaries.
pub(crate) const DISPLAY_P3_TO_XYZ: GamutMatrix = GamutMatrix([
    [0.4865709486, 0.2656676932, 0.1982172852],
    [0.2289745641, 0.6917385218, 0.0792869141],
    [0.0000000000, 0.0451133819, 1.0439443689],
]);

/// CIE XYZ (D65) → Display P3.
pub(crate) const XYZ_TO_DISPLAY_P3: GamutMatrix = GamutMatrix([
    [2.4934969119, -0.9313836179, -0.4027107845],
    [-0.8294889696, 1.7626640603, 0.0236246858],
    [0.0358458302, -0.0761723893, 0.9568845240],
]);

/// BT.2020 → CIE XYZ (D65).
pub(crate) const BT2020_TO_XYZ: GamutMatrix = GamutMatrix([
    [0.6369580484, 0.1446169036, 0.1688809752],
    [0.2627002120, 0.6779980715, 0.0593017165],
    [0.0000000000, 0.0280726930, 1.0609850578],
]);

/// CIE XYZ (D65) → BT.2020.
pub(crate) const XYZ_TO_BT2020: GamutMatrix = GamutMatrix([
    [1.7166511880, -0.3556707838, -0.2533662814],
    [-0.6666843518, 1.6164812366, 0.0157685458],
    [0.0176398574, -0.0427706133, 0.9421031212],
]);

/// [`GamutMatrix::between`], except that equal primaries give `None`: the plan
/// builders use it to decide whether a matrix step exists at all.
pub(crate) fn conversion_matrix(from: ColorPrimaries, to: ColorPrimaries) -> Option<GamutMatrix> {
    if from == to {
        return None;
    }
    GamutMatrix::between(from, to)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ColorPrimaries;

    /// Verify that forward × inverse ≈ identity for BT.709 ↔ BT.2020.
    #[test]
    fn bt709_bt2020_roundtrip() {
        let fwd = GamutMatrix::between(ColorPrimaries::Bt709, ColorPrimaries::Bt2020).unwrap();
        let inv = GamutMatrix::between(ColorPrimaries::Bt2020, ColorPrimaries::Bt709).unwrap();
        let test_rgb = [0.5f32, 0.3, 0.8];
        let mut rgb = test_rgb;
        rgb = fwd.apply(rgb);
        rgb = inv.apply(rgb);
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
        let fwd = GamutMatrix::between(ColorPrimaries::Bt709, ColorPrimaries::DisplayP3).unwrap();
        let inv = GamutMatrix::between(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709).unwrap();
        let test_rgb = [0.5f32, 0.3, 0.8];
        let mut rgb = test_rgb;
        rgb = fwd.apply(rgb);
        rgb = inv.apply(rgb);
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
        let m = GamutMatrix::between(ColorPrimaries::Bt709, ColorPrimaries::Bt2020).unwrap();
        let mut rgb = [1.0f32, 1.0, 1.0];
        rgb = m.apply(rgb);
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
        let m = GamutMatrix::between(ColorPrimaries::Bt709, ColorPrimaries::Bt2020).unwrap();
        let mut row = [0.5f32, 0.3, 0.8, 0.42, 0.1, 0.9, 0.2, 0.99];
        m.apply_row_rgba(&mut row.as_chunks_mut::<4>().0[..2]);
        assert_eq!(row[3], 0.42);
        assert_eq!(row[7], 0.99);
    }

    /// Verify XYZ→RGB × RGB→XYZ ≈ identity for BT.709.
    #[test]
    fn xyz_bt709_roundtrip() {
        let rgb = [0.5f32, 0.3, 0.8];
        let mut v = rgb;
        v = BT709_TO_XYZ.apply(v);
        v = XYZ_TO_BT709.apply(v);
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
        v = DISPLAY_P3_TO_XYZ.apply(v);
        v = XYZ_TO_DISPLAY_P3.apply(v);
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
        v = BT2020_TO_XYZ.apply(v);
        v = XYZ_TO_BT2020.apply(v);
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
            rgb = to.apply(rgb);
            rgb = from.apply(rgb);
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
        let identity = XYZ_TO_BT709.then(&BT709_TO_XYZ);
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
        let via_xyz = BT709_TO_XYZ.then(&XYZ_TO_BT2020);
        let direct = GamutMatrix::between(ColorPrimaries::Bt709, ColorPrimaries::Bt2020).unwrap();
        let rgb = [0.5f32, 0.3, 0.8];
        let mut v1 = rgb;
        v1 = via_xyz.apply(v1);
        let mut v2 = rgb;
        v2 = direct.apply(v2);
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
