#![allow(clippy::excessive_precision)]
//! Oklab relative to any supported RGB primaries.
//!
//! [`OklabSpace`] holds the two matrices that planar Oklab kernels need and does
//! scalar conversions. The constants are CSS Color 4's, for Björn Ottosson's Oklab.

use crate::ext::ColorPrimariesExt;
use crate::gamut::GamutMatrix;
use zenpixels::ColorPrimaries;

// ---------------------------------------------------------------------------
// Oklab matrices: CSS Color 4's conversions.js, which recomputes Björn Ottosson's
// so that D65 white gives L = 1 with no chroma.
// ---------------------------------------------------------------------------

/// M1 step 1: CIE XYZ (D65) → LMS.
pub const LMS_FROM_XYZ: GamutMatrix = GamutMatrix::from_rows([
    [0.8190224379967030, 0.3619062600528904, -0.1288737815209879],
    [0.0329836539323885, 0.9292868615863434, 0.0361446663506424],
    [0.0481771893596242, 0.2642395317527308, 0.6335478284694309],
]);

/// Inverse of M1: LMS → CIE XYZ (D65).
pub const XYZ_FROM_LMS: GamutMatrix = GamutMatrix::from_rows([
    [1.2268798758459243, -0.5578149944602171, 0.2813910456659647],
    [-0.0405757452148008, 1.1122868032803170, -0.0717110580655164],
    [-0.0763729366746601, -0.4214933324022432, 1.5869240198367816],
]);

/// M2: LMS^(1/3) → Oklab [L, a, b]. Universal, primaries-independent.
pub const OKLAB_FROM_LMS_CBRT: GamutMatrix = GamutMatrix::from_rows([
    [0.2104542683093140, 0.7936177747023054, -0.0040720430116193],
    [1.9779985324311684, -2.4285922420485799, 0.4505937096174110],
    [0.0259040424655478, 0.7827717124575296, -0.8086757549230774],
]);

/// Inverse of M2: Oklab [L, a, b] → LMS^(1/3).
pub const LMS_CBRT_FROM_OKLAB: GamutMatrix = GamutMatrix::from_rows([
    [1.0, 0.3963377773761749, 0.2158037573099136],
    [1.0, -0.1055613458156586, -0.0638541728258133],
    [1.0, -0.0894841775298119, -1.2914855480194092],
]);

// ---------------------------------------------------------------------------
// Combined matrices
// ---------------------------------------------------------------------------

/// Oklab relative to one set of RGB primaries.
///
/// Holds both directions so the two can't be swapped: a [`GamutMatrix`] alone
/// doesn't say which way it goes. Kernels take [`rgb_to_lms`](Self::rgb_to_lms)
/// or [`lms_to_rgb`](Self::lms_to_rgb) directly.
///
/// ```
/// use zenpixels_convert::{ColorPrimaries, oklab::OklabSpace};
///
/// let space = OklabSpace::for_primaries(ColorPrimaries::Bt709).unwrap();
/// let [l, a, b] = space.from_rgb([1.0, 1.0, 1.0]);
/// assert!((l - 1.0).abs() < 1e-3 && a.abs() < 1e-3 && b.abs() < 1e-3);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OklabSpace {
    rgb_to_lms: GamutMatrix,
    lms_to_rgb: GamutMatrix,
}

impl OklabSpace {
    /// The space for `primaries`, or `None` when they have no XYZ matrix here.
    #[must_use]
    pub fn for_primaries(primaries: ColorPrimaries) -> Option<Self> {
        Some(Self {
            rgb_to_lms: rgb_to_lms_matrix(primaries)?,
            lms_to_rgb: lms_to_rgb_matrix(primaries)?,
        })
    }

    /// Linear RGB to LMS, skipping an intermediate XYZ buffer.
    #[inline]
    #[must_use]
    pub const fn rgb_to_lms(&self) -> &GamutMatrix {
        &self.rgb_to_lms
    }

    /// LMS to linear RGB, skipping an intermediate XYZ buffer.
    #[inline]
    #[must_use]
    pub const fn lms_to_rgb(&self) -> &GamutMatrix {
        &self.lms_to_rgb
    }

    /// Linear RGB to `[L, a, b]`.
    #[inline]
    #[must_use]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_rgb(&self, [r, g, b]: [f32; 3]) -> [f32; 3] {
        rgb_to_oklab(r, g, b, &self.rgb_to_lms)
    }

    /// `[L, a, b]` to linear RGB, unclipped.
    #[inline]
    #[must_use]
    pub fn to_rgb(&self, [l, a, b]: [f32; 3]) -> [f32; 3] {
        oklab_to_rgb(l, a, b, &self.lms_to_rgb)
    }
}

/// `LMS_FROM_XYZ × RGB_TO_XYZ`.
pub(crate) fn rgb_to_lms_matrix(primaries: ColorPrimaries) -> Option<GamutMatrix> {
    Some(primaries.to_xyz_matrix()?.then(&LMS_FROM_XYZ))
}

/// `XYZ_TO_RGB × XYZ_FROM_LMS`.
pub(crate) fn lms_to_rgb_matrix(primaries: ColorPrimaries) -> Option<GamutMatrix> {
    Some(XYZ_FROM_LMS.then(primaries.from_xyz_matrix()?))
}

// ---------------------------------------------------------------------------
// Fast cube root
// ---------------------------------------------------------------------------

/// Fast approximate cube root with ~22 bits of precision.
///
/// Uses bit-manipulation for an initial estimate followed by two
/// Newton-Raphson iterations. Handles negative inputs and zero correctly.
#[inline]
pub(crate) fn fast_cbrt(x: f32) -> f32 {
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
#[inline]
pub(crate) fn rgb_to_oklab(r: f32, g: f32, b: f32, m1: &[[f32; 3]; 3]) -> [f32; 3] {
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
#[inline]
pub(crate) fn oklab_to_rgb(l: f32, a: f32, b: f32, m1_inv: &[[f32; 3]; 3]) -> [f32; 3] {
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
#[path = "oklab_cbrt_compare.rs"]
mod cbrt_compare;

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

    /// Neutrals have no chroma and L is the cube root of the gray, in every space with a matrix.
    #[test]
    fn oklab_neutrals_have_no_chroma() {
        for p in [
            ColorPrimaries::Bt709,
            ColorPrimaries::DisplayP3,
            ColorPrimaries::Bt2020,
        ] {
            let m1 = rgb_to_lms_matrix(p).unwrap();
            for i in 1..=4096 {
                let g = i as f32 / 4096.0;
                let [l, a, b] = rgb_to_oklab(g, g, g, &m1);
                assert!(a.hypot(b) < 1e-6, "{p:?} gray {g}: a {a}, b {b}");
                assert!((l - g.cbrt()).abs() < 2e-6, "{p:?} gray {g}: L {l}");
            }
        }
    }

    /// CSS Color 4's results for sRGB's primaries, from its conversions.js in f64.
    #[test]
    fn oklab_matches_css() {
        let m1 = rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();
        for (rgb, css) in [
            ([1.0, 0.0, 0.0], [0.627955, 0.224863, 0.125846]),
            ([0.0, 1.0, 0.0], [0.866440, -0.233888, 0.179498]),
            ([0.0, 0.0, 1.0], [0.452014, -0.032457, -0.311528]),
        ] {
            let lab = rgb_to_oklab(rgb[0], rgb[1], rgb[2], &m1);
            for (got, want) in lab.iter().zip(css) {
                assert!((got - want).abs() < 2e-6, "{rgb:?}: {lab:?} vs {css:?}");
            }
        }
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
