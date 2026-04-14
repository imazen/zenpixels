//! Color space registry — single source of truth for known color spaces.
//!
//! Maps between [`ColorPrimaries`], [`TransferFunction`], [`NamedProfile`],
//! and CICP codes. All the conversion functions (`from_cicp`, `to_cicp`,
//! `to_primaries_transfer`, etc.) should derive from this table.
//!
//! # Matrix computation
//!
//! Rather than hardcoding O(n²) pairwise matrices, we store one RGB→XYZ
//! matrix per primaries set and compute any src→dst matrix on demand:
//!
//! ```text
//! M = inv(dst_xyz) × adapt(src_wp, dst_wp) × src_xyz
//! ```
//!
//! The Bradford chromatic adaptation is only needed when white points differ.

use crate::color::NamedProfile;
use crate::{ColorPrimaries, TransferFunction};

/// A known color space entry in the registry.
#[derive(Clone, Copy, Debug)]
pub struct KnownColorSpace {
    /// Color primaries (gamut + white point).
    pub primaries: ColorPrimaries,
    /// Transfer function (EOTF encoding).
    pub transfer: TransferFunction,
    /// CICP codes (color_primaries, transfer_characteristics), if a mapping exists.
    pub cicp: Option<(u8, u8)>,
    /// Named profile variant, if one exists.
    pub named: Option<NamedProfile>,
}

/// The registry of all known color spaces.
///
/// This is the single source of truth. All `from_cicp`, `to_cicp`,
/// `to_primaries_transfer`, and `from_primaries_transfer` functions
/// should be derivable from this table.
pub const REGISTRY: &[KnownColorSpace] = &[
    KnownColorSpace {
        primaries: ColorPrimaries::Bt709,
        transfer: TransferFunction::Srgb,
        cicp: Some((1, 13)),
        named: Some(NamedProfile::Srgb),
    },
    KnownColorSpace {
        primaries: ColorPrimaries::DisplayP3,
        transfer: TransferFunction::Srgb,
        cicp: Some((12, 13)),
        named: Some(NamedProfile::DisplayP3),
    },
    KnownColorSpace {
        primaries: ColorPrimaries::Bt2020,
        transfer: TransferFunction::Bt709,
        cicp: Some((9, 1)),
        named: Some(NamedProfile::Bt2020),
    },
    KnownColorSpace {
        primaries: ColorPrimaries::Bt2020,
        transfer: TransferFunction::Pq,
        cicp: Some((9, 16)),
        named: Some(NamedProfile::Bt2020Pq),
    },
    KnownColorSpace {
        primaries: ColorPrimaries::AdobeRgb,
        transfer: TransferFunction::Gamma22,
        cicp: None,
        named: Some(NamedProfile::AdobeRgb),
    },
    KnownColorSpace {
        primaries: ColorPrimaries::Bt709,
        transfer: TransferFunction::Linear,
        cicp: Some((1, 8)),
        named: Some(NamedProfile::LinearSrgb),
    },
    KnownColorSpace {
        primaries: ColorPrimaries::DisplayP3,
        transfer: TransferFunction::Linear,
        cicp: Some((12, 8)),
        named: None,
    },
    KnownColorSpace {
        primaries: ColorPrimaries::Bt2020,
        transfer: TransferFunction::Linear,
        cicp: Some((9, 8)),
        named: None,
    },
    KnownColorSpace {
        primaries: ColorPrimaries::DisplayP3,
        transfer: TransferFunction::Pq,
        cicp: Some((12, 16)),
        named: None,
    },
];

// =========================================================================
// Lookup functions
// =========================================================================

/// Find a registry entry by CICP codes.
pub const fn find_by_cicp(cp: u8, tc: u8) -> Option<&'static KnownColorSpace> {
    let mut i = 0;
    while i < REGISTRY.len() {
        if let Some((rcp, rtc)) = REGISTRY[i].cicp {
            if rcp == cp && rtc == tc {
                return Some(&REGISTRY[i]);
            }
        }
        i += 1;
    }
    None
}

/// Find a registry entry by primaries + transfer.
pub const fn find_by_primaries_transfer(
    primaries: ColorPrimaries,
    transfer: TransferFunction,
) -> Option<&'static KnownColorSpace> {
    let mut i = 0;
    while i < REGISTRY.len() {
        if REGISTRY[i].primaries as u8 == primaries as u8
            && REGISTRY[i].transfer as u8 == transfer as u8
        {
            return Some(&REGISTRY[i]);
        }
        i += 1;
    }
    None
}

/// Find a registry entry by named profile.
pub const fn find_by_named(named: NamedProfile) -> Option<&'static KnownColorSpace> {
    let mut i = 0;
    while i < REGISTRY.len() {
        if let Some(rn) = REGISTRY[i].named {
            if rn as u8 == named as u8 {
                return Some(&REGISTRY[i]);
            }
        }
        i += 1;
    }
    None
}

// =========================================================================
// Matrix computation
// =========================================================================

/// 3×3 f32 matrix type.
pub type Mat3 = [[f32; 3]; 3];

/// Compute the RGB-to-XYZ matrix for a given set of primaries.
///
/// Uses the standard derivation from CIE xy chromaticity coordinates
/// and white point. Returns `None` for unknown primaries.
pub const fn rgb_to_xyz(primaries: ColorPrimaries) -> Option<Mat3> {
    let chrom = match primaries.chromaticity() {
        Some(c) => c,
        None => return None,
    };
    let ((rx, ry), (gx, gy), (bx, by)) = chrom;
    let (wx, wy) = primaries.white_point();

    let xr = rx / ry;
    let zr = (1.0 - rx - ry) / ry;
    let xg = gx / gy;
    let zg = (1.0 - gx - gy) / gy;
    let xb = bx / by;
    let zb = (1.0 - bx - by) / by;

    let xw = wx / wy;
    let zw = (1.0 - wx - wy) / wy;

    let m = [[xr, xg, xb], [1.0, 1.0, 1.0], [zr, zg, zb]];
    let w = [xw, 1.0, zw];
    let s = match solve_3x3(&m, &w) {
        Some(s) => s,
        None => return None,
    };

    Some([
        [xr * s[0], xg * s[1], xb * s[2]],
        [s[0], s[1], s[2]],
        [zr * s[0], zg * s[1], zb * s[2]],
    ])
}

/// Compute the gamut conversion matrix from `src` to `dst` primaries.
///
/// Handles chromatic adaptation (Bradford) when white points differ.
/// Returns `None` if either primaries set is unknown.
pub const fn gamut_matrix(src: ColorPrimaries, dst: ColorPrimaries) -> Option<Mat3> {
    let src_xyz = match rgb_to_xyz(src) {
        Some(m) => m,
        None => return None,
    };
    let dst_xyz = match rgb_to_xyz(dst) {
        Some(m) => m,
        None => return None,
    };
    let dst_inv = match invert_3x3(&dst_xyz) {
        Some(m) => m,
        None => return None,
    };

    if src.needs_chromatic_adaptation(dst) {
        let adapt = bradford_adapt(src.white_point(), dst.white_point());
        Some(mul_3x3(&mul_3x3(&dst_inv, &adapt), &src_xyz))
    } else {
        Some(mul_3x3(&dst_inv, &src_xyz))
    }
}

// =========================================================================
// Linear algebra helpers (f32, all const)
// =========================================================================

/// Bradford chromatic adaptation matrix coefficients.
const BRADFORD: Mat3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// Precomputed Bradford inverse (const — avoids runtime inversion).
const BRADFORD_INV: Mat3 = match invert_3x3(&BRADFORD) {
    Some(m) => m,
    None => panic!("Bradford matrix is singular"),
};

/// Compute a Bradford chromatic adaptation matrix from `src` to `dst` white point.
const fn bradford_adapt(src_wp: (f32, f32), dst_wp: (f32, f32)) -> Mat3 {
    let src_xyz = [
        src_wp.0 / src_wp.1,
        1.0,
        (1.0 - src_wp.0 - src_wp.1) / src_wp.1,
    ];
    let dst_xyz = [
        dst_wp.0 / dst_wp.1,
        1.0,
        (1.0 - dst_wp.0 - dst_wp.1) / dst_wp.1,
    ];

    let src_lms = mul_mv(&BRADFORD, &src_xyz);
    let dst_lms = mul_mv(&BRADFORD, &dst_xyz);

    let scale = [
        [dst_lms[0] / src_lms[0], 0.0, 0.0],
        [0.0, dst_lms[1] / src_lms[1], 0.0],
        [0.0, 0.0, dst_lms[2] / src_lms[2]],
    ];

    mul_3x3(&mul_3x3(&BRADFORD_INV, &scale), &BRADFORD)
}

/// 3×3 matrix × 3-vector.
pub const fn mul_mv(m: &Mat3, v: &[f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// 3×3 matrix multiply.
pub const fn mul_3x3(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut r = [[0.0f32; 3]; 3];
    let mut i = 0;
    while i < 3 {
        let mut j = 0;
        while j < 3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            j += 1;
        }
        i += 1;
    }
    r
}

/// 3×3 matrix inverse.
pub const fn invert_3x3(m: &Mat3) -> Option<Mat3> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-10 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ])
}

/// Solve A × x = b for x (3×3 system via Cramer's rule).
const fn solve_3x3(a: &Mat3, b: &[f32; 3]) -> Option<[f32; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-10 {
        return None;
    }
    let inv = 1.0 / det;

    let x = (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]))
        * inv;
    let y = (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]))
        * inv;
    let z = (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        * inv;

    Some([x, y, z])
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_registry_entries_unique() {
        for i in 0..REGISTRY.len() {
            for j in (i + 1)..REGISTRY.len() {
                let a = &REGISTRY[i];
                let b = &REGISTRY[j];
                assert!(
                    a.primaries as u8 != b.primaries as u8 || a.transfer as u8 != b.transfer as u8,
                    "duplicate registry entry: {:?} and {:?}",
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn cicp_lookup_roundtrips() {
        for entry in REGISTRY {
            if let Some((cp, tc)) = entry.cicp {
                let found = find_by_cicp(cp, tc).unwrap();
                assert_eq!(found.primaries as u8, entry.primaries as u8);
                assert_eq!(found.transfer as u8, entry.transfer as u8);
            }
        }
    }

    #[test]
    fn named_lookup_roundtrips() {
        for entry in REGISTRY {
            if let Some(named) = entry.named {
                let found = find_by_named(named).unwrap();
                assert_eq!(found.primaries as u8, entry.primaries as u8);
                assert_eq!(found.transfer as u8, entry.transfer as u8);
            }
        }
    }

    #[test]
    fn primaries_transfer_lookup_roundtrips() {
        for entry in REGISTRY {
            let found = find_by_primaries_transfer(entry.primaries, entry.transfer).unwrap();
            assert_eq!(found.primaries as u8, entry.primaries as u8);
            assert_eq!(found.transfer as u8, entry.transfer as u8);
        }
    }

    #[test]
    fn rgb_to_xyz_white_is_d65_for_srgb() {
        let m = rgb_to_xyz(ColorPrimaries::Bt709).unwrap();
        // White = R+G+B columns summed = XYZ of D65
        let w = [
            m[0][0] + m[0][1] + m[0][2],
            m[1][0] + m[1][1] + m[1][2],
            m[2][0] + m[2][1] + m[2][2],
        ];
        // D65 XYZ ≈ (0.9505, 1.0000, 1.0890)
        assert!((w[0] - 0.9505).abs() < 0.002, "X: {}", w[0]);
        assert!((w[1] - 1.0).abs() < 0.001, "Y: {}", w[1]);
        assert!((w[2] - 1.089).abs() < 0.002, "Z: {}", w[2]);
    }

    #[test]
    fn gamut_matrix_identity_for_same_primaries() {
        let m = gamut_matrix(ColorPrimaries::Bt709, ColorPrimaries::Bt709).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (m[i][j] - expected).abs() < 1e-5,
                    "identity[{i}][{j}] = {}, expected {expected}",
                    m[i][j]
                );
            }
        }
    }

    #[test]
    fn gamut_matrix_preserves_white() {
        let pairs = [
            (ColorPrimaries::Bt709, ColorPrimaries::DisplayP3),
            (ColorPrimaries::DisplayP3, ColorPrimaries::Bt709),
            (ColorPrimaries::Bt709, ColorPrimaries::Bt2020),
            (ColorPrimaries::Bt2020, ColorPrimaries::Bt709),
            (ColorPrimaries::Bt709, ColorPrimaries::AdobeRgb),
            (ColorPrimaries::AdobeRgb, ColorPrimaries::Bt709),
            (ColorPrimaries::DisplayP3, ColorPrimaries::Bt2020),
        ];
        for (src, dst) in pairs {
            let m = gamut_matrix(src, dst).unwrap();
            let w = mul_mv(&m, &[1.0, 1.0, 1.0]);
            assert!(
                (w[0] - 1.0).abs() < 1e-4 && (w[1] - 1.0).abs() < 1e-4 && (w[2] - 1.0).abs() < 1e-4,
                "{src:?}→{dst:?}: white → ({}, {}, {})",
                w[0],
                w[1],
                w[2]
            );
        }
    }

    // Proof: gamut_matrix is usable in const context.
    const P3_TO_SRGB: Mat3 = match gamut_matrix(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709) {
        Some(m) => m,
        None => panic!("failed to compute P3→sRGB matrix"),
    };

    #[test]
    fn const_matrix_is_correct() {
        // Verify the const-computed matrix matches CSS Color 4
        let css = [
            [1.2249401_f32, -0.2249402, 0.0],
            [-0.0420570, 1.0420571, 0.0],
            [-0.0196376, -0.0786361, 1.0982736],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (P3_TO_SRGB[i][j] - css[i][j]).abs() < 5e-5,
                    "const P3→sRGB[{i}][{j}]: {}  CSS={}",
                    P3_TO_SRGB[i][j],
                    css[i][j]
                );
            }
        }
    }

    #[test]
    fn gamut_matrix_matches_hardcoded_p3_srgb() {
        // Cross-validate runtime-computed matrix against the CSS Color 4 reference
        let m = gamut_matrix(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709).unwrap();
        let css = [
            [1.2249401_f32, -0.2249402, 0.0],
            [-0.0420570, 1.0420571, 0.0],
            [-0.0196376, -0.0786361, 1.0982736],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m[i][j] - css[i][j]).abs() < 5e-5,
                    "P3→sRGB[{i}][{j}]: computed={}, CSS={}",
                    m[i][j],
                    css[i][j]
                );
            }
        }
    }

    #[test]
    fn gamut_matrix_roundtrip() {
        let fwd = gamut_matrix(ColorPrimaries::DisplayP3, ColorPrimaries::Bt709).unwrap();
        let inv = gamut_matrix(ColorPrimaries::Bt709, ColorPrimaries::DisplayP3).unwrap();
        let identity = mul_3x3(&inv, &fwd);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (identity[i][j] - expected).abs() < 1e-4,
                    "roundtrip[{i}][{j}] = {}, expected {expected}",
                    identity[i][j]
                );
            }
        }
    }

    // gamut_matrix_with_chromatic_adaptation test removed:
    // DciP3 primaries were removed (theatrical projection only).
    // Bradford adaptation is still exercised by any future cross-white-
    // point pair; currently all remaining primaries share D65.
}
