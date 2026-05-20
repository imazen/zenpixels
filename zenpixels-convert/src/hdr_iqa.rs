#![allow(clippy::excessive_precision)]
//! HDR-IQA primitives — PU encoding (Mantiuk PU21) + named luma
//! coefficients for downstream perceptual quality metrics.
//!
//! Niche surface: this module exists so zensim / zenmetrics /
//! zenanalyze can score HDR image pairs through SDR-trained metrics
//! by front-ending them with PU-encoded luminance. The whole module
//! is `#[doc(hidden)]` because:
//!
//! - PU encoding is HDR-IQA-specific, not a general color-conversion
//!   primitive (it doesn't round-trip through normal pixel pipelines).
//! - The API contract is "experimental until validated against
//!   AIC-HDR2025 / UPIQ" (see imazen/zenmetrics#13).
//! - linear-srgb deliberately stays free of HDR-IQA-specific
//!   conversions; zenpixels-convert is the higher-level layer where
//!   perceptual converters naturally live.
//!
//! ## Reference
//!
//! Mantiuk, R. K., Azimi, M. (2021).
//! *PU21: A novel perceptually uniform encoding for adapting existing
//! quality metrics for HDR.* Picture Coding Symposium (PCS).
//! Reference impl: <https://github.com/gfxdisp/pu21>.
//!
//! ## Output range
//!
//! `pu_encode(Y_cd_m2)` returns a value in roughly `[0, 530]` for
//! `Y ∈ [0.005, 10_000]` cd/m². For downstream IQA work that wants a
//! normalized `[0, 1]` scale, divide by [`PU_PEAK`].
//!
//! ## Anchor identities
//!
//! | Input | PU value |
//! |---|---|
//! | 0.005 cd/m² (sub-black) | clamped to 0 |
//! | 1 cd/m² | ≈ 84.4 |
//! | 10 cd/m² | ≈ 158.5 |
//! | 80 cd/m² (legacy SDR display peak) | ≈ 250.5 |
//! | 100 cd/m² (BT.1886 SDR peak) | ≈ 261.8 |
//! | 203 cd/m² (BT.2408 HDR reference white) | ≈ 298.8 ([`PU_REF_WHITE_BT2408`]) |
//! | 1 000 cd/m² (HDR10 highlight) | ≈ 388.1 |
//! | 4 000 cd/m² | ≈ 468.5 |
//! | 10 000 cd/m² (PQ peak) | ≈ 520.5 ([`PU_PEAK`]) |

// `f32::powf` lives in `std` and behind `libm` in `no_std`. Mirror the
// rest of the crate (see fast_gamut.rs) — std-only at the moment. If
// no_std support is added later, gate this on `libm::Float`.

// =============================================================================
// PU21 constants (Mantiuk & Azimi 2021)
// =============================================================================
//
// PU21(Y) = par[7] * (((par[1] + par[2]·Y^par[4]) / (1 + par[3]·Y^par[4]))^par[5] - par[6])
//
// where Y is absolute luminance in cd/m². Coefficients lifted verbatim
// from gfxdisp/pu21 (banding_glare model — the variant tuned for IQA
// against both subtle banding and high-luminance flare).

const PU21_P1: f32 = 1.070_275_272;
const PU21_P2: f32 = 0.408_827_393_2;
const PU21_P3: f32 = 0.153_224_308;
const PU21_P4: f32 = 0.252_032_616_8;
const PU21_P5: f32 = 1.063_512_885;
const PU21_P6: f32 = 1.141_150_47;
const PU21_P7: f32 = 521.452_748_4;

/// PU value at the HDR10 peak luminance (10 000 cd/m²).
///
/// Use as the normaliser when a `[0, 1]` PU output is desired.
pub const PU_PEAK: f32 = 520.467_25;

/// PU value at BT.2408 HDR reference white (203 cd/m²).
///
/// Anchor identity for verifying PU implementations and for downstream
/// metric calibration.
pub const PU_REF_WHITE_BT2408: f32 = 298.761_14;

/// Lowest luminance the PU curve accepts before clamping to zero.
///
/// Matches gfxdisp/pu21's reference clamp.
pub const PU_LUMINANCE_MIN_CD_M2: f32 = 0.005;

/// Highest luminance the PU curve is calibrated for.
///
/// Coincides with the HDR10 / PQ ST 2084 peak. Inputs above this are
/// permitted (the formula remains monotone) but extrapolate beyond the
/// published validation range.
pub const PU_LUMINANCE_MAX_CD_M2: f32 = 10_000.0;

// =============================================================================
// Luma coefficients
// =============================================================================

/// BT.2020 NCL (non-constant-luminance) luma coefficients.
///
/// Used by HDR10, PQ, BT.2100, and the HDR gain-map specs. Order is
/// `[R, G, B]`; sums to 1.0.
pub const BT2020_NCL_LUMA: [f32; 3] = [0.2627, 0.6780, 0.0593];

/// BT.709 luma coefficients.
///
/// Used by sRGB-primary YCbCr at HDTV resolutions and as the default
/// for SDR JPEG / JPEG XL on Rec.709 primaries. Order is `[R, G, B]`;
/// sums to 1.0.
pub const BT709_LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// BT.601 luma coefficients (legacy SDTV).
///
/// Used by SDTV YCbCr and the canonical JPEG color-space matrix when
/// the encoder targets pre-HD content. Order is `[R, G, B]`;
/// sums to 1.0.
pub const BT601_LUMA: [f32; 3] = [0.299, 0.587, 0.114];

// =============================================================================
// Scalar PU
// =============================================================================

/// Encode absolute luminance (cd/m²) to PU space.
///
/// Inputs at or below [`PU_LUMINANCE_MIN_CD_M2`] (0.005 cd/m²) clamp to 0.
/// The curve is monotone-increasing across its valid range and
/// extrapolates smoothly above [`PU_LUMINANCE_MAX_CD_M2`].
#[inline]
pub fn pu_encode(luminance_cd_m2: f32) -> f32 {
    if luminance_cd_m2 <= PU_LUMINANCE_MIN_CD_M2 {
        return 0.0;
    }
    let yp = luminance_cd_m2.powf(PU21_P4);
    let num = PU21_P1 + PU21_P2 * yp;
    let den = 1.0 + PU21_P3 * yp;
    let inner = num / den;
    let val = PU21_P7 * (inner.powf(PU21_P5) - PU21_P6);
    val.max(0.0)
}

/// Decode a PU value back to absolute luminance (cd/m²).
///
/// Inverse of [`pu_encode`]. The closed-form inverse is well-defined
/// over the valid PU range `[0, ~530]`; inputs outside that clamp to
/// the corresponding luminance bounds.
#[inline]
pub fn pu_decode(pu: f32) -> f32 {
    if pu <= 0.0 {
        return 0.0;
    }
    let inner = ((pu / PU21_P7) + PU21_P6).powf(1.0 / PU21_P5);
    let num = inner - PU21_P1;
    let den = PU21_P2 - PU21_P3 * inner;
    if den <= 0.0 || num <= 0.0 {
        return PU_LUMINANCE_MIN_CD_M2;
    }
    (num / den).powf(1.0 / PU21_P4)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn pu_encode_f64(y: f64) -> f64 {
        if y <= PU_LUMINANCE_MIN_CD_M2 as f64 {
            return 0.0;
        }
        let yp = y.powf(PU21_P4 as f64);
        let num = (PU21_P1 as f64) + (PU21_P2 as f64) * yp;
        let den = 1.0 + (PU21_P3 as f64) * yp;
        let inner = num / den;
        let val = (PU21_P7 as f64) * (inner.powf(PU21_P5 as f64) - (PU21_P6 as f64));
        val.max(0.0)
    }

    #[test]
    fn sub_black_clamps_to_zero() {
        assert_eq!(pu_encode(-100.0), 0.0);
        assert_eq!(pu_encode(0.0), 0.0);
        assert_eq!(pu_encode(PU_LUMINANCE_MIN_CD_M2), 0.0);
    }

    #[test]
    fn bt2408_reference_white_anchor() {
        let pu = pu_encode(203.0);
        assert!(
            (pu - PU_REF_WHITE_BT2408).abs() < 0.05,
            "PU(203) = {pu}, expected ≈ {PU_REF_WHITE_BT2408}"
        );
    }

    #[test]
    fn hdr_peak_anchor() {
        let pu = pu_encode(PU_LUMINANCE_MAX_CD_M2);
        assert!(
            (pu - PU_PEAK).abs() < 0.05,
            "PU(10000) = {pu}, expected ≈ {PU_PEAK}"
        );
    }

    #[test]
    fn published_curve_anchors() {
        let cases = [
            (1.0_f32, 84.4_f32),
            (10.0, 158.5),
            (80.0, 250.5),
            (100.0, 261.8),
            (500.0, 348.5),
            (1_000.0, 388.1),
            (4_000.0, 468.5),
        ];
        for (input, expected) in cases {
            let pu = pu_encode(input);
            assert!(
                (pu - expected).abs() < 0.1,
                "PU({input}) = {pu}, expected ≈ {expected}"
            );
        }
    }

    #[test]
    fn monotone_across_decade_grid() {
        let grid = [0.01_f32, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0];
        let mut prev = pu_encode(grid[0]);
        for &y in &grid[1..] {
            let pu = pu_encode(y);
            assert!(
                pu > prev,
                "PU not monotone at Y = {y} (prev = {prev}, pu = {pu})"
            );
            prev = pu;
        }
    }

    #[test]
    fn round_trip_within_05_pct() {
        for &y in &[
            0.01_f32, 0.1, 1.0, 10.0, 80.0, 100.0, 203.0, 500.0, 1_000.0, 4_000.0, 10_000.0,
        ] {
            let pu = pu_encode(y);
            let back = pu_decode(pu);
            let rel_err = ((back - y) / y).abs();
            assert!(
                rel_err < 0.005,
                "Round-trip Y = {y}: PU = {pu}, decoded = {back}, rel_err = {rel_err}"
            );
        }
    }

    #[test]
    fn f32_path_matches_f64_reference() {
        let grid: Vec<f32> = (0..1000)
            .map(|i| {
                let t = i as f32 / 999.0;
                let log_min = PU_LUMINANCE_MIN_CD_M2.ln();
                let log_max = PU_LUMINANCE_MAX_CD_M2.ln();
                (log_min + (log_max - log_min) * t).exp()
            })
            .collect();
        let mut max_err = 0.0_f32;
        for &y in &grid {
            let fast = pu_encode(y);
            let slow = pu_encode_f64(y as f64) as f32;
            let err = (fast - slow).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 5e-3,
            "f32 PU max error vs f64 = {max_err}, expected < 5e-3"
        );
    }

    #[test]
    fn luma_sets_sum_to_one() {
        for (name, set) in [
            ("BT2020_NCL", BT2020_NCL_LUMA),
            ("BT709", BT709_LUMA),
            ("BT601", BT601_LUMA),
        ] {
            let sum: f32 = set.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "{name} luma coefficients sum to {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn green_is_largest_in_every_luma_standard() {
        for set in [BT2020_NCL_LUMA, BT709_LUMA, BT601_LUMA] {
            assert!(set[1] > set[0] && set[1] > set[2]);
        }
    }
}
