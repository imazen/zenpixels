//! One-call HDR → SDR conversion: BT.2020 → BT.709 matrix → Bt2446A
//! tone-map → soft chroma compression.
//!
//! See [`HdrToSdr`] for the full pipeline and defaults.

extern crate alloc;

use alloc::sync::Arc;

use crate::gamut::{GamutMatrix, apply_matrix_f32, conversion_matrix};
use crate::hdr::Bt2446A;
use crate::hdr::gamut_compress::{GamutBoundaryLut, SoftCompress};
use crate::oklab;
use zenpixels::ColorPrimaries;

/// One-call HDR → SDR conversion.
///
/// # Pipeline
///
/// Per pixel (in both [`apply_strip`](Self::apply_strip) and
/// [`apply_rgb`](Self::apply_rgb)):
///
/// 1. **BT.2020 → BT.709 matrix.** The input is assumed to be linear-light
///    BT.2020 HDR, source-normalized so `1.0 = source_peak_nits`. The
///    matrix converts the primaries (no transfer / no clamp); negative
///    values can appear briefly on saturated BT.2020 primaries that fall
///    outside the BT.709 hull and are addressed by the gamut compression
///    stage below.
/// 2. **BT.2446 Method A tone-map.** Applies the
///    [`Bt2446A`](crate::hdr::Bt2446A) curve constructed with
///    `(source_peak_nits, target_peak_nits)`. Output is target-normalized
///    (`1.0 = target_peak_nits`) linear-light SDR.
/// 3. **OKLch soft gamut compression.** Pulls residual out-of-gamut
///    excursions back into the BT.709 unit cube using a hue-preserving
///    rational knee curve. The knee defaults to `0.9` (compression kicks in
///    at 90 % of the max in-gamut chroma).
///
/// # Defaults
///
/// - `target_peak_nits = 100.0` (SDR reference white).
/// - `gamut_knee = 0.9`.
///
/// # Why this composition
///
/// The 76-sample HDR shootout (see `zentone`'s
/// `benchmarks/hdr_tone_map_shootout_full_2026-06-20.md`) measured 20
/// curve cells against producer-graded SDR. BT.2446 Method A was the
/// median winner (ΔE2000 = 3.17 vs producer SDR; the next-best curve,
/// BT.2390, scored 6.09). The matrix step happens *before* the curve so
/// the tone-map sees the BT.709 primaries the compressed output will
/// live in, and the soft chroma stage happens *last* so any matrix-
/// induced or curve-induced excursions stay hue-preserved.
///
/// # RGB-only
///
/// The strip variant is `&mut [[f32; 3]]`. For RGBA, run the alpha
/// channel separately — tone-mapping it is meaningless.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "hdr-experimental")]
/// # {
/// use zenpixels_convert::hdr::HdrToSdr;
///
/// // 1000-nit HDR source → 100-nit SDR target (default).
/// // Input is source-normalized: 1.0 = source_peak_nits = 1000 nits.
/// let converter = HdrToSdr::new(1000.0);
/// let mut pixels = vec![
///     [1.0_f32, 0.6, 0.3],
///     [0.1, 0.1, 0.1],
/// ];
/// converter.apply_strip(&mut pixels);
/// for px in &pixels {
///     for &c in px {
///         assert!((0.0..=1.0).contains(&c));
///     }
/// }
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct HdrToSdr {
    source_peak_nits: f32,
    target_peak_nits: f32,
    gamut_knee: f32,
    bt2446a: Bt2446A,
    bt2020_to_bt709: GamutMatrix,
    gamut_lut: Arc<GamutBoundaryLut>,
    bt709_m1: GamutMatrix,
    bt709_m1_inv: GamutMatrix,
}

impl HdrToSdr {
    /// Production defaults: `target_peak_nits = 100.0`, `gamut_knee = 0.9`.
    #[must_use]
    pub fn new(source_peak_nits: f32) -> Self {
        Self::with_params(source_peak_nits, 100.0, 0.9)
    }

    /// Full constructor with explicit target peak and gamut knee.
    ///
    /// `target_peak_nits`: SDR display peak luminance (typically 100).
    /// `gamut_knee`: fraction of max chroma at which the soft chroma
    /// compression kicks in (`0.0`–`1.0`, typical `0.9`).
    #[must_use]
    pub fn with_params(source_peak_nits: f32, target_peak_nits: f32, gamut_knee: f32) -> Self {
        let bt2446a = Bt2446A::new(source_peak_nits, target_peak_nits);
        let bt2020_to_bt709 = conversion_matrix(ColorPrimaries::Bt2020, ColorPrimaries::Bt709)
            .expect("BT.2020 and BT.709 are both well-known primaries");
        let bt709_m1 = oklab::rgb_to_lms_matrix(ColorPrimaries::Bt709)
            .expect("BT.709 has a defined LMS matrix");
        let bt709_m1_inv = oklab::lms_to_rgb_matrix(ColorPrimaries::Bt709)
            .expect("BT.709 has a defined inverse LMS matrix");
        let gamut_lut = Arc::new(GamutBoundaryLut::new(&bt709_m1_inv));
        Self {
            source_peak_nits,
            target_peak_nits,
            gamut_knee,
            bt2446a,
            bt2020_to_bt709,
            gamut_lut,
            bt709_m1,
            bt709_m1_inv,
        }
    }

    /// Source peak luminance (nits / cd/m²).
    #[inline]
    #[must_use]
    pub fn source_peak_nits(&self) -> f32 {
        self.source_peak_nits
    }

    /// Target peak luminance (nits / cd/m²).
    #[inline]
    #[must_use]
    pub fn target_peak_nits(&self) -> f32 {
        self.target_peak_nits
    }

    /// Gamut soft-compression knee (`0.0`–`1.0`).
    #[inline]
    #[must_use]
    pub fn gamut_knee(&self) -> f32 {
        self.gamut_knee
    }

    /// Borrow a [`SoftCompress`] view of the configured gamut compressor
    /// (constructs the wrapper on demand). Useful for callers that want
    /// to reuse the same LUT for downstream work without rebuilding.
    #[must_use]
    pub fn soft_compress(&self) -> SoftCompress {
        SoftCompress::from_matrices(&self.bt709_m1, &self.bt709_m1_inv, self.gamut_knee)
    }

    /// Apply the full HDR → SDR pipeline to a strip of linear `RGB f32`
    /// pixels in place.
    ///
    /// Input: linear-light BT.2020 RGB, source-normalized
    /// (`1.0 = source_peak_nits`). Output: linear-light BT.709 RGB,
    /// target-normalized (`1.0 = target_peak_nits`), guaranteed finite and
    /// in `[0, 1]` per channel — non-finite inputs (NaN / ±Inf) are
    /// scrubbed to `0` before the pipeline runs, and the post-compress
    /// stage clamps the final output to absorb the f32 EOTF / OKLab
    /// roundtrip noise that can push a saturated pixel ~1e-4 above 1.0.
    pub fn apply_strip(&self, rgb: &mut [[f32; 3]]) {
        // Step 0 — scrub non-finite inputs so the f32 transcendental chain
        // doesn't propagate NaN / ±Inf through the pipeline. Negatives are
        // also clamped to 0 because the upstream BT.2020 → BT.709 matrix
        // can introduce small negatives only for in-gamut transcoding;
        // genuinely-negative HDR pixels are not a meaningful input.
        for px in rgb.iter_mut() {
            for c in px.iter_mut() {
                if !c.is_finite() || *c < 0.0 {
                    *c = 0.0;
                }
            }
        }
        // Step 1 — BT.2020 → BT.709 matrix.
        for px in rgb.iter_mut() {
            apply_matrix_f32(px, &self.bt2020_to_bt709);
        }
        // Step 2 — Bt2446A tone-map (SIMD strip).
        self.bt2446a.map_strip_simd(rgb);
        // Step 3 — soft chroma compression in OKLch.
        // Use the cached LUT directly to avoid rebuilding the wrapper.
        soft_compress_strip_with_lut(
            rgb,
            &self.gamut_lut,
            &self.bt709_m1,
            &self.bt709_m1_inv,
            self.gamut_knee,
        );
        // Step 4 — hard clamp to `[0, 1]` to absorb f32 epsilon-level
        // overshoot at the saturated end (BT.2446-A's `f * (b_p - y_p)`
        // chain can push a near-peak pixel to ~1.0004 even though the
        // spec algorithm clamps internally; OKLab round-trip in the
        // compress stage adds a few more ULPs). Hard clamp here keeps
        // the `apply_strip` postcondition crisp.
        for px in rgb.iter_mut() {
            for c in px.iter_mut() {
                if !c.is_finite() {
                    *c = 0.0;
                } else {
                    *c = c.clamp(0.0, 1.0);
                }
            }
        }
    }

    /// Per-pixel variant of [`apply_strip`](Self::apply_strip).
    #[must_use]
    pub fn apply_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        let mut strip = [rgb];
        self.apply_strip(&mut strip);
        strip[0]
    }
}

/// In-place soft chroma compression on a strip, using a preconstructed LUT
/// and primary matrices. Used internally by [`HdrToSdr`] and friends.
#[inline]
fn soft_compress_strip_with_lut(
    rgb: &mut [[f32; 3]],
    lut: &GamutBoundaryLut,
    m1: &GamutMatrix,
    m1_inv: &GamutMatrix,
    knee: f32,
) {
    for px in rgb.iter_mut() {
        let lab = oklab::rgb_to_oklab(px[0], px[1], px[2], m1);
        let l = lab[0];
        let mut a = lab[1];
        let mut b = lab[2];

        let c = (a * a + b * b).sqrt();
        if c < 1e-10 {
            continue;
        }
        let h = b.atan2(a);
        let max_c = lut.max_chroma(l, h);
        if max_c < 1e-10 {
            a = 0.0;
            b = 0.0;
        } else {
            let knee_c = knee * max_c;
            if c > knee_c {
                let range = max_c - knee_c;
                let excess = c - knee_c;
                let compressed_c = knee_c + range * excess / (excess + range);
                let scale = compressed_c / c;
                a *= scale;
                b *= scale;
            }
        }
        let out = oklab::oklab_to_rgb(l, a, b, m1_inv);
        *px = out;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite_in_unit(out: [f32; 3]) {
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "ch{i}={v} out of [0,1]"
            );
        }
    }

    #[test]
    fn defaults_match_spec() {
        let c = HdrToSdr::new(1000.0);
        assert_eq!(c.source_peak_nits(), 1000.0);
        assert_eq!(c.target_peak_nits(), 100.0);
        assert!((c.gamut_knee() - 0.9).abs() < 1e-7);
    }

    #[test]
    fn zero_input_is_zero_output() {
        let c = HdrToSdr::new(1000.0);
        let out = c.apply_rgb([0.0, 0.0, 0.0]);
        for v in out {
            assert!(
                v.is_finite() && v.abs() < 1e-4,
                "black should stay black: {v}"
            );
        }
    }

    #[test]
    fn empty_strip_is_noop() {
        let c = HdrToSdr::new(1000.0);
        let mut strip: alloc::vec::Vec<[f32; 3]> = alloc::vec::Vec::new();
        c.apply_strip(&mut strip);
        assert!(strip.is_empty());
    }

    #[test]
    fn apply_strip_matches_apply_rgb() {
        let c = HdrToSdr::new(2000.0);
        let pixels = [
            [0.0_f32, 0.0, 0.0],
            [0.15, 0.25, 0.05],
            [0.5, 0.2, 0.1],
            [1.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ];
        let mut strip = pixels.to_vec();
        c.apply_strip(&mut strip);
        for (i, &p) in pixels.iter().enumerate() {
            let expected = c.apply_rgb(p);
            for k in 0..3 {
                let diff = (strip[i][k] - expected[k]).abs();
                assert!(
                    diff < 1e-5,
                    "strip vs per-pixel diverge at px[{i}]ch[{k}]: {} vs {} (diff {})",
                    strip[i][k],
                    expected[k],
                    diff
                );
            }
        }
    }

    #[test]
    fn pipeline_finite_on_extreme_inputs() {
        let c = HdrToSdr::with_params(10_000.0, 100.0, 0.9);
        // Pathological inputs: NaN, Inf, large positives, negatives, zero,
        // peak, slightly above peak.
        let pixels = [
            [0.0_f32, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 0.0],
            [-0.1, 0.5, 0.2],
            [f32::INFINITY, 0.5, 0.5],
            [0.5, f32::NAN, 0.5],
            [0.5, 0.5, -f32::INFINITY],
            [1e20, 1e20, 1e20],
            [1e-30, 1e-30, 1e-30],
        ];
        let mut strip = pixels.to_vec();
        c.apply_strip(&mut strip);
        // The pipeline must not propagate NaN or infinities, and must keep
        // every channel within [0, 1] regardless of input.
        for (i, px) in strip.iter().enumerate() {
            for (k, &v) in px.iter().enumerate() {
                assert!(
                    v.is_finite() && (0.0..=1.0).contains(&v),
                    "pixel {i} ch{k} = {v} out of [0,1] (input {:?})",
                    pixels[i]
                );
            }
        }
    }

    #[test]
    fn saturated_red_hdr_stays_in_srgb_gamut() {
        let c = HdrToSdr::new(1000.0);
        // Pure BT.2020 saturated red — its BT.709 primary projection has
        // negative G/B by construction; the soft-compress stage must
        // pull every channel back into [0, 1] with red dominant.
        let out = c.apply_rgb([1.0, 0.0, 0.0]);
        finite_in_unit(out);
        assert!(
            out[0] > out[1] && out[0] > out[2],
            "red should remain dominant: {out:?}"
        );
    }

    #[test]
    fn hdr_mid_grey_lands_in_sensible_sdr_range() {
        // Diffuse-white HDR (= 1.0 source-norm = 1000 nits at default
        // source peak). After tone-map + soft compress, the output's
        // luminance should land in a sensible SDR mid-bright range —
        // not under- or over-exposed.
        let c = HdrToSdr::new(1000.0);
        let out = c.apply_rgb([1.0, 1.0, 1.0]);
        finite_in_unit(out);
        // Peak HDR maps near SDR peak (BT.2446-A spec behavior).
        for v in out {
            assert!(
                v > 0.85 && v <= 1.0,
                "peak HDR should map near SDR peak: got {v}"
            );
        }
        // A more pertinent mid: HDR 0.18 maps to ~SDR 0.37 (linear),
        // which is sensible mid-grey.
        let mid = c.apply_rgb([0.18, 0.18, 0.18]);
        finite_in_unit(mid);
        for v in mid {
            assert!(
                v > 0.25 && v < 0.55,
                "HDR 0.18 mid should land in sensible SDR mid range: got {v}"
            );
        }
    }

    #[test]
    fn source_equals_target_is_near_identity() {
        // Source == target: Bt2446A is NOT exact identity (its 1.0770 ·
        // y_p boost in the low segment is intentional and applies
        // regardless of the rho ratio), but the output should stay
        // within ~25% of the input for mid-grey.
        let c = HdrToSdr::with_params(100.0, 100.0, 0.9);
        let rgb = [0.4_f32, 0.4, 0.4];
        let out = c.apply_rgb(rgb);
        finite_in_unit(out);
        // Neutral input stays neutral (within OKLab roundtrip noise).
        assert!((out[0] - out[1]).abs() < 5e-3, "channels diverge: {out:?}");
        assert!((out[1] - out[2]).abs() < 5e-3, "channels diverge: {out:?}");
        for k in 0..3 {
            assert!(
                (out[k] - rgb[k]).abs() < 0.25,
                "source==target output strayed far from input: out[{k}]={} input[{k}]={}",
                out[k],
                rgb[k]
            );
        }
    }

    #[test]
    fn knee_change_affects_near_edge_colors() {
        // Sanity-check that varying the gamut knee actually does
        // something: an aggressive (knee=0.3) HdrToSdr should diverge
        // from the default (knee=0.9) on near-gamut-edge red.
        let rgb = [0.95_f32, 0.05, 0.05];
        let default = HdrToSdr::new(1000.0).apply_rgb(rgb);
        let custom = HdrToSdr::with_params(1000.0, 100.0, 0.30).apply_rgb(rgb);
        finite_in_unit(default);
        finite_in_unit(custom);
        let mut differs = false;
        for k in 0..3 {
            if (default[k] - custom[k]).abs() > 1e-3 {
                differs = true;
                break;
            }
        }
        assert!(
            differs,
            "expected different output with different knee: default={default:?} custom={custom:?}"
        );
    }

    #[test]
    fn neutral_input_stays_finite_across_decades() {
        let c = HdrToSdr::new(4000.0);
        for shift in [0_i32, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10] {
            // shift==0  -> 1.0 source-norm (4000 nits)
            // shift=-10 -> ~1e-3 source-norm (~4 nits)
            let v = 2.0_f32.powi(shift);
            let out = c.apply_rgb([v, v, v]);
            finite_in_unit(out);
        }
    }
}
