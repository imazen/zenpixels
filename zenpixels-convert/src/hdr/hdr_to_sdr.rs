//! One-call HDR → SDR conversion. The pipeline is **`(source → BT.2020)
//! → Bt2446A → (BT.2020 → target_primaries) → SoftCompress (target OKLch)`**.
//!
//! See [`HdrToSdr`] for the full pipeline, source/target primary handling,
//! and defaults.

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
/// 1. **Source primaries → BT.2020 matrix** *(skipped if the source already
///    is BT.2020).* Input is linear-light source RGB, source-normalized so
///    `1.0 = source_peak_nits`. The matrix converts the primaries (no
///    transfer, no clamp) so the next step sees the BT.2020 RGB it was
///    designed for.
/// 2. **BT.2446 Method A tone-map.** Applies the
///    [`Bt2446A`](crate::hdr::Bt2446A) curve constructed with
///    `(source_peak_nits, target_peak_nits)`. ITU-R BT.2446-1 specifies the
///    curve in **BT.2020** R'G'B' / Y'Cb'Cr' — its luma weights
///    (`0.2627 / 0.6780 / 0.0593`) and YCbCr↔RGB coefficients are BT.2020-
///    specific. Feeding it BT.709 RGB produces a systematic hue shift on
///    saturated content, which is why it sits *before* the gamut matrix.
///    Output is BT.2020 RGB, target-normalized (`1.0 = target_peak_nits`).
/// 3. **BT.2020 → target primaries matrix** *(skipped when the target is
///    BT.2020 — wide-gamut output mode).* Gamut-converts the tone-mapped
///    SDR pixels into the target working space. Negatives can appear
///    briefly on saturated BT.2020 primaries that fall outside the target
///    hull and are absorbed by the gamut compression stage below.
/// 4. **OKLch soft gamut compression** *(skipped when the target is
///    BT.2020).* Pulls residual out-of-gamut excursions back into the
///    target unit cube using a hue-preserving rational knee curve. The
///    knee defaults to `0.9` (compression kicks in at 90 % of the max
///    in-gamut chroma).
///
/// # Source primaries
///
/// - [`new`](Self::new) assumes the source is **BT.2020** (the typical
///   HDR10 / HLG case). No source-gamut conversion step runs.
/// - [`with_source_primaries`](Self::with_source_primaries),
///   [`with_io`](Self::with_io), and [`with_params`](Self::with_params)
///   accept arbitrary [`ColorPrimaries`]. The constructor caches an extra
///   `source_to_bt2020` matrix that runs as step 1 above; the BT.2446
///   curve still operates in BT.2020 as it was designed. Display P3 HDR
///   (e.g. Apple ProRAW) and other non-BT.2020 HDR sources are supported
///   this way without distorting the curve.
///
/// # Target primaries
///
/// - [`new`](Self::new) and
///   [`with_source_primaries`](Self::with_source_primaries) target
///   **BT.709** (the default SDR output, sRGB primaries).
/// - [`with_io`](Self::with_io) and [`with_params`](Self::with_params)
///   accept an arbitrary target. When `target_primaries ==
///   ColorPrimaries::Bt2020`, steps 3 and 4 are **no-ops** — output stays
///   in BT.2020 linear-light at the target peak. This is the wide-gamut
///   "as lossless as physics allows" output mode: the only inherently
///   lossy stage is BT.2446-A's tone curve; no gamut narrowing or chroma
///   compression occurs. Use this when the downstream container can
///   carry BT.2020 primaries (BT.2020 PNG with cICP, HDR10 at SDR
///   luminance, etc.).
///
/// # Defaults
///
/// - `target_peak_nits = 100.0` (SDR reference white).
/// - `gamut_knee = 0.9`.
/// - `target_primaries = ColorPrimaries::Bt709`.
///
/// # Why this composition
///
/// The 76-sample HDR shootout (see `zentone`'s
/// `benchmarks/hdr_tone_map_shootout_full_2026-06-20.md`) measured 20
/// curve cells against producer-graded SDR. BT.2446 Method A was the
/// median winner (ΔE2000 = 3.17 vs producer SDR; the next-best curve,
/// BT.2390, scored 6.09). The tone-map runs *before* the gamut matrix so
/// the BT.2020 luma weights inside BT.2446-A's Y' / Cb' / Cr' arithmetic
/// see the BT.2020 RGB they were derived against, and the soft chroma
/// stage happens *last* so any matrix-induced or curve-induced excursions
/// stay hue-preserved in the final target working space.
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
/// // 1000-nit HDR source → 100-nit SDR target (default BT.709).
/// // Input is source-normalized: 1.0 = source_peak_nits = 1000 nits.
/// // BT.2020 source is assumed.
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
///
/// ## Wide-gamut output mode
///
/// Pass `target_primaries == ColorPrimaries::Bt2020` to keep the output
/// in BT.2020 — only the BT.2446-A tone curve runs; the gamut matrix and
/// soft-compress stages are skipped, so no chroma is narrowed.
///
/// ```
/// # #[cfg(feature = "hdr-experimental")]
/// # {
/// use zenpixels_convert::hdr::HdrToSdr;
/// use zenpixels::ColorPrimaries;
///
/// // 1000-nit BT.2020 HDR → 100-nit BT.2020 SDR (wide-gamut, no chroma loss).
/// let converter = HdrToSdr::with_io(
///     1000.0,
///     ColorPrimaries::Bt2020,
///     ColorPrimaries::Bt2020,
/// );
/// assert_eq!(converter.target_primaries(), ColorPrimaries::Bt2020);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct HdrToSdr {
    source_peak_nits: f32,
    target_peak_nits: f32,
    gamut_knee: f32,
    bt2446a: Bt2446A,
    /// `None` when the source is BT.2020 (no-op step 1).
    source_to_bt2020: Option<GamutMatrix>,
    /// `None` when the target is BT.2020 (wide-gamut output — no-op step 3).
    bt2020_to_target: Option<GamutMatrix>,
    /// Tracked for introspection and correctness pins. The presence of
    /// `bt2020_to_target` / `gamut_lut` / `target_m1*` mirrors whether this
    /// equals BT.2020.
    target_primaries: ColorPrimaries,
    /// `None` when the target is BT.2020 (no soft-compress step).
    gamut_lut: Option<Arc<GamutBoundaryLut>>,
    /// `None` when the target is BT.2020 (no soft-compress step).
    target_m1: Option<GamutMatrix>,
    /// `None` when the target is BT.2020 (no soft-compress step).
    target_m1_inv: Option<GamutMatrix>,
}

impl HdrToSdr {
    /// BT.2020 HDR source → BT.709 SDR target. `target_peak_nits = 100.0`,
    /// `gamut_knee = 0.9`.
    #[must_use]
    pub fn new(source_peak_nits: f32) -> Self {
        Self::with_params(
            source_peak_nits,
            ColorPrimaries::Bt2020,
            ColorPrimaries::Bt709,
            100.0,
            0.9,
        )
    }

    /// Arbitrary source primaries → BT.709 SDR target;
    /// `target_peak_nits = 100.0`, `gamut_knee = 0.9`.
    ///
    /// When `source_primaries != ColorPrimaries::Bt2020` the pipeline
    /// inserts a `source → BT.2020` matrix step before the BT.2446 curve so
    /// the curve still operates on the BT.2020 RGB it was designed for.
    #[must_use]
    pub fn with_source_primaries(source_peak_nits: f32, source_primaries: ColorPrimaries) -> Self {
        Self::with_params(
            source_peak_nits,
            source_primaries,
            ColorPrimaries::Bt709,
            100.0,
            0.9,
        )
    }

    /// Arbitrary source and target primaries; `target_peak_nits = 100.0`,
    /// `gamut_knee = 0.9`.
    ///
    /// When `target_primaries == ColorPrimaries::Bt2020`, the gamut matrix
    /// and soft-compress steps are skipped — the only lossy stage is the
    /// BT.2446-A tone curve. Use this for wide-gamut SDR containers
    /// (BT.2020 PNG with cICP tag, HDR10 at SDR luminance, etc.).
    #[must_use]
    pub fn with_io(
        source_peak_nits: f32,
        source_primaries: ColorPrimaries,
        target_primaries: ColorPrimaries,
    ) -> Self {
        Self::with_params(
            source_peak_nits,
            source_primaries,
            target_primaries,
            100.0,
            0.9,
        )
    }

    /// Full constructor with explicit source primaries, target primaries,
    /// target peak, and gamut knee.
    ///
    /// `source_primaries`: primaries of the input pixels (BT.2020 is the
    /// typical HDR10/HLG case; Display P3 covers Apple HDR).
    /// `target_primaries`: primaries of the output. `BT.709` is the SDR
    /// default; `BT.2020` is the wide-gamut "no chroma loss" mode (the
    /// gamut matrix + soft-compress stages are skipped).
    /// `target_peak_nits`: SDR display peak luminance (typically 100).
    /// `gamut_knee`: fraction of max chroma at which the soft chroma
    /// compression kicks in (`0.0`–`1.0`, typical `0.9`). Ignored when
    /// the target is BT.2020.
    #[must_use]
    pub fn with_params(
        source_peak_nits: f32,
        source_primaries: ColorPrimaries,
        target_primaries: ColorPrimaries,
        target_peak_nits: f32,
        gamut_knee: f32,
    ) -> Self {
        let bt2446a = Bt2446A::new(source_peak_nits, target_peak_nits);

        let source_to_bt2020 = if matches!(source_primaries, ColorPrimaries::Bt2020) {
            None
        } else {
            conversion_matrix(source_primaries, ColorPrimaries::Bt2020)
        };

        let (bt2020_to_target, gamut_lut, target_m1, target_m1_inv) =
            if matches!(target_primaries, ColorPrimaries::Bt2020) {
                // Wide-gamut output: skip the gamut matrix and the OKLch
                // soft-compress entirely. The BT.2446-A curve already
                // operates in BT.2020 so its output IS the answer.
                (None, None, None, None)
            } else {
                let m_gamut = conversion_matrix(ColorPrimaries::Bt2020, target_primaries)
                    .expect("BT.2020 and target are both well-known primaries");
                let m1 = oklab::rgb_to_lms_matrix(target_primaries)
                    .expect("target primaries have a defined LMS matrix");
                let m1_inv = oklab::lms_to_rgb_matrix(target_primaries)
                    .expect("target primaries have a defined inverse LMS matrix");
                let lut = Arc::new(GamutBoundaryLut::new(&m1_inv));
                (Some(m_gamut), Some(lut), Some(m1), Some(m1_inv))
            };

        Self {
            source_peak_nits,
            target_peak_nits,
            gamut_knee,
            bt2446a,
            source_to_bt2020,
            bt2020_to_target,
            target_primaries,
            gamut_lut,
            target_m1,
            target_m1_inv,
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

    /// Gamut soft-compression knee (`0.0`–`1.0`). Ignored when
    /// `target_primaries() == ColorPrimaries::Bt2020`.
    #[inline]
    #[must_use]
    pub fn gamut_knee(&self) -> f32 {
        self.gamut_knee
    }

    /// Configured target primaries — `BT.709` by default, `BT.2020` in
    /// the wide-gamut output mode.
    #[inline]
    #[must_use]
    pub fn target_primaries(&self) -> ColorPrimaries {
        self.target_primaries
    }

    /// Borrow a [`SoftCompress`] view of the configured gamut compressor
    /// (constructs the wrapper on demand). Useful for callers that want
    /// to reuse the same LUT for downstream work without rebuilding.
    ///
    /// Returns `None` when the target is BT.2020 (no soft-compress stage
    /// in the wide-gamut output mode).
    #[must_use]
    pub fn soft_compress(&self) -> Option<SoftCompress> {
        match (&self.target_m1, &self.target_m1_inv) {
            (Some(m1), Some(m1_inv)) => {
                Some(SoftCompress::from_matrices(m1, m1_inv, self.gamut_knee))
            }
            _ => None,
        }
    }

    /// Apply the full HDR → SDR pipeline to a strip of linear `RGB f32`
    /// pixels in place.
    ///
    /// Input: linear-light source RGB (per the `source_primaries` passed
    /// to the constructor), source-normalized (`1.0 = source_peak_nits`).
    /// Output: linear-light RGB in the configured target primaries,
    /// target-normalized (`1.0 = target_peak_nits`), guaranteed finite
    /// and in `[0, 1]` per channel — non-finite inputs (NaN / ±Inf) are
    /// scrubbed to `0` before the pipeline runs, and the final clamp
    /// absorbs the f32 EOTF / OKLab roundtrip noise that can push a
    /// saturated pixel ~1e-4 above 1.0.
    pub fn apply_strip(&self, rgb: &mut [[f32; 3]]) {
        // Step 0 — scrub non-finite inputs so the f32 transcendental chain
        // doesn't propagate NaN / ±Inf through the pipeline. Negatives are
        // also clamped to 0; the per-stage gamut matrices can introduce
        // small negatives on out-of-hull primaries downstream, but
        // genuinely-negative input pixels are not a meaningful input.
        for px in rgb.iter_mut() {
            for c in px.iter_mut() {
                if !c.is_finite() || *c < 0.0 {
                    *c = 0.0;
                }
            }
        }
        // Step 1 — source primaries → BT.2020 (no-op when source IS BT.2020).
        if let Some(m) = &self.source_to_bt2020 {
            for px in rgb.iter_mut() {
                apply_matrix_f32(px, m);
            }
        }
        // Step 2 — Bt2446A tone-map (BT.2020 HDR → BT.2020 SDR, SIMD).
        self.bt2446a.map_strip_simd(rgb);
        // Step 3 — BT.2020 SDR → target primaries (gamut convert).
        // No-op when target IS BT.2020 — wide-gamut output mode.
        if let Some(m) = &self.bt2020_to_target {
            for px in rgb.iter_mut() {
                apply_matrix_f32(px, m);
            }
        }
        // Step 4 — soft chroma compression in target-gamut OKLch.
        // No-op when target IS BT.2020 — output stays at BT.2446-A's
        // result with no chroma narrowing.
        if let (Some(lut), Some(m1), Some(m1_inv)) =
            (&self.gamut_lut, &self.target_m1, &self.target_m1_inv)
        {
            soft_compress_strip_with_lut(rgb, lut, m1, m1_inv, self.gamut_knee);
        }
        // Step 5 — hard clamp to `[0, 1]` to absorb f32 epsilon-level
        // overshoot at the saturated end (BT.2446-A's `f * (b_p - y_p)`
        // chain can push a near-peak pixel to ~1.0004 even though the
        // spec algorithm clamps internally; OKLab round-trip in the
        // compress stage adds a few more ULPs). Hard clamp here keeps
        // the `apply_strip` postcondition crisp. In wide-gamut output
        // mode (target == BT.2020) this also catches Bt2446A's
        // near-peak overshoot.
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
        assert_eq!(c.target_primaries(), ColorPrimaries::Bt709);
        // BT.2020 source → no source-to-BT.2020 matrix cached.
        assert!(c.source_to_bt2020.is_none());
        // BT.709 target → soft-compress stage active.
        assert!(c.bt2020_to_target.is_some());
        assert!(c.gamut_lut.is_some());
        assert!(c.target_m1.is_some());
        assert!(c.target_m1_inv.is_some());
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
        let c = HdrToSdr::with_params(
            2000.0,
            ColorPrimaries::Bt2020,
            ColorPrimaries::Bt709,
            100.0,
            0.9,
        );
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
        let c = HdrToSdr::with_params(
            10_000.0,
            ColorPrimaries::Bt2020,
            ColorPrimaries::Bt709,
            100.0,
            0.9,
        );
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
        let c = HdrToSdr::with_params(
            100.0,
            ColorPrimaries::Bt2020,
            ColorPrimaries::Bt709,
            100.0,
            0.9,
        );
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
        let custom = HdrToSdr::with_params(
            1000.0,
            ColorPrimaries::Bt2020,
            ColorPrimaries::Bt709,
            100.0,
            0.30,
        )
        .apply_rgb(rgb);
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

    // ----------------------------------------------------------------------
    // Color-space verification tests (added as part of the pipeline-order
    // fix). The "old order" pipeline applied the BT.2020 → BT.709 matrix
    // BEFORE BT.2446-A, even though the curve's luma weights and YCbCr↔RGB
    // matrices are hardcoded for BT.2020. These tests pin the math against
    // that regression.
    // ----------------------------------------------------------------------

    /// Build the OLD (buggy) pipeline manually so we can compare against
    /// the NEW order without re-introducing a public toggle.
    fn apply_old_buggy_order(c: &HdrToSdr, rgb: [f32; 3]) -> [f32; 3] {
        let mut px = [rgb];
        // Scrub.
        for p in px.iter_mut() {
            for v in p.iter_mut() {
                if !v.is_finite() || *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
        // OLD: matrix first.
        let m = c
            .bt2020_to_target
            .as_ref()
            .expect("BT.709 target should cache a BT.2020 -> target matrix");
        for p in px.iter_mut() {
            apply_matrix_f32(p, m);
        }
        // OLD: then curve in (wrong-primaries) BT.709.
        c.bt2446a.map_strip_simd(&mut px);
        // OLD: soft compress in BT.709 OKLch.
        let lut = c.gamut_lut.as_ref().expect("BT.709 target caches a LUT");
        let m1 = c.target_m1.as_ref().expect("BT.709 target caches m1");
        let m1_inv = c
            .target_m1_inv
            .as_ref()
            .expect("BT.709 target caches m1_inv");
        soft_compress_strip_with_lut(&mut px, lut, m1, m1_inv, c.gamut_knee);
        for p in px.iter_mut() {
            for v in p.iter_mut() {
                if !v.is_finite() {
                    *v = 0.0;
                } else {
                    *v = v.clamp(0.0, 1.0);
                }
            }
        }
        px[0]
    }

    #[test]
    fn neutral_grey_stays_neutral() {
        // For any neutral source-norm grey value, the output must be grey:
        // a wrong-primaries curve would shift hue on the achromatic axis
        // because BT.2446-A's Cb/Cr math is non-zero only when R≠G≠B
        // *after the matrix* — but neutral grey stays neutral under both
        // BT.2020 and BT.709 RGB by construction, so any divergence here
        // is a pipeline-order bug.
        let c = HdrToSdr::new(1000.0);
        for &g in &[0.0_f32, 0.1, 0.3, 0.5, 1.0, 2.0] {
            let out = c.apply_rgb([g, g, g]);
            for v in out {
                assert!(v.is_finite(), "grey {g} produced non-finite {out:?}");
            }
            // Allow ~1e-3 for the BT.2446-A Y'/Cb'/Cr' f32 round-trip and
            // the BT.2020→BT.709 + OKLab roundtrip noise. The OLD buggy
            // order produced shifts on the order of 0.05+ (well over this
            // tolerance) on saturated content — neutral grey was always
            // numerically close even in the wrong pipeline, but combined
            // with the saturated-primary tests above this still pins
            // the achromatic axis stays achromatic.
            assert!(
                (out[0] - out[1]).abs() < 1e-3,
                "grey {g} hue-shifted between R/G: {out:?}"
            );
            assert!(
                (out[1] - out[2]).abs() < 1e-3,
                "grey {g} hue-shifted between G/B: {out:?}"
            );
        }
    }

    #[test]
    fn hdr_diffuse_white_maps_to_sdr_diffuse_white() {
        // HDR pixel at full source peak (1000 nits) should land in a
        // sensible SDR range. The BT.2446-A curve at source=1000, target=100
        // maps the 1.0 source-norm input near 1.0 target-norm.
        let c = HdrToSdr::new(1000.0);
        let out = c.apply_rgb([1.0, 1.0, 1.0]);
        // Channels equal — diffuse white stays neutral (~1e-3 floor for
        // the BT.2446-A f32 Y'Cb'Cr' chain + BT.2020→BT.709 + OKLab).
        assert!(
            (out[0] - out[1]).abs() < 1e-3 && (out[1] - out[2]).abs() < 1e-3,
            "diffuse-white channels diverged: {out:?}"
        );
        // BT.2446-A specifies SDR peak when HDR input hits source peak;
        // the output should land near the top of the SDR range.
        assert!(
            (0.85..=1.0).contains(&out[0]),
            "peak HDR should land near SDR peak, got {}",
            out[0]
        );
    }

    #[test]
    fn saturated_bt2020_red_lands_red_dominant_in_bt709() {
        let c = HdrToSdr::new(1000.0);
        let out = c.apply_rgb([1.0, 0.0, 0.0]);
        finite_in_unit(out);
        assert!(
            out[0] > out[1] + 0.05,
            "red dominance lost vs green: {out:?}"
        );
        assert!(
            out[0] > out[2] + 0.05,
            "red dominance lost vs blue: {out:?}"
        );
    }

    #[test]
    fn saturated_bt2020_green_lands_green_dominant_in_bt709() {
        let c = HdrToSdr::new(1000.0);
        let out = c.apply_rgb([0.0, 1.0, 0.0]);
        finite_in_unit(out);
        // BT.2020 green is just inside BT.709 green, so the projection
        // stays clearly green-dominant. The output also crosses into
        // visible R/B because the BT.709 hull bends inward, but green
        // must remain the leader.
        assert!(
            out[1] > out[0] + 0.05,
            "green dominance lost vs red: {out:?}"
        );
        assert!(
            out[1] > out[2] + 0.05,
            "green dominance lost vs blue: {out:?}"
        );
    }

    #[test]
    fn saturated_bt2020_blue_lands_blue_dominant_in_bt709() {
        let c = HdrToSdr::new(1000.0);
        let out = c.apply_rgb([0.0, 0.0, 1.0]);
        finite_in_unit(out);
        assert!(
            out[2] > out[0] + 0.05,
            "blue dominance lost vs red: {out:?}"
        );
        assert!(
            out[2] > out[1] + 0.05,
            "blue dominance lost vs green: {out:?}"
        );
    }

    #[test]
    fn p3_source_pipeline_produces_neutral_grey_from_p3_grey() {
        // Display P3 source → grey in → grey out (no hue shift across
        // the source→BT.2020 conversion + BT.2446-A + BT.2020→BT.709 +
        // soft compress).
        let c = HdrToSdr::with_source_primaries(1000.0, ColorPrimaries::DisplayP3);
        let out = c.apply_rgb([0.5, 0.5, 0.5]);
        for v in out {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v), "{out:?}");
        }
        assert!(
            (out[0] - out[1]).abs() < 1e-3 && (out[1] - out[2]).abs() < 1e-3,
            "P3 grey hue-shifted: {out:?}"
        );
    }

    #[test]
    fn bt709_source_is_identity_matrix_to_bt2020() {
        // When the source is BT.709, the cached source_to_bt2020 matrix
        // should be Some(BT.709 -> BT.2020). Apply it to BT.709 red
        // (1, 0, 0) and verify the BT.2020 RGB matches the standard
        // BT.709→BT.2020 transform within 1e-3:
        //   (0.6274, 0.0691, 0.0164)  (computed from standards-grade
        //   BT.709_TO_XYZ × XYZ_TO_BT2020 matrices in src/gamut.rs).
        let c = HdrToSdr::with_source_primaries(1000.0, ColorPrimaries::Bt709);
        let m = c
            .source_to_bt2020
            .as_ref()
            .expect("BT.709 source should cache a source-to-BT.2020 matrix");
        let mut rgb = [1.0_f32, 0.0, 0.0];
        apply_matrix_f32(&mut rgb, m);
        let expected = [0.6274_f32, 0.0691, 0.0164];
        for k in 0..3 {
            assert!(
                (rgb[k] - expected[k]).abs() < 1e-3,
                "BT.709(1,0,0) → BT.2020 ch{k}: got {} expected {}",
                rgb[k],
                expected[k]
            );
        }
    }

    #[test]
    fn pipeline_order_regression_test() {
        // Pin that the new (correct) order produces measurably different
        // output from the old (buggy) order for a saturated content pixel.
        // If the orders ever match within 0.02, either the code regressed
        // or the test's expected difference is too tight.
        let c = HdrToSdr::new(1000.0);
        let input = [1.0_f32, 0.0, 0.0];
        let new_out = c.apply_rgb(input);
        let old_out = apply_old_buggy_order(&c, input);
        let mut max_diff = 0.0_f32;
        for k in 0..3 {
            let d = (new_out[k] - old_out[k]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff >= 0.02,
            "new and old pipeline orders should produce measurably different output on saturated red; got max_diff={max_diff:.4} (new={new_out:?} old={old_out:?})"
        );
    }

    // ----------------------------------------------------------------------
    // Wide-gamut output mode (target == BT.2020).
    // ----------------------------------------------------------------------

    #[test]
    fn bt2020_target_skips_gamut_steps() {
        // When the target is BT.2020, the constructor must take the
        // wide-gamut "no gamut narrowing" branch: source-to-BT.2020 is
        // also None because source==BT.2020, AND none of the post-curve
        // (gamut + soft-compress) state is allocated.
        let c = HdrToSdr::with_io(1000.0, ColorPrimaries::Bt2020, ColorPrimaries::Bt2020);
        assert_eq!(c.target_primaries(), ColorPrimaries::Bt2020);
        assert!(c.source_to_bt2020.is_none(), "source IS BT.2020");
        assert!(c.bt2020_to_target.is_none(), "wide-gamut → no gamut matrix");
        assert!(c.gamut_lut.is_none(), "wide-gamut → no soft-compress LUT");
        assert!(c.target_m1.is_none(), "wide-gamut → no target LMS matrix");
        assert!(c.target_m1_inv.is_none(), "wide-gamut → no inverse LMS");
        // soft_compress() returns None in this mode.
        assert!(c.soft_compress().is_none());
    }

    #[test]
    fn bt2020_target_output_equals_bt2446a_alone() {
        // Pin: in wide-gamut output mode, the only lossy stage is
        // BT.2446-A. The full pipeline output must be bit-identical to
        // running just `Bt2446A::map_strip_simd` on the scrubbed input.
        // (Steps 0 and 5 are the same scrub+clamp; steps 1, 3, 4 are no-ops.)
        let pipe_a = HdrToSdr::with_io(1000.0, ColorPrimaries::Bt2020, ColorPrimaries::Bt2020);
        // Deterministic xorshift32 — same seed as the SIMD parity test.
        struct Xorshift(u32);
        impl Xorshift {
            fn next_f32(&mut self) -> f32 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                self.0 = x;
                (x as f32 / u32::MAX as f32) * 2.0
            }
        }
        let mut rng = Xorshift(0xCAFEBABE);
        let n_pixels = 1000;
        let mut a: alloc::vec::Vec<[f32; 3]> = alloc::vec::Vec::with_capacity(n_pixels);
        for _ in 0..n_pixels {
            a.push([rng.next_f32(), rng.next_f32(), rng.next_f32()]);
        }
        let mut b = a.clone();

        // pipe_a: full HdrToSdr pipeline (scrub + Bt2446A + final clamp,
        // with steps 1/3/4 all no-ops).
        pipe_a.apply_strip(&mut a);

        // pipe_b: manually scrub (matches step 0), then Bt2446A directly,
        // then final clamp (matches step 5).
        for px in b.iter_mut() {
            for c in px.iter_mut() {
                if !c.is_finite() || *c < 0.0 {
                    *c = 0.0;
                }
            }
        }
        let bt2446a = Bt2446A::new(1000.0, 100.0);
        bt2446a.map_strip_simd(&mut b);
        for px in b.iter_mut() {
            for c in px.iter_mut() {
                if !c.is_finite() {
                    *c = 0.0;
                } else {
                    *c = c.clamp(0.0, 1.0);
                }
            }
        }

        // Bit-identical.
        for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
            for k in 0..3 {
                assert!(
                    pa[k].to_bits() == pb[k].to_bits(),
                    "px {i} ch{k} diverges: pipeline={} bt2446a_alone={}",
                    pa[k],
                    pb[k]
                );
            }
        }
    }

    #[test]
    fn bt2020_target_preserves_saturated_red() {
        // Saturated BT.2020 red into a BT.2020-target pipeline must
        // stay saturated: no gamut matrix runs, no soft-compress runs,
        // so the only change is BT.2446-A's tone curve scaling the
        // luminance down. Red dominates strongly, G/B near 0.
        let c = HdrToSdr::with_io(1000.0, ColorPrimaries::Bt2020, ColorPrimaries::Bt2020);
        let out = c.apply_rgb([1.0, 0.0, 0.0]);
        finite_in_unit(out);
        assert!(
            out[0] >= 0.5,
            "red should remain near peak, got r={}",
            out[0]
        );
        assert!(
            out[1] < 0.01,
            "green should stay near 0 (no gamut narrowing), got g={}",
            out[1]
        );
        assert!(
            out[2] < 0.01,
            "blue should stay near 0 (no gamut narrowing), got b={}",
            out[2]
        );
    }

    #[test]
    fn displayp3_target_runs_full_pipeline() {
        // P3 is narrower than BT.2020 on some hues, so the gamut matrix
        // + soft-compress stages MUST be active.
        let c = HdrToSdr::with_io(1000.0, ColorPrimaries::Bt2020, ColorPrimaries::DisplayP3);
        assert_eq!(c.target_primaries(), ColorPrimaries::DisplayP3);
        assert!(c.source_to_bt2020.is_none(), "source IS BT.2020");
        assert!(
            c.bt2020_to_target.is_some(),
            "P3 target → gamut matrix cached"
        );
        assert!(
            c.gamut_lut.is_some(),
            "P3 target → soft-compress LUT cached"
        );
        assert!(c.target_m1.is_some(), "P3 target → LMS matrix cached");
        assert!(c.target_m1_inv.is_some(), "P3 target → inverse LMS cached");
        // Saturated BT.2020 red into P3: still red-dominant, all channels
        // in [0, 1] because soft-compress kicks in.
        let out = c.apply_rgb([1.0, 0.0, 0.0]);
        finite_in_unit(out);
        assert!(
            out[0] > out[1] && out[0] > out[2],
            "red should remain dominant in P3 target: {out:?}"
        );
    }

    #[test]
    fn bt709_target_unchanged_from_old_behavior() {
        // Backward compat: HdrToSdr::new(p) === HdrToSdr::with_io(p,
        // BT.2020, BT.709). Bit-identical on a randomized strip.
        let pipe_default = HdrToSdr::new(1000.0);
        let pipe_explicit =
            HdrToSdr::with_io(1000.0, ColorPrimaries::Bt2020, ColorPrimaries::Bt709);
        // Same field state.
        assert_eq!(
            pipe_default.target_primaries(),
            pipe_explicit.target_primaries()
        );
        // Same outputs.
        struct Xorshift(u32);
        impl Xorshift {
            fn next_f32(&mut self) -> f32 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                self.0 = x;
                (x as f32 / u32::MAX as f32) * 2.0
            }
        }
        let mut rng = Xorshift(0xDECAF999);
        let n_pixels = 1000;
        let mut a: alloc::vec::Vec<[f32; 3]> = alloc::vec::Vec::with_capacity(n_pixels);
        for _ in 0..n_pixels {
            a.push([rng.next_f32(), rng.next_f32(), rng.next_f32()]);
        }
        let mut b = a.clone();
        pipe_default.apply_strip(&mut a);
        pipe_explicit.apply_strip(&mut b);
        for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
            for k in 0..3 {
                assert!(
                    pa[k].to_bits() == pb[k].to_bits(),
                    "px {i} ch{k} diverges: new()={} with_io={}",
                    pa[k],
                    pb[k]
                );
            }
        }
    }

    #[test]
    fn pipeline_finite_for_each_target_primaries() {
        // Extreme-input pipeline finiteness must hold across all
        // supported target primaries: BT.2020 (wide-gamut), BT.709
        // (default), DisplayP3.
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
        for &target in &[
            ColorPrimaries::Bt2020,
            ColorPrimaries::Bt709,
            ColorPrimaries::DisplayP3,
        ] {
            let c = HdrToSdr::with_io(10_000.0, ColorPrimaries::Bt2020, target);
            let mut strip = pixels.to_vec();
            c.apply_strip(&mut strip);
            for (i, px) in strip.iter().enumerate() {
                for (k, &v) in px.iter().enumerate() {
                    assert!(
                        v.is_finite() && (0.0..=1.0).contains(&v),
                        "target {target:?} px {i} ch{k}={v} out of [0,1] (input {:?})",
                        pixels[i]
                    );
                }
            }
        }
    }

    #[test]
    fn property_simd_matches_scalar_under_arbitrary_source_primaries() {
        // For each supported source-primary tier (BT.2020, BT.709, P3),
        // randomized strip processing through apply_strip must agree
        // with per-pixel apply_rgb within 5e-4 — the SIMD parity
        // contract carried from Bt2446A's own property test.
        //
        // Deterministic xorshift32 PRNG so the test is reproducible
        // across architectures without pulling a rand dep.
        struct Xorshift(u32);
        impl Xorshift {
            fn next_f32(&mut self) -> f32 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                self.0 = x;
                (x as f32 / u32::MAX as f32) * 2.0
            }
        }

        for &prim in &[
            ColorPrimaries::Bt2020,
            ColorPrimaries::Bt709,
            ColorPrimaries::DisplayP3,
        ] {
            let mut rng = Xorshift(0x1234_5678);
            let n_pixels = 12_000;
            let mut strip = alloc::vec::Vec::with_capacity(n_pixels);
            for _ in 0..n_pixels {
                strip.push([rng.next_f32(), rng.next_f32(), rng.next_f32()]);
            }
            let scalar: alloc::vec::Vec<[f32; 3]> = {
                let c = HdrToSdr::with_source_primaries(1000.0, prim);
                strip.iter().map(|p| c.apply_rgb(*p)).collect()
            };

            let c = HdrToSdr::with_source_primaries(1000.0, prim);
            c.apply_strip(&mut strip);

            for (i, (&sc, &sp)) in scalar.iter().zip(strip.iter()).enumerate() {
                for k in 0..3 {
                    let diff = (sc[k] - sp[k]).abs();
                    assert!(
                        diff < 5e-4,
                        "scalar/SIMD diverge under {prim:?} at px {i} ch{k}: scalar={} simd={} diff={}",
                        sc[k],
                        sp[k],
                        diff
                    );
                }
            }
        }
    }

    #[test]
    fn property_simd_matches_scalar_under_arbitrary_target_primaries() {
        // Extension of the SIMD-parity property test: SIMD/scalar
        // agreement must hold across every (source, target) primaries
        // pair. Catches divergence introduced by the per-target
        // conditional gamut/compress branches.
        struct Xorshift(u32);
        impl Xorshift {
            fn next_f32(&mut self) -> f32 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                self.0 = x;
                (x as f32 / u32::MAX as f32) * 2.0
            }
        }

        let primaries = [
            ColorPrimaries::Bt2020,
            ColorPrimaries::Bt709,
            ColorPrimaries::DisplayP3,
        ];

        for &source in &primaries {
            for &target in &primaries {
                let mut rng = Xorshift(0x1234_5678 ^ ((source as u32) << 8) ^ (target as u32));
                let n_pixels = 12_000;
                let mut strip = alloc::vec::Vec::with_capacity(n_pixels);
                for _ in 0..n_pixels {
                    strip.push([rng.next_f32(), rng.next_f32(), rng.next_f32()]);
                }
                let scalar: alloc::vec::Vec<[f32; 3]> = {
                    let c = HdrToSdr::with_io(1000.0, source, target);
                    strip.iter().map(|p| c.apply_rgb(*p)).collect()
                };

                let c = HdrToSdr::with_io(1000.0, source, target);
                c.apply_strip(&mut strip);

                for (i, (&sc, &sp)) in scalar.iter().zip(strip.iter()).enumerate() {
                    for k in 0..3 {
                        let diff = (sc[k] - sp[k]).abs();
                        assert!(
                            diff < 5e-4,
                            "scalar/SIMD diverge under source={source:?} target={target:?} at px {i} ch{k}: scalar={} simd={} diff={}",
                            sc[k],
                            sp[k],
                            diff
                        );
                    }
                }
            }
        }
    }
}
