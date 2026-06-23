//! Soft gamut compression in OKLch — precomputed gamut boundary LUT
//! plus a rational knee function that preserves hue and lightness.
//!
//! For a given color primaries set, [`GamutBoundaryLut`] tabulates the
//! maximum in-gamut OKLch chroma at each `(L, hue)` pair. [`SoftCompress`]
//! wraps a LUT with a configurable knee threshold and smoothly reduces
//! out-of-gamut chroma using a rational compression curve.
//!
//! # Provenance
//!
//! Extracted from `zenfilters::gamut_lut` (the previous home of these
//! types — the implementation is byte-identical to that crate's
//! `GamutBoundaryLut`, plus a thin [`SoftCompress`] wrapper that exposes
//! the same compression behavior under an explicit type. Tests carried
//! over verbatim from `zenfilters/src/gamut_lut.rs`).

extern crate alloc;

use crate::gamut::GamutMatrix;
use crate::oklab;

/// Precomputed sRGB (or P3 / BT.2020) gamut boundary in OKLch space.
///
/// Stores the maximum in-gamut chroma for a grid of `(L, hue)` values.
/// Constructed once per primaries set and reused across frames.
///
/// The grid is `64` lightness steps × `256` hue steps (16 384 floats =
/// 64 KiB). Construction takes ~30 ms on a single core for sRGB — amortize
/// by sharing one LUT across the lifetime of a converter.
#[derive(Debug, Clone)]
pub struct GamutBoundaryLut {
    /// Flattened `[L_STEPS][H_STEPS]` array of max chroma values.
    data: alloc::vec::Vec<f32>,
}

/// Number of lightness steps in the LUT (0..=1).
const L_STEPS: usize = 64;
/// Number of hue angle steps in the LUT (0..2π).
const H_STEPS: usize = 256;
/// Maximum chroma to search during LUT construction.
/// OKLch chroma rarely exceeds 0.4 for sRGB, but P3/BT.2020 can go higher.
const MAX_SEARCH_CHROMA: f32 = 0.5;
/// Binary search iterations for gamut boundary (2^-20 ≈ 1e-6 precision).
const BISECT_ITERS: u32 = 20;

impl GamutBoundaryLut {
    /// Build the gamut boundary LUT for a given primaries set.
    ///
    /// `m1_inv` is the combined LMS → linear RGB matrix for the target
    /// primaries, from [`crate::oklab::lms_to_rgb_matrix`].
    pub fn new(m1_inv: &GamutMatrix) -> Self {
        let mut data = alloc::vec![0.0f32; L_STEPS * H_STEPS];

        for li in 0..L_STEPS {
            let l = li as f32 / (L_STEPS - 1) as f32;
            for hi in 0..H_STEPS {
                let h = hi as f32 / H_STEPS as f32 * core::f32::consts::TAU;
                data[li * H_STEPS + hi] = find_max_chroma(l, h, m1_inv);
            }
        }

        Self { data }
    }

    /// Look up the maximum in-gamut chroma for a given `(L, hue)` with
    /// bilinear interpolation. `h` is in radians; out-of-range `L` is
    /// clamped to `[0, 1]` and `h` is wrapped modulo `2π`.
    #[inline]
    pub fn max_chroma(&self, l: f32, h: f32) -> f32 {
        let l_clamped = l.clamp(0.0, 1.0);
        let h_norm = h.rem_euclid(core::f32::consts::TAU);

        let l_f = l_clamped * (L_STEPS - 1) as f32;
        let h_f = h_norm / core::f32::consts::TAU * H_STEPS as f32;

        let l0 = (l_f as usize).min(L_STEPS - 2);
        let l1 = l0 + 1;
        let h0 = h_f as usize % H_STEPS;
        let h1 = (h0 + 1) % H_STEPS;

        let lt = l_f - l0 as f32;
        let ht = h_f - h0 as f32;

        let v00 = self.data[l0 * H_STEPS + h0];
        let v01 = self.data[l0 * H_STEPS + h1];
        let v10 = self.data[l1 * H_STEPS + h0];
        let v11 = self.data[l1 * H_STEPS + h1];

        let top = v00 + (v01 - v00) * ht;
        let bot = v10 + (v11 - v10) * ht;
        top + (bot - top) * lt
    }

    /// Apply soft chroma compression to OKLab planes in-place.
    ///
    /// For each pixel, if chroma exceeds `knee * max_chroma`, smoothly
    /// compresses it toward the gamut boundary using a rational function
    /// that preserves hue and lightness.
    ///
    /// `knee` is the fraction of max chroma where compression starts
    /// (`0.0`–`1.0`). Production default: `0.96` (start compressing at
    /// 96 % of gamut boundary), empirically calibrated against the
    /// imazen-26 gain-mapped HDR corpus on 2026-06-23 — the largest knee
    /// where the corpus-p90 fraction of pre-clamp out-of-gamut pixels
    /// stays under 0.1 %. Smaller values bring the rolloff in earlier
    /// (more desaturation, lower clipping); larger values let more
    /// clipping leak through.
    pub fn compress_planes(&self, l: &[f32], a: &mut [f32], b: &mut [f32], knee: f32) {
        let n = l.len();
        debug_assert!(a.len() == n && b.len() == n);

        for i in 0..n {
            let av = a[i];
            let bv = b[i];

            let c = (av * av + bv * bv).sqrt();
            if c < 1e-10 {
                continue; // achromatic, nothing to compress
            }

            let h = bv.atan2(av);

            let max_c = self.max_chroma(l[i], h);
            if max_c < 1e-10 {
                // At L=0 or L=1, max chroma is 0 — force achromatic
                a[i] = 0.0;
                b[i] = 0.0;
                continue;
            }

            let knee_c = knee * max_c;
            if c <= knee_c {
                continue; // within knee threshold, pass through
            }

            // Rational compression: maps [knee_c, ∞) → [knee_c, max_c)
            //
            // f(C) = knee_c + range * excess / (excess + range)
            //
            // Properties:
            //   f(knee_c) = knee_c          (C0 continuous)
            //   f'(knee_c) = 1              (C1 continuous — slope matches passthrough)
            //   f(∞) → max_c               (asymptotic limit)
            let range = max_c - knee_c;
            let excess = c - knee_c;
            let compressed_c = knee_c + range * excess / (excess + range);

            let scale = compressed_c / c;
            a[i] = av * scale;
            b[i] = bv * scale;
        }
    }
}

/// Binary search for the maximum in-gamut chroma at a given `(L, hue)`.
fn find_max_chroma(l: f32, h: f32, m1_inv: &GamutMatrix) -> f32 {
    let cos_h = libm::cosf(h);
    let sin_h = libm::sinf(h);

    let mut lo = 0.0f32;
    let mut hi = MAX_SEARCH_CHROMA;

    for _ in 0..BISECT_ITERS {
        let mid = (lo + hi) * 0.5;
        let a = mid * cos_h;
        let b = mid * sin_h;

        if is_in_gamut(l, a, b, m1_inv) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    lo
}

/// Check if an OKLab color is within the RGB gamut for the given primaries.
#[inline]
fn is_in_gamut(l: f32, a: f32, b: f32, m1_inv: &GamutMatrix) -> bool {
    let rgb = oklab::oklab_to_rgb(l, a, b, m1_inv);
    rgb[0] >= 0.0
        && rgb[0] <= 1.0
        && rgb[1] >= 0.0
        && rgb[1] <= 1.0
        && rgb[2] >= 0.0
        && rgb[2] <= 1.0
}

/// Soft chroma compression on linear-light RGB strips.
///
/// Wraps a [`GamutBoundaryLut`] with a `knee` threshold and exposes the
/// compression as an explicit per-strip API. Construction performs the
/// LUT build once; subsequent [`apply_strip`](Self::apply_strip) calls
/// reuse it.
///
/// # Pipeline
///
/// For each pixel:
/// 1. Convert linear RGB → OKLab (via [`crate::oklab::rgb_to_oklab`]).
/// 2. Compute chroma `c = √(a² + b²)` and hue `h = atan2(b, a)`.
/// 3. Look up max in-gamut chroma `c_max` at `(L, h)`.
/// 4. If `c > knee · c_max`, compress: `c' = knee·c_max + range · excess / (excess + range)`.
/// 5. Convert OKLab → linear RGB.
///
/// Hue and lightness are preserved within float precision; only chroma is
/// modified. The rational compression curve is C¹-continuous at the knee
/// (slope `1.0` on the inside, asymptote at the gamut boundary).
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "hdr-experimental")]
/// # {
/// use zenpixels_convert::hdr::SoftCompress;
/// use zenpixels_convert::oklab;
/// use zenpixels::ColorPrimaries;
///
/// let m1_inv = oklab::lms_to_rgb_matrix(ColorPrimaries::Bt709).unwrap();
/// let compress = SoftCompress::new(&m1_inv, 0.96);
///
/// let mut pixels = vec![[1.2_f32, 0.05, 0.05]]; // out-of-gamut red
/// compress.apply_strip(&mut pixels);
/// for px in &pixels {
///     for &c in px {
///         assert!(c <= 1.0 + 1e-2, "expected in-gamut output");
///     }
/// }
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct SoftCompress {
    lut: GamutBoundaryLut,
    knee: f32,
    m1: GamutMatrix,
    m1_inv: GamutMatrix,
}

impl SoftCompress {
    /// Construct a [`SoftCompress`] for the given primaries (via `m1_inv`,
    /// the LMS → RGB matrix from
    /// [`crate::oklab::lms_to_rgb_matrix`]) and
    /// `knee` threshold (`0.0`–`1.0`; production default `0.96`,
    /// corpus-validated 2026-06-23).
    ///
    /// The matching forward matrix is derived by inverting `m1_inv`. If you
    /// already have the forward matrix on hand (the `rgb_to_lms_matrix`
    /// output), prefer [`SoftCompress::from_matrices`].
    #[must_use]
    pub fn new(m1_inv: &GamutMatrix, knee: f32) -> Self {
        let m1 = invert_3x3(m1_inv).expect("LMS→RGB matrix must be invertible");
        Self {
            lut: GamutBoundaryLut::new(m1_inv),
            knee,
            m1,
            m1_inv: *m1_inv,
        }
    }

    /// Construct a [`SoftCompress`] from both forward and inverse matrices.
    /// `m1` is the linear-RGB → LMS matrix (from `oklab::rgb_to_lms_matrix`);
    /// `m1_inv` is the LMS → linear-RGB matrix.
    #[must_use]
    pub fn from_matrices(m1: &GamutMatrix, m1_inv: &GamutMatrix, knee: f32) -> Self {
        Self {
            lut: GamutBoundaryLut::new(m1_inv),
            knee,
            m1: *m1,
            m1_inv: *m1_inv,
        }
    }

    /// Apply soft gamut compression to a strip of linear RGB pixels in place.
    pub fn apply_strip(&self, rgb: &mut [[f32; 3]]) {
        // Convert to OKLab in planar form for the LUT — small temporaries
        // per pixel keep the API allocation-free even for short strips.
        for px in rgb.iter_mut() {
            let lab = oklab::rgb_to_oklab(px[0], px[1], px[2], &self.m1);
            let l = lab[0];
            let mut a = lab[1];
            let mut b = lab[2];

            let c = (a * a + b * b).sqrt();
            if c < 1e-10 {
                continue;
            }
            let h = b.atan2(a);
            let max_c = self.lut.max_chroma(l, h);
            if max_c < 1e-10 {
                a = 0.0;
                b = 0.0;
            } else {
                let knee_c = self.knee * max_c;
                if c > knee_c {
                    let range = max_c - knee_c;
                    let excess = c - knee_c;
                    let compressed_c = knee_c + range * excess / (excess + range);
                    let scale = compressed_c / c;
                    a *= scale;
                    b *= scale;
                }
            }
            let out = oklab::oklab_to_rgb(l, a, b, &self.m1_inv);
            *px = out;
        }
    }

    /// Borrow the inner [`GamutBoundaryLut`] for direct planar use (used by
    /// `zenfilters::Pipeline`, which keeps its own planar OKLab buffer).
    #[inline]
    #[must_use]
    pub fn lut(&self) -> &GamutBoundaryLut {
        &self.lut
    }

    /// Knee threshold (fraction of max chroma where compression starts).
    #[inline]
    #[must_use]
    pub fn knee(&self) -> f32 {
        self.knee
    }
}

/// Invert a 3×3 matrix via cofactor expansion. Returns `None` if singular.
fn invert_3x3(m: &GamutMatrix) -> Option<GamutMatrix> {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let i = m[2][2];

    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-30 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (e * i - f * h) * inv_det,
            -(b * i - c * h) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            -(d * i - f * g) * inv_det,
            (a * i - c * g) * inv_det,
            -(a * f - c * d) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            -(a * h - b * g) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oklab;
    use zenpixels::ColorPrimaries;

    fn bt709_lut() -> GamutBoundaryLut {
        let m1_inv = oklab::lms_to_rgb_matrix(ColorPrimaries::Bt709).unwrap();
        GamutBoundaryLut::new(&m1_inv)
    }

    fn bt709_soft_compress(knee: f32) -> SoftCompress {
        let m1 = oklab::rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();
        let m1_inv = oklab::lms_to_rgb_matrix(ColorPrimaries::Bt709).unwrap();
        SoftCompress::from_matrices(&m1, &m1_inv, knee)
    }

    // ---- Tests carried from zenfilters/src/gamut_lut.rs ----

    #[test]
    fn lut_boundary_at_extremes() {
        let lut = bt709_lut();

        // At L=0 (black) and L=1 (white), max chroma should be ~0
        for hi in 0..H_STEPS {
            let h = hi as f32 / H_STEPS as f32 * core::f32::consts::TAU;
            assert!(
                lut.max_chroma(0.0, h) < 0.01,
                "L=0 max chroma should be ~0, got {}",
                lut.max_chroma(0.0, h)
            );
            assert!(
                lut.max_chroma(1.0, h) < 0.01,
                "L=1 max chroma should be ~0, got {}",
                lut.max_chroma(1.0, h)
            );
        }
    }

    #[test]
    fn lut_boundary_has_positive_chroma_at_mid_l() {
        let lut = bt709_lut();

        let mut max_found = 0.0f32;
        for hi in 0..H_STEPS {
            let h = hi as f32 / H_STEPS as f32 * core::f32::consts::TAU;
            let mc = lut.max_chroma(0.5, h);
            max_found = max_found.max(mc);
        }
        assert!(
            max_found > 0.1,
            "mid-L should have substantial gamut, max chroma = {max_found}"
        );
    }

    #[test]
    fn lut_boundary_is_monotonic_toward_extremes() {
        let lut = bt709_lut();

        // For a fixed hue, chroma boundary should increase from L=0 to
        // some peak, then decrease to L=1 (spindle shape).
        let h = 0.5;
        let mut found_peak = false;
        let mut prev = 0.0f32;
        for li in 0..L_STEPS {
            let l = li as f32 / (L_STEPS - 1) as f32;
            let mc = lut.max_chroma(l, h);
            if mc < prev {
                found_peak = true;
            }
            if found_peak {
                assert!(
                    mc <= prev + 0.01,
                    "chroma should decrease after peak at L={l}"
                );
            }
            prev = mc;
        }
    }

    #[test]
    fn compress_preserves_in_gamut_colors() {
        let lut = bt709_lut();
        let m1_inv = oklab::lms_to_rgb_matrix(ColorPrimaries::Bt709).unwrap();

        let l = alloc::vec![0.5, 0.3, 0.7, 0.9];
        let mut a = alloc::vec![0.01, -0.02, 0.005, 0.0];
        let mut b = alloc::vec![0.01, 0.01, -0.01, 0.0];
        let a_orig = a.clone();
        let b_orig = b.clone();

        for i in 0..l.len() {
            assert!(
                is_in_gamut(l[i], a[i], b[i], &m1_inv),
                "test color {i} should be in gamut"
            );
        }

        lut.compress_planes(&l, &mut a, &mut b, 0.9);

        for i in 0..l.len() {
            assert!(
                (a[i] - a_orig[i]).abs() < 1e-6,
                "in-gamut color {i} a should be unchanged"
            );
            assert!(
                (b[i] - b_orig[i]).abs() < 1e-6,
                "in-gamut color {i} b should be unchanged"
            );
        }
    }

    #[test]
    fn compress_reduces_out_of_gamut_chroma() {
        let lut = bt709_lut();

        let l = alloc::vec![0.5, 0.5, 0.5];
        let mut a = alloc::vec![0.3, -0.3, 0.0];
        let mut b = alloc::vec![0.0, 0.0, 0.3];

        let orig_chroma: alloc::vec::Vec<f32> = a
            .iter()
            .zip(b.iter())
            .map(|(&av, &bv): (&f32, &f32)| (av * av + bv * bv).sqrt())
            .collect();

        lut.compress_planes(&l, &mut a, &mut b, 0.9);

        for i in 0..l.len() {
            let new_chroma = (a[i] * a[i] + b[i] * b[i]).sqrt();
            assert!(
                new_chroma < orig_chroma[i],
                "color {i} chroma should decrease: {:.4} -> {:.4}",
                orig_chroma[i],
                new_chroma
            );
        }
    }

    #[test]
    fn compress_preserves_hue() {
        let lut = bt709_lut();

        let l = alloc::vec![0.5, 0.5];
        let mut a = alloc::vec![0.3, -0.2];
        let mut b = alloc::vec![0.1, 0.25];

        let orig_hue: alloc::vec::Vec<f32> = a
            .iter()
            .zip(b.iter())
            .map(|(&av, &bv): (&f32, &f32)| bv.atan2(av))
            .collect();

        lut.compress_planes(&l, &mut a, &mut b, 0.9);

        for i in 0..l.len() {
            let new_hue = b[i].atan2(a[i]);
            let hue_diff = (new_hue - orig_hue[i]).abs();
            assert!(
                hue_diff < 1e-4,
                "hue should be preserved: {:.6} -> {:.6}",
                orig_hue[i],
                new_hue
            );
        }
    }

    #[test]
    fn compress_output_is_in_gamut() {
        let m1_inv = oklab::lms_to_rgb_matrix(ColorPrimaries::Bt709).unwrap();
        let lut = bt709_lut();

        let l = alloc::vec![0.5, 0.3, 0.8, 0.5, 0.1, 0.95];
        let mut a = alloc::vec![0.4, -0.3, 0.2, -0.4, 0.15, 0.05];
        let mut b = alloc::vec![0.3, 0.4, -0.3, -0.2, 0.2, -0.03];

        lut.compress_planes(&l, &mut a, &mut b, 0.9);

        for i in 0..l.len() {
            let rgb = oklab::oklab_to_rgb(l[i], a[i], b[i], &m1_inv);
            assert!(
                rgb[0] >= -0.01 && rgb[0] <= 1.01,
                "R out of gamut after compress: color {i} R={:.4}",
                rgb[0]
            );
            assert!(
                rgb[1] >= -0.01 && rgb[1] <= 1.01,
                "G out of gamut after compress: color {i} G={:.4}",
                rgb[1]
            );
            assert!(
                rgb[2] >= -0.01 && rgb[2] <= 1.01,
                "B out of gamut after compress: color {i} B={:.4}",
                rgb[2]
            );
        }
    }

    // ---- New tests for SoftCompress wrapper ----

    #[test]
    fn interior_pixels_pass_through_unchanged() {
        let compress = bt709_soft_compress(0.9);
        // Pixels well inside the gamut should be bit-identical (within
        // RGB→OKLab→RGB float roundtrip noise) after compress.
        let mut pixels = alloc::vec![
            [0.5_f32, 0.5, 0.5],
            [0.3, 0.2, 0.1],
            [0.6, 0.3, 0.2],
            [0.2, 0.4, 0.5],
        ];
        let originals = pixels.clone();
        compress.apply_strip(&mut pixels);
        for (i, (p, o)) in pixels.iter().zip(originals.iter()).enumerate() {
            for k in 0..3 {
                let diff = (p[k] - o[k]).abs();
                // 5e-4 tolerates the OKLab roundtrip f32 noise (cbrt + 3×3 matrices).
                assert!(
                    diff < 5e-4,
                    "interior pixel {i} ch{k}: input {} output {} diff {}",
                    o[k],
                    p[k],
                    diff
                );
            }
        }
    }

    #[test]
    fn out_of_gamut_pixels_land_in_gamut() {
        // Out-of-gamut **chroma** at sub-peak luminance — the realistic case
        // post-tone-mapping. Inputs were chosen so OKLab L stays in [0, 1]
        // (the LUT's defined domain); at L > 1, max_chroma collapses to 0
        // and the compressor correctly snaps to achromatic, but the OKLab →
        // RGB roundtrip on the L > 1 column emits RGB > 1 by construction.
        let compress = bt709_soft_compress(0.9);
        let mut pixels = alloc::vec![
            [0.85_f32, -0.05, -0.05], // out-of-gamut red (negative G/B from BT.2020 mapping)
            [-0.05, 0.85, -0.05],     // out-of-gamut green
            [-0.05, -0.05, 0.85],     // out-of-gamut blue
            [0.9, 0.9, -0.05],        // out-of-gamut yellow
        ];
        compress.apply_strip(&mut pixels);
        for (i, px) in pixels.iter().enumerate() {
            for (k, &v) in px.iter().enumerate() {
                assert!(
                    v.is_finite() && (-0.02..=1.02).contains(&v),
                    "out-of-gamut pixel {i} ch{k} = {v} did not land in `[0,1]`"
                );
            }
        }
    }

    #[test]
    fn hue_preservation_under_compression() {
        // Use sub-peak primaries so OKLab L stays well inside [0, 1]; at
        // L → 1 (or L → 0) the LUT's max_chroma collapses to 0 and the
        // compressor *correctly* forces a → b → 0, which by definition
        // throws hue away. The contract is "hue preserved when the LUT has
        // headroom"; we test that, not the degenerate L = boundary case.
        let m1 = oklab::rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();
        let compress = bt709_soft_compress(0.9);
        let inputs = [
            [0.85_f32, 0.05, 0.05], // saturated red
            [0.05, 0.85, 0.05],     // saturated green
            [0.05, 0.05, 0.85],     // saturated blue
            [0.7, 0.7, 0.05],       // saturated yellow
            [0.85, 0.05, 0.85],     // saturated magenta
        ];
        for rgb in inputs {
            let lab_before = oklab::rgb_to_oklab(rgb[0], rgb[1], rgb[2], &m1);
            let h_before = lab_before[2].atan2(lab_before[1]);

            let mut strip = alloc::vec![rgb];
            compress.apply_strip(&mut strip);

            let out = strip[0];
            let lab_after = oklab::rgb_to_oklab(out[0], out[1], out[2], &m1);
            let h_after = lab_after[2].atan2(lab_after[1]);

            let hue_diff = (h_after - h_before).abs();
            assert!(
                hue_diff < 0.001,
                "hue drift for {rgb:?}: before {h_before:.6} after {h_after:.6} diff {hue_diff}"
            );
        }
    }

    #[test]
    fn lightness_preservation_under_compression() {
        // Same input shape as hue test — sub-peak primaries with OKLab L in
        // [0, 1] so the compressor never has to snap chroma to zero.
        let m1 = oklab::rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();
        let compress = bt709_soft_compress(0.9);
        let inputs = [
            [0.85_f32, 0.05, 0.05],
            [0.05, 0.85, 0.05],
            [0.05, 0.05, 0.85],
            [0.7, 0.7, 0.05],
            [0.85, 0.05, 0.85],
        ];
        for rgb in inputs {
            let l_before = oklab::rgb_to_oklab(rgb[0], rgb[1], rgb[2], &m1)[0];
            let mut strip = alloc::vec![rgb];
            compress.apply_strip(&mut strip);
            let out = strip[0];
            let l_after = oklab::rgb_to_oklab(out[0], out[1], out[2], &m1)[0];
            let l_diff = (l_after - l_before).abs();
            assert!(
                l_diff < 0.005,
                "lightness drift for {rgb:?}: before {l_before} after {l_after} diff {l_diff}"
            );
        }
    }

    #[test]
    fn matrix_invert_round_trip() {
        let m_in = oklab::lms_to_rgb_matrix(ColorPrimaries::Bt709).unwrap();
        let m_out = invert_3x3(&m_in).unwrap();
        let m_back = invert_3x3(&m_out).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m_in[i][j] - m_back[i][j]).abs() < 1e-4,
                    "matrix invert roundtrip drift at [{i}][{j}]: in {} back {}",
                    m_in[i][j],
                    m_back[i][j]
                );
            }
        }
    }

    #[test]
    fn empty_strip_is_noop() {
        let compress = bt709_soft_compress(0.9);
        let mut empty: alloc::vec::Vec<[f32; 3]> = alloc::vec::Vec::new();
        compress.apply_strip(&mut empty);
        assert!(empty.is_empty());
    }
}
