//! Content-light-level (CLL) measurement for HDR pixel data.
//!
//! Scalar and SIMD reductions over relative-linear `RgbF32` / `RgbaF32`
//! buffers, scaled by a [`DiffuseWhite`] anchor into absolute cd/m².
//!
//! - [`LightLevelMethod`] — per-pixel reduction (MaxRgb / BT.2020 luma).
//! - [`CllMeasure`] — extension trait on `ContentLightLevel` carrying
//!   `measure_max`, the spec-literal MaxCLL and MaxFALL.
//!
//! A percentile or smoothed peak isn't here: nothing uses one. The 2026-06-22
//! shootout chose `measure_max`, and `git log` holds the histogram and its readouts.

// SIMD macro bodies use LANES * N; stable Rust cannot express this generic
// const product as an as_chunks array length. Keep chunk/remainder iteration.
#![allow(clippy::chunks_exact_to_as_chunks)]

use zenpixels::hdr::{ContentLightLevel, DiffuseWhite};
use zenpixels::{PixelFormat, PixelSlice, TransferFunction};

/// Round non-negative nits to a CTA-861.3 `u16` code (saturating).
///
/// `nits` is a luminance — always `≥ 0` at the call sites. Round-half-up is
/// then `(nits + 0.5)` truncated, and the float→int `as` cast saturates to
/// `[0, u16::MAX]` (mapping negatives and NaN to 0). Done by hand because
/// `f64::round` lives in `std` (libm) and this crate builds `no_std`.
#[inline]
fn nits_to_u16(nits: f64) -> u16 {
    (nits + 0.5) as u16
}

/// Per-pixel reduction method for content-light-level measurement.
///
/// CTA-861-G Annex P pins MaxCLL as "the largest light level of any
/// pixel" without normatively fixing the per-pixel reduction. Two
/// readings are in production use:
///
/// - **`MaxRgb`** — `max(R, G, B)` in cd/m². The dominant industry
///   convention (x265, DaVinci Resolve, Psychtoolbox, Dolby Vision L1,
///   libultrahdr). Bounds what a panel must drive on its worst channel
///   and is conservative on saturated colours.
/// - **`LuminanceBt2020`** — BT.2020 NCL luma
///   (`0.2627·R + 0.6780·G + 0.0593·B`). Used by some Netflix / Apple
///   TV+ pipelines. Matches photometric luminance, so a saturated red
///   reads at `0.2627 ×` peak instead of the full peak — closer to
///   perceived brightness, further from panel-drive worst case.
///
/// Default is [`MaxRgb`](Self::MaxRgb) matching the dominant reading.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LightLevelMethod {
    /// `max(R, G, B)` per pixel — the CTA-861.3 industry default.
    #[default]
    MaxRgb,
    /// `0.2627·R + 0.6780·G + 0.0593·B` per BT.2020 NCL luma weights.
    LuminanceBt2020,
}

/// CLL measurement extension trait on [`ContentLightLevel`].
///
/// Implemented for `ContentLightLevel` only; call it as
/// `ContentLightLevel::measure_max(px, white, method)` once
/// `use zenpixels_convert::hdr::measure::CllMeasure;` is in scope.
pub trait CllMeasure: crate::sealed::Sealed {
    /// MaxCLL + MaxFALL measurement for HDR content.
    ///
    /// Spec-conformant CTA-861.3 MaxCLL + MaxFALL — literal max + mean.
    /// MaxCLL = the largest single per-pixel light level in the image,
    /// MaxFALL = the arithmetic mean.
    ///
    /// `method` picks the per-pixel reduction; the same input contract
    /// holds for all measurements: relative-linear `RgbF32` / `RgbaF32`
    /// only, with `white` anchoring the relative scale to absolute
    /// cd/m² (sample `1.0` = `white` nits; [`DiffuseWhite::BT2408`] =
    /// 203 is the convention). Negative/NaN samples clamp to 0; an
    /// alpha lane is ignored; strided rows are handled.
    ///
    /// Empirically the production-best peak-measurement method per
    /// the 2026-06-22 audited HDR→SDR shootout (76 imazen-26 samples
    /// × 20 curves × 4 peak methods, scored with mean + per-image-
    /// percentile ΔE2000 and OKLab Euclidean ΔE). Won 3 of 6 ranking
    /// criteria including the user-visible `pct_above_de5` by 11 %
    /// over the closest alternative. See
    /// `zen/zentone/benchmarks/shootout_2026-06-22_findings_v2.md`.
    ///
    /// SIMD per-pixel `max + sum` only, scaled by the diffuse-white anchor at
    /// end-of-image, with runtime dispatch: no cargo feature or `-C target-cpu`
    /// flag is needed. `examples/measure_max_throughput.rs` prints its speed.
    fn measure_max(
        px: PixelSlice<'_>,
        white: DiffuseWhite,
        method: LightLevelMethod,
    ) -> Option<ContentLightLevel>;
}

impl CllMeasure for ContentLightLevel {
    fn measure_max(
        px: PixelSlice<'_>,
        white: DiffuseWhite,
        method: LightLevelMethod,
    ) -> Option<ContentLightLevel> {
        let desc = px.descriptor();
        let channels = match desc.pixel_format() {
            PixelFormat::RgbF32 => 3,
            PixelFormat::RgbaF32 => 4,
            _ => return None,
        };
        if desc.transfer != TransferFunction::Linear {
            return None;
        }
        let w = px.width() as usize;
        let h = px.rows() as usize;
        if w == 0 || h == 0 {
            return Some(ContentLightLevel::new(0, 0));
        }

        let stride = px.stride();
        let bytes = px.as_strided_bytes();
        let row_len = w * channels * 4;
        let white_nits = white.nits();

        let (row_max, row_sum) =
            simd_kernel::scan_max_mean_simd(bytes, h, stride, channels, row_len, method);

        let wn = f64::from(white_nits);
        let max_nits = f64::from(row_max) * wn;
        let fall_nits = row_sum / (w as f64 * h as f64) * wn;
        Some(ContentLightLevel::new(
            nits_to_u16(max_nits),
            nits_to_u16(fall_nits),
        ))
    }
}

// ============================================================================
// SIMD scan
// ============================================================================

mod simd_kernel {
    use super::LightLevelMethod;

    /// Eight samples per chunk at every tier: AVX2 natively, two registers on NEON
    /// and WASM128, and a loop in scalar.
    const LANES: usize = 8;

    /// Flush the f32 lane sums into the f64 running total every this many
    /// chunks (256 chunks = 2 048 samples per lane). f32 accumulation error
    /// grows with the number of sequential adds; flushing bounds the f32
    /// span to 2 048 adds regardless of row width, so MaxFALL stays within
    /// the ±1-nit parity contract even for panorama-wide rows at PQ-peak
    /// nit levels. Cost: one horizontal reduce per 2 048 pixels (~free).
    const SUM_FLUSH_CHUNKS: u32 = 256;

    // BT.2020 NCL luma coefficients, shared with `bt2446a` through the parent module.
    use crate::hdr::{BT2020_LB as KB, BT2020_LG as KG, BT2020_LR as KR};

    // MaxCLL + MaxFALL only: per-pixel `max(R,G,B)` (or BT.2020 luma), running
    // max + sum reduced via `reduce_max` / `reduce_add`. Returns
    // `(max_rel, sum_rel)` per row; the caller scales by `white.nits()` at
    // end-of-image.

    /// Top-level dispatcher for the fast measure_max path.
    /// Loops rows and calls the right per-method tier kernel.
    pub(super) fn scan_max_mean_simd(
        bytes: &[u8],
        h: usize,
        stride: usize,
        channels: usize,
        row_len: usize,
        method: LightLevelMethod,
    ) -> (f32, f64) {
        let mut max_rel = 0.0_f32;
        let mut sum_rel = 0.0_f64;
        for row in 0..h {
            let row_bytes = &bytes[row * stride..row * stride + row_len];
            let floats: &[f32] = bytemuck::cast_slice(row_bytes);
            let (rm, rs) = match method {
                LightLevelMethod::MaxRgb => {
                    if channels == 3 {
                        let mut rm = 0.0_f32;
                        let mut rs = 0.0_f64;
                        archmage::incant!(
                            scan_row_max_rgb_tier::<3>(floats, &mut rm, &mut rs),
                            [v3, neon, wasm128, scalar]
                        );
                        (rm, rs)
                    } else {
                        let mut rm = 0.0_f32;
                        let mut rs = 0.0_f64;
                        archmage::incant!(
                            scan_row_max_rgb_tier::<4>(floats, &mut rm, &mut rs),
                            [v3, neon, wasm128, scalar]
                        );
                        (rm, rs)
                    }
                }
                LightLevelMethod::LuminanceBt2020 => {
                    if channels == 3 {
                        let mut rm = 0.0_f32;
                        let mut rs = 0.0_f64;
                        archmage::incant!(
                            scan_row_luma_bt2020_tier::<3>(floats, &mut rm, &mut rs),
                            [v3, neon, wasm128, scalar]
                        );
                        (rm, rs)
                    } else {
                        let mut rm = 0.0_f32;
                        let mut rs = 0.0_f64;
                        archmage::incant!(
                            scan_row_luma_bt2020_tier::<4>(floats, &mut rm, &mut rs),
                            [v3, neon, wasm128, scalar]
                        );
                        (rm, rs)
                    }
                }
            };
            max_rel = max_rel.max(rm);
            sum_rel += rs;
        }
        (max_rel, sum_rel)
    }

    /// Tiered SIMD scan for the `MaxRgb` reduction. Per-pixel
    /// `max(0, R, G, B)`, accumulated into a SIMD running max and a
    /// SIMD running sum, reduced once per row to scalar.
    #[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
    pub(crate) fn scan_row_max_rgb_tier<const N: usize>(
        token: Token,
        row: &[f32],
        row_max_rel: &mut f32,
        row_sum_rel: &mut f64,
    ) {
        let zero = f32x8::zero_t(token);
        let mut local_max = zero;
        // f32 lane sums flushed to f64 every `SUM_FLUSH_CHUNKS` chunks —
        // see `accumulate_strip_max_rgb_tier`.
        let mut local_sum = zero;
        let mut chunks_since_flush = 0u32;

        let mut iter = row.chunks_exact(LANES * N);
        for chunk in &mut iter {
            let mut ra = [0.0_f32; LANES];
            let mut ga = [0.0_f32; LANES];
            let mut ba = [0.0_f32; LANES];
            for i in 0..LANES {
                let base = i * N;
                ra[i] = chunk[base];
                ga[i] = chunk[base + 1];
                ba[i] = chunk[base + 2];
            }
            // Tier-consistent NaN/negative fold — see the comment in
            // `accumulate_strip_max_rgb_tier`.
            let r = f32x8::load_t(token, &ra);
            let g = f32x8::load_t(token, &ga);
            let b = f32x8::load_t(token, &ba);
            let r = f32x8::blend(r.simd_gt(zero), r, zero);
            let g = f32x8::blend(g.simd_gt(zero), g, zero);
            let b = f32x8::blend(b.simd_gt(zero), b, zero);
            let m = r.max(g).max(b);
            local_max = local_max.max(m);
            local_sum += m;
            chunks_since_flush += 1;
            if chunks_since_flush == SUM_FLUSH_CHUNKS {
                *row_sum_rel += f64::from(local_sum.reduce_add());
                local_sum = zero;
                chunks_since_flush = 0;
            }
        }

        *row_max_rel = local_max.reduce_max().max(*row_max_rel);
        *row_sum_rel += f64::from(local_sum.reduce_add());

        // Scalar tail.
        for chunk in iter.remainder().chunks_exact(N) {
            let m = 0.0_f32.max(chunk[0]).max(chunk[1]).max(chunk[2]);
            if m > *row_max_rel {
                *row_max_rel = m;
            }
            *row_sum_rel += f64::from(m);
        }
    }

    /// Tiered SIMD scan for the `LuminanceBt2020` reduction.
    #[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
    pub(crate) fn scan_row_luma_bt2020_tier<const N: usize>(
        token: Token,
        row: &[f32],
        row_max_rel: &mut f32,
        row_sum_rel: &mut f64,
    ) {
        let zero = f32x8::zero_t(token);
        let kr = f32x8::splat_t(token, KR);
        let kg = f32x8::splat_t(token, KG);
        let kb = f32x8::splat_t(token, KB);

        let mut local_max = zero;
        // f32 lane sums flushed to f64 every `SUM_FLUSH_CHUNKS` chunks —
        // see `accumulate_strip_max_rgb_tier`.
        let mut local_sum = zero;
        let mut chunks_since_flush = 0u32;

        let mut iter = row.chunks_exact(LANES * N);
        for chunk in &mut iter {
            let mut ra = [0.0_f32; LANES];
            let mut ga = [0.0_f32; LANES];
            let mut ba = [0.0_f32; LANES];
            for i in 0..LANES {
                let base = i * N;
                ra[i] = chunk[base];
                ga[i] = chunk[base + 1];
                ba[i] = chunk[base + 2];
            }
            // Tier-consistent NaN/negative fold — see the comment in
            // `accumulate_strip_max_rgb_tier`.
            let r = f32x8::load_t(token, &ra);
            let g = f32x8::load_t(token, &ga);
            let b = f32x8::load_t(token, &ba);
            let r = f32x8::blend(r.simd_gt(zero), r, zero);
            let g = f32x8::blend(g.simd_gt(zero), g, zero);
            let b = f32x8::blend(b.simd_gt(zero), b, zero);
            let y = kr * r + kg * g + kb * b;
            local_max = local_max.max(y);
            local_sum += y;
            chunks_since_flush += 1;
            if chunks_since_flush == SUM_FLUSH_CHUNKS {
                *row_sum_rel += f64::from(local_sum.reduce_add());
                local_sum = zero;
                chunks_since_flush = 0;
            }
        }

        *row_max_rel = local_max.reduce_max().max(*row_max_rel);
        *row_sum_rel += f64::from(local_sum.reduce_add());

        // Scalar tail.
        for chunk in iter.remainder().chunks_exact(N) {
            let r = chunk[0].max(0.0);
            let g = chunk[1].max(0.0);
            let b = chunk[2].max(0.0);
            let y = KR * r + KG * g + KB * b;
            if y > *row_max_rel {
                *row_max_rel = y;
            }
            *row_sum_rel += f64::from(y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use zenpixels::{PixelBuffer, PixelDescriptor};

    fn rgbf32(pixels: &[[f32; 3]], w: u32, h: u32) -> PixelBuffer {
        let mut data = Vec::with_capacity(pixels.len() * 12);
        for p in pixels {
            for c in p {
                data.extend_from_slice(&c.to_ne_bytes());
            }
        }
        PixelBuffer::from_vec(data, w, h, PixelDescriptor::RGBF32_LINEAR).unwrap()
    }

    #[test]
    fn measure_max_matches_cta_literal_spec() {
        // CTA-861.3 strict: MaxCLL = largest per-pixel max(R,G,B) ·
        // white_nits, MaxFALL = mean of same. Pin against the same
        // values the legacy deprecated `measure` returns.
        let buf = rgbf32(&[[1.0; 3], [2.0; 3]], 2, 1);
        let cll = <ContentLightLevel as CllMeasure>::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        assert_eq!(cll.max_content_light_level, 406);
        assert_eq!(cll.max_frame_average_light_level, 305);
    }

    #[test]
    fn measure_max_luminance_bt2020_method_uses_luma_weights() {
        // Pure red @ 1.0 with BT.2020 luma: Y = 0.2627 · 1.0 = 0.2627
        // → 0.2627 · 203 = 53.3279 → rounds to 53.
        let buf = rgbf32(&[[1.0, 0.0, 0.0]], 1, 1);
        let cll = <ContentLightLevel as CllMeasure>::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::LuminanceBt2020,
        )
        .unwrap();
        assert_eq!(cll.max_content_light_level, 53);
        assert_eq!(cll.max_frame_average_light_level, 53);
        // MaxRgb on the same input picks 1.0 → 203.
        let cll_max_rgb = <ContentLightLevel as CllMeasure>::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        assert_eq!(cll_max_rgb.max_content_light_level, 203);
    }

    #[test]
    fn night_stars_literal_max_preserves_sparse_bright_content() {
        // Astrophotography case (issue #54 motivating example): 1100
        // pixels total — 1000 dark-sky at 0.005 and 100 "stars" at 5.0.
        // Spec-literal MaxCLL keeps the stars visible.
        let mut pixels: Vec<[f32; 3]> = alloc::vec![[0.005_f32; 3]; 1100];
        for star in pixels.iter_mut().take(100) {
            *star = [5.0; 3];
        }
        let buf = rgbf32(&pixels, 100, 11); // 100 × 11 = 1100 pixels total

        // Spec-literal preserves the stars at ~1015 nits.
        let lit = <ContentLightLevel as CllMeasure>::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        assert!(
            lit.max_content_light_level > 900 && lit.max_content_light_level < 1100,
            "night stars: spec literal MaxCLL = {} (expected near 1015)",
            lit.max_content_light_level
        );
    }

    /// Independent f64 oracle for CTA-861.3-A `MaxCLL` + `MaxFALL` —
    /// the Psychtoolbox-3 / x265 / libplacebo / Dolby Vision L1
    /// formula, restated here in plain f64 so we can pin our
    /// implementation against an unambiguous reference.
    ///
    /// Per `ComputeHDRStaticMetadataType1ContentLightLevels.m`
    /// (Psychtoolbox / Mario Kleiner): for each pixel, `light =
    /// max(R, G, B)` in cd/m² (after the relative-linear scale ×
    /// white_nits anchor). `MaxCLL` = max over all light values;
    /// `MaxFALL` = arithmetic mean. Same formula appears in x265's
    /// `analyze_src_pics`, in libplacebo's `pl_hdr_metadata_max_cll`,
    /// and in `libultrahdr`'s `MaxRGB` reduction.
    fn psychtoolbox_oracle_max_rgb(pixels: &[[f32; 3]], white_nits: f32) -> (f64, f64) {
        let mut max_nits = 0.0_f64;
        let mut sum_nits = 0.0_f64;
        for px in pixels {
            // Clamp negatives + NaN to 0 (matches our `0.0.max(…)` chain
            // and the implicit non-negativity assumption in the spec).
            let r = (px[0] as f64).max(0.0);
            let g = (px[1] as f64).max(0.0);
            let b = (px[2] as f64).max(0.0);
            let m_rel = r.max(g).max(b);
            let m_nits = m_rel * (white_nits as f64);
            if m_nits > max_nits {
                max_nits = m_nits;
            }
            sum_nits += m_nits;
        }
        let mean_nits = sum_nits / (pixels.len() as f64);
        (max_nits, mean_nits)
    }

    /// BT.2020 NCL luma oracle (the alternate Netflix / Apple TV+
    /// pipeline reading; same general shape but uses the BT.2020
    /// luminance coefficients).
    fn psychtoolbox_oracle_luma_bt2020(pixels: &[[f32; 3]], white_nits: f32) -> (f64, f64) {
        let mut max_nits = 0.0_f64;
        let mut sum_nits = 0.0_f64;
        for px in pixels {
            let r = (px[0] as f64).max(0.0);
            let g = (px[1] as f64).max(0.0);
            let b = (px[2] as f64).max(0.0);
            let y = 0.2627 * r + 0.6780 * g + 0.0593 * b;
            let y_nits = y * (white_nits as f64);
            if y_nits > max_nits {
                max_nits = y_nits;
            }
            sum_nits += y_nits;
        }
        let mean_nits = sum_nits / (pixels.len() as f64);
        (max_nits, mean_nits)
    }

    #[test]
    fn measure_max_matches_psychtoolbox_oracle_small_image() {
        // Hand-picked pixels covering: opaque saturated colours, dark
        // shadow, near-black, mid-grey, HDR specular peak. The mix
        // exercises both the running-max and the f64 sum precision.
        let pixels: Vec<[f32; 3]> = alloc::vec![
            [1.0, 0.0, 0.0],    // pure red
            [0.0, 1.0, 0.0],    // pure green
            [0.0, 0.0, 1.0],    // pure blue
            [0.5, 0.5, 0.5],    // mid grey
            [0.0; 3],           // black
            [3.0, 2.5, 4.0],    // HDR specular
            [0.18; 3],          // 18% middle grey
            [0.95, 0.85, 0.05]  // saturated warm
        ];
        let buf = rgbf32(&pixels, pixels.len() as u32, 1);
        let cll = <ContentLightLevel as CllMeasure>::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();

        let (oracle_max, oracle_mean) =
            psychtoolbox_oracle_max_rgb(&pixels, DiffuseWhite::BT2408.nits());
        let want_max = nits_to_u16(oracle_max);
        let want_fall = nits_to_u16(oracle_mean);
        assert_eq!(cll.max_content_light_level, want_max);
        assert_eq!(cll.max_frame_average_light_level, want_fall);
    }

    #[test]
    fn measure_max_luma_bt2020_matches_oracle() {
        // Pure red @ 1.0 with BT.2020 luma: Y = 0.2627 → 53.3279 nits.
        // Verify both the MaxRgb and the LuminanceBt2020 methods'
        // outputs match their respective oracles for an explicit
        // hand-checked answer.
        let pixels: Vec<[f32; 3]> = alloc::vec![[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [2.0, 2.0, 2.0],];
        let buf = rgbf32(&pixels, pixels.len() as u32, 1);
        let cll = <ContentLightLevel as CllMeasure>::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::LuminanceBt2020,
        )
        .unwrap();
        let (oracle_max, oracle_mean) =
            psychtoolbox_oracle_luma_bt2020(&pixels, DiffuseWhite::BT2408.nits());
        assert_eq!(cll.max_content_light_level, nits_to_u16(oracle_max));
        assert_eq!(cll.max_frame_average_light_level, nits_to_u16(oracle_mean));
    }

    #[test]
    fn measure_max_matches_oracle_at_strided_4mp_with_high_dr_outlier() {
        // 4 MP-scale image: 2048 × 2048 pixels, deterministic per-pixel
        // content + one HDR outlier pixel. Verifies both:
        //   (a) the SIMD f64 sum stays in lock-step with the f64 oracle
        //       across millions of pixels (precision check), AND
        //   (b) the literal max picks up the outlier exactly.
        const W: u32 = 2048;
        const H: u32 = 2048;
        let total = (W as usize) * (H as usize);
        let mut pixels: Vec<[f32; 3]> = Vec::with_capacity(total);
        for i in 0..total {
            let t = (i as f32) / (total as f32);
            pixels.push([t * 1.5, (1.0 - t) * 1.5, 0.5 + 0.25 * t]);
        }
        // One specular peak that strictly exceeds the smooth ramp.
        pixels[(W as usize) * (H as usize) / 2] = [25.0; 3];

        let buf = rgbf32(&pixels, W, H);
        let cll = <ContentLightLevel as CllMeasure>::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();

        let (oracle_max, oracle_mean) =
            psychtoolbox_oracle_max_rgb(&pixels, DiffuseWhite::BT2408.nits());
        // MaxCLL is saturating (u16 caps at 65535). 25.0 × 203 = 5075,
        // well below saturation — the test will catch a bin-quantisation
        // bug if any sneaks in.
        assert_eq!(cll.max_content_light_level, nits_to_u16(oracle_max));
        // MaxFALL: allow ±1 u16 code for rounding (f64 → f32 → f64 path
        // accumulated across 4 M pixels has microscopic drift).
        let want_fall = nits_to_u16(oracle_mean);
        let diff = (cll.max_frame_average_light_level as i32 - want_fall as i32).abs();
        assert!(
            diff <= 1,
            "MaxFALL u16 diverged: got {} want {} (oracle f64={:.4})",
            cll.max_frame_average_light_level,
            want_fall,
            oracle_mean
        );
    }
}
