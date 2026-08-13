//! Content-light-level (CLL) measurement for HDR pixel data.
//!
//! Histogram-based and SOTA scalar/SIMD reductions over relative-linear
//! `RgbF32` / `RgbaF32` buffers, scaled by a [`DiffuseWhite`] anchor into
//! absolute cd/m². The primitives are exposed as an extension trait
//! ([`CllMeasure`]) on [`ContentLightLevel`] so the call sites stay
//! identical to the (pre-relocation) inherent-impl shape — `cargo add
//! zenpixels-convert` + `use zenpixels_convert::CllMeasure` is the
//! whole upgrade path.
//!
//! This module owns:
//!
//! - [`LightLevelMethod`] — per-pixel reduction (MaxRgb / BT.2020 luma).
//! - [`LightLevelHistogram`] — log-scale histogram primitive with max,
//!   mean, and linearly-interpolated percentile readouts.
//! - [`CllMeasure`] — extension trait on `ContentLightLevel` carrying
//!   `measure_max`, `measure_max_smoothed`, `measure_robust`,
//!   `measure_percentile`, `measure_histogram`.
//! - The scalar and tiered-SIMD kernels behind the trait methods.
//!
//! The bit-exact deprecated `ContentLightLevel::measure(px, white)` 2-arg
//! method stays in `zenpixels::hdr` (frozen public surface for the
//! 0.2.14 release line). This module is the post-0.2.14 home for
//! everything richer.

use alloc::boxed::Box;
use alloc::vec;

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

/// Log-scale histogram of per-pixel light levels in cd/m².
///
/// Built by [`CllMeasure::measure_histogram`]. Exposes the
/// spec-literal max, arithmetic mean (the MaxFALL component), and
/// arbitrary percentile via a CDF walk over the binned distribution.
/// Bins are log2-spaced over `[BIN_MIN_NITS, BIN_MAX_NITS]` so the
/// high-DR range is well-resolved at ~0.02 stops per bin.
///
/// **Why the histogram is the primitive.** Defect-driven outliers
/// (stuck pixels, sensor noise spikes, specular blowouts) want a
/// percentile readout; naturally-sparse-bright content
/// (astrophotography, fireworks, candle in a dark room) wants the
/// literal max. A fixed-percentile API silently miscalibrates one
/// or the other; surfacing the histogram lets the caller commit to
/// a content policy explicitly. See
/// <https://github.com/imazen/zenpixels/issues/54> for the design.
///
/// The histogram is also the cheapest way to compute multiple
/// readouts — the per-pixel scan is the expensive step, and CDF
/// lookups are O(bins) after.
#[derive(Clone, Debug)]
#[non_exhaustive]
#[doc(hidden)]
pub struct LightLevelHistogram {
    bins: Box<[u32]>,
    total: u64,
    sum_nits: f64,
    literal_max_nits: f32,
    method: LightLevelMethod,
}

impl LightLevelHistogram {
    /// Lower edge of bin 0, in cd/m². Anything ≤ this (incl. 0 and
    /// negatives that survived clamping) lands in bin 0.
    pub const BIN_MIN_NITS: f32 = 0.005;
    /// Upper edge of the last bin, in cd/m². Anything ≥ this saturates
    /// into the last bin. The PQ container peak is 10 000 cd/m².
    pub const BIN_MAX_NITS: f32 = 10_000.0;
    /// Number of bins. 1024 covers `[0.005, 10000]` at ~0.0204 stops
    /// per bin (well below the cone JND), fits in L1 at 4 KiB.
    pub const NUM_BINS: usize = 1024;

    // log2 of the range endpoints (constants, not from libm at runtime).
    // log2(0.005) = -log2(200) = -(log2(128) + log2(1.5625)) ≈ -7.6438561
    // log2(10000) = log2(2^13 · 1.220703125) ≈ 13.287712
    // (computed in f64 at design time; pinned here so no_std builds need
    // no libm dep at runtime to know the bin geometry.)
    const LOG2_MIN: f32 = -7.643_856;
    const LOG2_MAX: f32 = 13.287_712;
    #[inline(always)]
    const fn log2_step() -> f32 {
        (Self::LOG2_MAX - Self::LOG2_MIN) / (Self::NUM_BINS as f32)
    }
    #[inline(always)]
    const fn inv_log2_step() -> f32 {
        1.0 / Self::log2_step()
    }

    /// Spec-literal MaxCLL — the largest per-pixel light level observed
    /// (CTA-861.3 strict reading). Exact, not bin-quantised.
    pub fn max(&self) -> f32 {
        self.literal_max_nits
    }

    /// Arithmetic mean of per-pixel light levels — the MaxFALL component
    /// for a single frame. `0.0` for an empty histogram.
    pub fn mean(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        (self.sum_nits / self.total as f64) as f32
    }

    /// Percentile of the light-level distribution, in cd/m².
    /// `percentile` is in `[0.0, 1.0]`; out-of-range inputs clamp, NaN
    /// maps to 0. `1.0` returns [`max`](Self::max) exactly (no bin
    /// quantisation at the spec-literal value).
    ///
    /// Intermediate percentiles walk the binned CDF, identify the bin
    /// where the cumulative count first crosses `percentile · total`,
    /// and **linearly interpolate within that bin** (in log2 space, to
    /// match the log2-uniform bin spacing). Returned values land
    /// strictly between the bin edges and the literal max is preserved
    /// when the threshold falls in the bin holding the maximum sample.
    /// Resolution is ~0.0006 stops at typical pixel counts (≥ 4 MP) and
    /// degrades smoothly as content fills fewer pixels per bin —
    /// always finer than the bin floor (~0.02 stops) returned by a
    /// naïve walk.
    pub fn percentile(&self, percentile: f32) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        // NaN check first — `clamp` panics on NaN bounds and propagates NaN
        // through the input; we've already documented NaN → 0 above.
        let p = if percentile.is_nan() {
            0.0
        } else {
            percentile.clamp(0.0, 1.0)
        };
        if p >= 1.0 {
            return self.literal_max_nits;
        }
        if p <= 0.0 {
            // The 0-th percentile is the floor of the distribution. We
            // don't track a literal-min, and reporting "the lower edge
            // of bin 0" (`BIN_MIN_NITS` ≈ 0.005) would surprise callers
            // who reasonably expect `p=0` → `0.0`. Pin to 0.
            return 0.0;
        }
        let threshold = (p as f64 * self.total as f64) as u64;
        let mut cum: u64 = 0;
        let inv_step = Self::inv_log2_step();
        for (i, &count) in self.bins.iter().enumerate() {
            let count_u64 = count as u64;
            cum += count_u64;
            if cum >= threshold {
                // Fraction of `count` pixels that fall ≤ threshold
                // within this bin. `count_before = cum - count_u64`
                // is the running total before this bin. Compute in f64
                // to keep precision when `count` reaches into the
                // millions on large frames (f32 mantissa is 23 bits).
                let count_before = cum - count_u64;
                let fraction = if count_u64 > 0 {
                    let inside = threshold.saturating_sub(count_before) as f64;
                    let f = (inside / count_u64 as f64).clamp(0.0, 1.0);
                    f as f32
                } else {
                    0.0
                };
                let log2_interp = Self::LOG2_MIN + (i as f32 + fraction) / inv_step;
                let interp = fast_exp2(log2_interp).max(0.0);
                // The bin holding the literal max must NEVER report a
                // value above it — `fast_exp2` rounding could otherwise
                // overshoot by a u16-nit code on the last reachable bin.
                return interp.min(self.literal_max_nits);
            }
        }
        self.literal_max_nits
    }

    /// The per-pixel reduction used when this histogram was built.
    pub fn method(&self) -> LightLevelMethod {
        self.method
    }

    /// Total pixels accumulated (equals `width × height` for the
    /// contiguous-RGB(A) measure path).
    pub fn total_pixels(&self) -> u64 {
        self.total
    }

    /// Raw bin counts; index `i` covers
    /// `[BIN_MIN_NITS · 2^(i·log2_step), BIN_MIN_NITS · 2^((i+1)·log2_step))`.
    /// Useful for plotting or composing custom readouts (multi-percentile,
    /// mode, etc.).
    pub fn bins(&self) -> &[u32] {
        &self.bins
    }
}

/// `no_std` `log2` for a positive `f32` — degree-2 minimax polynomial
/// on the mantissa. Max error ~0.01 stops over the input domain, well
/// below the 0.02-stop bin width of [`LightLevelHistogram`]. Inputs ≤ 0
/// return `f32::NEG_INFINITY` so the caller clamps into bin 0.
#[inline]
fn fast_log2(x: f32) -> f32 {
    use core::f32::consts::LOG2_E;
    // !(x > 0.0) catches NaN and ≤ 0 in one branch; the partial_cmp
    // rewrite clippy suggests doesn't read more clearly here.
    if let Some(core::cmp::Ordering::Greater) = x.partial_cmp(&0.0) {
        let bits = x.to_bits();
        let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
        // Mantissa reconstructed as a float in `[1.0, 2.0)`.
        let mantissa = f32::from_bits((bits & 0x7F_FFFF) | (127 << 23));
        let f = mantissa - 1.0;
        // log2(1+f) ≈ f · (log2(e) − (log2(e) − 1)·f), Horner-form
        // minimax (log2(e) ≈ 1.4426950, the leading constant).
        let log2_mantissa = f * (LOG2_E - (LOG2_E - 1.0) * f);
        (exponent as f32) + log2_mantissa
    } else {
        f32::NEG_INFINITY
    }
}

/// `no_std` `exp2` — degree-3 minimax polynomial on the fractional part
/// plus a bit-fiddle for the integer power-of-2 component. Accuracy
/// ample for the bin-edge → cd/m² conversion in
/// [`LightLevelHistogram::percentile`] (a percentile result is
/// quantised to a bin edge anyway). Inputs outside the f32 exponent
/// range saturate to 0 or `INFINITY` instead of wrapping.
#[inline]
fn fast_exp2(x: f32) -> f32 {
    if !x.is_finite() {
        return if x > 0.0 { f32::INFINITY } else { 0.0 };
    }
    // `as i32` truncates toward 0; floor differs for negative x.
    let mut i = x as i32;
    if (i as f32) > x {
        i -= 1;
    }
    let f = x - (i as f32);
    // 2^f ≈ 1 + ln(2)·f + 0.2402264·f² + 0.0554976·f³ (deg-3 minimax on [0,1]).
    // The leading coefficient is ln(2) by Taylor identity; the higher-order
    // terms are minimax-fit constants that don't match a named `consts`.
    let pf = 1.0 + f * (core::f32::consts::LN_2 + f * (0.240_226_4 + f * 0.055_497_6));
    // 2^i via f32 exponent bits (bias 127, shift 23). Saturate outside
    // the normal range; subnormals and overflow handled by the clamp.
    let biased = i + 127;
    if biased <= 0 {
        return 0.0;
    }
    if biased >= 255 {
        return f32::INFINITY;
    }
    let two_i = f32::from_bits((biased as u32) << 23);
    two_i * pf
}

/// Compute the bin index for a cd/m² value via the log2 mapping
/// pinned in [`LightLevelHistogram`]. Values ≤ `BIN_MIN_NITS` go to
/// bin 0; values ≥ `BIN_MAX_NITS` saturate to the last bin.
#[inline(always)]
fn bin_for_nits(value_nits: f32) -> usize {
    if value_nits <= LightLevelHistogram::BIN_MIN_NITS {
        return 0;
    }
    if value_nits >= LightLevelHistogram::BIN_MAX_NITS {
        return LightLevelHistogram::NUM_BINS - 1;
    }
    let log2 = fast_log2(value_nits);
    let bin =
        ((log2 - LightLevelHistogram::LOG2_MIN) * LightLevelHistogram::inv_log2_step()) as usize;
    if bin >= LightLevelHistogram::NUM_BINS {
        LightLevelHistogram::NUM_BINS - 1
    } else {
        bin
    }
}

/// Single-pass 3×1 horizontal-box-filtered row scan for
/// `measure_max_smoothed`: one row of `N`-channel f32 pixels →
/// `(smoothed_max_relative, unsmoothed_sum_relative)`.
///
/// Per-pixel `m[i] = reduce(R, G, B)` (per `method`) is computed once;
/// the smoothed running max tracks `max over i of mean(m[i-1], m[i], m[i+1])`
/// with mirror-padding at the row edges. The sum is the *unsmoothed*
/// arithmetic sum — `mean(mean(...))` is just the mean (linearity of
/// expectation) and CTA-861.3 MaxFALL is the literal arithmetic mean, so
/// the box filter only affects the max readout, not MaxFALL.
///
/// Why 3×1 over 3×3: a 3×3 mean needs an explicit row buffer (~16 KB for
/// 4K-wide rows) and doubles memory traffic; 3×1 is one sliding window of
/// three floats, single pass over the row, no allocation. 3×1 still
/// suppresses the dominant defect modes — single stuck pixels, denormal /
/// near-infinity values that escaped a poorly-clamped pipeline, specular
/// single-pixel blowouts. Real bright features that span ≥2 horizontal
/// pixels (small stars, sparks, candle flames) survive proportional to
/// their width.
///
/// State is two scalars (`prev`, `curr`); LLVM keeps them in registers, the
/// per-pixel cost is `reduce + 2 adds + 1 compare + 1 f64 add`. Memory
/// traffic matches `scan_row_max_mean` (no scratch buffer, no second pass).
///
/// Returns relative-linear units; the caller scales by `white_nits` at
/// end-of-image.
#[inline]
fn scan_row_max_mean_smoothed<const N: usize>(row: &[f32], method: LightLevelMethod) -> (f32, f64) {
    const ONE_THIRD: f32 = 1.0 / 3.0;
    let pixel_count = row.len() / N;
    if pixel_count == 0 {
        return (0.0, 0.0);
    }

    // Per-pixel reduce closure — `method` is loop-invariant Copy, LLVM
    // hoists the match out of the inner loop.
    let reduce_at = |i: usize| -> f32 {
        let px: &[f32; N] = row[i * N..(i + 1) * N].try_into().unwrap();
        match method {
            LightLevelMethod::MaxRgb => 0.0_f32.max(px[0]).max(px[1]).max(px[2]),
            LightLevelMethod::LuminanceBt2020 => {
                let r = 0.0_f32.max(px[0]);
                let g = 0.0_f32.max(px[1]);
                let b = 0.0_f32.max(px[2]);
                0.262_7 * r + 0.678_0 * g + 0.059_3 * b
            }
        }
    };

    // Degenerate widths: the 3-pixel window collapses, return the trivial
    // reading. Box filter at width=1 mirror-pads to (m,m,m) → mean = m.
    if pixel_count == 1 {
        let m = reduce_at(0);
        return (m, f64::from(m));
    }
    if pixel_count == 2 {
        let m0 = reduce_at(0);
        let m1 = reduce_at(1);
        // Mirror pad: m_smooth[0] = (m0+m0+m1)/3 ; m_smooth[1] = (m0+m1+m1)/3.
        let s0 = (2.0 * m0 + m1) * ONE_THIRD;
        let s1 = (m0 + 2.0 * m1) * ONE_THIRD;
        return (s0.max(s1), f64::from(m0) + f64::from(m1));
    }

    // pixel_count >= 3 — single-pass streaming with a 3-element sliding
    // window. `max_x3` holds the un-divided 3-sum; we divide by 3 once at
    // the end to keep the hot loop free of constant multiplies.
    let m0 = reduce_at(0);
    let m1 = reduce_at(1);
    let mut prev = m0;
    let mut curr = m1;
    // i=0: mirror-pad left → m_smooth_x3 = m0 + m0 + m1
    let mut max_x3 = 2.0 * m0 + m1;
    let mut sum = f64::from(m0);

    for i in 2..pixel_count {
        let next = reduce_at(i);
        // Smoothed value at pixel (i-1): (prev + curr + next) / 3.
        let s = prev + curr + next;
        if s > max_x3 {
            max_x3 = s;
        }
        sum += f64::from(curr);
        prev = curr;
        curr = next;
    }
    // i=pixel_count-1: mirror-pad right → m_smooth_x3 = prev + curr + curr.
    let s_last = prev + 2.0 * curr;
    if s_last > max_x3 {
        max_x3 = s_last;
    }
    sum += f64::from(curr);

    (max_x3 * ONE_THIRD, sum)
}

/// CLL measurement extension trait on [`ContentLightLevel`].
///
/// Carries the histogram-based and SOTA scalar/SIMD measurement
/// entrypoints. Implemented for `ContentLightLevel` only; users call
/// these as associated functions just like the (pre-relocation)
/// inherent impls — `ContentLightLevel::measure_max(px, white,
/// method)` etc., once `use zenpixels_convert::CllMeasure;` is in
/// scope.
///
/// MaxFALL is always the arithmetic mean (CTA-861.3 spec-literal),
/// independent of which entrypoint produces the MaxCLL reading.
pub trait CllMeasure {
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
    /// **SOTA performance.** This is the hot path for spec-conformant
    /// CLL metadata — the kind of measurement that runs on every frame
    /// of every encode. Implementation skips the histogram entirely:
    /// SIMD per-pixel `max + sum` only, scaled by the diffuse-white
    /// anchor at end-of-image. The SIMD path is unconditional (runtime
    /// dispatch — no cargo feature, no `-C target-cpu` flag needed): on
    /// a Ryzen 9 7950X via the AVX2 tier this reaches ≥1 Gpix/s
    /// sustained, several times the histogram path's throughput
    /// (`examples/measure_histogram_throughput.rs` prints both).
    fn measure_max(
        px: PixelSlice<'_>,
        white: DiffuseWhite,
        method: LightLevelMethod,
    ) -> Option<ContentLightLevel>;

    /// **Internal / experimental.** 3×1 horizontal box-filtered max as
    /// an alternative defect-rejection strategy to percentile-based
    /// [`measure_robust`](Self::measure_robust).
    ///
    /// **Kept on the trait but doc-hidden** because the 2026-06-22
    /// audited shootout did NOT crown this method under any of the 6
    /// ranking criteria (`mean_de2000`, `de2000_p95`, `de2000_p99`,
    /// `pct_above_de5`, `de_ok_mean`, `de_ok_p95`). On the 76-sample
    /// imazen-26 corpus it was a near-tie with `measure_max` and
    /// uniformly behind `measure_robust` on tail metrics. Production
    /// callers should pick [`measure_max`](Self::measure_max) (spec
    /// strict / sparse-bright) or [`measure_robust`](Self::measure_robust)
    /// (defect-tolerant) instead. May be removed in 0.3.0 if no usage
    /// case emerges.
    ///
    /// Each pixel's value contributes through the local 3-tap horizontal
    /// mean (`(m[i-1] + m[i] + m[i+1]) / 3`, mirror-padded at row edges).
    /// One stuck pixel at 10 000 cd/m² in a 0.005 cd/m² background reads
    /// as ~3 333 instead of 10 000; real bright features spanning ≥2
    /// horizontal pixels survive proportionally. MaxFALL is unchanged
    /// (mean of a 3×1 box-filtered image equals mean of the original).
    /// Same input contract as [`measure_max`](Self::measure_max).
    #[doc(hidden)]
    fn measure_max_smoothed(
        px: PixelSlice<'_>,
        white: DiffuseWhite,
        method: LightLevelMethod,
    ) -> Option<ContentLightLevel>;

    /// **Internal / experimental.** Convenience wrapper around
    /// [`measure_percentile`](Self::measure_percentile) at
    /// [`DEFAULT_PERCENTILE`](ContentLightLevel::DEFAULT_PERCENTILE).
    ///
    /// **Kept on the trait but doc-hidden** because the 2026-06-22
    /// audited shootout showed it splits 3-3 against
    /// [`measure_max`](Self::measure_max) on the corpus (winning the
    /// 3 tail-aware metrics by 1.4-1.8 % but losing
    /// `mean_de2000` / `pct_above_de5` / `de_ok_mean`). On the
    /// user-visible "clearly-different fraction" (`pct_above_de5`)
    /// it loses by 11 % relative. Production callers should use
    /// [`measure_max`](Self::measure_max) (default) or
    /// [`measure_percentile`](Self::measure_percentile) (explicit
    /// percentile with a documented content policy). May be removed
    /// in 0.3.0 if no usage case emerges.
    ///
    /// Same input contract as [`measure_max`](Self::measure_max).
    #[doc(hidden)]
    fn measure_robust(
        px: PixelSlice<'_>,
        white: DiffuseWhite,
        method: LightLevelMethod,
    ) -> Option<ContentLightLevel>;

    /// Percentile-aware MaxCLL + mean MaxFALL.
    ///
    /// **Secondary API** — the recommended production default is
    /// [`measure_max`](Self::measure_max). Use this when your content
    /// policy needs explicit percentile-based defect rejection (e.g.
    /// sensor-noisy capture path where single hot pixels would
    /// over-drive downstream tone-mapping).
    ///
    /// `percentile` is in `[0.0, 1.0]` and **has no default** — the
    /// caller commits to a percentile value explicitly per content
    /// policy. `1.0` is the spec-literal max (use
    /// [`measure_max`](Self::measure_max) directly if that's the goal).
    /// `0.99999` ([`DEFAULT_PERCENTILE`](ContentLightLevel::DEFAULT_PERCENTILE))
    /// is the tail-tightest tested value in the 2026-06-22 audited
    /// shootout — trades ~11 % more clearly-different pixels overall
    /// for ~1.5 % tighter worst-1-5 % tail.
    ///
    /// Same input contract as [`measure_histogram`](Self::measure_histogram).
    /// MaxFALL is always the arithmetic mean (CTA-861.3 / spec-literal),
    /// independent of `percentile`.
    #[doc(hidden)]
    fn measure_percentile(
        px: PixelSlice<'_>,
        white: DiffuseWhite,
        percentile: f32,
        method: LightLevelMethod,
    ) -> Option<ContentLightLevel>;

    /// Build a log-scale [`LightLevelHistogram`] of per-pixel light levels
    /// from relative-linear `RgbF32` / `RgbaF32` pixels.
    ///
    /// `white` anchors the relative scale to absolute cd/m² (sample `1.0`
    /// = `white` nits; [`DiffuseWhite::BT2408`] = 203 is the convention).
    /// `method` picks the per-pixel reduction (see [`LightLevelMethod`]).
    ///
    /// The histogram is the *primitive* — call [`LightLevelHistogram::max`],
    /// [`LightLevelHistogram::mean`], [`LightLevelHistogram::percentile`]
    /// (or [`bins`](LightLevelHistogram::bins) for custom CDF walks) to
    /// derive whatever readouts your content policy requires. See the
    /// issue #54 design rationale for why we don't bake a fixed
    /// percentile into a single-call API.
    ///
    /// Returns `None` for non-relative-linear `RgbF32`/`RgbaF32` input;
    /// `Some(empty)` for zero-area input (`total_pixels() == 0`,
    /// readouts return `0.0`). Strided rows handled; alpha ignored.
    #[doc(hidden)]
    fn measure_histogram(
        px: PixelSlice<'_>,
        white: DiffuseWhite,
        method: LightLevelMethod,
    ) -> Option<LightLevelHistogram>;
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

    fn measure_max_smoothed(
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

        // Scalar streaming path — auto-vectorises to ~1.3 Gpix/s on Zen 4.
        // A hand-rolled SIMD kernel built shifted-by-1 vectors via array
        // round-trips (magetypes f32x8 has no lane-shift/permute), and
        // the store→load forwarding on each chunk cost ~25% net vs the
        // auto-vectorised scalar. The right SIMD path is a two-pass
        // design (deinterleave+reduce into a row scratch, then 3-tap
        // box-max over the scratch), but that's a separate commit.
        let mut max_rel = 0.0_f32;
        let mut sum_rel = 0.0_f64;
        for row in 0..h {
            let row_bytes = &bytes[row * stride..row * stride + row_len];
            let floats: &[f32] = bytemuck::cast_slice(row_bytes);
            let (rm, rs) = if channels == 3 {
                scan_row_max_mean_smoothed::<3>(floats, method)
            } else {
                scan_row_max_mean_smoothed::<4>(floats, method)
            };
            max_rel = max_rel.max(rm);
            sum_rel += rs;
        }

        let wn = f64::from(white_nits);
        let max_nits = f64::from(max_rel) * wn;
        let fall_nits = sum_rel / (w as f64 * h as f64) * wn;
        Some(ContentLightLevel::new(
            nits_to_u16(max_nits),
            nits_to_u16(fall_nits),
        ))
    }

    fn measure_robust(
        px: PixelSlice<'_>,
        white: DiffuseWhite,
        method: LightLevelMethod,
    ) -> Option<ContentLightLevel> {
        <ContentLightLevel as CllMeasure>::measure_percentile(
            px,
            white,
            ContentLightLevel::DEFAULT_PERCENTILE,
            method,
        )
    }

    fn measure_percentile(
        px: PixelSlice<'_>,
        white: DiffuseWhite,
        percentile: f32,
        method: LightLevelMethod,
    ) -> Option<ContentLightLevel> {
        let h = <ContentLightLevel as CllMeasure>::measure_histogram(px, white, method)?;
        Some(ContentLightLevel::new(
            nits_to_u16(f64::from(h.percentile(percentile))),
            nits_to_u16(f64::from(h.mean())),
        ))
    }

    fn measure_histogram(
        px: PixelSlice<'_>,
        white: DiffuseWhite,
        method: LightLevelMethod,
    ) -> Option<LightLevelHistogram> {
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
            return Some(LightLevelHistogram {
                bins: vec![0u32; LightLevelHistogram::NUM_BINS].into_boxed_slice(),
                total: 0,
                sum_nits: 0.0,
                literal_max_nits: 0.0,
                method,
            });
        }

        let stride = px.stride();
        let bytes = px.as_strided_bytes();
        let row_len = w * channels * 4;
        let white_nits = white.nits();

        // SIMD path: 8 sub-histograms (one per SIMD lane on V3 / emulated
        // on NEON & WASM128) avoid the cross-lane scatter conflict on the
        // hot histogram increment. Reduces at the end. archmage + magetypes
        // are hard deps of zenpixels-convert, so SIMD dispatch is always
        // available (tier-fallback handles boxes without V3/NEON).
        Some(simd_kernel::measure_histogram_simd(
            bytes, w, h, stride, channels, row_len, white_nits, method,
        ))
    }
}

/// Test-only helper that derives the same `(MaxCLL, MaxFALL)` pair
/// via the histogram path, so the `measure_max_and_measure_histogram
/// _max_agree_bit_exact` test can cross-check the two paths against
/// each other.
#[cfg(test)]
fn measure_max_via_histogram_for_test(
    px: PixelSlice<'_>,
    white: DiffuseWhite,
    method: LightLevelMethod,
) -> Option<ContentLightLevel> {
    let h = <ContentLightLevel as CllMeasure>::measure_histogram(px, white, method)?;
    Some(ContentLightLevel::new(
        nits_to_u16(f64::from(h.max())),
        nits_to_u16(f64::from(h.mean())),
    ))
}

// ============================================================================
// SIMD measure_histogram path
// ============================================================================

mod simd_kernel {
    use super::{LightLevelHistogram, LightLevelMethod, bin_for_nits};

    /// One SIMD-lane-worth of sub-histogram. We allocate `LANES` of these
    /// so each lane writes to its own histogram and no cross-lane scatter
    /// conflict happens. Lane width is fixed at 8 across all tiers — V3
    /// (AVX2) is natively 8, NEON / WASM128 are 4 lanes wide so magetypes
    /// emulates 8-wide via two registers, and the scalar tier loops one
    /// pixel at a time. 8 × 1024 × 4 bytes = 32 KiB, which fits in a
    /// modern L1d (32–48 KiB) so the histogram pages stay hot through
    /// the scan.
    const LANES: usize = 8;

    /// Flush the f32 lane sums into the f64 running total every this many
    /// chunks (256 chunks = 2 048 samples per lane). f32 accumulation error
    /// grows with the number of sequential adds; flushing bounds the f32
    /// span to 2 048 adds regardless of row width, so MaxFALL stays within
    /// the ±1-nit parity contract even for panorama-wide rows at PQ-peak
    /// nit levels. Cost: one horizontal reduce per 2 048 pixels (~free).
    const SUM_FLUSH_CHUNKS: u32 = 256;

    // BT.2020 NCL luma coefficients — shared with `bt2446a` via the parent
    // module's `BT2020_L*` constants. Re-aliased here so the SIMD splat and
    // scalar tail use the same names that previously appeared in this kernel.
    use crate::hdr::{BT2020_LB as KB, BT2020_LG as KG, BT2020_LR as KR};

    /// Per-lane sub-histograms flattened into a single heap allocation
    /// of `LANES × NUM_BINS` u32s. We address sub-histogram `i` as
    /// `&mut sub_hists[i*NUM_BINS .. (i+1)*NUM_BINS]`. Flat storage
    /// keeps `#![forbid(unsafe_code)]` honoured (no array-shape
    /// transmute) while still giving each SIMD lane its own
    /// conflict-free histogram.
    type SubHists = alloc::boxed::Box<[u32]>;

    fn zero_subhists() -> SubHists {
        alloc::vec![0u32; LANES * LightLevelHistogram::NUM_BINS].into_boxed_slice()
    }

    /// Reduce the per-lane sub-histograms into the final flat histogram.
    fn merge_subhists(sub: &SubHists, out: &mut [u32]) {
        debug_assert_eq!(out.len(), LightLevelHistogram::NUM_BINS);
        for bin in 0..LightLevelHistogram::NUM_BINS {
            let mut total: u32 = 0;
            for lane in 0..LANES {
                total = total.wrapping_add(sub[lane * LightLevelHistogram::NUM_BINS + bin]);
            }
            out[bin] = total;
        }
    }

    /// Main entry. Builds the histogram via the tiered SIMD kernel
    /// (dispatched via `archmage::incant!`) and returns the populated
    /// `LightLevelHistogram`. Mirrors the scalar `measure_histogram`
    /// path's contract: same inputs, same output.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn measure_histogram_simd(
        bytes: &[u8],
        w: usize,
        h: usize,
        stride: usize,
        channels: usize,
        row_len: usize,
        white_nits: f32,
        method: LightLevelMethod,
    ) -> LightLevelHistogram {
        let mut sub = zero_subhists();
        let mut sum_nits = 0.0_f64;
        let mut literal_max_nits = 0.0_f32;

        for row in 0..h {
            let row_bytes = &bytes[row * stride..row * stride + row_len];
            let floats: &[f32] = bytemuck::cast_slice(row_bytes);
            match method {
                LightLevelMethod::MaxRgb => {
                    if channels == 3 {
                        archmage::incant!(
                            accumulate_strip_max_rgb_tier::<3>(
                                floats,
                                white_nits,
                                &mut sub,
                                &mut sum_nits,
                                &mut literal_max_nits,
                            ),
                            [v3, neon, wasm128, scalar]
                        );
                    } else {
                        archmage::incant!(
                            accumulate_strip_max_rgb_tier::<4>(
                                floats,
                                white_nits,
                                &mut sub,
                                &mut sum_nits,
                                &mut literal_max_nits,
                            ),
                            [v3, neon, wasm128, scalar]
                        );
                    }
                }
                LightLevelMethod::LuminanceBt2020 => {
                    if channels == 3 {
                        archmage::incant!(
                            accumulate_strip_luma_bt2020_tier::<3>(
                                floats,
                                white_nits,
                                &mut sub,
                                &mut sum_nits,
                                &mut literal_max_nits,
                            ),
                            [v3, neon, wasm128, scalar]
                        );
                    } else {
                        archmage::incant!(
                            accumulate_strip_luma_bt2020_tier::<4>(
                                floats,
                                white_nits,
                                &mut sub,
                                &mut sum_nits,
                                &mut literal_max_nits,
                            ),
                            [v3, neon, wasm128, scalar]
                        );
                    }
                }
            }
        }

        let mut bins = alloc::vec![0u32; LightLevelHistogram::NUM_BINS].into_boxed_slice();
        merge_subhists(&sub, &mut bins);

        LightLevelHistogram {
            bins,
            total: (w as u64) * (h as u64),
            sum_nits,
            literal_max_nits,
            method,
        }
    }

    /// Tiered SIMD kernel for the `MaxRgb` reduction. Processes one
    /// row of `N`-channel f32 pixels into the per-lane sub-histograms,
    /// the running max, and the running f64 sum. The `N` channel-count
    /// generic is the same shape as the scalar `accumulate_row_max_rgb`
    /// so the alpha lane (when `N == 4`) is ignored uniformly.
    #[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
    pub(crate) fn accumulate_strip_max_rgb_tier<const N: usize>(
        token: Token,
        row: &[f32],
        white_nits: f32,
        sub_hists: &mut [u32],
        sum_nits: &mut f64,
        literal_max_nits: &mut f32,
    ) {
        let zero = f32x8::zero(token);
        let wn = f32x8::splat(token, white_nits);
        let log2_min = f32x8::splat(token, LightLevelHistogram::LOG2_MIN);
        let inv_step = f32x8::splat(token, LightLevelHistogram::inv_log2_step());
        let bin_min_nits = f32x8::splat(token, LightLevelHistogram::BIN_MIN_NITS);
        let num_bins_minus_1 = f32x8::splat(token, (LightLevelHistogram::NUM_BINS - 1) as f32);

        let mut local_max = zero;
        // Accumulate in f32 lanes, flushing into the f64 running total
        // every `SUM_FLUSH_CHUNKS` chunks so the f32 error span is bounded
        // regardless of row width (the previous once-per-row conversion
        // assumed rows ≤ 4K pixels).
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
                // Alpha (chunk[base + 3] when N==4) is ignored.
            }
            let r = f32x8::load(token, &ra);
            let g = f32x8::load(token, &ga);
            let b = f32x8::load(token, &ba);

            // Tier-consistent NaN/negative fold: `v > 0` is an ORDERED
            // compare — false for NaN, for negatives, and for zero on
            // every tier — so the blend picks 0 for all three, matching
            // the scalar tail's `max(0.0)` semantics exactly. A bare
            // `zero.max(v)` chain is NOT tier-consistent for NaN input:
            // x86 `maxps` returns the second operand while NEON/WASM
            // propagate NaN, which zeroed MaxFALL (and could underreport
            // MaxCLL) whenever any sample was NaN. Pinned by the
            // wide-row NaN tests in tests/cll_measure.rs.
            let r = f32x8::blend(r.simd_gt(zero), r, zero);
            let g = f32x8::blend(g.simd_gt(zero), g, zero);
            let b = f32x8::blend(b.simd_gt(zero), b, zero);
            let m_rel = r.max(g).max(b);
            let m_nits = m_rel * wn;

            local_max = local_max.max(m_nits);
            local_sum += m_nits;
            chunks_since_flush += 1;
            if chunks_since_flush == SUM_FLUSH_CHUNKS {
                *sum_nits += f64::from(local_sum.reduce_add());
                local_sum = zero;
                chunks_since_flush = 0;
            }

            // SIMD log2 → bin index. Use `safe = max(m_nits, BIN_MIN_NITS)`
            // so log2(0) doesn't underflow into NaN/-inf.
            let safe = m_nits.max(bin_min_nits);
            let log2 = safe.log2_midp();
            // bin_f = ((log2 − log2_min) · inv_step), clamped to
            // `[0, NUM_BINS − 1]` in SIMD before the scalar bin write.
            let bin_f = ((log2 - log2_min) * inv_step)
                .max(zero)
                .min(num_bins_minus_1);

            let nits_arr = m_nits.to_array();
            let bin_arr = bin_f.to_array();
            // 8 independent scatter writes — one per lane / sub-histogram.
            // Lane `i` writes to `sub_hists[i]`, so no cross-lane
            // conflict is possible.  Same-bin runs WITHIN one sub-
            // histogram (smooth-tone content) still pay the load-add-
            // store latency, which is the dominant remaining cost and
            // the limit on throughput beyond what plain SIMD math gives.
            for i in 0..LANES {
                let bin = saturating_bin_scalar(nits_arr[i], bin_arr[i]);
                sub_hists[i * LightLevelHistogram::NUM_BINS + bin] += 1;
            }
        }

        // Reduce SIMD accumulators into the scalar running totals.
        let row_max = local_max.reduce_max();
        if row_max > *literal_max_nits {
            *literal_max_nits = row_max;
        }
        *sum_nits += f64::from(local_sum.reduce_add());

        // Scalar tail: pixels left over from the strip not divisible by
        // `LANES * N`. Reuses the `bin_for_nits` helper from the scalar
        // path to stay in lock-step with the scalar histogram's bin
        // boundaries.
        let remainder = iter.remainder();
        for chunk in remainder.chunks_exact(N) {
            let r = chunk[0].max(0.0);
            let g = chunk[1].max(0.0);
            let b = chunk[2].max(0.0);
            let m_rel = r.max(g).max(b);
            let m_nits = m_rel * white_nits;
            if m_nits > *literal_max_nits {
                *literal_max_nits = m_nits;
            }
            *sum_nits += f64::from(m_nits);
            // Tail pixels land in sub_hists[0]; merging at the end sums
            // all lanes so this is correct regardless of which lane the
            // tail "lives" in.
            sub_hists[bin_for_nits(m_nits)] += 1;
        }
    }

    /// Tiered SIMD kernel for the `LuminanceBt2020` reduction —
    /// `Y = 0.2627·R + 0.6780·G + 0.0593·B` (clamped non-negative).
    #[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
    pub(crate) fn accumulate_strip_luma_bt2020_tier<const N: usize>(
        token: Token,
        row: &[f32],
        white_nits: f32,
        sub_hists: &mut [u32],
        sum_nits: &mut f64,
        literal_max_nits: &mut f32,
    ) {
        let zero = f32x8::zero(token);
        let wn = f32x8::splat(token, white_nits);
        let kr = f32x8::splat(token, KR);
        let kg = f32x8::splat(token, KG);
        let kb = f32x8::splat(token, KB);
        let log2_min = f32x8::splat(token, LightLevelHistogram::LOG2_MIN);
        let inv_step = f32x8::splat(token, LightLevelHistogram::inv_log2_step());
        let bin_min_nits = f32x8::splat(token, LightLevelHistogram::BIN_MIN_NITS);
        let num_bins_minus_1 = f32x8::splat(token, (LightLevelHistogram::NUM_BINS - 1) as f32);

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
            // `accumulate_strip_max_rgb_tier`. A `.max(zero)` load fold
            // propagated NaN on NEON/WASM (and was order-dependent on
            // x86), poisoning the luminance dot product.
            let r = f32x8::load(token, &ra);
            let g = f32x8::load(token, &ga);
            let b = f32x8::load(token, &ba);
            let r = f32x8::blend(r.simd_gt(zero), r, zero);
            let g = f32x8::blend(g.simd_gt(zero), g, zero);
            let b = f32x8::blend(b.simd_gt(zero), b, zero);

            let y_rel = kr * r + kg * g + kb * b;
            let y_nits = y_rel * wn;

            local_max = local_max.max(y_nits);
            local_sum += y_nits;
            chunks_since_flush += 1;
            if chunks_since_flush == SUM_FLUSH_CHUNKS {
                *sum_nits += f64::from(local_sum.reduce_add());
                local_sum = zero;
                chunks_since_flush = 0;
            }

            let safe = y_nits.max(bin_min_nits);
            let log2 = safe.log2_midp();
            let bin_f = ((log2 - log2_min) * inv_step)
                .max(zero)
                .min(num_bins_minus_1);

            let nits_arr = y_nits.to_array();
            let bin_arr = bin_f.to_array();
            for i in 0..LANES {
                let bin = saturating_bin_scalar(nits_arr[i], bin_arr[i]);
                sub_hists[i * LightLevelHistogram::NUM_BINS + bin] += 1;
            }
        }

        let row_max = local_max.reduce_max();
        if row_max > *literal_max_nits {
            *literal_max_nits = row_max;
        }
        *sum_nits += f64::from(local_sum.reduce_add());

        let remainder = iter.remainder();
        for chunk in remainder.chunks_exact(N) {
            let r = chunk[0].max(0.0);
            let g = chunk[1].max(0.0);
            let b = chunk[2].max(0.0);
            let y_rel = KR * r + KG * g + KB * b;
            let y_nits = y_rel * white_nits;
            if y_nits > *literal_max_nits {
                *literal_max_nits = y_nits;
            }
            *sum_nits += f64::from(y_nits);
            sub_hists[bin_for_nits(y_nits)] += 1;
        }
    }

    /// Saturating scalar bin index — same semantics as `bin_for_nits` in
    /// the parent module, but inlined here so the SIMD hot loop doesn't
    /// pay a function-call overhead.
    #[inline(always)]
    fn saturating_bin_scalar(nits: f32, bin_f: f32) -> usize {
        if nits <= LightLevelHistogram::BIN_MIN_NITS {
            return 0;
        }
        if nits >= LightLevelHistogram::BIN_MAX_NITS {
            return LightLevelHistogram::NUM_BINS - 1;
        }
        let b = bin_f as usize;
        if b >= LightLevelHistogram::NUM_BINS {
            LightLevelHistogram::NUM_BINS - 1
        } else {
            b
        }
    }

    // ── SOTA fast-path: scan_max_mean (no histogram) ────────────────────
    //
    // For the spec-conformant CLL reading the caller only needs MaxCLL +
    // MaxFALL — the literal max and the arithmetic mean. The histogram
    // path's scatter step is wasted work. This pair of SIMD kernels
    // strips that out: per-pixel `max(R,G,B)` (or BT.2020 luma), running
    // max + sum reduced via `reduce_max` / `reduce_add`. No log2, no
    // bin index, no scatter. Returns `(max_rel, sum_rel)` per row; the
    // caller scales by `white.nits()` at end-of-image.

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
    /// SIMD running sum, reduced once per row to scalar. No histogram
    /// store — this is the gigapixel-class hot loop.
    #[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
    pub(crate) fn scan_row_max_rgb_tier<const N: usize>(
        token: Token,
        row: &[f32],
        row_max_rel: &mut f32,
        row_sum_rel: &mut f64,
    ) {
        let zero = f32x8::zero(token);
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
            let r = f32x8::load(token, &ra);
            let g = f32x8::load(token, &ga);
            let b = f32x8::load(token, &ba);
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
        let zero = f32x8::zero(token);
        let kr = f32x8::splat(token, KR);
        let kg = f32x8::splat(token, KG);
        let kb = f32x8::splat(token, KB);

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
            let r = f32x8::load(token, &ra);
            let g = f32x8::load(token, &ga);
            let b = f32x8::load(token, &ba);
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

    // ── Histogram primitive sanity ──────────────────────────────────────

    #[test]
    fn fast_log2_round_trips_through_fast_exp2_at_bin_edges() {
        // The percentile readout uses `fast_exp2(LOG2_MIN + i / inv_step)`
        // to recover the bin's lower-edge cd/m². Pin the round-trip
        // accuracy: at any bin edge the result should land within one
        // bin-width's relative tolerance of the canonical value.
        let inv_step = LightLevelHistogram::inv_log2_step();
        for &i in &[0_usize, 1, 100, 500, 1023] {
            let log2_edge = LightLevelHistogram::LOG2_MIN + (i as f32) / inv_step;
            let recovered = fast_exp2(log2_edge);
            // Verify against the f64 reference via the bit-trick identity.
            let want = libm_pow2_oracle(f64::from(log2_edge));
            let rel = (f64::from(recovered) - want).abs() / want;
            assert!(
                rel < 0.005,
                "bin {i}: fast_exp2 mismatch: got {recovered} want {want}"
            );
        }
    }

    /// Independent f64 oracle for `2^x` — we don't have libm in the
    /// crate but we do have `f64::powi` / std `f64::exp2`.
    #[cfg(feature = "std")]
    fn libm_pow2_oracle(x: f64) -> f64 {
        x.exp2()
    }
    /// no_std fallback oracle: split into integer/fraction, multiply.
    /// Less accurate than std's `exp2` but plenty for the bin-width
    /// tolerance the test demands.
    #[cfg(not(feature = "std"))]
    fn libm_pow2_oracle(x: f64) -> f64 {
        let i = x.floor() as i32;
        let f = x - (i as f64);
        let pf = 1.0
            + f * (0.693_147_180_559_945_3
                + f * (0.240_226_506_959_100_7 + f * 0.055_504_108_664_821_58));
        let two_i = (1u64 << (i + 1023)) as f64 / (1u64 << 1023) as f64;
        two_i * pf
    }

    #[test]
    fn measure_histogram_empty_input_returns_zero_readouts() {
        // Zero-area input is well-defined: total=0, all readouts are 0.
        // `PixelSlice` requires the byte view to satisfy the f32 alignment
        // even for zero rows, so route the empty case through an aligned
        // `&[f32]` (`Vec<f32>` is f32-aligned) cast to bytes.
        let owned: Vec<f32> = Vec::new();
        let bytes: &[u8] = bytemuck::cast_slice(&owned);
        let px = PixelSlice::new(bytes, 1, 0, 12, PixelDescriptor::RGBF32_LINEAR).unwrap();
        let h = <ContentLightLevel as CllMeasure>::measure_histogram(
            px,
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        assert_eq!(h.total_pixels(), 0);
        assert_eq!(h.max(), 0.0);
        assert_eq!(h.mean(), 0.0);
        assert_eq!(h.percentile(0.5), 0.0);
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
    fn defect_spike_percentile_drops_lone_outlier() {
        // Synthetic defect: 10×10 = 100 pixels at 0.5 (= 101.5 nits) plus
        // ONE stuck/specular pixel at 50.0 (= 10 150 nits, then saturated
        // to BIN_MAX_NITS = 10 000). Spec-literal MaxCLL pins to 10 000
        // (saturating-bin clipped from 10 150). p99.99 (drop the top
        // 0.01% = 0.01 pixels rounded down → drops the spike since the
        // threshold lands strictly below 100) returns the background
        // ~101.5. This is the defect-rejection use case.
        let mut pixels = alloc::vec![[0.5_f32; 3]; 100];
        pixels[0] = [50.0; 3]; // the outlier
        let buf = rgbf32(&pixels, 10, 10);

        let lit = <ContentLightLevel as CllMeasure>::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        // Spec-literal preserves the spike (saturated to the bin range).
        assert!(
            lit.max_content_light_level >= 9000,
            "defect spike: spec literal MaxCLL = {} (expected near 10000)",
            lit.max_content_light_level
        );

        // p99 drops the top 1% (~1 pixel) — the spike goes; background ≈ 101.
        let pct = <ContentLightLevel as CllMeasure>::measure_percentile(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            0.99,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        assert!(
            pct.max_content_light_level < 200,
            "p99 should drop the lone defect: got {}",
            pct.max_content_light_level
        );
    }

    #[test]
    fn night_stars_literal_max_preserves_sparse_bright_content() {
        // Astrophotography case (issue #54 motivating example): 1100
        // pixels total — 1000 dark-sky at 0.005 and 100 "stars" at 5.0.
        // Spec-literal MaxCLL keeps the stars visible; a fixed-percentile
        // API at p < 91% would silently clip them, exactly the failure
        // mode the issue calls out.
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

        // p99.99 also keeps them (only 0.01% = 0.11 pixels → 0 pixels
        // dropped, full literal max preserved through the percentile).
        let pct_high = <ContentLightLevel as CllMeasure>::measure_percentile(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            0.9999,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        assert!(
            pct_high.max_content_light_level > 900,
            "p99.99 must keep the stars (none are defects): got {}",
            pct_high.max_content_light_level
        );

        // A naive caller picking p90 would drop the stars (the threshold
        // is at the 990th pixel, which is in the dark-sky region). This
        // is the failure mode a fixed-percentile API would silently
        // create — we let the caller choose so they make the call
        // explicitly.
        let pct_low = <ContentLightLevel as CllMeasure>::measure_percentile(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            0.90,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        assert!(
            pct_low.max_content_light_level < 100,
            "p90 demonstrably loses sparse-bright content: got {}",
            pct_low.max_content_light_level
        );
    }

    #[test]
    fn percentile_zero_and_one_are_well_defined() {
        let buf = rgbf32(&[[0.0; 3], [0.5; 3], [1.0; 3]], 3, 1);
        let h = <ContentLightLevel as CllMeasure>::measure_histogram(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        // p=1.0 → spec-literal max, exact.
        assert!((h.percentile(1.0) - 203.0).abs() < 0.01);
        // p=0.0 → 0 (matches the documented contract).
        assert_eq!(h.percentile(0.0), 0.0);
    }

    #[test]
    fn percentile_interpolates_within_bin_when_threshold_lands_high() {
        // 10 000 pixels all at 5.0 (= 1015 nits exactly). The literal max
        // is 1015 — with linear interpolation the percentile readout at
        // p=0.9999 should land near the literal max (one bin ≈ 0.02 stops
        // wide; the threshold lands 99.99 % of the way through the bin,
        // putting the interpolated value within ~0.01 stops of the max).
        // Naïve floor-of-bin would read ≈ 1002 (one bin below).
        let buf = rgbf32(&[[5.0_f32; 3]; 10_000], 100, 100);
        let h = <ContentLightLevel as CllMeasure>::measure_histogram(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        let p = h.percentile(0.9999);
        // Allow [1010, 1015]: interpolation never overshoots the literal
        // max (cap inside `percentile`) and lands within one nit at this
        // density.
        assert!(
            (1010.0..=1015.0).contains(&p),
            "p99.99 interpolated within bin: expected ≈1015, got {p}"
        );
    }

    #[test]
    fn percentile_interpolation_never_exceeds_literal_max() {
        // Single bin gets the threshold-1 pixel inside it; with
        // interpolation the readout could round above `literal_max_nits`
        // if not capped. Pin the cap.
        let buf = rgbf32(&[[5.0; 3]; 1000], 100, 10);
        let h = <ContentLightLevel as CllMeasure>::measure_histogram(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        for &p in &[0.5_f32, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999] {
            let v = h.percentile(p);
            assert!(
                v <= h.max() + 1e-3,
                "p={p}: percentile {v} must not exceed literal max {}",
                h.max()
            );
        }
    }

    #[test]
    fn percentile_interpolation_beats_floor_precision_on_dense_content() {
        // 1 MP image of pure 5.0 (= 1015 nits). Floor-of-bin would
        // undershoot the literal by ~13 nits (~2 % = one log2 bin width
        // at this brightness). Interpolation should report within ~1 nit
        // of literal.
        let pixels: Vec<[f32; 3]> = alloc::vec![[5.0_f32; 3]; 1024 * 1024];
        let buf = rgbf32(&pixels, 1024, 1024);
        let h = <ContentLightLevel as CllMeasure>::measure_histogram(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        let p = h.percentile(0.9999);
        assert!(
            (p - h.max()).abs() < 2.0,
            "1 MP solid: interpolated p99.99 = {p}, literal max = {} \
             (expected within ~1 nit; floor-of-bin would be ~1002)",
            h.max()
        );
    }

    #[test]
    fn percentile_clamps_nan_and_out_of_range_inputs() {
        let buf = rgbf32(&[[0.5; 3]], 1, 1);
        let h = <ContentLightLevel as CllMeasure>::measure_histogram(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        assert_eq!(h.percentile(f32::NAN), 0.0); // NaN → 0 per doc
        assert!(h.percentile(2.0) > 0.0); // > 1.0 clamps to literal max
        assert_eq!(h.percentile(-0.5), 0.0); // < 0 clamps to 0
    }

    #[test]
    fn measure_histogram_rejects_non_linear_or_non_rgb_f32() {
        // Non-Linear transfer: rejected.
        let desc = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb);
        let mut data = Vec::new();
        for c in [0.5_f32; 3] {
            data.extend_from_slice(&c.to_ne_bytes());
        }
        let buf = PixelBuffer::from_vec(data, 1, 1, desc).unwrap();
        assert!(
            <ContentLightLevel as CllMeasure>::measure_histogram(
                buf.as_slice(),
                DiffuseWhite::BT2408,
                LightLevelMethod::MaxRgb,
            )
            .is_none()
        );
        // Non-f32 format: rejected.
        let desc = PixelDescriptor::RGB8_SRGB;
        let buf = PixelBuffer::from_vec(alloc::vec![0u8; 3], 1, 1, desc).unwrap();
        assert!(
            <ContentLightLevel as CllMeasure>::measure_histogram(
                buf.as_slice(),
                DiffuseWhite::BT2408,
                LightLevelMethod::MaxRgb,
            )
            .is_none()
        );
    }

    #[test]
    fn histogram_bins_exposed_and_sum_to_total() {
        let buf = rgbf32(&[[0.1; 3], [0.5; 3], [1.0; 3], [2.0; 3], [10.0; 3]], 5, 1);
        let h = <ContentLightLevel as CllMeasure>::measure_histogram(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        let bin_total: u64 = h.bins().iter().map(|&c| c as u64).sum();
        assert_eq!(bin_total, h.total_pixels());
        assert_eq!(h.total_pixels(), 5);
        assert_eq!(h.method(), LightLevelMethod::MaxRgb);
    }

    // ── Industry-standard accuracy parity ──────────────────────────────

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
        //   (b) the literal max picks up the outlier exactly (bit-exact
        //       via the `literal_max_nits` accumulator — no histogram
        //       quantisation).
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

    #[test]
    fn measure_max_and_measure_histogram_max_agree_bit_exact() {
        // The histogram path's `LightLevelHistogram::max()` returns
        // `literal_max_nits` (the bit-exact running max, not the
        // bin-quantised lookup). The fast `measure_max` path uses
        // the same f32 max accumulator under the hood. The two
        // values MUST be identical regardless of input — pin it.
        let pixels: Vec<[f32; 3]> = alloc::vec![
            [0.1, 0.2, 0.3],
            [1.5, 0.5, 0.25],
            [0.0, 3.0, 0.5],
            [0.7, 0.7, 0.7],
        ];
        let buf = rgbf32(&pixels, pixels.len() as u32, 1);
        let via_max = <ContentLightLevel as CllMeasure>::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        let via_hist = measure_max_via_histogram_for_test(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        assert_eq!(
            via_max.max_content_light_level,
            via_hist.max_content_light_level
        );
        assert_eq!(
            via_max.max_frame_average_light_level,
            via_hist.max_frame_average_light_level
        );
    }

    // ── measure_max_smoothed (3×1 horizontal box filter) ─────────────────

    #[test]
    fn measure_max_smoothed_suppresses_single_pixel_defect() {
        // 10-wide row, one stuck/specular pixel at column 5 = [50, 0, 0]
        // (= 10 150 nits, saturated to BIN_MAX_NITS = 10 000 for the
        // histogram path; the smoothed path keeps the raw f32 max-of-3
        // chain so the un-saturated 50.0 × 203 / 3 ≈ 3 383 nits shows up
        // after the box filter — that's the whole point).
        let mut pixels = alloc::vec![[0.0_f32; 3]; 10];
        pixels[5] = [50.0; 3];
        let buf = rgbf32(&pixels, 10, 1);

        // Spec-literal max keeps the spike (saturated at 10 000).
        let lit = <ContentLightLevel as CllMeasure>::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        assert!(
            lit.max_content_light_level >= 9000,
            "control: spec-literal keeps the spike, got {}",
            lit.max_content_light_level
        );

        // Smoothed max replaces m[5]=50 with (m[4]+m[5]+m[6])/3 = 50/3
        // ≈ 16.667. × 203 = 3 383.3, rounds to 3 383.
        let sm = <ContentLightLevel as CllMeasure>::measure_max_smoothed(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        let expected = (50.0_f64 / 3.0) * 203.0; // ≈ 3 383.3
        let got = f64::from(sm.max_content_light_level);
        assert!(
            (got - expected).abs() < 2.0,
            "3×1 mean of [0, 50, 0] = 50/3 → {expected:.1} nits, got {got}"
        );
    }

    #[test]
    fn measure_max_smoothed_preserves_three_pixel_cluster() {
        // 10-wide row, 3 adjacent pixels at 5.0 (centered around column 5).
        // Mean of [5, 5, 5] = 5 → spike preserved at full magnitude.
        let mut pixels = alloc::vec![[0.0_f32; 3]; 10];
        pixels[4] = [5.0; 3];
        pixels[5] = [5.0; 3];
        pixels[6] = [5.0; 3];
        let buf = rgbf32(&pixels, 10, 1);

        let sm = <ContentLightLevel as CllMeasure>::measure_max_smoothed(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        // 5.0 × 203 = 1 015 nits exactly.
        assert!(
            sm.max_content_light_level >= 1010 && sm.max_content_light_level <= 1020,
            "3-pixel cluster should preserve peak: got {}",
            sm.max_content_light_level
        );
    }

    #[test]
    fn measure_max_smoothed_two_pixel_cluster_drops_to_two_thirds() {
        // Two adjacent bright pixels in a dark row → smoothed peak is
        // (0 + hot + hot)/3 = 2·hot/3. Documents the trade-off for
        // sub-resolution features.
        let mut pixels = alloc::vec![[0.0_f32; 3]; 10];
        pixels[4] = [9.0; 3];
        pixels[5] = [9.0; 3];
        let buf = rgbf32(&pixels, 10, 1);

        let sm = <ContentLightLevel as CllMeasure>::measure_max_smoothed(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        let expected = (2.0_f64 * 9.0 / 3.0) * 203.0; // 6 · 203 = 1 218
        let got = f64::from(sm.max_content_light_level);
        assert!(
            (got - expected).abs() < 2.0,
            "2-pixel cluster: expected {expected:.0}, got {got}"
        );
    }

    #[test]
    fn measure_max_smoothed_mean_matches_measure_max_mean() {
        // MaxFALL is the literal arithmetic mean (CTA-861.3). Box-filtering
        // the input doesn't change the mean (linearity of expectation), and
        // we explicitly accumulate the unsmoothed sum, so the two paths
        // must agree exactly on MaxFALL for arbitrary content.
        let pixels: Vec<[f32; 3]> = alloc::vec![
            [0.1, 0.2, 0.3],
            [1.5, 0.5, 0.25],
            [0.0, 3.0, 0.5],
            [0.7, 0.7, 0.7],
            [50.0, 0.0, 0.0], // a defect
            [0.1, 0.2, 0.3],
        ];
        let buf = rgbf32(&pixels, pixels.len() as u32, 1);
        let strict = <ContentLightLevel as CllMeasure>::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        let smooth = <ContentLightLevel as CllMeasure>::measure_max_smoothed(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        assert_eq!(
            strict.max_frame_average_light_level, smooth.max_frame_average_light_level,
            "MaxFALL must match the spec-literal arithmetic mean"
        );
        // And the smoothed MaxCLL is strictly below the spec-literal here
        // because the defect drives the spec-literal reading.
        assert!(
            smooth.max_content_light_level < strict.max_content_light_level,
            "smoothed must suppress the defect: strict={}, smooth={}",
            strict.max_content_light_level,
            smooth.max_content_light_level
        );
    }

    #[test]
    fn measure_max_smoothed_mirror_pad_handles_edge_defect() {
        // Defect at column 0 (left edge). Mirror padding makes m[-1] = m[0],
        // so the smoothed value at i=0 is (m[0]+m[0]+m[1])/3 = (hot+hot+0)/3
        // = 2·hot/3. This is the dominant smoothed value, *not* hot/3.
        // The test pins the mirror-padded math.
        let mut pixels = alloc::vec![[0.0_f32; 3]; 10];
        pixels[0] = [30.0; 3];
        let buf = rgbf32(&pixels, 10, 1);
        let sm = <ContentLightLevel as CllMeasure>::measure_max_smoothed(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        let expected = (2.0_f64 * 30.0 / 3.0) * 203.0; // 20·203 = 4 060
        let got = f64::from(sm.max_content_light_level);
        assert!(
            (got - expected).abs() < 2.0,
            "edge defect with mirror pad: expected {expected:.0}, got {got}"
        );
    }

    #[test]
    fn measure_max_smoothed_degenerate_widths() {
        // 1-pixel-wide image: box filter collapses, smoothed == literal.
        let buf1 = rgbf32(&[[2.0; 3]], 1, 1);
        let sm1 = <ContentLightLevel as CllMeasure>::measure_max_smoothed(
            buf1.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        assert_eq!(sm1.max_content_light_level, 406); // 2.0 × 203

        // 2-pixel-wide image: both pixels get mirror padding from
        // themselves. (m0+m0+m1)/3 and (m0+m1+m1)/3; max picks whichever
        // is bigger. For [2.0, 1.0] the max is (2+2+1)/3 = 5/3 ≈ 1.667
        // → 1.667 × 203 = 338.3.
        let buf2 = rgbf32(&[[2.0; 3], [1.0; 3]], 2, 1);
        let sm2 = <ContentLightLevel as CllMeasure>::measure_max_smoothed(
            buf2.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        let expected2 = (5.0_f64 / 3.0) * 203.0;
        let got2 = f64::from(sm2.max_content_light_level);
        assert!(
            (got2 - expected2).abs() < 1.0,
            "width=2: expected {expected2:.0}, got {got2}"
        );
    }

    #[test]
    fn measure_max_smoothed_luma_bt2020_method() {
        // Pure red @ 5.0 with luma method: Y = 0.2627 · 5.0 = 1.3135.
        // Surround with 0 luma; defect at column 5 of a 10-wide row.
        // Smoothed peak = 1.3135 / 3 ≈ 0.4378 → · 203 = 88.9.
        let mut pixels = alloc::vec![[0.0_f32; 3]; 10];
        pixels[5] = [5.0, 0.0, 0.0];
        let buf = rgbf32(&pixels, 10, 1);
        let sm = <ContentLightLevel as CllMeasure>::measure_max_smoothed(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::LuminanceBt2020,
        )
        .unwrap();
        let expected = (0.262_7_f64 * 5.0 / 3.0) * 203.0; // ≈ 88.9
        let got = f64::from(sm.max_content_light_level);
        assert!(
            (got - expected).abs() < 2.0,
            "luma method smoothed: expected {expected:.1}, got {got}"
        );
    }

    #[test]
    fn measure_max_smoothed_zero_image_returns_zero() {
        let buf = rgbf32(&[[0.0; 3]; 4], 4, 1);
        let sm = <ContentLightLevel as CllMeasure>::measure_max_smoothed(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        assert_eq!(sm.max_content_light_level, 0);
        assert_eq!(sm.max_frame_average_light_level, 0);
    }

    #[test]
    fn measure_max_smoothed_rejects_non_linear_or_non_rgb_f32() {
        // Same rejection contract as measure_max — non-Linear transfer.
        let desc = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb);
        let mut data = Vec::new();
        for c in [0.5_f32; 3] {
            data.extend_from_slice(&c.to_ne_bytes());
        }
        let buf = PixelBuffer::from_vec(data, 1, 1, desc).unwrap();
        assert!(
            <ContentLightLevel as CllMeasure>::measure_max_smoothed(
                buf.as_slice(),
                DiffuseWhite::BT2408,
                LightLevelMethod::MaxRgb,
            )
            .is_none()
        );
    }

    // ── measure_robust (DEFAULT_PERCENTILE = 0.99999 convenience) ─────────

    #[test]
    fn default_percentile_constant_is_tail_tightest() {
        // Pin the constant so changing it requires a deliberate update.
        // 0.99999 = tail-tightest tested value in the 2026-06-22 audited
        // HDR→SDR shootout (76 imazen-26 samples × 20 curves × 4 peak
        // methods, scored on tail-aware percentiles + OKLab Euclidean ΔE).
        // The production default in zenpixels-convert is `measure_max`
        // (winning 3 of 6 criteria including the user-visible
        // `pct_above_de5`); this constant exists for callers who
        // explicitly opt into percentile-based defect rejection via
        // `measure_percentile`. See
        // `zen/zentone/benchmarks/shootout_2026-06-22_findings_v2.md`.
        assert_eq!(ContentLightLevel::DEFAULT_PERCENTILE, 0.99999);
    }

    #[test]
    fn measure_robust_equals_measure_percentile_at_default() {
        // Bit-exact alias contract: measure_robust must be the
        // measure_percentile(p=DEFAULT_PERCENTILE) reading for arbitrary
        // content. If they ever disagree, callers reading the alias get a
        // different answer from the explicit call.
        let pixels: Vec<[f32; 3]> = alloc::vec![
            [0.1, 0.2, 0.3],
            [1.5, 0.5, 0.25],
            [0.0, 3.0, 0.5],
            [0.7, 0.7, 0.7],
            [50.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
        ];
        let buf = rgbf32(&pixels, pixels.len() as u32, 1);

        for method in [LightLevelMethod::MaxRgb, LightLevelMethod::LuminanceBt2020] {
            let robust = <ContentLightLevel as CllMeasure>::measure_robust(
                buf.as_slice(),
                DiffuseWhite::BT2408,
                method,
            )
            .unwrap();
            let pct = <ContentLightLevel as CllMeasure>::measure_percentile(
                buf.as_slice(),
                DiffuseWhite::BT2408,
                ContentLightLevel::DEFAULT_PERCENTILE,
                method,
            )
            .unwrap();
            assert_eq!(robust.max_content_light_level, pct.max_content_light_level);
            assert_eq!(
                robust.max_frame_average_light_level,
                pct.max_frame_average_light_level
            );
        }
    }

    #[test]
    fn measure_robust_drops_dominant_defect_vs_measure_max() {
        // The motivating use case: dense content with one defect-driven
        // hot pixel. measure_max returns the spike (CTA-861.3 literal);
        // measure_robust returns the background.
        let mut pixels = alloc::vec![[0.5_f32; 3]; 100_000];
        pixels[0] = [50.0; 3]; // single defect pixel
        let buf = rgbf32(&pixels, 1000, 100);

        let strict = <ContentLightLevel as CllMeasure>::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        let robust = <ContentLightLevel as CllMeasure>::measure_robust(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();

        // Spec-literal preserves the spike (saturated near BIN_MAX_NITS).
        assert!(strict.max_content_light_level >= 9000);
        // Robust drops it — background = 0.5 × 203 ≈ 101.5 nits.
        // p=0.9999 over 100 000 pixels means threshold = 100 000 × 0.9999
        // = 99 990 pixels worth of CDF; the 99 990th pixel is in the
        // background bin (the defect is just 1 pixel). So robust should
        // land at background ≈ 101.5 nits.
        assert!(
            robust.max_content_light_level < 200,
            "measure_robust must drop the single defect: got {}",
            robust.max_content_light_level
        );
        // MaxFALL (literal mean) is unchanged by the percentile choice.
        assert_eq!(
            strict.max_frame_average_light_level,
            robust.max_frame_average_light_level
        );
    }

    #[test]
    fn measure_robust_preserves_dense_bright_content() {
        // 1100-pixel image, 100 stars at 5.0 (= 1015 nits) + 1000 dark.
        // Stars are 9 % of pixels — well above the 0.01 % outlier budget,
        // so they survive the percentile threshold. Readout lands at the
        // bin-edge of the bright bin (one log2 bin ≈ 2 % below the
        // literal max — DEFAULT_PERCENTILE docstring covers the
        // quantisation).
        let mut pixels: Vec<[f32; 3]> = alloc::vec![[0.005_f32; 3]; 1100];
        for star in pixels.iter_mut().take(100) {
            *star = [5.0; 3];
        }
        let buf = rgbf32(&pixels, 100, 11);
        let robust = <ContentLightLevel as CllMeasure>::measure_robust(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        // Linear interpolation within the bin: with the threshold landing
        // near the top of the bright bin, the readout is within ~1 nit of
        // the literal max (1015). A naïve floor-of-bin readout would
        // undershoot to ≈ 1002.
        assert!(
            robust.max_content_light_level >= 1010 && robust.max_content_light_level <= 1020,
            "dense bright content: robust must preserve the peak: got {}",
            robust.max_content_light_level
        );
    }

    #[test]
    fn measure_robust_sparse_bright_cliff() {
        // Image with one bright pixel and 99 dark. p=0.9999 over 100
        // pixels → threshold = 99; the dark bin cum hits 99 before the
        // bright bin → the bright pixel is dropped. Documents the
        // sparse-bright cliff: at small image sizes, single bright
        // pixels disappear. Astrophotography wants `measure_max` here.
        let mut pixels = alloc::vec![[0.005_f32; 3]; 100];
        pixels[0] = [5.0; 3]; // one bright "star"
        let buf = rgbf32(&pixels, 10, 10);
        let robust = <ContentLightLevel as CllMeasure>::measure_robust(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        // 0.005 × 203 = 1.015 — robust reports the dark-bin floor, not
        // the star.
        assert!(
            robust.max_content_light_level < 50,
            "sparse-bright cliff: 1-in-100 bright pixel must be dropped: got {}",
            robust.max_content_light_level
        );
        // And measure_max keeps the star.
        let strict = <ContentLightLevel as CllMeasure>::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        assert!(strict.max_content_light_level > 900);
    }

    #[test]
    fn measure_robust_rejects_non_linear_or_non_rgb_f32() {
        // Same rejection contract as the rest of the measure family.
        let desc = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb);
        let mut data = Vec::new();
        for c in [0.5_f32; 3] {
            data.extend_from_slice(&c.to_ne_bytes());
        }
        let buf = PixelBuffer::from_vec(data, 1, 1, desc).unwrap();
        assert!(
            <ContentLightLevel as CllMeasure>::measure_robust(
                buf.as_slice(),
                DiffuseWhite::BT2408,
                LightLevelMethod::MaxRgb,
            )
            .is_none()
        );
    }

    #[test]
    fn measure_robust_zero_image_returns_zero() {
        let buf = rgbf32(&[[0.0; 3]; 4], 4, 1);
        let robust = <ContentLightLevel as CllMeasure>::measure_robust(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        assert_eq!(robust.max_content_light_level, 0);
        assert_eq!(robust.max_frame_average_light_level, 0);
    }
}
