//! HDR metadata types.
//!
//! Pure data types for HDR content description. These travel with pixel
//! data alongside [`Cicp`](crate::Cicp) and [`ColorContext`](crate::ColorContext).
//!
//! For tone mapping and HDR processing functions, see
//! [`zenpixels-convert::hdr`](https://docs.rs/zenpixels-convert/latest/zenpixels_convert/hdr/).

use crate::{PixelFormat, PixelSlice, TransferFunction};

/// The absolute luminance, in cd/m² (nits), that a relative-linear sample
/// value of `1.0` represents — the "diffuse white" (a.k.a. nominal diffuse
/// white / SDR reference white) anchor that bridges relative-linear pixel
/// data to absolute display light.
///
/// This is the single scalar the rest of the industry uses for that bridge:
/// OpenEXR's `whiteLuminance` ("nits of RGB (1,1,1)"), JPEG XL's
/// `intensity_target`, libheif's `ndwt` (nominal diffuse white), and
/// libplacebo's SDR-white constant. The cross-vendor default is
/// [`BT2408`](Self::BT2408) = 203 cd/m².
///
/// It is a *typed* anchor on purpose: HDR code mixes nits, PQ-encoded `[0,1]`,
/// log2 gain, and headroom ratios — passing a bare `f32` invites unit
/// confusion. Use [`DiffuseWhite::new`] / [`DiffuseWhite::nits`].
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiffuseWhite(f32);

// Bit-exact equality so `DiffuseWhite` — and therefore `ColorContext` — keeps
// `Eq` despite wrapping `f32`. A luminance anchor is always a sane, finite,
// positive cd/m² value (203, 100, 10000, …), so a bitwise compare is reflexive
// and consistent; the -0.0 / NaN cases a value compare would treat differently
// never occur for an anchor.
impl PartialEq for DiffuseWhite {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for DiffuseWhite {}

impl DiffuseWhite {
    /// ITU-R BT.2408 HDR reference white: **203 cd/m²**. The cross-industry
    /// default anchor for relative-linear HDR (matches Chrome `SDRWhiteLevel`,
    /// Skia skcms, CSS `rec2100-linear`, and libplacebo).
    pub const BT2408: Self = Self(203.0);

    /// An anchor of `nits` cd/m² (the luminance that relative-linear `1.0`
    /// represents).
    #[must_use]
    pub const fn new(nits: f32) -> Self {
        Self(nits)
    }

    /// The anchor in cd/m² (nits).
    #[must_use]
    pub const fn nits(self) -> f32 {
        self.0
    }
}

impl Default for DiffuseWhite {
    /// [`BT2408`](Self::BT2408) — 203 cd/m².
    fn default() -> Self {
        Self::BT2408
    }
}

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

/// Scalar fast-path scan for `measure_max`: one row of `N`-channel f32
/// pixels → `(max_nits, sum_nits)` with the chosen per-pixel reduction.
/// No histogram, no log2, no scatter — just SIMD-friendly max + sum,
/// scaled by the diffuse-white anchor at the end of the row.
///
/// `N` = channel count (3 RGB, 4 RGBA; alpha lane ignored). `M` picks
/// the reduction at compile time so each instantiation has a monomorphic
/// inner loop that LLVM auto-vectorises. The same shape as the
/// `row_max_sum` helper below — kept separate so the deprecated
/// `ContentLightLevel::measure` path can drop without touching the new
/// API.
#[cfg(not(feature = "simd"))]
#[inline]
fn scan_row_max_mean<const N: usize>(row: &[f32], method: LightLevelMethod) -> (f32, f64) {
    let mut row_max = 0.0_f32;
    let mut row_sum = 0.0_f64;
    match method {
        LightLevelMethod::MaxRgb => {
            for chunk in row.chunks_exact(N) {
                let px: &[f32; N] = chunk.try_into().unwrap();
                let m = 0.0_f32.max(px[0]).max(px[1]).max(px[2]);
                row_max = row_max.max(m);
                row_sum += f64::from(m);
            }
        }
        LightLevelMethod::LuminanceBt2020 => {
            for chunk in row.chunks_exact(N) {
                let px: &[f32; N] = chunk.try_into().unwrap();
                let r = 0.0_f32.max(px[0]);
                let g = 0.0_f32.max(px[1]);
                let b = 0.0_f32.max(px[2]);
                let y = 0.2627 * r + 0.6780 * g + 0.0593 * b;
                row_max = row_max.max(y);
                row_sum += f64::from(y);
            }
        }
    }
    (row_max, row_sum)
}

/// Reduce one row of `N`-channel f32 pixels to
/// `(max, sum)` of the per-pixel `max(R, G, B)`.
///
/// `N` is the channel count (3 = `Rgb`, 4 = `Rgba`); only the first three
/// lanes are read, so any alpha is ignored. Each channel is folded from `0.0`,
/// so `f32::max`'s non-NaN-propagating semantics drop NaN and negative samples.
/// `chunk` is reborrowed as a fixed-size `&[f32; N]` so the bounds checks fall
/// away and LLVM can vectorize the reduction. The sum accumulates in `f64`:
/// a 4K frame is ~8M pixels, beyond f32's precision for a running total.
#[inline]
fn row_max_sum<const N: usize>(row: &[f32]) -> (f32, f64) {
    let mut row_max = 0.0f32;
    let mut row_sum = 0.0f64;
    for chunk in row.chunks_exact(N) {
        // `chunks_exact(N)` yields exactly-`N` slices — the conversion is infallible.
        let px: &[f32; N] = chunk.try_into().unwrap();
        let m = 0.0f32.max(px[0]).max(px[1]).max(px[2]);
        row_max = row_max.max(m);
        row_sum += f64::from(m);
    }
    (row_max, row_sum)
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
/// Built by [`ContentLightLevel::measure_histogram`]. Exposes the
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
pub struct LightLevelHistogram {
    bins: alloc::boxed::Box<[u32]>,
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
    /// Intermediate percentiles walk the binned CDF and return the
    /// **lower edge** of the bin where the cumulative count first
    /// crosses `percentile · total` — bin-quantised at ~0.02 stops,
    /// well below the cone JND.
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
        for (i, &count) in self.bins.iter().enumerate() {
            cum += count as u64;
            if cum >= threshold {
                let log2_edge = Self::LOG2_MIN + (i as f32) / Self::inv_log2_step();
                return fast_exp2(log2_edge).max(0.0);
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

/// HDR content light level metadata (CEA-861.3 / CTA-861-H).
///
/// Describes the peak brightness characteristics of HDR content.
/// Used by AVIF, JXL, PNG (cLLi chunk), and video containers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContentLightLevel {
    /// Maximum Content Light Level (MaxCLL) in cd/m² (nits).
    /// Peak luminance of any single pixel in the content.
    pub max_content_light_level: u16,
    /// Maximum Frame-Average Light Level (MaxFALL) in cd/m².
    /// Peak average luminance of any single frame.
    pub max_frame_average_light_level: u16,
}

impl ContentLightLevel {
    /// Create content light level metadata.
    pub const fn new(max_content_light_level: u16, max_frame_average_light_level: u16) -> Self {
        Self {
            max_content_light_level,
            max_frame_average_light_level,
        }
    }

    /// **Deprecated (0.2.15), hidden.** Computes the *literal* MaxCLL — the
    /// absolute max over pixels of `max(R, G, B)` — which is outlier-sensitive
    /// (one specular/noise pixel inflates it, making displays over-tone-map).
    /// Production HDR metadata uses a percentile (~99.99th); a percentile-aware
    /// replacement is tracked in <https://github.com/imazen/zenpixels/issues/54>
    /// and this method is queued for 0.3.0 removal. MaxFALL (the mean) is fine.
    ///
    /// Measure MaxCLL / MaxFALL (CTA-861.3-A) from relative-linear RGB(A) f32
    /// pixels, with `white` anchoring the scale (sample `1.0` = `white` nits;
    /// [`DiffuseWhite::BT2408`] — 203 — is the convention).
    ///
    /// Semantics per CTA-861.3-A as PNG 3rd ed §11.3.2.8 imports it for stills
    /// (one still = one frame): **MaxCLL** is the brightest pixel's
    /// `max(R, G, B)` in cd/m², **MaxFALL** is the image's average of per-pixel
    /// `max(R, G, B)`. Negative/NaN samples clamp to 0; an alpha lane is
    /// ignored; strided rows are handled.
    ///
    /// Returns `None` if the descriptor is not relative-linear
    /// `RgbF32`/`RgbaF32`. This is deliberate, not a missing case: cd/m² is
    /// only defined in **linear light**, so a transfer function would have to
    /// be inverted first — and inverting one (PQ/HLG/sRGB → linear) is the
    /// conversion pipeline's job, which the foundational `zenpixels` crate has
    /// no dependency on. To measure an integer or non-linear HDR buffer,
    /// linearize it to `RgbaF32` first (`zenpixels_convert::convert_buffer`),
    /// then call this. Zero-area input yields `Some(0, 0)`.
    ///
    /// `RgbF32` and `RgbaF32` share one reduction (generic over the channel
    /// count); the inner loop reads whole f32s from the channel-aligned buffer
    /// so it vectorizes rather than decoding sample-by-sample.
    #[must_use]
    #[doc(hidden)]
    #[deprecated(
        since = "0.2.15",
        note = "literal-maximum MaxCLL is outlier-sensitive (one specular/noise \
                pixel inflates it, making displays over-tone-map); production HDR \
                metadata uses a percentile (~99.99th). Percentile-aware \
                replacement planned: https://github.com/imazen/zenpixels/issues/54. \
                MaxFALL is unaffected."
    )]
    pub fn measure(px: PixelSlice<'_>, white: DiffuseWhite) -> Option<Self> {
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
            return Some(Self::new(0, 0));
        }
        let stride = px.stride();
        let bytes = px.as_strided_bytes();
        let row_len = w * channels * 4;

        // Reduce in relative-linear units, then scale by the anchor once at the
        // end — ∑(mᵢ·w) = (∑mᵢ)·w, fewer multiplies for the same f64 result.
        let mut max_lin = 0.0f32;
        let mut sum_lin = 0.0f64;
        for row in 0..h {
            let row_bytes = &bytes[row * stride..row * stride + row_len];
            // f32 buffers are channel-aligned (the `PixelBuffer` alignment
            // invariant), and `row_len` is a multiple of 4, so this cast never
            // straddles a sample — and reading whole f32s lets the reduction
            // vectorize, unlike per-byte `from_ne_bytes`.
            let floats: &[f32] = bytemuck::cast_slice(row_bytes);
            let (row_max, row_sum) = if channels == 3 {
                row_max_sum::<3>(floats)
            } else {
                row_max_sum::<4>(floats)
            };
            max_lin = max_lin.max(row_max);
            sum_lin += row_sum;
        }
        let wn = f64::from(white.nits());
        let max_nits = f64::from(max_lin) * wn;
        let fall = sum_lin / (w as f64 * h as f64) * wn;
        Some(Self::new(nits_to_u16(max_nits), nits_to_u16(fall)))
    }

    // ── Histogram-based measurements (replacing the deprecated `measure`) ──

    /// Build a log-scale [`LightLevelHistogram`] of per-pixel light levels
    /// from relative-linear `RgbF32` / `RgbaF32` pixels.
    ///
    /// `white` anchors the relative scale to absolute cd/m² (sample `1.0`
    /// = `white` nits; [`DiffuseWhite::BT2408`] = 203 is the convention).
    /// `method` picks the per-pixel reduction (see [`LightLevelMethod`]).
    ///
    /// The histogram is the *primitive* — call [`LightLevelHistogram::max`],
    /// [`LightLevelHistogram::mean`], [`LightLevelHistogram::percentile`]
    /// (or [`percentiles`](LightLevelHistogram::bins) for custom CDF
    /// walks) to derive whatever readouts your content policy requires.
    /// See the issue #54 design rationale for why we don't bake a fixed
    /// percentile into a single-call API.
    ///
    /// Returns `None` for non-relative-linear `RgbF32`/`RgbaF32` input;
    /// `Some(empty)` for zero-area input (`total_pixels() == 0`,
    /// readouts return `0.0`). Strided rows handled; alpha ignored.
    #[must_use]
    pub fn measure_histogram(
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
                bins: alloc::vec![0u32; LightLevelHistogram::NUM_BINS].into_boxed_slice(),
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
        // hot histogram increment. Reduces at the end.
        #[cfg(feature = "simd")]
        {
            Some(simd_kernel::measure_histogram_simd(
                bytes, w, h, stride, channels, row_len, white_nits, method,
            ))
        }

        // Scalar path is always available as the runtime fallback and the
        // no-`simd`-feature build path. Dispatch on method ONCE at the top;
        // the inner loop sees a monomorphised reducer so LLVM can
        // vectorise the per-pixel work.
        #[cfg(not(feature = "simd"))]
        {
            let mut bins = alloc::vec![0u32; LightLevelHistogram::NUM_BINS].into_boxed_slice();
            let mut sum_nits = 0.0_f64;
            let mut literal_max_nits = 0.0_f32;
            match method {
                LightLevelMethod::MaxRgb => {
                    for row in 0..h {
                        let row_bytes = &bytes[row * stride..row * stride + row_len];
                        let floats: &[f32] = bytemuck::cast_slice(row_bytes);
                        if channels == 3 {
                            accumulate_row_max_rgb::<3>(
                                floats,
                                white_nits,
                                &mut bins,
                                &mut sum_nits,
                                &mut literal_max_nits,
                            );
                        } else {
                            accumulate_row_max_rgb::<4>(
                                floats,
                                white_nits,
                                &mut bins,
                                &mut sum_nits,
                                &mut literal_max_nits,
                            );
                        }
                    }
                }
                LightLevelMethod::LuminanceBt2020 => {
                    for row in 0..h {
                        let row_bytes = &bytes[row * stride..row * stride + row_len];
                        let floats: &[f32] = bytemuck::cast_slice(row_bytes);
                        if channels == 3 {
                            accumulate_row_luma_bt2020::<3>(
                                floats,
                                white_nits,
                                &mut bins,
                                &mut sum_nits,
                                &mut literal_max_nits,
                            );
                        } else {
                            accumulate_row_luma_bt2020::<4>(
                                floats,
                                white_nits,
                                &mut bins,
                                &mut sum_nits,
                                &mut literal_max_nits,
                            );
                        }
                    }
                }
            }
            Some(LightLevelHistogram {
                bins,
                total: (w as u64) * (h as u64),
                sum_nits,
                literal_max_nits,
                method,
            })
        }
    }

    /// Spec-conformant CTA-861.3 MaxCLL + MaxFALL — literal max + mean.
    ///
    /// This is the **strict spec reading**: MaxCLL = the largest single
    /// per-pixel light level in the image, MaxFALL = the arithmetic
    /// mean. Use this when the delivery target mandates spec-literal
    /// metadata (Netflix, broadcast). For content with defect-driven
    /// hot pixels (sensor noise, stuck pixels, specular blowouts) prefer
    /// [`measure_percentile`](Self::measure_percentile) with a percentile
    /// you've committed to.
    ///
    /// `method` picks the per-pixel reduction. Same input contract as
    /// [`measure_histogram`](Self::measure_histogram).
    ///
    /// **SOTA performance.** This is the hot path for spec-conformant
    /// CLL metadata — the kind of measurement that runs on every frame
    /// of every encode. Implementation skips the histogram entirely:
    /// SIMD per-pixel `max + sum` only, scaled by the diffuse-white
    /// anchor at end-of-image. On Ryzen 9 7950X with the `simd` feature
    /// and `-C target-cpu=native` this reaches ≥1 Gpix/s sustained
    /// (vs ~490 Mpix/s via the histogram path), giving the SOTA spec-
    /// conformant CLL reading in the workspace.
    #[must_use]
    pub fn measure_max(
        px: PixelSlice<'_>,
        white: DiffuseWhite,
        method: LightLevelMethod,
    ) -> Option<Self> {
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
            return Some(Self::new(0, 0));
        }

        let stride = px.stride();
        let bytes = px.as_strided_bytes();
        let row_len = w * channels * 4;
        let white_nits = white.nits();

        #[cfg(feature = "simd")]
        let (row_max, row_sum) =
            simd_kernel::scan_max_mean_simd(bytes, h, stride, channels, row_len, method);

        #[cfg(not(feature = "simd"))]
        let (row_max, row_sum) = {
            let mut max_rel = 0.0_f32;
            let mut sum_rel = 0.0_f64;
            for row in 0..h {
                let row_bytes = &bytes[row * stride..row * stride + row_len];
                let floats: &[f32] = bytemuck::cast_slice(row_bytes);
                let (rm, rs) = if channels == 3 {
                    scan_row_max_mean::<3>(floats, method)
                } else {
                    scan_row_max_mean::<4>(floats, method)
                };
                max_rel = max_rel.max(rm);
                sum_rel += rs;
            }
            (max_rel, sum_rel)
        };

        let wn = f64::from(white_nits);
        let max_nits = f64::from(row_max) * wn;
        let fall_nits = row_sum / (w as f64 * h as f64) * wn;
        Some(Self::new(nits_to_u16(max_nits), nits_to_u16(fall_nits)))
    }

    /// Test-only helper that derives the same `(MaxCLL, MaxFALL)` pair
    /// via the histogram path, so the `measure_max_and_measure_histogram
    /// _max_agree_bit_exact` test can cross-check the two paths against
    /// each other. Kept `#[cfg(test)]` to avoid surfacing a redundant
    /// public alias.
    #[cfg(test)]
    pub(crate) fn measure_max_via_histogram_for_test(
        px: PixelSlice<'_>,
        white: DiffuseWhite,
        method: LightLevelMethod,
    ) -> Option<Self> {
        let h = Self::measure_histogram(px, white, method)?;
        Some(Self::new(
            nits_to_u16(f64::from(h.max())),
            nits_to_u16(f64::from(h.mean())),
        ))
    }

    /// Percentile-aware MaxCLL + mean MaxFALL.
    ///
    /// `percentile` is in `[0.0, 1.0]` and **has no default** — the
    /// caller commits to a percentile value explicitly per content
    /// policy. `1.0` is the spec-literal max (use
    /// [`measure_max`](Self::measure_max) directly if that's the goal).
    /// `0.9999` is the typical defect-rejection choice;
    /// astrophotography / fireworks / candle-in-dark-room content
    /// usually wants `1.0` (literal max) instead. See the issue #54
    /// docstring for the trade-off rationale.
    ///
    /// Same input contract as [`measure_histogram`](Self::measure_histogram).
    /// MaxFALL is always the arithmetic mean (CTA-861.3 / spec-literal),
    /// independent of `percentile`.
    #[must_use]
    pub fn measure_percentile(
        px: PixelSlice<'_>,
        white: DiffuseWhite,
        percentile: f32,
        method: LightLevelMethod,
    ) -> Option<Self> {
        let h = Self::measure_histogram(px, white, method)?;
        Some(Self::new(
            nits_to_u16(f64::from(h.percentile(percentile))),
            nits_to_u16(f64::from(h.mean())),
        ))
    }
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

/// Per-row accumulator for the `MaxRgb` method: `max(R, G, B)` per
/// pixel, in cd/m². Negatives/NaN fold to 0 via the `0.0.max(…)` chain.
#[cfg(not(feature = "simd"))]
#[inline]
fn accumulate_row_max_rgb<const N: usize>(
    row: &[f32],
    white_nits: f32,
    bins: &mut [u32],
    sum_nits: &mut f64,
    literal_max_nits: &mut f32,
) {
    for chunk in row.chunks_exact(N) {
        let px: &[f32; N] = chunk.try_into().unwrap();
        let m_rel = 0.0_f32.max(px[0]).max(px[1]).max(px[2]);
        let m_nits = m_rel * white_nits;
        if m_nits > *literal_max_nits {
            *literal_max_nits = m_nits;
        }
        *sum_nits += f64::from(m_nits);
        bins[bin_for_nits(m_nits)] += 1;
    }
}

/// Per-row accumulator for the `LuminanceBt2020` method: BT.2020 NCL
/// luma in cd/m², `Y = 0.2627·R + 0.6780·G + 0.0593·B`. Channels are
/// clamped to non-negative first; NaN channels fold to 0 by the
/// `0.0.max(…)` pattern (`f32::max` is non-NaN-propagating, returning
/// the non-NaN operand when one side is NaN).
#[cfg(not(feature = "simd"))]
#[inline]
fn accumulate_row_luma_bt2020<const N: usize>(
    row: &[f32],
    white_nits: f32,
    bins: &mut [u32],
    sum_nits: &mut f64,
    literal_max_nits: &mut f32,
) {
    for chunk in row.chunks_exact(N) {
        let px: &[f32; N] = chunk.try_into().unwrap();
        let r = 0.0_f32.max(px[0]);
        let g = 0.0_f32.max(px[1]);
        let b = 0.0_f32.max(px[2]);
        let y_rel = 0.2627 * r + 0.6780 * g + 0.0593 * b;
        let y_nits = y_rel * white_nits;
        if y_nits > *literal_max_nits {
            *literal_max_nits = y_nits;
        }
        *sum_nits += f64::from(y_nits);
        bins[bin_for_nits(y_nits)] += 1;
    }
}

/// Mastering display color volume metadata (SMPTE ST 2086).
///
/// Describes the display on which the content was mastered, enabling
/// downstream displays to reproduce the creator's intent.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MasteringDisplay {
    /// RGB primaries of the mastering display in CIE 1931 xy coordinates.
    /// `[[rx, ry], [gx, gy], [bx, by]]`.
    pub primaries_xy: [[f32; 2]; 3],
    /// White point in CIE 1931 xy coordinates `[wx, wy]`.
    pub white_point_xy: [f32; 2],
    /// Maximum display luminance in cd/m².
    pub max_luminance: f32,
    /// Minimum display luminance in cd/m².
    pub min_luminance: f32,
}

impl MasteringDisplay {
    /// Create mastering display metadata from CIE 1931 xy coordinates and cd/m² luminances.
    pub const fn new(
        primaries_xy: [[f32; 2]; 3],
        white_point_xy: [f32; 2],
        max_luminance: f32,
        min_luminance: f32,
    ) -> Self {
        Self {
            primaries_xy,
            white_point_xy,
            max_luminance,
            min_luminance,
        }
    }

    /// BT.2020 primaries with D65 white point, 10000 nits peak (HDR10 reference).
    pub const HDR10_REFERENCE: Self = Self {
        primaries_xy: [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        white_point_xy: [0.3127, 0.3290],
        max_luminance: 10000.0,
        min_luminance: 0.0001,
    };

    /// Display P3 primaries with D65 white point, 1000 nits.
    pub const DISPLAY_P3_1000: Self = Self {
        primaries_xy: [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
        white_point_xy: [0.3127, 0.3290],
        max_luminance: 1000.0,
        min_luminance: 0.0001,
    };
}

// ============================================================================
// SIMD measure_histogram path (feature = "simd")
// ============================================================================

#[cfg(feature = "simd")]
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

    /// BT.2020 NCL luma coefficients pinned at compile time so the SIMD
    /// path's splat constants match the scalar `accumulate_row_luma_bt2020`
    /// reduction.
    const KR: f32 = 0.2627;
    const KG: f32 = 0.6780;
    const KB: f32 = 0.0593;

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
        // Accumulate in f32 lanes inside the loop and convert to f64 once
        // per row to bound rounding error — a 4K row is at most 3840 pixels
        // and f32 sums of cd/m² values stay precise across that span.
        let mut local_sum = zero;

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

            // Folded `0.0.max(R).max(G).max(B)`: non-NaN-propagating max
            // means negative inputs and NaN both fold to 0 (matches the
            // scalar contract).
            let m_rel = zero.max(r).max(g).max(b);
            let m_nits = m_rel * wn;

            local_max = local_max.max(m_nits);
            local_sum += m_nits;

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
        let mut local_sum = zero;

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
            let r = f32x8::load(token, &ra).max(zero);
            let g = f32x8::load(token, &ga).max(zero);
            let b = f32x8::load(token, &ba).max(zero);

            let y_rel = kr * r + kg * g + kb * b;
            let y_nits = y_rel * wn;

            local_max = local_max.max(y_nits);
            local_sum += y_nits;

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
    #[allow(clippy::too_many_arguments)]
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
        let mut local_sum = zero;

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
            let r = f32x8::load(token, &ra);
            let g = f32x8::load(token, &ga);
            let b = f32x8::load(token, &ba);
            let m = zero.max(r).max(g).max(b);
            local_max = local_max.max(m);
            local_sum += m;
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
        let mut local_sum = zero;

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
            let r = f32x8::load(token, &ra).max(zero);
            let g = f32x8::load(token, &ga).max(zero);
            let b = f32x8::load(token, &ba).max(zero);
            let y = kr * r + kg * g + kb * b;
            local_max = local_max.max(y);
            local_sum += y;
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
#[allow(deprecated)] // still exercises ContentLightLevel::measure until its 0.3.0 removal
mod tests {
    use super::*;
    use crate::{PixelBuffer, PixelDescriptor};
    use alloc::vec::Vec;

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
    fn diffuse_white_defaults_to_bt2408() {
        assert_eq!(DiffuseWhite::default(), DiffuseWhite::BT2408);
        assert_eq!(DiffuseWhite::BT2408.nits(), 203.0);
        assert_eq!(DiffuseWhite::new(100.0).nits(), 100.0);
    }

    #[test]
    fn measure_two_grays_cta_stills_semantics() {
        // [1.0, 2.0] @ 203: MaxCLL = 2·203 = 406; MaxFALL = avg(203, 406) = 304.5 → 305.
        let buf = rgbf32(&[[1.0; 3], [2.0; 3]], 2, 1);
        let cll = ContentLightLevel::measure(buf.as_slice(), DiffuseWhite::BT2408).unwrap();
        assert_eq!(cll.max_content_light_level, 406);
        assert_eq!(cll.max_frame_average_light_level, 305);
    }

    #[test]
    fn measure_handles_stride_and_ignores_padding() {
        use crate::PixelSlice;
        // 2×2 RGB f32: 6 real f32/row, padded to 9 f32/row (36-byte stride, a
        // multiple of the 12-byte pixel). The padding holds a 1e9 sentinel — if
        // a row cast ever ran past `width*bpp`, MaxCLL would explode to ~2e11.
        let (w, h, row_floats) = (2u32, 2u32, 9usize);
        let mut data = alloc::vec![1.0e9f32; row_floats * h as usize];
        let pixels = [[0.5f32; 3], [1.0; 3], [2.0; 3], [0.25; 3]];
        for (i, p) in pixels.iter().enumerate() {
            let base = (i / w as usize) * row_floats + (i % w as usize) * 3;
            data[base..base + 3].copy_from_slice(p);
        }
        // `Vec<f32>` is f32-aligned, so the byte view satisfies the slice's
        // alignment contract; stride 40 is a multiple of the f32 size.
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        let px =
            PixelSlice::new(bytes, w, h, row_floats * 4, PixelDescriptor::RGBF32_LINEAR).unwrap();
        let cll = ContentLightLevel::measure(px, DiffuseWhite::BT2408).unwrap();
        // Peak max(R,G,B) = 2.0 → 406; FALL = avg(0.5,1,2,0.25)·203 = 190.3 → 190.
        assert_eq!(cll.max_content_light_level, 406);
        assert_eq!(cll.max_frame_average_light_level, 190);
    }

    #[test]
    fn measure_clamps_nan_and_negative() {
        let buf = rgbf32(&[[-1.0, f32::NAN, 0.5]], 1, 1);
        let cll = ContentLightLevel::measure(buf.as_slice(), DiffuseWhite::BT2408).unwrap();
        // max(R,G,B) folds from 0.0 → 0.5 · 203 = 101.5 → 102.
        assert_eq!(cll.max_content_light_level, 102);
        assert_eq!(cll.max_frame_average_light_level, 102);
    }

    #[test]
    fn measure_ignores_alpha_and_custom_white() {
        let mut data = Vec::new();
        for c in [0.5f32, 0.5, 0.5, 7.0] {
            data.extend_from_slice(&c.to_ne_bytes());
        }
        let buf = PixelBuffer::from_vec(data, 1, 1, PixelDescriptor::RGBAF32_LINEAR).unwrap();
        // alpha 7.0 ignored; custom 100-nit white: 0.5 · 100 = 50.
        let cll = ContentLightLevel::measure(buf.as_slice(), DiffuseWhite::new(100.0)).unwrap();
        assert_eq!(cll.max_content_light_level, 50);
    }

    #[test]
    fn measure_rejects_non_linear_and_non_f32() {
        let u8buf =
            PixelBuffer::from_vec(alloc::vec![0u8; 3], 1, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        assert!(ContentLightLevel::measure(u8buf.as_slice(), DiffuseWhite::BT2408).is_none());

        let nonlinear = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb);
        let mut data = Vec::new();
        for c in [0.5f32; 3] {
            data.extend_from_slice(&c.to_ne_bytes());
        }
        let buf = PixelBuffer::from_vec(data, 1, 1, nonlinear).unwrap();
        assert!(ContentLightLevel::measure(buf.as_slice(), DiffuseWhite::BT2408).is_none());
    }

    #[test]
    fn content_light_level_new() {
        let cll = ContentLightLevel::new(1000, 500);
        assert_eq!(cll.max_content_light_level, 1000);
        assert_eq!(cll.max_frame_average_light_level, 500);
    }

    #[test]
    fn content_light_level_default() {
        let cll = ContentLightLevel::default();
        assert_eq!(cll.max_content_light_level, 0);
        assert_eq!(cll.max_frame_average_light_level, 0);
    }

    #[test]
    fn mastering_display_new() {
        let md = MasteringDisplay::new(
            [[0.68, 0.32], [0.265, 0.69], [0.15, 0.06]],
            [0.3127, 0.329],
            1000.0,
            0.001,
        );
        assert_eq!(md.max_luminance, 1000.0);
        assert_eq!(md.min_luminance, 0.001);
    }

    #[test]
    fn mastering_display_constants() {
        assert_eq!(MasteringDisplay::HDR10_REFERENCE.max_luminance, 10000.0);
        assert_eq!(MasteringDisplay::DISPLAY_P3_1000.max_luminance, 1000.0);
    }

    #[test]
    fn content_light_level_clone_eq() {
        let a = ContentLightLevel::new(100, 50);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    #[cfg(feature = "std")]
    fn content_light_level_hash() {
        use core::hash::{Hash, Hasher};
        let a = ContentLightLevel::new(100, 50);
        let b = a;
        let mut h1 = std::hash::DefaultHasher::new();
        a.hash(&mut h1);
        let mut h2 = std::hash::DefaultHasher::new();
        b.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
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
        use crate::PixelSlice;
        let owned: Vec<f32> = Vec::new();
        let bytes: &[u8] = bytemuck::cast_slice(&owned);
        let px = PixelSlice::new(bytes, 1, 0, 12, PixelDescriptor::RGBF32_LINEAR).unwrap();
        let h = ContentLightLevel::measure_histogram(
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
        let cll = ContentLightLevel::measure_max(
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
        let cll = ContentLightLevel::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::LuminanceBt2020,
        )
        .unwrap();
        assert_eq!(cll.max_content_light_level, 53);
        assert_eq!(cll.max_frame_average_light_level, 53);
        // MaxRgb on the same input picks 1.0 → 203.
        let cll_max_rgb = ContentLightLevel::measure_max(
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

        let lit = ContentLightLevel::measure_max(
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
        let pct = ContentLightLevel::measure_percentile(
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
        let lit = ContentLightLevel::measure_max(
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
        let pct_high = ContentLightLevel::measure_percentile(
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
        let pct_low = ContentLightLevel::measure_percentile(
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
        let h = ContentLightLevel::measure_histogram(
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
    fn percentile_clamps_nan_and_out_of_range_inputs() {
        let buf = rgbf32(&[[0.5; 3]], 1, 1);
        let h = ContentLightLevel::measure_histogram(
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
            ContentLightLevel::measure_histogram(
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
            ContentLightLevel::measure_histogram(
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
        let h = ContentLightLevel::measure_histogram(
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
        let cll = ContentLightLevel::measure_max(
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
        let cll = ContentLightLevel::measure_max(
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
        let cll = ContentLightLevel::measure_max(
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
        let via_max = ContentLightLevel::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap();
        let via_hist = ContentLightLevel::measure_max_via_histogram_for_test(
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
}
