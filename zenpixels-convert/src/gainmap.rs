//! ISO 21496-1 / UltraHDR dual-layer gain-map reconstruction, resampling, orientation,
//! and extraction.
//!
//! Subsumes pure ISO 21496-1 pixel math in `no_std + alloc`, `#![forbid(unsafe_code)]`
//! with **0-bit `Accelerators::ALL.with_gainmap() == Accelerators::SCALAR` parity** and
//! O(width) streaming memory:
//!
//! - **Direction-complete ISO 21496-1:** exact `i128` rational headroom ordering supports
//!   both **Forward** (SDR base → HDR alternate, `H_alt > H_base`) and **Reverse** (HDR base →
//!   SDR alternate, `H_base > H_alt`) with signed weight, exact [`GainMapTarget::AlternateRendition`]
//!   / [`GainMapTarget::BaseRendition`] endpoints, and zero-weight identity bypass.
//! - **Multiple orientation models:**
//!   - O(1) EXIF passthrough metadata rotate via [`DualLayerHdr::rotate_metadata`] /
//!     [`DualLayerHdrView::rotate_metadata`] (zero pixel rebake, both layers kept aligned).
//!   - Whole-image apply policy [`GainMapOrientation`] (`Bake` to upright display pixels,
//!     `Passthrough` returning `(PixelBuf, Orientation)` with zero transpose via
//!     [`apply_gain_map_oriented`], or `RequireIdentity`).
//!   - Coordinated pixel-level dual-layer baking & rotation via [`bake_dual_layer_orientation`]
//!     and [`rotate_dual_layer`] keeping `base` and `gain_map` strictly registered.
//! - **Separable resampling (`zenresize` center-sited math):** [`GainMapResample::Bilinear`],
//!   [`GainMapResample::Bicubic`], [`GainMapResample::Mitchell`], and [`GainMapResample::Nearest`],
//!   with cached horizontally-upsampled gain rows and 8-element padded scratch rows so SIMD
//!   kernels execute **pure SIMD-only loops** with zero scalar tails.
//! - **Streaming row workers & extraction:** [`GainMapApplier`] for pull/streaming decode
//!   with fallible row providers, [`GainMapRowExtractor`] for fused row-level encode, and
//!   [`extract_gain_map_only`] supporting `Box` / `Bilinear` / `Point` downsampling,
//!   2-pass `content_fit`, 1-ch/3-ch, U8/U16/10-bit/12-bit maps, and [`enough::Stop`].

use alloc::vec::Vec;
use core::num::NonZeroU32;
use enough::{Stop, Unstoppable};
use whereat::At;
use zenpixels::{
    AlphaMode, BitPacking, ChannelOrder, ColorSpace, DiffuseWhite, DualLayerHdr, DualLayerHdrView,
    GainMapChannels, GainMapColorSpace, GainMapComponent, GainMapDirection, GainMapMetadata,
    HdrMetadata, ImageSpec, Orientation, PixelBuf, PixelSink, PixelSource, PixelView, Primaries,
    SampleFacts, SampleLayout, SampleType, SignalRange, SignedFraction, Transfer, UnsignedFraction,
};

use crate::accel::{self, Accelerators};
use crate::error::{ConvertError, err_at};
use crate::gainmap_simd::{
    MAX_LOG2_GAIN, blend_rows_2_scalar, blend_rows_4_scalar, combine_linear_f32_scalar,
    decode_gain_1ch_scalar, exp2_gain_scalar,
};
use crate::orient;
use crate::plan::RowPlan;
use crate::policy::{AlphaPolicy, ConvertPolicy};

/// Minimum log₂ span (`1/16` stop) enforced by content-fitted extraction grids so uniform
/// images never degenerate to a zero-width quantization range.
pub const CONTENT_FIT_MIN_SPAN_STOPS: f32 = 1.0 / 16.0;

/// Maximum resampling kernel taps allowed in [`GainMapApplier`] (covers unlimited upscaling
/// and up to 32× Bilinear / 16× Bicubic gain-map downscaling while preventing cache DoS).
const MAX_APPLY_RESAMPLE_TAPS: usize = 64;

/// Maximum resampling kernel taps allowed in [`extract_gain_map_only`] (`GainMapDownsample::Bilinear`,
/// covering every valid `scale_factor: u8` `1..=255`).
const MAX_EXTRACT_RESAMPLE_TAPS: usize = 512;

const INV_255: f32 = 1.0 / 255.0;

// ---------------------------------------------------------------------------
// Public math, target, orientation & boost helpers
// ---------------------------------------------------------------------------

/// Target display headroom or explicit rendition endpoint for ISO 21496-1 reconstruction.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum GainMapTarget {
    /// Target display HDR headroom in log₂ stops (`0.0` = SDR, `2.0` = 4× HDR, `2.32` ≈ 5×).
    HeadroomStops(f32),
    /// Exact full alternate rendition (`signed_weight = +1.0` for Forward, `-1.0` for Reverse),
    /// immune to floating-point rounding of `metadata.alternate_hdr_headroom`.
    AlternateRendition,
    /// Exact base rendition (`signed_weight = 0.0` identity bypass, skipping gain-map sampling
    /// and additive offsets completely).
    BaseRendition,
}

/// Policy controlling how pending EXIF [`Orientation`] on a [`DualLayerHdr`] /
/// [`DualLayerHdrView`] is handled during whole-image gain-map reconstruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GainMapOrientation {
    /// Reconstruct in stored-frame coordinates (preserving exact subsampled gain-map alignment),
    /// then physically apply `dual.orientation()` so the returned pixels are display-oriented
    /// (`Orientation::Identity`). Default.
    #[default]
    Bake,
    /// Reconstruct in stored-frame coordinates with **zero pixel transpose cost**, returning
    /// the pending `dual.orientation()` alongside the buffer via [`apply_gain_map_oriented`]
    /// for EXIF passthrough or GPU UV orientation.
    Passthrough,
    /// Refuse inputs whose pending orientation is not [`Orientation::Identity`]
    /// (`ConvertError::UnsupportedRoute`).
    RequireIdentity,
}

/// Convert a linear display boost ratio (e.g. `1.0` = SDR, `4.0` = 4× HDR) into
/// ISO 21496-1 log₂ headroom stops (`0.0` for `<= 1.0` or non-finite).
#[inline]
#[must_use]
pub fn display_boost_to_headroom_stops(display_boost: f32) -> f32 {
    if !display_boost.is_finite() || display_boost <= 1.0 {
        0.0
    } else {
        libm::log2f(display_boost)
    }
}

/// Full-reconstruction display headroom in log₂ stops (`metadata.alternate_hdr_headroom`).
///
/// Note: for guaranteed exact full alternate reconstruction (`|weight| == 1.0`) even when
/// `alternate_hdr_headroom` is not exactly representable in `f32`, prefer
/// [`GainMapApplyConfig::alternate_rendition`] / [`GainMapTarget::AlternateRendition`].
#[inline]
#[must_use]
pub fn full_reconstruction_headroom_stops(metadata: &GainMapMetadata) -> f32 {
    metadata.alternate_hdr_headroom.to_f32()
}

/// Compute the ISO 21496-1 signed application weight for `display_headroom_stops` using
/// exact `i128` rational cross-products (never divides by zero or yields `NaN` on close
/// valid rationals):
///
/// - Returns `None` if `display_headroom_stops` is non-finite or `H_base == H_alt`.
/// - Returns `0.0` at or beyond the base endpoint (`amount == 0.0`), signaling exact
///   base-rendition identity bypass.
/// - Returns `sign(H_alt - H_base) * clamp((H - H_base) / (H_alt - H_base), 0, 1)`
///   (`+amount` for Forward SDR→HDR, `-amount` for Reverse HDR→SDR).
#[inline]
#[must_use]
pub fn signed_weight(metadata: &GainMapMetadata, display_headroom_stops: f32) -> Option<f32> {
    if !display_headroom_stops.is_finite() {
        return None;
    }
    let direction = metadata.direction()?;
    let hb_n = i128::from(metadata.base_hdr_headroom.num);
    let hb_d = i128::from(metadata.base_hdr_headroom.den.get());
    let ha_n = i128::from(metadata.alternate_hdr_headroom.num);
    let ha_d = i128::from(metadata.alternate_hdr_headroom.den.get());

    // Exact non-zero numerator of (H_alt - H_base) * (hb_d * ha_d):
    //   (H - hb_n/hb_d) / (ha_n/ha_d - hb_n/hb_d)
    // = ((H * hb_d - hb_n) * ha_d) / (ha_n * hb_d - hb_n * ha_d)
    let span_num = ha_n * hb_d - hb_n * ha_d;
    if span_num == 0 {
        return None;
    }
    let num = (f64::from(display_headroom_stops) * (hb_d as f64) - (hb_n as f64)) * (ha_d as f64);
    let amount = (num / (span_num as f64)).clamp(0.0, 1.0) as f32;
    if amount == 0.0 {
        return Some(0.0);
    }
    Some(match direction {
        GainMapDirection::Forward => amount,
        GainMapDirection::Reverse => -amount,
    })
}

/// Compute the signed application weight for any [`GainMapTarget`].
#[inline]
#[must_use]
pub fn signed_weight_for_target(metadata: &GainMapMetadata, target: GainMapTarget) -> Option<f32> {
    match target {
        GainMapTarget::HeadroomStops(stops) => signed_weight(metadata, stops),
        GainMapTarget::BaseRendition => {
            metadata.direction()?;
            Some(0.0)
        }
        GainMapTarget::AlternateRendition => Some(match metadata.direction()? {
            GainMapDirection::Forward => 1.0,
            GainMapDirection::Reverse => -1.0,
        }),
    }
}

// ---------------------------------------------------------------------------
// Exact rational encoding helpers (dynamic denominator, zero silent clamp)
// ---------------------------------------------------------------------------

#[inline(never)]
fn signed_fraction_from_f32(v: f32) -> SignedFraction {
    let v = if v.is_finite() { v } else { 0.0 };
    if v == 0.0 {
        return SignedFraction::ZERO;
    }
    let vf = f64::from(v);
    for den in [1u32, 2, 4, 8, 16, 32, 64, 128, 256, 1024] {
        let scaled = vf * f64::from(den);
        let rounded = libm::round(scaled);
        if (scaled - rounded).abs() <= 1e-7
            && rounded >= f64::from(i32::MIN)
            && rounded <= f64::from(i32::MAX)
        {
            return SignedFraction::new(rounded as i32, NonZeroU32::new(den).unwrap());
        }
    }
    for den in [1_000_000u32, 100_000, 10_000, 1_000, 100, 10, 1] {
        let max_abs = (i32::MAX as f64) / f64::from(den);
        if vf.abs() <= max_abs || den == 1 {
            let clamped = vf.clamp(-max_abs, max_abs);
            return SignedFraction::new(
                libm::round(clamped * f64::from(den)) as i32,
                NonZeroU32::new(den).unwrap(),
            );
        }
    }
    SignedFraction::ZERO
}

#[inline(never)]
fn unsigned_fraction_from_f32(v: f32) -> UnsignedFraction {
    let v = if v.is_finite() && v > 0.0 { v } else { 0.0 };
    if v == 0.0 {
        return UnsignedFraction::ZERO;
    }
    let vf = f64::from(v);
    for den in [1u32, 2, 4, 8, 16, 32, 64, 128, 256, 1024] {
        let scaled = vf * f64::from(den);
        let rounded = libm::round(scaled);
        if (scaled - rounded).abs() <= 1e-7 && rounded >= 0.0 && rounded <= f64::from(u32::MAX) {
            return UnsignedFraction::new(rounded as u32, NonZeroU32::new(den).unwrap());
        }
    }
    for den in [1_000_000u32, 100_000, 10_000, 1_000, 100, 10, 1] {
        let max_val = (u32::MAX as f64) / f64::from(den);
        if vf <= max_val || den == 1 {
            let clamped = vf.clamp(0.0, max_val);
            return UnsignedFraction::new(
                libm::round(clamped * f64::from(den)) as u32,
                NonZeroU32::new(den).unwrap(),
            );
        }
    }
    UnsignedFraction::ZERO
}

#[inline(never)]
fn alloc_f32(n: usize) -> Result<Vec<f32>, At<ConvertError>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| err_at(ConvertError::AllocationFailed))?;
    v.resize(n, 0.0f32);
    Ok(v)
}

#[inline]
fn color_alpha_mode(color: &ColorSpace) -> AlphaMode {
    match *color {
        ColorSpace::Rgb { alpha, .. }
        | ColorSpace::Gray { alpha, .. }
        | ColorSpace::Icc { alpha, .. } => alpha,
        _ => AlphaMode::None,
    }
}

#[inline]
fn is_valid_gain_map_order(order: ChannelOrder) -> bool {
    matches!(
        order,
        ChannelOrder::Gray
            | ChannelOrder::Rgb
            | ChannelOrder::Generic(1)
            | ChannelOrder::Generic(3)
    )
}

// ---------------------------------------------------------------------------
// Separable 1D resampling weight table (zenresize pixel-center siting)
// ---------------------------------------------------------------------------

/// Resampling filter used when the gain map resolution differs from the base image.
///
/// Resampling is performed on **encoded gain-map samples `[0, 1]`** before applying
/// `gamma` and `exp2`, matching ISO 21496-1 / Android Ultra HDR / `libavif`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GainMapResample {
    /// Nearest-neighbor (1-tap box). Fastest; useful for preview/diagnostics or 1:1 maps.
    Nearest,
    /// Separable bilinear / triangle filter (2-tap upsample). ISO 21496-1 & Android Ultra HDR
    /// baseline; zero overshoot.
    #[default]
    Bilinear,
    /// Separable Catmull-Rom bicubic spline (`B=0, C=0.5`, 4-tap upsample, clamped to `[0, 1]`).
    /// Sharper transition edges on subsampled gain maps.
    Bicubic,
    /// Separable Mitchell-Netravali cubic (`B=1/3, C=1/3`, 4-tap upsample, clamped to `[0, 1]`).
    /// Balanced sharpness with minimal ringing (~1.3% negative lobe vs ~6.5% for Catmull-Rom).
    Mitchell,
}

#[inline]
fn eval_cubic(x: f64, b: f64, c: f64) -> f64 {
    let ax = if x < 0.0 { -x } else { x };
    if ax < 1.0 {
        let p3 = (12.0 - 9.0 * b - 6.0 * c) / 6.0;
        let p2 = (-18.0 + 12.0 * b + 6.0 * c) / 6.0;
        let p0 = (6.0 - 2.0 * b) / 6.0;
        (p3 * ax + p2) * ax * ax + p0
    } else if ax < 2.0 {
        let q3 = (-b - 6.0 * c) / 6.0;
        let q2 = (6.0 * b + 30.0 * c) / 6.0;
        let q1 = (-12.0 * b - 48.0 * c) / 6.0;
        let q0 = (8.0 * b + 24.0 * c) / 6.0;
        ((q3 * ax + q2) * ax + q1) * ax + q0
    } else {
        0.0
    }
}

#[inline]
fn eval_resample_kernel(mode: GainMapResample, x: f64) -> f64 {
    match mode {
        GainMapResample::Nearest => {
            let ax = if x < 0.0 { -x } else { x };
            if ax <= 0.5 { 1.0 } else { 0.0 }
        }
        GainMapResample::Bilinear => {
            let ax = if x < 0.0 { -x } else { x };
            if ax < 1.0 { 1.0 - ax } else { 0.0 }
        }
        GainMapResample::Bicubic => eval_cubic(x, 0.0, 0.5),
        GainMapResample::Mitchell => eval_cubic(x, 1.0 / 3.0, 1.0 / 3.0),
    }
}

/// Precomputed 1D separable resampling weights using `zenresize` pixel-center siting:
/// `center = (out + 0.5) / scale - 0.5`, edge-clamped before kernel evaluation and
/// normalized to exact unit sum. Single flat `weights_flat` allocation (zero `Vec<Vec<_>>`).
#[derive(Clone, Debug)]
struct AxisWeights {
    left: Vec<u32>,
    tap_counts: Vec<u16>,
    weights_flat: Vec<f32>,
    weights_stride: usize,
    max_taps: usize,
}

/// Compute exact trimmed nonzero resampling taps for a single output coordinate `o` on stack
/// (`0` heap allocations). Only rejects with `UnsupportedRoute` if the **actual** trimmed nonzero
/// tap count exceeds `max_allowed_taps` (fixes B30: 128→4 bilinear has 64 actual taps and succeeds!).
#[inline]
fn compute_axis_taps_stack(
    in_size: u32,
    out_size: u32,
    mode: GainMapResample,
    o: u32,
    max_allowed_taps: usize,
    out_weights: &mut [f32; MAX_APPLY_RESAMPLE_TAPS],
) -> Result<(u32, usize), ConvertError> {
    const WEIGHT_EPS: f64 = 2e-8;
    if in_size == 0 || out_size == 0 {
        return Err(ConvertError::DimensionMismatch);
    }
    let max_in = i64::from(in_size - 1);
    if mode == GainMapResample::Nearest {
        let ratio = f64::from(in_size) / f64::from(out_size);
        let idx = (libm::floor((f64::from(o) + 0.5) * ratio) as i64).clamp(0, max_in) as u32;
        out_weights[0] = 1.0;
        return Ok((idx, 1));
    }

    let scale = f64::from(out_size) / f64::from(in_size);
    let downscale = scale.min(1.0);
    let base_win = match mode {
        GainMapResample::Bilinear => 1.0,
        GainMapResample::Bicubic | GainMapResample::Mitchell => 2.0,
        GainMapResample::Nearest => 0.5,
    };
    let eff_window = base_win / downscale;
    // Fast conservative reject only when even the strict interior support exceeds budget.
    if (2.0 * eff_window) > (max_allowed_taps as f64) + 2.0
        && u64::from(in_size) > max_allowed_taps as u64
    {
        return Err(ConvertError::UnsupportedRoute);
    }

    let center = (f64::from(o) + 0.5) / scale - 0.5;
    let lp = (libm::ceil(center - eff_window - 1e-4) as i64).max(0);
    let rp = (libm::floor(center + eff_window + 1e-4) as i64).min(max_in);

    // Collect nonzero weights on stack, trimming leading/trailing exact zeros before checking count.
    let mut tmp = [0.0f64; MAX_APPLY_RESAMPLE_TAPS + 6];
    let mut len = 0usize;
    let mut total = 0.0f64;
    let mut trimmed_left = lp.min(max_in) as u32;
    let mut seen_nonzero = false;

    if lp <= rp {
        let mut inp = lp;
        while inp <= rp {
            let x = downscale * ((inp as f64) - center);
            let mut w = eval_resample_kernel(mode, x);
            if w.abs() <= WEIGHT_EPS {
                w = 0.0;
            }
            if !seen_nonzero {
                if w == 0.0 {
                    trimmed_left += 1;
                    inp += 1;
                    continue;
                }
                seen_nonzero = true;
            }
            if len >= tmp.len() {
                return Err(ConvertError::UnsupportedRoute);
            }
            tmp[len] = w;
            len += 1;
            total += w;
            inp += 1;
        }
        while len > 0 && tmp[len - 1] == 0.0 {
            len -= 1;
        }
        if total != 0.0 && len > 0 {
            let inv = 1.0 / total;
            for w in &mut tmp[..len] {
                *w *= inv;
            }
        } else {
            len = 0;
        }
    }

    if len == 0 {
        let nearest = (libm::round(center) as i64).clamp(0, max_in) as u32;
        out_weights[0] = 1.0;
        return Ok((nearest, 1));
    }
    if len > max_allowed_taps {
        return Err(ConvertError::UnsupportedRoute);
    }
    let mut sum = 0.0f32;
    for j in 0..len {
        let wf = tmp[j] as f32;
        out_weights[j] = wf;
        sum += wf;
    }
    if sum != 0.0 {
        let inv = 1.0f32 / sum;
        for v in &mut out_weights[..len] {
            *v *= inv;
        }
    }
    Ok((trimmed_left, len))
}

/// Probe exact maximum vertical tap count across `0..out_size` in **O(1) stack memory**
/// (zero heap allocation, independent of image height — B30).
fn probe_max_axis_taps(
    in_size: u32,
    out_size: u32,
    mode: GainMapResample,
    max_allowed_taps: usize,
) -> Result<usize, ConvertError> {
    if in_size == 0 || out_size == 0 {
        return Err(ConvertError::DimensionMismatch);
    }
    if mode == GainMapResample::Nearest {
        return Ok(1);
    }
    let mut ws = [0.0f32; MAX_APPLY_RESAMPLE_TAPS];
    let mut max_taps = 1usize;
    for o in 0..out_size {
        let (_, count) =
            compute_axis_taps_stack(in_size, out_size, mode, o, max_allowed_taps, &mut ws)?;
        max_taps = max_taps.max(count);
    }
    Ok(max_taps)
}

impl AxisWeights {
    #[inline(never)]
    fn try_new(
        in_size: u32,
        out_size: u32,
        mode: GainMapResample,
        max_allowed_taps: usize,
    ) -> Result<Self, ConvertError> {
        if in_size == 0 || out_size == 0 {
            return Err(ConvertError::DimensionMismatch);
        }
        let out_n = out_size as usize;

        let mut left = Vec::new();
        left.try_reserve_exact(out_n)
            .map_err(|_| ConvertError::AllocationFailed)?;

        if mode == GainMapResample::Nearest {
            let ratio = f64::from(in_size) / f64::from(out_size);
            let max_in = i64::from(in_size - 1);
            for o in 0..out_size {
                let idx =
                    (libm::floor((f64::from(o) + 0.5) * ratio) as i64).clamp(0, max_in) as u32;
                left.push(idx);
            }
            return Ok(Self {
                left,
                tap_counts: Vec::new(),
                weights_flat: Vec::new(),
                weights_stride: 1,
                max_taps: 1,
            });
        }

        // First pass (zero heap alloc): measure exact max_taps and reject only if actual > max_allowed_taps!
        let max_taps = probe_max_axis_taps(in_size, out_size, mode, max_allowed_taps)?;
        let weights_stride = max_taps.max(1);

        let mut tap_counts = Vec::new();
        tap_counts
            .try_reserve_exact(out_n)
            .map_err(|_| ConvertError::AllocationFailed)?;
        let flat_len = out_n
            .checked_mul(weights_stride)
            .ok_or(ConvertError::AllocationFailed)?;
        let mut weights_flat = Vec::new();
        weights_flat
            .try_reserve_exact(flat_len)
            .map_err(|_| ConvertError::AllocationFailed)?;
        weights_flat.resize(flat_len, 0.0f32);

        let mut ws = [0.0f32; MAX_APPLY_RESAMPLE_TAPS];
        for o in 0..out_size {
            let (l, count) =
                compute_axis_taps_stack(in_size, out_size, mode, o, max_allowed_taps, &mut ws)?;
            left.push(l);
            tap_counts.push(count as u16);
            let off = (o as usize) * weights_stride;
            weights_flat[off..off + count].copy_from_slice(&ws[..count]);
        }

        Ok(Self {
            left,
            tap_counts,
            weights_flat,
            weights_stride,
            max_taps,
        })
    }

    #[inline(always)]
    fn taps(&self, out_idx: usize) -> (u32, &[f32]) {
        if self.tap_counts.is_empty() {
            static ONE: [f32; 1] = [1.0];
            return (self.left[out_idx], &ONE);
        }
        let count = usize::from(self.tap_counts[out_idx]);
        let off = out_idx * self.weights_stride;
        (self.left[out_idx], &self.weights_flat[off..off + count])
    }
}

// ---------------------------------------------------------------------------
// Apply configuration & streaming GainMapApplier
// ---------------------------------------------------------------------------

/// Configuration for ISO 21496-1 gain-map reconstruction via [`GainMapApplier`],
/// [`apply_gain_map_oriented`], or [`apply_gain_map_with`].
#[derive(Clone, Debug)]
pub struct GainMapApplyConfig {
    /// Target display HDR headroom in log₂ stops or explicit rendition endpoint.
    pub target: GainMapTarget,
    /// Resampling kernel used when gain-map dimensions differ from base dimensions.
    pub resample: GainMapResample,
    /// Pending EXIF orientation policy (`Bake`, `Passthrough`, or `RequireIdentity`).
    pub orientation: GainMapOrientation,
    /// Conversion policy controlling SIMD [`Accelerators`], `sdr_white`, and CMS backends.
    ///
    /// Defaults to [`ConvertPolicy::working_linear`] (`accelerators: None`, respecting the
    /// process-wide `accel::enable*` registry or scalar fallback for dead-stripping); override
    /// via [`with_policy`](Self::with_policy) (e.g. `Accelerators::ALL.with_gainmap()` or
    /// `Accelerators::WEB_SDR.with_gainmap()`).
    pub policy: ConvertPolicy,
    /// Optional target output [`ImageSpec`].
    ///
    /// - `None`: outputs relative-linear `RGBF32` (or `RGBAF32` when base has alpha) in the
    ///   resolved gain-map application primaries.
    /// - `Some(spec)`: fuses conversion directly to `spec` (e.g. `RGBAF16`, `RGB16` PQ,
    ///   `RGB16` HLG, `RGBA8` sRGB, `DisplayP3`) in O(width) row memory with zero full-frame
    ///   intermediate buffer.
    pub dst_spec: Option<ImageSpec>,
}

impl GainMapApplyConfig {
    /// Create a default configuration for `display_headroom_stops` (log₂ stops).
    #[inline(always)]
    #[must_use]
    pub fn new(display_headroom_stops: f32) -> Self {
        Self::with_target_default(GainMapTarget::HeadroomStops(display_headroom_stops))
    }

    /// Create a configuration from a linear display boost ratio (e.g. `4.0` → `2.0` stops).
    #[inline(always)]
    #[must_use]
    pub fn from_display_boost(display_boost: f32) -> Self {
        Self::new(display_boost_to_headroom_stops(display_boost))
    }

    /// Create a configuration targeting exact full alternate reconstruction (`|weight| == 1.0`).
    #[inline(always)]
    #[must_use]
    pub fn alternate_rendition() -> Self {
        Self::with_target_default(GainMapTarget::AlternateRendition)
    }

    /// Create a configuration targeting exact base rendition bypass (`weight == 0.0`).
    #[inline(always)]
    #[must_use]
    pub fn base_rendition() -> Self {
        Self::with_target_default(GainMapTarget::BaseRendition)
    }

    #[inline(always)]
    fn with_target_default(target: GainMapTarget) -> Self {
        Self {
            target,
            resample: GainMapResample::Bilinear,
            orientation: GainMapOrientation::Bake,
            policy: ConvertPolicy::working_linear(),
            dst_spec: None,
        }
    }

    /// Replace the target headroom / rendition endpoint.
    #[inline(always)]
    #[must_use]
    pub fn with_target(mut self, target: GainMapTarget) -> Self {
        self.target = target;
        self
    }

    /// Select the gain-map resampling filter (`Bilinear`, `Bicubic`, `Mitchell`, `Nearest`).
    #[inline(always)]
    #[must_use]
    pub fn with_resample(mut self, resample: GainMapResample) -> Self {
        self.resample = resample;
        self
    }

    /// Select how pending EXIF orientation on [`DualLayerHdr`] / [`DualLayerHdrView`] is handled.
    #[inline(always)]
    #[must_use]
    pub fn with_orientation(mut self, orientation: GainMapOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Set the conversion policy (accelerators, `sdr_white`, CMS backend, HLG rendering).
    #[inline(always)]
    #[must_use]
    pub fn with_policy(mut self, policy: ConvertPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Fuse row conversion straight into `dst_spec` (e.g. `RGBAF16`, `RGB16` PQ, `RGBA8` sRGB).
    #[inline(always)]
    #[must_use]
    pub fn with_dst_spec(mut self, dst_spec: ImageSpec) -> Self {
        self.dst_spec = Some(dst_spec);
        self
    }
}

/// Reusable, O(width)-memory, row-streaming ISO 21496-1 gain-map reconstructor.
///
/// Operates in stored-frame coordinates (`base_w × base_h`) and exposes [`orientation`](Self::orientation)
/// so streaming callers can pass pending EXIF orientation through or bake it after decode.
/// All internal scratch rows are padded to a multiple of 8 (`w_pad`), allowing SIMD vertical
/// blend, `exp2` gain decode, and linear RGB(A) combine kernels to run **pure SIMD-only loops**.
#[derive(Clone, Debug)]
pub struct GainMapApplier {
    base_spec: ImageSpec,
    base_w: u32,
    base_h: u32,
    w_pad: usize,
    base_row_bytes: usize,
    gm_w: u32,
    gm_h: u32,
    gm_spec: ImageSpec,
    gm_row_bytes: usize,
    gm_ch: usize,
    gm_max: [f32; 3],
    has_alpha: bool,
    dst_spec: ImageSpec,
    dst_row_bytes: usize,
    orientation: Orientation,
    accel: Accelerators,

    signed_weight: f32,
    is_zero_weight: bool,
    same_size: bool,
    resample: GainMapResample,
    shared_meta: bool,

    base_w_g: [f32; 3],
    slope_w_g: [f32; 3],
    inv_gamma: [f32; 3],
    k_base: [f32; 3],
    k_alt: [f32; 3],

    /// Exact 256-entry gain LUT used when gain map is full-packing U8 and 1:1 or `Nearest`.
    u8_gain_lut: Option<[[f32; 256]; 3]>,

    base_to_app: RowPlan,
    app_to_dst: Option<RowPlan>,
    zero_weight_direct: Option<RowPlan>,

    h_weights: Option<AxisWeights>,
    /// Cached gain-map row index per slot (`u32::MAX` = empty/invalid).
    cache_gm_y: Vec<u32>,
    /// Flat cache storage: `[slot * gm_ch * w_pad + c * w_pad .. +w_pad]`, lanes `w..w_pad` zeroed.
    cache_planes: Vec<f32>,
    cache_cursor: usize,
    v_slot_indices: Vec<usize>,

    // Flat O(w_pad) row scratch buffers (all multiples of 8 elements)
    base_lin_row: Vec<f32>,
    out_lin_row: Vec<f32>,
    gm_decode_scratch: Vec<f32>,
    /// Flat vertically-blended gain samples: `[c * w_pad .. (c + 1) * w_pad]`.
    v_blend: Vec<f32>,
    /// Padded scratch for SIMD `gamma_pow` when `inv_gamma != 1.0` (`0` bytes when `gamma == 1.0`).
    gamma_tmp: Vec<f32>,
    /// Flat linear gain multipliers: `[c * w_pad .. (c + 1) * w_pad]`.
    gains: Vec<f32>,
}

#[inline(never)]
fn build_u8_gain_lut(
    base_w_g: [f32; 3],
    slope_w_g: [f32; 3],
    inv_gamma: [f32; 3],
) -> [[f32; 256]; 3] {
    let mut lut = [[1.0f32; 256]; 3];
    for c in 0..3 {
        let unit_gamma = (inv_gamma[c] - 1.0).abs() < 1e-7;
        for (i, slot) in lut[c].iter_mut().enumerate() {
            let s = f32::from(i as u8) * INV_255;
            let g = if unit_gamma || i == 0 || i == 255 {
                s
            } else {
                libm::powf(s, inv_gamma[c])
            };
            let lg = base_w_g[c] + slope_w_g[c] * g;
            *slot = exp2_gain_scalar(lg);
        }
    }
    lut
}

#[cold]
#[inline(never)]
fn decode_gain_gamma_scalar(
    samples: &[f32],
    w: usize,
    inv_gamma: f32,
    base_w: f32,
    slope_w: f32,
    out_gains: &mut [f32],
) {
    for i in 0..w {
        let raw = samples[i];
        let s = if raw > 0.0 {
            if raw < 1.0 { raw } else { 1.0 }
        } else {
            0.0
        };
        let g = if s == 0.0 || s == 1.0 {
            s
        } else {
            libm::powf(s, inv_gamma)
        };
        out_gains[i] = exp2_gain_scalar(base_w + slope_w * g);
    }
}

#[cold]
#[inline(never)]
fn blend_rows_general(
    cache_planes: &[f32],
    slot_stride: usize,
    ch_off: usize,
    slots: &[usize],
    ws: &[f32],
    w: usize,
    dst: &mut [f32],
) {
    dst[..w].fill(0.0);
    for (t, &wt) in ws.iter().enumerate() {
        let base = slots[t] * slot_stride + ch_off;
        let sr = &cache_planes[base..base + w];
        for (o, &v) in dst[..w].iter_mut().zip(sr.iter()) {
            *o += v * wt;
        }
    }
    for o in &mut dst[..w] {
        *o = if *o > 0.0 {
            if *o < 1.0 { *o } else { 1.0 }
        } else {
            0.0
        };
    }
}

struct ApplierScratch {
    cache_gm_y: Vec<u32>,
    cache_planes: Vec<f32>,
    v_slot_indices: Vec<usize>,
    base_lin_row: Vec<f32>,
    out_lin_row: Vec<f32>,
    gm_decode_scratch: Vec<f32>,
    v_blend: Vec<f32>,
    gains: Vec<f32>,
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn alloc_applier_scratch(
    w_pad: usize,
    ch: usize,
    gm_w: usize,
    gm_ch: usize,
    gains_ch: usize,
    cache_slots: usize,
    need_base_lin: bool,
    need_out_lin: bool,
    need_gm_decode: bool,
    need_v_blend: bool,
    need_gains: bool,
) -> Result<ApplierScratch, At<ConvertError>> {
    let row_floats_pad = w_pad
        .checked_mul(ch)
        .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
    let base_lin_row = if need_base_lin {
        alloc_f32(row_floats_pad)?
    } else {
        Vec::new()
    };
    let out_lin_row = if need_out_lin {
        alloc_f32(row_floats_pad)?
    } else {
        Vec::new()
    };
    let gm_decode_scratch = if need_gm_decode {
        let len = gm_w
            .checked_mul(gm_ch)
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        alloc_f32(len)?
    } else {
        Vec::new()
    };

    let mut cache_gm_y = Vec::new();
    let mut cache_planes = Vec::new();
    let mut v_slot_indices = Vec::new();
    if cache_slots > 0 {
        let total = cache_slots
            .checked_mul(gm_ch)
            .and_then(|v| v.checked_mul(w_pad))
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        cache_gm_y
            .try_reserve_exact(cache_slots)
            .map_err(|_| err_at(ConvertError::AllocationFailed))?;
        cache_gm_y.resize(cache_slots, u32::MAX);
        cache_planes = alloc_f32(total)?;
        v_slot_indices
            .try_reserve_exact(cache_slots)
            .map_err(|_| err_at(ConvertError::AllocationFailed))?;
        v_slot_indices.resize(cache_slots, 0);
    }

    let v_blend = if need_v_blend {
        let len = gm_ch
            .checked_mul(w_pad)
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        alloc_f32(len)?
    } else {
        Vec::new()
    };
    let gains = if need_gains {
        let len = gains_ch
            .checked_mul(w_pad)
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        alloc_f32(len)?
    } else {
        Vec::new()
    };

    Ok(ApplierScratch {
        cache_gm_y,
        cache_planes,
        v_slot_indices,
        base_lin_row,
        out_lin_row,
        gm_decode_scratch,
        v_blend,
        gains,
    })
}

impl GainMapApplier {
    /// Construct a streaming applier directly from a validated [`DualLayerHdrView`].
    ///
    /// Rejects non-`Identity` orientation when `config.orientation == GainMapOrientation::RequireIdentity`.
    pub fn for_view(
        dual: &DualLayerHdrView<'_>,
        config: &GainMapApplyConfig,
    ) -> Result<Self, At<ConvertError>> {
        if config.orientation == GainMapOrientation::RequireIdentity
            && dual.orientation() != Orientation::Identity
        {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        let mut applier = Self::new(
            dual.base().spec().clone(),
            dual.base().width(),
            dual.base().rows(),
            dual.gain_map().spec().clone(),
            dual.gain_map().width(),
            dual.gain_map().rows(),
            dual.metadata(),
            dual.alternate_color_space(),
            config,
        )?;
        applier.orientation = dual.orientation();
        Ok(applier)
    }

    /// Construct a streaming applier from explicit layer specs, dimensions, and metadata.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base_spec: ImageSpec,
        base_w: u32,
        base_h: u32,
        gm_spec: ImageSpec,
        gm_w: u32,
        gm_h: u32,
        metadata: GainMapMetadata,
        alternate_color_space: Option<&ColorSpace>,
        config: &GainMapApplyConfig,
    ) -> Result<Self, At<ConvertError>> {
        // Phase 1: Pure validation & scalar math BEFORE allocating any `Vec` or `RowPlan`
        // (eliminates per-`?` drop-glue ladders for `RowPlan` & scratch vectors).
        if base_w == 0 || base_h == 0 || gm_w == 0 || gm_h == 0 {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        gm_spec
            .validate()
            .map_err(|e| err_at((*e.error()).into()))?;
        if !metadata.is_valid() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        let sw = signed_weight_for_target(&metadata, config.target)
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        validate_gain_map_spec(&gm_spec)?;

        let (_, base_white_opt) =
            resolve_linear_rgb_primaries(&base_spec.color).unwrap_or((Primaries::Bt709, None));
        let base_white = base_white_opt
            .or(config.policy.sdr_white)
            .unwrap_or(DiffuseWhite::BT2408);

        let app_primaries = if sw == 0.0 {
            resolve_linear_rgb_primaries(&base_spec.color)
                .map(|(p, _)| p)
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?
        } else {
            match metadata.application_space {
                GainMapColorSpace::Base => resolve_linear_rgb_primaries(&base_spec.color)
                    .map(|(p, _)| p)
                    .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?,
                GainMapColorSpace::Alternate => {
                    let alt = alternate_color_space
                        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
                    resolve_linear_rgb_primaries(alt)
                        .map(|(p, _)| p)
                        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?
                }
            }
        };

        let mut base_w_g = [0.0f32; 3];
        let mut slope_w_g = [0.0f32; 3];
        let mut inv_gamma = [1.0f32; 3];
        let mut k_base = [0.0f32; 3];
        let mut k_alt = [0.0f32; 3];
        for c in 0..3 {
            let comp = metadata.channel(c);
            let min_g = comp.min_gain_stops.to_f32();
            let span_g = comp.max_gain_stops.to_f32() - min_g;
            base_w_g[c] = min_g * sw;
            slope_w_g[c] = span_g * sw;
            inv_gamma[c] = 1.0 / comp.gamma.to_f32();
            k_base[c] = comp.base_offset.to_f32();
            k_alt[c] = comp.alternate_offset.to_f32();
            if !base_w_g[c].is_finite()
                || !slope_w_g[c].is_finite()
                || !inv_gamma[c].is_finite()
                || !k_base[c].is_finite()
                || !k_alt[c].is_finite()
            {
                return Err(err_at(ConvertError::UnsupportedRoute));
            }
        }

        let has_alpha = base_spec.layout.order.has_alpha_slot();
        let ch = if has_alpha { 4usize } else { 3usize };
        let w_usize = base_w as usize;
        let w_pad = w_usize
            .checked_next_multiple_of(8)
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        let base_row_bytes = w_usize
            .checked_mul(base_spec.layout.bytes_per_pixel())
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        let gm_row_bytes = (gm_w as usize)
            .checked_mul(gm_spec.layout.bytes_per_pixel())
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        let dst_bpp = config
            .dst_spec
            .as_ref()
            .map_or(ch * 4, |t| t.layout.bytes_per_pixel());
        let dst_row_bytes = w_usize
            .checked_mul(dst_bpp)
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;

        let gm_layout = gm_spec.layout;
        let gm_ch = gm_layout.channels();
        let gm_max = [
            gm_layout.packing.max_code(gm_layout.sample, 0).max(1) as f32,
            gm_layout.packing.max_code(gm_layout.sample, 1).max(1) as f32,
            gm_layout.packing.max_code(gm_layout.sample, 2).max(1) as f32,
        ];
        let shared_meta = matches!(metadata.channels, GainMapChannels::Shared(_));
        let same_size = base_w == gm_w && base_h == gm_h;
        let is_zero_weight = sw == 0.0;
        let use_u8_lut = !is_zero_weight
            && gm_layout.sample == SampleType::U8
            && gm_layout.packing.is_full()
            && (same_size || config.resample == GainMapResample::Nearest);

        // Phase 2: Build horizontal resampling weights (O(width)) and probe vertical max taps
        // in O(1) stack memory (zero O(height * taps) heap allocation — B30!).
        let (h_weights, cache_slots) = if is_zero_weight || same_size {
            (None, 0usize)
        } else {
            let mode = if use_u8_lut {
                GainMapResample::Nearest
            } else {
                config.resample
            };
            let hw = AxisWeights::try_new(gm_w, base_w, mode, MAX_APPLY_RESAMPLE_TAPS)
                .map_err(err_at)?;
            let v_max_taps =
                probe_max_axis_taps(gm_h, base_h, mode, MAX_APPLY_RESAMPLE_TAPS).map_err(err_at)?;
            let slots = if use_u8_lut { 0 } else { v_max_taps.max(1) };
            (Some(hw), slots)
        };

        // Phase 3: Build `RowPlan`s (`RowPlan::build` validates `base_spec` & `dst_spec`).
        let app_alpha = if !has_alpha {
            AlphaMode::None
        } else if color_alpha_mode(&base_spec.color) == AlphaMode::Opaque {
            AlphaMode::Opaque
        } else {
            AlphaMode::Straight
        };
        let app_spec = ImageSpec::new(
            if has_alpha {
                SampleLayout::RGBAF32
            } else {
                SampleLayout::RGBF32
            },
            ColorSpace::Rgb {
                primaries: app_primaries,
                transfer: Transfer::Linear,
                range: SignalRange::Full,
                alpha: app_alpha,
                white: Some(base_white),
            },
        );
        let mut row_pol = config.policy.clone();
        row_pol.sdr_white = Some(base_white);
        let accel = accel::resolve(row_pol.accelerators);

        let base_to_app = RowPlan::build(base_spec.clone(), app_spec.clone(), &row_pol, None)?;
        let (dst_spec, app_to_dst, zero_weight_direct) = if let Some(target) = &config.dst_spec {
            let a2d = RowPlan::build(app_spec, target.clone(), &row_pol, None)?;
            let direct = if is_zero_weight {
                RowPlan::build(base_spec.clone(), target.clone(), &row_pol, None).ok()
            } else {
                None
            };
            (target.clone(), Some(a2d), direct)
        } else {
            (app_spec, None, None)
        };

        // Phase 4: Single out-of-line flat scratch allocation + optional U8 LUT.
        let gains_ch = if gm_ch == 3 || !shared_meta { 3 } else { 1 };
        let scratch = alloc_applier_scratch(
            w_pad,
            ch,
            gm_w as usize,
            gm_ch,
            gains_ch,
            cache_slots,
            !(is_zero_weight && zero_weight_direct.is_some()),
            !is_zero_weight && (app_to_dst.is_some() || w_pad != w_usize),
            !is_zero_weight && !use_u8_lut,
            !is_zero_weight && !use_u8_lut,
            !is_zero_weight,
        )?;
        let need_gamma_tmp = !is_zero_weight
            && !use_u8_lut
            && inv_gamma
                .iter()
                .any(|&g| (g - 1.0).abs() >= 1e-7 && (0.25..=3.0).contains(&g));
        let gamma_tmp = if need_gamma_tmp {
            alloc_f32(w_pad)?
        } else {
            Vec::new()
        };
        let u8_gain_lut = if use_u8_lut {
            Some(build_u8_gain_lut(base_w_g, slope_w_g, inv_gamma))
        } else {
            None
        };

        Ok(Self {
            base_spec,
            base_w,
            base_h,
            w_pad,
            base_row_bytes,
            gm_w,
            gm_h,
            gm_spec,
            gm_row_bytes,
            gm_ch,
            gm_max,
            has_alpha,
            dst_spec,
            dst_row_bytes,
            orientation: Orientation::Identity,
            accel,
            signed_weight: sw,
            is_zero_weight,
            same_size,
            resample: config.resample,
            shared_meta,
            base_w_g,
            slope_w_g,
            inv_gamma,
            k_base,
            k_alt,
            u8_gain_lut,
            base_to_app,
            app_to_dst,
            zero_weight_direct,
            h_weights,
            cache_gm_y: scratch.cache_gm_y,
            cache_planes: scratch.cache_planes,
            cache_cursor: 0,
            v_slot_indices: scratch.v_slot_indices,
            base_lin_row: scratch.base_lin_row,
            out_lin_row: scratch.out_lin_row,
            gm_decode_scratch: scratch.gm_decode_scratch,
            v_blend: scratch.v_blend,
            gamma_tmp,
            gains: scratch.gains,
        })
    }

    /// Output [`ImageSpec`] produced by this applier (in stored-frame coordinates).
    #[inline]
    #[must_use]
    pub fn dst_spec(&self) -> &ImageSpec {
        &self.dst_spec
    }

    /// Pending EXIF [`Orientation`] captured from [`DualLayerHdrView`] (`Orientation::Identity`
    /// when constructed directly via [`GainMapApplier::new`]).
    #[inline]
    #[must_use]
    pub const fn orientation(&self) -> Orientation {
        self.orientation
    }

    /// Signed ISO 21496-1 application weight (`0.0` = exact base bypass, `>0` = Forward, `<0` = Reverse).
    #[inline]
    #[must_use]
    pub const fn signed_weight(&self) -> f32 {
        self.signed_weight
    }

    /// Whether gain-map sampling and additive offsets are bypassed completely (`signed_weight == 0.0`).
    #[inline]
    #[must_use]
    pub const fn is_zero_weight(&self) -> bool {
        self.is_zero_weight
    }

    /// Clear cached horizontally-upsampled gain rows (call before reusing on a different gain map).
    pub fn reset_cache(&mut self) {
        self.cache_gm_y.fill(u32::MAX);
        self.cache_cursor = 0;
    }

    /// Reconstruct row `y` from `base_row` and a borrowed decoded `gain_map` view into `dst_row`.
    pub fn apply_row(
        &mut self,
        y: u32,
        base_row: &[u8],
        gain_map: &PixelView<'_>,
        dst_row: &mut [u8],
    ) -> Result<(), At<ConvertError>> {
        if gain_map.width() != self.gm_w
            || gain_map.rows() != self.gm_h
            || gain_map.spec() != &self.gm_spec
        {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        self.apply_row_dyn(y, base_row, &mut |gy| Ok(gain_map.row(gy)), dst_row)
    }

    /// Reconstruct row `y` while lazily pulling gain-map rows `gy` on demand via a fallible
    /// `fetch_gm_row` provider.
    ///
    /// On resampled paths (`Bilinear`, `Bicubic`, `Mitchell`), horizontally-upsampled gain rows
    /// are cached across consecutive `y` so `fetch_gm_row` is invoked at most once per unique
    /// gain-map row during sequential top-to-bottom decode. Short provider rows return
    /// `Err(ConvertError::DimensionMismatch)` rather than panicking.
    #[inline]
    pub fn apply_row_with_provider<'a, F>(
        &mut self,
        y: u32,
        base_row: &[u8],
        mut fetch_gm_row: F,
        dst_row: &mut [u8],
    ) -> Result<(), At<ConvertError>>
    where
        F: FnMut(u32) -> Result<&'a [u8], At<ConvertError>>,
    {
        self.apply_row_dyn(y, base_row, &mut fetch_gm_row, dst_row)
    }

    #[inline(never)]
    fn apply_row_dyn<'a>(
        &mut self,
        y: u32,
        base_row: &[u8],
        fetch_gm_row: &mut dyn FnMut(u32) -> Result<&'a [u8], At<ConvertError>>,
        dst_row: &mut [u8],
    ) -> Result<(), At<ConvertError>> {
        if y >= self.base_h
            || base_row.len() < self.base_row_bytes
            || dst_row.len() < self.dst_row_bytes
        {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        let w = self.base_w;
        let w_u = w as usize;
        let ch = if self.has_alpha { 4usize } else { 3usize };
        let active_lin_bytes = w_u * ch * 4;

        // Fast exact identity bypass at zero weight: no gain-map fetch, no offset drift!
        if self.is_zero_weight {
            if let Some(direct) = &self.zero_weight_direct {
                return direct.convert_row(base_row, dst_row, w, y);
            }
            if let Some(a2d) = &self.app_to_dst {
                let lin_bytes = &mut bytemuck::cast_slice_mut::<f32, u8>(&mut self.base_lin_row)
                    [..active_lin_bytes];
                self.base_to_app.convert_row(base_row, lin_bytes, w, y)?;
                return a2d.convert_row(
                    &bytemuck::cast_slice(&self.base_lin_row)[..active_lin_bytes],
                    dst_row,
                    w,
                    y,
                );
            }
            return self.base_to_app.convert_row(base_row, dst_row, w, y);
        }

        // 1. Decode base row into linear RGB(A) in application primaries (padding lanes stay 0.0).
        {
            let lin_bytes = &mut bytemuck::cast_slice_mut::<f32, u8>(&mut self.base_lin_row)
                [..active_lin_bytes];
            self.base_to_app.convert_row(base_row, lin_bytes, w, y)?;
        }

        // 2. Obtain linear gain multipliers into `self.gains` (padded to `w_pad`).
        if let Some(lut) = self.u8_gain_lut {
            self.fill_gains_u8_lut(y, &lut, fetch_gm_row)?;
        } else {
            self.fill_samples_for_row(y, fetch_gm_row)?;
            self.decode_samples_to_gains();
        }

        // 3. Apply ISO 21496-1 gain formula `(base + k_base) * gain - k_alt` on padded `w_pad`.
        let w_pad = self.w_pad;
        let need_3 = self.gm_ch == 3 || !self.shared_meta;
        let gr = &self.gains[0..w_pad];
        let gg = if need_3 {
            &self.gains[w_pad..2 * w_pad]
        } else {
            gr
        };
        let gb = if need_3 {
            &self.gains[2 * w_pad..3 * w_pad]
        } else {
            gr
        };

        let combine_fn = self
            .accel
            .gainmap_combine
            .unwrap_or(combine_linear_f32_scalar);

        if let Some(a2d) = &self.app_to_dst {
            combine_fn(
                &self.base_lin_row,
                gr,
                gg,
                gb,
                self.k_base,
                self.k_alt,
                self.has_alpha,
                &mut self.out_lin_row,
            );
            a2d.convert_row(
                &bytemuck::cast_slice(&self.out_lin_row)[..active_lin_bytes],
                dst_row,
                w,
                y,
            )?;
        } else if w_u == w_pad
            && let Ok(dst_f32) =
                bytemuck::try_cast_slice_mut::<u8, f32>(&mut dst_row[..self.dst_row_bytes])
        {
            // Zero-copy aligned fast path (e.g. 4K UHD width 3840 % 8 == 0)!
            combine_fn(
                &self.base_lin_row,
                gr,
                gg,
                gb,
                self.k_base,
                self.k_alt,
                self.has_alpha,
                dst_f32,
            );
        } else {
            // Odd width (w != w_pad) or unaligned dst_row: combine into padded `out_lin_row`.
            if self.out_lin_row.len() < self.base_lin_row.len() {
                self.out_lin_row = alloc_f32(self.base_lin_row.len())?;
            }
            combine_fn(
                &self.base_lin_row,
                gr,
                gg,
                gb,
                self.k_base,
                self.k_alt,
                self.has_alpha,
                &mut self.out_lin_row,
            );
            dst_row[..self.dst_row_bytes]
                .copy_from_slice(&bytemuck::cast_slice(&self.out_lin_row)[..self.dst_row_bytes]);
        }

        Ok(())
    }

    /// Stream an entire image from any [`PixelSource`] + borrowed `gain_map` into any
    /// [`PixelSink`] (`PixelBuf`, `PixelViewMut`, `BandedBuf`) in stored-frame coordinates
    /// using O(width) scratch memory, checking `stop` once per row.
    pub fn apply_into<S: PixelSource + ?Sized, D: PixelSink + ?Sized>(
        &mut self,
        base: &S,
        gain_map: &PixelView<'_>,
        dst: &mut D,
        stop: &dyn Stop,
    ) -> Result<(), At<ConvertError>> {
        if base.source_width() != self.base_w
            || base.source_height() != self.base_h
            || base.source_spec() != &self.base_spec
            || gain_map.width() != self.gm_w
            || gain_map.rows() != self.gm_h
            || gain_map.spec() != &self.gm_spec
            || dst.sink_width() != self.base_w
            || dst.sink_height() != self.base_h
            || dst.sink_spec() != &self.dst_spec
        {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        self.reset_cache();
        for y in 0..self.base_h {
            stop.check().map_err(|r| err_at(r.into()))?;
            self.apply_row_dyn(
                y,
                base.source_row(y),
                &mut |gy| Ok(gain_map.row(gy)),
                dst.sink_row_mut(y),
            )?;
        }
        dst.sink_finish(SampleFacts::NONE, HdrMetadata::default());
        Ok(())
    }

    #[inline(never)]
    fn fill_gains_u8_lut<'a>(
        &mut self,
        y: u32,
        lut: &[[f32; 256]; 3],
        fetch_gm_row: &mut dyn FnMut(u32) -> Result<&'a [u8], At<ConvertError>>,
    ) -> Result<(), At<ConvertError>> {
        let gy = if self.same_size {
            y
        } else {
            let ratio = f64::from(self.gm_h) / f64::from(self.base_h);
            let max_in = i64::from(self.gm_h - 1);
            (libm::floor((f64::from(y) + 0.5) * ratio) as i64).clamp(0, max_in) as u32
        };
        let row = fetch_gm_row(gy)?;
        if row.len() < self.gm_row_bytes {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        let w = self.base_w as usize;
        let w_pad = self.w_pad;
        let need_3 = self.gm_ch == 3 || !self.shared_meta;

        if self.same_size && self.gm_ch == 1 && !need_3 {
            // Hot 1:1 1-channel shared UltraHDR LUT path (branch-free tight loop).
            for (out, &b) in self.gains[..w].iter_mut().zip(row[..w].iter()) {
                *out = lut[0][usize::from(b)];
            }
            return Ok(());
        }

        let hw_left = self.h_weights.as_ref().map(|hw| hw.left.as_slice());
        if self.gm_ch == 1 {
            for x in 0..w {
                let sx = hw_left.map_or(x, |l| l[x] as usize);
                let b = usize::from(row[sx]);
                self.gains[x] = lut[0][b];
                if need_3 {
                    self.gains[w_pad + x] = lut[1][b];
                    self.gains[2 * w_pad + x] = lut[2][b];
                }
            }
        } else {
            for x in 0..w {
                let o = hw_left.map_or(x, |l| l[x] as usize) * 3;
                self.gains[x] = lut[0][usize::from(row[o])];
                self.gains[w_pad + x] = lut[1][usize::from(row[o + 1])];
                self.gains[2 * w_pad + x] = lut[2][usize::from(row[o + 2])];
            }
        }
        Ok(())
    }

    #[inline(never)]
    fn ensure_horiz_row<'a>(
        &mut self,
        gm_y: u32,
        keep_start: u32,
        keep_end: u32,
        fetch_gm_row: &mut dyn FnMut(u32) -> Result<&'a [u8], At<ConvertError>>,
    ) -> Result<usize, At<ConvertError>> {
        if let Some(idx) = self.cache_gm_y.iter().position(|&gy| gy == gm_y) {
            return Ok(idx);
        }
        let n_slots = self.cache_gm_y.len();
        let mut slot_idx = self.cache_cursor;
        if let Some(empty) = self.cache_gm_y.iter().position(|&gy| gy == u32::MAX) {
            slot_idx = empty;
        } else {
            for step in 0..n_slots {
                let cand = (self.cache_cursor + step) % n_slots;
                let gy = self.cache_gm_y[cand];
                if gy < keep_start || gy >= keep_end {
                    slot_idx = cand;
                    break;
                }
            }
        }
        self.cache_cursor = (slot_idx + 1) % n_slots;

        let raw_row = fetch_gm_row(gm_y)?;
        if raw_row.len() < self.gm_row_bytes {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        decode_gm_row_planar(
            raw_row,
            self.gm_w as usize,
            self.gm_spec.layout,
            self.gm_max,
            &mut self.gm_decode_scratch,
        );

        let w = self.base_w as usize;
        let w_pad = self.w_pad;
        let gm_w = self.gm_w as usize;
        let slot_stride = self.gm_ch * w_pad;
        let hw = self.h_weights.as_ref().unwrap();
        let dst_slot = &mut self.cache_planes[slot_idx * slot_stride..(slot_idx + 1) * slot_stride];

        for c in 0..self.gm_ch {
            let src_plane = &self.gm_decode_scratch[c * gm_w..(c + 1) * gm_w];
            let dst_plane = &mut dst_slot[c * w_pad..c * w_pad + w];
            if self.resample == GainMapResample::Nearest {
                for (x, out) in dst_plane.iter_mut().enumerate() {
                    *out = src_plane[hw.left[x] as usize];
                }
            } else if hw.max_taps == 2 {
                // Fast unrolled 2-tap horizontal bilinear loop (eliminates inner slice/loop overhead).
                let lefts = hw.left.as_slice();
                let counts = hw.tap_counts.as_slice();
                let wflat = hw.weights_flat.as_slice();
                for x in 0..w {
                    let l = lefts[x] as usize;
                    let off = x * 2;
                    dst_plane[x] = if counts[x] == 2 {
                        src_plane[l] * wflat[off] + src_plane[l + 1] * wflat[off + 1]
                    } else {
                        src_plane[l] * wflat[off]
                    };
                }
            } else {
                for (x, out) in dst_plane.iter_mut().enumerate() {
                    let (left, ws) = hw.taps(x);
                    let l = left as usize;
                    let mut sum = 0.0f32;
                    for (t, &wt) in ws.iter().enumerate() {
                        sum += src_plane[l + t] * wt;
                    }
                    *out = sum;
                }
            }
        }
        self.cache_gm_y[slot_idx] = gm_y;
        Ok(slot_idx)
    }

    #[inline(never)]
    fn fill_samples_for_row<'a>(
        &mut self,
        y: u32,
        fetch_gm_row: &mut dyn FnMut(u32) -> Result<&'a [u8], At<ConvertError>>,
    ) -> Result<(), At<ConvertError>> {
        let w = self.base_w as usize;
        let w_pad = self.w_pad;
        if self.same_size {
            let raw = fetch_gm_row(y)?;
            if raw.len() < self.gm_row_bytes {
                return Err(err_at(ConvertError::DimensionMismatch));
            }
            decode_gm_row_planar(
                raw,
                w,
                self.gm_spec.layout,
                self.gm_max,
                &mut self.gm_decode_scratch,
            );
            for c in 0..self.gm_ch {
                self.v_blend[c * w_pad..c * w_pad + w]
                    .copy_from_slice(&self.gm_decode_scratch[c * w..(c + 1) * w]);
            }
            return Ok(());
        }

        let mut v_ws_buf = [0.0f32; MAX_APPLY_RESAMPLE_TAPS];
        let (v_left, v_count) = compute_axis_taps_stack(
            self.gm_h,
            self.base_h,
            self.resample,
            y,
            MAX_APPLY_RESAMPLE_TAPS,
            &mut v_ws_buf,
        )
        .map_err(err_at)?;
        let keep_end = v_left + v_count as u32;

        for t in 0..v_count {
            let idx = self.ensure_horiz_row(v_left + t as u32, v_left, keep_end, fetch_gm_row)?;
            self.v_slot_indices[t] = idx;
        }

        let ws = &v_ws_buf[..v_count];
        let blend2 = self.accel.gainmap_blend_2.unwrap_or(blend_rows_2_scalar);
        let blend4 = self.accel.gainmap_blend_4.unwrap_or(blend_rows_4_scalar);
        let slot_stride = self.gm_ch * w_pad;

        for c in 0..self.gm_ch {
            let co = c * w_pad;
            let dst = &mut self.v_blend[co..co + w_pad];
            match v_count {
                1 => {
                    let b0 = self.v_slot_indices[0] * slot_stride + co;
                    let s0 = &self.cache_planes[b0..b0 + w_pad];
                    blend2(s0, s0, 1.0, 0.0, dst);
                }
                2 => {
                    let b0 = self.v_slot_indices[0] * slot_stride + co;
                    let b1 = self.v_slot_indices[1] * slot_stride + co;
                    blend2(
                        &self.cache_planes[b0..b0 + w_pad],
                        &self.cache_planes[b1..b1 + w_pad],
                        ws[0],
                        ws[1],
                        dst,
                    );
                }
                4 => {
                    let b0 = self.v_slot_indices[0] * slot_stride + co;
                    let b1 = self.v_slot_indices[1] * slot_stride + co;
                    let b2 = self.v_slot_indices[2] * slot_stride + co;
                    let b3 = self.v_slot_indices[3] * slot_stride + co;
                    blend4(
                        &self.cache_planes[b0..b0 + w_pad],
                        &self.cache_planes[b1..b1 + w_pad],
                        &self.cache_planes[b2..b2 + w_pad],
                        &self.cache_planes[b3..b3 + w_pad],
                        [ws[0], ws[1], ws[2], ws[3]],
                        dst,
                    );
                }
                _ => {
                    blend_rows_general(
                        &self.cache_planes,
                        slot_stride,
                        co,
                        &self.v_slot_indices,
                        ws,
                        w,
                        dst,
                    );
                }
            }
        }
        Ok(())
    }

    #[inline(never)]
    fn decode_samples_to_gains(&mut self) {
        let w = self.base_w as usize;
        let w_pad = self.w_pad;
        let decode_fn = self.accel.gainmap_decode.unwrap_or(decode_gain_1ch_scalar);
        let pow_fn = self
            .accel
            .gamma_pow
            .unwrap_or(crate::transfers::gamma_pow_scalar_slice);
        let channels = if self.gm_ch == 3 || !self.shared_meta {
            3usize
        } else {
            1usize
        };
        for c in 0..channels {
            let src_off = if self.gm_ch == 3 { c * w_pad } else { 0 };
            let samples = &self.v_blend[src_off..src_off + w_pad];
            let dst = &mut self.gains[c * w_pad..(c + 1) * w_pad];
            let inv_g = self.inv_gamma[c];
            if (inv_g - 1.0).abs() < 1e-7 {
                decode_fn(samples, self.base_w_g[c], self.slope_w_g[c], dst);
            } else if (0.25..=3.0).contains(&inv_g) && self.gamma_tmp.len() >= w_pad {
                // Vectorized zero-codegen-overhead `powf` reusing `transfers::TRANSFER_POW`
                // (`gamma_pow_simd_slice` / `gamma_pow_scalar_slice` with 0-bit parity!).
                let tmp = &mut self.gamma_tmp[..w_pad];
                for i in 0..w_pad {
                    let raw = samples[i];
                    tmp[i] = if raw > 0.0 {
                        if raw < 1.0 { raw } else { 1.0 }
                    } else {
                        0.0
                    };
                }
                pow_fn(tmp, inv_g, f64::from(inv_g));
                decode_fn(tmp, self.base_w_g[c], self.slope_w_g[c], dst);
            } else {
                decode_gain_gamma_scalar(
                    samples,
                    w,
                    inv_g,
                    self.base_w_g[c],
                    self.slope_w_g[c],
                    dst,
                );
            }
        }
    }
}

/// Decode one gain-map scanline (`U8` / `U16` / `F32`, full or `Lsb`/`Msb` bit-packed,
/// 1-ch or 3-ch) into planar `[0, 1]` f32 channels inside `dst_planar` (`ch * width`).
#[inline(never)]
fn decode_gm_row_planar(
    row: &[u8],
    width: usize,
    layout: SampleLayout,
    max_code: [f32; 3],
    dst_planar: &mut [f32],
) {
    let ch = layout.channels();
    let inv = [1.0 / max_code[0], 1.0 / max_code[1], 1.0 / max_code[2]];
    let full = layout.packing.is_full();

    // Fast path for the dominant UltraHDR 1-ch full-packing U8 gain map.
    if layout.sample == SampleType::U8 && full && ch == 1 {
        let inv0 = inv[0];
        for (dst, &b) in dst_planar[..width].iter_mut().zip(row[..width].iter()) {
            *dst = f32::from(b) * inv0;
        }
        return;
    }

    // Compact unified decoder for 3-ch U8, bit-packed U8/U16 (10/12-bit), full U16, and F32.
    let bps = match layout.sample {
        SampleType::U8 => 1usize,
        SampleType::U16 => 2usize,
        SampleType::F32 => 4usize,
        _ => return,
    };
    for x in 0..width {
        for c in 0..ch {
            let elem = x * ch + c;
            let o = elem * bps;
            let v = match layout.sample {
                SampleType::U8 => {
                    let raw = u32::from(row[o]);
                    let code = if full {
                        raw
                    } else {
                        layout.packing.extract_code(SampleType::U8, raw, c)
                    };
                    (code as f32) * inv[c]
                }
                SampleType::U16 => {
                    let raw = u32::from(u16::from_ne_bytes([row[o], row[o + 1]]));
                    let code = if full {
                        raw
                    } else {
                        layout.packing.extract_code(SampleType::U16, raw, c)
                    };
                    (code as f32) * inv[c]
                }
                SampleType::F32 => {
                    let f = f32::from_ne_bytes([row[o], row[o + 1], row[o + 2], row[o + 3]]);
                    if f.is_finite() { f } else { 0.0 }
                }
                _ => 0.0,
            };
            dst_planar[c * width + x] = v;
        }
    }
}

fn validate_gain_map_spec(spec: &ImageSpec) -> Result<(), At<ConvertError>> {
    let ok_color = matches!(
        spec.color,
        ColorSpace::Untagged {
            channels: 1 | 3,
            alpha: AlphaMode::None,
        }
    );
    let ok_sample = matches!(
        spec.layout.sample,
        SampleType::U8 | SampleType::U16 | SampleType::F32
    );
    let ok_packing = matches!(
        spec.layout.packing,
        BitPacking::Full | BitPacking::Lsb { .. } | BitPacking::Msb { .. }
    );
    let ok_order = is_valid_gain_map_order(spec.layout.order);
    let ok_ch = matches!(spec.layout.channels(), 1 | 3);
    if !ok_color || !ok_sample || !ok_packing || !ok_order || !ok_ch {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// High-level apply & dual-layer orientation functions
// ---------------------------------------------------------------------------

#[inline(never)]
fn apply_gain_map_stored(
    dual: &DualLayerHdrView<'_>,
    config: &GainMapApplyConfig,
    stop: &dyn Stop,
) -> Result<PixelBuf, At<ConvertError>> {
    stop.check().map_err(|r| err_at(r.into()))?;
    let mut applier = GainMapApplier::for_view(dual, config)?;
    let mut out = PixelBuf::new(
        dual.base().width(),
        dual.base().rows(),
        applier.dst_spec().clone(),
    )
    .map_err(|e| err_at((*e.error()).into()))?;
    applier.apply_into(dual.base(), dual.gain_map(), &mut out, stop)?;
    Ok(out)
}

/// Reconstruct a [`PixelBuf`] and its resulting pending [`Orientation`] from a zero-copy
/// [`DualLayerHdrView`] according to `config.orientation`:
///
/// - [`GainMapOrientation::Bake`] (default): reconstructs in stored coordinates, then
///   physically applies `dual.orientation()`, returning `(display_buf, Orientation::Identity)`.
/// - [`GainMapOrientation::Passthrough`]: reconstructs in stored coordinates with zero
///   transpose cost, returning `(stored_buf, dual.orientation())` for EXIF passthrough or
///   GPU UV rotation.
/// - [`GainMapOrientation::RequireIdentity`]: returns `Err(ConvertError::UnsupportedRoute)`
///   if `dual.orientation() != Orientation::Identity`.
#[inline]
pub fn apply_gain_map_oriented(
    dual: &DualLayerHdrView<'_>,
    config: &GainMapApplyConfig,
    stop: &dyn Stop,
) -> Result<(PixelBuf, Orientation), At<ConvertError>> {
    let orient = dual.orientation();
    if config.orientation == GainMapOrientation::RequireIdentity && orient != Orientation::Identity
    {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    let out = apply_gain_map_stored(dual, config, stop)?;
    if config.orientation == GainMapOrientation::Bake && orient != Orientation::Identity {
        stop.check().map_err(|r| err_at(r.into()))?;
        let baked = orient::apply_orientation(&out.as_view(), orient)?;
        Ok((baked, Orientation::Identity))
    } else {
        Ok((out, orient))
    }
}

/// Reconstruct a linear-light [`PixelBuf`] (`RGBF32` / `RGBAF32`) from an owned
/// [`DualLayerHdr`] at `display_headroom_stops` (log₂ stops, e.g. `0.0` = SDR,
/// `2.32` ≈ 5× / 1000-nit HDR display), baking `dual.orientation()` to display orientation.
pub fn apply_gain_map(
    dual: &DualLayerHdr,
    display_headroom_stops: f32,
) -> Result<PixelBuf, At<ConvertError>> {
    let cfg = GainMapApplyConfig::new(display_headroom_stops);
    apply_gain_map_with(&dual.as_view(), &cfg, &Unstoppable)
}

/// Reconstruct a linear-light [`PixelBuf`] (`RGBF32` / `RGBAF32`) from a zero-copy
/// [`DualLayerHdrView`] using `policy` (allowing CMS backends and SIMD selection),
/// baking `dual.orientation()` to display orientation.
pub fn apply_gain_map_view(
    dual: &DualLayerHdrView<'_>,
    display_headroom_stops: f32,
    policy: &ConvertPolicy,
) -> Result<PixelBuf, At<ConvertError>> {
    let cfg = GainMapApplyConfig::new(display_headroom_stops).with_policy(policy.clone());
    apply_gain_map_with(dual, &cfg, &Unstoppable)
}

/// Reconstruct a [`PixelBuf`] from a zero-copy [`DualLayerHdrView`] with full control over
/// target headroom/rendition, resampling kernel, orientation policy, conversion policy,
/// optional target [`ImageSpec`], and cooperative [`enough::Stop`] cancellation.
///
/// Refuses [`GainMapOrientation::Passthrough`] when `dual.orientation() != Orientation::Identity`
/// so a pending rotation is never silently discarded (use [`apply_gain_map_oriented`] when you
/// want the stored-coordinate buffer paired with its pending [`Orientation`]).
#[inline]
pub fn apply_gain_map_with(
    dual: &DualLayerHdrView<'_>,
    config: &GainMapApplyConfig,
    stop: &dyn Stop,
) -> Result<PixelBuf, At<ConvertError>> {
    if config.orientation == GainMapOrientation::Passthrough
        && dual.orientation() != Orientation::Identity
    {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    let (buf, _) = apply_gain_map_oriented(dual, config, stop)?;
    Ok(buf)
}

/// Physically bake `dual.orientation()` into both `base` and `gain_map`, returning an owned
/// aligned [`DualLayerHdr`] with [`Orientation::Identity`].
///
/// Both layers receive the exact same dihedral transform so their stored-frame coordinates
/// stay registered (including subsampled ¼-res gain maps under pixel-center siting).
/// For O(1) lossless EXIF passthrough without rebaking pixels, use
/// [`DualLayerHdr::rotate_metadata`] instead.
#[inline(never)]
pub fn bake_dual_layer_orientation(
    dual: &DualLayerHdrView<'_>,
) -> Result<DualLayerHdr, At<ConvertError>> {
    rotate_dual_layer(dual, Orientation::Identity)
}

/// Physically apply `dual.orientation().then(delta)` to both `base` and `gain_map`, keeping
/// both layers strictly registered and returning an owned [`DualLayerHdr`] with
/// [`Orientation::Identity`].
///
/// Preserves `metadata`, `alternate_color_space`, and per-layer `HdrMetadata` / `SampleFacts`.
/// For O(1) EXIF passthrough metadata rotation without pixel rebake, use
/// [`DualLayerHdr::rotate_metadata`] / [`DualLayerHdrView::rotate_metadata`].
#[inline(never)]
pub fn rotate_dual_layer(
    dual: &DualLayerHdrView<'_>,
    delta: Orientation,
) -> Result<DualLayerHdr, At<ConvertError>> {
    let total = dual.orientation().then(delta);
    let base = orient::apply_orientation(dual.base(), total)?;
    let gain_map = orient::apply_orientation(dual.gain_map(), total)?;
    DualLayerHdr::new_with_alternate(
        base,
        gain_map,
        dual.metadata(),
        dual.alternate_color_space().cloned(),
        Orientation::Identity,
    )
    .map_err(|e| err_at((*e.error()).into()))
}

// ---------------------------------------------------------------------------
// Extraction: downsampling modes, configuration, streaming row extractor
// ---------------------------------------------------------------------------

/// Downsampling filter used during gain-map extraction when `scale_factor > 1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GainMapDownsample {
    /// Area-averaging box filter over each `scale × scale` block in linear light (default).
    /// Prevents aliasing and single-pixel highlight sparkle on subsampled gain maps.
    #[default]
    Box,
    /// Separable center-sited triangle/bilinear filter in linear light.
    Bilinear,
    /// Center-pixel point sampling `(gx * scale + scale / 2).min(w - 1)`, matching legacy
    /// `ultrahdr-core` / `zenjpeg` subsampling.
    Point,
}

/// Configuration for ISO 21496-1 gain-map extraction ([`extract_gain_map_only`],
/// [`GainMapRowExtractor`]).
#[derive(Clone, Debug)]
pub struct GainMapExtractConfig {
    /// Subsampling divisor for the output gain map (`1` = full resolution, `4` = ¼-res
    /// UltraHDR standard; output dimensions are `div_ceil(scale_factor)`).
    pub scale_factor: u8,
    /// Downsampling filter applied when `scale_factor > 1`.
    pub downsample: GainMapDownsample,
    /// `false` = 1-channel luminance gain map (`GainMapChannels::Shared`);
    /// `true` = 3-channel RGB gain map (`GainMapChannels::PerChannel`).
    pub multi_channel: bool,
    /// Sample storage type (`U8` default, or `U16` for 10/12/16-bit AVIF/JXL maps).
    pub sample_type: SampleType,
    /// Bit packing for integer gain-map samples (`BitPacking::Full` default; e.g.
    /// `BitPacking::Lsb { bits: 10 }` for 10-bit AVIF/JXL maps).
    pub packing: BitPacking,
    /// Minimum encoded log₂ gain stop (code `0`).
    pub min_gain_stops: f32,
    /// Maximum encoded log₂ gain stop (code `max_code`).
    pub max_gain_stops: f32,
    /// Gain-map gamma (`> 0.0`, default `1.0`).
    pub gamma: f32,
    /// Linear base offset (`k_base`, default `1/64`).
    pub base_offset: f32,
    /// Linear alternate offset (`k_alt`, default `1/64`).
    pub alternate_offset: f32,
    /// Base rendition HDR headroom in log₂ stops (`0.0` = SDR base).
    pub base_hdr_headroom_stops: f32,
    /// Alternate rendition HDR headroom in log₂ stops (`> base` = Forward, `< base` = Reverse).
    pub alternate_hdr_headroom_stops: f32,
    /// Primaries in which channelwise gain is computed (`Base` or `Alternate`).
    pub application_space: GainMapColorSpace,
    /// When `true`, performs a 2-pass content-fit scan that narrows the quantization grid
    /// to the content's observed gain range within `[min_gain_stops, max_gain_stops]`
    /// (minimum span [`CONTENT_FIT_MIN_SPAN_STOPS`]).
    pub content_fit: bool,
    /// Conversion policy used to linearize base and alternate inputs.
    pub policy: ConvertPolicy,
}

impl Default for GainMapExtractConfig {
    fn default() -> Self {
        Self {
            scale_factor: 1,
            downsample: GainMapDownsample::Box,
            multi_channel: false,
            sample_type: SampleType::U8,
            packing: BitPacking::Full,
            min_gain_stops: 0.0,
            max_gain_stops: 2.5,
            gamma: 1.0,
            base_offset: 1.0 / 64.0,
            alternate_offset: 1.0 / 64.0,
            base_hdr_headroom_stops: 0.0,
            alternate_hdr_headroom_stops: 2.5,
            application_space: GainMapColorSpace::Base,
            content_fit: false,
            policy: ConvertPolicy::working_linear(),
        }
    }
}

impl GainMapExtractConfig {
    /// Standard UltraHDR preset (¼-res `scale_factor = 4`, 1-ch U8, `[0, ~2.58]` stops =
    /// `1×..=6×` linear boost, `1/64` offsets, `Box` area downsampling).
    #[must_use]
    pub fn ultrahdr() -> Self {
        Self::default()
            .with_scale_factor(4)
            .with_boost_range_linear(1.0, 6.0)
            .with_headroom_boost_linear(1.0, 6.0)
    }

    /// Set gain-map subsampling divisor (`>= 1`).
    #[inline]
    #[must_use]
    pub fn with_scale_factor(mut self, scale_factor: u8) -> Self {
        self.scale_factor = scale_factor.max(1);
        self
    }

    /// Set downsampling filter when `scale_factor > 1`.
    #[inline]
    #[must_use]
    pub fn with_downsample(mut self, downsample: GainMapDownsample) -> Self {
        self.downsample = downsample;
        self
    }

    /// Select 1-channel (`false`) vs 3-channel RGB (`true`) gain map.
    #[inline]
    #[must_use]
    pub fn with_multi_channel(mut self, multi_channel: bool) -> Self {
        self.multi_channel = multi_channel;
        self
    }

    /// Select integer storage type & packing (e.g. `(SampleType::U16, BitPacking::Lsb { bits: 10 })`).
    #[inline]
    #[must_use]
    pub fn with_sample_storage(mut self, sample_type: SampleType, packing: BitPacking) -> Self {
        self.sample_type = sample_type;
        self.packing = packing;
        self
    }

    /// Set quantization grid in log₂ stops.
    #[inline]
    #[must_use]
    pub fn with_gain_range_stops(mut self, min_stops: f32, max_stops: f32) -> Self {
        self.min_gain_stops = min_stops;
        self.max_gain_stops = max_stops;
        self
    }

    /// Set quantization grid from positive linear boost ratios (e.g. `1.0`, `6.0`).
    #[inline]
    #[must_use]
    pub fn with_boost_range_linear(mut self, min_boost: f32, max_boost: f32) -> Self {
        self.min_gain_stops = libm::log2f(min_boost.max(1e-6));
        self.max_gain_stops = libm::log2f(max_boost.max(1e-6));
        self
    }

    /// Set gain-map encoding gamma (`> 0.0`).
    #[inline]
    #[must_use]
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set linear base & alternate offsets.
    #[inline]
    #[must_use]
    pub fn with_offsets(mut self, base_offset: f32, alternate_offset: f32) -> Self {
        self.base_offset = base_offset;
        self.alternate_offset = alternate_offset;
        self
    }

    /// Set base and alternate HDR headrooms in log₂ stops.
    #[inline]
    #[must_use]
    pub fn with_headroom_stops(mut self, base_stops: f32, alternate_stops: f32) -> Self {
        self.base_hdr_headroom_stops = base_stops;
        self.alternate_hdr_headroom_stops = alternate_stops;
        self
    }

    /// Set base and alternate HDR headrooms from linear boost ratios (`1.0` → `0.0` stops).
    #[inline]
    #[must_use]
    pub fn with_headroom_boost_linear(mut self, base_boost: f32, alternate_boost: f32) -> Self {
        self.base_hdr_headroom_stops = display_boost_to_headroom_stops(base_boost);
        self.alternate_hdr_headroom_stops = display_boost_to_headroom_stops(alternate_boost);
        self
    }

    /// Choose whether gain is computed in `Base` or `Alternate` linear primaries.
    #[inline]
    #[must_use]
    pub fn with_application_space(mut self, space: GainMapColorSpace) -> Self {
        self.application_space = space;
        self
    }

    /// Enable 2-pass content-fit grid narrowing to the observed content gain range.
    #[inline]
    #[must_use]
    pub fn with_content_fit(mut self, content_fit: bool) -> Self {
        self.content_fit = content_fit;
        self
    }

    /// Set conversion policy used to linearize base and alternate images.
    #[inline]
    #[must_use]
    pub fn with_policy(mut self, policy: ConvertPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Build the exact [`GainMapMetadata`] matching this configuration's quantization grid.
    #[inline(never)]
    pub fn to_metadata(&self) -> Result<GainMapMetadata, At<ConvertError>> {
        if !self.min_gain_stops.is_finite()
            || !self.max_gain_stops.is_finite()
            || self.max_gain_stops < self.min_gain_stops
            || !self.gamma.is_finite()
            || self.gamma <= 0.0
            || !self.base_offset.is_finite()
            || !self.alternate_offset.is_finite()
            || !self.base_hdr_headroom_stops.is_finite()
            || !self.alternate_hdr_headroom_stops.is_finite()
            || self.base_hdr_headroom_stops < 0.0
            || self.alternate_hdr_headroom_stops < 0.0
        {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        let comp = GainMapComponent {
            min_gain_stops: signed_fraction_from_f32(self.min_gain_stops),
            max_gain_stops: signed_fraction_from_f32(self.max_gain_stops),
            gamma: unsigned_fraction_from_f32(self.gamma),
            base_offset: signed_fraction_from_f32(self.base_offset),
            alternate_offset: signed_fraction_from_f32(self.alternate_offset),
        };
        let channels = if self.multi_channel {
            GainMapChannels::PerChannel([comp; 3])
        } else {
            GainMapChannels::Shared(comp)
        };
        let h_base = unsigned_fraction_from_f32(self.base_hdr_headroom_stops);
        let h_alt = unsigned_fraction_from_f32(self.alternate_hdr_headroom_stops);
        GainMapMetadata::try_new(channels, h_base, h_alt, self.application_space)
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))
    }

    fn gain_map_spec(&self) -> Result<ImageSpec, At<ConvertError>> {
        let layout = match (self.sample_type, self.multi_channel) {
            (SampleType::U8, false) => SampleLayout::GRAY8.with_packing(self.packing),
            (SampleType::U8, true) => SampleLayout::RGB8.with_packing(self.packing),
            (SampleType::U16, false) => SampleLayout::GRAY16.with_packing(self.packing),
            (SampleType::U16, true) => SampleLayout::RGB16.with_packing(self.packing),
            _ => return Err(err_at(ConvertError::UnsupportedRoute)),
        };
        let spec = ImageSpec::new(
            layout,
            ColorSpace::Untagged {
                channels: if self.multi_channel { 3 } else { 1 },
                alpha: AlphaMode::None,
            },
        );
        spec.validate().map_err(|e| err_at((*e.error()).into()))?;
        validate_gain_map_spec(&spec)?;
        Ok(spec)
    }
}

/// Row-streaming linear-f32 ISO 21496-1 gain-map encoder.
#[derive(Clone, Debug)]
pub struct GainMapRowExtractor {
    metadata: GainMapMetadata,
    multichannel: bool,
    sign: f32,
    luma: [f32; 3],
    min_g: [f32; 3],
    span_g: [f32; 3],
    gamma: [f32; 3],
    k_base: [f32; 3],
    k_alt: [f32; 3],
    observed_min_stops: f32,
    observed_max_stops: f32,
}

impl GainMapRowExtractor {
    /// Construct a row extractor using exact [`GainMapMetadata`] and application primaries.
    #[inline(never)]
    pub fn new(
        metadata: GainMapMetadata,
        app_primaries: Primaries,
    ) -> Result<Self, At<ConvertError>> {
        if !metadata.is_valid() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        let direction = metadata
            .direction()
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        let sign = match direction {
            GainMapDirection::Forward => 1.0_f32,
            GainMapDirection::Reverse => -1.0_f32,
        };
        let mut min_g = [0.0f32; 3];
        let mut span_g = [0.0f32; 3];
        let mut gamma = [1.0f32; 3];
        let mut k_base = [0.0f32; 3];
        let mut k_alt = [0.0f32; 3];
        for c in 0..3 {
            let comp = metadata.channel(c);
            min_g[c] = comp.min_gain_stops.to_f32();
            span_g[c] = comp.max_gain_stops.to_f32() - min_g[c];
            gamma[c] = comp.gamma.to_f32();
            k_base[c] = comp.base_offset.to_f32();
            k_alt[c] = comp.alternate_offset.to_f32();
        }
        Ok(Self {
            multichannel: metadata.channels.is_multichannel(),
            metadata,
            sign,
            luma: app_primaries.luma_weights(),
            min_g,
            span_g,
            gamma,
            k_base,
            k_alt,
            observed_min_stops: f32::INFINITY,
            observed_max_stops: f32::NEG_INFINITY,
        })
    }

    /// Construct a row extractor from a [`GainMapExtractConfig`] and application primaries.
    pub fn from_config(
        config: &GainMapExtractConfig,
        app_primaries: Primaries,
    ) -> Result<Self, At<ConvertError>> {
        Self::new(config.to_metadata()?, app_primaries)
    }

    /// Quantize one row of paired linear RGB(A) `f32` pixels (`channels` is `3` or `4`)
    /// into packed gain-map sample bytes matching `gm_layout` (`U8` or `U16`, `Full`/`Lsb`/`Msb`,
    /// `Gray`/`Rgb`/`Generic(1|3)`).
    ///
    /// Computes exact `f64` signed-offset ratios `(alt + k_alt) / (base + k_base)` without
    /// near-black floor clamping.
    #[inline(never)]
    pub fn encode_row_linear_f32(
        &mut self,
        base_linear: &[f32],
        alt_linear: &[f32],
        channels: usize,
        gm_layout: SampleLayout,
        dst_row: &mut [u8],
    ) -> Result<(), At<ConvertError>> {
        if channels != 3 && channels != 4 {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        #[allow(clippy::manual_is_multiple_of)]
        if base_linear.len() != alt_linear.len() || base_linear.len() % channels != 0 {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        let gm_ch = if self.multichannel { 3usize } else { 1usize };
        if !matches!(gm_layout.sample, SampleType::U8 | SampleType::U16)
            || !matches!(
                gm_layout.packing,
                BitPacking::Full | BitPacking::Lsb { .. } | BitPacking::Msb { .. }
            )
            || !is_valid_gain_map_order(gm_layout.order)
        {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        if gm_layout.channels() != gm_ch {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        ImageSpec::new(
            gm_layout,
            ColorSpace::Untagged {
                channels: gm_ch as u8,
                alpha: AlphaMode::None,
            },
        )
        .validate()
        .map_err(|e| err_at((*e.error()).into()))?;

        let pixels = base_linear.len() / channels;
        let need_bytes = pixels
            .checked_mul(gm_layout.bytes_per_pixel())
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        if dst_row.len() < need_bytes {
            return Err(err_at(ConvertError::DimensionMismatch));
        }

        let max_code = gm_layout.packing.max_code(gm_layout.sample, 0).max(1);
        let max_f = max_code as f32;

        if self.multichannel {
            for x in 0..pixels {
                let o = x * channels;
                for c in 0..3 {
                    let log2_r = compute_log2_ratio(
                        self.sign,
                        base_linear[o + c],
                        alt_linear[o + c],
                        self.k_base[c],
                        self.k_alt[c],
                        self.min_g[c],
                        self.min_g[c] + self.span_g[c],
                    );
                    self.observed_min_stops = self.observed_min_stops.min(log2_r);
                    self.observed_max_stops = self.observed_max_stops.max(log2_r);
                    let code = quantize_log2_gain(
                        log2_r,
                        self.min_g[c],
                        self.span_g[c],
                        self.gamma[c],
                        max_code,
                        max_f,
                    );
                    write_gm_code(gm_layout, dst_row, x * 3 + c, c, code);
                }
            }
        } else {
            let [yr, yg, yb] = self.luma;
            let max_g0 = self.min_g[0] + self.span_g[0];
            for x in 0..pixels {
                let o = x * channels;
                let lb = yr * base_linear[o] + yg * base_linear[o + 1] + yb * base_linear[o + 2];
                let la = yr * alt_linear[o] + yg * alt_linear[o + 1] + yb * alt_linear[o + 2];
                let log2_r = compute_log2_ratio(
                    self.sign,
                    lb,
                    la,
                    self.k_base[0],
                    self.k_alt[0],
                    self.min_g[0],
                    max_g0,
                );
                self.observed_min_stops = self.observed_min_stops.min(log2_r);
                self.observed_max_stops = self.observed_max_stops.max(log2_r);
                let code = quantize_log2_gain(
                    log2_r,
                    self.min_g[0],
                    self.span_g[0],
                    self.gamma[0],
                    max_code,
                    max_f,
                );
                write_gm_code(gm_layout, dst_row, x, 0, code);
            }
        }

        Ok(())
    }

    /// Observe the `(min_log2_stops, max_log2_stops)` gain range accumulated across all
    /// rows encoded so far (`None` if zero pixels were processed).
    #[inline]
    #[must_use]
    pub fn observed_gain_stops(&self) -> Option<(f32, f32)> {
        if self.observed_min_stops.is_finite() && self.observed_max_stops.is_finite() {
            Some((self.observed_min_stops, self.observed_max_stops))
        } else {
            None
        }
    }

    /// Return the exact [`GainMapMetadata`] matching the encoded bytes.
    #[inline]
    #[must_use]
    pub fn metadata(&self) -> GainMapMetadata {
        self.metadata
    }

    /// Build [`GainMapMetadata`] for the encoded bytes, optionally widening
    /// `alternate_hdr_headroom` (in Forward mode) if observed content gain exceeded the
    /// configured alternate headroom (matching `ultrahdr-core`'s contract `#33`).
    #[must_use]
    pub fn build_metadata_widening_headroom(&self) -> GainMapMetadata {
        let mut out = self.metadata;
        if self.sign > 0.0
            && self.observed_max_stops.is_finite()
            && self.observed_max_stops > out.alternate_hdr_headroom.to_f32()
        {
            let grid_lo = self.min_g[0].min(self.min_g[1]).min(self.min_g[2]);
            let grid_hi = (self.min_g[0] + self.span_g[0])
                .max(self.min_g[1] + self.span_g[1])
                .max(self.min_g[2] + self.span_g[2]);
            let clamped = self.observed_max_stops.clamp(grid_lo, grid_hi);
            if clamped > out.alternate_hdr_headroom.to_f32() {
                let candidate = unsigned_fraction_from_f32(clamped);
                if let Some(updated) = GainMapMetadata::try_new(
                    out.channels,
                    out.base_hdr_headroom,
                    candidate,
                    out.application_space,
                ) {
                    out = updated;
                }
            }
        }
        out
    }
}

#[inline]
fn compute_log2_ratio(
    sign: f32,
    base_val: f32,
    alt_val: f32,
    k_base: f32,
    k_alt: f32,
    min_g: f32,
    max_g: f32,
) -> f32 {
    let num = f64::from(alt_val) + f64::from(k_alt);
    let den = f64::from(base_val) + f64::from(k_base);
    if !num.is_finite() || !den.is_finite() {
        return 0.0;
    }
    if (num > 0.0 && den > 0.0) || (num < 0.0 && den < 0.0) {
        let lg = (f64::from(sign) * libm::log2(num / den)) as f32;
        if lg.is_finite() {
            return lg.clamp(-MAX_LOG2_GAIN, MAX_LOG2_GAIN);
        }
    }
    if num == den {
        0.0
    } else if num > den {
        if sign > 0.0 { max_g } else { min_g }
    } else if sign > 0.0 {
        min_g
    } else {
        max_g
    }
}

#[inline(always)]
fn quantize_log2_gain(
    log2_r: f32,
    min_g: f32,
    span_g: f32,
    gamma: f32,
    max_code: u32,
    max_f: f32,
) -> u32 {
    let norm = if span_g.abs() < 1e-12 {
        0.0
    } else {
        ((log2_r - min_g) / span_g).clamp(0.0, 1.0)
    };
    let encoded = if (gamma - 1.0).abs() < 1e-7 {
        norm
    } else {
        libm::powf(norm, gamma)
    };
    ((encoded * max_f + 0.5) as u32).min(max_code)
}

#[inline(always)]
fn write_gm_code(layout: SampleLayout, dst: &mut [u8], elem_idx: usize, comp: usize, code: u32) {
    let packed = layout.packing.pack_code(layout.sample, comp, code);
    match layout.sample {
        SampleType::U8 => dst[elem_idx] = packed as u8,
        SampleType::U16 => {
            let o = elem_idx * 2;
            dst[o..o + 2].copy_from_slice(&(packed as u16).to_ne_bytes());
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Whole-image extraction (zero base copy + subsampling + 2-pass content-fit)
// ---------------------------------------------------------------------------

/// Extract only the numeric gain-map [`PixelBuf`] and exact [`GainMapMetadata`] from
/// borrowed `base_src` and `alternate_src` views without duplicating the base raster.
pub fn extract_gain_map_only(
    base_src: &PixelView<'_>,
    alternate_src: &PixelView<'_>,
    config: &GainMapExtractConfig,
    stop: &dyn Stop,
) -> Result<(PixelBuf, GainMapMetadata), At<ConvertError>> {
    stop.check().map_err(|r| err_at(r.into()))?;
    if config.content_fit {
        let initial_meta = config.to_metadata()?;
        let (_, observed) = extract_gain_map_internal(
            base_src,
            alternate_src,
            initial_meta,
            config.scale_factor.max(1),
            config.downsample,
            config.gain_map_spec()?,
            &config.policy,
            stop,
            true,
        )?;
        let fitted = content_fit_config(config, observed);
        let fitted_meta = fitted.to_metadata()?;
        let (buf, _) = extract_gain_map_internal(
            base_src,
            alternate_src,
            fitted_meta,
            fitted.scale_factor.max(1),
            fitted.downsample,
            fitted.gain_map_spec()?,
            &fitted.policy,
            stop,
            false,
        )?;
        return Ok((buf, fitted_meta));
    }

    let meta = config.to_metadata()?;
    let (buf, _) = extract_gain_map_internal(
        base_src,
        alternate_src,
        meta,
        config.scale_factor.max(1),
        config.downsample,
        config.gain_map_spec()?,
        &config.policy,
        stop,
        false,
    )?;
    Ok((buf, meta))
}

fn content_fit_config(
    config: &GainMapExtractConfig,
    observed: Option<(f32, f32)>,
) -> GainMapExtractConfig {
    let mut out = config.clone();
    let Some((obs_min, obs_max)) = observed else {
        return out;
    };
    if !obs_min.is_finite() || !obs_max.is_finite() {
        return out;
    }
    let hi_bound = config.max_gain_stops;
    let lo_bound = config.min_gain_stops;
    let top = obs_max.clamp(lo_bound, hi_bound);
    let bottom = obs_min.clamp(lo_bound, top);
    let (bottom, top) = if top - bottom < CONTENT_FIT_MIN_SPAN_STOPS {
        let widened_top = (bottom + CONTENT_FIT_MIN_SPAN_STOPS).min(hi_bound);
        if widened_top - bottom >= CONTENT_FIT_MIN_SPAN_STOPS - 1e-6 {
            (bottom, widened_top)
        } else {
            (
                (widened_top - CONTENT_FIT_MIN_SPAN_STOPS).max(lo_bound),
                widened_top,
            )
        }
    } else {
        (bottom, top)
    };
    out.min_gain_stops = bottom;
    out.max_gain_stops = top;
    out
}

/// Extract an ISO 21496-1 [`DualLayerHdr`] at 1:1 resolution from `base_src` and
/// `alternate_src` using exact caller-supplied `metadata` verbatim.
///
/// For subsampled maps, content-fit grids, 10/12-bit maps, or avoiding a base buffer copy,
/// use [`extract_gain_map_only`].
pub fn extract_gain_map(
    base_src: &PixelView<'_>,
    alternate_src: &PixelView<'_>,
    metadata: GainMapMetadata,
) -> Result<DualLayerHdr, At<ConvertError>> {
    let gm_spec = if metadata.channels.is_multichannel() {
        ImageSpec::new(
            SampleLayout::RGB8,
            ColorSpace::Untagged {
                channels: 3,
                alpha: AlphaMode::None,
            },
        )
    } else {
        ImageSpec::new(
            SampleLayout::GRAY8,
            ColorSpace::Untagged {
                channels: 1,
                alpha: AlphaMode::None,
            },
        )
    };
    let (gain_buf, _) = extract_gain_map_internal(
        base_src,
        alternate_src,
        metadata,
        1,
        GainMapDownsample::Point,
        gm_spec,
        &ConvertPolicy::working_linear(),
        &Unstoppable,
        false,
    )?;

    let base_owned = base_src
        .try_to_tight_buf()
        .map_err(|e| err_at((*e.error()).into()))?;
    let alt_color = if metadata.application_space == GainMapColorSpace::Alternate
        || alternate_src.spec().color != base_src.spec().color
    {
        Some(alternate_src.spec().color.clone())
    } else {
        None
    };
    DualLayerHdr::new_with_alternate(
        base_owned,
        gain_buf,
        metadata,
        alt_color,
        Orientation::Identity,
    )
    .map_err(|e| err_at((*e.error()).into()))
}

type ObservedGainRange = Option<(f32, f32)>;

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn extract_gain_map_internal(
    base_src: &PixelView<'_>,
    alternate_src: &PixelView<'_>,
    metadata: GainMapMetadata,
    scale_factor: u8,
    downsample: GainMapDownsample,
    gm_spec: ImageSpec,
    base_policy: &ConvertPolicy,
    stop: &dyn Stop,
    measure_only: bool,
) -> Result<(PixelBuf, ObservedGainRange), At<ConvertError>> {
    stop.check().map_err(|r| err_at(r.into()))?;
    let w = base_src.width();
    let h = base_src.rows();
    if w == 0 || h == 0 || alternate_src.width() != w || alternate_src.rows() != h {
        return Err(err_at(ConvertError::DimensionMismatch));
    }
    if !metadata.is_valid() {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    let _ = metadata
        .direction()
        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;

    let (base_primaries, base_white_opt) = resolve_linear_rgb_primaries(&base_src.spec().color)
        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
    let base_white = base_white_opt
        .or(base_policy.sdr_white)
        .unwrap_or(DiffuseWhite::BT2408);

    let app_primaries = match metadata.application_space {
        GainMapColorSpace::Base => base_primaries,
        GainMapColorSpace::Alternate => resolve_linear_rgb_primaries(&alternate_src.spec().color)
            .map(|(p, _)| p)
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?,
    };

    let linear_rgb = ImageSpec::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: app_primaries,
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: Some(base_white),
        },
    );
    let policy = base_policy
        .clone()
        .with_sdr_white(base_white)
        .with_alpha(AlphaPolicy::DiscardUnchecked);
    let base_plan = RowPlan::build(base_src.spec().clone(), linear_rgb.clone(), &policy, None)?;
    let alt_plan = RowPlan::build(alternate_src.spec().clone(), linear_rgb, &policy, None)?;

    let scale = u32::from(scale_factor.max(1));
    let gm_w = w.div_ceil(scale);
    let gm_h = h.div_ceil(scale);

    let mut extractor = GainMapRowExtractor::new(metadata, app_primaries)?;
    let mut gain_buf = if measure_only {
        PixelBuf::new(1, 1, gm_spec.clone()).map_err(|e| err_at((*e.error()).into()))?
    } else {
        PixelBuf::new(gm_w, gm_h, gm_spec.clone()).map_err(|e| err_at((*e.error()).into()))?
    };

    let row_floats = (w as usize)
        .checked_mul(3)
        .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
    let gm_floats = (gm_w as usize)
        .checked_mul(3)
        .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
    let mut base_row = alloc_f32(row_floats)?;
    let mut alt_row = alloc_f32(row_floats)?;

    let mut dummy_row = Vec::new();
    if measure_only {
        let b = (gm_w as usize)
            .checked_mul(gm_spec.layout.bytes_per_pixel())
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        dummy_row
            .try_reserve_exact(b)
            .map_err(|_| err_at(ConvertError::AllocationFailed))?;
        dummy_row.resize(b, 0u8);
    }

    if scale == 1 {
        let mut gm_v = gain_buf.as_view_mut();
        for y in 0..h {
            stop.check().map_err(|r| err_at(r.into()))?;
            base_plan.convert_row(
                base_src.row(y),
                bytemuck::cast_slice_mut(&mut base_row),
                w,
                y,
            )?;
            alt_plan.convert_row(
                alternate_src.row(y),
                bytemuck::cast_slice_mut(&mut alt_row),
                w,
                y,
            )?;
            let dst_slice = if measure_only {
                &mut dummy_row[..]
            } else {
                gm_v.row_mut(y)
            };
            extractor.encode_row_linear_f32(&base_row, &alt_row, 3, gm_spec.layout, dst_slice)?;
        }
        return Ok((gain_buf, extractor.observed_gain_stops()));
    }

    // Subsampled extraction (scale > 1)
    let mut gm_base_row = alloc_f32(gm_floats)?;
    let mut gm_alt_row = alloc_f32(gm_floats)?;
    let mut gm_v = gain_buf.as_view_mut();

    match downsample {
        GainMapDownsample::Point => {
            for gy in 0..gm_h {
                stop.check().map_err(|r| err_at(r.into()))?;
                let y = (gy * scale + scale / 2).min(h - 1);
                base_plan.convert_row(
                    base_src.row(y),
                    bytemuck::cast_slice_mut(&mut base_row),
                    w,
                    y,
                )?;
                alt_plan.convert_row(
                    alternate_src.row(y),
                    bytemuck::cast_slice_mut(&mut alt_row),
                    w,
                    y,
                )?;
                for gx in 0..gm_w as usize {
                    let sx = ((gx as u32 * scale + scale / 2).min(w - 1)) as usize * 3;
                    let dx = gx * 3;
                    gm_base_row[dx..dx + 3].copy_from_slice(&base_row[sx..sx + 3]);
                    gm_alt_row[dx..dx + 3].copy_from_slice(&alt_row[sx..sx + 3]);
                }
                let dst_slice = if measure_only {
                    &mut dummy_row[..]
                } else {
                    gm_v.row_mut(gy)
                };
                extractor.encode_row_linear_f32(
                    &gm_base_row,
                    &gm_alt_row,
                    3,
                    gm_spec.layout,
                    dst_slice,
                )?;
            }
        }
        GainMapDownsample::Box => {
            for gy in 0..gm_h {
                stop.check().map_err(|r| err_at(r.into()))?;
                let y0 = gy * scale;
                let y1 = (y0 + scale).min(h);
                let rows_in_box = (y1 - y0).max(1) as f32;
                gm_base_row.fill(0.0);
                gm_alt_row.fill(0.0);
                for y in y0..y1 {
                    base_plan.convert_row(
                        base_src.row(y),
                        bytemuck::cast_slice_mut(&mut base_row),
                        w,
                        y,
                    )?;
                    alt_plan.convert_row(
                        alternate_src.row(y),
                        bytemuck::cast_slice_mut(&mut alt_row),
                        w,
                        y,
                    )?;
                    for gx in 0..gm_w {
                        let x0 = gx * scale;
                        let x1 = (x0 + scale).min(w);
                        let inv_x = 1.0 / ((x1 - x0).max(1) as f32);
                        let mut br = 0.0f32;
                        let mut bg = 0.0f32;
                        let mut bb = 0.0f32;
                        let mut ar = 0.0f32;
                        let mut ag = 0.0f32;
                        let mut ab = 0.0f32;
                        for x in x0..x1 {
                            let o = (x as usize) * 3;
                            br += base_row[o];
                            bg += base_row[o + 1];
                            bb += base_row[o + 2];
                            ar += alt_row[o];
                            ag += alt_row[o + 1];
                            ab += alt_row[o + 2];
                        }
                        let d = (gx as usize) * 3;
                        gm_base_row[d] += br * inv_x;
                        gm_base_row[d + 1] += bg * inv_x;
                        gm_base_row[d + 2] += bb * inv_x;
                        gm_alt_row[d] += ar * inv_x;
                        gm_alt_row[d + 1] += ag * inv_x;
                        gm_alt_row[d + 2] += ab * inv_x;
                    }
                }
                let inv_y = 1.0 / rows_in_box;
                for v in &mut gm_base_row {
                    *v *= inv_y;
                }
                for v in &mut gm_alt_row {
                    *v *= inv_y;
                }
                let dst_slice = if measure_only {
                    &mut dummy_row[..]
                } else {
                    gm_v.row_mut(gy)
                };
                extractor.encode_row_linear_f32(
                    &gm_base_row,
                    &gm_alt_row,
                    3,
                    gm_spec.layout,
                    dst_slice,
                )?;
            }
        }
        GainMapDownsample::Bilinear => {
            let hw = AxisWeights::try_new(
                w,
                gm_w,
                GainMapResample::Bilinear,
                MAX_EXTRACT_RESAMPLE_TAPS,
            )
            .map_err(err_at)?;
            let vw = AxisWeights::try_new(
                h,
                gm_h,
                GainMapResample::Bilinear,
                MAX_EXTRACT_RESAMPLE_TAPS,
            )
            .map_err(err_at)?;
            for gy in 0..gm_h {
                stop.check().map_err(|r| err_at(r.into()))?;
                let (y_left, y_ws) = vw.taps(gy as usize);
                gm_base_row.fill(0.0);
                gm_alt_row.fill(0.0);
                for (yt, &wy) in y_ws.iter().enumerate() {
                    let y = y_left + yt as u32;
                    base_plan.convert_row(
                        base_src.row(y),
                        bytemuck::cast_slice_mut(&mut base_row),
                        w,
                        y,
                    )?;
                    alt_plan.convert_row(
                        alternate_src.row(y),
                        bytemuck::cast_slice_mut(&mut alt_row),
                        w,
                        y,
                    )?;
                    for gx in 0..gm_w as usize {
                        let (x_left, x_ws) = hw.taps(gx);
                        let xl = x_left as usize;
                        let mut br = 0.0f32;
                        let mut bg = 0.0f32;
                        let mut bb = 0.0f32;
                        let mut ar = 0.0f32;
                        let mut ag = 0.0f32;
                        let mut ab = 0.0f32;
                        for (xt, &wx) in x_ws.iter().enumerate() {
                            let o = (xl + xt) * 3;
                            br += base_row[o] * wx;
                            bg += base_row[o + 1] * wx;
                            bb += base_row[o + 2] * wx;
                            ar += alt_row[o] * wx;
                            ag += alt_row[o + 1] * wx;
                            ab += alt_row[o + 2] * wx;
                        }
                        let d = gx * 3;
                        gm_base_row[d] += br * wy;
                        gm_base_row[d + 1] += bg * wy;
                        gm_base_row[d + 2] += bb * wy;
                        gm_alt_row[d] += ar * wy;
                        gm_alt_row[d + 1] += ag * wy;
                        gm_alt_row[d + 2] += ab * wy;
                    }
                }
                let dst_slice = if measure_only {
                    &mut dummy_row[..]
                } else {
                    gm_v.row_mut(gy)
                };
                extractor.encode_row_linear_f32(
                    &gm_base_row,
                    &gm_alt_row,
                    3,
                    gm_spec.layout,
                    dst_slice,
                )?;
            }
        }
    }

    Ok((gain_buf, extractor.observed_gain_stops()))
}

fn resolve_linear_rgb_primaries(color: &ColorSpace) -> Option<(Primaries, Option<DiffuseWhite>)> {
    match color {
        ColorSpace::Rgb {
            primaries, white, ..
        } if *primaries != Primaries::Unknown => Some((*primaries, *white)),
        ColorSpace::Gray { white, .. } => Some((Primaries::Bt709, *white)),
        ColorSpace::Icc { profile, .. } => {
            let id = zenpixels::icc::IccIdentification::from_icc(profile)?;
            if id.primaries != Primaries::Unknown {
                Some((id.primaries, None))
            } else {
                None
            }
        }
        _ => None,
    }
}
