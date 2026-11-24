//! ISO 21496-1 / UltraHDR dual-layer gain-map reconstruction, resampling, and extraction.
//!
//! Subsumes pure ISO 21496-1 pixel math in `no_std + alloc`, `#![forbid(unsafe_code)]`
//! with **0-bit `Accelerators::ALL == Accelerators::SCALAR` parity** and O(width) streaming
//! memory:
//!
//! - **Direction-complete ISO 21496-1:** exact rational headroom ordering supports both
//!   **Forward** (SDR base → HDR alternate, `H_alt > H_base`) and **Reverse** (HDR base →
//!   SDR alternate, `H_base > H_alt`) with signed weight, plus exact zero-weight identity
//!   bypass (no map sampling or offset drift when `weight == 0`).
//! - **Color-space & alpha complete:** handles `GainMapColorSpace::Base` and
//!   `GainMapColorSpace::Alternate` (converting base into alternate linear primaries
//!   *before* channelwise gain), ICC-described bases, arbitrary base/destination formats
//!   (`RGBAF32`, `RGBAF16`, `RGB16` PQ/HLG, `RGBA8` sRGB, etc.) via fused [`RowPlan`],
//!   and preserves base alpha untouched.
//! - **Separable resampling (`zenresize` center-sited math):** [`GainMapResample::Bilinear`]
//!   (ISO 21496-1 / Android Ultra HDR default), [`GainMapResample::Bicubic`] (Catmull-Rom),
//!   [`GainMapResample::Mitchell`] (low-ringing Mitchell-Netravali), and
//!   [`GainMapResample::Nearest`], with cached horizontally-upsampled gain rows so ¼-res
//!   maps decode and horizontally filter each gain row only once.
//! - **Fast paths with 0-bit parity:** 1:1 / `Nearest` U8 maps use an exact 256-entry LUT;
//!   upsampled maps use SIMD vertical blending + portable `magemath::Expf` `exp2` + native
//!   SIMD `deinterleave_{3,4}ch` combine.
//! - **Streaming row workers & extraction:** [`GainMapApplier`] for pull/streaming decode,
//!   [`GainMapRowExtractor`] for fused row-level encode (e.g. alongside a tone-map splitter),
//!   and [`extract_gain_map_only`] supporting subsampling (`Box` area-average, `Bilinear`,
//!   `Point`), 2-pass `content_fit` grid narrowing, 1-ch/3-ch, U8/U16/10-bit/12-bit maps,
//!   and cooperative [`enough::Stop`] cancellation.

use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroU32;
use enough::{Stop, Unstoppable};
use whereat::At;
use zenpixels::{
    AlphaMode, BitPacking, ColorSpace, DiffuseWhite, DualLayerHdr, DualLayerHdrView,
    GainMapChannels, GainMapColorSpace, GainMapComponent, GainMapDirection, GainMapMetadata,
    HdrMetadata, ImageSpec, PixelBuf, PixelSink, PixelSource, PixelView, Primaries, SampleFacts,
    SampleLayout, SampleType, SignalRange, SignedFraction, Transfer, UnsignedFraction,
};

use crate::accel::{self, Accelerators};
use crate::error::{ConvertError, err_at};
use crate::gainmap_simd::{
    MAX_LOG2_GAIN, blend_rows_2_scalar, blend_rows_4_scalar, combine_linear_f32_scalar,
    decode_gain_1ch_scalar, exp2_gain_scalar,
};
use crate::plan::RowPlan;
use crate::policy::{AlphaPolicy, ConvertPolicy};

/// Minimum log₂ span (`1/16` stop) enforced by content-fitted extraction grids so uniform
/// images never degenerate to a zero-width quantization range.
pub const CONTENT_FIT_MIN_SPAN_STOPS: f32 = 1.0 / 16.0;

// ---------------------------------------------------------------------------
// Public math & boost helpers
// ---------------------------------------------------------------------------

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
#[inline]
#[must_use]
pub fn full_reconstruction_headroom_stops(metadata: &GainMapMetadata) -> f32 {
    metadata.alternate_hdr_headroom.to_f32()
}

/// Compute the ISO 21496-1 signed application weight for `display_headroom_stops`:
///
/// - Compares exact rational `base_hdr_headroom` vs `alternate_hdr_headroom`.
/// - Returns `None` if `display_headroom_stops` is non-finite or headrooms are equal.
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
    let h_base = metadata.base_hdr_headroom.to_f64();
    let h_alt = metadata.alternate_hdr_headroom.to_f64();
    let amount =
        ((f64::from(display_headroom_stops) - h_base) / (h_alt - h_base)).clamp(0.0, 1.0) as f32;
    if amount == 0.0 {
        return Some(0.0);
    }
    Some(match direction {
        GainMapDirection::Forward => amount,
        GainMapDirection::Reverse => -amount,
    })
}

// ---------------------------------------------------------------------------
// Exact rational encoding helpers
// ---------------------------------------------------------------------------

fn signed_fraction_from_f32(v: f32) -> SignedFraction {
    let v = if v.is_finite() { v } else { 0.0 };
    if v == 0.0 {
        return SignedFraction::ZERO;
    }
    // Preserve exact binary fractions (e.g. 1/64, 1/2, 2.5, -1.25) cleanly.
    for den in [1u32, 2, 4, 8, 16, 32, 64, 128, 256, 1024] {
        let scaled = f64::from(v) * f64::from(den);
        let rounded = libm::round(scaled);
        if (scaled - rounded).abs() <= 1e-7
            && rounded >= f64::from(i32::MIN)
            && rounded <= f64::from(i32::MAX)
        {
            return SignedFraction::new(rounded as i32, NonZeroU32::new(den).unwrap());
        }
    }
    let den = 1_000_000u32;
    let max_abs = (i32::MAX as f64) / f64::from(den);
    let clamped = f64::from(v).clamp(-max_abs, max_abs);
    SignedFraction::new(
        libm::round(clamped * f64::from(den)) as i32,
        NonZeroU32::new(den).unwrap(),
    )
}

fn unsigned_fraction_from_f32(v: f32) -> UnsignedFraction {
    let v = if v.is_finite() && v > 0.0 { v } else { 0.0 };
    if v == 0.0 {
        return UnsignedFraction::ZERO;
    }
    for den in [1u32, 2, 4, 8, 16, 32, 64, 128, 256, 1024] {
        let scaled = f64::from(v) * f64::from(den);
        let rounded = libm::round(scaled);
        if (scaled - rounded).abs() <= 1e-7 && rounded >= 0.0 && rounded <= f64::from(u32::MAX) {
            return UnsignedFraction::new(rounded as u32, NonZeroU32::new(den).unwrap());
        }
    }
    let den = 1_000_000u32;
    let max_val = (u32::MAX as f64) / f64::from(den);
    let clamped = f64::from(v).clamp(0.0, max_val);
    UnsignedFraction::new(
        libm::round(clamped * f64::from(den)) as u32,
        NonZeroU32::new(den).unwrap(),
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

#[inline]
fn resample_window(mode: GainMapResample) -> f64 {
    match mode {
        GainMapResample::Nearest => 0.5,
        GainMapResample::Bilinear => 1.0,
        GainMapResample::Bicubic | GainMapResample::Mitchell => 2.0,
    }
}

/// Precomputed 1D separable resampling weights using `zenresize` pixel-center siting:
/// `center = (out + 0.5) / scale - 0.5`, edge-clamped before kernel evaluation and
/// normalized to exact unit sum in `f64`.
#[derive(Clone, Debug)]
struct AxisWeights {
    left: Vec<u32>,
    tap_counts: Vec<u16>,
    weights_flat: Vec<f32>,
    max_taps: usize,
}

impl AxisWeights {
    fn try_new(in_size: u32, out_size: u32, mode: GainMapResample) -> Result<Self, ConvertError> {
        const WEIGHT_EPS: f64 = 2e-8;
        if in_size == 0 || out_size == 0 {
            return Err(ConvertError::DimensionMismatch);
        }
        let out_n = out_size as usize;

        if mode == GainMapResample::Nearest {
            let mut left = Vec::new();
            left.try_reserve_exact(out_n)
                .map_err(|_| ConvertError::AllocationFailed)?;
            let mut tap_counts = Vec::new();
            tap_counts
                .try_reserve_exact(out_n)
                .map_err(|_| ConvertError::AllocationFailed)?;
            let mut weights_flat = Vec::new();
            weights_flat
                .try_reserve_exact(out_n)
                .map_err(|_| ConvertError::AllocationFailed)?;
            let ratio = f64::from(in_size) / f64::from(out_size);
            for o in 0..out_size {
                let idx = (libm::floor((f64::from(o) + 0.5) * ratio) as i64)
                    .clamp(0, i64::from(in_size - 1)) as u32;
                left.push(idx);
                tap_counts.push(1);
                weights_flat.push(1.0);
            }
            return Ok(Self {
                left,
                tap_counts,
                weights_flat,
                max_taps: 1,
            });
        }

        let scale = f64::from(out_size) / f64::from(in_size);
        let downscale = scale.min(1.0);
        let eff_window = resample_window(mode) / downscale;

        let mut all_left = Vec::new();
        all_left
            .try_reserve_exact(out_n)
            .map_err(|_| ConvertError::AllocationFailed)?;
        let mut all_counts = Vec::new();
        all_counts
            .try_reserve_exact(out_n)
            .map_err(|_| ConvertError::AllocationFailed)?;
        let mut all_weights: Vec<Vec<f64>> = Vec::new();
        all_weights
            .try_reserve_exact(out_n)
            .map_err(|_| ConvertError::AllocationFailed)?;

        let mut max_taps = 1usize;
        let mut tmp: Vec<f64> = Vec::new();

        for o in 0..out_size {
            tmp.clear();
            let center = (f64::from(o) + 0.5) / scale - 0.5;
            let left_edge = libm::ceil(center - eff_window - 1e-4) as i32;
            let right_edge = libm::floor(center + eff_window + 1e-4) as i32;
            let lp = left_edge.max(0) as u32;
            let rp = right_edge.min(in_size as i32 - 1) as u32;

            let mut trimmed_left = lp;
            if lp <= rp {
                let mut total = 0.0f64;
                for inp in lp..=rp {
                    let x = downscale * (f64::from(inp) - center);
                    let mut w = eval_resample_kernel(mode, x);
                    if w.abs() <= WEIGHT_EPS {
                        w = 0.0;
                    }
                    tmp.push(w);
                    total += w;
                }
                if total != 0.0 {
                    let inv = 1.0 / total;
                    for w in &mut tmp {
                        *w *= inv;
                    }
                } else {
                    tmp.clear();
                }
                while tmp.last() == Some(&0.0) {
                    tmp.pop();
                }
                while tmp.first() == Some(&0.0) {
                    tmp.remove(0);
                    trimmed_left += 1;
                }
            }
            if tmp.is_empty() {
                let nearest = (libm::round(center) as i64).clamp(0, i64::from(in_size - 1)) as u32;
                trimmed_left = nearest;
                tmp.push(1.0);
            }
            max_taps = max_taps.max(tmp.len());
            all_left.push(trimmed_left);
            all_counts.push(tmp.len().min(u16::MAX as usize) as u16);
            all_weights.push(tmp.clone());
        }

        let flat_len = out_n
            .checked_mul(max_taps)
            .ok_or(ConvertError::AllocationFailed)?;
        let mut weights_flat = Vec::new();
        weights_flat
            .try_reserve_exact(flat_len)
            .map_err(|_| ConvertError::AllocationFailed)?;
        weights_flat.resize(flat_len, 0.0f32);

        for (i, ws) in all_weights.iter().enumerate() {
            let off = i * max_taps;
            let mut sum = 0.0f32;
            for (j, &w) in ws.iter().enumerate() {
                let wf = w as f32;
                weights_flat[off + j] = wf;
                sum += wf;
            }
            if sum != 0.0 {
                let inv = 1.0f32 / sum;
                for v in &mut weights_flat[off..off + ws.len()] {
                    *v *= inv;
                }
            }
        }

        Ok(Self {
            left: all_left,
            tap_counts: all_counts,
            weights_flat,
            max_taps,
        })
    }

    #[inline(always)]
    fn taps(&self, out_idx: usize) -> (u32, &[f32]) {
        let count = usize::from(self.tap_counts[out_idx]);
        let off = out_idx * self.max_taps;
        (self.left[out_idx], &self.weights_flat[off..off + count])
    }
}

// ---------------------------------------------------------------------------
// Apply configuration & streaming GainMapApplier
// ---------------------------------------------------------------------------

/// Configuration for ISO 21496-1 gain-map reconstruction via [`GainMapApplier`] or
/// [`apply_gain_map_with`].
#[derive(Clone, Debug)]
pub struct GainMapApplyConfig {
    /// Target display HDR headroom in log₂ stops (`0.0` = SDR, `2.0` = 4× HDR, `2.32` ≈ 5×).
    pub display_headroom_stops: f32,
    /// Resampling kernel used when gain-map dimensions differ from base dimensions.
    pub resample: GainMapResample,
    /// Conversion policy controlling SIMD [`Accelerators`], `sdr_white`, and CMS backends.
    pub policy: ConvertPolicy,
    /// Optional target output [`ImageSpec`].
    ///
    /// - `None`: outputs relative-linear `RGBF32` (or `RGBAF32` when base has alpha) in the
    ///   resolved gain-map application primaries.
    /// - `Some(spec)`: fuses conversion directly to `spec` (e.g. `RGBAF16`, `RGB16` PQ,
    ///   `RGB16` HLG, `RGBA8` sRGB, `DisplayP3`) in O(width) row memory with zero full-frame
    ///   intermediate buffer!
    pub dst_spec: Option<ImageSpec>,
}

impl GainMapApplyConfig {
    /// Create a default configuration for `display_headroom_stops` (log₂ stops).
    #[inline]
    #[must_use]
    pub fn new(display_headroom_stops: f32) -> Self {
        Self {
            display_headroom_stops,
            resample: GainMapResample::Bilinear,
            policy: ConvertPolicy::working_linear(),
            dst_spec: None,
        }
    }

    /// Create a configuration from a linear display boost ratio (e.g. `4.0` → `2.0` stops).
    #[inline]
    #[must_use]
    pub fn from_display_boost(display_boost: f32) -> Self {
        Self::new(display_boost_to_headroom_stops(display_boost))
    }

    /// Select the gain-map resampling filter (`Bilinear`, `Bicubic`, `Mitchell`, `Nearest`).
    #[inline]
    #[must_use]
    pub fn with_resample(mut self, resample: GainMapResample) -> Self {
        self.resample = resample;
        self
    }

    /// Set the conversion policy (accelerators, `sdr_white`, CMS backend, HLG rendering).
    #[inline]
    #[must_use]
    pub fn with_policy(mut self, policy: ConvertPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Fuse row conversion straight into `dst_spec` (e.g. `RGBAF16`, `RGB16` PQ, `RGBA8` sRGB).
    #[inline]
    #[must_use]
    pub fn with_dst_spec(mut self, dst_spec: ImageSpec) -> Self {
        self.dst_spec = Some(dst_spec);
        self
    }
}

#[derive(Clone, Debug)]
struct CachedHorizRow {
    gm_y: u32,
    valid: bool,
    /// Planar horizontally-upsampled gain values: `[ch0: 0..base_w, ch1: ..., ch2: ...]`.
    planes: Vec<f32>,
}

/// Reusable, O(width)-memory, row-streaming ISO 21496-1 gain-map reconstructor.
///
/// Replaces `ultrahdr-core`'s `RowDecoder`, `StreamDecoder`, `GainMapLut`, `ShepardsLut`,
/// and `apply_gain_row_*` with one unified, direction-complete, SIMD-accelerated engine:
///
/// - Handles Forward (SDR→HDR) and Reverse (HDR→SDR) gain maps via exact rational headroom
///   ordering and signed weight.
/// - Bypasses map sampling and offsets completely when `signed_weight == 0.0`.
/// - Caches horizontally-upsampled gain rows across consecutive output scanlines so
///   ¼-res maps decode and horizontally filter each gain row only once.
/// - Fuses base decode (`base_to_app`) and optional destination encode (`app_to_dst`) in
///   O(width) scratch memory.
#[derive(Clone, Debug)]
pub struct GainMapApplier {
    base_w: u32,
    base_h: u32,
    gm_w: u32,
    gm_h: u32,
    gm_spec: ImageSpec,
    gm_ch: usize,
    gm_max: [f32; 3],
    has_alpha: bool,
    dst_spec: ImageSpec,
    accel: Accelerators,

    signed_weight: f32,
    is_zero_weight: bool,
    same_size: bool,
    resample: GainMapResample,
    shared_meta: bool,

    min_g: [f32; 3],
    span_g: [f32; 3],
    inv_gamma: [f32; 3],
    k_base: [f32; 3],
    k_alt: [f32; 3],

    /// Exact 256-entry gain LUT used when gain map is full-packing U8 and 1:1 or `Nearest`.
    u8_gain_lut: Option<[[f32; 256]; 3]>,

    base_to_app: RowPlan,
    app_to_dst: Option<RowPlan>,
    zero_weight_direct: Option<RowPlan>,

    h_weights: Option<AxisWeights>,
    v_weights: Option<AxisWeights>,
    cache: Vec<CachedHorizRow>,
    cache_cursor: usize,
    v_slot_indices: Vec<usize>,

    // Reusable O(width) row scratch buffers
    base_lin_row: Vec<f32>,
    out_lin_row: Vec<f32>,
    gm_decode_scratch: Vec<f32>,
    v_blend_planes: [Vec<f32>; 3],
    gains_planes: [Vec<f32>; 3],
}

impl GainMapApplier {
    /// Construct a streaming applier directly from a validated [`DualLayerHdrView`].
    pub fn for_view(
        dual: &DualLayerHdrView<'_>,
        config: &GainMapApplyConfig,
    ) -> Result<Self, At<ConvertError>> {
        Self::new(
            dual.base().spec().clone(),
            dual.base().width(),
            dual.base().rows(),
            dual.gain_map().spec().clone(),
            dual.gain_map().width(),
            dual.gain_map().rows(),
            dual.metadata(),
            dual.alternate_color_space(),
            config,
        )
    }

    /// Construct a streaming applier from explicit layer specs, dimensions, and metadata.
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
        if base_w == 0 || base_h == 0 || gm_w == 0 || gm_h == 0 {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        base_spec
            .validate()
            .map_err(|e| err_at((*e.error()).into()))?;
        gm_spec
            .validate()
            .map_err(|e| err_at((*e.error()).into()))?;
        if !metadata.is_valid() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        let sw = signed_weight(&metadata, config.display_headroom_stops)
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        validate_gain_map_spec(&gm_spec)?;

        let (_, base_white_opt) =
            resolve_linear_rgb_primaries(&base_spec.color).unwrap_or((Primaries::Bt709, None));
        let base_white = base_white_opt
            .or(config.policy.sdr_white)
            .unwrap_or(DiffuseWhite::BT2408);

        let app_primaries = match metadata.application_space {
            GainMapColorSpace::Base => resolve_linear_rgb_primaries(&base_spec.color)
                .map(|(p, _)| p)
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?,
            GainMapColorSpace::Alternate => {
                let alt =
                    alternate_color_space.ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
                resolve_linear_rgb_primaries(alt)
                    .map(|(p, _)| p)
                    .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?
            }
        };

        let has_alpha = base_spec.layout.order.has_alpha_slot();
        let ch = if has_alpha { 4usize } else { 3usize };
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
                alpha: if has_alpha {
                    AlphaMode::Straight
                } else {
                    AlphaMode::None
                },
                white: Some(base_white),
            },
        );

        let mut row_pol = config.policy.clone();
        row_pol.sdr_white = Some(base_white);
        let accel = accel::resolve(row_pol.accelerators);

        let base_to_app = RowPlan::build(base_spec.clone(), app_spec.clone(), &row_pol, None)?;
        let (dst_spec, app_to_dst, zero_weight_direct) = if let Some(target) = &config.dst_spec {
            let a2d = RowPlan::build(app_spec.clone(), target.clone(), &row_pol, None)?;
            let direct = if sw == 0.0 {
                RowPlan::build(base_spec, target.clone(), &row_pol, None).ok()
            } else {
                None
            };
            (target.clone(), Some(a2d), direct)
        } else {
            (app_spec, None, None)
        };

        let gm_layout = gm_spec.layout;
        let gm_ch = gm_layout.channels();
        let gm_max = [
            gm_layout.packing.max_code(gm_layout.sample, 0).max(1) as f32,
            gm_layout.packing.max_code(gm_layout.sample, 1).max(1) as f32,
            gm_layout.packing.max_code(gm_layout.sample, 2).max(1) as f32,
        ];

        let mut min_g = [0.0f32; 3];
        let mut span_g = [0.0f32; 3];
        let mut inv_gamma = [1.0f32; 3];
        let mut k_base = [0.0f32; 3];
        let mut k_alt = [0.0f32; 3];
        for c in 0..3 {
            let comp = metadata.channel(c);
            min_g[c] = comp.min_gain_stops.to_f32();
            span_g[c] = comp.max_gain_stops.to_f32() - min_g[c];
            inv_gamma[c] = 1.0 / comp.gamma.to_f32();
            k_base[c] = comp.base_offset.to_f32();
            k_alt[c] = comp.alternate_offset.to_f32();
        }
        let shared_meta = matches!(metadata.channels, GainMapChannels::Shared(_));
        let same_size = base_w == gm_w && base_h == gm_h;
        let is_zero_weight = sw == 0.0;

        // Exact 256-entry LUT for full-packing U8 gain maps when no fractional sample
        // interpolation is performed (1:1 same-size OR Nearest resampling).
        let use_u8_lut = !is_zero_weight
            && gm_layout.sample == SampleType::U8
            && gm_layout.packing.is_full()
            && (same_size || config.resample == GainMapResample::Nearest);
        let u8_gain_lut = if use_u8_lut {
            let mut lut = [[1.0f32; 256]; 3];
            for c in 0..3 {
                let unit_gamma = (inv_gamma[c] - 1.0).abs() < 1e-7;
                for (i, slot) in lut[c].iter_mut().enumerate() {
                    let s = (i as f32) / 255.0;
                    let g = if unit_gamma || i == 0 || i == 255 {
                        s
                    } else {
                        libm::powf(s, inv_gamma[c])
                    };
                    let lg = ((min_g[c] + span_g[c] * g) * sw).clamp(-MAX_LOG2_GAIN, MAX_LOG2_GAIN);
                    *slot = exp2_gain_scalar(lg);
                }
            }
            Some(lut)
        } else {
            None
        };

        let (h_weights, v_weights, cache_slots) = if is_zero_weight || same_size {
            (None, None, 0usize)
        } else if use_u8_lut {
            // Nearest U8 LUT needs only index mapping, zero float row cache!
            let hw =
                AxisWeights::try_new(gm_w, base_w, GainMapResample::Nearest).map_err(err_at)?;
            let vw =
                AxisWeights::try_new(gm_h, base_h, GainMapResample::Nearest).map_err(err_at)?;
            (Some(hw), Some(vw), 0usize)
        } else {
            let hw = AxisWeights::try_new(gm_w, base_w, config.resample).map_err(err_at)?;
            let vw = AxisWeights::try_new(gm_h, base_h, config.resample).map_err(err_at)?;
            let slots = vw.max_taps.max(1);
            (Some(hw), Some(vw), slots)
        };

        let w_usize = base_w as usize;
        let row_floats = w_usize
            .checked_mul(ch)
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;

        let alloc_f32 = |n: usize| -> Result<Vec<f32>, At<ConvertError>> {
            let mut v = Vec::new();
            v.try_reserve_exact(n)
                .map_err(|_| err_at(ConvertError::AllocationFailed))?;
            v.resize(n, 0.0f32);
            Ok(v)
        };

        let base_lin_row = if is_zero_weight && zero_weight_direct.is_some() {
            Vec::new()
        } else {
            alloc_f32(row_floats)?
        };
        let out_lin_row = if app_to_dst.is_some() && !is_zero_weight {
            alloc_f32(row_floats)?
        } else {
            Vec::new()
        };

        let gm_decode_len = if is_zero_weight || use_u8_lut {
            0
        } else {
            (gm_w as usize)
                .checked_mul(gm_ch)
                .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?
        };
        let gm_decode_scratch = if gm_decode_len > 0 {
            alloc_f32(gm_decode_len)?
        } else {
            Vec::new()
        };

        let mut cache = Vec::new();
        let mut v_slot_indices = Vec::new();
        if cache_slots > 0 {
            let plane_len = w_usize
                .checked_mul(gm_ch)
                .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
            cache
                .try_reserve_exact(cache_slots)
                .map_err(|_| err_at(ConvertError::AllocationFailed))?;
            for _ in 0..cache_slots {
                cache.push(CachedHorizRow {
                    gm_y: u32::MAX,
                    valid: false,
                    planes: alloc_f32(plane_len)?,
                });
            }
            v_slot_indices
                .try_reserve_exact(cache_slots)
                .map_err(|_| err_at(ConvertError::AllocationFailed))?;
            v_slot_indices.resize(cache_slots, 0);
        }

        let need_sample_planes = !is_zero_weight && !use_u8_lut;
        let v_blend_planes = [
            if need_sample_planes {
                alloc_f32(w_usize)?
            } else {
                Vec::new()
            },
            if need_sample_planes && gm_ch == 3 {
                alloc_f32(w_usize)?
            } else {
                Vec::new()
            },
            if need_sample_planes && gm_ch == 3 {
                alloc_f32(w_usize)?
            } else {
                Vec::new()
            },
        ];

        let need_3_gains = gm_ch == 3 || !shared_meta;
        let gains_planes = [
            if !is_zero_weight {
                alloc_f32(w_usize)?
            } else {
                Vec::new()
            },
            if !is_zero_weight && need_3_gains {
                alloc_f32(w_usize)?
            } else {
                Vec::new()
            },
            if !is_zero_weight && need_3_gains {
                alloc_f32(w_usize)?
            } else {
                Vec::new()
            },
        ];

        Ok(Self {
            base_w,
            base_h,
            gm_w,
            gm_h,
            gm_spec,
            gm_ch,
            gm_max,
            has_alpha,
            dst_spec,
            accel,
            signed_weight: sw,
            is_zero_weight,
            same_size,
            resample: config.resample,
            shared_meta,
            min_g,
            span_g,
            inv_gamma,
            k_base,
            k_alt,
            u8_gain_lut,
            base_to_app,
            app_to_dst,
            zero_weight_direct,
            h_weights,
            v_weights,
            cache,
            cache_cursor: 0,
            v_slot_indices,
            base_lin_row,
            out_lin_row,
            gm_decode_scratch,
            v_blend_planes,
            gains_planes,
        })
    }

    /// Output [`ImageSpec`] produced by this applier.
    #[inline]
    #[must_use]
    pub fn dst_spec(&self) -> &ImageSpec {
        &self.dst_spec
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
        for slot in &mut self.cache {
            slot.valid = false;
        }
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
        self.apply_row_with_provider(y, base_row, |gy| gain_map.row(gy), dst_row)
    }

    /// Reconstruct row `y` while lazily pulling gain-map rows `gy` on demand via `fetch_gm_row`.
    ///
    /// On resampled paths (`Bilinear`, `Bicubic`, `Mitchell`), horizontally-upsampled gain rows
    /// are cached across consecutive `y` so `fetch_gm_row` is invoked at most once per unique
    /// gain-map row during sequential top-to-bottom decode. On 1:1 and `Nearest` full-packing U8
    /// LUT paths, `fetch_gm_row` borrows the raw gain-map row directly with zero row cache.
    pub fn apply_row_with_provider<'a, F>(
        &mut self,
        y: u32,
        base_row: &[u8],
        mut fetch_gm_row: F,
        dst_row: &mut [u8],
    ) -> Result<(), At<ConvertError>>
    where
        F: FnMut(u32) -> &'a [u8],
    {
        if y >= self.base_h {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        let w = self.base_w;
        let need_dst = (w as usize) * self.dst_spec.layout.bytes_per_pixel();
        if dst_row.len() < need_dst {
            return Err(err_at(ConvertError::DimensionMismatch));
        }

        // Fast exact identity bypass at zero weight: no gain-map fetch, no offset drift!
        if self.is_zero_weight {
            if let Some(direct) = &self.zero_weight_direct {
                return direct.convert_row(base_row, dst_row, w, y);
            }
            if let Some(a2d) = &self.app_to_dst {
                self.base_to_app.convert_row(
                    base_row,
                    bytemuck::cast_slice_mut(&mut self.base_lin_row),
                    w,
                    y,
                )?;
                return a2d.convert_row(bytemuck::cast_slice(&self.base_lin_row), dst_row, w, y);
            }
            return self.base_to_app.convert_row(base_row, dst_row, w, y);
        }

        // 1. Decode base row into linear RGB(A) in application primaries.
        self.base_to_app.convert_row(
            base_row,
            bytemuck::cast_slice_mut(&mut self.base_lin_row),
            w,
            y,
        )?;

        // 2. Obtain linear gain multipliers into `self.gains_planes`.
        if let Some(lut) = self.u8_gain_lut {
            self.fill_gains_u8_lut(y, &lut, &mut fetch_gm_row);
        } else {
            self.fill_samples_for_row(y, &mut fetch_gm_row);
            self.decode_samples_to_gains();
        }

        // 3. Apply ISO 21496-1 gain formula `(base + k_base) * gain - k_alt`.
        let w_u = w as usize;
        let need_3 = self.gm_ch == 3 || !self.shared_meta;
        let gr = &self.gains_planes[0][..w_u];
        let gg = if need_3 {
            &self.gains_planes[1][..w_u]
        } else {
            gr
        };
        let gb = if need_3 {
            &self.gains_planes[2][..w_u]
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
            a2d.convert_row(bytemuck::cast_slice(&self.out_lin_row), dst_row, w, y)?;
        } else if let Ok(dst_f32) =
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut dst_row[..need_dst])
        {
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
            // Unaligned destination byte slice fallback (zero-UB safe).
            if self.out_lin_row.len() < self.base_lin_row.len() {
                self.out_lin_row
                    .try_reserve_exact(self.base_lin_row.len() - self.out_lin_row.len())
                    .map_err(|_| err_at(ConvertError::AllocationFailed))?;
                self.out_lin_row.resize(self.base_lin_row.len(), 0.0f32);
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
            dst_row[..need_dst].copy_from_slice(bytemuck::cast_slice(&self.out_lin_row));
        }

        Ok(())
    }

    /// Stream an entire image from any [`PixelSource`] + borrowed `gain_map` into any
    /// [`PixelSink`] (`PixelBuf`, `PixelViewMut`, `BandedBuf`) in O(width) scratch memory,
    /// checking `stop` once per row.
    pub fn apply_into<S: PixelSource + ?Sized, D: PixelSink + ?Sized>(
        &mut self,
        base: &S,
        gain_map: &PixelView<'_>,
        dst: &mut D,
        stop: &dyn Stop,
    ) -> Result<(), At<ConvertError>> {
        if base.source_width() != self.base_w
            || base.source_height() != self.base_h
            || dst.sink_width() != self.base_w
            || dst.sink_height() != self.base_h
            || dst.sink_spec() != &self.dst_spec
        {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        self.reset_cache();
        for y in 0..self.base_h {
            stop.check().map_err(|r| err_at(r.into()))?;
            self.apply_row(y, base.source_row(y), gain_map, dst.sink_row_mut(y))?;
        }
        dst.sink_finish(SampleFacts::NONE, HdrMetadata::default());
        Ok(())
    }

    fn fill_gains_u8_lut<'a, F>(&mut self, y: u32, lut: &[[f32; 256]; 3], fetch_gm_row: &mut F)
    where
        F: FnMut(u32) -> &'a [u8],
    {
        let gy = if let Some(vw) = &self.v_weights {
            vw.left[y as usize]
        } else {
            y
        };
        let row = fetch_gm_row(gy);
        let w = self.base_w as usize;
        let need_3 = self.gm_ch == 3 || !self.shared_meta;

        if self.same_size {
            if self.gm_ch == 1 {
                if need_3 {
                    let [g0, g1, g2] = &mut self.gains_planes;
                    for x in 0..w {
                        let b = usize::from(row[x]);
                        g0[x] = lut[0][b];
                        g1[x] = lut[1][b];
                        g2[x] = lut[2][b];
                    }
                } else {
                    let g0 = &mut self.gains_planes[0][..w];
                    for x in 0..w {
                        g0[x] = lut[0][usize::from(row[x])];
                    }
                }
            } else {
                let [g0, g1, g2] = &mut self.gains_planes;
                for x in 0..w {
                    let o = x * 3;
                    g0[x] = lut[0][usize::from(row[o])];
                    g1[x] = lut[1][usize::from(row[o + 1])];
                    g2[x] = lut[2][usize::from(row[o + 2])];
                }
            }
        } else if let Some(hw) = &self.h_weights {
            if self.gm_ch == 1 {
                if need_3 {
                    let [g0, g1, g2] = &mut self.gains_planes;
                    for x in 0..w {
                        let b = usize::from(row[hw.left[x] as usize]);
                        g0[x] = lut[0][b];
                        g1[x] = lut[1][b];
                        g2[x] = lut[2][b];
                    }
                } else {
                    for (out, &gx) in self.gains_planes[0][..w]
                        .iter_mut()
                        .zip(hw.left[..w].iter())
                    {
                        *out = lut[0][usize::from(row[gx as usize])];
                    }
                }
            } else {
                let [g0, g1, g2] = &mut self.gains_planes;
                for x in 0..w {
                    let o = (hw.left[x] as usize) * 3;
                    g0[x] = lut[0][usize::from(row[o])];
                    g1[x] = lut[1][usize::from(row[o + 1])];
                    g2[x] = lut[2][usize::from(row[o + 2])];
                }
            }
        }
    }

    fn ensure_horiz_row<'a, F>(
        &mut self,
        gm_y: u32,
        keep_start: u32,
        keep_end: u32,
        fetch_gm_row: &mut F,
    ) -> usize
    where
        F: FnMut(u32) -> &'a [u8],
    {
        for (idx, slot) in self.cache.iter().enumerate() {
            if slot.valid && slot.gm_y == gm_y {
                return idx;
            }
        }
        // Pick an invalid slot first; otherwise scan from `cache_cursor` for a slot outside
        // the active `[keep_start, keep_end)` range so taps needed by the current output row
        // can never be evicted even under non-monotonic row access.
        let n_slots = self.cache.len();
        let mut slot_idx = self.cache_cursor;
        if let Some(empty) = self.cache.iter().position(|s| !s.valid) {
            slot_idx = empty;
        } else {
            for step in 0..n_slots {
                let cand = (self.cache_cursor + step) % n_slots;
                let gy = self.cache[cand].gm_y;
                if gy < keep_start || gy >= keep_end {
                    slot_idx = cand;
                    break;
                }
            }
        }
        self.cache_cursor = (slot_idx + 1) % n_slots;

        let raw_row = fetch_gm_row(gm_y);
        decode_gm_row_planar(
            raw_row,
            self.gm_w as usize,
            self.gm_spec.layout,
            self.gm_max,
            &mut self.gm_decode_scratch,
        );

        let w = self.base_w as usize;
        let gm_w = self.gm_w as usize;
        let hw = self.h_weights.as_ref().unwrap();
        let dst_planes = &mut self.cache[slot_idx].planes;

        for c in 0..self.gm_ch {
            let src_plane = &self.gm_decode_scratch[c * gm_w..(c + 1) * gm_w];
            let dst_plane = &mut dst_planes[c * w..(c + 1) * w];
            if self.resample == GainMapResample::Nearest {
                for (x, out) in dst_plane.iter_mut().enumerate() {
                    *out = src_plane[hw.left[x] as usize];
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
        self.cache[slot_idx].gm_y = gm_y;
        self.cache[slot_idx].valid = true;
        slot_idx
    }

    fn fill_samples_for_row<'a, F>(&mut self, y: u32, fetch_gm_row: &mut F)
    where
        F: FnMut(u32) -> &'a [u8],
    {
        let w = self.base_w as usize;
        if self.same_size {
            let raw = fetch_gm_row(y);
            decode_gm_row_planar(
                raw,
                w,
                self.gm_spec.layout,
                self.gm_max,
                &mut self.gm_decode_scratch,
            );
            for c in 0..self.gm_ch {
                self.v_blend_planes[c][..w]
                    .copy_from_slice(&self.gm_decode_scratch[c * w..(c + 1) * w]);
            }
            return;
        }

        let (v_left, v_count) = {
            let vw = self.v_weights.as_ref().unwrap();
            let (l, ws) = vw.taps(y as usize);
            (l, ws.len())
        };
        let keep_end = v_left + v_count as u32;

        // Ensure all required gain-map rows are horizontally resampled & cached (zero allocs).
        for t in 0..v_count {
            let idx = self.ensure_horiz_row(v_left + t as u32, v_left, keep_end, fetch_gm_row);
            self.v_slot_indices[t] = idx;
        }

        let vw = self.v_weights.as_ref().unwrap();
        let (_, ws) = vw.taps(y as usize);
        let blend2 = self.accel.gainmap_blend_2.unwrap_or(blend_rows_2_scalar);
        let blend4 = self.accel.gainmap_blend_4.unwrap_or(blend_rows_4_scalar);

        for c in 0..self.gm_ch {
            let co = c * w;
            let dst = &mut self.v_blend_planes[c][..w];
            match v_count {
                1 => {
                    let s0 = &self.cache[self.v_slot_indices[0]].planes[co..co + w];
                    for (o, &v) in dst.iter_mut().zip(s0.iter()) {
                        *o = v.clamp(0.0, 1.0);
                    }
                }
                2 => {
                    let s0 = &self.cache[self.v_slot_indices[0]].planes[co..co + w];
                    let s1 = &self.cache[self.v_slot_indices[1]].planes[co..co + w];
                    blend2(s0, s1, ws[0], ws[1], dst);
                }
                4 => {
                    let s0 = &self.cache[self.v_slot_indices[0]].planes[co..co + w];
                    let s1 = &self.cache[self.v_slot_indices[1]].planes[co..co + w];
                    let s2 = &self.cache[self.v_slot_indices[2]].planes[co..co + w];
                    let s3 = &self.cache[self.v_slot_indices[3]].planes[co..co + w];
                    blend4(s0, s1, s2, s3, [ws[0], ws[1], ws[2], ws[3]], dst);
                }
                _ => {
                    dst.fill(0.0);
                    for (t, &wt) in ws.iter().enumerate() {
                        let sr = &self.cache[self.v_slot_indices[t]].planes[co..co + w];
                        for (o, &v) in dst.iter_mut().zip(sr.iter()) {
                            *o += v * wt;
                        }
                    }
                    for o in dst.iter_mut() {
                        *o = if *o > 0.0 {
                            if *o < 1.0 { *o } else { 1.0 }
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
    }

    fn decode_samples_to_gains(&mut self) {
        let w = self.base_w as usize;
        let decode_fn = self.accel.gainmap_decode.unwrap_or(decode_gain_1ch_scalar);
        let need_3 = self.gm_ch == 3 || !self.shared_meta;

        decode_fn(
            &self.v_blend_planes[0][..w],
            self.min_g[0],
            self.span_g[0],
            self.signed_weight,
            self.inv_gamma[0],
            &mut self.gains_planes[0][..w],
        );
        if need_3 {
            let s1 = if self.gm_ch == 3 {
                &self.v_blend_planes[1][..w]
            } else {
                &self.v_blend_planes[0][..w]
            };
            let s2 = if self.gm_ch == 3 {
                &self.v_blend_planes[2][..w]
            } else {
                &self.v_blend_planes[0][..w]
            };
            decode_fn(
                s1,
                self.min_g[1],
                self.span_g[1],
                self.signed_weight,
                self.inv_gamma[1],
                &mut self.gains_planes[1][..w],
            );
            decode_fn(
                s2,
                self.min_g[2],
                self.span_g[2],
                self.signed_weight,
                self.inv_gamma[2],
                &mut self.gains_planes[2][..w],
            );
        }
    }
}

/// Decode one gain-map scanline (`U8` / `U16` / `F32`, full or `Lsb`/`Msb` bit-packed,
/// 1-ch or 3-ch) into planar `[0, 1]` f32 channels inside `dst_planar` (`ch * width`).
fn decode_gm_row_planar(
    row: &[u8],
    width: usize,
    layout: SampleLayout,
    max_code: [f32; 3],
    dst_planar: &mut [f32],
) {
    let ch = layout.channels();
    let inv = [1.0 / max_code[0], 1.0 / max_code[1], 1.0 / max_code[2]];
    match (layout.sample, layout.packing.is_full(), ch) {
        (SampleType::U8, true, 1) => {
            let inv0 = inv[0];
            for (dst, &b) in dst_planar[..width].iter_mut().zip(row[..width].iter()) {
                *dst = f32::from(b) * inv0;
            }
        }
        (SampleType::U8, true, 3) => {
            let (p0, rest) = dst_planar.split_at_mut(width);
            let (p1, p2) = rest.split_at_mut(width);
            for (x, px) in row[..width * 3].as_chunks::<3>().0.iter().enumerate() {
                p0[x] = f32::from(px[0]) * inv[0];
                p1[x] = f32::from(px[1]) * inv[1];
                p2[x] = f32::from(px[2]) * inv[2];
            }
        }
        (SampleType::U8, false, _) => {
            for x in 0..width {
                for c in 0..ch {
                    let code =
                        layout
                            .packing
                            .extract_code(SampleType::U8, u32::from(row[x * ch + c]), c);
                    dst_planar[c * width + x] = (code as f32) * inv[c];
                }
            }
        }
        (SampleType::U16, true, _) => {
            for x in 0..width {
                for c in 0..ch {
                    let o = (x * ch + c) * 2;
                    let v = u16::from_ne_bytes([row[o], row[o + 1]]);
                    dst_planar[c * width + x] = f32::from(v) * inv[c];
                }
            }
        }
        (SampleType::U16, false, _) => {
            for x in 0..width {
                for c in 0..ch {
                    let o = (x * ch + c) * 2;
                    let w16 = u32::from(u16::from_ne_bytes([row[o], row[o + 1]]));
                    let code = layout.packing.extract_code(SampleType::U16, w16, c);
                    dst_planar[c * width + x] = (code as f32) * inv[c];
                }
            }
        }
        (SampleType::F32, _, _) => {
            for x in 0..width {
                for c in 0..ch {
                    let o = (x * ch + c) * 4;
                    let v = f32::from_ne_bytes([row[o], row[o + 1], row[o + 2], row[o + 3]]);
                    dst_planar[c * width + x] = if v.is_finite() { v } else { 0.0 };
                }
            }
        }
        _ => {
            dst_planar[..width * ch].fill(0.0);
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
    let ok_ch = matches!(spec.layout.channels(), 1 | 3);
    if !ok_color || !ok_sample || !ok_packing || !ok_ch {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// High-level apply convenience functions
// ---------------------------------------------------------------------------

/// Reconstruct a linear-light [`PixelBuf`] (`RGBF32` / `RGBAF32`) from an owned
/// [`DualLayerHdr`] at `display_headroom_stops` (log₂ stops, e.g. `0.0` = SDR,
/// `2.32` ≈ 5× / 1000-nit HDR display).
pub fn apply_gain_map(
    dual: &DualLayerHdr,
    display_headroom_stops: f32,
) -> Result<PixelBuf, At<ConvertError>> {
    apply_gain_map_view(
        &dual.as_view(),
        display_headroom_stops,
        &ConvertPolicy::working_linear(),
    )
}

/// Reconstruct a linear-light [`PixelBuf`] (`RGBF32` / `RGBAF32`) from a zero-copy
/// [`DualLayerHdrView`] using `policy` (allowing CMS backends and SIMD selection).
pub fn apply_gain_map_view(
    dual: &DualLayerHdrView<'_>,
    display_headroom_stops: f32,
    policy: &ConvertPolicy,
) -> Result<PixelBuf, At<ConvertError>> {
    let cfg = GainMapApplyConfig::new(display_headroom_stops).with_policy(policy.clone());
    apply_gain_map_with(dual, &cfg, &Unstoppable)
}

/// Reconstruct a [`PixelBuf`] from a zero-copy [`DualLayerHdrView`] with full control over
/// resampling kernel, conversion policy, optional target [`ImageSpec`] (e.g. `RGBAF16`,
/// `RGB16` PQ, `RGBA8` sRGB), and cooperative [`enough::Stop`] cancellation.
pub fn apply_gain_map_with(
    dual: &DualLayerHdrView<'_>,
    config: &GainMapApplyConfig,
    stop: &dyn Stop,
) -> Result<PixelBuf, At<ConvertError>> {
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
///
/// All gain and headroom fields are in **log₂ stops** (matching ISO 21496-1 &
/// [`GainMapMetadata`]). Ergonomic linear-boost builders
/// ([`with_boost_range_linear`](Self::with_boost_range_linear),
/// [`with_headroom_boost_linear`](Self::with_headroom_boost_linear)) convert linear
/// ratios (`4.0` → `2.0` stops) once at construction time.
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
    /// (minimum span [`CONTENT_FIT_MIN_SPAN_STOPS`]), spending the full code space on
    /// gains that actually occur while declaring the exact narrowed grid in metadata.
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
///
/// Directly replaces `ultrahdr-core::compute_gain_row` / `RowEncoder` so callers such as
/// `zenjpeg` can fuse tone-map splitting + gain quantization in a single row pass while
/// tracking observed content gain bounds and building contract-exact [`GainMapMetadata`].
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
    /// 1-ch for `Shared` or 3-ch for `PerChannel`).
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
        let pixels = (base_linear.len() / channels).min(alt_linear.len() / channels);
        let gm_ch = if self.multichannel { 3usize } else { 1usize };
        if gm_layout.channels() != gm_ch {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        let need_bytes = pixels * gm_layout.bytes_per_pixel();
        if dst_row.len() < need_bytes {
            return Err(err_at(ConvertError::DimensionMismatch));
        }

        let max_code = gm_layout.packing.max_code(gm_layout.sample, 0).max(1);
        let max_f = max_code as f32;

        if self.multichannel {
            for x in 0..pixels {
                let o = x * channels;
                for c in 0..3 {
                    let b = base_linear[o + c].max(0.0);
                    let a = alt_linear[o + c].max(0.0);
                    let log2_r = self.sign
                        * libm::log2f(
                            (a + self.k_alt[c]).max(1e-6) / (b + self.k_base[c]).max(1e-6),
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
            for x in 0..pixels {
                let o = x * channels;
                let lb = (yr * base_linear[o] + yg * base_linear[o + 1] + yb * base_linear[o + 2])
                    .max(0.0);
                let la =
                    (yr * alt_linear[o] + yg * alt_linear[o + 1] + yb * alt_linear[o + 2]).max(0.0);
                let log2_r = self.sign
                    * libm::log2f((la + self.k_alt[0]).max(1e-6) / (lb + self.k_base[0]).max(1e-6));
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

    /// Return the exact [`GainMapMetadata`] matching the encoded bytes (never narrows the
    /// declared quantization grid behind the bytes' back; preserves exact metadata).
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
///
/// Supports:
/// - Subsampling (`config.scale_factor`, `div_ceil` output dimensions) with
///   [`GainMapDownsample::Box`] (area average), [`GainMapDownsample::Bilinear`], or
///   [`GainMapDownsample::Point`] (legacy center-pixel).
/// - 2-pass [`content_fit`](GainMapExtractConfig::content_fit) grid narrowing to the
///   content's observed gain range (minimum span [`CONTENT_FIT_MIN_SPAN_STOPS`]).
/// - 1-channel (`GRAY8`/`GRAY16`/`Lsb`) and 3-channel (`RGB8`/`RGB16`/`Lsb`) numeric
///   `ColorSpace::Untagged` gain maps.
/// - Forward (SDR→HDR) and Reverse (HDR→SDR) directions, Base and Alternate application
///   primaries, and cooperative [`enough::Stop`] cancellation.
pub fn extract_gain_map_only(
    base_src: &PixelView<'_>,
    alternate_src: &PixelView<'_>,
    config: &GainMapExtractConfig,
    stop: &dyn Stop,
) -> Result<(PixelBuf, GainMapMetadata), At<ConvertError>> {
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
    let dual = DualLayerHdr::new(base_owned, gain_buf, metadata)
        .map_err(|e| err_at((*e.error()).into()))?;
    if metadata.application_space == GainMapColorSpace::Alternate
        || alternate_src.spec().color != base_src.spec().color
    {
        dual.with_alternate_color_space(alternate_src.spec().color.clone())
            .map_err(|e| err_at((*e.error()).into()))
    } else {
        Ok(dual)
    }
}

type ObservedGainRange = Option<(f32, f32)>;

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
    let mut base_row = vec![0.0f32; 0];
    base_row
        .try_reserve_exact(row_floats)
        .map_err(|_| err_at(ConvertError::AllocationFailed))?;
    base_row.resize(row_floats, 0.0f32);
    let mut alt_row = vec![0.0f32; 0];
    alt_row
        .try_reserve_exact(row_floats)
        .map_err(|_| err_at(ConvertError::AllocationFailed))?;
    alt_row.resize(row_floats, 0.0f32);

    let mut dummy_row = vec![0u8; 0];
    if measure_only {
        let b = (gm_w as usize) * gm_spec.layout.bytes_per_pixel();
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
    let mut gm_base_row = Vec::new();
    gm_base_row
        .try_reserve_exact(gm_floats)
        .map_err(|_| err_at(ConvertError::AllocationFailed))?;
    gm_base_row.resize(gm_floats, 0.0f32);
    let mut gm_alt_row = Vec::new();
    gm_alt_row
        .try_reserve_exact(gm_floats)
        .map_err(|_| err_at(ConvertError::AllocationFailed))?;
    gm_alt_row.resize(gm_floats, 0.0f32);

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
            let hw = AxisWeights::try_new(w, gm_w, GainMapResample::Bilinear).map_err(err_at)?;
            let vw = AxisWeights::try_new(h, gm_h, GainMapResample::Bilinear).map_err(err_at)?;
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
                return Some((id.primaries, None));
            }
            None
        }
        _ => None,
    }
}
