//! Streaming HDR content-light-level (MaxCLL / MaxFALL) measurement in O(1) row memory.
//!
//! Operates directly over any [`PixelSource`] (`PixelView`, `PixelBuf`, `BandedBuf`)
//! using a stack-resident 64-pixel tile — **zero full-image F32 temporary buffer**.

use alloc::boxed::Box;
use whereat::At;
use zenpixels::{
    ColorSpace, ContentLightLevel, DiffuseWhite, HlgMode, HlgRendering, ImageSpec, PixelSource,
    SampleType, Transfer,
};

use crate::accel::Accelerators;
use crate::error::ConvertError;
use crate::policy::{ConvertPolicy, DepthPolicy, GamutPolicy};
use crate::tonemap::{BT2020_LB, BT2020_LG, BT2020_LR};

/// Per-pixel reduction method for content-light-level measurement.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LightLevelMethod {
    /// `max(R, G, B)` per pixel — CTA-861.3 / HDR10 default.
    #[default]
    MaxRgb,
    /// BT.2020 NCL luminance (`0.2627·R + 0.6780·G + 0.0593·B`).
    LuminanceBt2020,
}

/// 1024-bin log2 histogram of per-pixel luminance in cd/m² over `[0.005, 10_000]` nits,
/// tracking exact zero / non-positive counts separately from positive sub-bin samples.
#[derive(Clone, Debug)]
pub struct LightLevelHistogram {
    bins: Box<[u64]>,
    zero_count: u64,
    total: u64,
    sum_nits: f64,
    literal_max_nits: f32,
    method: LightLevelMethod,
}

impl LightLevelHistogram {
    /// Lower bin edge in cd/m².
    pub const BIN_MIN_NITS: f32 = 0.005;
    /// Upper bin edge in cd/m² (ST 2084 PQ peak).
    pub const BIN_MAX_NITS: f32 = 10_000.0;
    /// Number of log2-spaced bins (~0.02 stops per bin).
    pub const NUM_BINS: usize = 1024;

    const LOG2_MIN: f32 = -7.643_856;
    const LOG2_MAX: f32 = 13.287_712;
    const INV_SPAN: f32 = (Self::NUM_BINS - 1) as f32 / (Self::LOG2_MAX - Self::LOG2_MIN);

    /// Fallibly construct an empty histogram (returns `Err(ConvertError::AllocationFailed)` on OOM).
    pub fn try_new(method: LightLevelMethod) -> Result<Self, At<ConvertError>> {
        let mut bins = alloc::vec::Vec::new();
        bins.try_reserve_exact(Self::NUM_BINS)
            .map_err(|_| crate::error::err_at(ConvertError::AllocationFailed))?;
        bins.resize(Self::NUM_BINS, 0u64);
        Ok(Self {
            bins: bins.into_boxed_slice(),
            zero_count: 0,
            total: 0,
            sum_nits: 0.0,
            literal_max_nits: 0.0,
            method,
        })
    }

    /// Construct an empty histogram.
    #[must_use]
    pub fn new(method: LightLevelMethod) -> Self {
        Self {
            bins: alloc::vec![0u64; Self::NUM_BINS].into_boxed_slice(),
            zero_count: 0,
            total: 0,
            sum_nits: 0.0,
            literal_max_nits: 0.0,
            method,
        }
    }

    /// Record a single pixel's light level in cd/m².
    #[inline]
    pub fn record(&mut self, nits: f32) {
        if !nits.is_finite() || nits <= 0.0 {
            self.zero_count += 1;
            self.total += 1;
            return;
        }
        if nits > self.literal_max_nits {
            self.literal_max_nits = nits;
        }
        self.sum_nits += f64::from(nits);
        self.total += 1;
        let idx = if nits <= Self::BIN_MIN_NITS {
            0
        } else if nits >= Self::BIN_MAX_NITS {
            Self::NUM_BINS - 1
        } else {
            let l2 = libm::log2f(nits);
            (((l2 - Self::LOG2_MIN) * Self::INV_SPAN) as usize).min(Self::NUM_BINS - 1)
        };
        self.bins[idx] += 1;
    }

    /// Read-only view of the 1024 log2-spaced luminance histogram bins.
    #[inline]
    #[must_use]
    pub fn bins(&self) -> &[u64] {
        &self.bins
    }

    /// Merge another [`LightLevelHistogram`] accumulated with the same [`LightLevelMethod`]
    /// into `self` (useful for parallel tile/strip reduction).
    ///
    /// Returns [`ConvertError::UnsupportedRoute`] if `self.method() != other.method()`.
    pub fn merge(&mut self, other: &Self) -> Result<(), At<ConvertError>> {
        if self.method != other.method {
            return Err(crate::error::err_at(ConvertError::UnsupportedRoute));
        }
        for (dst, &src) in self.bins.iter_mut().zip(other.bins.iter()) {
            *dst = dst.saturating_add(src);
        }
        self.zero_count = self.zero_count.saturating_add(other.zero_count);
        self.total = self.total.saturating_add(other.total);
        self.sum_nits += other.sum_nits;
        if other.literal_max_nits > self.literal_max_nits {
            self.literal_max_nits = other.literal_max_nits;
        }
        Ok(())
    }

    /// Reduction method used to build this histogram.
    #[inline]
    #[must_use]
    pub const fn method(&self) -> LightLevelMethod {
        self.method
    }

    /// Total pixels recorded.
    #[inline]
    #[must_use]
    pub const fn total_pixels(&self) -> u64 {
        self.total
    }

    /// Number of recorded zero / non-positive / non-finite samples.
    #[inline]
    #[must_use]
    pub const fn zero_pixels(&self) -> u64 {
        self.zero_count
    }

    /// Exact literal maximum luminance observed in cd/m².
    #[inline]
    #[must_use]
    pub const fn max_nits(&self) -> f32 {
        self.literal_max_nits
    }

    /// Arithmetic mean luminance (MaxFALL) in cd/m².
    #[inline]
    #[must_use]
    pub fn mean_nits(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.sum_nits / self.total as f64) as f32
        }
    }

    /// Nearest-rank order-statistic percentile readout (`p` in `0.0..=1.0`, e.g. `0.9999`).
    ///
    /// Uses rank `k = ceil(p * total).max(1)` over the sorted sample set (so `p = 0.0`
    /// returns `0.0` when any black pixel was recorded, and `p = 0.5` on `[0, 1, 100]`
    /// selects the 2nd sample ≈ `1.0`).
    #[must_use]
    pub fn percentile_nits(&self, p: f32) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        if p >= 1.0 {
            return self.literal_max_nits;
        }
        let p_clamped = if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) };
        let rank = (libm::ceil((self.total as f64) * f64::from(p_clamped)) as u64).max(1);
        if rank <= self.zero_count {
            return 0.0;
        }
        let mut acc = self.zero_count;
        for (i, &count) in self.bins.iter().enumerate() {
            acc += count;
            if acc >= rank {
                if i == 0 {
                    return Self::BIN_MIN_NITS.min(self.literal_max_nits);
                }
                let l2 = Self::LOG2_MIN + (i as f32) / Self::INV_SPAN;
                return libm::exp2f(l2).min(self.literal_max_nits);
            }
        }
        self.literal_max_nits
    }

    /// Convert literal max + mean into ITU-T H.274 / CTA-861.3 upper-bound [`ContentLightLevel`].
    #[must_use]
    pub fn to_content_light_level(&self) -> ContentLightLevel {
        ContentLightLevel::new(
            ceil_cll_u16(self.max_nits()),
            ceil_cll_u16(self.mean_nits()),
        )
    }

    /// Convert percentile peak + mean into [`ContentLightLevel`].
    #[must_use]
    pub fn to_robust_cll(&self, percentile: f32) -> ContentLightLevel {
        ContentLightLevel::new(
            ceil_cll_u16(self.percentile_nits(percentile)),
            ceil_cll_u16(self.mean_nits()),
        )
    }
}

#[inline]
fn ceil_cll_u16(nits: f32) -> u16 {
    if !nits.is_finite() || nits <= 0.0 {
        return 0;
    }
    // Ceiling upper bound per ITU-T H.274 / CTA-861.3, with tiny relative guard so exact
    // integer light levels with 1-ULP float rounding (e.g. 812.00006) do not jump by +1.
    let v = f64::from(nits);
    let adjusted = if v >= 1.0 { v * (1.0 - 1e-6) } else { v };
    libm::ceil(adjusted).clamp(0.0, f64::from(u16::MAX)) as u16
}

fn measurement_primaries(spec: &ImageSpec, method: LightLevelMethod) -> zenpixels::Primaries {
    match method {
        LightLevelMethod::LuminanceBt2020 => zenpixels::Primaries::Bt2020,
        LightLevelMethod::MaxRgb => match spec.color {
            ColorSpace::Rgb { primaries, .. } if primaries != zenpixels::Primaries::Unknown => {
                primaries
            }
            _ => zenpixels::Primaries::Bt2020,
        },
    }
}

/// Measure a full [`LightLevelHistogram`] over any [`PixelSource`] in O(1) row scratch memory.
///
/// For [`LightLevelMethod::MaxRgb`], RGB inputs are measured in their **native RGB primaries**
/// (CTA-861.3 / ITU-T H.274 MaxCLL definition); for [`LightLevelMethod::LuminanceBt2020`],
/// pixels are linearized into BT.2020 to compute CIE Y luminance.
pub fn measure_histogram<S: PixelSource + ?Sized>(
    src: &S,
    white: DiffuseWhite,
    hlg_rendering: Option<HlgRendering>,
    method: LightLevelMethod,
) -> Result<LightLevelHistogram, At<ConvertError>> {
    let spec = src.source_spec();
    let w = src.source_width();
    let h = src.source_height();
    let mut hist = LightLevelHistogram::try_new(method)?;
    if w == 0 || h == 0 {
        return Ok(hist);
    }

    let linear_spec = ImageSpec::new(
        zenpixels::SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: measurement_primaries(spec, method),
            transfer: Transfer::Linear,
            range: zenpixels::SignalRange::Full,
            alpha: zenpixels::AlphaMode::None,
            white: Some(white),
        },
    );
    let mut policy = ConvertPolicy::working_linear()
        .with_sdr_white(white)
        .with_depth(DepthPolicy::Round)
        .with_gamut(GamutPolicy::PreserveExtended)
        .with_alpha(crate::policy::AlphaPolicy::DiscardUnchecked)
        .with_accelerators(Accelerators::default());
    if let Some(r) = hlg_rendering {
        policy = policy.with_hlg_rendering(r);
    }
    let row_plan = crate::plan::RowPlan::build(spec.clone(), linear_spec, &policy, None)?;

    // Single reusable fallibly-allocated, 4-byte-aligned f32 scratch row
    // (O(width) = a few KiB, NEVER O(width * height), safe on odd allocators!).
    let num_floats = (w as usize)
        .checked_mul(3)
        .ok_or_else(|| crate::error::err_at(ConvertError::DimensionMismatch))?;
    let mut scratch_f32: alloc::vec::Vec<f32> = alloc::vec::Vec::new();
    scratch_f32
        .try_reserve_exact(num_floats)
        .map_err(|_| crate::error::err_at(ConvertError::AllocationFailed))?;
    scratch_f32.resize(num_floats, 0.0f32);
    let white_nits = white.nits();
    for y in 0..h {
        let scratch_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut scratch_f32);
        row_plan.convert_row(src.source_row(y), scratch_bytes, w, y)?;
        for px in scratch_f32.as_chunks::<3>().0 {
            let v = match method {
                LightLevelMethod::MaxRgb => px[0].max(px[1]).max(px[2]),
                LightLevelMethod::LuminanceBt2020 => {
                    BT2020_LR * px[0] + BT2020_LG * px[1] + BT2020_LB * px[2]
                }
            };
            hist.record(v * white_nits);
        }
    }
    Ok(hist)
}

/// Measure [`ContentLightLevel`] (literal MaxCLL + MaxFALL) over any [`PixelSource`] in O(1) row memory.
pub fn measure_cll<S: PixelSource + ?Sized>(
    src: &S,
    white: DiffuseWhite,
    hlg_rendering: Option<HlgRendering>,
    method: LightLevelMethod,
) -> Result<ContentLightLevel, At<ConvertError>> {
    Ok(measure_histogram(src, white, hlg_rendering, method)?.to_content_light_level())
}

/// 3-tap mirror-padded horizontal box filter over one linear-light RGB(A) f32 row.
/// Returns `(max_smoothed_rel, sum_rel)` in diffuse-white-relative units.
#[inline]
fn scan_row_max_mean_smoothed<const N: usize>(row: &[f32], method: LightLevelMethod) -> (f32, f64) {
    const ONE_THIRD: f32 = 1.0 / 3.0;
    let pixel_count = row.len() / N;
    if pixel_count == 0 {
        return (0.0, 0.0);
    }

    let reduce_at = |i: usize| -> f32 {
        let px = &row[i * N..(i + 1) * N];
        let r = if px[0].is_finite() {
            px[0].max(0.0)
        } else {
            0.0
        };
        let g = if px[1].is_finite() {
            px[1].max(0.0)
        } else {
            0.0
        };
        let b = if px[2].is_finite() {
            px[2].max(0.0)
        } else {
            0.0
        };
        match method {
            LightLevelMethod::MaxRgb => r.max(g).max(b),
            LightLevelMethod::LuminanceBt2020 => BT2020_LR * r + BT2020_LG * g + BT2020_LB * b,
        }
    };

    if pixel_count == 1 {
        let m = reduce_at(0);
        return (m, f64::from(m));
    }
    if pixel_count == 2 {
        let m0 = reduce_at(0);
        let m1 = reduce_at(1);
        let s0 = (2.0 * m0 + m1) * ONE_THIRD;
        let s1 = (m0 + 2.0 * m1) * ONE_THIRD;
        return (s0.max(s1), f64::from(m0) + f64::from(m1));
    }

    let m0 = reduce_at(0);
    let m1 = reduce_at(1);
    let mut prev = m0;
    let mut curr = m1;
    let mut max_x3 = 2.0 * m0 + m1;
    let mut sum = f64::from(m0);

    for i in 2..pixel_count {
        let next = reduce_at(i);
        let s = prev + curr + next;
        if s > max_x3 {
            max_x3 = s;
        }
        sum += f64::from(curr);
        prev = curr;
        curr = next;
    }
    let s_last = prev + 2.0 * curr;
    if s_last > max_x3 {
        max_x3 = s_last;
    }
    sum += f64::from(curr);

    (max_x3 * ONE_THIRD, sum)
}

/// Measure defect-tolerant [`ContentLightLevel`] using a 3-tap horizontal mirror-padded
/// box filter on per-pixel light levels (`(m[i-1] + m[i] + m[i+1]) / 3`) in O(1) row memory.
///
/// Suppresses isolated stuck/hot pixels by 3× while preserving bright features spanning
/// ≥2 horizontal pixels; MaxFALL remains the exact arithmetic mean across all pixels.
pub fn measure_max_smoothed<S: PixelSource + ?Sized>(
    src: &S,
    white: DiffuseWhite,
    hlg_rendering: Option<HlgRendering>,
    method: LightLevelMethod,
) -> Result<ContentLightLevel, At<ConvertError>> {
    let spec = src.source_spec();
    let w = src.source_width();
    let h = src.source_height();
    if w == 0 || h == 0 {
        return Ok(ContentLightLevel::new(0, 0));
    }
    let total_px = (w as f64) * (h as f64);

    // Zero-allocation fast path for unpremultiplied full-packing Linear F32 RGB(A) in matching primaries.
    let target_primaries = measurement_primaries(spec, method);
    if spec.layout.packing.is_full()
        && spec.layout.sample == SampleType::F32
        && matches!(
            spec.layout.order,
            zenpixels::ChannelOrder::Rgb | zenpixels::ChannelOrder::Rgba
        )
        && let ColorSpace::Rgb {
            primaries,
            transfer: Transfer::Linear,
            range: zenpixels::SignalRange::Full,
            alpha,
            white: src_white,
        } = spec.color
        && primaries == target_primaries
        && alpha != zenpixels::AlphaMode::Premultiplied
    {
        let ch = spec.layout.channels();
        let eff_white = src_white.unwrap_or(white).nits();
        let mut max_rel = 0.0_f32;
        let mut sum_rel = 0.0_f64;
        let mut aligned_scratch: alloc::vec::Vec<f32> = alloc::vec::Vec::new();
        for y in 0..h {
            let row = src.source_row(y);
            let floats: &[f32] = if let Ok(f) = bytemuck::try_cast_slice::<u8, f32>(row) {
                f
            } else {
                let need = (w as usize) * ch;
                if aligned_scratch.len() < need {
                    aligned_scratch
                        .try_reserve_exact(need - aligned_scratch.len())
                        .map_err(|_| crate::error::err_at(ConvertError::AllocationFailed))?;
                    aligned_scratch.resize(need, 0.0);
                }
                for (dst_f, chunk) in aligned_scratch[..need]
                    .iter_mut()
                    .zip(row.as_chunks::<4>().0)
                {
                    *dst_f = f32::from_ne_bytes(*chunk);
                }
                &aligned_scratch[..need]
            };
            let (rm, rs) = if ch == 3 {
                scan_row_max_mean_smoothed::<3>(floats, method)
            } else {
                scan_row_max_mean_smoothed::<4>(floats, method)
            };
            if rm > max_rel {
                max_rel = rm;
            }
            sum_rel += rs;
        }
        return Ok(ContentLightLevel::new(
            ceil_cll_u16(max_rel * eff_white),
            ceil_cll_u16(((sum_rel / total_px) * f64::from(eff_white)) as f32),
        ));
    }

    // General streaming path: 1 fallibly-allocated RGBF32 row via RowPlan.
    let linear_spec = ImageSpec::new(
        zenpixels::SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: target_primaries,
            transfer: Transfer::Linear,
            range: zenpixels::SignalRange::Full,
            alpha: zenpixels::AlphaMode::None,
            white: Some(white),
        },
    );
    let mut policy = ConvertPolicy::working_linear()
        .with_sdr_white(white)
        .with_depth(DepthPolicy::Round)
        .with_gamut(GamutPolicy::PreserveExtended)
        .with_alpha(crate::policy::AlphaPolicy::DiscardUnchecked)
        .with_accelerators(Accelerators::default());
    if let Some(r) = hlg_rendering {
        policy = policy.with_hlg_rendering(r);
    }
    let row_plan = crate::plan::RowPlan::build(spec.clone(), linear_spec, &policy, None)?;

    let num_floats = (w as usize)
        .checked_mul(3)
        .ok_or_else(|| crate::error::err_at(ConvertError::DimensionMismatch))?;
    let mut scratch_f32: alloc::vec::Vec<f32> = alloc::vec::Vec::new();
    scratch_f32
        .try_reserve_exact(num_floats)
        .map_err(|_| crate::error::err_at(ConvertError::AllocationFailed))?;
    scratch_f32.resize(num_floats, 0.0f32);

    let white_nits = white.nits();
    let mut max_rel = 0.0_f32;
    let mut sum_rel = 0.0_f64;
    for y in 0..h {
        let scratch_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut scratch_f32);
        row_plan.convert_row(src.source_row(y), scratch_bytes, w, y)?;
        let (rm, rs) = scan_row_max_mean_smoothed::<3>(&scratch_f32, method);
        if rm > max_rel {
            max_rel = rm;
        }
        sum_rel += rs;
    }
    Ok(ContentLightLevel::new(
        ceil_cll_u16(max_rel * white_nits),
        ceil_cll_u16(((sum_rel / total_px) * f64::from(white_nits)) as f32),
    ))
}

/// Fast zero-histogram streaming max-luminance peak scan in cd/m² (O(1) row scratch).
///
/// Measures native-primaries `MaxRgb` peak luminance without building a log histogram,
/// allocating at most one fallibly-reserved `f32` row (or zero bytes on unpremultiplied
/// Linear F32 / Full PQ U16 RGB sources).
pub fn measure_peak<S: PixelSource + ?Sized>(
    src: &S,
    white: DiffuseWhite,
    hlg_rendering: Option<HlgRendering>,
) -> Result<f32, At<ConvertError>> {
    measure_max_nits_streaming(src, white, hlg_rendering)
}

/// Fast zero-histogram streaming max-luminance peak scan in cd/m² (O(1) row scratch).
pub(crate) fn measure_max_nits_streaming<S: PixelSource + ?Sized>(
    src: &S,
    white: DiffuseWhite,
    hlg_rendering: Option<HlgRendering>,
) -> Result<f32, At<ConvertError>> {
    let spec = src.source_spec();
    let w = src.source_width();
    let h = src.source_height();
    if w == 0 || h == 0 {
        return Ok(0.0);
    }

    // Zero-allocation fast paths for unpremultiplied native-primaries Linear F32 or Full PQ U16 RGB.
    if spec.layout.packing.is_full() {
        if let ColorSpace::Rgb {
            primaries,
            transfer: Transfer::Linear,
            range: zenpixels::SignalRange::Full,
            alpha,
            white: src_white,
        } = spec.color
            && primaries != zenpixels::Primaries::Unknown
            && alpha != zenpixels::AlphaMode::Premultiplied
            && spec.layout.sample == SampleType::F32
        {
            let ch = spec.layout.channels();
            let eff_white = src_white.unwrap_or(white).nits();
            let mut max_v = 0.0f32;
            for y in 0..h {
                let row = src.source_row(y);
                for px in row.chunks_exact(ch * 4) {
                    for c in 0..3 {
                        let v = f32::from_ne_bytes(px[c * 4..c * 4 + 4].try_into().unwrap());
                        if v.is_finite() && v > max_v {
                            max_v = v;
                        }
                    }
                }
            }
            return Ok(max_v * eff_white);
        }
        if let ColorSpace::Rgb {
            primaries,
            transfer: Transfer::Pq,
            range: zenpixels::SignalRange::Full,
            alpha,
            ..
        } = spec.color
            && primaries != zenpixels::Primaries::Unknown
            && alpha != zenpixels::AlphaMode::Premultiplied
            && spec.layout.sample == SampleType::U16
        {
            let ch = spec.layout.channels();
            let mut max_code = 0u16;
            for y in 0..h {
                let row = src.source_row(y);
                for px in row.chunks_exact(ch * 2) {
                    for c in 0..3 {
                        let v = u16::from_ne_bytes([px[c * 2], px[c * 2 + 1]]);
                        if v > max_code {
                            max_code = v;
                        }
                    }
                }
            }
            // Use the exact same PQ EOTF as the row pipeline for 100% consistent peak definition.
            let mut norm = [f32::from(max_code) * (1.0 / 65535.0)];
            #[cfg(feature = "hdr-transfers")]
            crate::transfers::hdr::pq_eotf_scalar_slice(&mut norm, 10_000.0);
            #[cfg(not(feature = "hdr-transfers"))]
            {
                norm[0] = crate::strip::pq_eotf_scalar(norm[0]) * 10_000.0;
            }
            return Ok(norm[0]);
        }
    }

    // General zero-histogram streaming max reducer: 1 fallibly-allocated f32 row via RowPlan!
    let eff_hlg = match spec.color {
        ColorSpace::Rgb {
            transfer: Transfer::Hlg(HlgMode::Display(r)),
            ..
        } => Some(r),
        _ => hlg_rendering,
    };
    let linear_spec = ImageSpec::new(
        zenpixels::SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: measurement_primaries(spec, LightLevelMethod::MaxRgb),
            transfer: Transfer::Linear,
            range: zenpixels::SignalRange::Full,
            alpha: zenpixels::AlphaMode::None,
            white: Some(white),
        },
    );
    let mut policy = ConvertPolicy::working_linear()
        .with_sdr_white(white)
        .with_depth(DepthPolicy::Round)
        .with_gamut(GamutPolicy::PreserveExtended)
        .with_alpha(crate::policy::AlphaPolicy::DiscardUnchecked)
        .with_accelerators(Accelerators::default());
    if let Some(r) = eff_hlg {
        policy = policy.with_hlg_rendering(r);
    }
    let row_plan = crate::plan::RowPlan::build(spec.clone(), linear_spec, &policy, None)?;

    let num_floats = (w as usize)
        .checked_mul(3)
        .ok_or_else(|| crate::error::err_at(ConvertError::DimensionMismatch))?;
    let mut scratch_f32: alloc::vec::Vec<f32> = alloc::vec::Vec::new();
    scratch_f32
        .try_reserve_exact(num_floats)
        .map_err(|_| crate::error::err_at(ConvertError::AllocationFailed))?;
    scratch_f32.resize(num_floats, 0.0f32);

    let white_nits = white.nits();
    let mut max_v = 0.0f32;
    for y in 0..h {
        let scratch_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut scratch_f32);
        row_plan.convert_row(src.source_row(y), scratch_bytes, w, y)?;
        for &v in &scratch_f32 {
            if v.is_finite() && v > max_v {
                max_v = v;
            }
        }
    }
    Ok(max_v * white_nits)
}
