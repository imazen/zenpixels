//! Extension traits that add conversion methods to zenpixels interchange types.
//!
//! These traits bridge the type–conversion boundary: the types live in
//! `zenpixels` (no heavy deps), while the conversion math lives here
//! (depends on `linear-srgb`).

use zenpixels::{ColorPrimaries, TransferFunction};

use crate::convert::{hlg_eotf, hlg_oetf, pq_eotf, pq_oetf};
use crate::gamut::GamutMatrix;

// ---------------------------------------------------------------------------
// TransferFunctionExt
// ---------------------------------------------------------------------------

/// Adds scalar EOTF/OETF methods to [`TransferFunction`].
pub trait TransferFunctionExt {
    /// Scalar EOTF: encoded signal → linear light.
    ///
    /// Canonical reference implementation for testing SIMD paths.
    #[must_use]
    fn linearize(&self, v: f32) -> f32;

    /// Scalar OETF: linear light → encoded signal.
    ///
    /// Canonical reference implementation for testing SIMD paths.
    #[must_use]
    fn delinearize(&self, v: f32) -> f32;
}

impl TransferFunctionExt for TransferFunction {
    #[allow(unreachable_patterns)]
    fn linearize(&self, v: f32) -> f32 {
        match self {
            Self::Linear | Self::Unknown => v,
            Self::Srgb => linear_srgb::precise::srgb_to_linear(v),
            Self::Bt709 => linear_srgb::tf::bt709_to_linear(v),
            Self::Pq => pq_eotf(v),
            Self::Hlg => hlg_eotf(v),
            _ => v,
        }
    }

    #[allow(unreachable_patterns)]
    fn delinearize(&self, v: f32) -> f32 {
        match self {
            Self::Linear | Self::Unknown => v,
            Self::Srgb => linear_srgb::precise::linear_to_srgb(v),
            Self::Bt709 => linear_srgb::tf::linear_to_bt709(v),
            Self::Pq => pq_oetf(v),
            Self::Hlg => hlg_oetf(v),
            _ => v,
        }
    }
}

// ---------------------------------------------------------------------------
// ColorPrimariesExt
// ---------------------------------------------------------------------------

/// Adds XYZ matrix lookups to [`ColorPrimaries`].
#[allow(clippy::wrong_self_convention)]
pub trait ColorPrimariesExt {
    /// Linear RGB → CIE XYZ (D65 white point).
    ///
    /// Returns `None` for [`Unknown`](ColorPrimaries::Unknown).
    fn to_xyz_matrix(&self) -> Option<&'static GamutMatrix>;

    /// CIE XYZ (D65 white point) → linear RGB.
    ///
    /// Returns `None` for [`Unknown`](ColorPrimaries::Unknown).
    fn from_xyz_matrix(&self) -> Option<&'static GamutMatrix>;
}

impl ColorPrimariesExt for ColorPrimaries {
    #[allow(unreachable_patterns)]
    fn to_xyz_matrix(&self) -> Option<&'static GamutMatrix> {
        match self {
            Self::Bt709 => Some(&crate::gamut::BT709_TO_XYZ),
            Self::DisplayP3 => Some(&crate::gamut::DISPLAY_P3_TO_XYZ),
            Self::Bt2020 => Some(&crate::gamut::BT2020_TO_XYZ),
            _ => None,
        }
    }

    #[allow(unreachable_patterns)]
    fn from_xyz_matrix(&self) -> Option<&'static GamutMatrix> {
        match self {
            Self::Bt709 => Some(&crate::gamut::XYZ_TO_BT709),
            Self::DisplayP3 => Some(&crate::gamut::XYZ_TO_DISPLAY_P3),
            Self::Bt2020 => Some(&crate::gamut::XYZ_TO_BT2020),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// PixelBufferConvertExt
// ---------------------------------------------------------------------------

use alloc::sync::Arc;
use whereat::{At, ResultAtExt};
use zenpixels::PixelDescriptor;
use zenpixels::buffer::PixelBuffer;
use zenpixels::descriptor::{AlphaMode, ChannelLayout, ChannelType};

/// Adds format conversion methods to type-erased [`PixelBuffer`].
pub trait PixelBufferConvertExt {
    /// Convert pixel data to a different layout and depth.
    ///
    /// Uses [`RowConverter`](crate::RowConverter) for transfer-function-aware
    /// conversion. Color metadata is preserved.
    ///
    /// **Allocates** a new [`PixelBuffer`].
    fn convert_to(&self, target: PixelDescriptor) -> Result<PixelBuffer, At<crate::ConvertError>>;

    /// Add an alpha channel. **Allocates** a new `PixelBuffer`.
    ///
    /// - Gray → GrayAlpha (opaque alpha)
    /// - Rgb → Rgba (opaque alpha)
    /// - Already has alpha → identity copy
    fn try_add_alpha(&self) -> Result<PixelBuffer, At<crate::ConvertError>>;

    /// Widen to U16 depth (lossless, ×257). **Allocates** a new `PixelBuffer`.
    fn try_widen_to_u16(&self) -> Result<PixelBuffer, At<crate::ConvertError>>;

    /// Narrow to U8 depth (lossy, rounded). **Allocates** a new `PixelBuffer`.
    fn try_narrow_to_u8(&self) -> Result<PixelBuffer, At<crate::ConvertError>>;

    /// Convert to linear-light F32, preserving channel layout and primaries.
    ///
    /// This is the EOTF step of a scene-referred pipeline: decoded pixels
    /// (sRGB, BT.709, PQ, HLG) are converted to linear light for processing.
    ///
    /// **Allocates** a new `PixelBuffer`.
    fn linearize(&self) -> Result<PixelBuffer, At<crate::ConvertError>>;

    /// Apply a transfer function to a linear-light buffer.
    ///
    /// This is the OETF step: linear-light pixels are encoded for display
    /// or storage. The buffer should be in F32 linear light; if it is in a
    /// different transfer function, the conversion goes through linear as
    /// an intermediate step.
    ///
    /// **Allocates** a new `PixelBuffer`.
    fn delinearize(
        &self,
        transfer: TransferFunction,
    ) -> Result<PixelBuffer, At<crate::ConvertError>>;

    /// Estimate the cost of [`convert_to(target)`](Self::convert_to)
    /// **without running it**.
    ///
    /// Cheap to call — builds a [`ConvertPlan`](crate::ConvertPlan)
    /// internally and walks its steps. No row work, no large
    /// allocations. The returned [`ResourceEstimate`](crate::ResourceEstimate)
    /// reports peak working-set memory + median wall-clock time on the
    /// reference machine, with a ±30 % design tolerance — see
    /// [`crate::estimate`] for the accuracy contract.
    ///
    /// If no conversion path exists (the same `Err` that `convert_to`
    /// would return), the estimate falls back to
    /// [`EstimateConfidence::Unknown`](crate::EstimateConfidence::Unknown)
    /// with zero memory and zero time. Callers can read
    /// [`ResourceEstimate::confidence`](crate::ResourceEstimate::confidence)
    /// to detect that case.
    fn estimate_convert_to(&self, target: &PixelDescriptor) -> crate::ResourceEstimate;

    /// Estimate the cost of [`try_add_alpha`](Self::try_add_alpha) without
    /// running it.
    ///
    /// Builds the same target descriptor `try_add_alpha` would (Gray →
    /// GrayAlpha, Rgb → Rgba, otherwise identity) and delegates to
    /// [`estimate_convert_to`](Self::estimate_convert_to). The returned
    /// [`ResourceEstimate`](crate::ResourceEstimate) carries the
    /// **±30 % design tolerance** of the underlying plan estimator (see
    /// [`crate::estimate`]); check
    /// [`ResourceEstimate::confidence`](crate::ResourceEstimate::confidence)
    /// for the calibration tier.
    fn estimate_try_add_alpha(&self) -> crate::ResourceEstimate;

    /// Estimate the cost of [`try_widen_to_u16`](Self::try_widen_to_u16)
    /// without running it.
    ///
    /// Same shape as `try_widen_to_u16` but produces a
    /// [`ResourceEstimate`](crate::ResourceEstimate) instead of an
    /// allocated buffer. **±30 % design tolerance** per
    /// [`crate::estimate`]; if the source is already U16 the estimate is
    /// the (memcpy-only) identity path. Inspect
    /// [`ResourceEstimate::confidence`](crate::ResourceEstimate::confidence)
    /// for the calibration tier.
    fn estimate_try_widen_to_u16(&self) -> crate::ResourceEstimate;

    /// Estimate the cost of [`try_narrow_to_u8`](Self::try_narrow_to_u8)
    /// without running it.
    ///
    /// Same shape as `try_narrow_to_u8` but produces a
    /// [`ResourceEstimate`](crate::ResourceEstimate) instead of an
    /// allocated buffer. **±30 % design tolerance** per
    /// [`crate::estimate`]; if the source is already U8 the estimate is
    /// the (memcpy-only) identity path.
    fn estimate_try_narrow_to_u8(&self) -> crate::ResourceEstimate;

    /// Estimate the cost of [`linearize`](Self::linearize) without
    /// running it.
    ///
    /// Same target descriptor as `linearize` (linear-light F32,
    /// preserving layout / alpha / primaries) and delegates to
    /// [`estimate_convert_to`](Self::estimate_convert_to). **±30 %
    /// design tolerance** per [`crate::estimate`].
    fn estimate_linearize(&self) -> crate::ResourceEstimate;

    /// Estimate the cost of [`delinearize(transfer)`](Self::delinearize)
    /// without running it.
    ///
    /// Same target descriptor as `delinearize` (source descriptor with
    /// the new transfer function) and delegates to
    /// [`estimate_convert_to`](Self::estimate_convert_to). **±30 %
    /// design tolerance** per [`crate::estimate`].
    fn estimate_delinearize(&self, transfer: TransferFunction) -> crate::ResourceEstimate;
}

/// Typed convenience conversions that return `PixelBuffer<P>`.
///
/// Requires the `rgb` feature for the concrete pixel types.
#[cfg(feature = "rgb")]
pub trait PixelBufferConvertTypedExt: PixelBufferConvertExt {
    /// Convert to RGB8, allocating a new buffer.
    fn to_rgb8(&self) -> PixelBuffer<rgb::Rgb<u8>>;

    /// Convert to RGBA8, allocating a new buffer.
    fn to_rgba8(&self) -> PixelBuffer<rgb::Rgba<u8>>;

    /// Convert to Gray8, allocating a new buffer.
    fn to_gray8(&self) -> PixelBuffer<rgb::Gray<u8>>;

    /// Convert to BGRA8, allocating a new buffer.
    fn to_bgra8(&self) -> PixelBuffer<rgb::alt::BGRA<u8>>;

    /// Estimate the cost of [`to_rgb8`](Self::to_rgb8) without running it.
    ///
    /// Delegates to
    /// [`estimate_convert_to(&RGB8_SRGB)`](PixelBufferConvertExt::estimate_convert_to)
    /// — the underlying allocation path is the same `convert_to_typed`
    /// call. The typed buffer wrapper itself is a zero-cost cast, so the
    /// returned [`ResourceEstimate`](crate::ResourceEstimate) is the
    /// untyped estimate verbatim. **±30 % design tolerance** per
    /// [`crate::estimate`].
    fn estimate_to_rgb8(&self) -> crate::ResourceEstimate {
        self.estimate_convert_to(&PixelDescriptor::RGB8_SRGB)
    }

    /// Estimate the cost of [`to_rgba8`](Self::to_rgba8) without running
    /// it. Delegates to
    /// [`estimate_convert_to(&RGBA8_SRGB)`](PixelBufferConvertExt::estimate_convert_to).
    /// **±30 % design tolerance** per [`crate::estimate`].
    fn estimate_to_rgba8(&self) -> crate::ResourceEstimate {
        self.estimate_convert_to(&PixelDescriptor::RGBA8_SRGB)
    }

    /// Estimate the cost of [`to_gray8`](Self::to_gray8) without running
    /// it. Delegates to
    /// [`estimate_convert_to(&GRAY8_SRGB)`](PixelBufferConvertExt::estimate_convert_to).
    /// **±30 % design tolerance** per [`crate::estimate`].
    fn estimate_to_gray8(&self) -> crate::ResourceEstimate {
        self.estimate_convert_to(&PixelDescriptor::GRAY8_SRGB)
    }

    /// Estimate the cost of [`to_bgra8`](Self::to_bgra8) without running
    /// it. Delegates to
    /// [`estimate_convert_to(&BGRA8_SRGB)`](PixelBufferConvertExt::estimate_convert_to).
    /// **±30 % design tolerance** per [`crate::estimate`].
    fn estimate_to_bgra8(&self) -> crate::ResourceEstimate {
        self.estimate_convert_to(&PixelDescriptor::BGRA8_SRGB)
    }
}

/// Assert that a descriptor is not CMYK.
fn assert_not_cmyk(desc: &PixelDescriptor) {
    assert!(
        desc.color_model() != crate::ColorModel::Cmyk,
        "CMYK pixel data cannot be processed by zenpixels-convert. \
         Use a CMS (e.g., moxcms) with an ICC profile for CMYK↔RGB conversion."
    );
}

impl PixelBufferConvertExt for PixelBuffer {
    #[track_caller]
    fn convert_to(&self, target: PixelDescriptor) -> Result<PixelBuffer, At<crate::ConvertError>> {
        let src_desc = self.descriptor();
        assert_not_cmyk(&src_desc);
        assert_not_cmyk(&target);
        if src_desc == target {
            // Identity — just copy.
            let dst_stride = target.aligned_stride(self.width());
            let total = dst_stride
                .checked_mul(self.height() as usize)
                .ok_or_else(|| whereat::at!(crate::ConvertError::AllocationFailed))?;
            let mut out = alloc::vec![0u8; total];
            let src_slice = self.as_slice();
            for y in 0..self.height() {
                let src_row = src_slice.row(y);
                let dst_start = y as usize * dst_stride;
                out[dst_start..dst_start + src_row.len()].copy_from_slice(src_row);
            }
            let mut buf = PixelBuffer::from_vec(out, self.width(), self.height(), target)
                .map_err_at(crate::ConvertError::from)?;
            if let Some(ctx) = self.color_context() {
                buf = buf.with_color_context(Arc::clone(ctx));
            }
            return Ok(buf);
        }

        let mut converter = crate::RowConverter::new(src_desc, target).at()?;

        let dst_stride = target.aligned_stride(self.width());
        let total = dst_stride
            .checked_mul(self.height() as usize)
            .ok_or_else(|| whereat::at!(crate::ConvertError::AllocationFailed))?;
        let mut out = alloc::vec![0u8; total];

        let src_slice = self.as_slice();
        for y in 0..self.height() {
            let src_row = src_slice.row(y);
            let dst_start = y as usize * dst_stride;
            let dst_end = dst_start + dst_stride;
            converter.convert_row(src_row, &mut out[dst_start..dst_end], self.width());
        }

        let mut buf = PixelBuffer::from_vec(out, self.width(), self.height(), target)
            .map_err_at(crate::ConvertError::from)?;
        if let Some(ctx) = self.color_context() {
            buf = buf.with_color_context(Arc::clone(ctx));
        }
        Ok(buf)
    }

    #[track_caller]
    fn try_add_alpha(&self) -> Result<PixelBuffer, At<crate::ConvertError>> {
        let desc = self.descriptor();
        let target_layout = match desc.layout() {
            ChannelLayout::Gray => ChannelLayout::GrayAlpha,
            ChannelLayout::Rgb => ChannelLayout::Rgba,
            other => other,
        };
        let alpha = if target_layout.has_alpha() && desc.alpha().is_none() {
            Some(AlphaMode::Straight)
        } else {
            desc.alpha()
        };
        let target =
            PixelDescriptor::new(desc.channel_type(), target_layout, alpha, desc.transfer());
        self.convert_to(target)
    }

    #[track_caller]
    fn try_widen_to_u16(&self) -> Result<PixelBuffer, At<crate::ConvertError>> {
        let desc = self.descriptor();
        let target = PixelDescriptor::new(
            ChannelType::U16,
            desc.layout(),
            desc.alpha(),
            desc.transfer(),
        );
        self.convert_to(target)
    }

    #[track_caller]
    fn try_narrow_to_u8(&self) -> Result<PixelBuffer, At<crate::ConvertError>> {
        let desc = self.descriptor();
        let target = PixelDescriptor::new(
            ChannelType::U8,
            desc.layout(),
            desc.alpha(),
            desc.transfer(),
        );
        self.convert_to(target)
    }

    #[track_caller]
    fn linearize(&self) -> Result<PixelBuffer, At<crate::ConvertError>> {
        let desc = self.descriptor();
        let target = PixelDescriptor::new_full(
            ChannelType::F32,
            desc.layout(),
            desc.alpha(),
            TransferFunction::Linear,
            desc.primaries,
        );
        self.convert_to(target)
    }

    #[track_caller]
    fn delinearize(
        &self,
        transfer: TransferFunction,
    ) -> Result<PixelBuffer, At<crate::ConvertError>> {
        let target = self.descriptor().with_transfer(transfer);
        self.convert_to(target)
    }

    fn estimate_convert_to(&self, target: &PixelDescriptor) -> crate::ResourceEstimate {
        let src = self.descriptor();
        // CMYK is rejected by the plan builder too; we don't panic in the
        // estimator — surface as Unknown so callers reading the field can
        // detect "I can't tell" without a panic.
        if src.color_model() == crate::ColorModel::Cmyk
            || target.color_model() == crate::ColorModel::Cmyk
        {
            return crate::ResourceEstimate {
                peak_memory_bytes: 0,
                wall_time_ms: 0.0,
                breakdown: alloc::vec::Vec::new(),
                confidence: crate::EstimateConfidence::Unknown,
            };
        }
        match crate::ConvertPlan::new(src, *target) {
            Ok(plan) => plan.estimate_resources(self.width(), self.height()),
            Err(_) => crate::ResourceEstimate {
                peak_memory_bytes: 0,
                wall_time_ms: 0.0,
                breakdown: alloc::vec::Vec::new(),
                confidence: crate::EstimateConfidence::Unknown,
            },
        }
    }

    fn estimate_try_add_alpha(&self) -> crate::ResourceEstimate {
        let desc = self.descriptor();
        let target_layout = match desc.layout() {
            ChannelLayout::Gray => ChannelLayout::GrayAlpha,
            ChannelLayout::Rgb => ChannelLayout::Rgba,
            other => other,
        };
        let alpha = if target_layout.has_alpha() && desc.alpha().is_none() {
            Some(AlphaMode::Straight)
        } else {
            desc.alpha()
        };
        let target =
            PixelDescriptor::new(desc.channel_type(), target_layout, alpha, desc.transfer());
        self.estimate_convert_to(&target)
    }

    fn estimate_try_widen_to_u16(&self) -> crate::ResourceEstimate {
        let desc = self.descriptor();
        if desc.channel_type() == ChannelType::U16 {
            // No-op shortcut: same descriptor → identity memcpy path.
            return self.estimate_convert_to(&desc);
        }
        let target = PixelDescriptor::new(
            ChannelType::U16,
            desc.layout(),
            desc.alpha(),
            desc.transfer(),
        );
        self.estimate_convert_to(&target)
    }

    fn estimate_try_narrow_to_u8(&self) -> crate::ResourceEstimate {
        let desc = self.descriptor();
        if desc.channel_type() == ChannelType::U8 {
            return self.estimate_convert_to(&desc);
        }
        let target = PixelDescriptor::new(
            ChannelType::U8,
            desc.layout(),
            desc.alpha(),
            desc.transfer(),
        );
        self.estimate_convert_to(&target)
    }

    fn estimate_linearize(&self) -> crate::ResourceEstimate {
        let desc = self.descriptor();
        let target = PixelDescriptor::new_full(
            ChannelType::F32,
            desc.layout(),
            desc.alpha(),
            TransferFunction::Linear,
            desc.primaries,
        );
        self.estimate_convert_to(&target)
    }

    fn estimate_delinearize(&self, transfer: TransferFunction) -> crate::ResourceEstimate {
        let target = self.descriptor().with_transfer(transfer);
        self.estimate_convert_to(&target)
    }
}

/// Adds HDR-aware conversion methods to [`PixelBuffer`].
///
/// HDR→SDR conversions need a source-peak luminance to parameterize the
/// BT.2446-A tone-map curve. Plain
/// [`convert_to`](PixelBufferConvertExt::convert_to) refuses such cases
/// with [`ConvertError::HdrSourceRequiresPeak`](crate::ConvertError::HdrSourceRequiresPeak).
/// These methods supply the peak — either explicitly
/// ([`convert_to_with_hdr_config`](Self::convert_to_with_hdr_config)) or
/// by measuring MaxCLL from the buffer itself
/// ([`convert_to_sdr`](Self::convert_to_sdr)).
///
/// Gated behind `hdr-experimental`.
#[cfg(feature = "hdr-experimental")]
pub trait PixelBufferHdrConvertExt {
    /// Convert this HDR buffer to `target` (typically an SDR descriptor —
    /// sRGB / BT.709 / Gamma22), auto-measuring source peak via
    /// [`CllMeasure::measure_max`](crate::hdr::CllMeasure::measure_max)
    /// (the production-default per the 2026-06-22 audited shootout —
    /// wins 3 of 6 ranking criteria including the user-visible
    /// `pct_above_de5`, see `DEFAULT_PERCENTILE` docs for the alternative
    /// percentile path).
    ///
    /// For non-HDR sources this falls back to
    /// [`convert_to`](PixelBufferConvertExt::convert_to) (so the call is
    /// safe to use when the source's HDR-ness isn't known up front).
    ///
    /// **Allocates** a new [`PixelBuffer`].
    fn convert_to_sdr(
        &self,
        target: PixelDescriptor,
    ) -> Result<PixelBuffer, At<crate::ConvertError>>;

    /// Convert this HDR buffer to `target` with explicit HDR knobs.
    ///
    /// `hdr.source_peak_nits` is mandatory and parameterizes the BT.2446-A
    /// curve. `target_peak_nits` defaults to `100.0` (SDR), `gamut_knee`
    /// to `0.9` — pass via [`HdrConfig::default()`](crate::HdrConfig).
    ///
    /// For non-HDR sources the `hdr` argument is ignored and the call
    /// behaves like [`convert_to`](PixelBufferConvertExt::convert_to).
    ///
    /// **Allocates** a new [`PixelBuffer`].
    fn convert_to_with_hdr_config(
        &self,
        target: PixelDescriptor,
        hdr: crate::HdrConfig,
    ) -> Result<PixelBuffer, At<crate::ConvertError>>;

    /// Estimate the cost of [`convert_to_sdr(target)`](Self::convert_to_sdr)
    /// without running it.
    ///
    /// For non-HDR sources this is equivalent to
    /// [`estimate_convert_to(&target)`](PixelBufferConvertExt::estimate_convert_to).
    ///
    /// For HDR sources (PQ / HLG transfer) the estimate accounts for
    /// the three legs of `convert_to_sdr` separately:
    /// 1. The source linearization (a `convert_to` into linear-light
    ///    F32 RGB/RGBA), needed before measuring CLL;
    /// 2. The
    ///    [`CllMeasure::measure_max`](crate::hdr::CllMeasure::measure_max)
    ///    scan over the linear source. Calibration is from the
    ///    2026-06-19 `measure_max_throughput` bench at
    ///    `benchmarks/measure_max_throughput_2026-06-19.md`
    ///    (~2.7 Gpix/s on the default-build SIMD path on Ryzen 9 7950X,
    ///    no `-C target-cpu=native`);
    /// 3. The downstream
    ///    [`convert_to_with_hdr_config`](Self::convert_to_with_hdr_config)
    ///    plan.
    ///
    /// The returned [`ResourceEstimate`](crate::ResourceEstimate)
    /// reports the sum of those legs in `wall_time_ms`, the **maximum**
    /// of their working sets in `peak_memory_bytes` (linear scratch is
    /// freed before the HDR plan allocates its destination), and
    /// concatenates their breakdowns into a single
    /// [`StepEstimate`](crate::StepEstimate) list. Confidence is the
    /// most-conservative tier across the legs (see
    /// [`EstimateConfidence`](crate::EstimateConfidence)). **±30 %
    /// design tolerance** per [`crate::estimate`].
    fn estimate_convert_to_sdr(&self, target: &PixelDescriptor) -> crate::ResourceEstimate;

    /// Estimate the cost of
    /// [`convert_to_with_hdr_config(target, hdr)`](Self::convert_to_with_hdr_config)
    /// without running it.
    ///
    /// Builds the same
    /// [`ConvertPlan`](crate::ConvertPlan) the wrapper would
    /// (via
    /// [`ConvertPlan::new_with_hdr_config`](crate::ConvertPlan::new_with_hdr_config))
    /// and walks its steps. **±30 % design tolerance** per
    /// [`crate::estimate`]; check
    /// [`ResourceEstimate::confidence`](crate::ResourceEstimate::confidence)
    /// for the calibration tier (the BT.2446-A tone-map cell is
    /// calibrated from `benchmarks/bt2446a_throughput_2026-06-20.md`;
    /// the OKLch soft-compress cell is heuristic).
    fn estimate_convert_to_with_hdr_config(
        &self,
        target: &PixelDescriptor,
        hdr: crate::HdrConfig,
    ) -> crate::ResourceEstimate;
}

#[cfg(feature = "hdr-experimental")]
impl PixelBufferHdrConvertExt for PixelBuffer {
    #[track_caller]
    fn convert_to_sdr(
        &self,
        target: PixelDescriptor,
    ) -> Result<PixelBuffer, At<crate::ConvertError>> {
        use crate::hdr::{CllMeasure, LightLevelMethod};
        use zenpixels::hdr::{ContentLightLevel, DiffuseWhite};

        let src_desc = self.descriptor();
        assert_not_cmyk(&src_desc);
        assert_not_cmyk(&target);

        // Non-HDR source: short-circuit to the regular convert_to path
        // (which now rejects HDR→SDR loudly, so this is purely the
        // "doesn't matter, source isn't HDR" branch).
        if !matches!(
            src_desc.transfer(),
            TransferFunction::Pq | TransferFunction::Hlg
        ) {
            return self.convert_to(target);
        }

        // Measure source peak. For PQ buffers we need linear-light F32
        // first (CllMeasure operates on relative-linear RGB f32).
        // For HLG, same thing.
        let lin_desc = PixelDescriptor::new_full(
            ChannelType::F32,
            if src_desc.has_alpha() {
                ChannelLayout::Rgba
            } else {
                ChannelLayout::Rgb
            },
            src_desc.alpha(),
            TransferFunction::Linear,
            src_desc.primaries,
        );
        let linear_src = self.convert_to(lin_desc)?;
        let lin_slice = linear_src.as_slice();
        let diffuse_white = self
            .color_context()
            .and_then(|c| c.diffuse_white)
            .unwrap_or(DiffuseWhite::BT2408);
        let cll =
            ContentLightLevel::measure_max(lin_slice, diffuse_white, LightLevelMethod::MaxRgb)
                .unwrap_or(ContentLightLevel::new(1000, 0));
        let source_peak_nits = f32::from(cll.max_content_light_level).max(100.0);
        self.convert_to_with_hdr_config(
            target,
            crate::HdrConfig {
                source_peak_nits,
                ..crate::HdrConfig::default()
            },
        )
    }

    #[track_caller]
    fn convert_to_with_hdr_config(
        &self,
        target: PixelDescriptor,
        hdr: crate::HdrConfig,
    ) -> Result<PixelBuffer, At<crate::ConvertError>> {
        let src_desc = self.descriptor();
        assert_not_cmyk(&src_desc);
        assert_not_cmyk(&target);
        // Do NOT short-circuit on `src_desc == target` — the HDR-aware
        // constructor still needs to run the tone-map + soft-compress
        // chain when both descriptors are e.g. `RGBF32_LINEAR`. The plan
        // itself decides whether HDR work is needed based on the source's
        // transfer function (SDR-encoded sources fall through to plain
        // `ConvertPlan::new`).

        let plan = crate::ConvertPlan::new_with_hdr_config(src_desc, target, hdr).at()?;
        let mut converter = crate::RowConverter::from_plan(plan);

        let dst_stride = target.aligned_stride(self.width());
        let total = dst_stride
            .checked_mul(self.height() as usize)
            .ok_or_else(|| whereat::at!(crate::ConvertError::AllocationFailed))?;
        let mut out = alloc::vec![0u8; total];

        let src_slice = self.as_slice();
        for y in 0..self.height() {
            let src_row = src_slice.row(y);
            let dst_start = y as usize * dst_stride;
            let dst_end = dst_start + dst_stride;
            converter.convert_row(src_row, &mut out[dst_start..dst_end], self.width());
        }

        let mut buf = PixelBuffer::from_vec(out, self.width(), self.height(), target)
            .map_err_at(crate::ConvertError::from)?;
        if let Some(ctx) = self.color_context() {
            buf = buf.with_color_context(Arc::clone(ctx));
        }
        Ok(buf)
    }

    fn estimate_convert_to_sdr(&self, target: &PixelDescriptor) -> crate::ResourceEstimate {
        let src_desc = self.descriptor();
        // CMYK is rejected by the underlying plan builder. Surface
        // Unknown rather than panic so the estimator never aborts.
        if src_desc.color_model() == crate::ColorModel::Cmyk
            || target.color_model() == crate::ColorModel::Cmyk
        {
            return crate::ResourceEstimate::zero(crate::EstimateConfidence::Unknown);
        }

        // Non-HDR short-circuit: matches the runtime path.
        if !matches!(
            src_desc.transfer(),
            TransferFunction::Pq | TransferFunction::Hlg
        ) {
            return self.estimate_convert_to(target);
        }

        // HDR source: model the three legs.
        // (1) Linearize to F32 RGB/RGBA for the CLL scan.
        let lin_desc = PixelDescriptor::new_full(
            ChannelType::F32,
            if src_desc.has_alpha() {
                ChannelLayout::Rgba
            } else {
                ChannelLayout::Rgb
            },
            src_desc.alpha(),
            TransferFunction::Linear,
            src_desc.primaries,
        );
        let linearize_est = self.estimate_convert_to(&lin_desc);

        // (2) measure_max scan over the linear source. Default-build
        // SIMD throughput from benchmarks/measure_max_throughput_2026-06-19.md
        // is ~2.7 Gpix/s on the 7950X AVX2 path (no -C target-cpu=native).
        let pixels = u64::from(self.width()) * u64::from(self.height());
        let measure_step = crate::estimate::measure_max_step_estimate(pixels);

        // (3) Downstream HDR plan with a placeholder source peak
        // (the estimator's wall-clock model doesn't depend on the
        // exact peak — same step set, same pixel work either way).
        let hdr_est = self.estimate_convert_to_with_hdr_config(
            target,
            crate::HdrConfig {
                source_peak_nits: 1000.0,
                ..crate::HdrConfig::default()
            },
        );

        // Sum wall-clock across the three legs.
        let wall_time_ms = linearize_est.wall_time_ms + measure_step.time_ms + hdr_est.wall_time_ms;
        // Peak memory: the linear scratch is freed before the HDR plan
        // allocates its destination, so peak = max(linear, hdr).
        let peak_memory_bytes = linearize_est
            .peak_memory_bytes
            .max(hdr_est.peak_memory_bytes);

        // Concatenate per-step breakdowns; insert the measure_max
        // step between linearize and HDR plan steps so callers can
        // see it.
        let mut breakdown = alloc::vec::Vec::with_capacity(
            linearize_est.breakdown.len() + 1 + hdr_est.breakdown.len(),
        );
        breakdown.extend(linearize_est.breakdown.iter().cloned());
        breakdown.push(measure_step);
        breakdown.extend(hdr_est.breakdown.iter().cloned());

        // Confidence: most-conservative tier (Unknown > Heuristic >
        // Calibrated). measure_max is calibrated from the bench above.
        let confidence = worst_confidence(&[linearize_est.confidence, hdr_est.confidence]);

        crate::ResourceEstimate {
            peak_memory_bytes,
            wall_time_ms,
            breakdown,
            confidence,
        }
    }

    fn estimate_convert_to_with_hdr_config(
        &self,
        target: &PixelDescriptor,
        hdr: crate::HdrConfig,
    ) -> crate::ResourceEstimate {
        let src = self.descriptor();
        if src.color_model() == crate::ColorModel::Cmyk
            || target.color_model() == crate::ColorModel::Cmyk
        {
            return crate::ResourceEstimate::zero(crate::EstimateConfidence::Unknown);
        }
        match crate::ConvertPlan::new_with_hdr_config(src, *target, hdr) {
            Ok(plan) => plan.estimate_resources(self.width(), self.height()),
            Err(_) => crate::ResourceEstimate::zero(crate::EstimateConfidence::Unknown),
        }
    }
}

/// Return the most-conservative confidence tier (worst case).
/// Order: Unknown > Heuristic > Calibrated.
#[cfg(feature = "hdr-experimental")]
fn worst_confidence(tiers: &[crate::EstimateConfidence]) -> crate::EstimateConfidence {
    let mut worst = crate::EstimateConfidence::Calibrated;
    for &t in tiers {
        worst = match (worst, t) {
            (crate::EstimateConfidence::Unknown, _) | (_, crate::EstimateConfidence::Unknown) => {
                crate::EstimateConfidence::Unknown
            }
            (crate::EstimateConfidence::Heuristic, _)
            | (_, crate::EstimateConfidence::Heuristic) => crate::EstimateConfidence::Heuristic,
            _ => crate::EstimateConfidence::Calibrated,
        };
    }
    worst
}

#[cfg(feature = "rgb")]
use zenpixels::buffer::Pixel;

#[cfg(feature = "rgb")]
impl PixelBufferConvertTypedExt for PixelBuffer {
    fn to_rgb8(&self) -> PixelBuffer<rgb::Rgb<u8>> {
        convert_to_typed(self, PixelDescriptor::RGB8_SRGB)
    }

    fn to_rgba8(&self) -> PixelBuffer<rgb::Rgba<u8>> {
        convert_to_typed(self, PixelDescriptor::RGBA8_SRGB)
    }

    fn to_gray8(&self) -> PixelBuffer<rgb::Gray<u8>> {
        convert_to_typed(self, PixelDescriptor::GRAY8_SRGB)
    }

    fn to_bgra8(&self) -> PixelBuffer<rgb::alt::BGRA<u8>> {
        convert_to_typed(self, PixelDescriptor::BGRA8_SRGB)
    }
}

/// Internal: convert to any target descriptor, returning a typed buffer.
#[cfg(feature = "rgb")]
fn convert_to_typed<Q: Pixel>(buf: &PixelBuffer, target: PixelDescriptor) -> PixelBuffer<Q> {
    use alloc::vec;
    let mut conv = crate::RowConverter::new(buf.descriptor(), target)
        .expect("RowConverter: no conversion path");
    let dst_bpp = target.bytes_per_pixel();
    let dst_stride = target.aligned_stride(buf.width());
    let total = dst_stride * buf.height() as usize;
    let mut out = vec![0u8; total];
    let src_slice = buf.as_slice();
    for y in 0..buf.height() {
        let src_row = src_slice.row(y);
        let dst_start = y as usize * dst_stride;
        let dst_end = dst_start + buf.width() as usize * dst_bpp;
        conv.convert_row(src_row, &mut out[dst_start..dst_end], buf.width());
    }
    // We need to construct PixelBuffer<Q> from raw parts.
    // Use from_vec to build the erased form, then reinterpret.
    let erased = PixelBuffer::from_vec(out, buf.width(), buf.height(), target)
        .expect("convert_to_typed: buffer construction failed");
    // Carry over color context
    let erased = if let Some(ctx) = buf.color_context() {
        erased.with_color_context(Arc::clone(ctx))
    } else {
        erased
    };
    erased
        .try_typed::<Q>()
        .expect("convert_to_typed: type mismatch after conversion")
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CMYK guard tests ---

    #[test]
    #[should_panic(expected = "CMYK pixel data cannot be processed")]
    fn cmyk_rejected_by_convert_to() {
        let cmyk_data = vec![0u8; 4 * 4]; // 4 pixels
        let buf = PixelBuffer::from_vec(cmyk_data, 2, 2, PixelDescriptor::CMYK8).unwrap();
        let _ = buf.convert_to(PixelDescriptor::RGB8_SRGB);
    }

    #[test]
    #[should_panic(expected = "CMYK pixel data cannot be processed")]
    fn cmyk_rejected_as_convert_target() {
        let rgb_data = vec![0u8; 3 * 4]; // 4 pixels
        let buf = PixelBuffer::from_vec(rgb_data, 2, 2, PixelDescriptor::RGB8_SRGB).unwrap();
        let _ = buf.convert_to(PixelDescriptor::CMYK8);
    }

    // --- TransferFunction linearize/delinearize tests ---

    #[test]
    fn srgb_linearize_roundtrip() {
        let tf = TransferFunction::Srgb;
        for &v in &[0.0, 0.04045, 0.1, 0.5, 0.73, 1.0] {
            let lin = tf.linearize(v);
            let back = tf.delinearize(lin);
            assert!(
                (v - back).abs() < 1e-5,
                "sRGB roundtrip failed for {v}: linearize={lin}, delinearize={back}"
            );
        }
    }

    #[test]
    fn pq_linearize_roundtrip() {
        let tf = TransferFunction::Pq;
        // linear-srgb 0.6 rational poly: ~3e-4 roundtrip error at low signal.
        // Tighten to 1e-5 after upgrading to linear-srgb with two-range EOTF.
        for &v in &[0.0, 0.1, 0.5, 0.75, 1.0] {
            let lin = tf.linearize(v);
            let back = tf.delinearize(lin);
            assert!(
                (v - back).abs() < 5e-4,
                "PQ roundtrip failed for {v}: linearize={lin}, delinearize={back}"
            );
        }
    }

    #[test]
    fn hlg_linearize_roundtrip() {
        let tf = TransferFunction::Hlg;
        for &v in &[0.0, 0.1, 0.3, 0.5, 0.8, 1.0] {
            let lin = tf.linearize(v);
            let back = tf.delinearize(lin);
            assert!(
                (v - back).abs() < 1e-4,
                "HLG roundtrip failed for {v}: linearize={lin}, delinearize={back}"
            );
        }
    }

    #[test]
    fn linear_identity() {
        let tf = TransferFunction::Linear;
        for &v in &[0.0, 0.5, 1.0] {
            assert_eq!(tf.linearize(v), v);
            assert_eq!(tf.delinearize(v), v);
        }
    }

    // --- ColorPrimaries XYZ matrix tests ---

    #[test]
    fn xyz_matrix_availability() {
        assert!(ColorPrimaries::Bt709.to_xyz_matrix().is_some());
        assert!(ColorPrimaries::Bt709.from_xyz_matrix().is_some());
        assert!(ColorPrimaries::DisplayP3.to_xyz_matrix().is_some());
        assert!(ColorPrimaries::Bt2020.to_xyz_matrix().is_some());
        assert!(ColorPrimaries::Unknown.to_xyz_matrix().is_none());
        assert!(ColorPrimaries::Unknown.from_xyz_matrix().is_none());
    }

    #[test]
    fn xyz_roundtrip_bt709() {
        let to = ColorPrimaries::Bt709.to_xyz_matrix().unwrap();
        let from = ColorPrimaries::Bt709.from_xyz_matrix().unwrap();
        let rgb = [0.5f32, 0.3, 0.8];
        let mut v = rgb;
        crate::gamut::apply_matrix_f32(&mut v, to);
        crate::gamut::apply_matrix_f32(&mut v, from);
        for c in 0..3 {
            assert!(
                (v[c] - rgb[c]).abs() < 1e-4,
                "XYZ roundtrip BT.709 ch{c}: {:.6} vs {:.6}",
                v[c],
                rgb[c]
            );
        }
    }

    // --- Bt709 and Unknown transfer function tests ---

    #[test]
    fn bt709_linearize_roundtrip() {
        let tf = TransferFunction::Bt709;
        for &v in &[0.0, 0.04045, 0.1, 0.5, 0.73, 1.0] {
            let lin = tf.linearize(v);
            let back = tf.delinearize(lin);
            assert!(
                (v - back).abs() < 1e-5,
                "BT.709 roundtrip failed for {v}: linearize={lin}, delinearize={back}"
            );
        }
    }

    #[test]
    fn unknown_transfer_identity() {
        let tf = TransferFunction::Unknown;
        for &v in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(
                tf.linearize(v),
                v,
                "Unknown linearize should be identity for {v}"
            );
            assert_eq!(
                tf.delinearize(v),
                v,
                "Unknown delinearize should be identity for {v}"
            );
        }
    }

    // --- PixelBufferConvertExt tests ---

    use super::PixelBufferConvertExt;

    #[test]
    fn convert_to_identity() {
        let data = vec![100u8, 150, 200, 50, 100, 150];
        let buf = PixelBuffer::from_vec(data.clone(), 2, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        let out = buf.convert_to(PixelDescriptor::RGB8_SRGB).unwrap();
        assert_eq!(out.descriptor(), PixelDescriptor::RGB8_SRGB);
        assert_eq!(out.width(), 2);
        assert_eq!(out.height(), 1);
        assert_eq!(&out.as_slice().row(0)[..6], &data[..]);
    }

    #[test]
    fn convert_to_rgba8() {
        let data = vec![100u8, 150, 200, 50, 100, 150];
        let buf = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        let out = buf.convert_to(PixelDescriptor::RGBA8_SRGB).unwrap();
        assert_eq!(out.descriptor(), PixelDescriptor::RGBA8_SRGB);
        let slice = out.as_slice();
        let row = slice.row(0);
        // Pixel 0: R=100, G=150, B=200, A=255
        assert_eq!(row[0], 100);
        assert_eq!(row[1], 150);
        assert_eq!(row[2], 200);
        assert_eq!(row[3], 255);
        // Pixel 1: R=50, G=100, B=150, A=255
        assert_eq!(row[4], 50);
        assert_eq!(row[5], 100);
        assert_eq!(row[6], 150);
        assert_eq!(row[7], 255);
    }

    #[test]
    fn try_add_alpha_rgb() {
        let data = vec![100u8, 150, 200, 50, 100, 150];
        let buf = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        let out = buf.try_add_alpha().unwrap();
        // Should now be RGBA with straight alpha
        assert_eq!(
            out.descriptor().layout(),
            zenpixels::descriptor::ChannelLayout::Rgba
        );
        let slice = out.as_slice();
        let row = slice.row(0);
        assert_eq!(row[3], 255);
        assert_eq!(row[7], 255);
    }

    #[test]
    fn try_widen_to_u16() {
        let data = vec![100u8, 150, 200, 50, 100, 150];
        let buf = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        let out = buf.try_widen_to_u16().unwrap();
        assert_eq!(
            out.descriptor().channel_type(),
            zenpixels::descriptor::ChannelType::U16
        );
        let slice = out.as_slice();
        let row = slice.row(0);
        // U16 little-endian: value * 257
        for (i, &expected_u8) in [100u8, 150, 200, 50, 100, 150].iter().enumerate() {
            let lo = row[i * 2];
            let hi = row[i * 2 + 1];
            let val = u16::from_le_bytes([lo, hi]);
            let expected = expected_u8 as u16 * 257;
            assert_eq!(
                val, expected,
                "channel {i}: expected {expected} (u8={expected_u8}*257), got {val}"
            );
        }
    }

    #[test]
    fn linearize_srgb_to_linear_f32() {
        let data = vec![128u8, 128, 128, 64, 64, 64];
        let buf = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        let lin = buf.linearize().unwrap();
        assert_eq!(lin.descriptor().transfer(), TransferFunction::Linear);
        assert_eq!(
            lin.descriptor().channel_type(),
            zenpixels::descriptor::ChannelType::F32
        );
        assert_eq!(lin.descriptor().primaries, ColorPrimaries::Bt709);
        // sRGB 128/255 ≈ 0.502 → linear ≈ 0.216
        let slice = lin.as_slice();
        let row = slice.row(0);
        let r = f32::from_le_bytes([row[0], row[1], row[2], row[3]]);
        assert!(
            (r - 0.216).abs() < 0.01,
            "sRGB 128 should linearize to ~0.216, got {r}"
        );
    }

    #[test]
    fn delinearize_linear_to_srgb() {
        // Create linear F32 buffer
        let linear_val: f32 = 0.216;
        let mut data = vec![0u8; 24]; // 2 pixels × 3 channels × 4 bytes
        for i in 0..6 {
            let bytes = linear_val.to_le_bytes();
            data[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }
        let buf = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGBF32_LINEAR).unwrap();
        let srgb = buf.delinearize(TransferFunction::Srgb).unwrap();
        assert_eq!(srgb.descriptor().transfer(), TransferFunction::Srgb);
        // Linear 0.216 → sRGB ≈ 0.502
        let slice = srgb.as_slice();
        let row = slice.row(0);
        let r = f32::from_le_bytes([row[0], row[1], row[2], row[3]]);
        assert!(
            (r - 0.502).abs() < 0.01,
            "linear 0.216 should delinearize to ~0.502, got {r}"
        );
    }

    #[test]
    fn linearize_delinearize_roundtrip() {
        let data = vec![100u8, 150, 200, 50, 100, 150];
        let buf = PixelBuffer::from_vec(data.clone(), 2, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        let lin = buf.linearize().unwrap();
        // Now delinearize back to sRGB F32
        let back = lin.delinearize(TransferFunction::Srgb).unwrap();
        // Values should round-trip within F32 precision
        let slice = back.as_slice();
        let row = slice.row(0);
        let r = f32::from_le_bytes([row[0], row[1], row[2], row[3]]);
        let expected = 100.0 / 255.0;
        assert!(
            (r - expected).abs() < 0.005,
            "roundtrip pixel 0 R: expected ~{expected}, got {r}"
        );
    }

    #[test]
    fn linearize_preserves_alpha() {
        let data = vec![100u8, 150, 200, 128, 50, 100, 150, 64];
        let buf = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGBA8_SRGB).unwrap();
        let lin = buf.linearize().unwrap();
        assert_eq!(
            lin.descriptor().layout(),
            zenpixels::descriptor::ChannelLayout::Rgba
        );
        assert!(lin.descriptor().alpha().is_some());
    }

    #[test]
    fn linearize_preserves_primaries() {
        let data = vec![100u8, 150, 200, 50, 100, 150];
        let desc = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
        let buf = PixelBuffer::from_vec(data, 2, 1, desc).unwrap();
        let lin = buf.linearize().unwrap();
        assert_eq!(lin.descriptor().primaries, ColorPrimaries::DisplayP3);
    }

    #[test]
    fn linearize_already_linear_is_identity() {
        let val: f32 = 0.5;
        let mut data = vec![0u8; 12]; // 1 pixel × 3 channels × 4 bytes
        for i in 0..3 {
            data[i * 4..i * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }
        let buf = PixelBuffer::from_vec(data, 1, 1, PixelDescriptor::RGBF32_LINEAR).unwrap();
        let lin = buf.linearize().unwrap();
        let slice = lin.as_slice();
        let row = slice.row(0);
        let r = f32::from_le_bytes([row[0], row[1], row[2], row[3]]);
        assert!(
            (r - val).abs() < 1e-6,
            "already-linear should be identity, got {r}"
        );
    }

    #[test]
    fn try_narrow_to_u8() {
        // Create RGB16 buffer with known values
        let values: [u16; 6] = [
            100 * 257,
            150 * 257,
            200 * 257,
            50 * 257,
            100 * 257,
            150 * 257,
        ];
        let mut data = vec![0u8; 12];
        for (i, &v) in values.iter().enumerate() {
            let bytes = v.to_le_bytes();
            data[i * 2] = bytes[0];
            data[i * 2 + 1] = bytes[1];
        }
        let buf = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGB16_SRGB).unwrap();
        let out = buf.try_narrow_to_u8().unwrap();
        assert_eq!(
            out.descriptor().channel_type(),
            zenpixels::descriptor::ChannelType::U8
        );
        let slice = out.as_slice();
        let row = slice.row(0);
        assert_eq!(row[0], 100);
        assert_eq!(row[1], 150);
        assert_eq!(row[2], 200);
        assert_eq!(row[3], 50);
        assert_eq!(row[4], 100);
        assert_eq!(row[5], 150);
    }

    #[test]
    #[cfg(feature = "rgb")]
    fn to_rgb8() {
        // Start with RGBA8 buffer, convert to typed RGB8
        let data = vec![100u8, 150, 200, 255, 50, 100, 150, 255];
        let buf = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGBA8_SRGB).unwrap();
        let typed: PixelBuffer<rgb::Rgb<u8>> = buf.to_rgb8();
        assert_eq!(typed.width(), 2);
        assert_eq!(typed.height(), 1);
        let slice = typed.as_slice();
        let row = slice.row(0);
        // Alpha should be dropped: 3 bytes per pixel
        assert_eq!(row[0], 100);
        assert_eq!(row[1], 150);
        assert_eq!(row[2], 200);
        assert_eq!(row[3], 50);
        assert_eq!(row[4], 100);
        assert_eq!(row[5], 150);
    }

    #[test]
    #[cfg(feature = "rgb")]
    fn to_rgba8() {
        let data = vec![100u8, 150, 200, 50, 100, 150];
        let buf = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        let typed: PixelBuffer<rgb::Rgba<u8>> = buf.to_rgba8();
        assert_eq!(typed.width(), 2);
        assert_eq!(typed.height(), 1);
        let slice = typed.as_slice();
        let row = slice.row(0);
        // RGB -> RGBA with alpha=255
        assert_eq!(row[0], 100);
        assert_eq!(row[1], 150);
        assert_eq!(row[2], 200);
        assert_eq!(row[3], 255);
        assert_eq!(row[4], 50);
        assert_eq!(row[5], 100);
        assert_eq!(row[6], 150);
        assert_eq!(row[7], 255);
    }

    #[test]
    #[cfg(feature = "rgb")]
    fn to_gray8() {
        let data = vec![100u8, 150, 200, 50, 100, 150];
        let buf = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        let typed: PixelBuffer<rgb::Gray<u8>> = buf.to_gray8();
        assert_eq!(typed.width(), 2);
        assert_eq!(typed.height(), 1);
        let slice = typed.as_slice();
        let row = slice.row(0);
        // Gray values should be luminance-weighted, not zero
        assert!(row[0] > 0, "gray pixel 0 should be non-zero");
        assert!(row[1] > 0, "gray pixel 1 should be non-zero");
    }

    #[test]
    #[cfg(feature = "rgb")]
    fn to_bgra8() {
        let data = vec![100u8, 150, 200, 50, 100, 150];
        let buf = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        let typed: PixelBuffer<rgb::alt::BGRA<u8>> = buf.to_bgra8();
        assert_eq!(typed.width(), 2);
        assert_eq!(typed.height(), 1);
        let slice = typed.as_slice();
        let row = slice.row(0);
        // BGRA layout: B, G, R, A
        // Pixel 0: R=100, G=150, B=200 -> BGRA = 200, 150, 100, 255
        assert_eq!(row[0], 200);
        assert_eq!(row[1], 150);
        assert_eq!(row[2], 100);
        assert_eq!(row[3], 255);
        // Pixel 1: R=50, G=100, B=150 -> BGRA = 150, 100, 50, 255
        assert_eq!(row[4], 150);
        assert_eq!(row[5], 100);
        assert_eq!(row[6], 50);
        assert_eq!(row[7], 255);
    }
}
