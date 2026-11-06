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
pub trait TransferFunctionExt: crate::sealed::Sealed {
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
            Self::Gamma22 => linear_srgb::default::gamma_to_linear(v, 563.0 / 256.0),
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
            Self::Gamma22 => linear_srgb::default::linear_to_gamma(v, 563.0 / 256.0),
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
pub trait ColorPrimariesExt: crate::sealed::Sealed {
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

/// Retain an ICC only while its samples' color meaning is unchanged.
fn context_after_conversion(
    source: &PixelSlice<'_>,
    target: PixelDescriptor,
) -> Option<Arc<zenpixels::ColorContext>> {
    let from = source.descriptor();
    let pq_decoded =
        from.transfer() == TransferFunction::Pq && target.transfer() == TransferFunction::Linear;
    let empty = zenpixels::ColorContext::default();
    let current = match source.color_context() {
        Some(c) => c.as_ref(),
        None if pq_decoded => &empty,
        None => return None,
    };
    if from.transfer() == target.transfer()
        && from.primaries == target.primaries
        && from.signal_range == target.signal_range
    {
        return source.color_context().map(Arc::clone);
    }
    let mut next = zenpixels::ColorContext::default();
    next.diffuse_white = if pq_decoded {
        Some(zenpixels::hdr::DiffuseWhite::new(10_000.0))
    } else {
        current.diffuse_white
    };
    if let (Some(p), Some(t)) = (target.primaries.to_cicp(), target.transfer().to_cicp()) {
        next.cicp = Some(zenpixels::Cicp::new(
            p,
            t,
            0,
            target.signal_range == zenpixels::SignalRange::Full,
        ));
    }
    Some(Arc::new(next))
}

// ---------------------------------------------------------------------------
// PixelBufferConvertExt
// ---------------------------------------------------------------------------

use alloc::sync::Arc;
#[cfg(feature = "bt2446a")]
use alloc::vec::Vec;
use whereat::{At, ResultAtExt};
use zenpixels::PixelDescriptor;
use zenpixels::buffer::{PixelBuffer, PixelCow, PixelSlice, PixelSliceMut};
use zenpixels::descriptor::{AlphaMode, ChannelLayout, ChannelType};

/// Adds format conversion methods to type-erased [`PixelBuffer`].
pub trait PixelBufferConvertExt: crate::sealed::Sealed + Sized {
    /// Convert pixel data to a different layout and depth.
    ///
    /// Uses [`RowConverter`](crate::RowConverter) for transfer-function-aware
    /// conversion. Layout/depth-only changes retain current color metadata;
    /// color changes replace it with the destination signaling. An attached ICC
    /// requiring a full CMS returns `NeedsCms`; use `finalize_for_output_with`.
    /// Contradictory current CICP/descriptor signaling is refused.
    ///
    /// **Allocates** a new [`PixelBuffer`].
    fn convert_to(&self, target: PixelDescriptor) -> Result<PixelBuffer, At<crate::ConvertError>>;

    /// Convert with explicit loss and HDR intent. Source context supplies a
    /// missing white anchor; HDR-to-SDR buffers measure their peak by default.
    fn convert_to_with_options(
        &self,
        target: PixelDescriptor,
        options: &zenpixels::ConvertOptions,
    ) -> Result<PixelBuffer, At<crate::ConvertError>>;

    /// Change format in place when no conversion math is needed, with no allocation.
    ///
    /// Handles metadata-only changes, the `Rgba8` and `Bgra8` byte swap, and removing a
    /// padding or declared-opaque alpha lane. Anything else returns
    /// [`ConvertError::NoPath`](crate::ConvertError::NoPath) with the buffer untouched:
    /// fall back to [`convert_to`](Self::convert_to). Live alpha is refused, so use
    /// [`reduce_to_load_bearing_format_in_place`](crate::PixelBufferLoadBearingExt::reduce_to_load_bearing_format_in_place)
    /// to scan for it and drop it when it's opaque.
    fn try_adapt_in_place(
        &mut self,
        target: PixelDescriptor,
    ) -> Result<(), At<crate::ConvertError>>;

    /// Physically apply `orientation`, reusing this allocation where possible.
    ///
    /// Identity returns the buffer untouched. Orientations permute in place, with
    /// a visited scratch for non-square transposes; pixels wider than 16 bytes
    /// fall back to a copy.
    fn into_oriented(
        self,
        orientation: zenpixels::Orientation,
    ) -> Result<PixelBuffer, At<crate::ConvertError>>;

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
}

/// Reject conversions that require a CMS plugin from these
/// no-CMS-argument extension entry points.
///
/// The trait-level methods (`convert_to`, `try_widen_to_u16`, …) don't
/// take a [`PluggableCms`](crate::cms::PluggableCms), so CMYK / Lab /
/// XYZ / any other non-native color model surfaces as a typed
/// [`ConvertError::NeedsCms`] here — pre-0.2.16 this was an
/// `assert_not_cmyk` panic. Callers that need CMS dispatch should
/// build a [`RowConverter`](crate::RowConverter) directly via
/// [`new_explicit_with_cms`](crate::RowConverter::new_explicit_with_cms).
#[inline]
fn check_needs_cms(
    from: &PixelDescriptor,
    to: &PixelDescriptor,
) -> Result<(), At<crate::ConvertError>> {
    if crate::convert::needs_cms_for_color_model(from, to) {
        return Err(whereat::at!(crate::ConvertError::NeedsCms {
            from: *from,
            to: *to,
        }));
    }
    Ok(())
}

/// Evidence for samples produced from `exact`-evidenced samples by a
/// metadata-proven reversible representation change, or `None`.
///
/// Swizzles, gray expansion and alpha insertion copy sample values; inserted
/// alpha is opaque, which lies on every grid. U8→U16 multiplies by 257, which
/// keeps an N-bit grid only when N divides 8 and otherwise lands on the 8-bit
/// grid. Anything else, including alpha bytes that are only padding, gets none.
fn exact_after_preserving(
    from: PixelDescriptor,
    to: PixelDescriptor,
    exact: zenpixels::sample::ExactDepth,
) -> Option<zenpixels::sample::ExactDepth> {
    if from.layout().has_alpha() && !from.has_alpha() {
        return None;
    }
    // The carry rule below models the full-scale U8 to U16 kernel (x257).
    // Narrow anchors scale by 2^(M-N) instead, and ConvertPlan refuses narrow
    // depth changes and range crossings. Never carry evidence across either,
    // even if a later plan accepts one.
    if from.signal_range != to.signal_range
        || (from.signal_range != zenpixels::SignalRange::Full
            && from.channel_type() != to.channel_type())
    {
        return None;
    }
    if crate::ConvertPlan::new_preserving_samples(from, to).is_err() {
        return None;
    }
    let carry = |bits: u8| match (from.channel_type(), to.channel_type()) {
        (ChannelType::U8, ChannelType::U8) | (ChannelType::U16, ChannelType::U16) => Some(bits),
        (ChannelType::U8, ChannelType::U16) => {
            Some(if 8u8.is_multiple_of(bits) { bits } else { 8 })
        }
        _ => None,
    };
    let color = carry(exact.color_bits())?;
    let alpha = if from.has_alpha() {
        carry(exact.alpha_bits())?
    } else {
        1
    };
    zenpixels::sample::ExactDepth::new(to.channel_type(), to.signal_range, color, alpha).ok()
}

/// A converter for `source`, using its current color when it has an ICC profile.
fn converter_for(
    source: &PixelSlice<'_>,
    target: PixelDescriptor,
    options: &zenpixels::ConvertOptions,
) -> Result<crate::RowConverter, At<crate::ConvertError>> {
    let src_desc = source.descriptor();
    check_needs_cms(&src_desc, &target)?;
    let mut options = *options;
    if options.sdr_white().is_none()
        && let Some(white) = source.color_context().and_then(|c| c.diffuse_white)
    {
        options = options.with_sdr_white(white);
    }
    let source_profile =
        crate::output::current_profile_for(src_desc, source.color_context().map(Arc::as_ref))?;
    let changes_color =
        src_desc.transfer() != target.transfer() || src_desc.primaries != target.primaries;
    if changes_color && matches!(source_profile, zenpixels::ColorProfileSource::Icc(_)) {
        // The profile-only CMS interface cannot express absolute-light intent.
        if matches!(
            (src_desc.transfer(), target.transfer()),
            (TransferFunction::Pq | TransferFunction::Hlg, _)
                | (_, TransferFunction::Pq | TransferFunction::Hlg)
        ) {
            return Err(whereat::at!(crate::ConvertError::NeedsCms {
                from: src_desc,
                to: target
            }));
        }
        crate::RowConverter::new_with_sources(
            src_desc,
            target,
            source_profile,
            target.color_profile_source(),
            &options,
            None,
        )
    } else {
        crate::RowConverter::new_explicit(src_desc, target, &options).at()
    }
}

fn default_options() -> zenpixels::ConvertOptions {
    zenpixels::ConvertOptions::permissive()
        .with_alpha_policy(zenpixels::AlphaPolicy::DiscardUnchecked)
}

/// Adds whole-image conversion methods to type-erased [`PixelSlice`].
///
/// Every method handles strided rows. Row-at-a-time work goes through
/// [`RowConverter`](crate::RowConverter).
pub trait PixelSliceConvertExt<'a>: crate::sealed::Sealed {
    /// Convert to `target`. **Allocates** a new `PixelBuffer`.
    ///
    /// Uses the slice's current ICC profile when it has one. Conversions that need
    /// a CMS return [`ConvertError::NeedsCms`](crate::ConvertError::NeedsCms).
    fn convert_to(&self, target: PixelDescriptor) -> Result<PixelBuffer, At<crate::ConvertError>>;

    /// Convert with explicit loss and HDR intent. Source context supplies a
    /// missing white anchor; HDR-to-SDR buffers measure their peak by default.
    fn convert_to_with_options(
        &self,
        target: PixelDescriptor,
        options: &zenpixels::ConvertOptions,
    ) -> Result<PixelBuffer, At<crate::ConvertError>>;

    /// Convert into `dst`, whose descriptor is the target. No output allocation.
    ///
    /// # Errors
    ///
    /// [`ConvertError::BufferSize`](crate::ConvertError::BufferSize) when `dst` has
    /// different dimensions, leaving it untouched, and as for
    /// [`convert_to`](Self::convert_to).
    fn convert_into<'d>(
        &self,
        dst: PixelSliceMut<'d>,
    ) -> Result<PixelSliceMut<'d>, At<crate::ConvertError>>;

    /// Pick the cheapest of `supported` and return these pixels in it, borrowed
    /// when they already match.
    ///
    /// Use [`ConvertOptions::permissive`](zenpixels::ConvertOptions::permissive)
    /// with `AlphaPolicy::DiscardUnchecked` to let any conversion through.
    ///
    /// # Errors
    ///
    /// [`ConvertError::EmptyFormatList`](crate::ConvertError::EmptyFormatList), any
    /// refusal from `options`, and allocation failure.
    fn adapt_for_encode(
        &self,
        supported: &[PixelDescriptor],
        options: &zenpixels::ConvertOptions,
    ) -> Result<PixelCow<'a>, At<crate::ConvertError>>;

    /// Scan visible alpha, stopping at the first non-opaque pixel. Padding is
    /// ignored, and nothing allocates. Floating alpha must equal 1.0 exactly.
    ///
    /// Run this before planning with `AlphaPolicy::DiscardUnchecked` when a
    /// destination must stay unchanged on failure.
    ///
    /// # Errors
    ///
    /// [`ConvertError::AlphaNotOpaque`](crate::ConvertError::AlphaNotOpaque).
    fn check_opaque(&self) -> Result<(), At<crate::ConvertError>>;

    /// Read these samples for what a decoder couldn't state: 8-bit content in `u16` storage, or
    /// whether alpha is really used. Row padding is ignored, and nothing allocates.
    ///
    /// This is a whole pass, so it is the caller's choice, and `stop` is checked between rows. A
    /// format with no canonical grid to measure against, such as a float, gets
    /// [`SourcePrecision::from_storage`](zenpixels::precision::SourcePrecision::from_storage). The
    /// result describes these samples, so call it before anything edits them.
    ///
    /// # Errors
    ///
    /// The [`StopReason`](enough::StopReason) when `stop` fires.
    fn scan_source_precision(
        &self,
        stop: &dyn enough::Stop,
    ) -> Result<zenpixels::precision::SourcePrecision, At<enough::StopReason>>;

    /// Quantize PQ-anchored linear pixels to `target`. **Allocates**.
    ///
    /// The anchor comes from the slice's `ColorContext`. See `docs/hdr-anchor-api-design.md`.
    fn quantize_to(&self, target: PixelDescriptor) -> Result<PixelBuffer, At<crate::ConvertError>>;

    /// Physically apply `orientation`. **Allocates**, also for identity.
    ///
    /// Width and height swap for the four transposing orientations. Whole pixels
    /// move, so any format works. To skip the copy for identity, check
    /// `orientation.is_identity()`.
    fn oriented(
        &self,
        orientation: zenpixels::Orientation,
    ) -> Result<PixelBuffer, At<crate::ConvertError>>;

    /// [`oriented`](Self::oriented) into a caller's buffer of the output dimensions.
    fn oriented_into(
        &self,
        orientation: zenpixels::Orientation,
        dst: PixelSliceMut<'_>,
    ) -> Result<(), At<crate::ConvertError>>;
}

impl<'a> PixelSliceConvertExt<'a> for PixelSlice<'a> {
    #[track_caller]
    fn convert_to(&self, target: PixelDescriptor) -> Result<PixelBuffer, At<crate::ConvertError>> {
        self.convert_to_with_options(target, &default_options())
    }

    #[track_caller]
    fn convert_to_with_options(
        &self,
        target: PixelDescriptor,
        options: &zenpixels::ConvertOptions,
    ) -> Result<PixelBuffer, At<crate::ConvertError>> {
        let src_desc = self.descriptor();
        let width = self.width();
        if src_desc == target {
            check_needs_cms(&src_desc, &target)?;
            crate::output::current_profile_for(src_desc, self.color_context().map(Arc::as_ref))?;
            let mut buf = PixelBuffer::try_new(width, self.rows(), target)
                .map_err_at(crate::ConvertError::from)?;
            let mut destination = buf.as_slice_mut();
            for y in 0..self.rows() {
                destination.row_mut(y).copy_from_slice(self.row(y));
            }
            if let Some(ctx) = self.color_context() {
                buf = buf.with_color_context(Arc::clone(ctx));
            }
            // The same samples under the same descriptor keep their evidence.
            buf.set_exact_depth(self.exact_depth())
                .expect("identical descriptors accept the same evidence");
            return Ok(buf);
        }
        let mut resolved = *options;
        if crate::convert::is_hdr_to_sdr(src_desc.transfer(), target.transfer()) {
            if src_desc.transfer() == TransferFunction::Hlg {
                return Err(whereat::at!(crate::ConvertError::HlgOotfNotImplemented));
            }
            if resolved.sdr_peak().is_none() {
                return Err(whereat::at!(crate::ConvertError::SdrPeakRequired));
            }
            let peak = match resolved.hdr_peak() {
                zenpixels::hdr::HdrPeak::Nits(peak) => peak,
                zenpixels::hdr::HdrPeak::Declared => self
                    .color_context()
                    .and_then(|c| c.hdr_peak)
                    .ok_or_else(|| whereat::at!(crate::ConvertError::HdrPeakRequired))?,
                zenpixels::hdr::HdrPeak::Measured => {
                    #[cfg(feature = "bt2446a")]
                    {
                        measured_peak(self, resolved.sdr_peak().expect("validated target peak"))?
                    }
                    #[cfg(not(feature = "bt2446a"))]
                    {
                        return Err(whereat::at!(crate::ConvertError::HdrFeatureRequired));
                    }
                }
                _ => return Err(whereat::at!(crate::ConvertError::HdrPeakRequired)),
            };
            resolved = resolved.with_hdr_peak(zenpixels::hdr::HdrPeak::Nits(peak));
        }
        let mut converter = converter_for(self, target, &resolved)?;
        let mut buf = PixelBuffer::try_new(width, self.rows(), target)
            .map_err_at(crate::ConvertError::from)?;
        let mut destination = buf.as_slice_mut();
        for y in 0..self.rows() {
            converter.try_convert_row(self.row(y), destination.row_mut(y), width)?;
        }
        let anchor = if crate::convert::is_hdr_to_sdr(src_desc.transfer(), target.transfer()) {
            resolved
                .sdr_peak()
                .map(|n| zenpixels::hdr::DiffuseWhite::new(n.nits()))
        } else {
            resolved
                .sdr_white()
                .or_else(|| self.color_context().and_then(|c| c.diffuse_white))
        };
        let context = context_after_conversion(self, target);
        if let Some(white) = anchor {
            let context = context
                .as_deref()
                .cloned()
                .unwrap_or_default()
                .with_diffuse_white(white);
            buf = buf.with_color_context(Arc::new(context));
        } else if let Some(context) = context {
            buf = buf.with_color_context(context);
        }
        if let Some(exact) = self
            .exact_depth()
            .and_then(|exact| exact_after_preserving(src_desc, target, exact))
        {
            buf.set_exact_depth(Some(exact))
                .expect("transferred evidence names the target storage and range");
        }
        Ok(buf)
    }

    #[track_caller]
    fn convert_into<'d>(
        &self,
        mut dst: PixelSliceMut<'d>,
    ) -> Result<PixelSliceMut<'d>, At<crate::ConvertError>> {
        if dst.width() != self.width() || dst.rows() != self.rows() {
            let bytes = |width: u32, rows: u32, descriptor: PixelDescriptor| {
                (width as usize)
                    .saturating_mul(rows as usize)
                    .saturating_mul(descriptor.bytes_per_pixel())
            };
            return Err(whereat::at!(crate::ConvertError::BufferSize {
                expected: bytes(self.width(), self.rows(), dst.descriptor()),
                actual: bytes(dst.width(), dst.rows(), dst.descriptor()),
            }));
        }
        let target = dst.descriptor();
        let mut converter = converter_for(self, target, &default_options())?;
        for y in 0..self.rows() {
            converter.try_convert_row(self.row(y), dst.row_mut(y), self.width())?;
        }
        // The pixels are new, so nothing the destination said about its old ones still holds.
        Ok(match context_after_conversion(self, target) {
            Some(ctx) => dst.with_color_context(ctx),
            None if dst.color_context().is_some() => {
                dst.with_color_context(Arc::new(zenpixels::ColorContext::default()))
            }
            None => dst,
        })
    }

    #[track_caller]
    fn adapt_for_encode(
        &self,
        supported: &[PixelDescriptor],
        options: &zenpixels::ConvertOptions,
    ) -> Result<PixelCow<'a>, At<crate::ConvertError>> {
        // A current ICC profile outranks the descriptor, as it does in `convert_to`.
        if let Some(target) = crate::adapt::encode_conversion_target(self.descriptor(), supported) {
            let source = self.descriptor();
            let profile =
                crate::output::current_profile_for(source, self.color_context().map(Arc::as_ref))?;
            let changes_color =
                source.transfer() != target.transfer() || source.primaries != target.primaries;
            if changes_color && matches!(profile, zenpixels::ColorProfileSource::Icc(_)) {
                let checked = crate::adapt::options_after_opacity_check(self, target, options)?;
                let mut converter = crate::RowConverter::new_with_sources(
                    source,
                    target,
                    profile,
                    target.color_profile_source(),
                    &checked,
                    None,
                )?;
                let mut output = PixelBuffer::try_new(self.width(), self.rows(), target)
                    .map_err(|_| whereat::at!(crate::ConvertError::AllocationFailed))?;
                {
                    let mut destination = output.as_slice_mut();
                    for y in 0..self.rows() {
                        converter.try_convert_row(
                            self.row(y),
                            destination.row_mut(y),
                            self.width(),
                        )?;
                    }
                }
                return Ok(PixelCow::Owned(
                    match context_after_conversion(self, target) {
                        Some(ctx) => output.with_color_context(ctx),
                        None => output,
                    },
                ));
            }
        }
        let adapted = crate::adapt::adapt_raw_rows_for_encode(
            self.as_strided_bytes(),
            self.descriptor(),
            self.width(),
            self.rows(),
            self.stride(),
            supported,
            options,
        )?;
        Ok(match adapted {
            PixelCow::Borrowed(borrowed) => PixelCow::Borrowed(match self.color_context() {
                Some(ctx) => borrowed.with_color_context(Arc::clone(ctx)),
                None => borrowed,
            }),
            PixelCow::Owned(owned) => {
                PixelCow::Owned(match context_after_conversion(self, owned.descriptor()) {
                    Some(ctx) => owned.with_color_context(ctx),
                    None => owned,
                })
            }
        })
    }

    #[track_caller]
    fn scan_source_precision(
        &self,
        stop: &dyn enough::Stop,
    ) -> Result<zenpixels::precision::SourcePrecision, At<enough::StopReason>> {
        crate::adapt::scan_source_precision(self, stop)
    }

    #[track_caller]
    fn check_opaque(&self) -> Result<(), At<crate::ConvertError>> {
        crate::adapt::check_opaque(self)
    }

    #[track_caller]
    fn quantize_to(&self, target: PixelDescriptor) -> Result<PixelBuffer, At<crate::ConvertError>> {
        crate::hdr::quantize_to(self.clone(), target)
    }

    #[track_caller]
    fn oriented(
        &self,
        orientation: zenpixels::Orientation,
    ) -> Result<PixelBuffer, At<crate::ConvertError>> {
        crate::orient::apply_orientation(self.clone(), orientation).map_err(|e| whereat::at!(e))
    }

    #[track_caller]
    fn oriented_into(
        &self,
        orientation: zenpixels::Orientation,
        dst: PixelSliceMut<'_>,
    ) -> Result<(), At<crate::ConvertError>> {
        crate::orient::apply_orientation_into(self.clone(), orientation, dst)
            .map_err(|e| whereat::at!(e))
    }
}

impl PixelBufferConvertExt for PixelBuffer {
    #[track_caller]
    fn convert_to(&self, target: PixelDescriptor) -> Result<PixelBuffer, At<crate::ConvertError>> {
        self.as_slice().convert_to(target)
    }

    #[track_caller]
    fn convert_to_with_options(
        &self,
        target: PixelDescriptor,
        options: &zenpixels::ConvertOptions,
    ) -> Result<PixelBuffer, At<crate::ConvertError>> {
        self.as_slice().convert_to_with_options(target, options)
    }

    #[track_caller]
    fn try_adapt_in_place(
        &mut self,
        target: PixelDescriptor,
    ) -> Result<(), At<crate::ConvertError>> {
        crate::adapt::try_adapt_in_place(self, target)
    }

    #[track_caller]
    fn into_oriented(
        mut self,
        orientation: zenpixels::Orientation,
    ) -> Result<PixelBuffer, At<crate::ConvertError>> {
        if orientation.is_identity() {
            return Ok(self);
        }
        if crate::orient::apply_orientation_in_place(&mut self, orientation).is_ok() {
            return Ok(self);
        }
        self.as_slice().oriented(orientation)
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
            PixelDescriptor::new(desc.channel_type(), target_layout, alpha, desc.transfer())
                .with_primaries(desc.primaries)
                .with_signal_range(desc.signal_range);
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
        )
        .with_primaries(desc.primaries)
        .with_signal_range(desc.signal_range);
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
        )
        .with_primaries(desc.primaries)
        .with_signal_range(desc.signal_range);
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
/// ([`convert_to_sdr_measuring_peak`](Self::convert_to_sdr_measuring_peak)).
///
/// Gated behind `hdr-experimental`.
#[cfg(feature = "hdr-experimental")]
pub trait PixelBufferHdrConvertExt: crate::sealed::Sealed {
    /// Explicitly measure peak luminance in a separate rowwise pass, then map
    /// to SDR. Allocates one output image plus reusable row scratch.
    fn convert_to_sdr_measuring_peak(
        &self,
        target: PixelDescriptor,
    ) -> Result<PixelBuffer, At<crate::ConvertError>>;

    /// Convert this HDR buffer to `target` with explicit HDR knobs.
    ///
    /// `hdr.source_peak_nits` is mandatory and parameterizes the BT.2446-A
    /// curve. `target_peak_nits` defaults to `100.0` (SDR), `gamut_knee`
    /// to `0.96` — start from
    /// [`HdrConfig::for_source_peak`](crate::HdrConfig::for_source_peak).
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
}

#[cfg(feature = "hdr-experimental")]
impl PixelBufferHdrConvertExt for PixelBuffer {
    fn convert_to_sdr_measuring_peak(
        &self,
        target: PixelDescriptor,
    ) -> Result<PixelBuffer, At<crate::ConvertError>> {
        self.convert_to_with_options(
            target,
            &default_options().with_sdr_peak(zenpixels::hdr::Nits::new(100.0)),
        )
    }

    #[track_caller]
    fn convert_to_with_hdr_config(
        &self,
        target: PixelDescriptor,
        hdr: crate::HdrConfig,
    ) -> Result<PixelBuffer, At<crate::ConvertError>> {
        let src_desc = self.descriptor();
        check_needs_cms(&src_desc, &target)?;
        if matches!(
            crate::output::current_profile(self)?,
            zenpixels::ColorProfileSource::Icc(_)
        ) {
            return Err(whereat::at!(crate::ConvertError::NeedsCms {
                from: src_desc,
                to: target
            }));
        }
        // Do NOT short-circuit on `src_desc == target` — the HDR-aware
        // constructor still needs to run the tone-map + soft-compress
        // chain when both descriptors are e.g. `RGBF32_LINEAR`. The plan
        // itself decides whether HDR work is needed based on the source's
        // transfer function (SDR-encoded sources fall through to plain
        // `ConvertPlan::new`).

        let mut plan = crate::ConvertPlan::new_with_hdr_config(src_desc, target, hdr).at()?;
        if let Some(white) = self.color_context().and_then(|c| c.diffuse_white) {
            plan = plan.with_linear_input_white(white, hdr.source_peak_nits);
        }
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
            converter.try_convert_row(src_row, &mut out[dst_start..dst_end], self.width())?;
        }

        let mut buf = PixelBuffer::from_vec(out, self.width(), self.height(), target)
            .map_err_at(crate::ConvertError::from)?;
        if matches!(
            src_desc.transfer(),
            TransferFunction::Linear | TransferFunction::Pq
        ) {
            let mut context = context_after_conversion(&self.as_slice(), target)
                .map(|c| (*c).clone())
                .unwrap_or_default();
            context.diffuse_white = Some(zenpixels::hdr::DiffuseWhite::new(hdr.target_peak_nits));
            buf = buf.with_color_context(Arc::new(context));
        } else if let Some(ctx) = context_after_conversion(&self.as_slice(), target) {
            buf = buf.with_color_context(ctx);
        }
        Ok(buf)
    }
}

/// Measure absolute MaxRGB luminance from current PQ samples, rowwise with a
/// reused F32 row. Metadata never substitutes for the measured peak.
#[cfg(feature = "bt2446a")]
fn measured_peak(
    source: &PixelSlice<'_>,
    black_frame_axis: zenpixels::hdr::Nits,
) -> Result<zenpixels::hdr::Nits, At<crate::ConvertError>> {
    let from = source.descriptor();
    crate::output::current_profile_for(from, source.color_context().map(Arc::as_ref))?;
    if source.color_context().is_some_and(|c| c.icc.is_some()) {
        return Err(whereat::at!(crate::ConvertError::NeedsCms {
            from,
            to: from
        }));
    }
    let linear = PixelDescriptor::new_full(
        ChannelType::F32,
        from.layout(),
        from.alpha().map(|_| AlphaMode::Straight),
        TransferFunction::Linear,
        from.primaries,
    )
    .with_signal_range(from.signal_range);
    let options = default_options().with_sdr_white(zenpixels::hdr::DiffuseWhite::new(10_000.0));
    let mut converter = crate::RowConverter::new_explicit(from, linear, &options)?;
    let count = (source.width() as usize)
        .checked_mul(linear.channels())
        .ok_or_else(|| whereat::at!(crate::ConvertError::AllocationFailed))?;
    let mut row = Vec::new();
    row.try_reserve_exact(count)
        .map_err(|_| whereat::at!(crate::ConvertError::AllocationFailed))?;
    row.resize(count, 0_f32);
    let mut peak = 0_f32;
    for y in 0..source.rows() {
        converter.try_convert_row(
            source.row(y),
            bytemuck::cast_slice_mut(&mut row),
            source.width(),
        )?;
        for pixel in row.chunks_exact(linear.channels()) {
            for &value in &pixel[..usize::from(linear.color_model().color_channels())] {
                if !value.is_finite() {
                    return Err(whereat::at!(crate::ConvertError::HdrPeakRequired));
                }
                peak = peak.max(value * 10_000.0);
            }
        }
    }
    // All-black samples map to zero on every finite positive axis. Reuse the
    // caller's explicit target axis so mapper coefficients remain well defined.
    Ok(zenpixels::hdr::Nits::try_new(peak).unwrap_or(black_frame_axis))
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
    buf.convert_to(target)
        .expect("convert_to_typed: conversion failed; use convert_to for fallible conversion")
        .try_typed::<Q>()
        .expect("convert_to_typed: type mismatch after conversion")
}

/// Conversion of one [`Color`](zenpixels::Color).
pub trait ColorConvertExt: crate::sealed::Sealed + Sized {
    /// This color in `target`'s format, transfer and primaries, without a one-pixel buffer.
    ///
    /// Linear converted to PQ or HLG takes 1.0 as the format's peak, which is what those
    /// transfers decode to.
    ///
    /// # Errors
    ///
    /// [`ConvertError::NoPath`](crate::ConvertError::NoPath) from an SDR transfer to PQ or HLG:
    /// neither says where SDR white belongs, and passing the signal through would put it at the
    /// peak, 10,000 cd/m² for PQ. For a matte, `CompositeOnto::reference_white` places it.
    ///
    /// Otherwise whatever [`RowConverter::new_explicit`](crate::RowConverter::new_explicit)
    /// refuses for this color's descriptor and `target`, or
    /// [`ConvertError::Buffer`](crate::ConvertError::Buffer) when `target`'s pixels are too wide
    /// for a color.
    fn try_convert_to(
        &self,
        target: PixelDescriptor,
        options: &zenpixels::ConvertOptions,
    ) -> Result<Self, whereat::At<crate::ConvertError>>;

    /// Parse a CSS color: hex, a name, `rgb()`, `hsl()`, `hwb()`, `color()`, `lab()`,
    /// `lch()`, `oklab()` or `oklch()`. Case doesn't matter.
    ///
    /// Spaces with no descriptor, such as `lab()`, resolve to linear BT.2020 `f32`, which
    /// keeps out-of-gamut values. See [`map_to_gamut`](Self::map_to_gamut) for output.
    ///
    /// ```
    /// use zenpixels::Color;
    /// use zenpixels_convert::ColorConvertExt;
    ///
    /// assert_eq!(Color::parse_css("#ff0000").unwrap(), Color::srgb8(255, 0, 0));
    /// assert_eq!(Color::parse_css("rebeccapurple").unwrap(), Color::srgb8(102, 51, 153));
    /// let wide = Color::parse_css("color(display-p3 1 0 0 / 50%)").unwrap();
    /// assert!(wide.descriptor().has_alpha());
    /// ```
    ///
    /// # Errors
    ///
    /// [`CssColorError::Unsupported`](crate::css::CssColorError::Unsupported) for forms that
    /// need context, such as `currentcolor`, and
    /// [`CssColorError::Invalid`](crate::css::CssColorError::Invalid) otherwise.
    fn parse_css(css: &str) -> Result<Self, crate::css::CssColorError>;

    /// This color in `target`, moved into its gamut if needed, keeping hue and lightness.
    ///
    /// Follows the CSS Color 4 gamut mapping algorithm: OKLCh chroma reduction with a
    /// 0.02 tolerance in OKLab, then a clip. For one color at a time; it's far slower
    /// per pixel than [`SoftCompress`](crate::hdr::SoftCompress) and not meant for images.
    /// Alpha is kept.
    ///
    /// # Errors
    ///
    /// [`ConvertError::NoPath`](crate::ConvertError::NoPath) for HDR transfers, unknown
    /// primaries, non-RGB targets, or primaries with no OKLab matrices here.
    fn map_to_gamut(
        &self,
        target: PixelDescriptor,
    ) -> Result<Self, whereat::At<crate::ConvertError>>;

    /// Whether both are the same color, whatever their formats.
    ///
    /// `==` asks whether format and samples match, so opaque red as `RGB8` and as `RGBA8`
    /// differ under it and are the same here. This compares linear light with 1e-6 of
    /// tolerance, which is below one 16-bit step, and above the rounding of a conversion.
    ///
    /// Fully transparent colors are all the same, since no composite can tell them apart.
    /// A color with no way to linear light, such as CMYK or an unknown transfer or primaries,
    /// is the same only as an identical one. SDR and HDR never match, nor do PQ and HLG, since
    /// their linear scales differ.
    ///
    /// ```
    /// use zenpixels::{Color, PixelDescriptor};
    /// use zenpixels_convert::ColorConvertExt;
    ///
    /// let red = Color::srgb8(255, 0, 0);
    /// assert!(red.same_color(&Color::srgba8(255, 0, 0, 255)));
    /// assert_ne!(red, Color::srgba8(255, 0, 0, 255));
    /// assert!(!red.same_color(&Color::srgb8(254, 0, 0)));
    /// ```
    fn same_color(&self, other: &Self) -> bool;
}

/// CSS Color 4 gamut mapping of linear RGB, in the RGB of `space`.
///
/// Clipping encoded or linear channels to `0..=1` gives the same color, since transfer
/// functions are monotonic and fix both ends, so this stays in linear light.
fn css_gamut_map(space: &crate::oklab::OklabSpace, rgb: [f32; 3]) -> [f32; 3] {
    const JND: f32 = 0.02;
    const EPSILON: f32 = 0.0001;
    const ROUNDOFF: f32 = 1e-6;
    let delta = |x: [f32; 3], y: [f32; 3]| {
        let d: [f32; 3] = core::array::from_fn(|i| x[i] - y[i]);
        libm::sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    };
    let clip = |c: [f32; 3]| c.map(|v| v.clamp(0.0, 1.0));
    let in_gamut = |c: [f32; 3]| c.iter().all(|v| (-ROUNDOFF..=1.0 + ROUNDOFF).contains(v));

    let origin @ [l, a, b] = space.from_rgb(rgb);
    if l >= 1.0 {
        return [1.0; 3];
    }
    if l <= 0.0 {
        return [0.0; 3];
    }
    if in_gamut(rgb) {
        return clip(rgb);
    }
    let mut clipped = clip(rgb);
    if delta(space.from_rgb(clipped), origin) < JND {
        return clipped;
    }
    let chroma = libm::sqrtf(a * a + b * b);
    let (ca, sa) = (a / chroma, b / chroma);
    let (mut min, mut max, mut min_in_gamut) = (0.0f32, chroma, true);
    while max - min > EPSILON {
        let c = (min + max) / 2.0;
        let current = space.to_rgb([l, c * ca, c * sa]);
        if min_in_gamut && in_gamut(current) {
            min = c;
            continue;
        }
        clipped = clip(current);
        let e = delta(space.from_rgb(clipped), space.from_rgb(current));
        if e < JND {
            if JND - e < EPSILON {
                return clipped;
            }
            min_in_gamut = false;
            min = c;
        } else {
            max = c;
        }
    }
    clipped
}

impl ColorConvertExt for zenpixels::Color {
    fn parse_css(css: &str) -> Result<Self, crate::css::CssColorError> {
        crate::css::parse(css)
    }

    fn same_color(&self, other: &Self) -> bool {
        use zenpixels::TransferFunction as Tf;
        if self == other {
            return true;
        }
        let scale = |d: PixelDescriptor| match d.transfer() {
            Tf::Pq => 1,
            Tf::Hlg => 2,
            _ => 0,
        };
        let known = |d: PixelDescriptor| {
            d.transfer() != Tf::Unknown && d.primaries != ColorPrimaries::Unknown
        };
        let (a, b) = (self.descriptor(), other.descriptor());
        if !known(a) || !known(b) || scale(a) != scale(b) {
            return false;
        }
        let linear = |color: &Self| {
            let target = PixelDescriptor::RGBAF32_LINEAR.with_primaries(ColorPrimaries::Bt2020);
            // Out-of-range channels are part of the color, so they must reach the comparison.
            let options = zenpixels::ConvertOptions::permissive()
                .with_alpha_policy(zenpixels::AlphaPolicy::DiscardUnchecked)
                .with_clip_out_of_gamut(false);
            let converted = color.try_convert_to(target, &options).ok()?;
            let bytes: [u8; 16] = converted.as_bytes().try_into().ok()?;
            Some(core::array::from_fn::<f32, 4, _>(|i| {
                f32::from_ne_bytes([
                    bytes[i * 4],
                    bytes[i * 4 + 1],
                    bytes[i * 4 + 2],
                    bytes[i * 4 + 3],
                ])
            }))
        };
        let (Some(a), Some(b)) = (linear(self), linear(other)) else {
            return false;
        };
        const TOLERANCE: f32 = 1e-6;
        if a[3] == 0.0 && b[3] == 0.0 {
            return true;
        }
        a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= TOLERANCE)
    }

    fn map_to_gamut(
        &self,
        target: PixelDescriptor,
    ) -> Result<Self, whereat::At<crate::ConvertError>> {
        use zenpixels::{AlphaMode, ChannelLayout, ChannelType, TransferFunction as Tf};
        let from = self.descriptor();
        let no_path = || whereat::at!(crate::ConvertError::NoPath { from, to: target });
        let hdr = |t: Tf| matches!(t, Tf::Pq | Tf::Hlg);
        if hdr(from.transfer())
            || hdr(target.transfer())
            || !matches!(
                target.layout(),
                ChannelLayout::Rgb | ChannelLayout::Rgba | ChannelLayout::Bgra
            )
        {
            return Err(no_path());
        }
        let space =
            crate::oklab::OklabSpace::for_primaries(target.primaries).ok_or_else(no_path)?;
        let options = zenpixels::ConvertOptions::permissive()
            .with_alpha_policy(zenpixels::AlphaPolicy::DiscardUnchecked);

        // Linear RGBA in the target's primaries, with out-of-gamut values kept.
        let linear = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            Tf::Linear,
        )
        .with_primaries(target.primaries);
        let mut pixel = [0f32; 4];
        crate::RowConverter::new_explicit(from, linear, &options.with_clip_out_of_gamut(false))?
            .try_convert_row(self.as_bytes(), bytemuck::cast_slice_mut(&mut pixel), 1)?;
        let [r, g, b, alpha] = pixel;
        let [r, g, b] = css_gamut_map(&space, [r, g, b]);

        let mapped = [r, g, b, alpha];
        let mut out = [[0f32; 4]; 1];
        let bytes = bytemuck::cast_slice_mut::<[f32; 4], u8>(&mut out);
        if target.bytes_per_pixel() > bytes.len() {
            return Err(no_path());
        }
        let bytes = &mut bytes[..target.bytes_per_pixel()];
        crate::RowConverter::new_explicit(linear, target, &options)?.try_convert_row(
            bytemuck::cast_slice(&mapped),
            bytes,
            1,
        )?;
        Self::try_new(target, bytes).map_err(|e| whereat::at!(crate::ConvertError::Buffer(e)))
    }

    fn try_convert_to(
        &self,
        target: PixelDescriptor,
        options: &zenpixels::ConvertOptions,
    ) -> Result<Self, whereat::At<crate::ConvertError>> {
        let mut converter = crate::RowConverter::new_explicit(self.descriptor(), target, options)?;
        let mut pixel = [[0f32; 4]; 1];
        let pixel = bytemuck::cast_slice_mut::<[f32; 4], u8>(&mut pixel);
        if target.bytes_per_pixel() > pixel.len() {
            return Err(whereat::at!(crate::ConvertError::Buffer(
                zenpixels::buffer::BufferError::InvalidDimensions
            )));
        }
        let pixel = &mut pixel[..target.bytes_per_pixel()];
        converter.try_convert_row(self.as_bytes(), pixel, 1)?;
        Self::try_new(target, pixel).map_err(|e| whereat::at!(crate::ConvertError::Buffer(e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CMYK guard tests ---
    //
    // These used to be `#[should_panic]` against `assert_not_cmyk` (the
    // pre-#44 ABORT behaviour). Per the 0.2.16 NeedsCms migration the
    // trait-level entry points return a typed `ConvertError::NeedsCms`
    // instead — callers wanting CMS dispatch build a `RowConverter` with
    // `new_explicit_with_cms(_, _, _, Some(&MoxCms))` and re-issue.

    #[test]
    fn cmyk_source_returns_needs_cms_from_convert_to() {
        let cmyk_data = vec![0u8; 4 * 4]; // 4 pixels
        let buf = PixelBuffer::from_vec(cmyk_data, 2, 2, PixelDescriptor::CMYK8).unwrap();
        let err = match buf.convert_to(PixelDescriptor::RGB8_SRGB) {
            Ok(_) => panic!("CMYK→RGB on the no-CMS extension entry must error"),
            Err(e) => e,
        };
        assert!(
            matches!(*err.error(), crate::ConvertError::NeedsCms { .. }),
            "expected NeedsCms, got {:?}",
            err.error(),
        );
    }

    #[test]
    fn cmyk_target_returns_needs_cms_from_convert_to() {
        let rgb_data = vec![0u8; 3 * 4]; // 4 pixels
        let buf = PixelBuffer::from_vec(rgb_data, 2, 2, PixelDescriptor::RGB8_SRGB).unwrap();
        let err = match buf.convert_to(PixelDescriptor::CMYK8) {
            Ok(_) => panic!("RGB→CMYK on the no-CMS extension entry must error"),
            Err(e) => e,
        };
        assert!(
            matches!(*err.error(), crate::ConvertError::NeedsCms { .. }),
            "expected NeedsCms, got {:?}",
            err.error(),
        );
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
        let v = from.apply(to.apply(rgb));
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
        let alphas = [128u8, 64];
        let data = vec![100u8, 150, 200, alphas[0], 50, 100, 150, alphas[1]];
        let buf = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGBA8_SRGB).unwrap();
        let lin = buf.linearize().unwrap();
        assert_eq!(
            lin.descriptor().layout(),
            zenpixels::descriptor::ChannelLayout::Rgba
        );
        assert!(lin.descriptor().alpha().is_some());

        // The alpha *values* must survive too, not just the descriptor:
        // alpha is coverage, not light, so it is rescaled (a / 255) and
        // never pushed through the sRGB EOTF (which would turn 128 into
        // ~0.2158 instead of ~0.502).
        let slice = lin.as_slice();
        let row: &[f32] = bytemuck::cast_slice(slice.row(0));
        for (i, &a) in alphas.iter().enumerate() {
            let want = f32::from(a) / 255.0;
            assert_eq!(
                row[i * 4 + 3].to_bits(),
                want.to_bits(),
                "pixel {i}: alpha must be carried linearly, got {} want {want}",
                row[i * 4 + 3]
            );
        }
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
