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
    if crate::convert::requires_cms(from, to) {
        return Err(whereat::at!(crate::ConvertError::NeedsCms {
            from: *from,
            to: *to,
        }));
    }
    Ok(())
}

impl PixelBufferConvertExt for PixelBuffer {
    #[track_caller]
    fn convert_to(&self, target: PixelDescriptor) -> Result<PixelBuffer, At<crate::ConvertError>> {
        let src_desc = self.descriptor();
        check_needs_cms(&src_desc, &target)?;
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
    #[track_caller]
    fn convert_to_sdr(
        &self,
        target: PixelDescriptor,
    ) -> Result<PixelBuffer, At<crate::ConvertError>> {
        use zenpixels::hdr::DiffuseWhite;

        let src_desc = self.descriptor();
        check_needs_cms(&src_desc, &target)?;

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
        //
        // The linearized image is only needed to find the brightest pixel,
        // so it is never materialized: each row is decoded into one reused
        // F32 scratch row and measured immediately (imazen/zenpixels#69 —
        // this used to allocate a full F32 copy, 4× an RGBA8 source, and
        // throw it away). MaxCLL is a max, so the per-row maximum of
        // `measure_max` composes exactly with the whole-image reading
        // (`nits_to_u16` is monotone); the MaxFALL half is unused here.
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
        let diffuse_white = self
            .color_context()
            .and_then(|c| c.diffuse_white)
            .unwrap_or(DiffuseWhite::BT2408);
        let max_cll = measure_peak_rowwise(self, lin_desc, diffuse_white)?;
        let source_peak_nits = f32::from(max_cll).max(100.0);
        self.convert_to_with_hdr_config(target, crate::HdrConfig::for_source_peak(source_peak_nits))
    }

    #[track_caller]
    fn convert_to_with_hdr_config(
        &self,
        target: PixelDescriptor,
        hdr: crate::HdrConfig,
    ) -> Result<PixelBuffer, At<crate::ConvertError>> {
        let src_desc = self.descriptor();
        check_needs_cms(&src_desc, &target)?;
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
}

/// Spec-literal MaxCLL (cd/m², CTA-861.3) of `src` after decoding it to
/// `lin_desc` (relative-linear `RgbF32` / `RgbaF32`), one row at a time
/// through a single reused scratch row — no full-image intermediate.
///
/// Byte-parity with measuring a fully materialized `src.convert_to(lin_desc)`:
/// each row goes through the same `RowConverter`, and MaxCLL is a maximum
/// (`max_y nits_to_u16(max_row_y · white) == nits_to_u16(max_all · white)`
/// because `nits_to_u16` is monotone non-decreasing).
#[cfg(feature = "hdr-experimental")]
#[track_caller]
fn measure_peak_rowwise(
    src: &PixelBuffer,
    lin_desc: PixelDescriptor,
    white: zenpixels::hdr::DiffuseWhite,
) -> Result<u16, At<crate::ConvertError>> {
    use crate::hdr::{CllMeasure, LightLevelMethod};
    use zenpixels::hdr::ContentLightLevel;

    let src_desc = src.descriptor();
    check_needs_cms(&src_desc, &lin_desc)?;
    let mut converter = crate::RowConverter::new(src_desc, lin_desc).at()?;

    let width = src.width();
    // `Vec<f32>` (not `Vec<u8>`) so the scratch row satisfies the F32
    // alignment `PixelSlice::new` validates.
    let floats_per_row = (width as usize)
        .checked_mul(lin_desc.channels())
        .ok_or_else(|| whereat::at!(crate::ConvertError::AllocationFailed))?;
    let mut scratch = alloc::vec![0f32; floats_per_row];
    let row_bytes = floats_per_row * 4;

    let src_slice = src.as_slice();
    let mut max_cll = 0u16;
    for y in 0..src.height() {
        let row: &mut [u8] = bytemuck::cast_slice_mut(&mut scratch);
        converter.convert_row(src_slice.row(y), row, width);
        let lin_row = zenpixels::PixelSlice::new(row, width, 1, row_bytes, lin_desc)
            .map_err_at(crate::ConvertError::from)?;
        let row_cll = ContentLightLevel::measure_max(lin_row, white, LightLevelMethod::MaxRgb)
            .map_or(1000, |c| c.max_content_light_level);
        max_cll = max_cll.max(row_cll);
    }
    Ok(max_cll)
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
