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
            Self::Srgb | Self::Bt709 => linear_srgb::precise::srgb_to_linear(v),
            Self::Pq => pq_eotf(v),
            Self::Hlg => hlg_eotf(v),
            _ => v,
        }
    }

    #[allow(unreachable_patterns)]
    fn delinearize(&self, v: f32) -> f32 {
        match self {
            Self::Linear | Self::Unknown => v,
            Self::Srgb | Self::Bt709 => linear_srgb::precise::linear_to_srgb(v),
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

#[cfg(feature = "buffer")]
use alloc::sync::Arc;
#[cfg(feature = "buffer")]
use alloc::vec;
#[cfg(feature = "buffer")]
use zenpixels::PixelDescriptor;
#[cfg(feature = "buffer")]
use zenpixels::buffer::{Pixel, PixelBuffer};
#[cfg(feature = "buffer")]
use zenpixels::descriptor::{AlphaMode, ChannelLayout, ChannelType};

/// Adds format conversion methods to type-erased [`PixelBuffer`].
#[cfg(feature = "buffer")]
pub trait PixelBufferConvertExt {
    /// Convert pixel data to a different layout and depth.
    ///
    /// Uses [`RowConverter`](crate::RowConverter) for transfer-function-aware
    /// conversion. Color metadata is preserved.
    ///
    /// **Allocates** a new [`PixelBuffer`].
    fn convert_to(&self, target: PixelDescriptor) -> Result<PixelBuffer, crate::ConvertError>;

    /// Add an alpha channel. **Allocates** a new `PixelBuffer`.
    ///
    /// - Gray → GrayAlpha (opaque alpha)
    /// - Rgb → Rgba (opaque alpha)
    /// - Already has alpha → identity copy
    fn try_add_alpha(&self) -> Result<PixelBuffer, crate::ConvertError>;

    /// Widen to U16 depth (lossless, ×257). **Allocates** a new `PixelBuffer`.
    fn try_widen_to_u16(&self) -> Result<PixelBuffer, crate::ConvertError>;

    /// Narrow to U8 depth (lossy, rounded). **Allocates** a new `PixelBuffer`.
    fn try_narrow_to_u8(&self) -> Result<PixelBuffer, crate::ConvertError>;

    /// Convert to RGB8, allocating a new buffer.
    fn to_rgb8(&self) -> PixelBuffer<rgb::Rgb<u8>>;

    /// Convert to RGBA8, allocating a new buffer.
    fn to_rgba8(&self) -> PixelBuffer<rgb::Rgba<u8>>;

    /// Convert to Gray8, allocating a new buffer.
    fn to_gray8(&self) -> PixelBuffer<rgb::Gray<u8>>;

    /// Convert to BGRA8, allocating a new buffer.
    fn to_bgra8(&self) -> PixelBuffer<rgb::alt::BGRA<u8>>;
}

#[cfg(feature = "buffer")]
impl PixelBufferConvertExt for PixelBuffer {
    fn convert_to(&self, target: PixelDescriptor) -> Result<PixelBuffer, crate::ConvertError> {
        let src_desc = self.descriptor();
        if src_desc == target {
            // Identity — just copy.
            let dst_stride = target.aligned_stride(self.width());
            let total = dst_stride
                .checked_mul(self.height() as usize)
                .ok_or(crate::ConvertError::AllocationFailed)?;
            let mut out = alloc::vec![0u8; total];
            let src_slice = self.as_slice();
            for y in 0..self.height() {
                let src_row = src_slice.row(y);
                let dst_start = y as usize * dst_stride;
                out[dst_start..dst_start + src_row.len()].copy_from_slice(src_row);
            }
            let mut buf = PixelBuffer::from_vec(out, self.width(), self.height(), target)
                .map_err(|_| crate::ConvertError::AllocationFailed)?;
            if let Some(ctx) = self.color_context() {
                buf = buf.with_color_context(Arc::clone(ctx));
            }
            return Ok(buf);
        }

        let converter = crate::RowConverter::new(src_desc, target)?;

        let dst_stride = target.aligned_stride(self.width());
        let total = dst_stride
            .checked_mul(self.height() as usize)
            .ok_or(crate::ConvertError::AllocationFailed)?;
        let mut out = alloc::vec![0u8; total];

        let src_slice = self.as_slice();
        for y in 0..self.height() {
            let src_row = src_slice.row(y);
            let dst_start = y as usize * dst_stride;
            let dst_end = dst_start + dst_stride;
            converter.convert_row(src_row, &mut out[dst_start..dst_end], self.width());
        }

        let mut buf = PixelBuffer::from_vec(out, self.width(), self.height(), target)
            .map_err(|_| crate::ConvertError::AllocationFailed)?;
        if let Some(ctx) = self.color_context() {
            buf = buf.with_color_context(Arc::clone(ctx));
        }
        Ok(buf)
    }

    fn try_add_alpha(&self) -> Result<PixelBuffer, crate::ConvertError> {
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

    fn try_widen_to_u16(&self) -> Result<PixelBuffer, crate::ConvertError> {
        let desc = self.descriptor();
        let target = PixelDescriptor::new(
            ChannelType::U16,
            desc.layout(),
            desc.alpha(),
            desc.transfer(),
        );
        self.convert_to(target)
    }

    fn try_narrow_to_u8(&self) -> Result<PixelBuffer, crate::ConvertError> {
        let desc = self.descriptor();
        let target = PixelDescriptor::new(
            ChannelType::U8,
            desc.layout(),
            desc.alpha(),
            desc.transfer(),
        );
        self.convert_to(target)
    }

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
#[cfg(feature = "buffer")]
fn convert_to_typed<Q: Pixel>(buf: &PixelBuffer, target: PixelDescriptor) -> PixelBuffer<Q> {
    let conv = crate::RowConverter::new(buf.descriptor(), target)
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
        for &v in &[0.0, 0.1, 0.5, 0.75, 1.0] {
            let lin = tf.linearize(v);
            let back = tf.delinearize(lin);
            assert!(
                (v - back).abs() < 1e-4,
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
}
