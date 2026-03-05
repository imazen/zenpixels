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

    #[cfg(feature = "buffer")]
    use super::PixelBufferConvertExt;

    #[cfg(feature = "buffer")]
    use zenpixels::PixelDescriptor;

    #[cfg(feature = "buffer")]
    use zenpixels::buffer::PixelBuffer;

    #[test]
    #[cfg(feature = "buffer")]
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
    #[cfg(feature = "buffer")]
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
    #[cfg(feature = "buffer")]
    fn try_add_alpha_rgb() {
        let data = vec![100u8, 150, 200, 50, 100, 150];
        let buf = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        let out = buf.try_add_alpha().unwrap();
        // Should now be RGBA with straight alpha
        assert_eq!(out.descriptor().layout(), zenpixels::descriptor::ChannelLayout::Rgba);
        let slice = out.as_slice();
        let row = slice.row(0);
        assert_eq!(row[3], 255);
        assert_eq!(row[7], 255);
    }

    #[test]
    #[cfg(feature = "buffer")]
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
    #[cfg(feature = "buffer")]
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
        let buf =
            PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGB16_SRGB).unwrap();
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
    #[cfg(feature = "buffer")]
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
    #[cfg(feature = "buffer")]
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
    #[cfg(feature = "buffer")]
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
    #[cfg(feature = "buffer")]
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
