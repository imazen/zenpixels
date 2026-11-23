//! Zero-copy ecosystem interchange (`rgb`, `imgref`, and `image` 0.25).
//!
//! Every ownership move checks exact `bytemuck` allocation/deallocation
//! alignment and capacity rules so zero-copy handoffs are guaranteed sound under
//! `#![forbid(unsafe_code)]`.

#[cfg(feature = "imgref")]
use alloc::vec::Vec;
#[cfg(any(feature = "imgref", feature = "image"))]
use whereat::At;

#[cfg(any(feature = "imgref", feature = "image"))]
use crate::buffer::{BufferError, PixelBuf, PixelView, err_at};
#[cfg(feature = "imgref")]
use crate::color::ImageSpec as ImgrefImageSpec;
#[cfg(feature = "image")]
use crate::color::{AlphaMode, ColorSpace, ImageSpec, Primaries, SignalRange, Transfer};
#[cfg(feature = "image")]
use crate::layout::{ChannelOrder, InteropTarget, SampleLayout, SampleType};
#[cfg(feature = "image")]
use crate::storage::StorageSample;

/// Compile-time pixel type marker for zero-copy `imgref` / typed helpers.
#[cfg(feature = "rgb")]
pub trait PixelType: bytemuck::Pod {
    /// Expected [`SampleLayout`](crate::SampleLayout) for this Rust pixel struct.
    const LAYOUT: crate::SampleLayout;
}

#[cfg(feature = "rgb")]
mod rgb_impls {
    use super::PixelType;
    use crate::buffer::{Bgrx, Rgbx};
    use crate::layout::{ChannelOrder, SampleLayout, SampleType};
    use crate::pixel_types::{F16Sample, GrayAlpha8, GrayAlpha16, GrayAlphaF16, GrayAlphaF32};
    use rgb::{Bgr, Bgra, Gray, Rgb, Rgba};

    impl PixelType for Rgbx {
        const LAYOUT: SampleLayout = SampleLayout::RGBX8;
    }
    impl PixelType for Bgrx {
        const LAYOUT: SampleLayout = SampleLayout::BGRX8;
    }
    impl PixelType for GrayAlpha8 {
        const LAYOUT: SampleLayout = SampleLayout::GRAYA8;
    }
    impl PixelType for GrayAlpha16 {
        const LAYOUT: SampleLayout = SampleLayout::GRAYA16;
    }
    impl PixelType for GrayAlphaF16 {
        const LAYOUT: SampleLayout = SampleLayout::GRAYAF16;
    }
    impl PixelType for GrayAlphaF32 {
        const LAYOUT: SampleLayout = SampleLayout::GRAYAF32;
    }

    impl PixelType for Gray<u8> {
        const LAYOUT: SampleLayout = SampleLayout::GRAY8;
    }
    impl PixelType for Gray<u16> {
        const LAYOUT: SampleLayout = SampleLayout::GRAY16;
    }
    impl PixelType for Gray<F16Sample> {
        const LAYOUT: SampleLayout = SampleLayout::GRAYF16;
    }
    impl PixelType for Gray<f32> {
        const LAYOUT: SampleLayout = SampleLayout::GRAYF32;
    }

    impl PixelType for Rgb<u8> {
        const LAYOUT: SampleLayout = SampleLayout::RGB8;
    }
    impl PixelType for Rgb<u16> {
        const LAYOUT: SampleLayout = SampleLayout::RGB16;
    }
    impl PixelType for Rgb<F16Sample> {
        const LAYOUT: SampleLayout = SampleLayout::RGBF16;
    }
    impl PixelType for Rgb<f32> {
        const LAYOUT: SampleLayout = SampleLayout::RGBF32;
    }

    impl PixelType for Rgba<u8> {
        const LAYOUT: SampleLayout = SampleLayout::RGBA8;
    }
    impl PixelType for Rgba<u16> {
        const LAYOUT: SampleLayout = SampleLayout::RGBA16;
    }
    impl PixelType for Rgba<F16Sample> {
        const LAYOUT: SampleLayout = SampleLayout::RGBAF16;
    }
    impl PixelType for Rgba<f32> {
        const LAYOUT: SampleLayout = SampleLayout::RGBAF32;
    }

    impl PixelType for Bgr<u8> {
        const LAYOUT: SampleLayout = SampleLayout::BGR8;
    }
    impl PixelType for Bgr<u16> {
        const LAYOUT: SampleLayout = SampleLayout::new(SampleType::U16, ChannelOrder::Bgr);
    }
    impl PixelType for Bgr<F16Sample> {
        const LAYOUT: SampleLayout = SampleLayout::new(SampleType::F16, ChannelOrder::Bgr);
    }
    impl PixelType for Bgr<f32> {
        const LAYOUT: SampleLayout = SampleLayout::new(SampleType::F32, ChannelOrder::Bgr);
    }

    impl PixelType for Bgra<u8> {
        const LAYOUT: SampleLayout = SampleLayout::BGRA8;
    }
    impl PixelType for Bgra<u16> {
        const LAYOUT: SampleLayout = SampleLayout::new(SampleType::U16, ChannelOrder::Bgra);
    }
    impl PixelType for Bgra<F16Sample> {
        const LAYOUT: SampleLayout = SampleLayout::new(SampleType::F16, ChannelOrder::Bgra);
    }
    impl PixelType for Bgra<f32> {
        const LAYOUT: SampleLayout = SampleLayout::new(SampleType::F32, ChannelOrder::Bgra);
    }
}

#[cfg(feature = "imgref")]
impl<'a> PixelView<'a> {
    /// Borrow as a zero-copy [`imgref::ImgRef<'a, P>`] when layout, alignment, and pixel-whole stride match.
    pub fn try_as_imgref<P: PixelType>(&self) -> Result<imgref::ImgRef<'a, P>, At<BufferError>> {
        if self.spec().layout != P::LAYOUT {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let bpp = core::mem::size_of::<P>();
        if self.stride() % bpp != 0 {
            return Err(err_at(BufferError::StrideNotPixelAligned));
        }
        let stride_px = self.stride() / bpp;
        let pixels =
            bytemuck::try_cast_slice::<u8, P>(self.as_strided_bytes()).map_err(|e| match e {
                bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                    err_at(BufferError::AlignmentViolation)
                }
                _ => err_at(BufferError::IncompatibleLayout),
            })?;
        Ok(imgref::ImgRef::new_stride(
            pixels,
            self.width() as usize,
            self.rows() as usize,
            stride_px,
        ))
    }
}

#[cfg(feature = "imgref")]
impl PixelBuf {
    /// Adopt an [`imgref::ImgVec<P>`] **zero-copy** when tightly packed and alignment-compatible.
    pub fn from_imgvec<P: PixelType>(
        img: imgref::ImgVec<P>,
        color: crate::ColorSpace,
    ) -> Result<Self, At<BufferError>> {
        let w = u32::try_from(img.width()).map_err(|_| err_at(BufferError::InvalidDimensions))?;
        let h = u32::try_from(img.height()).map_err(|_| err_at(BufferError::InvalidDimensions))?;
        let spec = ImgrefImageSpec::new(P::LAYOUT, color);
        let (buf, _, _) = img.into_contiguous_buf();
        Self::from_vec(buf, w, h, spec)
    }

    /// Move into an [`imgref::ImgVec<P>`], zero-copy when allocation alignment & capacity match.
    pub fn try_into_imgvec<P: PixelType>(self) -> Result<imgref::ImgVec<P>, At<BufferError>> {
        if self.spec().layout != P::LAYOUT {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let w = self.width() as usize;
        let h = self.height() as usize;
        let vec: Vec<P> = self.try_into_vec()?;
        Ok(imgref::ImgVec::new(vec, w, h))
    }
}

#[cfg(feature = "image")]
impl PixelBuf {
    /// Move into an `image::ImageBuffer<P, Vec<P::Subpixel>>` zero-copy when layout and color match.
    pub fn try_into_image_buffer<P>(
        self,
    ) -> Result<image::ImageBuffer<P, alloc::vec::Vec<P::Subpixel>>, At<BufferError>>
    where
        P: image::Pixel + 'static,
        P::Subpixel: crate::storage::StorageSample,
    {
        if !self.spec().supports_interop(InteropTarget::ImageBuffer) {
            return Err(err_at(BufferError::InteropUnsupported));
        }
        if self.spec().layout.sample != P::Subpixel::SAMPLE_TYPE
            || self.spec().layout.channels() != P::CHANNEL_COUNT as usize
        {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let w = self.width();
        let h = self.height();
        let samples = self.try_into_vec::<P::Subpixel>()?;
        image::ImageBuffer::from_raw(w, h, samples)
            .ok_or_else(|| err_at(BufferError::InsufficientData))
    }

    /// Adopt an `image::ImageBuffer<image::Rgb<u8>, Vec<u8>>` zero-copy as sRGB.
    pub fn from_rgb8_image_buffer(
        img: image::RgbImage,
        color: ColorSpace,
    ) -> Result<Self, At<BufferError>> {
        let (w, h) = img.dimensions();
        Self::from_bytes(
            img.into_raw(),
            w,
            h,
            ImageSpec::new(SampleLayout::RGB8, color),
        )
    }

    /// Adopt an `image::ImageBuffer<image::Rgba<u8>, Vec<u8>>` zero-copy.
    pub fn from_rgba8_image_buffer(
        img: image::RgbaImage,
        color: ColorSpace,
    ) -> Result<Self, At<BufferError>> {
        let (w, h) = img.dimensions();
        Self::from_bytes(
            img.into_raw(),
            w,
            h,
            ImageSpec::new(SampleLayout::RGBA8, color),
        )
    }

    /// Move into an `image::DynamicImage` zero-copy when [`InteropTarget::DynamicImage`] is satisfied.
    pub fn try_into_dynamic_image(self) -> Result<image::DynamicImage, At<BufferError>> {
        if !self.spec().supports_interop(InteropTarget::DynamicImage) {
            return Err(err_at(BufferError::InteropUnsupported));
        }
        match (self.spec().layout.sample, self.spec().layout.order) {
            (SampleType::U8, ChannelOrder::Gray) => Ok(image::DynamicImage::ImageLuma8(
                self.try_into_image_buffer()?,
            )),
            (SampleType::U8, ChannelOrder::GrayAlpha) => Ok(image::DynamicImage::ImageLumaA8(
                self.try_into_image_buffer()?,
            )),
            (SampleType::U8, ChannelOrder::Rgb) => Ok(image::DynamicImage::ImageRgb8(
                self.try_into_image_buffer()?,
            )),
            (SampleType::U8, ChannelOrder::Rgba) => Ok(image::DynamicImage::ImageRgba8(
                self.try_into_image_buffer()?,
            )),
            (SampleType::U16, ChannelOrder::Gray) => Ok(image::DynamicImage::ImageLuma16(
                self.try_into_image_buffer()?,
            )),
            (SampleType::U16, ChannelOrder::GrayAlpha) => Ok(image::DynamicImage::ImageLumaA16(
                self.try_into_image_buffer()?,
            )),
            (SampleType::U16, ChannelOrder::Rgb) => Ok(image::DynamicImage::ImageRgb16(
                self.try_into_image_buffer()?,
            )),
            (SampleType::U16, ChannelOrder::Rgba) => Ok(image::DynamicImage::ImageRgba16(
                self.try_into_image_buffer()?,
            )),
            (SampleType::F32, ChannelOrder::Rgb) => Ok(image::DynamicImage::ImageRgb32F(
                self.try_into_image_buffer()?,
            )),
            (SampleType::F32, ChannelOrder::Rgba) => Ok(image::DynamicImage::ImageRgba32F(
                self.try_into_image_buffer()?,
            )),
            _ => Err(err_at(BufferError::InteropUnsupported)),
        }
    }

    /// Helper: construct an sRGB `ColorSpace` matching `image`'s default convention.
    #[must_use]
    pub const fn image_default_color(order: ChannelOrder) -> ColorSpace {
        match order {
            ChannelOrder::Gray => ColorSpace::GRAY_SRGB,
            ChannelOrder::GrayAlpha => ColorSpace::Gray {
                transfer: Transfer::Srgb,
                range: SignalRange::Full,
                alpha: AlphaMode::Straight,
                white: None,
            },
            ChannelOrder::Rgb => ColorSpace::Rgb {
                primaries: Primaries::Bt709,
                transfer: Transfer::Srgb,
                range: SignalRange::Full,
                alpha: AlphaMode::None,
                white: None,
            },
            _ => ColorSpace::SRGB_ALPHA,
        }
    }
}
