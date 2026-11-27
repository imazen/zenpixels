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
use crate::buffer::{BufferError, PixelBuf, PixelView, PixelViewMut, err_at};
#[cfg(feature = "image")]
use crate::cicp::Cicp;
#[cfg(any(feature = "imgref", feature = "image"))]
use crate::color::FormatCow;
#[cfg(feature = "imgref")]
use crate::color::PixelFormat as ImgrefImageSpec;
#[cfg(feature = "image")]
use crate::color::{AlphaMode, ColorSpace, PixelFormat, Primaries, SignalRange, Transfer};
#[cfg(feature = "image")]
use crate::layout::{ChannelOrder, InteropTarget, SampleLayout, SampleType};
#[cfg(feature = "image")]
use crate::orientation::Orientation;
#[cfg(any(feature = "imgref", feature = "image"))]
use crate::storage::StorageSample;

/// Compile-time pixel type marker for zero-copy `imgref` / typed helpers.
#[cfg(feature = "rgb")]
pub trait PixelType: bytemuck::Pod {
    /// Expected [`SampleLayout`] for this Rust pixel struct.
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
    /// Borrow a typed [`imgref::ImgRef<'a, P>`] **zero-copy** with true `'a` lifetime.
    ///
    /// Uses a zero-allocation `&'static PixelFormat` for all standard built-in presets, and
    /// allocates an `Arc<PixelFormat>` only for custom/ICC color spaces.
    pub fn from_imgref<P: PixelType>(
        img: imgref::ImgRef<'a, P>,
        color: crate::ColorSpace,
    ) -> Result<Self, At<BufferError>> {
        let bpp = core::mem::size_of::<P>();
        if bpp == 0 || bpp != P::LAYOUT.bytes_per_pixel() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let w = u32::try_from(img.width()).map_err(|_| err_at(BufferError::InvalidDimensions))?;
        let h = u32::try_from(img.height()).map_err(|_| err_at(BufferError::InvalidDimensions))?;
        let stride = img
            .stride()
            .checked_mul(bpp)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        Self::new_with_format_cow(
            bytemuck::cast_slice(img.buf()),
            w,
            h,
            stride,
            FormatCow::from_owned(ImgrefImageSpec::new(P::LAYOUT, color)),
        )
    }

    /// Borrow as a zero-copy [`imgref::ImgRef<'a, P>`] when layout, alignment, and pixel-whole stride match.
    pub fn as_imgref<P: PixelType>(&self) -> Result<imgref::ImgRef<'a, P>, At<BufferError>> {
        let bpp = core::mem::size_of::<P>();
        if bpp == 0 || bpp != P::LAYOUT.bytes_per_pixel() || self.format().layout != P::LAYOUT {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        if self.stride_bytes() % bpp != 0 {
            return Err(err_at(BufferError::StrideNotPixelAligned));
        }
        let stride_px = self.stride_bytes() / bpp;
        let req =
            crate::buffer::required_bytes(self.height(), self.stride_bytes(), self.row_bytes())
                .map_err(err_at)?;
        let active = self
            .as_strided_bytes()
            .get(..req)
            .ok_or_else(|| err_at(BufferError::InsufficientData))?;
        let pixels = bytemuck::try_cast_slice::<u8, P>(active).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
        })?;
        Ok(imgref::ImgRef::new_stride(
            pixels,
            self.width() as usize,
            self.height() as usize,
            stride_px,
        ))
    }

    /// Borrow as a sample-strided zero-copy [`imgref::ImgRef<'a, S>`] where
    /// `img.width() == self.width() * channels` and `img.stride() == self.stride_bytes() / size_of::<S>()`.
    pub fn as_sample_imgref<S: StorageSample>(
        &self,
    ) -> Result<imgref::ImgRef<'a, S>, At<BufferError>> {
        let layout = self.format().layout;
        if layout.packing.is_packed_word() || layout.sample != S::SAMPLE_TYPE {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let sz = core::mem::size_of::<S>();
        if sz == 0 || self.stride_bytes() % sz != 0 {
            return Err(err_at(BufferError::AlignmentViolation));
        }
        let width_samples = (self.width() as usize)
            .checked_mul(layout.order.channels())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let stride_samples = self.stride_bytes() / sz;
        let req =
            crate::buffer::required_bytes(self.height(), self.stride_bytes(), self.row_bytes())
                .map_err(err_at)?;
        let active = self
            .as_strided_bytes()
            .get(..req)
            .ok_or_else(|| err_at(BufferError::InsufficientData))?;
        let samples = bytemuck::try_cast_slice::<u8, S>(active).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
        })?;
        Ok(imgref::ImgRef::new_stride(
            samples,
            width_samples,
            self.height() as usize,
            stride_samples,
        ))
    }
}

#[cfg(feature = "imgref")]
impl<'a> PixelViewMut<'a> {
    /// Borrow a mutable typed [`imgref::ImgRefMut<'a, P>`] **zero-copy** with true `'a` lifetime.
    ///
    /// Uses a zero-allocation `&'static PixelFormat` for all standard built-in presets, and
    /// allocates an `Arc<PixelFormat>` only for custom/ICC color spaces.
    pub fn from_imgref_mut<P: PixelType>(
        img: imgref::ImgRefMut<'a, P>,
        color: crate::ColorSpace,
    ) -> Result<Self, At<BufferError>> {
        let bpp = core::mem::size_of::<P>();
        if bpp == 0 || bpp != P::LAYOUT.bytes_per_pixel() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let w = u32::try_from(img.width()).map_err(|_| err_at(BufferError::InvalidDimensions))?;
        let h = u32::try_from(img.height()).map_err(|_| err_at(BufferError::InvalidDimensions))?;
        let stride = img
            .stride()
            .checked_mul(bpp)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        Self::new_with_format_cow(
            bytemuck::cast_slice_mut(img.into_buf()),
            w,
            h,
            stride,
            FormatCow::from_owned(ImgrefImageSpec::new(P::LAYOUT, color)),
        )
    }

    /// Borrow as a mutable zero-copy [`imgref::ImgRefMut<'_, P>`].
    pub fn as_imgref_mut<P: PixelType>(
        &mut self,
    ) -> Result<imgref::ImgRefMut<'_, P>, At<BufferError>> {
        let bpp = core::mem::size_of::<P>();
        if bpp == 0 || bpp != P::LAYOUT.bytes_per_pixel() || self.format().layout != P::LAYOUT {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        if self.stride_bytes() % bpp != 0 {
            return Err(err_at(BufferError::StrideNotPixelAligned));
        }
        let stride_px = self.stride_bytes() / bpp;
        let w = self.width() as usize;
        let h = self.height() as usize;
        let req =
            crate::buffer::required_bytes(self.height(), self.stride_bytes(), self.row_bytes())
                .map_err(err_at)?;
        let active = self
            .as_strided_bytes_mut()
            .get_mut(..req)
            .ok_or_else(|| err_at(BufferError::InsufficientData))?;
        let pixels = bytemuck::try_cast_slice_mut::<u8, P>(active).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
        })?;
        Ok(imgref::ImgRefMut::new_stride(pixels, w, h, stride_px))
    }

    /// Consume this mutable view and convert into a zero-copy [`imgref::ImgRefMut<'a, P>`] retaining `'a`.
    pub fn into_imgref_mut<P: PixelType>(
        self,
    ) -> Result<imgref::ImgRefMut<'a, P>, At<BufferError>> {
        let bpp = core::mem::size_of::<P>();
        if bpp == 0 || bpp != P::LAYOUT.bytes_per_pixel() || self.format().layout != P::LAYOUT {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        if self.stride_bytes() % bpp != 0 {
            return Err(err_at(BufferError::StrideNotPixelAligned));
        }
        let stride_px = self.stride_bytes() / bpp;
        let w = self.width() as usize;
        let h = self.height() as usize;
        let req =
            crate::buffer::required_bytes(self.height(), self.stride_bytes(), self.row_bytes())
                .map_err(err_at)?;
        let bytes = self.into_strided_bytes_mut();
        let active = bytes
            .get_mut(..req)
            .ok_or_else(|| err_at(BufferError::InsufficientData))?;
        let pixels = bytemuck::try_cast_slice_mut::<u8, P>(active).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
        })?;
        Ok(imgref::ImgRefMut::new_stride(pixels, w, h, stride_px))
    }

    /// Borrow as a mutable sample-strided zero-copy [`imgref::ImgRefMut<'_, S>`].
    pub fn as_sample_imgref_mut<S: StorageSample>(
        &mut self,
    ) -> Result<imgref::ImgRefMut<'_, S>, At<BufferError>> {
        let layout = self.format().layout;
        if layout.packing.is_packed_word() || layout.sample != S::SAMPLE_TYPE {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let sz = core::mem::size_of::<S>();
        if sz == 0 || self.stride_bytes() % sz != 0 {
            return Err(err_at(BufferError::AlignmentViolation));
        }
        let width_samples = (self.width() as usize)
            .checked_mul(layout.order.channels())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let stride_samples = self.stride_bytes() / sz;
        let h = self.height() as usize;
        let req =
            crate::buffer::required_bytes(self.height(), self.stride_bytes(), self.row_bytes())
                .map_err(err_at)?;
        let active = self
            .as_strided_bytes_mut()
            .get_mut(..req)
            .ok_or_else(|| err_at(BufferError::InsufficientData))?;
        let samples = bytemuck::try_cast_slice_mut::<u8, S>(active).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
        })?;
        Ok(imgref::ImgRefMut::new_stride(
            samples,
            width_samples,
            h,
            stride_samples,
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
        let bpp = core::mem::size_of::<P>();
        if bpp == 0 || bpp != P::LAYOUT.bytes_per_pixel() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let w = u32::try_from(img.width()).map_err(|_| err_at(BufferError::InvalidDimensions))?;
        let h = u32::try_from(img.height()).map_err(|_| err_at(BufferError::InvalidDimensions))?;
        let spec = ImgrefImageSpec::new(P::LAYOUT, color);
        let (buf, _, _) = img.into_contiguous_buf();
        Self::from_pod_bytes_unchecked(buf, w, h, spec)
    }

    /// Move into an [`imgref::ImgVec<P>`] **strictly zero-copy** without copying or allocating.
    ///
    /// Returns `Err(self)` completely untouched if layout, contiguity, alignment, or allocation
    /// capacity prevent a zero-copy `Vec<P>` cast.
    #[allow(clippy::result_large_err)]
    pub fn into_imgvec_zero_copy<P: PixelType>(self) -> Result<imgref::ImgVec<P>, Self> {
        let bpp = core::mem::size_of::<P>();
        if bpp == 0 || bpp != P::LAYOUT.bytes_per_pixel() || self.format().layout != P::LAYOUT {
            return Err(self);
        }
        let w = self.width() as usize;
        let h = self.height() as usize;
        match self.into_vec_zero_copy::<P>() {
            Ok(vec) => Ok(imgref::ImgVec::new(vec, w, h)),
            Err(buf) => Err(buf),
        }
    }

    /// Move into an [`imgref::ImgVec<P>`], zero-copy when allocation alignment & capacity match.
    pub fn into_imgvec<P: PixelType>(self) -> Result<imgref::ImgVec<P>, At<BufferError>> {
        let bpp = core::mem::size_of::<P>();
        if bpp == 0 || bpp != P::LAYOUT.bytes_per_pixel() || self.format().layout != P::LAYOUT {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let w = self.width() as usize;
        let h = self.height() as usize;
        let vec: Vec<P> = self.into_vec()?;
        Ok(imgref::ImgVec::new(vec, w, h))
    }
}

#[cfg(feature = "image")]
impl From<image::metadata::Orientation> for Orientation {
    fn from(o: image::metadata::Orientation) -> Self {
        use image::metadata::Orientation as ImgO;
        match o {
            ImgO::NoTransforms => Self::Identity,
            ImgO::FlipHorizontal => Self::FlipH,
            ImgO::Rotate180 => Self::Rotate180,
            ImgO::FlipVertical => Self::FlipV,
            ImgO::Rotate90FlipH => Self::Transpose,
            ImgO::Rotate90 => Self::Rotate90,
            ImgO::Rotate270FlipH => Self::Transverse,
            ImgO::Rotate270 => Self::Rotate270,
        }
    }
}

#[cfg(feature = "image")]
impl From<Orientation> for image::metadata::Orientation {
    fn from(o: Orientation) -> Self {
        match o {
            Orientation::Identity => Self::NoTransforms,
            Orientation::FlipH => Self::FlipHorizontal,
            Orientation::Rotate180 => Self::Rotate180,
            Orientation::FlipV => Self::FlipVertical,
            Orientation::Transpose => Self::Rotate90FlipH,
            Orientation::Rotate90 => Self::Rotate90,
            Orientation::Transverse => Self::Rotate270FlipH,
            Orientation::Rotate270 => Self::Rotate270,
        }
    }
}

#[cfg(feature = "image")]
impl From<image::metadata::Cicp> for Cicp {
    fn from(c: image::metadata::Cicp) -> Self {
        use image::metadata::{
            CicpColorPrimaries as ImgP, CicpMatrixCoefficients as ImgM,
            CicpTransferCharacteristics as ImgT, CicpVideoFullRangeFlag as ImgR,
        };
        let primaries = match c.primaries {
            ImgP::SRgb => 1,
            ImgP::Unspecified => 2,
            ImgP::RgbM => 4,
            ImgP::RgbB => 5,
            ImgP::Bt601 => 6,
            ImgP::Rgb240m => 7,
            ImgP::GenericFilm => 8,
            ImgP::Rgb2020 => 9,
            ImgP::Xyz => 10,
            ImgP::SmpteRp431 => 11,
            ImgP::SmpteRp432 => 12,
            ImgP::Industry22 => 22,
            _ => 2,
        };
        let transfer = match c.transfer {
            ImgT::Bt709 => 1,
            ImgT::Unspecified => 2,
            ImgT::Bt470M => 4,
            ImgT::Bt470BG => 5,
            ImgT::Bt601 => 6,
            ImgT::Smpte240m => 7,
            ImgT::Linear => 8,
            ImgT::Log100 => 9,
            ImgT::LogSqrt => 10,
            ImgT::Iec61966_2_4 => 11,
            ImgT::Bt1361 => 12,
            ImgT::SRgb => 13,
            ImgT::Bt2020_10bit => 14,
            ImgT::Bt2020_12bit => 15,
            ImgT::Smpte2084 => 16,
            ImgT::Smpte428 => 17,
            ImgT::Bt2100Hlg => 18,
            _ => 2,
        };
        let matrix = match c.matrix {
            ImgM::Identity => 0,
            ImgM::Bt709 => 1,
            ImgM::Unspecified => 2,
            ImgM::UsFCC => 4,
            ImgM::Bt470BG => 5,
            ImgM::Smpte170m => 6,
            ImgM::Smpte240m => 7,
            ImgM::YCgCo => 8,
            ImgM::Bt2020NonConstant => 9,
            ImgM::Bt2020Constant => 10,
            ImgM::Smpte2085 => 11,
            ImgM::ChromaticityDerivedNonConstant => 12,
            ImgM::ChromaticityDerivedConstant => 13,
            ImgM::Bt2100 => 14,
            ImgM::IptPqC2 => 15,
            ImgM::YCgCoRe => 16,
            ImgM::YCgCoRo => 17,
            _ => 2,
        };
        Self::new(
            primaries,
            transfer,
            matrix,
            matches!(c.full_range, ImgR::FullRange),
        )
    }
}

#[cfg(feature = "image")]
impl TryFrom<Cicp> for image::metadata::Cicp {
    type Error = BufferError;

    fn try_from(c: Cicp) -> Result<Self, BufferError> {
        use image::metadata::{
            CicpColorPrimaries as ImgP, CicpMatrixCoefficients as ImgM,
            CicpTransferCharacteristics as ImgT, CicpVideoFullRangeFlag as ImgR,
        };
        let primaries = match c.color_primaries {
            1 => ImgP::SRgb,
            2 => ImgP::Unspecified,
            4 => ImgP::RgbM,
            5 => ImgP::RgbB,
            6 => ImgP::Bt601,
            7 => ImgP::Rgb240m,
            8 => ImgP::GenericFilm,
            9 => ImgP::Rgb2020,
            10 => ImgP::Xyz,
            11 => ImgP::SmpteRp431,
            12 => ImgP::SmpteRp432,
            22 => ImgP::Industry22,
            _ => return Err(BufferError::InteropUnsupported),
        };
        let transfer = match c.transfer_characteristics {
            1 => ImgT::Bt709,
            2 => ImgT::Unspecified,
            4 => ImgT::Bt470M,
            5 => ImgT::Bt470BG,
            6 => ImgT::Bt601,
            7 => ImgT::Smpte240m,
            8 => ImgT::Linear,
            9 => ImgT::Log100,
            10 => ImgT::LogSqrt,
            11 => ImgT::Iec61966_2_4,
            12 => ImgT::Bt1361,
            13 => ImgT::SRgb,
            14 => ImgT::Bt2020_10bit,
            15 => ImgT::Bt2020_12bit,
            16 => ImgT::Smpte2084,
            17 => ImgT::Smpte428,
            18 => ImgT::Bt2100Hlg,
            _ => return Err(BufferError::InteropUnsupported),
        };
        let matrix = match c.matrix_coefficients {
            0 => ImgM::Identity,
            1 => ImgM::Bt709,
            2 => ImgM::Unspecified,
            4 => ImgM::UsFCC,
            5 => ImgM::Bt470BG,
            6 => ImgM::Smpte170m,
            7 => ImgM::Smpte240m,
            8 => ImgM::YCgCo,
            9 => ImgM::Bt2020NonConstant,
            10 => ImgM::Bt2020Constant,
            11 => ImgM::Smpte2085,
            12 => ImgM::ChromaticityDerivedNonConstant,
            13 => ImgM::ChromaticityDerivedConstant,
            14 => ImgM::Bt2100,
            15 => ImgM::IptPqC2,
            16 => ImgM::YCgCoRe,
            17 => ImgM::YCgCoRo,
            _ => return Err(BufferError::InteropUnsupported),
        };
        let full_range = if c.full_range {
            ImgR::FullRange
        } else {
            ImgR::NarrowRange
        };
        Ok(Self {
            primaries,
            transfer,
            matrix,
            full_range,
        })
    }
}

#[cfg(feature = "image")]
fn image_color_type_layout(
    ct: image::ExtendedColorType,
) -> Option<(ChannelOrder, AlphaMode, bool)> {
    match ct {
        image::ExtendedColorType::L8 | image::ExtendedColorType::L16 => {
            Some((ChannelOrder::Gray, AlphaMode::None, true))
        }
        image::ExtendedColorType::La8 | image::ExtendedColorType::La16 => {
            Some((ChannelOrder::GrayAlpha, AlphaMode::Straight, true))
        }
        image::ExtendedColorType::Rgb8
        | image::ExtendedColorType::Rgb16
        | image::ExtendedColorType::Rgb32F => Some((ChannelOrder::Rgb, AlphaMode::None, false)),
        image::ExtendedColorType::Rgba8
        | image::ExtendedColorType::Rgba16
        | image::ExtendedColorType::Rgba32F => {
            Some((ChannelOrder::Rgba, AlphaMode::Straight, false))
        }
        _ => None,
    }
}

#[cfg(feature = "image")]
fn color_space_to_image_cicp(cs: &ColorSpace) -> Option<image::metadata::Cicp> {
    use image::metadata::{
        Cicp as ImgCicp, CicpColorPrimaries as ImgP, CicpMatrixCoefficients as ImgM,
        CicpTransferCharacteristics as ImgT, CicpVideoFullRangeFlag as ImgR,
    };
    let map_primaries = |p: Primaries| match p {
        Primaries::Bt709 => Some(ImgP::SRgb),
        Primaries::DciP3 => Some(ImgP::SmpteRp431),
        Primaries::DisplayP3 => Some(ImgP::SmpteRp432),
        Primaries::Bt2020 => Some(ImgP::Rgb2020),
        _ => None,
    };
    let map_transfer = |t: Transfer| match t {
        Transfer::Srgb => Some(ImgT::SRgb),
        Transfer::Linear => Some(ImgT::Linear),
        Transfer::Bt709 => Some(ImgT::Bt709),
        Transfer::Gamma22 => Some(ImgT::Bt470M),
        Transfer::Pq => Some(ImgT::Smpte2084),
        Transfer::DciCinema => Some(ImgT::Smpte428),
        Transfer::Hlg(crate::HlgMode::Scene) => Some(ImgT::Bt2100Hlg),
        _ => None,
    };
    match cs {
        ColorSpace::Rgb {
            primaries,
            transfer,
            range: SignalRange::Full,
            white: None,
            ..
        } => Some(ImgCicp {
            primaries: map_primaries(*primaries)?,
            transfer: map_transfer(*transfer)?,
            matrix: ImgM::Identity,
            full_range: ImgR::FullRange,
        }),
        ColorSpace::Gray {
            transfer,
            range: SignalRange::Full,
            white: None,
            ..
        } => Some(ImgCicp {
            primaries: ImgP::SRgb,
            transfer: map_transfer(*transfer)?,
            matrix: ImgM::Identity,
            full_range: ImgR::FullRange,
        }),
        _ => None,
    }
}

#[cfg(feature = "image")]
fn canonicalize_opaque_alpha_samples<S: StorageSample>(samples: &mut [S], channels: usize) {
    if channels < 2 {
        return;
    }
    let alpha_idx = channels - 1;
    let bytes = bytemuck::cast_slice_mut::<S, u8>(samples);
    match S::SAMPLE_TYPE {
        SampleType::U8 => {
            for px in bytes.chunks_exact_mut(channels) {
                px[alpha_idx] = 255;
            }
        }
        SampleType::U16 => {
            let max_bytes = u16::MAX.to_ne_bytes();
            for px in bytes.chunks_exact_mut(channels * 2) {
                px[alpha_idx * 2..alpha_idx * 2 + 2].copy_from_slice(&max_bytes);
            }
        }
        SampleType::F32 => {
            let max_bytes = 1.0f32.to_ne_bytes();
            for px in bytes.chunks_exact_mut(channels * 4) {
                px[alpha_idx * 4..alpha_idx * 4 + 4].copy_from_slice(&max_bytes);
            }
        }
        _ => {}
    }
}

#[cfg(feature = "image")]
impl PixelBuf {
    /// Move into an `image::ImageBuffer<P, Vec<P::Subpixel>>` zero-copy when layout and color match,
    /// propagating CICP color-space metadata onto the resulting `ImageBuffer`.
    ///
    /// When `self.format().color.alpha() == AlphaMode::Opaque` and cached facts do not already prove
    /// `all_opaque`, canonicalizes the stored alpha channel to full opacity so `image`'s straight-alpha
    /// `Rgba`/`LumaA` buffer never receives non-opaque alpha samples.
    pub fn into_image_buffer<P>(
        self,
    ) -> Result<image::ImageBuffer<P, alloc::vec::Vec<P::Subpixel>>, At<BufferError>>
    where
        P: image::Pixel + image::PixelWithColorType + 'static,
        P::Subpixel: crate::storage::StorageSample,
    {
        if !self.format().supports_interop(InteropTarget::ImageBuffer) {
            return Err(err_at(BufferError::InteropUnsupported));
        }
        let img_cicp = color_space_to_image_cicp(&self.format().color)
            .ok_or_else(|| err_at(BufferError::InteropUnsupported))?;
        let order_matches = matches!(
            (self.format().layout.order, P::COLOR_TYPE),
            (
                ChannelOrder::Gray,
                image::ExtendedColorType::L8 | image::ExtendedColorType::L16
            ) | (
                ChannelOrder::GrayAlpha,
                image::ExtendedColorType::La8 | image::ExtendedColorType::La16
            ) | (
                ChannelOrder::Rgb,
                image::ExtendedColorType::Rgb8
                    | image::ExtendedColorType::Rgb16
                    | image::ExtendedColorType::Rgb32F
            ) | (
                ChannelOrder::Rgba,
                image::ExtendedColorType::Rgba8
                    | image::ExtendedColorType::Rgba16
                    | image::ExtendedColorType::Rgba32F
            )
        );
        if !order_matches || self.format().layout.sample != P::Subpixel::SAMPLE_TYPE {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let w = self.width();
        let h = self.height();
        let needs_opaque_fix =
            self.format().color.alpha() == AlphaMode::Opaque && !self.facts().all_opaque;
        let channels = self.format().layout.order.channels();
        let mut samples = self.into_vec::<P::Subpixel>()?;
        if needs_opaque_fix {
            canonicalize_opaque_alpha_samples(&mut samples, channels);
        }
        let mut img = image::ImageBuffer::from_raw(w, h, samples)
            .ok_or_else(|| err_at(BufferError::InsufficientData))?;
        img.set_color_space(img_cicp)
            .map_err(|_| err_at(BufferError::InteropUnsupported))?;
        Ok(img)
    }
}

#[cfg(feature = "image")]
fn image_cicp_to_color_space(
    cicp: image::metadata::Cicp,
    alpha: AlphaMode,
    is_gray: bool,
) -> Option<ColorSpace> {
    use image::metadata::{
        CicpColorPrimaries as ImgP, CicpMatrixCoefficients as ImgM,
        CicpTransferCharacteristics as ImgT, CicpVideoFullRangeFlag as ImgR,
    };
    if cicp.matrix != ImgM::Identity || cicp.full_range != ImgR::FullRange {
        return None;
    }
    let transfer = match cicp.transfer {
        ImgT::SRgb => Transfer::Srgb,
        ImgT::Linear => Transfer::Linear,
        ImgT::Bt709 | ImgT::Bt601 | ImgT::Bt2020_10bit | ImgT::Bt2020_12bit => Transfer::Bt709,
        ImgT::Bt470M => Transfer::Gamma22,
        ImgT::Smpte2084 => Transfer::Pq,
        ImgT::Smpte428 => Transfer::DciCinema,
        ImgT::Bt2100Hlg => Transfer::Hlg(crate::HlgMode::Scene),
        _ => return None,
    };
    if is_gray {
        return Some(ColorSpace::Gray {
            transfer,
            range: SignalRange::Full,
            alpha,
            white: None,
        });
    }
    let primaries = match cicp.primaries {
        ImgP::SRgb => Primaries::Bt709,
        ImgP::SmpteRp431 => Primaries::DciP3,
        ImgP::SmpteRp432 => Primaries::DisplayP3,
        ImgP::Rgb2020 => Primaries::Bt2020,
        _ => return None,
    };
    Some(ColorSpace::Rgb {
        primaries,
        transfer,
        range: SignalRange::Full,
        alpha,
        white: None,
    })
}

#[cfg(feature = "image")]
impl<'a> PixelView<'a> {
    /// Borrow any `image::ImageBuffer<P, C>` **zero-copy** for all 10 standard formats
    /// (`Luma`, `LumaA`, `Rgb`, `Rgba` × `u8`/`u16`/`f32`), reading its declared CICP color space.
    ///
    /// Uses a zero-allocation `&'static PixelFormat` for all standard sRGB / linear formats, and
    /// allocates an `Arc<PixelFormat>` only for non-preset CICP color spaces.
    pub fn from_image_buffer<P, C>(
        img: &'a image::ImageBuffer<P, C>,
    ) -> Result<Self, At<BufferError>>
    where
        P: image::Pixel + image::PixelWithColorType + 'static,
        P::Subpixel: StorageSample,
        C: core::ops::Deref<Target = [P::Subpixel]>,
    {
        let (order, alpha, is_gray) = image_color_type_layout(P::COLOR_TYPE)
            .ok_or_else(|| err_at(BufferError::IncompatibleLayout))?;
        let color = image_cicp_to_color_space(img.color_space(), alpha, is_gray)
            .ok_or_else(|| err_at(BufferError::InteropUnsupported))?;
        let (w, h) = img.dimensions();
        let layout = SampleLayout::new(P::Subpixel::SAMPLE_TYPE, order);
        let stride = (w as usize)
            .checked_mul(layout.bytes_per_pixel())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        Self::new_with_format_cow(
            bytemuck::cast_slice(img.as_raw().deref()),
            w,
            h,
            stride,
            FormatCow::from_owned(PixelFormat::new(layout, color)),
        )
    }

    /// Borrow an `image::DynamicImage` **zero-copy** for all 10 standard variants.
    pub fn from_dynamic_image(img: &'a image::DynamicImage) -> Result<Self, At<BufferError>> {
        match img {
            image::DynamicImage::ImageLuma8(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageLumaA8(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageRgb8(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageRgba8(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageLuma16(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageLumaA16(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageRgb16(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageRgba16(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageRgb32F(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageRgba32F(b) => Self::from_image_buffer(b),
            _ => Err(err_at(BufferError::InteropUnsupported)),
        }
    }
}

#[cfg(feature = "image")]
impl<'a> PixelViewMut<'a> {
    /// Mutably borrow any `image::ImageBuffer<P, C>` **zero-copy** for all 10 standard formats.
    ///
    /// Uses a zero-allocation `&'static PixelFormat` for all standard sRGB / linear formats, and
    /// allocates an `Arc<PixelFormat>` only for non-preset CICP color spaces.
    pub fn from_image_buffer_mut<P, C>(
        img: &'a mut image::ImageBuffer<P, C>,
    ) -> Result<Self, At<BufferError>>
    where
        P: image::Pixel + image::PixelWithColorType + 'static,
        P::Subpixel: StorageSample,
        C: core::ops::Deref<Target = [P::Subpixel]> + core::ops::DerefMut,
    {
        let (order, alpha, is_gray) = image_color_type_layout(P::COLOR_TYPE)
            .ok_or_else(|| err_at(BufferError::IncompatibleLayout))?;
        let color = image_cicp_to_color_space(img.color_space(), alpha, is_gray)
            .ok_or_else(|| err_at(BufferError::InteropUnsupported))?;
        let (w, h) = img.dimensions();
        let layout = SampleLayout::new(P::Subpixel::SAMPLE_TYPE, order);
        let stride = (w as usize)
            .checked_mul(layout.bytes_per_pixel())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let samples: &'a mut [P::Subpixel] = img;
        Self::new_with_format_cow(
            bytemuck::cast_slice_mut(samples),
            w,
            h,
            stride,
            FormatCow::from_owned(PixelFormat::new(layout, color)),
        )
    }

    /// Mutably borrow an `image::DynamicImage` **zero-copy** for all 10 standard variants.
    pub fn from_dynamic_image_mut(
        img: &'a mut image::DynamicImage,
    ) -> Result<Self, At<BufferError>> {
        match img {
            image::DynamicImage::ImageLuma8(b) => Self::from_image_buffer_mut(b),
            image::DynamicImage::ImageLumaA8(b) => Self::from_image_buffer_mut(b),
            image::DynamicImage::ImageRgb8(b) => Self::from_image_buffer_mut(b),
            image::DynamicImage::ImageRgba8(b) => Self::from_image_buffer_mut(b),
            image::DynamicImage::ImageLuma16(b) => Self::from_image_buffer_mut(b),
            image::DynamicImage::ImageLumaA16(b) => Self::from_image_buffer_mut(b),
            image::DynamicImage::ImageRgb16(b) => Self::from_image_buffer_mut(b),
            image::DynamicImage::ImageRgba16(b) => Self::from_image_buffer_mut(b),
            image::DynamicImage::ImageRgb32F(b) => Self::from_image_buffer_mut(b),
            image::DynamicImage::ImageRgba32F(b) => Self::from_image_buffer_mut(b),
            _ => Err(err_at(BufferError::InteropUnsupported)),
        }
    }
}

#[cfg(feature = "image")]
impl PixelBuf {
    /// Adopt any `image::ImageBuffer<P, Vec<P::Subpixel>>` **zero-copy** across all 10 standard
    /// formats (`Luma`, `LumaA`, `Rgb`, `Rgba` × `u8`/`u16`/`f32`), preserving its declared
    /// `image::metadata::Cicp` color space.
    pub fn from_image_buffer<P>(
        img: image::ImageBuffer<P, alloc::vec::Vec<P::Subpixel>>,
    ) -> Result<Self, At<BufferError>>
    where
        P: image::Pixel + image::PixelWithColorType + 'static,
        P::Subpixel: StorageSample,
    {
        let (order, alpha, is_gray) = image_color_type_layout(P::COLOR_TYPE)
            .ok_or_else(|| err_at(BufferError::IncompatibleLayout))?;
        let color = image_cicp_to_color_space(img.color_space(), alpha, is_gray)
            .ok_or_else(|| err_at(BufferError::InteropUnsupported))?;
        let (w, h) = img.dimensions();
        let layout = SampleLayout::new(P::Subpixel::SAMPLE_TYPE, order);
        Self::from_samples(img.into_raw(), w, h, PixelFormat::new(layout, color))
    }

    /// Adopt any `image::ImageBuffer<P, Vec<P::Subpixel>>` **zero-copy** with an explicit
    /// `color`, rejecting contradictory non-default CICP declarations.
    pub fn from_image_buffer_with_color<P>(
        img: image::ImageBuffer<P, alloc::vec::Vec<P::Subpixel>>,
        color: ColorSpace,
    ) -> Result<Self, At<BufferError>>
    where
        P: image::Pixel + image::PixelWithColorType + 'static,
        P::Subpixel: StorageSample,
    {
        let (order, _, _) = image_color_type_layout(P::COLOR_TYPE)
            .ok_or_else(|| err_at(BufferError::IncompatibleLayout))?;
        let declared = img.color_space();
        if declared != image::metadata::Cicp::SRGB
            && color_space_to_image_cicp(&color) != Some(declared)
        {
            return Err(err_at(BufferError::InteropUnsupported));
        }
        let (w, h) = img.dimensions();
        let layout = SampleLayout::new(P::Subpixel::SAMPLE_TYPE, order);
        Self::from_samples(img.into_raw(), w, h, PixelFormat::new(layout, color))
    }

    /// Adopt an `image::DynamicImage` **zero-copy** across all 10 standard variants, reading
    /// its declared `image::metadata::Cicp` color space.
    ///
    /// Note: `image` 0.25 defaults `DynamicImage::color_space()` to `Cicp::SRGB` even for
    /// `ImageRgb32F`/`ImageRgba32F`. Use [`from_dynamic_image_with_color`](Self::from_dynamic_image_with_color)
    /// when adopting untagged linear float `DynamicImage` buffers whose CICP was left default.
    pub fn from_dynamic_image(img: image::DynamicImage) -> Result<Self, At<BufferError>> {
        match img {
            image::DynamicImage::ImageLuma8(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageLumaA8(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageRgb8(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageRgba8(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageLuma16(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageLumaA16(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageRgb16(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageRgba16(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageRgb32F(b) => Self::from_image_buffer(b),
            image::DynamicImage::ImageRgba32F(b) => Self::from_image_buffer(b),
            _ => Err(err_at(BufferError::InteropUnsupported)),
        }
    }

    /// Adopt an `image::DynamicImage` **zero-copy** across all 10 standard variants with an
    /// explicit `color` (e.g. `ColorSpace::LINEAR_SRGB` for `ImageRgb32F`), rejecting
    /// contradictory non-default CICP declarations.
    pub fn from_dynamic_image_with_color(
        img: image::DynamicImage,
        color: ColorSpace,
    ) -> Result<Self, At<BufferError>> {
        match img {
            image::DynamicImage::ImageLuma8(b) => Self::from_image_buffer_with_color(b, color),
            image::DynamicImage::ImageLumaA8(b) => Self::from_image_buffer_with_color(b, color),
            image::DynamicImage::ImageRgb8(b) => Self::from_image_buffer_with_color(b, color),
            image::DynamicImage::ImageRgba8(b) => Self::from_image_buffer_with_color(b, color),
            image::DynamicImage::ImageLuma16(b) => Self::from_image_buffer_with_color(b, color),
            image::DynamicImage::ImageLumaA16(b) => Self::from_image_buffer_with_color(b, color),
            image::DynamicImage::ImageRgb16(b) => Self::from_image_buffer_with_color(b, color),
            image::DynamicImage::ImageRgba16(b) => Self::from_image_buffer_with_color(b, color),
            image::DynamicImage::ImageRgb32F(b) => Self::from_image_buffer_with_color(b, color),
            image::DynamicImage::ImageRgba32F(b) => Self::from_image_buffer_with_color(b, color),
            _ => Err(err_at(BufferError::InteropUnsupported)),
        }
    }

    /// Adopt an `image::RgbImage` zero-copy, reading its declared `image::metadata::Cicp` color space.
    pub fn from_rgb8_image(img: image::RgbImage) -> Result<Self, At<BufferError>> {
        Self::from_image_buffer(img)
    }

    /// Adopt an `image::RgbaImage` zero-copy, reading its declared `image::metadata::Cicp` color space.
    pub fn from_rgba8_image(img: image::RgbaImage) -> Result<Self, At<BufferError>> {
        Self::from_image_buffer(img)
    }

    /// Adopt an `image::ImageBuffer<image::Rgb<u8>, Vec<u8>>` zero-copy with explicit `color`.
    ///
    /// Rejects contradictory color spaces when `img` declares a non-default CICP space
    /// that disagrees with `color` (use [`from_rgb8_image`](Self::from_rgb8_image) to
    /// preserve `img`'s own declared CICP automatically).
    pub fn from_rgb8_image_with_color(
        img: image::RgbImage,
        color: ColorSpace,
    ) -> Result<Self, At<BufferError>> {
        Self::from_image_buffer_with_color(img, color)
    }

    /// Adopt an `image::ImageBuffer<image::Rgba<u8>, Vec<u8>>` zero-copy with explicit `color`.
    ///
    /// Rejects contradictory color spaces when `img` declares a non-default CICP space
    /// that disagrees with `color` (use [`from_rgba8_image`](Self::from_rgba8_image) to
    /// preserve `img`'s own declared CICP automatically).
    pub fn from_rgba8_image_with_color(
        img: image::RgbaImage,
        color: ColorSpace,
    ) -> Result<Self, At<BufferError>> {
        Self::from_image_buffer_with_color(img, color)
    }

    /// Move into an `image::DynamicImage` zero-copy when [`InteropTarget::DynamicImage`] is satisfied.
    pub fn into_dynamic_image(self) -> Result<image::DynamicImage, At<BufferError>> {
        if !self.format().supports_interop(InteropTarget::DynamicImage) {
            return Err(err_at(BufferError::InteropUnsupported));
        }
        match (self.format().layout.sample, self.format().layout.order) {
            (SampleType::U8, ChannelOrder::Gray) => {
                Ok(image::DynamicImage::ImageLuma8(self.into_image_buffer()?))
            }
            (SampleType::U8, ChannelOrder::GrayAlpha) => {
                Ok(image::DynamicImage::ImageLumaA8(self.into_image_buffer()?))
            }
            (SampleType::U8, ChannelOrder::Rgb) => {
                Ok(image::DynamicImage::ImageRgb8(self.into_image_buffer()?))
            }
            (SampleType::U8, ChannelOrder::Rgba) => {
                Ok(image::DynamicImage::ImageRgba8(self.into_image_buffer()?))
            }
            (SampleType::U16, ChannelOrder::Gray) => {
                Ok(image::DynamicImage::ImageLuma16(self.into_image_buffer()?))
            }
            (SampleType::U16, ChannelOrder::GrayAlpha) => {
                Ok(image::DynamicImage::ImageLumaA16(self.into_image_buffer()?))
            }
            (SampleType::U16, ChannelOrder::Rgb) => {
                Ok(image::DynamicImage::ImageRgb16(self.into_image_buffer()?))
            }
            (SampleType::U16, ChannelOrder::Rgba) => {
                Ok(image::DynamicImage::ImageRgba16(self.into_image_buffer()?))
            }
            (SampleType::F32, ChannelOrder::Rgb) => {
                Ok(image::DynamicImage::ImageRgb32F(self.into_image_buffer()?))
            }
            (SampleType::F32, ChannelOrder::Rgba) => {
                Ok(image::DynamicImage::ImageRgba32F(self.into_image_buffer()?))
            }
            _ => Err(err_at(BufferError::InteropUnsupported)),
        }
    }

    /// Helper: construct an sRGB [`ColorSpace`] matching `image` 0.25's default `Cicp::SRGB`
    /// convention for standard grayscale and RGB/BGR channel orders.
    ///
    /// Returns `Some(...)` for `Gray`, `GrayAlpha`, `Rgb`, `Bgr`, `Rgba`, `Bgra`, `Rgbx`, and
    /// `Bgrx` (mapping `Rgb`/`Bgr` to `AlphaMode::None`, `Rgbx`/`Bgrx` to `AlphaMode::Undefined`,
    /// and `Rgba`/`Bgra`/`GrayAlpha` to `AlphaMode::Straight`), and `None` for non-sRGB orders
    /// (`Cmyk`, `Generic`).
    ///
    /// Note: [`from_image_buffer`](Self::from_image_buffer) and [`from_dynamic_image`](Self::from_dynamic_image)
    /// read each buffer's declared `image::metadata::Cicp` rather than calling this helper.
    #[must_use]
    #[doc(alias = "image_srgb_default_color")]
    pub const fn try_image_srgb_default_color(order: ChannelOrder) -> Option<ColorSpace> {
        match order {
            ChannelOrder::Gray => Some(ColorSpace::GRAY_SRGB),
            ChannelOrder::GrayAlpha => Some(ColorSpace::Gray {
                transfer: Transfer::Srgb,
                range: SignalRange::Full,
                alpha: AlphaMode::Straight,
                white: None,
            }),
            ChannelOrder::Rgb | ChannelOrder::Bgr => Some(ColorSpace::Rgb {
                primaries: Primaries::Bt709,
                transfer: Transfer::Srgb,
                range: SignalRange::Full,
                alpha: AlphaMode::None,
                white: None,
            }),
            ChannelOrder::Rgba | ChannelOrder::Bgra => Some(ColorSpace::SRGB_ALPHA),
            ChannelOrder::Rgbx | ChannelOrder::Bgrx => Some(ColorSpace::Rgb {
                primaries: Primaries::Bt709,
                transfer: Transfer::Srgb,
                range: SignalRange::Full,
                alpha: AlphaMode::Undefined,
                white: None,
            }),
            ChannelOrder::Cmyk | ChannelOrder::Generic(_) => None,
            _ => None,
        }
    }
}

#[cfg(feature = "image")]
impl TryFrom<image::DynamicImage> for PixelBuf {
    type Error = At<BufferError>;

    #[inline]
    fn try_from(img: image::DynamicImage) -> Result<Self, At<BufferError>> {
        Self::from_dynamic_image(img)
    }
}
