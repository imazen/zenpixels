//! Conversions to and from the `image` crate.
//!
//! `TryFrom` impls only. Nothing here copies samples. Color travels with packed
//! buffers in both directions, and a conversion that would have to relabel the
//! pixels refuses instead: `image` has no "unknown" color space and defaults to
//! sRGB.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};

#[cfg(feature = "planar")]
use image::Primitive;
use image::flat::{FlatSamples, SampleLayout};
use image::metadata::{
    Cicp as ImageCicp, CicpColorPrimaries, CicpMatrixCoefficients, CicpTransferCharacteristics,
    CicpVideoFullRangeFlag, Orientation as ImageOrientation,
};
use image::{ColorType, DynamicImage, ExtendedColorType, ImageBuffer, Luma, LumaA, Rgb, Rgba};
use whereat::{At, at};

use crate::buffer::{
    BufferError, ByteStorage, FromPartsError, PixelBuffer, PixelBufferParts, PixelSlice,
    PixelSliceMut, PixelStorage,
};
use crate::cicp::Cicp;
use crate::color::ColorContext;
use crate::descriptor::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, PixelFormat,
    SignalRange, TransferFunction,
};
use crate::orientation::Orientation;
use crate::pixel_types::F16Sample;
#[cfg(feature = "planar")]
use crate::plane::{Plane, PlaneError, PlaneMut, PlaneRef, PlaneSample};
use crate::storage::SampleStorage;

#[cfg(feature = "planar")]
fn u32_dim(n: usize) -> Result<u32, PlaneError> {
    u32::try_from(n).map_err(|_| PlaneError::GeometryOverflow)
}

// ---- Planes: `Luma` images and strided views ------------------------------

/// Move a gray image's container into a plane.
#[cfg(feature = "planar")]
impl<T, C> TryFrom<ImageBuffer<Luma<T>, C>> for Plane<C>
where
    T: PlaneSample + Primitive,
    C: SampleStorage<Sample = T> + Deref<Target = [T]>,
{
    type Error = PlaneError;
    fn try_from(image: ImageBuffer<Luma<T>, C>) -> Result<Self, PlaneError> {
        let (width, height) = image.dimensions();
        Plane::new(
            image.into_raw(),
            width as usize,
            height as usize,
            width as usize,
        )
    }
}

/// Borrow a gray image as a plane.
#[cfg(feature = "planar")]
impl<'a, T, C> TryFrom<&'a ImageBuffer<Luma<T>, C>> for PlaneRef<'a, T>
where
    T: PlaneSample + Primitive,
    C: Deref<Target = [T]>,
{
    type Error = PlaneError;
    fn try_from(image: &'a ImageBuffer<Luma<T>, C>) -> Result<Self, PlaneError> {
        let (width, height) = image.dimensions();
        Plane::new(&**image, width as usize, height as usize, width as usize)
    }
}

/// Borrow a gray image as a mutable plane.
#[cfg(feature = "planar")]
impl<'a, T, C> TryFrom<&'a mut ImageBuffer<Luma<T>, C>> for PlaneMut<'a, T>
where
    T: PlaneSample + Primitive,
    C: DerefMut<Target = [T]>,
{
    type Error = PlaneError;
    fn try_from(image: &'a mut ImageBuffer<Luma<T>, C>) -> Result<Self, PlaneError> {
        let (width, height) = image.dimensions();
        Plane::new(
            &mut **image,
            width as usize,
            height as usize,
            width as usize,
        )
    }
}

/// Move a plane's storage into a gray image.
///
/// `ImageBuffer` has no stride and would read a strided buffer as wrong pixels,
/// so the error is the plane, unchanged, unless row 0 is at index 0 and the rows
/// are tight. Use `FlatSamples` for a strided plane. The image is labeled sRGB,
/// `image`'s default: a plane carries no color, so set it on the result.
#[cfg(feature = "planar")]
impl<T, C> TryFrom<Plane<C>> for ImageBuffer<Luma<T>, C>
where
    T: PlaneSample + Primitive,
    C: SampleStorage<Sample = T> + Deref<Target = [T]>,
{
    type Error = Plane<C>;
    fn try_from(plane: Plane<C>) -> Result<Self, Plane<C>> {
        let (width, height, stride) = (plane.width(), plane.height(), plane.stride());
        let (Ok(w), Ok(h)) = (u32_dim(width), u32_dim(height)) else {
            return Err(plane);
        };
        if stride != width || plane.samples_start() != 0 {
            return Err(plane);
        }
        let buf = plane.into_storage();
        Ok(ImageBuffer::from_raw(w, h, buf).expect("the plane's extent covers width * height"))
    }
}

#[cfg(feature = "planar")]
fn gray_layout(width: usize, height: usize, stride: usize) -> Result<SampleLayout, PlaneError> {
    Ok(SampleLayout {
        channels: 1,
        channel_stride: 1,
        width: u32_dim(width)?,
        width_stride: 1,
        height: u32_dim(height)?,
        height_stride: stride,
    })
}

/// View a plane, strided or not, through `image`'s strided type.
#[cfg(feature = "planar")]
impl<'a, T: PlaneSample> TryFrom<PlaneRef<'a, T>> for FlatSamples<&'a [T]> {
    type Error = PlaneError;
    fn try_from(plane: PlaneRef<'a, T>) -> Result<Self, PlaneError> {
        let layout = gray_layout(plane.width(), plane.height(), plane.stride())?;
        Ok(FlatSamples {
            samples: plane.into_strided_samples(),
            layout,
            color_hint: None,
        })
    }
}

/// View a mutable plane, strided or not, through `image`'s strided type.
#[cfg(feature = "planar")]
impl<'a, T: PlaneSample> TryFrom<PlaneMut<'a, T>> for FlatSamples<&'a mut [T]> {
    type Error = PlaneError;
    fn try_from(plane: PlaneMut<'a, T>) -> Result<Self, PlaneError> {
        let layout = gray_layout(plane.width(), plane.height(), plane.stride())?;
        Ok(FlatSamples {
            samples: plane.into_strided_samples_mut(),
            layout,
            color_hint: None,
        })
    }
}

// ---- Metadata: Orientation & CICP -----------------------------------------

/// Exact 1:1 EXIF orientation bijection (`1..=8`) from `image` to `zenpixels`.
impl From<ImageOrientation> for Orientation {
    fn from(o: ImageOrientation) -> Self {
        match o {
            ImageOrientation::NoTransforms => Self::Identity,
            ImageOrientation::FlipHorizontal => Self::FlipH,
            ImageOrientation::Rotate180 => Self::Rotate180,
            ImageOrientation::FlipVertical => Self::FlipV,
            ImageOrientation::Rotate90FlipH => Self::Transpose,
            ImageOrientation::Rotate90 => Self::Rotate90,
            ImageOrientation::Rotate270FlipH => Self::Transverse,
            ImageOrientation::Rotate270 => Self::Rotate270,
        }
    }
}

/// Exact 1:1 EXIF orientation bijection (`1..=8`) from `zenpixels` to `image`.
impl From<Orientation> for ImageOrientation {
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

/// Convert an `image::metadata::Cicp` tag to `zenpixels::Cicp`.
impl From<ImageCicp> for Cicp {
    fn from(c: ImageCicp) -> Self {
        Self::new(
            primaries_code(c.primaries),
            transfer_code(c.transfer),
            matrix_code(c.matrix),
            matches!(c.full_range, CicpVideoFullRangeFlag::FullRange),
        )
    }
}

/// Convert a `zenpixels::Cicp` tag to `image::metadata::Cicp`, returning
/// [`BufferError::IncompatibleDescriptor`] if a reserved ITU-T H.273 code point
/// has no variant in `image::metadata`.
impl TryFrom<Cicp> for ImageCicp {
    type Error = BufferError;
    fn try_from(c: Cicp) -> Result<Self, BufferError> {
        let primaries = match c.color_primaries {
            1 => CicpColorPrimaries::SRgb,
            2 => CicpColorPrimaries::Unspecified,
            4 => CicpColorPrimaries::RgbM,
            5 => CicpColorPrimaries::RgbB,
            6 => CicpColorPrimaries::Bt601,
            7 => CicpColorPrimaries::Rgb240m,
            8 => CicpColorPrimaries::GenericFilm,
            9 => CicpColorPrimaries::Rgb2020,
            10 => CicpColorPrimaries::Xyz,
            11 => CicpColorPrimaries::SmpteRp431,
            12 => CicpColorPrimaries::SmpteRp432,
            22 => CicpColorPrimaries::Industry22,
            _ => return Err(BufferError::IncompatibleDescriptor),
        };
        let transfer = match c.transfer_characteristics {
            1 => CicpTransferCharacteristics::Bt709,
            2 => CicpTransferCharacteristics::Unspecified,
            4 => CicpTransferCharacteristics::Bt470M,
            5 => CicpTransferCharacteristics::Bt470BG,
            6 => CicpTransferCharacteristics::Bt601,
            7 => CicpTransferCharacteristics::Smpte240m,
            8 => CicpTransferCharacteristics::Linear,
            9 => CicpTransferCharacteristics::Log100,
            10 => CicpTransferCharacteristics::LogSqrt,
            11 => CicpTransferCharacteristics::Iec61966_2_4,
            12 => CicpTransferCharacteristics::Bt1361,
            13 => CicpTransferCharacteristics::SRgb,
            14 => CicpTransferCharacteristics::Bt2020_10bit,
            15 => CicpTransferCharacteristics::Bt2020_12bit,
            16 => CicpTransferCharacteristics::Smpte2084,
            17 => CicpTransferCharacteristics::Smpte428,
            18 => CicpTransferCharacteristics::Bt2100Hlg,
            _ => return Err(BufferError::IncompatibleDescriptor),
        };
        let matrix = match c.matrix_coefficients {
            0 => CicpMatrixCoefficients::Identity,
            1 => CicpMatrixCoefficients::Bt709,
            2 => CicpMatrixCoefficients::Unspecified,
            4 => CicpMatrixCoefficients::UsFCC,
            5 => CicpMatrixCoefficients::Bt470BG,
            6 => CicpMatrixCoefficients::Smpte170m,
            7 => CicpMatrixCoefficients::Smpte240m,
            8 => CicpMatrixCoefficients::YCgCo,
            9 => CicpMatrixCoefficients::Bt2020NonConstant,
            10 => CicpMatrixCoefficients::Bt2020Constant,
            11 => CicpMatrixCoefficients::Smpte2085,
            12 => CicpMatrixCoefficients::ChromaticityDerivedNonConstant,
            13 => CicpMatrixCoefficients::ChromaticityDerivedConstant,
            14 => CicpMatrixCoefficients::Bt2100,
            15 => CicpMatrixCoefficients::IptPqC2,
            16 => CicpMatrixCoefficients::YCgCoRe,
            17 => CicpMatrixCoefficients::YCgCoRo,
            _ => return Err(BufferError::IncompatibleDescriptor),
        };
        let full_range = if c.full_range {
            CicpVideoFullRangeFlag::FullRange
        } else {
            CicpVideoFullRangeFlag::NarrowRange
        };
        Ok(Self {
            primaries,
            transfer,
            matrix,
            full_range,
        })
    }
}

// ---- Color ----------------------------------------------------------------

fn primaries_code(p: CicpColorPrimaries) -> u8 {
    match p {
        CicpColorPrimaries::SRgb => 1,
        CicpColorPrimaries::Unspecified => 2,
        CicpColorPrimaries::RgbM => 4,
        CicpColorPrimaries::RgbB => 5,
        CicpColorPrimaries::Bt601 => 6,
        CicpColorPrimaries::Rgb240m => 7,
        CicpColorPrimaries::GenericFilm => 8,
        CicpColorPrimaries::Rgb2020 => 9,
        CicpColorPrimaries::Xyz => 10,
        CicpColorPrimaries::SmpteRp431 => 11,
        CicpColorPrimaries::SmpteRp432 => 12,
        CicpColorPrimaries::Industry22 => 22,
        _ => 2,
    }
}

fn transfer_code(t: CicpTransferCharacteristics) -> u8 {
    match t {
        CicpTransferCharacteristics::Bt709 => 1,
        CicpTransferCharacteristics::Unspecified => 2,
        CicpTransferCharacteristics::Bt470M => 4,
        CicpTransferCharacteristics::Bt470BG => 5,
        CicpTransferCharacteristics::Bt601 => 6,
        CicpTransferCharacteristics::Smpte240m => 7,
        CicpTransferCharacteristics::Linear => 8,
        CicpTransferCharacteristics::Log100 => 9,
        CicpTransferCharacteristics::LogSqrt => 10,
        CicpTransferCharacteristics::Iec61966_2_4 => 11,
        CicpTransferCharacteristics::Bt1361 => 12,
        CicpTransferCharacteristics::SRgb => 13,
        CicpTransferCharacteristics::Bt2020_10bit => 14,
        CicpTransferCharacteristics::Bt2020_12bit => 15,
        CicpTransferCharacteristics::Smpte2084 => 16,
        CicpTransferCharacteristics::Smpte428 => 17,
        CicpTransferCharacteristics::Bt2100Hlg => 18,
        _ => 2,
    }
}

fn matrix_code(m: CicpMatrixCoefficients) -> u8 {
    match m {
        CicpMatrixCoefficients::Identity => 0,
        CicpMatrixCoefficients::Bt709 => 1,
        CicpMatrixCoefficients::Unspecified => 2,
        CicpMatrixCoefficients::UsFCC => 4,
        CicpMatrixCoefficients::Bt470BG => 5,
        CicpMatrixCoefficients::Smpte170m => 6,
        CicpMatrixCoefficients::Smpte240m => 7,
        CicpMatrixCoefficients::YCgCo => 8,
        CicpMatrixCoefficients::Bt2020NonConstant => 9,
        CicpMatrixCoefficients::Bt2020Constant => 10,
        CicpMatrixCoefficients::Smpte2085 => 11,
        CicpMatrixCoefficients::ChromaticityDerivedNonConstant => 12,
        CicpMatrixCoefficients::ChromaticityDerivedConstant => 13,
        CicpMatrixCoefficients::Bt2100 => 14,
        CicpMatrixCoefficients::IptPqC2 => 15,
        CicpMatrixCoefficients::YCgCoRe => 16,
        CicpMatrixCoefficients::YCgCoRo => 17,
        _ => 2,
    }
}

fn image_primaries(p: ColorPrimaries) -> Option<CicpColorPrimaries> {
    match p {
        ColorPrimaries::Bt709 => Some(CicpColorPrimaries::SRgb),
        ColorPrimaries::Bt2020 => Some(CicpColorPrimaries::Rgb2020),
        ColorPrimaries::DisplayP3 => Some(CicpColorPrimaries::SmpteRp432),
        _ => None,
    }
}

fn image_transfer(t: TransferFunction) -> Option<CicpTransferCharacteristics> {
    match t {
        TransferFunction::Bt709 => Some(CicpTransferCharacteristics::Bt709),
        TransferFunction::Linear => Some(CicpTransferCharacteristics::Linear),
        TransferFunction::Srgb => Some(CicpTransferCharacteristics::SRgb),
        TransferFunction::Pq => Some(CicpTransferCharacteristics::Smpte2084),
        TransferFunction::Hlg => Some(CicpTransferCharacteristics::Bt2100Hlg),
        _ => None,
    }
}

/// The descriptor for an image's color, and the CICP to attach when the
/// descriptor alone would lose part of it. Common spaces allocate nothing.
fn describe(
    channel: ChannelType,
    layout: ChannelLayout,
    alpha: Option<AlphaMode>,
    color: ImageCicp,
) -> (PixelDescriptor, Option<Cicp>) {
    let prim = primaries_code(color.primaries);
    let tc = transfer_code(color.transfer);
    let full = matches!(color.full_range, CicpVideoFullRangeFlag::FullRange);
    let matrix = matrix_code(color.matrix);
    let primaries = ColorPrimaries::from_cicp(prim).unwrap_or(ColorPrimaries::Unknown);
    let transfer = TransferFunction::from_cicp(tc).unwrap_or(TransferFunction::Unknown);
    let descriptor = PixelDescriptor::new_full(channel, layout, alpha, transfer, primaries)
        .with_signal_range(if full {
            SignalRange::Full
        } else {
            SignalRange::Narrow
        });
    let lossless =
        matrix == 0 && primaries.to_cicp() == Some(prim) && transfer.to_cicp() == Some(tc);
    let cicp = (!lossless).then(|| Cicp::new(prim, tc, matrix, full));
    (descriptor, cicp)
}

/// The `image` labels for a buffer, or `None` when `image` cannot state its color
/// exactly: unknown color, narrow range, an ICC profile, or CICP that disagrees.
fn image_color(
    descriptor: PixelDescriptor,
    context: Option<&Arc<ColorContext>>,
) -> Option<(CicpColorPrimaries, CicpTransferCharacteristics)> {
    if descriptor.signal_range != SignalRange::Full {
        return None;
    }
    let primaries = image_primaries(descriptor.primaries)?;
    let transfer = image_transfer(descriptor.transfer)?;
    if let Some(context) = context {
        if context
            .rendering_intent()
            .is_some_and(|i| i != crate::RenderingIntent::RelativeColorimetric)
        {
            return None;
        }
        match context.profile() {
            Some(crate::ColorProfile::Icc(_) | crate::ColorProfile::Parametric(_)) => return None,
            Some(crate::ColorProfile::Cicp(cicp)) => {
                let matches = Some(cicp.color_primaries) == descriptor.primaries.to_cicp()
                    && Some(cicp.transfer_characteristics) == descriptor.transfer.to_cicp()
                    && cicp.matrix_coefficients == 0
                    && cicp.full_range;
                if !matches {
                    return None;
                }
            }
            None => {}
        }
    }
    Some((primaries, transfer))
}

// ---- Packed buffers -------------------------------------------------------

/// The `image` labels when this buffer can move into an `ImageBuffer` of this format:
/// same format, straight alpha, tight rows, first pixel at index 0, color `image` can
/// state exactly.
#[allow(clippy::too_many_arguments)]
fn movable_erased(
    descriptor: PixelDescriptor,
    width: u32,
    stride: usize,
    offset_bytes: usize,
    color_context: Option<&Arc<ColorContext>>,
    channel: ChannelType,
    layout: ChannelLayout,
    alpha: Option<AlphaMode>,
) -> Option<(CicpColorPrimaries, CicpTransferCharacteristics)> {
    let alpha_ok = match alpha {
        None => descriptor.alpha().is_none(),
        Some(_) => matches!(
            descriptor.alpha(),
            Some(AlphaMode::Straight | AlphaMode::Opaque)
        ),
    };
    let tight = stride == width as usize * descriptor.bytes_per_pixel();
    if descriptor.channel_type() != channel
        || descriptor.layout() != layout
        || !alpha_ok
        || !tight
        || offset_bytes != 0
    {
        return None;
    }
    image_color(descriptor, color_context)
}

#[inline]
fn movable<S: ByteStorage>(
    buffer: &PixelBuffer<(), S>,
    channel: ChannelType,
    layout: ChannelLayout,
    alpha: Option<AlphaMode>,
) -> Option<(CicpColorPrimaries, CicpTransferCharacteristics)> {
    movable_erased(
        buffer.descriptor(),
        buffer.width(),
        buffer.stride(),
        buffer.offset_bytes(),
        buffer.color_context(),
        channel,
        layout,
        alpha,
    )
}

fn adopt_image_storage<S: ByteStorage>(
    data: S,
    width: u32,
    height: u32,
    channel: ChannelType,
    layout: ChannelLayout,
    alpha: Option<AlphaMode>,
    color: ImageCicp,
) -> Result<PixelBuffer<(), S>, FromPartsError<S>> {
    let (descriptor, cicp) = describe(channel, layout, alpha, color);
    let stride = (width as usize).saturating_mul(descriptor.bytes_per_pixel());
    let parts = PixelBufferParts::new(data, width, height, stride, descriptor);
    let buffer = PixelBuffer::try_from_parts(parts)?;
    Ok(match cicp {
        Some(cicp) => buffer.with_cicp(cicp),
        None => buffer,
    })
}

fn slice_from_image_bytes(
    bytes: &[u8],
    width: u32,
    height: u32,
    channel: ChannelType,
    layout: ChannelLayout,
    alpha: Option<AlphaMode>,
    color: ImageCicp,
) -> Result<PixelSlice<'_>, At<BufferError>> {
    let (descriptor, cicp) = describe(channel, layout, alpha, color);
    let stride = (width as usize)
        .checked_mul(descriptor.bytes_per_pixel())
        .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
    let slice = PixelSlice::new(bytes, width, height, stride, descriptor)?;
    Ok(match cicp {
        Some(cicp) => slice.with_cicp(cicp),
        None => slice,
    })
}

fn slice_mut_from_image_bytes(
    bytes: &mut [u8],
    width: u32,
    height: u32,
    channel: ChannelType,
    layout: ChannelLayout,
    alpha: Option<AlphaMode>,
    color: ImageCicp,
) -> Result<PixelSliceMut<'_>, At<BufferError>> {
    let (descriptor, cicp) = describe(channel, layout, alpha, color);
    let stride = (width as usize)
        .checked_mul(descriptor.bytes_per_pixel())
        .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
    let slice = PixelSliceMut::new(bytes, width, height, stride, descriptor)?;
    Ok(match cicp {
        Some(cicp) => slice.with_cicp(cicp),
        None => slice,
    })
}

macro_rules! packed_conversions {
    ($($pixel:ident<$sample:ty> => $channel:ident, $layout:ident, $alpha:expr;)*) => {$(
        /// Move an image's container into a buffer, with its color.
        impl<C> TryFrom<ImageBuffer<$pixel<$sample>, C>> for PixelBuffer<(), C>
        where
            C: SampleStorage<Sample = $sample> + Deref<Target = [$sample]>,
        {
            type Error = FromPartsError<C>;
            #[inline]
            fn try_from(image: ImageBuffer<$pixel<$sample>, C>) -> Result<Self, FromPartsError<C>> {
                let color = image.color_space();
                let (width, height) = image.dimensions();
                adopt_image_storage(
                    image.into_raw(),
                    width,
                    height,
                    ChannelType::$channel,
                    ChannelLayout::$layout,
                    $alpha,
                    color,
                )
            }
        }

        /// Move an image's `Vec` into a default [`PixelBuffer`], with its color.
        impl TryFrom<ImageBuffer<$pixel<$sample>, Vec<$sample>>> for PixelBuffer {
            type Error = FromPartsError;
            #[inline]
            fn try_from(image: ImageBuffer<$pixel<$sample>, Vec<$sample>>) -> Result<Self, FromPartsError> {
                let color = image.color_space();
                let (width, height) = image.dimensions();
                adopt_image_storage(
                    PixelStorage::from(image.into_raw()),
                    width,
                    height,
                    ChannelType::$channel,
                    ChannelLayout::$layout,
                    $alpha,
                    color,
                )
            }
        }

        /// Move a default [`PixelBuffer`] into an image, with its color. Refuses as
        /// the storage-generic form does, and also when the buffer's storage isn't
        /// this sample type.
        impl TryFrom<PixelBuffer> for ImageBuffer<$pixel<$sample>, Vec<$sample>> {
            type Error = PixelBuffer;
            fn try_from(buffer: PixelBuffer) -> Result<Self, PixelBuffer> {
                let Some((primaries, transfer)) =
                    movable(&buffer, ChannelType::$channel, ChannelLayout::$layout, $alpha)
                else {
                    return Err(buffer);
                };
                if !buffer.storage_holds::<$sample>() {
                    return Err(buffer);
                }
                let (width, height) = (buffer.width(), buffer.height());
                let samples = buffer
                    .into_parts()
                    .data
                    .try_into_vec::<$sample>()
                    .ok()
                    .expect("checked above");
                let mut image: Self = ImageBuffer::from_raw(width, height, samples)
                    .expect("validated storage covers width * height pixels");
                image.set_rgb_primaries(primaries);
                image.set_transfer_function(transfer);
                Ok(image)
            }
        }

        /// Borrow an image as a slice, with its color.
        impl<'a, C> TryFrom<&'a ImageBuffer<$pixel<$sample>, C>> for PixelSlice<'a>
        where
            C: Deref<Target = [$sample]>,
        {
            type Error = At<BufferError>;
            #[inline]
            fn try_from(image: &'a ImageBuffer<$pixel<$sample>, C>) -> Result<Self, At<BufferError>> {
                let (width, height) = image.dimensions();
                slice_from_image_bytes(
                    bytemuck::cast_slice(&**image),
                    width,
                    height,
                    ChannelType::$channel,
                    ChannelLayout::$layout,
                    $alpha,
                    image.color_space(),
                )
            }
        }

        /// Borrow an image as a mutable slice, with its color.
        impl<'a, C> TryFrom<&'a mut ImageBuffer<$pixel<$sample>, C>> for PixelSliceMut<'a>
        where
            C: DerefMut<Target = [$sample]>,
        {
            type Error = At<BufferError>;
            #[inline]
            fn try_from(image: &'a mut ImageBuffer<$pixel<$sample>, C>) -> Result<Self, At<BufferError>> {
                let color = image.color_space();
                let (width, height) = image.dimensions();
                slice_mut_from_image_bytes(
                    bytemuck::cast_slice_mut(&mut **image),
                    width,
                    height,
                    ChannelType::$channel,
                    ChannelLayout::$layout,
                    $alpha,
                    color,
                )
            }
        }

        /// Move a buffer's storage into an image, with its color.
        ///
        /// The error is the buffer, unchanged, when the format differs, the rows
        /// are not tight, the first pixel is past index 0, or `image` cannot state
        /// the color exactly. Relabel with `with_transfer` and `with_primaries`
        /// when the color is known and only the label is missing. Premultiplied
        /// and undefined alpha are refused because `image` means straight alpha
        /// (`AlphaMode::Opaque` is accepted because all-`1.0` alpha samples are
        /// numerically identical under straight alpha, and reads back as `Straight`).
        impl<C> TryFrom<PixelBuffer<(), C>> for ImageBuffer<$pixel<$sample>, C>
        where
            C: SampleStorage<Sample = $sample> + Deref<Target = [$sample]>,
        {
            type Error = PixelBuffer<(), C>;
            fn try_from(buffer: PixelBuffer<(), C>) -> Result<Self, PixelBuffer<(), C>> {
                let Some((primaries, transfer)) =
                    movable(&buffer, ChannelType::$channel, ChannelLayout::$layout, $alpha)
                else {
                    return Err(buffer);
                };
                let (width, height) = (buffer.width(), buffer.height());
                let parts = buffer.into_parts();
                let mut image: Self = ImageBuffer::from_raw(width, height, parts.data)
                    .expect("validated storage covers width * height pixels");
                image.set_rgb_primaries(primaries);
                image.set_transfer_function(transfer);
                Ok(image)
            }
        }
    )*};
}

packed_conversions! {
    Luma<u8> => U8, Gray, None;
    Luma<u16> => U16, Gray, None;
    Luma<f32> => F32, Gray, None;
    LumaA<u8> => U8, GrayAlpha, Some(AlphaMode::Straight);
    LumaA<u16> => U16, GrayAlpha, Some(AlphaMode::Straight);
    LumaA<f32> => F32, GrayAlpha, Some(AlphaMode::Straight);
    Rgb<u8> => U8, Rgb, None;
    Rgb<u16> => U16, Rgb, None;
    Rgb<f32> => F32, Rgb, None;
    Rgba<u8> => U8, Rgba, Some(AlphaMode::Straight);
    Rgba<u16> => U16, Rgba, Some(AlphaMode::Straight);
    Rgba<f32> => F32, Rgba, Some(AlphaMode::Straight);
}

fn flat_channels(descriptor: PixelDescriptor, expected: ChannelType) -> Result<u8, BufferError> {
    if descriptor.channel_type() != expected {
        return Err(BufferError::IncompatibleDescriptor);
    }
    match descriptor.layout() {
        ChannelLayout::Gray => Ok(1),
        ChannelLayout::GrayAlpha => Ok(2),
        ChannelLayout::Rgb => Ok(3),
        ChannelLayout::Rgba => Ok(4),
        _ => Err(BufferError::IncompatibleDescriptor),
    }
}

fn flat_descriptor_and_stride(
    layout: &SampleLayout,
    color_hint: Option<ColorType>,
    channel: ChannelType,
    sample_size: usize,
) -> Result<(PixelDescriptor, usize), BufferError> {
    if layout.channel_stride != 1 {
        return Err(BufferError::IncompatibleDescriptor);
    }
    let ch_layout = match layout.channels {
        1 => ChannelLayout::Gray,
        2 => ChannelLayout::GrayAlpha,
        3 => ChannelLayout::Rgb,
        4 => ChannelLayout::Rgba,
        _ => return Err(BufferError::IncompatibleDescriptor),
    };
    if layout.width_stride != usize::from(layout.channels) {
        return Err(BufferError::IncompatibleDescriptor);
    }
    if let Some(hint) = color_hint {
        let Ok(hint_fmt) = PixelFormat::try_from(hint) else {
            return Err(BufferError::IncompatibleDescriptor);
        };
        if hint_fmt.channel_type() != channel || hint_fmt.layout() != ch_layout {
            return Err(BufferError::IncompatibleDescriptor);
        }
    }
    let stride_bytes = layout
        .height_stride
        .checked_mul(sample_size)
        .ok_or(BufferError::InvalidDimensions)?;
    // FlatSamples carries no primaries or transfer function; declare both Unknown
    // so downstream converters never silently treat untagged FFI samples as sRGB/BT.709.
    let descriptor = PixelDescriptor::new_full(
        channel,
        ch_layout,
        ch_layout.has_alpha().then_some(AlphaMode::Straight),
        TransferFunction::Unknown,
        ColorPrimaries::Unknown,
    );
    Ok((descriptor, stride_bytes))
}

fn flat_layout_from_geom(
    descriptor: PixelDescriptor,
    width: u32,
    height: u32,
    stride: usize,
    channel: ChannelType,
    sample_size: usize,
) -> Result<(SampleLayout, Option<ColorType>), At<BufferError>> {
    let channels = flat_channels(descriptor, channel).map_err(|e| at!(e))?;
    if stride % sample_size != 0 {
        return Err(at!(BufferError::AlignmentViolation));
    }
    Ok((
        SampleLayout {
            channels,
            channel_stride: 1,
            width,
            width_stride: usize::from(channels),
            height,
            height_stride: stride / sample_size,
        },
        ColorType::try_from(descriptor.format).ok(),
    ))
}

fn slice_from_flat_bytes<'a>(
    bytes: &'a [u8],
    layout: &SampleLayout,
    color_hint: Option<ColorType>,
    channel: ChannelType,
    sample_size: usize,
) -> Result<PixelSlice<'a>, At<BufferError>> {
    let (descriptor, stride_bytes) =
        flat_descriptor_and_stride(layout, color_hint, channel, sample_size).map_err(|e| at!(e))?;
    PixelSlice::new(bytes, layout.width, layout.height, stride_bytes, descriptor)
}

fn slice_mut_from_flat_bytes<'a>(
    bytes: &'a mut [u8],
    layout: &SampleLayout,
    color_hint: Option<ColorType>,
    channel: ChannelType,
    sample_size: usize,
) -> Result<PixelSliceMut<'a>, At<BufferError>> {
    let (descriptor, stride_bytes) =
        flat_descriptor_and_stride(layout, color_hint, channel, sample_size).map_err(|e| at!(e))?;
    PixelSliceMut::new(bytes, layout.width, layout.height, stride_bytes, descriptor)
}

macro_rules! flat_views {
    ($($sample:ty => $channel:ident;)*) => {$(
        /// View a slice, strided or not, through `image`'s strided [`FlatSamples`].
        ///
        /// `FlatSamples` carries no color profile, so read color metadata from the
        /// slice first. Only gray, gray-alpha, RGB and RGBA layouts are accepted.
        impl<'a> TryFrom<PixelSlice<'a>> for FlatSamples<&'a [$sample]> {
            type Error = At<BufferError>;
            fn try_from(slice: PixelSlice<'a>) -> Result<Self, At<BufferError>> {
                let (layout, color_hint) = flat_layout_from_geom(
                    slice.descriptor(),
                    slice.width(),
                    slice.rows(),
                    slice.stride(),
                    ChannelType::$channel,
                    core::mem::size_of::<$sample>(),
                )?;
                let samples: &'a [$sample] = if slice.width() == 0 || slice.rows() == 0 {
                    &[]
                } else {
                    bytemuck::try_cast_slice(slice.as_strided_bytes())
                        .map_err(|_| at!(BufferError::AlignmentViolation))?
                };
                Ok(FlatSamples {
                    samples,
                    layout,
                    color_hint,
                })
            }
        }

        /// View a mutable slice, strided or not, through `image`'s strided [`FlatSamples`].
        impl<'a> TryFrom<PixelSliceMut<'a>> for FlatSamples<&'a mut [$sample]> {
            type Error = At<BufferError>;
            fn try_from(slice: PixelSliceMut<'a>) -> Result<Self, At<BufferError>> {
                let (layout, color_hint) = flat_layout_from_geom(
                    slice.descriptor(),
                    slice.width(),
                    slice.rows(),
                    slice.stride(),
                    ChannelType::$channel,
                    core::mem::size_of::<$sample>(),
                )?;
                let raw = slice.into_strided_bytes_mut();
                let samples: &'a mut [$sample] = if raw.is_empty() {
                    &mut []
                } else {
                    bytemuck::try_cast_slice_mut(raw)
                        .map_err(|_| at!(BufferError::AlignmentViolation))?
                };
                Ok(FlatSamples {
                    samples,
                    layout,
                    color_hint,
                })
            }
        }

        /// Move a [`PixelBuffer`]'s sample `Vec` (strided or packed, including 64-byte
        /// SIMD-padded 3-channel buffers) into `FlatSamples<Vec<S>>` without copying.
        ///
        /// Returns `Err(buffer)` unchanged when the format isn't `Gray`/`GrayAlpha`/`Rgb`/`Rgba`,
        /// `offset != 0`, `stride` isn't sample-aligned, or storage holds another sample type.
        impl TryFrom<PixelBuffer> for FlatSamples<Vec<$sample>> {
            type Error = PixelBuffer;
            fn try_from(buffer: PixelBuffer) -> Result<Self, PixelBuffer> {
                if buffer.offset_bytes() != 0 || !buffer.storage_holds::<$sample>() {
                    return Err(buffer);
                }
                let Ok((layout, color_hint)) = flat_layout_from_geom(
                    buffer.descriptor(),
                    buffer.width(),
                    buffer.height(),
                    buffer.stride(),
                    ChannelType::$channel,
                    core::mem::size_of::<$sample>(),
                ) else {
                    return Err(buffer);
                };
                let samples = buffer
                    .into_parts()
                    .data
                    .try_into_vec::<$sample>()
                    .ok()
                    .expect("checked above");
                Ok(FlatSamples {
                    samples,
                    layout,
                    color_hint,
                })
            }
        }

        /// Borrow an interleaved row-major [`FlatSamples`] slice as a [`PixelSlice`].
        ///
        /// Because `FlatSamples` carries no CICP or ICC color signaling, the returned
        /// slice has `ColorPrimaries::Unknown` and `TransferFunction::Unknown`.
        impl<'a> TryFrom<FlatSamples<&'a [$sample]>> for PixelSlice<'a> {
            type Error = At<BufferError>;
            #[inline]
            fn try_from(flat: FlatSamples<&'a [$sample]>) -> Result<Self, At<BufferError>> {
                slice_from_flat_bytes(
                    bytemuck::cast_slice(flat.samples),
                    &flat.layout,
                    flat.color_hint,
                    ChannelType::$channel,
                    core::mem::size_of::<$sample>(),
                )
            }
        }

        /// Borrow a mutable interleaved row-major [`FlatSamples`] slice as a [`PixelSliceMut`].
        ///
        /// Stamps `ColorPrimaries::Unknown` and `TransferFunction::Unknown`; relabel with
        /// `with_primaries` / `with_transfer` when the color space is known.
        impl<'a> TryFrom<FlatSamples<&'a mut [$sample]>> for PixelSliceMut<'a> {
            type Error = At<BufferError>;
            #[inline]
            fn try_from(flat: FlatSamples<&'a mut [$sample]>) -> Result<Self, At<BufferError>> {
                slice_mut_from_flat_bytes(
                    bytemuck::cast_slice_mut(flat.samples),
                    &flat.layout,
                    flat.color_hint,
                    ChannelType::$channel,
                    core::mem::size_of::<$sample>(),
                )
            }
        }

        /// Move an owned interleaved row-major `FlatSamples<Vec<S>>` into a [`PixelBuffer`]
        /// without copying, returning `Err(flat)` unchanged if the layout is non-interleaved,
        /// unsupported, or undersized.
        impl TryFrom<FlatSamples<Vec<$sample>>> for PixelBuffer {
            type Error = FlatSamples<Vec<$sample>>;
            fn try_from(flat: FlatSamples<Vec<$sample>>) -> Result<Self, FlatSamples<Vec<$sample>>> {
                let Ok((descriptor, stride_bytes)) = flat_descriptor_and_stride(
                    &flat.layout,
                    flat.color_hint,
                    ChannelType::$channel,
                    core::mem::size_of::<$sample>(),
                ) else {
                    return Err(flat);
                };
                let FlatSamples {
                    samples,
                    layout,
                    color_hint,
                } = flat;
                let parts = PixelBufferParts::new(
                    PixelStorage::from(samples),
                    layout.width,
                    layout.height,
                    stride_bytes,
                    descriptor,
                );
                match PixelBuffer::try_from_parts(parts) {
                    Ok(buffer) => Ok(buffer),
                    Err(mut err) => {
                        let parts = err.take_parts().expect("parts retained");
                        let samples = parts
                            .data
                            .try_into_vec::<$sample>()
                            .ok()
                            .expect("same sample type");
                        Err(FlatSamples {
                            samples,
                            layout,
                            color_hint,
                        })
                    }
                }
            }
        }
    )*};
}

flat_views! {
    u8 => U8;
    u16 => U16;
    F16Sample => F16;
    f32 => F32;
}

// ---- DynamicImage ---------------------------------------------------------

macro_rules! dynamic_image {
    ($($variant:ident: $pixel:ident<$sample:ty> => $channel:ident, $layout:ident;)*) => {
        /// Move a [`DynamicImage`]'s samples into a buffer, with its color.
        impl TryFrom<DynamicImage> for PixelBuffer {
            type Error = At<BufferError>;
            fn try_from(image: DynamicImage) -> Result<Self, At<BufferError>> {
                match image {
                    $(DynamicImage::$variant(image) => {
                        PixelBuffer::try_from(image).map_err(FromPartsError::without_buffer)
                    })*
                    _ => Err(at!(BufferError::IncompatibleDescriptor)),
                }
            }
        }

        /// Borrow a [`DynamicImage`] as a slice, with its color.
        impl<'a> TryFrom<&'a DynamicImage> for PixelSlice<'a> {
            type Error = At<BufferError>;
            fn try_from(image: &'a DynamicImage) -> Result<Self, At<BufferError>> {
                match image {
                    $(DynamicImage::$variant(image) => PixelSlice::try_from(image),)*
                    _ => Err(at!(BufferError::IncompatibleDescriptor)),
                }
            }
        }

        /// Borrow a [`DynamicImage`] as a mutable slice, with its color.
        impl<'a> TryFrom<&'a mut DynamicImage> for PixelSliceMut<'a> {
            type Error = At<BufferError>;
            fn try_from(image: &'a mut DynamicImage) -> Result<Self, At<BufferError>> {
                match image {
                    $(DynamicImage::$variant(image) => PixelSliceMut::try_from(image),)*
                    _ => Err(at!(BufferError::IncompatibleDescriptor)),
                }
            }
        }

        /// Move a buffer into a [`DynamicImage`], with its color.
        ///
        /// The error is the buffer, unchanged, for a format `DynamicImage` has no
        /// variant for, such as gray `f32`, or whenever the matching `ImageBuffer`
        /// conversion refuses.
        impl TryFrom<PixelBuffer> for DynamicImage {
            type Error = PixelBuffer;
            fn try_from(buffer: PixelBuffer) -> Result<Self, PixelBuffer> {
                let descriptor = buffer.descriptor();
                let format = (descriptor.channel_type(), descriptor.layout());
                $(if format == (ChannelType::$channel, ChannelLayout::$layout) {
                    return ImageBuffer::<$pixel<$sample>, Vec<$sample>>::try_from(buffer)
                        .map(DynamicImage::$variant);
                })*
                Err(buffer)
            }
        }
    };
}

dynamic_image! {
    ImageLuma8: Luma<u8> => U8, Gray;
    ImageLumaA8: LumaA<u8> => U8, GrayAlpha;
    ImageRgb8: Rgb<u8> => U8, Rgb;
    ImageRgba8: Rgba<u8> => U8, Rgba;
    ImageLuma16: Luma<u16> => U16, Gray;
    ImageLumaA16: LumaA<u16> => U16, GrayAlpha;
    ImageRgb16: Rgb<u16> => U16, Rgb;
    ImageRgba16: Rgba<u16> => U16, Rgba;
    ImageRgb32F: Rgb<f32> => F32, Rgb;
    ImageRgba32F: Rgba<f32> => F32, Rgba;
}

// ---- Color types ----------------------------------------------------------

/// The format of an `image` color type.
impl TryFrom<ColorType> for PixelFormat {
    type Error = BufferError;
    fn try_from(color: ColorType) -> Result<Self, BufferError> {
        Ok(match color {
            ColorType::L8 => Self::Gray8,
            ColorType::La8 => Self::GrayA8,
            ColorType::Rgb8 => Self::Rgb8,
            ColorType::Rgba8 => Self::Rgba8,
            ColorType::L16 => Self::Gray16,
            ColorType::La16 => Self::GrayA16,
            ColorType::Rgb16 => Self::Rgb16,
            ColorType::Rgba16 => Self::Rgba16,
            ColorType::Rgb32F => Self::RgbF32,
            ColorType::Rgba32F => Self::RgbaF32,
            _ => return Err(BufferError::IncompatibleDescriptor),
        })
    }
}

/// The `image` color type of a format, when it has one.
impl TryFrom<PixelFormat> for ColorType {
    type Error = BufferError;
    fn try_from(format: PixelFormat) -> Result<Self, BufferError> {
        Ok(match format {
            PixelFormat::Gray8 => Self::L8,
            PixelFormat::GrayA8 => Self::La8,
            PixelFormat::Rgb8 => Self::Rgb8,
            PixelFormat::Rgba8 => Self::Rgba8,
            PixelFormat::Gray16 => Self::L16,
            PixelFormat::GrayA16 => Self::La16,
            PixelFormat::Rgb16 => Self::Rgb16,
            PixelFormat::Rgba16 => Self::Rgba16,
            PixelFormat::RgbF32 => Self::Rgb32F,
            PixelFormat::RgbaF32 => Self::Rgba32F,
            _ => return Err(BufferError::IncompatibleDescriptor),
        })
    }
}

/// The format of an `image` extended color type, for the ones with byte-addressable
/// samples and a packed format. The rest are refused: sub-byte layouts, `Rgb5x1` and
/// `Unknown` have no samples to address, and `A8` isn't a packed format: it is a
/// `plane::Plane` with `plane::PlaneRole::Alpha`.
impl TryFrom<ExtendedColorType> for PixelFormat {
    type Error = BufferError;
    fn try_from(color: ExtendedColorType) -> Result<Self, BufferError> {
        Ok(match color {
            ExtendedColorType::L8 => Self::Gray8,
            ExtendedColorType::La8 => Self::GrayA8,
            ExtendedColorType::Rgb8 => Self::Rgb8,
            ExtendedColorType::Rgba8 => Self::Rgba8,
            ExtendedColorType::L16 => Self::Gray16,
            ExtendedColorType::La16 => Self::GrayA16,
            ExtendedColorType::Rgb16 => Self::Rgb16,
            ExtendedColorType::Rgba16 => Self::Rgba16,
            ExtendedColorType::Bgra8 => Self::Bgra8,
            ExtendedColorType::Rgb32F => Self::RgbF32,
            ExtendedColorType::Rgba32F => Self::RgbaF32,
            ExtendedColorType::Cmyk8 => Self::CMYK8,
            ExtendedColorType::Cmyk16 => Self::CMYK16,
            ExtendedColorType::Bgr8 => Self::Bgr8,
            _ => return Err(BufferError::IncompatibleDescriptor),
        })
    }
}

/// The `image` extended color type of a format, when it has one.
impl TryFrom<PixelFormat> for ExtendedColorType {
    type Error = BufferError;
    fn try_from(format: PixelFormat) -> Result<Self, BufferError> {
        Ok(match format {
            PixelFormat::Gray8 => Self::L8,
            PixelFormat::GrayA8 => Self::La8,
            PixelFormat::Rgb8 => Self::Rgb8,
            PixelFormat::Rgba8 => Self::Rgba8,
            PixelFormat::Gray16 => Self::L16,
            PixelFormat::GrayA16 => Self::La16,
            PixelFormat::Rgb16 => Self::Rgb16,
            PixelFormat::Rgba16 => Self::Rgba16,
            PixelFormat::Bgra8 => Self::Bgra8,
            PixelFormat::RgbF32 => Self::Rgb32F,
            PixelFormat::RgbaF32 => Self::Rgba32F,
            PixelFormat::CMYK8 => Self::Cmyk8,
            PixelFormat::CMYK16 => Self::Cmyk16,
            PixelFormat::Bgr8 => Self::Bgr8,
            _ => return Err(BufferError::IncompatibleDescriptor),
        })
    }
}
