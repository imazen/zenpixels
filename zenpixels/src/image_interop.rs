//! Conversions to and from the `image` crate.
//!
//! `TryFrom` impls only. Nothing here copies samples. Color travels with packed
//! buffers in both directions, and a conversion that would have to relabel the
//! pixels refuses instead: `image` has no "unknown" color space and defaults to
//! sRGB.

use alloc::sync::Arc;
use core::ops::{Deref, DerefMut};
use whereat::ResultAtExt as _;

use image::flat::{FlatSamples, SampleLayout};
use image::metadata::{
    Cicp as ImageCicp, CicpColorPrimaries, CicpMatrixCoefficients, CicpTransferCharacteristics,
    CicpVideoFullRangeFlag,
};
use image::{
    ColorType, DynamicImage, ExtendedColorType, ImageBuffer, Luma, LumaA, Primitive, Rgb, Rgba,
};
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
use crate::plane::{Plane, PlaneError, PlaneMut, PlaneRef, PlaneSample, SampleStorage};

fn u32_dim(n: usize) -> Result<u32, PlaneError> {
    u32::try_from(n).map_err(|_| PlaneError::GeometryOverflow)
}

// ---- Planes: `Luma` images and strided views ------------------------------

/// Move a gray image's container into a plane.
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

// ---- Color ----------------------------------------------------------------

fn primaries_code(p: CicpColorPrimaries) -> u8 {
    match p {
        CicpColorPrimaries::SRgb => 1,
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
        if context.icc.is_some() {
            return None;
        }
        if let Some(cicp) = context.cicp {
            let matches = Some(cicp.color_primaries) == descriptor.primaries.to_cicp()
                && Some(cicp.transfer_characteristics) == descriptor.transfer.to_cicp()
                && cicp.matrix_coefficients == 0
                && cicp.full_range;
            if !matches {
                return None;
            }
        }
    }
    Some((primaries, transfer))
}

// ---- Packed buffers -------------------------------------------------------

/// The `image` labels when this buffer can move into an `ImageBuffer` of this format:
/// same format, straight alpha, tight rows, first pixel at index 0, color `image` can
/// state exactly.
fn movable<S: ByteStorage>(
    buffer: &PixelBuffer<(), S>,
    channel: ChannelType,
    layout: ChannelLayout,
    alpha: Option<AlphaMode>,
) -> Option<(CicpColorPrimaries, CicpTransferCharacteristics)> {
    let descriptor = buffer.descriptor();
    let alpha_ok = match alpha {
        None => descriptor.alpha().is_none(),
        Some(_) => matches!(
            descriptor.alpha(),
            Some(AlphaMode::Straight | AlphaMode::Opaque)
        ),
    };
    let tight = buffer.stride() == buffer.width() as usize * descriptor.bytes_per_pixel();
    if descriptor.channel_type() != channel
        || descriptor.layout() != layout
        || !alpha_ok
        || !tight
        || buffer.offset_bytes() != 0
    {
        return None;
    }
    image_color(descriptor, buffer.color_context())
}

macro_rules! packed_conversions {
    ($($pixel:ident<$sample:ty> => $channel:ident, $layout:ident, $alpha:expr;)*) => {$(
        /// Move an image's container into a buffer, with its color.
        impl<C> TryFrom<ImageBuffer<$pixel<$sample>, C>> for PixelBuffer<(), C>
        where
            C: SampleStorage<Sample = $sample> + Deref<Target = [$sample]>,
        {
            type Error = FromPartsError<C>;
            fn try_from(image: ImageBuffer<$pixel<$sample>, C>) -> Result<Self, FromPartsError<C>> {
                let (descriptor, cicp) =
                    describe(ChannelType::$channel, ChannelLayout::$layout, $alpha, image.color_space());
                let (width, height) = image.dimensions();
                let stride = width as usize * descriptor.bytes_per_pixel();
                let parts = PixelBufferParts::new(image.into_raw(), width, height, stride, descriptor);
                let buffer = PixelBuffer::try_from_parts(parts)?;
                Ok(match cicp {
                    Some(cicp) => buffer.with_cicp(cicp),
                    None => buffer,
                })
            }
        }

        /// Move an image's `Vec` into a default [`PixelBuffer`], with its color.
        impl TryFrom<ImageBuffer<$pixel<$sample>, Vec<$sample>>> for PixelBuffer {
            type Error = FromPartsError;
            fn try_from(image: ImageBuffer<$pixel<$sample>, Vec<$sample>>) -> Result<Self, FromPartsError> {
                let (descriptor, cicp) =
                    describe(ChannelType::$channel, ChannelLayout::$layout, $alpha, image.color_space());
                let (width, height) = image.dimensions();
                let stride = width as usize * descriptor.bytes_per_pixel();
                let parts = PixelBufferParts::new(PixelStorage::from(image.into_raw()), width, height, stride, descriptor);
                let buffer = PixelBuffer::try_from_parts(parts)?;
                Ok(match cicp {
                    Some(cicp) => buffer.with_cicp(cicp),
                    None => buffer,
                })
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
            fn try_from(image: &'a ImageBuffer<$pixel<$sample>, C>) -> Result<Self, At<BufferError>> {
                let (descriptor, cicp) =
                    describe(ChannelType::$channel, ChannelLayout::$layout, $alpha, image.color_space());
                let (width, height) = image.dimensions();
                let stride = width as usize * descriptor.bytes_per_pixel();
                let slice = PixelSlice::new(bytemuck::cast_slice(&**image), width, height, stride, descriptor)?;
                Ok(match cicp {
                    Some(cicp) => slice.with_cicp(cicp),
                    None => slice,
                })
            }
        }

        /// Borrow an image as a mutable slice, with its color.
        impl<'a, C> TryFrom<&'a mut ImageBuffer<$pixel<$sample>, C>> for PixelSliceMut<'a>
        where
            C: DerefMut<Target = [$sample]>,
        {
            type Error = At<BufferError>;
            fn try_from(image: &'a mut ImageBuffer<$pixel<$sample>, C>) -> Result<Self, At<BufferError>> {
                let (descriptor, cicp) =
                    describe(ChannelType::$channel, ChannelLayout::$layout, $alpha, image.color_space());
                let (width, height) = image.dimensions();
                let stride = width as usize * descriptor.bytes_per_pixel();
                let slice = PixelSliceMut::new(bytemuck::cast_slice_mut(&mut **image), width, height, stride, descriptor)?;
                Ok(match cicp {
                    Some(cicp) => slice.with_cicp(cicp),
                    None => slice,
                })
            }
        }

        /// Move a buffer's storage into an image, with its color.
        ///
        /// The error is the buffer, unchanged, when the format differs, the rows
        /// are not tight, the first pixel is past index 0, or `image` cannot state
        /// the color exactly. Relabel with `with_transfer` and `with_primaries`
        /// when the color is known and only the label is missing. Premultiplied
        /// and undefined alpha are refused because `image` means straight alpha.
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

macro_rules! flat_views {
    ($($sample:ty => $channel:ident;)*) => {$(
        /// View a slice, strided or not, through `image`'s strided type.
        ///
        /// `FlatSamples` carries no color, so read it from the slice first. Only
        /// gray, gray-alpha, RGB and RGBA have an `image` pixel type to read with.
        impl<'a> TryFrom<PixelSlice<'a>> for FlatSamples<&'a [$sample]> {
            type Error = At<BufferError>;
            fn try_from(slice: PixelSlice<'a>) -> Result<Self, At<BufferError>> {
                let descriptor = slice.descriptor();
                if descriptor.channel_type() != ChannelType::$channel {
                    return Err(at!(BufferError::IncompatibleDescriptor));
                }
                let channels = match descriptor.layout() {
                    ChannelLayout::Gray => 1u8,
                    ChannelLayout::GrayAlpha => 2,
                    ChannelLayout::Rgb => 3,
                    ChannelLayout::Rgba => 4,
                    _ => return Err(at!(BufferError::IncompatibleDescriptor)),
                };
                let samples: &'a [$sample] = bytemuck::try_cast_slice(slice.as_strided_bytes())
                    .map_err(|_| BufferError::AlignmentViolation).map_err(whereat::At::wrap).at_crate(crate::at_crate_info()).at()?;
                Ok(FlatSamples {
                    samples,
                    layout: SampleLayout {
                        channels,
                        channel_stride: 1,
                        width: slice.width(),
                        width_stride: usize::from(channels),
                        height: slice.rows(),
                        height_stride: slice.stride() / core::mem::size_of::<$sample>(),
                    },
                    color_hint: None,
                })
            }
        }
    )*};
}

flat_views! {
    u8 => U8;
    u16 => U16;
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
/// `Unknown` have no samples to address, and `A8` isn't a packed format: it is a [`Plane`]
/// with [`PlaneRole::Alpha`](crate::plane::PlaneRole::Alpha).
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
            ExtendedColorType::Cmyk8 => Self::Cmyk8,
            ExtendedColorType::Cmyk16 => Self::Cmyk16,
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
            PixelFormat::Cmyk8 => Self::Cmyk8,
            PixelFormat::Cmyk16 => Self::Cmyk16,
            PixelFormat::Bgr8 => Self::Bgr8,
            _ => return Err(BufferError::IncompatibleDescriptor),
        })
    }
}
