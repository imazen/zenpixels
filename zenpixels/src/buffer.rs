//! Unparameterized pixel buffers, zero-copy strided views, bands, planes, and dual-layer HDR.
//!
//! - [`PixelBuf`] owns an alignment-exact [`PixelStorage`], optional [`HdrMetadata`],
//!   [`ColorOrigin`], and cached [`SampleFacts`] (cleared automatically on `&mut`).
//! - [`PixelView<'a>`] and [`PixelViewMut<'a>`] borrow rectangular (possibly strided
//!   or cropped) subimages with lifetimes tied directly to `'a` backing storage.
//! - Zero-copy typed pixel access is provided by methods (`row_as::<Rgb<u8>>`,
//!   `from_typed`, `into_vec_zero_copy`) rather than a `<P = ()>` type parameter.
//! - [`BandedBuf`] handles gigapixel images in bounded-memory horizontal bands.
//! - [`PlanarBuf`] / [`PlanarView`] handle 1..=4 planes (YCbCr, NV12, P010, planar RGB+A)
//!   including N-bit [`BitPacking`] code extraction.
//! - [`DualLayerHdrBuf`] carries ISO 21496-1 / UltraHDR base + gain map + headroom metadata.

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use whereat::At;

use crate::cicp::MatrixCoefficients;
use crate::color::{
    AlphaMode, ColorError, ColorOrigin, ColorSpace, FormatCow, GainMapMetadata, HdrMetadata,
    PixelFormat, SignalRange,
};
use crate::facts::{FloatFacts, SampleFacts};
use crate::layout::{
    BitPacking, ChannelOrder, InteropTarget, LayoutError, PixelRect, PlaneFormat, PlaneRole,
    RowAlign, SampleLayout, SampleStride, SampleType,
};
use crate::orientation::Orientation;
use crate::pixel_types::{F16Sample, GrayAlpha8, GrayAlpha16, GrayAlphaF16, GrayAlphaF32};
use crate::sample::{ChromaSiting, CodePlane, CodePlaneError, CodePlaneMut, SampleEncoding};
use crate::storage::{
    AlignedSamples, ByteStorage, ByteStorageMut, PixelStorage, StorageError, StorageSample,
};

mod sealed {
    pub trait Sealed {}
}

/// Sealed marker trait for Rust scalar, channel-array, or pixel types that carry a
/// known [`SampleLayout`] compatibility contract.
///
/// Used by [`PixelBuf::from_typed`] and [`PixelView::from_typed`] to prevent silent
/// native-endian reinterpretation across unrelated scalar types (e.g. `Vec<i32>`
/// adopted as `RGBF32`). For intentional raw byte reinterpretation, use
/// [`PixelBuf::from_pod_bytes_unchecked`] or [`PixelView::from_pod_bytes_unchecked`].
pub trait TypedElement: bytemuck::Pod + sealed::Sealed {
    /// Whether this element type is structurally compatible with `layout`.
    fn matches_layout(layout: SampleLayout) -> bool;
}

macro_rules! impl_typed_scalar {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl sealed::Sealed for $ty {}
            impl TypedElement for $ty {
                #[inline]
                fn matches_layout(layout: SampleLayout) -> bool {
                    layout.sample == <$ty as StorageSample>::SAMPLE_TYPE
                }
            }
        )+
    };
}

impl_typed_scalar!(u8, u16, u32, u64, i8, i16, i32, i64, F16Sample, f32, f64);

macro_rules! impl_typed_array {
    ($($n:literal),+ $(,)?) => {
        $(
            impl<S: StorageSample> sealed::Sealed for [S; $n] {}
            impl<S: StorageSample> TypedElement for [S; $n] {
                #[inline]
                fn matches_layout(layout: SampleLayout) -> bool {
                    layout.sample == S::SAMPLE_TYPE
                        && !layout.packing.is_packed_word()
                        && layout.order.channels() == $n
                }
            }
        )+
    };
}

impl_typed_array!(1, 2, 3, 4, 5);

macro_rules! impl_typed_exact_layout {
    ($($ty:ty => $layout:expr),+ $(,)?) => {
        $(
            impl sealed::Sealed for $ty {}
            impl TypedElement for $ty {
                #[inline]
                fn matches_layout(layout: SampleLayout) -> bool {
                    layout == $layout
                }
            }
        )+
    };
}

impl_typed_exact_layout!(
    Rgbx => SampleLayout::RGBX8,
    Bgrx => SampleLayout::BGRX8,
    GrayAlpha8 => SampleLayout::GRAYA8,
    GrayAlpha16 => SampleLayout::GRAYA16,
    GrayAlphaF16 => SampleLayout::GRAYAF16,
    GrayAlphaF32 => SampleLayout::GRAYAF32,
);

#[cfg(feature = "rgb")]
mod rgb_typed_impls {
    use super::{SampleLayout, SampleType, TypedElement, sealed};
    use crate::layout::ChannelOrder;
    use crate::pixel_types::F16Sample;
    use rgb::{Bgr, Bgra, Gray, Rgb, Rgba};

    macro_rules! impl_rgb_typed {
        ($($ty:ty => $layout:expr),+ $(,)?) => {
            $(
                impl sealed::Sealed for $ty {}
                impl TypedElement for $ty {
                    #[inline]
                    fn matches_layout(layout: SampleLayout) -> bool {
                        layout == $layout
                    }
                }
            )+
        };
    }

    impl_rgb_typed!(
        Gray<u8> => SampleLayout::GRAY8,
        Gray<u16> => SampleLayout::GRAY16,
        Gray<F16Sample> => SampleLayout::GRAYF16,
        Gray<f32> => SampleLayout::GRAYF32,
        Rgb<u8> => SampleLayout::RGB8,
        Rgb<u16> => SampleLayout::RGB16,
        Rgb<F16Sample> => SampleLayout::RGBF16,
        Rgb<f32> => SampleLayout::RGBF32,
        Rgba<u8> => SampleLayout::RGBA8,
        Rgba<u16> => SampleLayout::RGBA16,
        Rgba<F16Sample> => SampleLayout::RGBAF16,
        Rgba<f32> => SampleLayout::RGBAF32,
        Bgr<u8> => SampleLayout::BGR8,
        Bgr<u16> => SampleLayout::new(SampleType::U16, ChannelOrder::Bgr),
        Bgr<F16Sample> => SampleLayout::new(SampleType::F16, ChannelOrder::Bgr),
        Bgr<f32> => SampleLayout::new(SampleType::F32, ChannelOrder::Bgr),
        Bgra<u8> => SampleLayout::BGRA8,
        Bgra<u16> => SampleLayout::new(SampleType::U16, ChannelOrder::Bgra),
        Bgra<F16Sample> => SampleLayout::new(SampleType::F16, ChannelOrder::Bgra),
        Bgra<f32> => SampleLayout::new(SampleType::F32, ChannelOrder::Bgra),
    );
}

/// 4-byte RGB + unused padding pixel (`#[repr(C)]`, `align = 1`).
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Rgbx {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Unspecified padding byte.
    pub x: u8,
}

/// 4-byte BGR + unused padding pixel (`#[repr(C)]`, `align = 1`).
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Bgrx {
    /// Blue channel.
    pub b: u8,
    /// Green channel.
    pub g: u8,
    /// Red channel.
    pub r: u8,
    /// Unspecified padding byte.
    pub x: u8,
}

/// Initial pixel fill color for freshly allocated [`PixelBuf`]s
/// ([`PixelBuf::new_with_fill`], [`PixelBuf::new_white`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BufFill {
    /// Zeroed black pixels (alpha slot filled to channel maximum when `AlphaMode::Opaque`).
    #[default]
    Black,
    /// Solid opaque white pixels (`MAX` / `1.0` for full range, studio white `235 << shift`
    /// for narrow integer range; alpha slot filled to channel maximum).
    White,
}

/// Errors returned by pixel buffer, view, band, and plane operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferError {
    /// Image specification failed validation.
    InvalidFormat(ColorError),
    /// Layout or stride calculation error.
    InvalidLayout(LayoutError),
    /// Base pointer or row stride does not satisfy the sample or pixel alignment.
    AlignmentViolation,
    /// Provided byte or pixel slice is shorter than required for `(width, height, stride)`.
    InsufficientData,
    /// Row stride is smaller than `width * bytes_per_pixel`.
    StrideTooSmall,
    /// Stride is not a whole multiple of `size_of::<P>()` for a pixel-strided API (e.g. `imgref`).
    StrideNotPixelAligned,
    /// Zero dimension or arithmetic overflow.
    InvalidDimensions,
    /// Requested crop, band, or plane index is out of bounds.
    OutOfBounds,
    /// Typed element size does not divide the active row or buffer extent cleanly.
    IncompatibleLayout,
    /// Spec does not satisfy the requested [`InteropTarget`]'s zero-copy contract.
    InteropUnsupported,
    /// Heap allocation failed (OOM or allocation layout overflow).
    AllocationFailed,
    /// ISO 21496-1 gain-map metadata failed invariant validation (`gamma == 0`, `min > max`,
    /// equal headrooms, or contradictory `backward_direction`).
    InvalidGainMapMetadata,
    /// Buffer declares or requested [`AlphaMode::Opaque`] without verified channel-maximum alpha
    /// (for example `PixelBuf::new_zeroed` with `AlphaMode::Opaque`, or `into_opaque_verified`
    /// encountering a non-maximum alpha sample).
    UnverifiedOpaqueAlpha,
}

impl From<ColorError> for BufferError {
    fn from(e: ColorError) -> Self {
        Self::InvalidFormat(e)
    }
}

impl From<LayoutError> for BufferError {
    fn from(e: LayoutError) -> Self {
        Self::InvalidLayout(e)
    }
}

impl From<StorageError> for BufferError {
    fn from(_: StorageError) -> Self {
        Self::AllocationFailed
    }
}

impl fmt::Display for BufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat(e) => e.fmt(f),
            Self::InvalidLayout(e) => e.fmt(f),
            Self::AlignmentViolation => f.write_str("pointer or stride violates sample alignment"),
            Self::InsufficientData => f.write_str("buffer is smaller than required image extent"),
            Self::StrideTooSmall => f.write_str("stride is smaller than active row bytes"),
            Self::StrideNotPixelAligned => {
                f.write_str("stride is not a whole multiple of bytes_per_pixel")
            }
            Self::InvalidDimensions => f.write_str("width or height is zero or overflows"),
            Self::OutOfBounds => f.write_str("crop, row range, or plane index is out of bounds"),
            Self::IncompatibleLayout => {
                f.write_str("typed element layout is incompatible with pixel buffer")
            }
            Self::InteropUnsupported => {
                f.write_str("image spec does not satisfy requested zero-copy interop contract")
            }
            Self::AllocationFailed => f.write_str("pixel buffer allocation failed"),
            Self::InvalidGainMapMetadata => {
                f.write_str("ISO 21496-1 gain-map metadata failed invariant validation")
            }
            Self::UnverifiedOpaqueAlpha => f.write_str(
                "AlphaMode::Opaque requires black/white fill, verification, or explicit assume_opaque",
            ),
        }
    }
}

impl core::error::Error for BufferError {}

#[cold]
#[inline(never)]
#[track_caller]
pub(crate) fn err_at(e: BufferError) -> At<BufferError> {
    whereat::at!(e)
}

/// Minimum bytes needed for `(rows - 1) * stride + row_bytes` (last row needs no trailing padding).
pub(crate) fn required_bytes(
    rows: u32,
    stride: usize,
    row_bytes: usize,
) -> Result<usize, BufferError> {
    if rows == 0 || row_bytes == 0 {
        return Ok(0);
    }
    let extent = u64::from(rows - 1)
        .checked_mul(stride as u64)
        .and_then(|v| v.checked_add(row_bytes as u64))
        .filter(|&b| b <= isize::MAX as u64)
        .ok_or(BufferError::InvalidDimensions)?;
    Ok(extent as usize)
}

fn validate_geometry(
    data_len: usize,
    data_ptr: *const u8,
    width: u32,
    rows: u32,
    stride: usize,
    spec: &PixelFormat,
) -> Result<(usize, usize), At<BufferError>> {
    spec.validate()
        .map_err(|e| whereat::at!(BufferError::InvalidFormat(*e.error())))?;
    if width == 0 || rows == 0 {
        return Err(err_at(BufferError::InvalidDimensions));
    }
    let bpp = spec.layout.bytes_per_pixel();
    let row_bytes = (width as usize)
        .checked_mul(bpp)
        .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
    if stride < row_bytes {
        return Err(err_at(BufferError::StrideTooSmall));
    }
    let align = spec.layout.min_alignment();
    if stride % align != 0 || (data_ptr as usize) % align != 0 {
        return Err(err_at(BufferError::AlignmentViolation));
    }
    let req = required_bytes(rows, stride, row_bytes).map_err(err_at)?;
    if data_len < req {
        return Err(err_at(BufferError::InsufficientData));
    }
    Ok((row_bytes, req))
}

/// Fill the alpha channel of every active pixel to its canonical maximum value
/// when `spec.color.alpha() == AlphaMode::Opaque`.
fn fill_opaque_alpha(bytes: &mut [u8], width: u32, height: u32, stride: usize, spec: &PixelFormat) {
    if spec.color.alpha() != AlphaMode::Opaque || !spec.layout.order.has_alpha_slot() {
        return;
    }
    let bpp = spec.layout.bytes_per_pixel();
    let channels = spec.layout.order.channels();
    let alpha_idx = channels - 1;
    let sample = spec.layout.sample;
    let packing = spec.layout.packing;

    if packing.is_packed_word() {
        let word_bytes = sample.byte_size();
        let alpha_bits = packing.pack_code(sample, alpha_idx, packing.max_code(sample, alpha_idx));
        for y in 0..height as usize {
            let row = &mut bytes[y * stride..y * stride + (width as usize) * bpp];
            for px in row.chunks_exact_mut(word_bytes) {
                let w = read_sample_u32(sample, px) | alpha_bits;
                match sample {
                    SampleType::U8 => px[0] = w as u8,
                    SampleType::U16 => px.copy_from_slice(&(w as u16).to_ne_bytes()),
                    SampleType::U32 => px.copy_from_slice(&w.to_ne_bytes()),
                    _ => {}
                }
            }
        }
        return;
    }

    let sz = sample.byte_size();
    let alpha_off = alpha_idx * sz;
    let max_bytes: [u8; 8] = match (sample, packing) {
        (SampleType::U8, BitPacking::Full) => [255, 0, 0, 0, 0, 0, 0, 0],
        (SampleType::U16, BitPacking::Full) => {
            let b = u16::MAX.to_ne_bytes();
            [b[0], b[1], 0, 0, 0, 0, 0, 0]
        }
        (SampleType::U32, BitPacking::Full) => {
            let b = u32::MAX.to_ne_bytes();
            [b[0], b[1], b[2], b[3], 0, 0, 0, 0]
        }
        (SampleType::U64, BitPacking::Full) => u64::MAX.to_ne_bytes(),
        (SampleType::I8, BitPacking::Full) => [i8::MAX as u8, 0, 0, 0, 0, 0, 0, 0],
        (SampleType::I16, BitPacking::Full) => {
            let b = i16::MAX.to_ne_bytes();
            [b[0], b[1], 0, 0, 0, 0, 0, 0]
        }
        (SampleType::I32, BitPacking::Full) => {
            let b = i32::MAX.to_ne_bytes();
            [b[0], b[1], b[2], b[3], 0, 0, 0, 0]
        }
        (SampleType::I64, BitPacking::Full) => i64::MAX.to_ne_bytes(),
        (SampleType::F16, BitPacking::Full) => {
            let b = 0x3C00u16.to_ne_bytes();
            [b[0], b[1], 0, 0, 0, 0, 0, 0]
        }
        (SampleType::F32, BitPacking::Full) => {
            let b = 1.0f32.to_ne_bytes();
            [b[0], b[1], b[2], b[3], 0, 0, 0, 0]
        }
        (SampleType::F64, BitPacking::Full) => 1.0f64.to_ne_bytes(),
        (
            SampleType::U8 | SampleType::U16 | SampleType::U32,
            BitPacking::Lsb { .. } | BitPacking::Msb { .. } | BitPacking::Shifted { .. },
        ) => {
            let max_code = packing.max_code(sample, alpha_idx);
            let packed = packing.pack_code(sample, alpha_idx, max_code);
            let b = packed.to_ne_bytes();
            [b[0], b[1], b[2], b[3], 0, 0, 0, 0]
        }
        _ => return,
    };

    for y in 0..height as usize {
        let row = &mut bytes[y * stride..y * stride + (width as usize) * bpp];
        for px in row.chunks_exact_mut(bpp) {
            px[alpha_off..alpha_off + sz].copy_from_slice(&max_bytes[..sz]);
        }
    }
}

/// Fill the alpha channel of every active pixel to its canonical maximum value
/// regardless of `spec.color.alpha()` (used by [`PixelBuf::into_opaque_filled`]).
fn fill_alpha_max_unconditional(
    bytes: &mut [u8],
    width: u32,
    height: u32,
    stride: usize,
    layout: SampleLayout,
) {
    if !layout.order.has_alpha_slot() {
        return;
    }
    let tmp_spec = PixelFormat::new(
        layout,
        ColorSpace::Untagged {
            channels: layout.channels() as u8,
            alpha: AlphaMode::Opaque,
        },
    );
    fill_opaque_alpha(bytes, width, height, stride, &tmp_spec);
}

/// Return `true` iff every active pixel's alpha channel equals its canonical maximum value.
fn check_alpha_all_max(
    bytes: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    layout: SampleLayout,
) -> bool {
    if !layout.order.has_alpha_slot() {
        return true;
    }
    let bpp = layout.bytes_per_pixel();
    let channels = layout.order.channels();
    let alpha_idx = channels - 1;
    let sample = layout.sample;
    let packing = layout.packing;

    if packing.is_packed_word() {
        let word_bytes = sample.byte_size();
        let want = packing.max_code(sample, alpha_idx);
        for y in 0..height as usize {
            let row = &bytes[y * stride..y * stride + (width as usize) * bpp];
            for px in row.chunks_exact(word_bytes) {
                let w = read_sample_u32(sample, px);
                if packing.extract_code(sample, w, alpha_idx) != want {
                    return false;
                }
            }
        }
        return true;
    }

    let sz = sample.byte_size();
    let alpha_off = alpha_idx * sz;
    if !packing.is_full() {
        let want = packing.max_code(sample, alpha_idx);
        for y in 0..height as usize {
            let row = &bytes[y * stride..y * stride + (width as usize) * bpp];
            for px in row.chunks_exact(bpp) {
                let w = read_sample_u32(sample, &px[alpha_off..alpha_off + sz]);
                if packing.extract_code(sample, w, alpha_idx) != want {
                    return false;
                }
            }
        }
        return true;
    }

    for y in 0..height as usize {
        let row = &bytes[y * stride..y * stride + (width as usize) * bpp];
        for px in row.chunks_exact(bpp) {
            let a = &px[alpha_off..alpha_off + sz];
            let ok = match sample {
                SampleType::U8 => a[0] == u8::MAX,
                SampleType::U16 => u16::from_ne_bytes([a[0], a[1]]) == u16::MAX,
                SampleType::U32 => u32::from_ne_bytes([a[0], a[1], a[2], a[3]]) == u32::MAX,
                SampleType::U64 => {
                    u64::from_ne_bytes([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]]) == u64::MAX
                }
                SampleType::I8 => (a[0] as i8) == i8::MAX,
                SampleType::I16 => i16::from_ne_bytes([a[0], a[1]]) == i16::MAX,
                SampleType::I32 => i32::from_ne_bytes([a[0], a[1], a[2], a[3]]) == i32::MAX,
                SampleType::I64 => {
                    i64::from_ne_bytes([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]]) == i64::MAX
                }
                SampleType::F16 => u16::from_ne_bytes([a[0], a[1]]) == 0x3C00,
                SampleType::F32 => f32::from_ne_bytes([a[0], a[1], a[2], a[3]]) == 1.0,
                SampleType::F64 => {
                    f64::from_ne_bytes([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]]) == 1.0
                }
            };
            if !ok {
                return false;
            }
        }
    }
    true
}

/// Fill every active pixel with reference white (and channel-maximum alpha on any alpha slot).
fn fill_white_pixels(bytes: &mut [u8], width: u32, height: u32, stride: usize, spec: &PixelFormat) {
    let layout = spec.layout;
    let bpp = layout.bytes_per_pixel();
    let channels = layout.order.channels();
    let sample = layout.sample;
    let packing = layout.packing;
    let has_alpha = layout.order.has_alpha_slot();
    let has_padding = layout.order.has_padding_slot();
    let color_ch = channels - usize::from(has_alpha || has_padding);

    if packing.is_packed_word() {
        let word_bytes = sample.byte_size();
        let mut word = 0u32;
        for c in 0..channels {
            let max_c = packing.max_code(sample, c);
            let code = if c < color_ch && spec.color.range() == Some(SignalRange::Narrow) {
                let bits = (32 - max_c.leading_zeros()).max(8);
                (235u32 << (bits - 8)).min(max_c)
            } else {
                max_c
            };
            word |= packing.pack_code(sample, c, code);
        }
        let wb = word.to_ne_bytes();
        for y in 0..height as usize {
            let row = &mut bytes[y * stride..y * stride + (width as usize) * bpp];
            for px in row.chunks_exact_mut(word_bytes) {
                px.copy_from_slice(&wb[..word_bytes]);
            }
        }
        return;
    }

    let sz = sample.byte_size();
    let mut white_px = [0u8; 128]; // max 16 channels × 8 B = 128 B
    let is_narrow = spec.color.range() == Some(SignalRange::Narrow);

    let encode_sample = |dst: &mut [u8], ch_idx: usize, is_alpha: bool| {
        if !packing.is_full() {
            let max_c = packing.max_code(sample, ch_idx);
            let code = if !is_alpha && is_narrow {
                let bits = (32 - max_c.leading_zeros()).max(8);
                (235u32 << (bits - 8)).min(max_c)
            } else {
                max_c
            };
            let packed = packing.pack_code(sample, ch_idx, code);
            dst.copy_from_slice(&packed.to_ne_bytes()[..sz]);
            return;
        }
        match sample {
            SampleType::U8 => dst[0] = if !is_alpha && is_narrow { 235 } else { u8::MAX },
            SampleType::U16 => {
                let v: u16 = if !is_alpha && is_narrow {
                    235u16 << 8
                } else {
                    u16::MAX
                };
                dst.copy_from_slice(&v.to_ne_bytes());
            }
            SampleType::U32 => {
                let v: u32 = if !is_alpha && is_narrow {
                    235u32 << 24
                } else {
                    u32::MAX
                };
                dst.copy_from_slice(&v.to_ne_bytes());
            }
            SampleType::U64 => dst.copy_from_slice(&u64::MAX.to_ne_bytes()),
            SampleType::I8 => dst[0] = i8::MAX as u8,
            SampleType::I16 => dst.copy_from_slice(&i16::MAX.to_ne_bytes()),
            SampleType::I32 => dst.copy_from_slice(&i32::MAX.to_ne_bytes()),
            SampleType::I64 => dst.copy_from_slice(&i64::MAX.to_ne_bytes()),
            SampleType::F16 => dst.copy_from_slice(&0x3C00u16.to_ne_bytes()),
            SampleType::F32 => dst.copy_from_slice(&1.0f32.to_ne_bytes()),
            SampleType::F64 => dst.copy_from_slice(&1.0f64.to_ne_bytes()),
        }
    };

    match &spec.color {
        ColorSpace::Oklab { .. } => {
            // Oklab white: L = 1.0, a = 0.0, b = 0.0 (+ alpha = MAX).
            encode_sample(&mut white_px[0..sz], 0, false);
            if has_alpha {
                encode_sample(
                    &mut white_px[(channels - 1) * sz..channels * sz],
                    channels - 1,
                    true,
                );
            }
        }
        ColorSpace::Xyz { white_point, .. } => {
            let (wx, wy) = white_point.xy();
            let xyz = [wx / wy, 1.0_f32, (1.0 - wx - wy) / wy];
            for (c, &v) in xyz.iter().enumerate() {
                match sample {
                    SampleType::F32 => {
                        white_px[c * 4..(c + 1) * 4].copy_from_slice(&v.to_ne_bytes());
                    }
                    SampleType::F16 => {
                        let h = crate::pixel_types::F16Sample::from_f32(v).to_bits();
                        white_px[c * 2..(c + 1) * 2].copy_from_slice(&h.to_ne_bytes());
                    }
                    SampleType::F64 => {
                        white_px[c * 8..(c + 1) * 8].copy_from_slice(&f64::from(v).to_ne_bytes());
                    }
                    _ => encode_sample(&mut white_px[c * sz..(c + 1) * sz], c, false),
                }
            }
            if has_alpha {
                encode_sample(&mut white_px[3 * sz..4 * sz], 3, true);
            }
        }
        ColorSpace::Cmyk {
            inverted: false, ..
        } => {
            // Non-inverted CMYK: 0 = no ink = white paper! Only alpha is MAX.
            if has_alpha {
                encode_sample(
                    &mut white_px[(channels - 1) * sz..channels * sz],
                    channels - 1,
                    true,
                );
            }
        }
        _ => {
            for c in 0..color_ch {
                encode_sample(&mut white_px[c * sz..(c + 1) * sz], c, false);
            }
            if has_alpha {
                encode_sample(
                    &mut white_px[(channels - 1) * sz..channels * sz],
                    channels - 1,
                    true,
                );
            }
        }
    }

    let pat = &white_px[..bpp];
    for y in 0..height as usize {
        let row = &mut bytes[y * stride..y * stride + (width as usize) * bpp];
        for px in row.chunks_exact_mut(bpp) {
            px.copy_from_slice(pat);
        }
    }
}

// ---------------------------------------------------------------------------
// PixelView (borrowed, immutable, true `'a` lifetimes)
// ---------------------------------------------------------------------------

static ASSUMED_ORIGIN: ColorOrigin = ColorOrigin::assumed();

#[inline]
fn box_hdr(hdr: HdrMetadata) -> Option<Arc<HdrMetadata>> {
    if hdr.is_empty() {
        None
    } else {
        Some(Arc::new(hdr))
    }
}

#[inline]
fn unbox_hdr(hdr: &Option<Arc<HdrMetadata>>) -> HdrMetadata {
    match hdr {
        Some(h) => **h,
        None => HdrMetadata::default(),
    }
}

#[inline]
fn box_origin(origin: ColorOrigin) -> Option<Arc<ColorOrigin>> {
    if origin.is_assumed() {
        None
    } else {
        Some(Arc::new(origin))
    }
}

#[inline]
fn origin_ref(origin: &Option<Arc<ColorOrigin>>) -> &ColorOrigin {
    match origin {
        Some(o) => o,
        None => &ASSUMED_ORIGIN,
    }
}

#[inline]
fn unbox_origin(origin: Option<Arc<ColorOrigin>>) -> ColorOrigin {
    match origin {
        Some(o) => Arc::unwrap_or_clone(o),
        None => ColorOrigin::assumed(),
    }
}

/// Zero-copy borrowed view of an interleaved image or rectangular crop.
///
/// Every slicing method (`row`, `row_as`, `crop`, `sub_rows`,
/// `as_contiguous_bytes`) returns references or subviews tied directly to `'a`
/// rather than `&self`, so chaining (`buf.as_view().crop(...)?.row(0)`) works cleanly.
#[derive(Clone, Debug)]
pub struct PixelView<'a> {
    data: &'a [u8],
    width: u32,
    height: u32,
    stride_bytes: usize,
    spec: FormatCow<'a>,
    facts: SampleFacts,
    hdr: Option<Arc<HdrMetadata>>,
    origin: Option<&'a ColorOrigin>,
    orientation: Orientation,
}

impl<'a> PixelView<'a> {
    /// Borrow strided bytes with checked dimensions, alignment, and length (**zero heap allocation**).
    ///
    /// Pass [`&formats::RGB8_SRGB`](crate::formats::RGB8_SRGB) (or any `crate::formats::*` preset) or a borrowed
    /// `&'a PixelFormat`. If a view genuinely needs to own a dynamically constructed [`PixelFormat`],
    /// use [`new_with_format_cow`](Self::new_with_format_cow) with [`FormatCow::from_owned`] or
    /// [`FormatCow::from_arc`].
    ///
    /// Trims any trailing bytes beyond the active `(height - 1) * stride_bytes + row_bytes`
    /// extent so zero-copy interop bridges (`as_imgref`) never touch trailing slop.
    pub fn new(
        data: &'a [u8],
        width: u32,
        height: u32,
        stride_bytes: usize,
        format: &'a PixelFormat,
    ) -> Result<Self, At<BufferError>> {
        Self::new_with_format_cow(
            data,
            width,
            height,
            stride_bytes,
            FormatCow::Borrowed(format),
        )
    }

    /// Borrow strided bytes with an explicit [`FormatCow<'a>`] (borrowed `&'a PixelFormat` or
    /// shared `Arc<PixelFormat>` via [`FormatCow::from_owned`] / [`FormatCow::from_parametric`] /
    /// [`FormatCow::from_arc`]).
    ///
    /// Trims any trailing bytes beyond the active `(height - 1) * stride_bytes + row_bytes` extent.
    pub fn new_with_format_cow(
        data: &'a [u8],
        width: u32,
        height: u32,
        stride_bytes: usize,
        format: FormatCow<'a>,
    ) -> Result<Self, At<BufferError>> {
        let (_rb, req) = validate_geometry(
            data.len(),
            data.as_ptr(),
            width,
            height,
            stride_bytes,
            format.as_format(),
        )?;
        Ok(Self {
            data: &data[..req],
            width,
            height,
            stride_bytes,
            spec: format,
            facts: SampleFacts::NONE,
            hdr: None,
            origin: None,
            orientation: Orientation::Identity,
        })
    }

    /// Borrow tightly packed native-endian bytes (`stride_bytes == width * bytes_per_pixel`, **zero allocation**).
    pub fn from_bytes(
        data: &'a [u8],
        width: u32,
        height: u32,
        format: &'a PixelFormat,
    ) -> Result<Self, At<BufferError>> {
        let stride_bytes = (width as usize)
            .checked_mul(format.layout.bytes_per_pixel())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        Self::new(data, width, height, stride_bytes, format)
    }

    /// Borrow a tightly packed typed pixel or sample slice zero-copy via `bytemuck` (**zero allocation**).
    ///
    /// Validates that `P` is structurally compatible with `format.layout` (rejecting
    /// e.g. `&[i32]` as `RGBF32`). For intentional raw byte reinterpretation, use
    /// [`from_pod_bytes_unchecked`](Self::from_pod_bytes_unchecked).
    pub fn from_typed<P: TypedElement>(
        pixels: &'a [P],
        width: u32,
        height: u32,
        format: &'a PixelFormat,
    ) -> Result<Self, At<BufferError>> {
        if !P::matches_layout(format.layout) {
            return Err(err_at(BufferError::IncompatibleLayout).at_string(|| {
                alloc::format!(
                    "element `{}` does not match {:?} layout (annotate integer literals e.g. `0u8`, or use `from_pod_bytes_unchecked`)",
                    core::any::type_name::<P>(),
                    format.layout
                )
            }));
        }
        Self::from_bytes(bytemuck::cast_slice(pixels), width, height, format)
    }

    /// Borrow a tightly packed scalar sample slice (`&[u8]`, `&[u16]`, `&[f32]`, etc.) zero-copy (**zero allocation**).
    pub fn from_samples<S: StorageSample>(
        samples: &'a [S],
        width: u32,
        height: u32,
        format: &'a PixelFormat,
    ) -> Result<Self, At<BufferError>> {
        if format.layout.sample != S::SAMPLE_TYPE {
            return Err(err_at(BufferError::IncompatibleLayout).at_string(|| {
                alloc::format!(
                    "sample `{}` ({:?}) does not match {:?} layout",
                    core::any::type_name::<S>(),
                    S::SAMPLE_TYPE,
                    format.layout
                )
            }));
        }
        Self::from_bytes(bytemuck::cast_slice(samples), width, height, format)
    }

    /// Borrow a tightly packed `rgb` pixel slice (`&[rgb::Rgb<u8>]`, etc.) zero-copy with an
    /// explicit [`ColorSpace`] (deriving [`SampleLayout`] from `P::LAYOUT`).
    ///
    /// Uses a zero-allocation `&'static PixelFormat` for all standard built-in presets, and
    /// allocates an `Arc<PixelFormat>` only for custom/ICC color spaces.
    #[cfg(feature = "rgb")]
    pub fn from_pixels_with_color<P: crate::interop::PixelType>(
        pixels: &'a [P],
        width: u32,
        height: u32,
        color: crate::ColorSpace,
    ) -> Result<Self, At<BufferError>> {
        let stride_bytes = (width as usize)
            .checked_mul(P::LAYOUT.bytes_per_pixel())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        Self::new_with_format_cow(
            bytemuck::cast_slice(pixels),
            width,
            height,
            stride_bytes,
            FormatCow::from_owned(PixelFormat::new(P::LAYOUT, color)),
        )
    }

    /// Explicitly reinterpret any tightly packed `bytemuck::Pod` slice in native endianness
    /// as raw pixel bytes matching `format` without checking element-to-layout compatibility
    /// (**zero allocation**).
    pub fn from_pod_bytes_unchecked<P: bytemuck::Pod>(
        pixels: &'a [P],
        width: u32,
        height: u32,
        format: &'a PixelFormat,
    ) -> Result<Self, At<BufferError>> {
        Self::from_bytes(bytemuck::cast_slice(pixels), width, height, format)
    }

    /// Image width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Image height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Byte distance between consecutive row starts.
    #[inline]
    #[must_use]
    pub const fn stride_bytes(&self) -> usize {
        self.stride_bytes
    }

    /// Active bytes per row (`width * bytes_per_pixel`).
    #[inline]
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        (self.width as usize) * self.spec.as_format().layout.bytes_per_pixel()
    }

    /// Authoritative image specification (`layout` + `color`).
    #[inline]
    #[must_use]
    pub fn format(&self) -> &PixelFormat {
        self.spec.as_format()
    }

    /// Proven sample facts attached to this view.
    #[inline]
    #[must_use]
    pub const fn facts(&self) -> SampleFacts {
        self.facts
    }

    /// Descriptive HDR container metadata attached to this view.
    #[inline]
    #[must_use]
    pub fn hdr(&self) -> HdrMetadata {
        unbox_hdr(&self.hdr)
    }

    /// Original container color provenance borrowed from the owning image, if attached.
    #[inline]
    #[must_use]
    pub const fn origin(&self) -> Option<&'a ColorOrigin> {
        self.origin
    }

    /// Pending EXIF display orientation attached to this view (defaults to [`Orientation::Identity`]).
    #[inline]
    #[must_use]
    pub const fn orientation(&self) -> Orientation {
        self.orientation
    }

    /// Display-oriented `(width, height)` after applying [`orientation`](Self::orientation).
    #[inline]
    #[must_use]
    pub const fn display_dimensions(&self) -> (u32, u32) {
        self.orientation.output_dimensions(self.width, self.height)
    }

    /// Attach caller-asserted [`SampleFacts`] to this view (caller assumes responsibility for validity).
    #[inline]
    #[must_use]
    pub fn with_assumed_facts(mut self, facts: SampleFacts) -> Self {
        self.facts = facts;
        self
    }

    /// Attach descriptive HDR metadata to this view.
    #[inline]
    #[must_use]
    pub fn with_hdr(mut self, hdr: HdrMetadata) -> Self {
        self.hdr = box_hdr(hdr);
        self
    }

    /// Attach borrowed container color provenance to this view.
    #[inline]
    #[must_use]
    pub const fn with_origin(mut self, origin: &'a ColorOrigin) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Replace the pending EXIF display orientation on this view.
    #[inline]
    #[must_use]
    pub const fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Compose an additional display transform (including reflections) onto this view's
    /// pending orientation (`self.orientation.then(delta)`) in O(1) without touching pixels.
    #[inline]
    #[must_use]
    pub const fn compose_orientation(mut self, delta: Orientation) -> Self {
        self.orientation = self.orientation.then(delta);
        self
    }

    /// Active bytes of row `y` (excludes inter-row padding), tied to backing lifetime `'a`.
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height()`.
    #[inline]
    #[must_use]
    pub fn row(&self, y: u32) -> &'a [u8] {
        assert!(
            y < self.height,
            "row {y} out of bounds (height: {})",
            self.height
        );
        let start = (y as usize) * self.stride_bytes;
        &self.data[start..start + self.row_bytes()]
    }

    /// Zero-copy typed slice view of row `y` (e.g. `view.row_as::<rgb::Rgb<u8>>(y)`).
    ///
    /// Validates `bytemuck::try_cast_slice` alignment and exact element divisibility.
    pub fn row_as<P: bytemuck::Pod>(&self, y: u32) -> Result<&'a [P], At<BufferError>> {
        if y >= self.height {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let bytes = self.row(y);
        bytemuck::try_cast_slice(bytes).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout).at_string(|| {
                alloc::format!(
                    "row byte length {} is not compatible with element `{}` (size {})",
                    bytes.len(),
                    core::any::type_name::<P>(),
                    core::mem::size_of::<P>()
                )
            }),
        })
    }

    /// Raw strided byte backing starting at `(0, 0)` (includes inter-row padding).
    #[inline]
    #[must_use]
    pub const fn as_strided_bytes(&self) -> &'a [u8] {
        self.data
    }

    /// Whether rows are tightly packed (`stride_bytes == row_bytes` or `height <= 1`).
    #[inline]
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.height <= 1 || self.stride_bytes == self.row_bytes()
    }

    /// Contiguous packed bytes across all rows, or `None` if padded.
    #[inline]
    #[must_use]
    pub fn as_contiguous_bytes(&self) -> Option<&'a [u8]> {
        if self.is_contiguous() {
            let total = self.row_bytes() * (self.height as usize);
            Some(&self.data[..total])
        } else {
            None
        }
    }

    /// Zero-copy contiguous typed slice across all rows (e.g. `&[rgb::Rgba<f32>]`).
    ///
    /// Validates `bytemuck::try_cast_slice` alignment and exact element divisibility (raw POD
    /// reinterpretation, not semantic channel-order validation).
    pub fn as_contiguous_typed<P: bytemuck::Pod>(&self) -> Result<&'a [P], At<BufferError>> {
        let bytes = self
            .as_contiguous_bytes()
            .ok_or_else(|| err_at(BufferError::StrideNotPixelAligned))?;
        bytemuck::try_cast_slice(bytes).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout).at_string(|| {
                alloc::format!(
                    "contiguous byte length {} is not compatible with element `{}` (size {})",
                    bytes.len(),
                    core::any::type_name::<P>(),
                    core::mem::size_of::<P>()
                )
            }),
        })
    }

    /// Borrow contiguous bytes if tightly packed, or fallibly copy active rows into a `Vec<u8>`.
    pub fn to_contiguous_bytes(&self) -> Result<Cow<'a, [u8]>, At<BufferError>> {
        if let Some(bytes) = self.as_contiguous_bytes() {
            return Ok(Cow::Borrowed(bytes));
        }
        let rb = self.row_bytes();
        let total = rb
            .checked_mul(self.height as usize)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let mut out = Vec::new();
        out.try_reserve_exact(total)
            .map_err(|_| err_at(BufferError::AllocationFailed))?;
        for y in 0..self.height {
            out.extend_from_slice(self.row(y));
        }
        Ok(Cow::Owned(out))
    }

    /// Zero-copy rectangular subimage view retaining parent stride and `'a` lifetime.
    pub fn crop(&self, rect: PixelRect) -> Result<PixelView<'a>, At<BufferError>> {
        if rect.width == 0 || rect.height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if rect
            .x
            .checked_add(rect.width)
            .is_none_or(|e| e > self.width)
            || rect
                .y
                .checked_add(rect.height)
                .is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let bpp = self.spec.layout.bytes_per_pixel();
        let start = (rect.y as usize) * self.stride_bytes + (rect.x as usize) * bpp;
        let end = (rect.y as usize + rect.height as usize - 1) * self.stride_bytes
            + (rect.x as usize + rect.width as usize) * bpp;
        Ok(PixelView {
            data: &self.data[start..end],
            width: rect.width,
            height: rect.height,
            stride_bytes: self.stride_bytes,
            spec: self.spec.clone(),
            facts: self.facts,
            hdr: self.hdr.clone(),
            origin: self.origin,
            orientation: self.orientation,
        })
    }

    /// Zero-copy full-width horizontal strip view retaining `'a` lifetime.
    pub fn sub_rows(&self, start_y: u32, row_count: u32) -> Result<PixelView<'a>, At<BufferError>> {
        self.crop(PixelRect::new(0, start_y, self.width, row_count))
    }

    /// Fallibly copy active pixels into a freshly allocated, tightly packed [`PixelBuf`].
    pub fn to_tight_buf(&self) -> Result<PixelBuf, At<BufferError>> {
        let rb = self.row_bytes();
        let total = rb
            .checked_mul(self.height as usize)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let storage = PixelStorage::from_rows(
            self.spec.layout.sample,
            total,
            (0..self.height).map(|y| self.row(y)),
        )
        .map_err(|e| err_at(e.into()))?;
        Ok(PixelBuf {
            storage,
            offset: 0,
            width: self.width,
            height: self.height,
            stride: rb,
            spec: self.spec.as_format().clone(),
            facts: self.facts,
            hdr: self.hdr.clone(),
            origin: self.origin.cloned().and_then(box_origin),
            orientation: self.orientation,
        })
    }

    /// Measure exact [`FloatFacts`] across all active `f32` samples (returns `NONE` for non-`F32`).
    #[must_use]
    pub fn measure_float_facts(&self) -> FloatFacts {
        if self.spec.layout.sample != SampleType::F32 {
            return FloatFacts::NONE;
        }
        let mut acc = FloatFacts::UNIT_INTERVAL | FloatFacts::POSITIVE;
        for y in 0..self.height {
            let row = self.row(y);
            if let Ok(floats) = bytemuck::try_cast_slice::<u8, f32>(row) {
                acc = acc & FloatFacts::measure_f32_slice(floats);
            } else {
                return FloatFacts::NONE;
            }
        }
        acc
    }
}

// ---------------------------------------------------------------------------
// PixelViewMut (borrowed, mutable)
// ---------------------------------------------------------------------------

/// Zero-copy mutable view of an interleaved image or rectangular crop.
#[derive(Debug)]
pub struct PixelViewMut<'a> {
    data: &'a mut [u8],
    width: u32,
    height: u32,
    stride_bytes: usize,
    spec: FormatCow<'a>,
    hdr: Option<Arc<HdrMetadata>>,
    orientation: Orientation,
}

impl<'a> PixelViewMut<'a> {
    /// Borrow mutable strided bytes with checked dimensions, alignment, and length (**zero heap allocation**).
    ///
    /// Pass [`&formats::RGB8_SRGB`](crate::formats::RGB8_SRGB) (or any `crate::formats::*` preset) or a borrowed
    /// `&'a PixelFormat`. If a view genuinely needs to own a dynamically constructed [`PixelFormat`],
    /// use [`new_with_format_cow`](Self::new_with_format_cow) with [`FormatCow::from_owned`] or
    /// [`FormatCow::from_arc`].
    pub fn new(
        data: &'a mut [u8],
        width: u32,
        height: u32,
        stride_bytes: usize,
        format: &'a PixelFormat,
    ) -> Result<Self, At<BufferError>> {
        Self::new_with_format_cow(
            data,
            width,
            height,
            stride_bytes,
            FormatCow::Borrowed(format),
        )
    }

    /// Borrow mutable strided bytes with an explicit [`FormatCow<'a>`].
    pub fn new_with_format_cow(
        data: &'a mut [u8],
        width: u32,
        height: u32,
        stride_bytes: usize,
        format: FormatCow<'a>,
    ) -> Result<Self, At<BufferError>> {
        let (_rb, req) = validate_geometry(
            data.len(),
            data.as_ptr(),
            width,
            height,
            stride_bytes,
            format.as_format(),
        )?;
        Ok(Self {
            data: &mut data[..req],
            width,
            height,
            stride_bytes,
            spec: format,
            hdr: None,
            orientation: Orientation::Identity,
        })
    }

    /// Descriptive HDR container metadata attached to this mutable view.
    #[inline]
    #[must_use]
    pub fn hdr(&self) -> HdrMetadata {
        unbox_hdr(&self.hdr)
    }

    /// Attach descriptive HDR metadata to this mutable view.
    #[inline]
    #[must_use]
    pub fn with_hdr(mut self, hdr: HdrMetadata) -> Self {
        self.hdr = box_hdr(hdr);
        self
    }

    /// Pending EXIF display orientation attached to this mutable view.
    #[inline]
    #[must_use]
    pub const fn orientation(&self) -> Orientation {
        self.orientation
    }

    /// Display-oriented `(width, height)` after applying [`orientation`](Self::orientation).
    #[inline]
    #[must_use]
    pub const fn display_dimensions(&self) -> (u32, u32) {
        self.orientation.output_dimensions(self.width, self.height)
    }

    /// Replace the pending EXIF display orientation on this mutable view.
    #[inline]
    #[must_use]
    pub const fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Set the pending EXIF display orientation on this mutable view in place.
    #[inline]
    pub fn set_orientation(&mut self, orientation: Orientation) {
        self.orientation = orientation;
    }

    /// Compose an additional display transform (including reflections) onto this view's
    /// pending orientation (`self.orientation.then(delta)`) in O(1) without touching pixels.
    #[inline]
    #[must_use]
    pub const fn compose_orientation(mut self, delta: Orientation) -> Self {
        self.orientation = self.orientation.then(delta);
        self
    }

    /// Width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Row stride in bytes.
    #[inline]
    #[must_use]
    pub const fn stride_bytes(&self) -> usize {
        self.stride_bytes
    }

    /// Active bytes per row.
    #[inline]
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        (self.width as usize) * self.spec.as_format().layout.bytes_per_pixel()
    }

    /// Authoritative image specification.
    #[inline]
    #[must_use]
    pub fn format(&self) -> &PixelFormat {
        self.spec.as_format()
    }

    /// Mutable active bytes of row `y`.
    #[inline]
    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(
            y < self.height,
            "row {y} out of bounds (height: {})",
            self.height
        );
        let start = (y as usize) * self.stride_bytes;
        let rb = self.row_bytes();
        &mut self.data[start..start + rb]
    }

    /// Zero-copy mutable typed slice of row `y` (e.g. `&mut [rgb::Rgba<u8>]`).
    pub fn row_as_mut<P: bytemuck::Pod>(&mut self, y: u32) -> Result<&mut [P], At<BufferError>> {
        if y >= self.height {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let bytes = self.row_mut(y);
        let len = bytes.len();
        bytemuck::try_cast_slice_mut(bytes).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout).at_string(|| {
                alloc::format!(
                    "row byte length {len} is not compatible with element `{}` (size {})",
                    core::any::type_name::<P>(),
                    core::mem::size_of::<P>()
                )
            }),
        })
    }

    /// Mutable raw strided byte backing starting at `(0, 0)` (includes inter-row padding).
    #[inline]
    pub fn as_strided_bytes_mut(&mut self) -> &mut [u8] {
        self.data
    }

    /// Mutable contiguous bytes if rows are tightly packed.
    #[inline]
    pub fn as_contiguous_bytes_mut(&mut self) -> Option<&mut [u8]> {
        if self.height <= 1 || self.stride_bytes == self.row_bytes() {
            let total = self.row_bytes() * (self.height as usize);
            Some(&mut self.data[..total])
        } else {
            None
        }
    }

    /// Borrow as an immutable [`PixelView`].
    #[inline]
    #[must_use]
    pub fn as_view(&self) -> PixelView<'_> {
        PixelView {
            data: self.data,
            width: self.width,
            height: self.height,
            stride_bytes: self.stride_bytes,
            spec: self.spec.to_borrowed(),
            facts: SampleFacts::NONE,
            hdr: self.hdr.clone(),
            origin: None,
            orientation: self.orientation,
        }
    }

    /// Downgrade into an immutable [`PixelView<'a>`] keeping `'a`.
    #[inline]
    #[must_use]
    pub fn into_view(self) -> PixelView<'a> {
        PixelView {
            data: self.data,
            width: self.width,
            height: self.height,
            stride_bytes: self.stride_bytes,
            spec: self.spec,
            facts: SampleFacts::NONE,
            hdr: self.hdr,
            origin: None,
            orientation: self.orientation,
        }
    }

    /// Consume this mutable view and return the underlying strided backing slice tied to `'a`.
    #[inline]
    #[must_use]
    pub fn into_strided_bytes_mut(self) -> &'a mut [u8] {
        self.data
    }

    /// Mutable rectangular subimage view.
    pub fn crop_mut(&mut self, rect: PixelRect) -> Result<PixelViewMut<'_>, At<BufferError>> {
        if rect.width == 0 || rect.height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if rect
            .x
            .checked_add(rect.width)
            .is_none_or(|e| e > self.width)
            || rect
                .y
                .checked_add(rect.height)
                .is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let bpp = self.spec.layout.bytes_per_pixel();
        let start = (rect.y as usize) * self.stride_bytes + (rect.x as usize) * bpp;
        let end = (rect.y as usize + rect.height as usize - 1) * self.stride_bytes
            + (rect.x as usize + rect.width as usize) * bpp;
        Ok(PixelViewMut {
            data: &mut self.data[start..end],
            width: rect.width,
            height: rect.height,
            stride_bytes: self.stride_bytes,
            spec: self.spec.to_borrowed(),
            hdr: self.hdr.clone(),
            orientation: self.orientation,
        })
    }

    /// Mutable full-width strip view.
    pub fn sub_rows_mut(
        &mut self,
        start_y: u32,
        row_count: u32,
    ) -> Result<PixelViewMut<'_>, At<BufferError>> {
        self.crop_mut(PixelRect::new(0, start_y, self.width, row_count))
    }

    /// Consume this mutable view and narrow it to a rectangular crop retaining `'a`.
    pub fn into_crop(self, rect: PixelRect) -> Result<PixelViewMut<'a>, At<BufferError>> {
        if rect.width == 0 || rect.height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if rect
            .x
            .checked_add(rect.width)
            .is_none_or(|e| e > self.width)
            || rect
                .y
                .checked_add(rect.height)
                .is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let bpp = self.spec.layout.bytes_per_pixel();
        let start = (rect.y as usize) * self.stride_bytes + (rect.x as usize) * bpp;
        let end = (rect.y as usize + rect.height as usize - 1) * self.stride_bytes
            + (rect.x as usize + rect.width as usize) * bpp;
        Ok(PixelViewMut {
            data: &mut self.data[start..end],
            width: rect.width,
            height: rect.height,
            stride_bytes: self.stride_bytes,
            spec: self.spec,
            hdr: self.hdr,
            orientation: self.orientation,
        })
    }

    /// Consume this mutable view and narrow it to a full-width strip retaining `'a`.
    pub fn into_sub_rows(
        self,
        start_y: u32,
        row_count: u32,
    ) -> Result<PixelViewMut<'a>, At<BufferError>> {
        let width = self.width;
        self.into_crop(PixelRect::new(0, start_y, width, row_count))
    }
}

// ---------------------------------------------------------------------------
// PixelBuf (unparameterized owned buffer, fallible by default)
// ---------------------------------------------------------------------------

/// Current buffer geometry and metadata passed to the closure in [`PixelBuf::transform_in_place`].
#[derive(Clone, Copy, Debug)]
pub struct InPlaceTransformInput<'a> {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in rows.
    pub height: u32,
    /// Current row stride in bytes.
    pub stride_bytes: usize,
    /// Current pixel format (layout + color space).
    pub format: &'a PixelFormat,
    /// Current container color provenance.
    pub origin: &'a ColorOrigin,
}

/// Replacement stride and metadata returned by the closure in [`PixelBuf::transform_in_place`].
#[derive(Clone, Debug)]
pub struct InPlaceTransformOutput {
    /// New row stride in bytes (must fit within existing storage and satisfy alignment).
    pub stride_bytes: usize,
    /// New pixel format (layout + color space).
    pub format: PixelFormat,
    /// New container color provenance.
    pub origin: ColorOrigin,
    /// New proven sample facts for the transformed pixels.
    pub facts: SampleFacts,
}

/// Decomposed parts returned by [`PixelBuf::into_parts`].
#[derive(Debug)]
#[non_exhaustive]
pub struct PixelBufParts {
    /// Underlying alignment-exact storage.
    pub storage: PixelStorage,
    /// Byte offset of `(0, 0)` within `storage.as_bytes()`.
    pub byte_offset: usize,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in rows.
    pub height: u32,
    /// Byte stride between rows.
    pub stride_bytes: usize,
    /// Pixel format (layout + color space).
    pub format: PixelFormat,
    /// Cached sample facts.
    pub facts: SampleFacts,
    /// Descriptive HDR metadata.
    pub hdr: HdrMetadata,
    /// Container provenance.
    pub origin: ColorOrigin,
    /// Pending EXIF display orientation.
    pub orientation: Orientation,
}

/// Owned, unparameterized, alignment-exact pixel image buffer.
///
/// Every allocating constructor is fallible and returns `Result<Self, At<BufferError>>`.
/// Mutable access automatically invalidates cached [`SampleFacts`].
#[derive(Debug)]
pub struct PixelBuf {
    storage: PixelStorage,
    offset: usize,
    width: u32,
    height: u32,
    stride: usize,
    spec: PixelFormat,
    facts: SampleFacts,
    hdr: Option<Arc<HdrMetadata>>,
    origin: Option<Arc<ColorOrigin>>,
    orientation: Orientation,
}

impl Clone for PixelBuf {
    /// Clone this buffer, preserving alignment, metadata, facts, origin, and orientation.
    ///
    /// # Panics
    ///
    /// Panics on allocation failure when repacking misaligned backing storage; use
    /// [`PixelBuf::try_clone`] for fallible cloning.
    fn clone(&self) -> Self {
        let align = self.spec.layout.min_alignment();
        let cloned_storage = self.storage.clone();
        // Guard against cloning an accidentally-aligned `Vec<u8>` backing on a wider sample layout:
        // if the cloned byte Vec lands misaligned, re-pack into natively aligned sample storage.
        let (storage, offset, stride) = if align > 1
            && ((cloned_storage.as_bytes().as_ptr() as usize + self.offset) % align != 0
                || (self.height > 1 && self.stride % align != 0))
        {
            let rb = self.row_bytes();
            let total = rb * (self.height as usize);
            let view = self.as_view();
            let native = PixelStorage::from_rows(
                self.spec.layout.sample,
                total,
                (0..self.height).map(|y| view.row(y)),
            )
            .expect("cloning PixelBuf into native-aligned storage failed");
            (native, 0, rb)
        } else {
            (cloned_storage, self.offset, self.stride)
        };
        Self {
            storage,
            offset,
            width: self.width,
            height: self.height,
            stride,
            spec: self.spec.clone(),
            facts: self.facts,
            hdr: self.hdr.clone(),
            origin: self.origin.clone(),
            orientation: self.orientation,
        }
    }
}

impl PixelBuf {
    /// Fallibly clone this buffer, preserving alignment, metadata, facts, origin, and orientation.
    pub fn try_clone(&self) -> Result<Self, At<BufferError>> {
        let align = self.spec.layout.min_alignment();
        let cloned_storage = self.storage.try_clone().map_err(|e| err_at(e.into()))?;
        let (storage, offset, stride) = if align > 1
            && ((cloned_storage.as_bytes().as_ptr() as usize + self.offset) % align != 0
                || (self.height > 1 && self.stride % align != 0))
        {
            let rb = self.row_bytes();
            let total = rb
                .checked_mul(self.height as usize)
                .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
            let view = self.as_view();
            let native = PixelStorage::from_rows(
                self.spec.layout.sample,
                total,
                (0..self.height).map(|y| view.row(y)),
            )
            .map_err(|e| err_at(e.into()))?;
            (native, 0, rb)
        } else {
            (cloned_storage, self.offset, self.stride)
        };
        Ok(Self {
            storage,
            offset,
            width: self.width,
            height: self.height,
            stride,
            spec: self.spec.clone(),
            facts: self.facts,
            hdr: self.hdr.clone(),
            origin: self.origin.clone(),
            orientation: self.orientation,
        })
    }

    /// Fallibly allocate a tightly packed black-filled buffer backed by native sample storage.
    ///
    /// When `spec.color.alpha() == AlphaMode::Opaque`, the alpha channel is initialized to
    /// canonical full opacity (`255`, `65535`, `1.0`, or packed max code) so stored pixels
    /// honor the [`AlphaMode::Opaque`] invariant immediately.
    ///
    /// For solid opaque white initialization, use [`new_white`](Self::new_white) or
    /// [`new_with_fill`](Self::new_with_fill). To skip the alpha-fill pass when you will
    /// overwrite every pixel, allocate via [`new_zeroed`](Self::new_zeroed) with
    /// [`AlphaMode::Straight`], fill all pixels, and retag zero-copy with
    /// [`into_opaque_verified`](Self::into_opaque_verified) or [`assume_opaque`](Self::assume_opaque).
    pub fn new(
        width: u32,
        height: u32,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, At<BufferError>> {
        Self::new_with_fill(width, height, format, RowAlign::Tight, BufFill::Black)
    }

    /// Fallibly allocate a tightly packed white-filled buffer (reference white on color channels
    /// and canonical maximum on any alpha slot).
    pub fn new_white(
        width: u32,
        height: u32,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, At<BufferError>> {
        Self::new_with_fill(width, height, format, RowAlign::Tight, BufFill::White)
    }

    /// Fallibly allocate a black-filled buffer with explicit row alignment
    /// (`Tight`, `Cache64`, `PixelCache64`, `Samples(n)`).
    ///
    /// For `Cache64` and `PixelCache64`, storage is allocated as 64-byte aligned
    /// [`AlignedSamples`] so both the first pixel and every
    /// row start land on a 64-byte boundary. When `format.color.alpha() == AlphaMode::Opaque`,
    /// alpha samples are initialized to full opacity.
    pub fn new_aligned(
        width: u32,
        height: u32,
        format: impl Into<PixelFormat>,
        align: RowAlign,
    ) -> Result<Self, At<BufferError>> {
        Self::new_with_fill(width, height, format, align, BufFill::Black)
    }

    /// Fallibly allocate a buffer with explicit row alignment and initial [`BufFill`]
    /// (`Black` or `White`).
    pub fn new_with_fill(
        width: u32,
        height: u32,
        format: impl Into<PixelFormat>,
        align: RowAlign,
        fill: BufFill,
    ) -> Result<Self, At<BufferError>> {
        let spec = format.into();
        spec.validate()
            .map_err(|e| whereat::at!(BufferError::InvalidFormat(*e.error())))?;
        if width == 0 || height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let stride = spec
            .layout
            .aligned_stride(width, align)
            .map_err(|e| err_at(e.into()))?;
        let row_bytes = (width as usize)
            .checked_mul(spec.layout.bytes_per_pixel())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let total = match align {
            RowAlign::Cache64 | RowAlign::PixelCache64 => stride
                .checked_mul(height as usize)
                .ok_or_else(|| err_at(BufferError::InvalidDimensions))?,
            _ => required_bytes(height, stride, row_bytes).map_err(err_at)?,
        };
        let mut storage = match align {
            RowAlign::Cache64 | RowAlign::PixelCache64 => {
                PixelStorage::zeroed_aligned64_bytes(total)
            }
            _ => PixelStorage::zeroed_bytes(spec.layout.sample, total),
        }
        .map_err(|e| err_at(e.into()))?;
        let facts = match fill {
            BufFill::Black => {
                if spec.color.alpha() == AlphaMode::Opaque {
                    fill_opaque_alpha(storage.as_bytes_mut(), width, height, stride, &spec);
                }
                SampleFacts::for_zeroed_format(&spec)
            }
            BufFill::White => {
                fill_white_pixels(storage.as_bytes_mut(), width, height, stride, &spec);
                SampleFacts::for_white_format(&spec)
            }
        };
        Ok(Self {
            storage,
            offset: 0,
            width,
            height,
            stride,
            spec,
            facts,
            hdr: None,
            origin: None,
            orientation: Orientation::Identity,
        })
    }

    /// Fallibly allocate raw zero-filled OS pages with **no alpha-fill pass** for maximum
    /// decoder/renderer write throughput.
    ///
    /// Refuses [`AlphaMode::Opaque`] on layouts with an alpha slot with
    /// [`BufferError::UnverifiedOpaqueAlpha`] so an opaque-tagged buffer can never silently
    /// contain zero alpha. Allocate with [`AlphaMode::Straight`], write every
    /// pixel including full alpha, then retag zero-copy via
    /// [`into_opaque_verified`](Self::into_opaque_verified),
    /// [`into_opaque_filled`](Self::into_opaque_filled), or [`assume_opaque`](Self::assume_opaque).
    pub fn new_zeroed(
        width: u32,
        height: u32,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, At<BufferError>> {
        Self::new_zeroed_aligned(width, height, format, RowAlign::Tight)
    }

    /// Fallibly allocate raw zero-filled aligned OS pages with **no alpha-fill pass**,
    /// refusing [`AlphaMode::Opaque`] on layouts with an alpha slot.
    pub fn new_zeroed_aligned(
        width: u32,
        height: u32,
        format: impl Into<PixelFormat>,
        align: RowAlign,
    ) -> Result<Self, At<BufferError>> {
        let spec = format.into();
        spec.validate()
            .map_err(|e| whereat::at!(BufferError::InvalidFormat(*e.error())))?;
        if width == 0 || height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if spec.color.alpha() == AlphaMode::Opaque && spec.layout.order.has_alpha_slot() {
            return Err(err_at(BufferError::UnverifiedOpaqueAlpha));
        }
        let stride = spec
            .layout
            .aligned_stride(width, align)
            .map_err(|e| err_at(e.into()))?;
        let row_bytes = (width as usize)
            .checked_mul(spec.layout.bytes_per_pixel())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let total = match align {
            RowAlign::Cache64 | RowAlign::PixelCache64 => stride
                .checked_mul(height as usize)
                .ok_or_else(|| err_at(BufferError::InvalidDimensions))?,
            _ => required_bytes(height, stride, row_bytes).map_err(err_at)?,
        };
        let storage = match align {
            RowAlign::Cache64 | RowAlign::PixelCache64 => {
                PixelStorage::zeroed_aligned64_bytes(total)
            }
            _ => PixelStorage::zeroed_bytes(spec.layout.sample, total),
        }
        .map_err(|e| err_at(e.into()))?;
        let facts = SampleFacts::for_zeroed_format(&spec);
        Ok(Self {
            storage,
            offset: 0,
            width,
            height,
            stride,
            spec,
            facts,
            hdr: None,
            origin: None,
            orientation: Orientation::Identity,
        })
    }

    /// Fallibly allocate tight native storage validated upfront for a zero-copy [`InteropTarget`].
    pub fn new_for_interop(
        width: u32,
        height: u32,
        format: impl Into<PixelFormat>,
        target: InteropTarget,
    ) -> Result<Self, At<BufferError>> {
        let spec = format.into();
        if !spec.supports_interop(target) {
            return Err(err_at(BufferError::InteropUnsupported));
        }
        Self::new(width, height, spec)
    }

    /// Adopt tightly packed bytes (`Vec<u8>`), handling odd-address allocators safely.
    ///
    /// - For 1-byte sample formats (`U8`, `I8`), adopts `bytes` **zero-copy** at any address.
    /// - For wider samples (`U16`, `F16`, `F32`, `F64`), adopts zero-copy when the pointer
    ///   happens to be sample-aligned, and fallibly copies into properly aligned native
    ///   [`PixelStorage`] when the allocator returned a misaligned address (e.g. odd address).
    ///
    /// To recover `bytes` intact on error, use [`from_bytes_recover`](Self::from_bytes_recover).
    pub fn from_bytes(
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, At<BufferError>> {
        Self::from_bytes_recover(bytes, width, height, format).map_err(|(e, _)| e)
    }

    /// Adopt tightly packed bytes (`Vec<u8>`), returning `Err((err, bytes))` with the original
    /// `Vec<u8>` untouched on validation or allocation failure.
    #[allow(clippy::result_large_err)]
    pub fn from_bytes_recover(
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, (At<BufferError>, Vec<u8>)> {
        let spec = format.into();
        if let Err(e) = spec.validate() {
            return Err((whereat::at!(BufferError::InvalidFormat(*e.error())), bytes));
        }
        if width == 0 || height == 0 {
            return Err((err_at(BufferError::InvalidDimensions), bytes));
        }
        let bpp = spec.layout.bytes_per_pixel();
        let Some(row_bytes) = (width as usize).checked_mul(bpp) else {
            return Err((err_at(BufferError::InvalidDimensions), bytes));
        };
        let req = match required_bytes(height, row_bytes, row_bytes) {
            Ok(r) => r,
            Err(e) => return Err((err_at(e), bytes)),
        };
        if bytes.len() < req {
            return Err((err_at(BufferError::InsufficientData), bytes));
        }
        let align = spec.layout.min_alignment();
        let storage = if (bytes.as_ptr() as usize) % align == 0 {
            PixelStorage::U8(bytes)
        } else {
            // Odd-allocator-safe fallible copy into natively aligned storage (first byte = first pixel).
            match PixelStorage::from_rows(spec.layout.sample, req, core::iter::once(&bytes[..req]))
            {
                Ok(s) => s,
                Err(e) => return Err((err_at(e.into()), bytes)),
            }
        };
        Ok(Self {
            storage,
            offset: 0,
            width,
            height,
            stride: row_bytes,
            spec,
            facts: SampleFacts::NONE,
            hdr: None,
            origin: None,
            orientation: Orientation::Identity,
        })
    }

    /// Adopt a tightly packed typed `Vec<P>` (pixels like `Vec<rgb::Rgb<u16>>`, arrays `[u8; 3]`,
    /// or raw samples like `Vec<f32>`).
    ///
    /// Validates that `P` is structurally compatible with `format.layout` (preventing silent
    /// reinterpretation such as `Vec<i32>` as `RGBF32`). Note that untyped integer literals
    /// (`vec![255, 0, 0]`) default to `Vec<i32>` in Rust and will be rejected for `U8` layouts;
    /// annotate the first element (e.g. `vec![255u8, 0, 0]`) or use [`from_bytes`](Self::from_bytes).
    /// For intentional raw byte reinterpretation, use [`from_pod_bytes_unchecked`](Self::from_pod_bytes_unchecked).
    ///
    /// **Zero-copy** whenever `align_of::<P>() == format.layout.min_alignment()` and
    /// `bytemuck::allocation::try_cast_vec` succeeds (preserving exact deallocation alignment!).
    /// Otherwise fallibly copies into native-aligned [`PixelStorage`].
    ///
    /// To recover `pixels` intact on error, use [`from_typed_recover`](Self::from_typed_recover).
    pub fn from_typed<P: TypedElement>(
        pixels: Vec<P>,
        width: u32,
        height: u32,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, At<BufferError>> {
        Self::from_typed_recover(pixels, width, height, format).map_err(|(e, _)| e)
    }

    /// Adopt a tightly packed typed `Vec<P>`, returning `Err((err, pixels))` with the original
    /// `Vec<P>` untouched on validation or allocation failure.
    #[allow(clippy::result_large_err)]
    pub fn from_typed_recover<P: TypedElement>(
        pixels: Vec<P>,
        width: u32,
        height: u32,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, (At<BufferError>, Vec<P>)> {
        let spec = format.into();
        if !P::matches_layout(spec.layout) {
            return Err((
                err_at(BufferError::IncompatibleLayout).at_string(|| {
                    alloc::format!(
                        "element `{}` does not match {:?} layout (annotate integer literals e.g. `0u8`, or use `from_pod_bytes_unchecked`)",
                        core::any::type_name::<P>(),
                        spec.layout
                    )
                }),
                pixels,
            ));
        }
        if let Err(e) = spec.validate() {
            return Err((whereat::at!(BufferError::InvalidFormat(*e.error())), pixels));
        }
        if width == 0 || height == 0 {
            return Err((err_at(BufferError::InvalidDimensions), pixels));
        }
        let bpp = spec.layout.bytes_per_pixel();
        let Some(row_bytes) = (width as usize).checked_mul(bpp) else {
            return Err((err_at(BufferError::InvalidDimensions), pixels));
        };
        let req = match required_bytes(height, row_bytes, row_bytes) {
            Ok(r) => r,
            Err(e) => return Err((err_at(e), pixels)),
        };
        let Some(total_bytes) = pixels.len().checked_mul(core::mem::size_of::<P>()) else {
            return Err((err_at(BufferError::InvalidDimensions), pixels));
        };
        if total_bytes < req {
            return Err((err_at(BufferError::InsufficientData), pixels));
        }
        let storage = match PixelStorage::adopt_vec(pixels, spec.layout.sample) {
            Ok(s) => s,
            Err(pixels) => {
                let copied = {
                    let raw = bytemuck::cast_slice::<P, u8>(&pixels);
                    PixelStorage::from_rows(spec.layout.sample, req, core::iter::once(&raw[..req]))
                };
                match copied {
                    Ok(s) => s,
                    Err(e) => return Err((err_at(e.into()), pixels)),
                }
            }
        };
        Ok(Self {
            storage,
            offset: 0,
            width,
            height,
            stride: row_bytes,
            spec,
            facts: SampleFacts::NONE,
            hdr: None,
            origin: None,
            orientation: Orientation::Identity,
        })
    }

    /// Adopt a tightly packed scalar sample `Vec<S>` (`Vec<u8>`, `Vec<u16>`, `Vec<f32>`, etc.).
    pub fn from_samples<S: StorageSample>(
        samples: Vec<S>,
        width: u32,
        height: u32,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, At<BufferError>> {
        let spec = format.into();
        if spec.layout.sample != S::SAMPLE_TYPE {
            return Err(err_at(BufferError::IncompatibleLayout).at_string(|| {
                alloc::format!(
                    "sample `{}` ({:?}) does not match {:?} layout",
                    core::any::type_name::<S>(),
                    S::SAMPLE_TYPE,
                    spec.layout
                )
            }));
        }
        Self::from_pod_bytes_unchecked(samples, width, height, spec)
    }

    /// Adopt a tightly packed `rgb` pixel `Vec<P>` (`Vec<rgb::Rgb<u8>>`, etc.) with an explicit
    /// [`ColorSpace`] (deriving [`SampleLayout`] from `P::LAYOUT`).
    #[cfg(feature = "rgb")]
    pub fn from_pixels_with_color<P: crate::interop::PixelType>(
        pixels: Vec<P>,
        width: u32,
        height: u32,
        color: crate::ColorSpace,
    ) -> Result<Self, At<BufferError>> {
        Self::from_pod_bytes_unchecked(pixels, width, height, PixelFormat::new(P::LAYOUT, color))
    }

    /// Explicitly reinterpret any tightly packed `Vec<P: bytemuck::Pod>` in native endianness
    /// as raw pixel bytes matching `format` without checking element-to-layout compatibility.
    pub fn from_pod_bytes_unchecked<P: bytemuck::Pod>(
        pixels: Vec<P>,
        width: u32,
        height: u32,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, At<BufferError>> {
        let spec = format.into();
        spec.validate()
            .map_err(|e| whereat::at!(BufferError::InvalidFormat(*e.error())))?;
        if width == 0 || height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let bpp = spec.layout.bytes_per_pixel();
        let row_bytes = (width as usize)
            .checked_mul(bpp)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let req = required_bytes(height, row_bytes, row_bytes).map_err(err_at)?;
        let total_bytes = pixels
            .len()
            .checked_mul(core::mem::size_of::<P>())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        if total_bytes < req {
            return Err(err_at(BufferError::InsufficientData));
        }
        let storage = match PixelStorage::adopt_vec(pixels, spec.layout.sample) {
            Ok(s) => s,
            Err(pixels) => {
                let raw = bytemuck::cast_slice::<P, u8>(&pixels);
                PixelStorage::from_rows(spec.layout.sample, req, core::iter::once(&raw[..req]))
                    .map_err(|e| err_at(e.into()))?
            }
        };
        Ok(Self {
            storage,
            offset: 0,
            width,
            height,
            stride: row_bytes,
            spec,
            facts: SampleFacts::NONE,
            hdr: None,
            origin: None,
            orientation: Orientation::Identity,
        })
    }

    /// Checked construction from owned [`PixelStorage`], byte offset, dimensions, byte stride, and
    /// format with fresh default metadata (`facts = NONE`, `hdr = empty`, `origin = assumed`,
    /// `orientation = Identity`).
    ///
    /// To restore full decomposed metadata from [`PixelBuf::into_parts`], use
    /// [`from_parts`](Self::from_parts) or [`from_parts_recover`](Self::from_parts_recover).
    pub fn from_storage_parts(
        storage: PixelStorage,
        byte_offset: usize,
        width: u32,
        height: u32,
        stride_bytes: usize,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, At<BufferError>> {
        let spec = format.into();
        let bytes = storage.as_bytes();
        let slice = bytes
            .get(byte_offset..)
            .ok_or_else(|| err_at(BufferError::InsufficientData))?;
        validate_geometry(
            slice.len(),
            slice.as_ptr(),
            width,
            height,
            stride_bytes,
            &spec,
        )?;
        Ok(Self {
            storage,
            offset: byte_offset,
            width,
            height,
            stride: stride_bytes,
            spec,
            facts: SampleFacts::NONE,
            hdr: None,
            origin: None,
            orientation: Orientation::Identity,
        })
    }

    /// Checked reconstruction from a [`PixelBufParts`] struct (preserving `hdr`, `origin`,
    /// `orientation`, and `parts.facts`).
    ///
    /// On validation failure, `parts` is dropped; use
    /// [`from_parts_recover`](Self::from_parts_recover) to recover the allocation intact.
    pub fn from_parts(parts: PixelBufParts) -> Result<Self, At<BufferError>> {
        Self::from_parts_recover(parts).map_err(|(err, _)| err)
    }

    /// Checked reconstruction from [`PixelBufParts`], returning `Err((err, parts))` with the
    /// original allocation and metadata intact on validation failure.
    #[allow(clippy::result_large_err)]
    pub fn from_parts_recover(
        parts: PixelBufParts,
    ) -> Result<Self, (At<BufferError>, PixelBufParts)> {
        let bytes = parts.storage.as_bytes();
        let Some(slice) = bytes.get(parts.byte_offset..) else {
            return Err((err_at(BufferError::InsufficientData), parts));
        };
        if let Err(e) = validate_geometry(
            slice.len(),
            slice.as_ptr(),
            parts.width,
            parts.height,
            parts.stride_bytes,
            &parts.format,
        ) {
            return Err((e, parts));
        }
        Ok(Self {
            storage: parts.storage,
            offset: parts.byte_offset,
            width: parts.width,
            height: parts.height,
            stride: parts.stride_bytes,
            spec: parts.format,
            facts: parts.facts,
            hdr: box_hdr(parts.hdr),
            origin: box_origin(parts.origin),
            orientation: parts.orientation,
        })
    }

    /// Adopt an immutable external owned backing (`Arc<[u8]>`, `Arc<[u16]>`, `&'static [u8]`,
    /// memory-mapped file wrapper, shared buffer pool handle, etc.) **zero-copy**.
    ///
    /// Validates geometry and sample alignment BEFORE boxing `storage`; on error, returns
    /// `Err((err, storage))` with zero heap allocation and `storage` completely untouched.
    ///
    /// Read access (`as_view`, `row`, `PixelSource`) is strictly zero-copy. Requesting
    /// mutable access (`as_view_mut`, `try_as_view_mut`, `make_mut`) transparently
    /// performs copy-on-write into owned aligned storage without mutating `storage`.
    /// Call [`into_external`](Self::into_external) to recover `S` zero-copy.
    #[allow(clippy::result_large_err)]
    pub fn from_external<S: ByteStorage + Send + Sync + 'static>(
        storage: S,
        width: u32,
        height: u32,
        stride_bytes: usize,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, (At<BufferError>, S)> {
        Self::from_external_with_byte_offset(storage, 0, width, height, stride_bytes, format)
    }

    /// Adopt an immutable external owned backing starting at `byte_offset` bytes **zero-copy**.
    ///
    /// Returns `Err((err, storage))` untouched on validation failure with zero heap allocation.
    #[allow(clippy::result_large_err)]
    pub fn from_external_with_byte_offset<S: ByteStorage + Send + Sync + 'static>(
        storage: S,
        byte_offset: usize,
        width: u32,
        height: u32,
        stride_bytes: usize,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, (At<BufferError>, S)> {
        let spec = format.into();
        let check = {
            let bytes = storage.as_bytes();
            match bytes.get(byte_offset..) {
                None => Err(err_at(BufferError::InsufficientData)),
                Some(slice) => validate_geometry(
                    slice.len(),
                    slice.as_ptr(),
                    width,
                    height,
                    stride_bytes,
                    &spec,
                ),
            }
        };
        if let Err(e) = check {
            return Err((e, storage));
        }
        Ok(Self {
            storage: PixelStorage::from_external(storage),
            offset: byte_offset,
            width,
            height,
            stride: stride_bytes,
            spec,
            facts: SampleFacts::NONE,
            hdr: None,
            origin: None,
            orientation: Orientation::Identity,
        })
    }

    /// Adopt a mutable external owned backing (`Box<[T]>`, custom-allocator wrapper, mutable
    /// pool lease, FFI writable buffer, etc.) **zero-copy** for both reads and in-place writes.
    ///
    /// Validates geometry and sample alignment BEFORE boxing `storage`; on error, returns
    /// `Err((err, storage))` with zero heap allocation and `storage` completely untouched.
    /// Call [`into_external_mut`](Self::into_external_mut) to recover `S` zero-copy.
    #[allow(clippy::result_large_err)]
    pub fn from_external_mut<S: ByteStorageMut + Send + Sync + 'static>(
        storage: S,
        width: u32,
        height: u32,
        stride_bytes: usize,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, (At<BufferError>, S)> {
        Self::from_external_mut_with_byte_offset(storage, 0, width, height, stride_bytes, format)
    }

    /// Adopt a mutable external owned backing starting at `byte_offset` bytes **zero-copy**.
    ///
    /// Returns `Err((err, storage))` untouched on validation failure with zero heap allocation.
    #[allow(clippy::result_large_err)]
    pub fn from_external_mut_with_byte_offset<S: ByteStorageMut + Send + Sync + 'static>(
        storage: S,
        byte_offset: usize,
        width: u32,
        height: u32,
        stride_bytes: usize,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, (At<BufferError>, S)> {
        let spec = format.into();
        let check = {
            let bytes = storage.as_bytes();
            match bytes.get(byte_offset..) {
                None => Err(err_at(BufferError::InsufficientData)),
                Some(slice) => validate_geometry(
                    slice.len(),
                    slice.as_ptr(),
                    width,
                    height,
                    stride_bytes,
                    &spec,
                ),
            }
        };
        if let Err(e) = check {
            return Err((e, storage));
        }
        Ok(Self {
            storage: PixelStorage::from_external_mut(storage),
            offset: byte_offset,
            width,
            height,
            stride: stride_bytes,
            spec,
            facts: SampleFacts::NONE,
            hdr: None,
            origin: None,
            orientation: Orientation::Identity,
        })
    }

    /// Whether this buffer is backed by an external owner (`from_external` or `from_external_mut`).
    #[inline]
    #[must_use]
    pub const fn is_external(&self) -> bool {
        self.storage.is_external()
    }

    /// Whether mutable access modifies the current backing in place without copy-on-write.
    #[inline]
    #[must_use]
    pub const fn is_mutable_in_place(&self) -> bool {
        self.storage.is_mutable_in_place()
    }

    /// Borrow the external owner if backed by an external owner of concrete type `S`.
    #[inline]
    #[must_use]
    pub fn external_ref<S: 'static>(&self) -> Option<&S> {
        self.storage.external_ref::<S>()
    }

    /// Mutably borrow the external owner if backed by `from_external_mut` of concrete type `S`,
    /// automatically clearing cached [`SampleFacts`].
    #[inline]
    pub fn external_mut<S: 'static>(&mut self) -> Option<&mut S> {
        if self.storage.external_ref::<S>().is_some() {
            self.facts = SampleFacts::NONE;
        }
        self.storage.external_mut::<S>()
    }

    /// Recover an immutable external owner `S` zero-copy; returns `Err(self)` untouched otherwise.
    #[allow(clippy::result_large_err)]
    pub fn into_external<S: 'static>(mut self) -> Result<S, Self> {
        match self.storage.into_external::<S>() {
            Ok(owner) => Ok(owner),
            Err(storage) => {
                self.storage = storage;
                Err(self)
            }
        }
    }

    /// Recover a mutable external owner `S` zero-copy; returns `Err(self)` untouched otherwise.
    #[allow(clippy::result_large_err)]
    pub fn into_external_mut<S: 'static>(mut self) -> Result<S, Self> {
        match self.storage.into_external_mut::<S>() {
            Ok(owner) => Ok(owner),
            Err(storage) => {
                self.storage = storage;
                Err(self)
            }
        }
    }

    /// Ensure backing storage is mutable in place, fallibly promoting immutable external
    /// storage ([`PixelStorage::External`]) via copy-on-write into properly aligned owned storage.
    ///
    /// No-op (zero copy, zero allocation) for native `Vec`, `Aligned64`, and `ExternalMut` buffers.
    #[doc(alias = "cow")]
    #[doc(alias = "copy_on_write")]
    pub fn make_mut(&mut self) -> Result<(), At<BufferError>> {
        if !matches!(self.storage, PixelStorage::External(_)) {
            return Ok(());
        }
        let align = self.spec.layout.min_alignment();
        let rb = self.row_bytes();
        let req = required_bytes(self.height, self.stride, rb).map_err(err_at)?;
        if self.offset == 0 && (self.height <= 1 || self.stride % align == 0) {
            let mut aligned = AlignedSamples::zeroed_bytes(req).map_err(|e| err_at(e.into()))?;
            aligned
                .as_bytes_mut()
                .copy_from_slice(&self.storage.as_bytes()[..req]);
            self.storage = PixelStorage::Aligned64(aligned);
        } else {
            let total = rb
                .checked_mul(self.height as usize)
                .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
            let native = {
                let view = self.as_view();
                PixelStorage::from_rows(
                    self.spec.layout.sample,
                    total,
                    (0..self.height).map(|y| view.row(y)),
                )
                .map_err(|e| err_at(e.into()))?
            };
            self.storage = native;
            self.offset = 0;
            self.stride = rb;
        }
        Ok(())
    }

    /// Width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Stride in bytes.
    #[inline]
    #[must_use]
    pub const fn stride_bytes(&self) -> usize {
        self.stride
    }

    /// Active bytes per row.
    #[inline]
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        (self.width as usize) * self.spec.layout.bytes_per_pixel()
    }

    /// Leading byte offset inside backing storage.
    #[inline]
    #[must_use]
    pub const fn byte_offset(&self) -> usize {
        self.offset
    }

    /// Authoritative image specification.
    #[inline]
    #[must_use]
    pub fn format(&self) -> &PixelFormat {
        &self.spec
    }

    /// Relabel specification metadata without modifying pixel bytes, preserving numeric sample
    /// type, bit packing, and `bytes_per_pixel` (e.g. relabeling color space or channel order).
    ///
    /// Rejects changes to `sample` type or `packing` (use
    /// [`reinterpret_bytes`](Self::reinterpret_bytes) for deliberate raw-byte reinterpretation
    /// across sample types). Clears cached [`SampleFacts`] and resets [`ColorOrigin`] to
    /// [`ColorOrigin::assumed`] whenever `format.color` changes.
    pub fn relabel_format(
        mut self,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, At<BufferError>> {
        let spec = format.into();
        if spec.layout.sample != self.spec.layout.sample
            || spec.layout.packing != self.spec.layout.packing
            || spec.layout.bytes_per_pixel() != self.spec.layout.bytes_per_pixel()
        {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let bytes = &self.storage.as_bytes()[self.offset..];
        validate_geometry(
            bytes.len(),
            bytes.as_ptr(),
            self.width,
            self.height,
            self.stride,
            &spec,
        )?;
        let color_changed = spec.color != self.spec.color;
        self.spec = spec;
        self.facts = SampleFacts::NONE;
        if color_changed {
            self.origin = None;
        }
        Ok(self)
    }

    /// Relabel only the color space metadata without modifying pixel bytes, keeping the exact
    /// [`SampleLayout`].
    ///
    /// Does **not** convert pixel values (use `zenpixels-convert` to convert colors). Resets
    /// [`ColorOrigin`] to [`ColorOrigin::assumed`] when `color` changes and clears cached
    /// [`SampleFacts`] when signal range, alpha mode, or color model changes.
    pub fn relabel_color(mut self, color: crate::ColorSpace) -> Result<Self, At<BufferError>> {
        let new_spec = PixelFormat::new(self.spec.layout, color);
        new_spec
            .validate()
            .map_err(|e| whereat::at!(BufferError::InvalidFormat(*e.error())))?;
        let color_changed = new_spec.color != self.spec.color;
        let same_facts_domain = new_spec.color.range() == self.spec.color.range()
            && new_spec.color.alpha() == self.spec.color.alpha()
            && core::mem::discriminant(&new_spec.color)
                == core::mem::discriminant(&self.spec.color);
        self.spec = new_spec;
        if !same_facts_domain {
            self.facts = SampleFacts::NONE;
        }
        if color_changed {
            self.origin = None;
        }
        Ok(self)
    }

    /// Deliberately reinterpret the raw pixel bytes under a new [`PixelFormat`] with identical
    /// `bytes_per_pixel` and satisfied sample alignment (permits changing `sample` type and
    /// `packing`, e.g. `U8` ↔ `I8` or `U16` ↔ `F16`).
    ///
    /// Does **not** convert pixel values. Always clears cached [`SampleFacts`], [`ColorOrigin`],
    /// and [`HdrMetadata`].
    pub fn reinterpret_bytes(
        mut self,
        format: impl Into<PixelFormat>,
    ) -> Result<Self, At<BufferError>> {
        let spec = format.into();
        if spec.layout.bytes_per_pixel() != self.spec.layout.bytes_per_pixel() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let bytes = &self.storage.as_bytes()[self.offset..];
        validate_geometry(
            bytes.len(),
            bytes.as_ptr(),
            self.width,
            self.height,
            self.stride,
            &spec,
        )?;
        self.spec = spec;
        self.facts = SampleFacts::NONE;
        self.origin = None;
        self.hdr = None;
        Ok(self)
    }

    /// Scan every active pixel to verify its alpha slot equals its canonical maximum value, then
    /// retag this buffer as [`AlphaMode::Opaque`] with `facts.all_opaque = true` (zero copy).
    ///
    /// Performs the full-pixel scan itself — no prior [`assume_facts`](Self::assume_facts) call is
    /// required. Returns [`BufferError::IncompatibleLayout`] if the layout has no alpha slot, or
    /// [`BufferError::UnverifiedOpaqueAlpha`] if any alpha sample is not at channel maximum.
    #[doc(alias = "verify_opaque")]
    #[doc(alias = "mark_opaque")]
    pub fn into_opaque_verified(mut self) -> Result<Self, At<BufferError>> {
        if !self.spec.layout.order.has_alpha_slot() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let new_spec = self.spec.clone().with_alpha(AlphaMode::Opaque);
        new_spec
            .validate()
            .map_err(|e| whereat::at!(BufferError::InvalidFormat(*e.error())))?;
        let bytes = &self.storage.as_bytes()[self.offset..];
        if !check_alpha_all_max(
            bytes,
            self.width,
            self.height,
            self.stride,
            self.spec.layout,
        ) {
            return Err(err_at(BufferError::UnverifiedOpaqueAlpha));
        }
        self.spec = new_spec;
        self.facts.all_opaque = true;
        self.facts.binary_alpha = true;
        Ok(self)
    }

    /// Fill every active pixel's alpha slot to its canonical maximum value and retag this buffer
    /// as [`AlphaMode::Opaque`] with `facts.all_opaque = true` (in place, promoting immutable
    /// external storage via copy-on-write if needed).
    pub fn into_opaque_filled(mut self) -> Result<Self, At<BufferError>> {
        if !self.spec.layout.order.has_alpha_slot() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let new_spec = self.spec.clone().with_alpha(AlphaMode::Opaque);
        new_spec
            .validate()
            .map_err(|e| whereat::at!(BufferError::InvalidFormat(*e.error())))?;
        self.make_mut()?;
        let layout = self.spec.layout;
        let off = self.offset;
        let w = self.width;
        let h = self.height;
        let stride = self.stride;
        fill_alpha_max_unconditional(
            &mut self.storage.as_bytes_mut()[off..],
            w,
            h,
            stride,
            layout,
        );
        self.spec = new_spec;
        self.facts.all_opaque = true;
        self.facts.binary_alpha = true;
        Ok(self)
    }

    /// Retag this buffer as [`AlphaMode::Opaque`] in O(1) without scanning pixels, asserting
    /// that the caller already filled every active alpha sample to channel maximum.
    ///
    /// Pairs with [`new_zeroed`](Self::new_zeroed) for maximum-speed decoder/renderer fill loops:
    /// allocate zeroed with [`AlphaMode::Straight`], write every pixel including `A = max`,
    /// then call `assume_opaque()`.
    pub fn assume_opaque(mut self) -> Result<Self, At<BufferError>> {
        if !self.spec.layout.order.has_alpha_slot() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let new_spec = self.spec.clone().with_alpha(AlphaMode::Opaque);
        new_spec
            .validate()
            .map_err(|e| whereat::at!(BufferError::InvalidFormat(*e.error())))?;
        self.spec = new_spec;
        self.facts.all_opaque = true;
        self.facts.binary_alpha = true;
        Ok(self)
    }

    /// Cached sample facts.
    #[inline]
    #[must_use]
    pub const fn facts(&self) -> SampleFacts {
        self.facts
    }

    /// Attach caller-asserted [`SampleFacts`] (caller assumes responsibility for validity;
    /// cleared on next mutable pixel access).
    #[inline]
    pub fn assume_facts(&mut self, facts: SampleFacts) {
        self.facts = facts;
    }

    /// Descriptive HDR container metadata.
    #[inline]
    #[must_use]
    pub fn hdr(&self) -> HdrMetadata {
        unbox_hdr(&self.hdr)
    }

    /// Attach descriptive HDR container metadata.
    #[inline]
    #[must_use]
    pub fn with_hdr(mut self, hdr: HdrMetadata) -> Self {
        self.hdr = box_hdr(hdr);
        self
    }

    /// Original container color provenance.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> &ColorOrigin {
        origin_ref(&self.origin)
    }

    /// Attach container color provenance.
    #[inline]
    #[must_use]
    pub fn with_origin(mut self, origin: ColorOrigin) -> Self {
        self.origin = box_origin(origin);
        self
    }

    /// Pending EXIF display orientation attached to this buffer (defaults to [`Orientation::Identity`]).
    #[inline]
    #[must_use]
    pub const fn orientation(&self) -> Orientation {
        self.orientation
    }

    /// Display-oriented `(width, height)` after applying [`orientation`](Self::orientation).
    #[inline]
    #[must_use]
    pub const fn display_dimensions(&self) -> (u32, u32) {
        self.orientation.output_dimensions(self.width, self.height)
    }

    /// Replace the pending EXIF display orientation on this buffer.
    #[inline]
    #[must_use]
    pub const fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Set the pending EXIF display orientation on this buffer in place.
    #[inline]
    pub fn set_orientation(&mut self, orientation: Orientation) {
        self.orientation = orientation;
    }

    /// Compose an additional display transform (including reflections) onto this buffer's
    /// pending orientation (`self.orientation.then(delta)`) in O(1) without touching or copying pixels.
    #[inline]
    #[must_use]
    pub const fn compose_orientation(mut self, delta: Orientation) -> Self {
        self.orientation = self.orientation.then(delta);
        self
    }

    /// Borrow active pixel bytes for row `y` (`len == row_bytes()`).
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height()`.
    #[inline]
    #[must_use]
    pub fn row(&self, y: u32) -> &[u8] {
        assert!(
            y < self.height,
            "row index {y} out of bounds ({})",
            self.height
        );
        let start = self.offset + (y as usize) * self.stride;
        &self.storage.as_bytes()[start..start + self.row_bytes()]
    }

    /// Mutably borrow active pixel bytes for row `y` (`len == row_bytes()`), clearing cached
    /// [`SampleFacts`] and copy-on-write promoting immutable external storage if needed.
    ///
    /// For fallible external-storage CoW without panicking on OOM, use
    /// `buf.try_as_view_mut()?.row_mut(y)`.
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height()`, or if copy-on-write promotion of immutable external storage
    /// fails to allocate.
    #[inline]
    #[must_use]
    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(
            y < self.height,
            "row index {y} out of bounds ({})",
            self.height
        );
        self.make_mut()
            .expect("copy-on-write promotion of external PixelBuf failed");
        self.facts = SampleFacts::NONE;
        let rb = self.row_bytes();
        let start = self.offset + (y as usize) * self.stride;
        &mut self.storage.as_bytes_mut()[start..start + rb]
    }

    /// Borrow an immutable [`PixelView`] of the whole image (zero `Arc` refcount bump).
    #[inline]
    #[must_use]
    pub fn as_view(&self) -> PixelView<'_> {
        let bytes = self.storage.as_bytes();
        let end =
            self.offset + required_bytes(self.height, self.stride, self.row_bytes()).unwrap_or(0);
        PixelView {
            data: &bytes[self.offset..end],
            width: self.width,
            height: self.height,
            stride_bytes: self.stride,
            spec: FormatCow::Borrowed(&self.spec),
            facts: self.facts,
            hdr: self.hdr.clone(),
            origin: Some(origin_ref(&self.origin)),
            orientation: self.orientation,
        }
    }

    /// Fallibly borrow a mutable [`PixelViewMut`], clearing cached [`SampleFacts`] and
    /// copy-on-write promoting immutable external storage without panicking on OOM.
    #[inline]
    #[doc(alias = "cow")]
    #[doc(alias = "copy_on_write")]
    pub fn try_as_view_mut(&mut self) -> Result<PixelViewMut<'_>, At<BufferError>> {
        self.make_mut()?;
        self.facts = SampleFacts::NONE;
        let rb = self.row_bytes();
        let end = self.offset + required_bytes(self.height, self.stride, rb).unwrap_or(0);
        let bytes = self.storage.as_bytes_mut();
        Ok(PixelViewMut {
            data: &mut bytes[self.offset..end],
            width: self.width,
            height: self.height,
            stride_bytes: self.stride,
            spec: FormatCow::Borrowed(&self.spec),
            hdr: self.hdr.clone(),
            orientation: self.orientation,
        })
    }

    /// Borrow a mutable [`PixelViewMut`], automatically clearing cached [`SampleFacts`].
    ///
    /// If backed by immutable external storage (`from_external`), promotes via copy-on-write;
    /// use [`try_as_view_mut`](Self::try_as_view_mut) for fallible OOM handling.
    #[inline]
    #[must_use]
    pub fn as_view_mut(&mut self) -> PixelViewMut<'_> {
        self.try_as_view_mut()
            .expect("copy-on-write promotion of external PixelBuf failed")
    }

    /// Borrow a full-width horizontal strip of `row_count` rows starting at `start_y` without copying pixels.
    #[inline]
    pub fn sub_rows(&self, start_y: u32, row_count: u32) -> Result<PixelView<'_>, At<BufferError>> {
        self.as_view().sub_rows(start_y, row_count)
    }

    /// Borrow a mutable full-width horizontal strip of `row_count` rows starting at `start_y`,
    /// automatically clearing cached [`SampleFacts`] and fallibly copy-on-write promoting
    /// immutable external storage if needed.
    #[inline]
    pub fn sub_rows_mut(
        &mut self,
        start_y: u32,
        row_count: u32,
    ) -> Result<PixelViewMut<'_>, At<BufferError>> {
        if row_count == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if start_y
            .checked_add(row_count)
            .is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        self.try_as_view_mut()?.into_sub_rows(start_y, row_count)
    }

    /// Borrow a rectangular cropped subimage view without copying pixels.
    #[inline]
    pub fn crop_view(&self, rect: PixelRect) -> Result<PixelView<'_>, At<BufferError>> {
        self.as_view().crop(rect)
    }

    /// Borrow a mutable rectangular cropped subimage view without copying pixels,
    /// automatically clearing cached [`SampleFacts`] and fallibly copy-on-write promoting
    /// immutable external storage if needed.
    #[inline]
    pub fn crop_view_mut(&mut self, rect: PixelRect) -> Result<PixelViewMut<'_>, At<BufferError>> {
        if rect.width == 0 || rect.height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if rect
            .x
            .checked_add(rect.width)
            .is_none_or(|e| e > self.width)
            || rect
                .y
                .checked_add(rect.height)
                .is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        self.try_as_view_mut()?.into_crop(rect)
    }

    /// Consume this buffer and narrow it to a rectangular crop in **O(1) with zero pixel copies**.
    ///
    /// Keeps the underlying [`PixelStorage`] and row `stride_bytes` intact while advancing the active
    /// byte offset and narrowing `(width, height)`. Preserves sample alignment, cached
    /// [`SampleFacts`], [`HdrMetadata`], [`ColorOrigin`], and [`Orientation`]. Chain
    /// [`into_tight_buf`](Self::into_tight_buf) afterwards to compact the cropped region into
    /// tightly packed storage if desired.
    #[inline]
    pub fn into_crop(mut self, rect: PixelRect) -> Result<Self, At<BufferError>> {
        if rect.width == 0 || rect.height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if rect
            .x
            .checked_add(rect.width)
            .is_none_or(|e| e > self.width)
            || rect
                .y
                .checked_add(rect.height)
                .is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let bpp = self.spec.layout.bytes_per_pixel();
        let delta = (rect.y as usize) * self.stride + (rect.x as usize) * bpp;
        self.offset += delta;
        self.width = rect.width;
        self.height = rect.height;
        Ok(self)
    }

    /// Atomically transform this buffer's bytes and metadata in place with zero reallocation.
    ///
    /// The closure receives `(active_bytes, input)` where `active_bytes` starts at pixel `(0, 0)`.
    /// If it returns `Some(output)`, the new format and geometry are validated against the
    /// existing storage and adopted atomically; if it returns `None`, the buffer's metadata is
    /// unchanged.
    pub fn transform_in_place<F>(&mut self, f: F) -> Result<(), At<BufferError>>
    where
        F: FnOnce(&mut [u8], InPlaceTransformInput<'_>) -> Option<InPlaceTransformOutput>,
    {
        self.make_mut()?;
        let active = &mut self.storage.as_bytes_mut()[self.offset..];
        let input = InPlaceTransformInput {
            width: self.width,
            height: self.height,
            stride_bytes: self.stride,
            format: &self.spec,
            origin: origin_ref(&self.origin),
        };
        let Some(output) = f(active, input) else {
            return Ok(());
        };
        let InPlaceTransformOutput {
            stride_bytes: new_stride,
            format: new_spec,
            origin: new_origin,
            facts: new_facts,
        } = output;
        new_spec
            .validate()
            .map_err(|e| whereat::at!(BufferError::InvalidFormat(*e.error())))?;
        let new_bpp = new_spec.layout.bytes_per_pixel();
        let new_rb = (self.width as usize)
            .checked_mul(new_bpp)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        if new_stride < new_rb {
            return Err(err_at(BufferError::StrideTooSmall));
        }
        let align = new_spec.layout.min_alignment();
        let ptr_addr = (self.storage.as_bytes().as_ptr() as usize)
            .checked_add(self.offset)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        if align > 1 && (ptr_addr % align != 0 || (self.height > 1 && new_stride % align != 0)) {
            return Err(err_at(BufferError::AlignmentViolation));
        }
        let need = required_bytes(self.height, new_stride, new_rb).map_err(err_at)?;
        let avail = self.storage.as_bytes().len().saturating_sub(self.offset);
        if avail < need {
            return Err(err_at(BufferError::InsufficientData));
        }
        self.stride = new_stride;
        self.spec = new_spec;
        self.origin = box_origin(new_origin);
        self.facts = new_facts;
        Ok(())
    }

    /// Fallibly compact strided rows into tight row packing (`stride == row_bytes()`).
    ///
    /// Reuses the existing allocation in-place for owned mutable storage; promotes shared or
    /// immutable external storage via fallible copy-on-write (`BufferError::AllocationFailed` on
    /// OOM). Preserves `offset = 0` whenever the backing base pointer satisfies
    /// `spec.layout.min_alignment()`, and otherwise retains the smallest valid aligned offset
    /// (`< min_alignment()`) so sample alignment is never violated.
    pub fn into_tight_buf(mut self) -> Result<Self, At<BufferError>> {
        let rb = self.row_bytes();
        let align = self.spec.layout.min_alignment();
        let ptr_rem = (self.storage.as_bytes().as_ptr() as usize) % align;
        let target_offset = if ptr_rem == 0 { 0 } else { align - ptr_rem };
        if self.offset == target_offset && (self.height <= 1 || self.stride == rb) {
            self.stride = rb;
            self.storage
                .truncate_bytes(target_offset + rb * self.height as usize);
            return Ok(self);
        }
        self.make_mut()?;
        let ptr_rem = (self.storage.as_bytes().as_ptr() as usize) % align;
        let target_offset = if ptr_rem == 0 { 0 } else { align - ptr_rem };
        let bytes = self.storage.as_bytes_mut();
        for y in 0..self.height as usize {
            let src = self.offset + y * self.stride;
            let dst = target_offset + y * rb;
            if src != dst {
                bytes.copy_within(src..src + rb, dst);
            }
        }
        self.offset = target_offset;
        self.stride = rb;
        self.storage
            .truncate_bytes(target_offset + rb * self.height as usize);
        Ok(self)
    }

    /// Permute or compact channels in place without reallocating for compatible same-depth orders.
    ///
    /// Supports:
    /// - Same-width R↔B channel swaps (`Rgb ↔ Bgr`, `Rgba ↔ Bgra`, `Rgbx ↔ Bgrx`) for any
    ///   unpacked sample type (`U8`, `U16`, `F16`, `F32`, etc.), preserving [`SampleFacts`] and
    ///   [`ColorSpace`].
    /// - Lossless 4→3 padding/opaque drops (`Rgbx`/`Bgrx` → `Rgb`/`Bgr`, or `Rgba`/`Bgra` with
    ///   [`AlphaMode::Opaque`] → `Rgb`/`Bgr`), compacting each row in place and updating
    ///   `AlphaMode::Opaque` to `AlphaMode::None`.
    ///
    /// Refuses lossy alpha drops (`Straight`/`Premultiplied`), packed-word layouts, and
    /// unsupported channel orders with [`BufferError::IncompatibleLayout`].
    pub fn swizzle_in_place(&mut self, target_order: ChannelOrder) -> Result<(), At<BufferError>> {
        let src_order = self.spec.layout.order;
        if src_order == target_order {
            return Ok(());
        }
        if self.spec.layout.packing.is_packed_word() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let sz = self.spec.layout.sample.byte_size();
        match (src_order, target_order) {
            (ChannelOrder::Rgb, ChannelOrder::Bgr)
            | (ChannelOrder::Bgr, ChannelOrder::Rgb)
            | (ChannelOrder::Rgba, ChannelOrder::Bgra)
            | (ChannelOrder::Bgra, ChannelOrder::Rgba)
            | (ChannelOrder::Rgbx, ChannelOrder::Bgrx)
            | (ChannelOrder::Bgrx, ChannelOrder::Rgbx) => {
                let new_layout = SampleLayout {
                    order: target_order,
                    ..self.spec.layout
                };
                let new_spec = PixelFormat::new(new_layout, self.spec.color.clone());
                new_spec
                    .validate()
                    .map_err(|e| whereat::at!(BufferError::InvalidFormat(*e.error())))?;
                self.make_mut()?;
                let bpp = new_layout.bytes_per_pixel();
                let rb = (self.width as usize) * bpp;
                let off = self.offset;
                let stride = self.stride;
                let bytes = self.storage.as_bytes_mut();
                for y in 0..self.height as usize {
                    let row = &mut bytes[off + y * stride..off + y * stride + rb];
                    if sz == 1 {
                        for px in row.chunks_exact_mut(bpp) {
                            px.swap(0, 2);
                        }
                    } else {
                        let c2 = 2 * sz;
                        for px in row.chunks_exact_mut(bpp) {
                            let (first, rest) = px.split_at_mut(c2);
                            first[..sz].swap_with_slice(&mut rest[..sz]);
                        }
                    }
                }
                self.spec = new_spec;
                Ok(())
            }
            (
                ChannelOrder::Rgbx | ChannelOrder::Bgrx | ChannelOrder::Rgba | ChannelOrder::Bgra,
                ChannelOrder::Rgb | ChannelOrder::Bgr,
            ) => {
                let is_alpha_src = matches!(src_order, ChannelOrder::Rgba | ChannelOrder::Bgra);
                if is_alpha_src && self.spec.color.alpha() != AlphaMode::Opaque {
                    return Err(err_at(BufferError::IncompatibleLayout));
                }
                let new_color = if is_alpha_src {
                    self.spec.color.clone().with_alpha(AlphaMode::None)
                } else {
                    self.spec.color.clone()
                };
                let new_layout = SampleLayout {
                    order: target_order,
                    ..self.spec.layout
                };
                let new_spec = PixelFormat::new(new_layout, new_color);
                new_spec
                    .validate()
                    .map_err(|e| whereat::at!(BufferError::InvalidFormat(*e.error())))?;
                self.make_mut()?;
                let swap_rb = matches!(
                    (src_order, target_order),
                    (ChannelOrder::Rgbx | ChannelOrder::Rgba, ChannelOrder::Bgr)
                        | (ChannelOrder::Bgrx | ChannelOrder::Bgra, ChannelOrder::Rgb)
                );
                let src_bpp = 4 * sz;
                let dst_bpp = 3 * sz;
                let w = self.width as usize;
                let off = self.offset;
                let stride = self.stride;
                let bytes = self.storage.as_bytes_mut();
                let mut tmp = [0u8; 24];
                for y in 0..self.height as usize {
                    let row_start = off + y * stride;
                    let row = &mut bytes[row_start..row_start + w * src_bpp];
                    for i in 0..w {
                        let s = i * src_bpp;
                        let d = i * dst_bpp;
                        if swap_rb {
                            tmp[..sz].copy_from_slice(&row[s + 2 * sz..s + 3 * sz]);
                            tmp[sz..2 * sz].copy_from_slice(&row[s + sz..s + 2 * sz]);
                            tmp[2 * sz..3 * sz].copy_from_slice(&row[s..s + sz]);
                            row[d..d + dst_bpp].copy_from_slice(&tmp[..dst_bpp]);
                        } else if s != d {
                            row.copy_within(s..s + dst_bpp, d);
                        }
                    }
                }
                self.spec = new_spec;
                self.facts = SampleFacts::NONE;
                Ok(())
            }
            _ => Err(err_at(BufferError::IncompatibleLayout)),
        }
    }

    /// Move the backing allocation into a tightly packed `Vec<P>` **strictly zero-copy**.
    ///
    /// Succeeds only when `offset == 0`, rows are contiguous, active byte length
    /// matches the backing storage length, `align_of::<P>()` equals the backing
    /// allocation's exact alignment, and capacity in bytes is a multiple of
    /// `size_of::<P>()`. Returns `Err(self)` untouched otherwise.
    #[allow(clippy::result_large_err)]
    pub fn into_vec_zero_copy<P: bytemuck::Pod>(mut self) -> Result<Vec<P>, Self> {
        let total = self.row_bytes() * (self.height as usize);
        if self.offset != 0
            || (self.height > 1 && self.stride != self.row_bytes())
            || !self.storage.can_cast_into_vec::<P>(total)
        {
            return Err(self);
        }
        // Only trim trailing length AFTER proving the zero-copy cast will succeed,
        // so `Err(self)` is ALWAYS returned completely untouched!
        if self.storage.byte_len() > total {
            self.storage.truncate_bytes(total);
        }
        match self.storage.cast_into_vec::<P>() {
            Ok(vec) => Ok(vec),
            Err(storage) => {
                self.storage = storage;
                Err(self)
            }
        }
    }

    /// Extract tightly packed pixels into `Vec<P>`, moving **zero-copy** when
    /// allocation alignment & capacity match, or fallibly copying active rows otherwise.
    ///
    /// Never violates deallocation alignment, never drops row tails on cross-row elements,
    /// and never panics on OOM.
    pub fn into_vec<P: bytemuck::Pod>(self) -> Result<Vec<P>, At<BufferError>> {
        let elem = core::mem::size_of::<P>();
        let rb = self.row_bytes();
        let total = rb
            .checked_mul(self.height as usize)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        if elem == 0 || total % elem != 0 {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        match self.into_vec_zero_copy::<P>() {
            Ok(v) => Ok(v),
            Err(buf) => {
                let count = total / elem;
                if rb % elem == 0 {
                    let mut out: Vec<P> = Vec::new();
                    out.try_reserve_exact(count)
                        .map_err(|_| err_at(BufferError::AllocationFailed))?;
                    let view = buf.as_view();
                    for y in 0..view.height() {
                        let row = view.row(y);
                        if let Ok(typed) = bytemuck::try_cast_slice::<u8, P>(row) {
                            out.extend_from_slice(typed);
                        } else {
                            for chunk in row.chunks_exact(elem) {
                                out.push(bytemuck::pod_read_unaligned(chunk));
                            }
                        }
                    }
                    Ok(out)
                } else {
                    // Element spans row boundaries (e.g. strided RGB16 extracted as `[u16; 2]`):
                    // stream active row bytes into zeroed `Vec<P>` so tight and strided agree 100%.
                    core::alloc::Layout::array::<P>(count)
                        .map_err(|_| err_at(BufferError::AllocationFailed))?;
                    let mut out: Vec<P> = bytemuck::allocation::try_zeroed_vec(count)
                        .map_err(|()| err_at(BufferError::AllocationFailed))?;
                    let dst_bytes = bytemuck::cast_slice_mut::<P, u8>(&mut out);
                    let view = buf.as_view();
                    let mut pos = 0usize;
                    for y in 0..view.height() {
                        let row = view.row(y);
                        dst_bytes[pos..pos + row.len()].copy_from_slice(row);
                        pos += row.len();
                    }
                    Ok(out)
                }
            }
        }
    }

    /// Decompose into owned backing storage and metadata.
    #[inline]
    #[must_use]
    pub fn into_parts(self) -> PixelBufParts {
        PixelBufParts {
            storage: self.storage,
            byte_offset: self.offset,
            width: self.width,
            height: self.height,
            stride_bytes: self.stride,
            format: self.spec,
            facts: self.facts,
            hdr: unbox_hdr(&self.hdr),
            origin: unbox_origin(self.origin),
            orientation: self.orientation,
        }
    }

    /// Access underlying storage reference (for diagnostics and interop checks).
    #[inline]
    #[must_use]
    pub fn storage(&self) -> &PixelStorage {
        &self.storage
    }
}

/// Borrowed-or-owned pixel image returned by zero-copy negotiation / adaptation helpers.
#[derive(Clone, Debug)]
pub enum PixelCow<'a> {
    /// Zero-copy borrowed view.
    Borrowed(PixelView<'a>),
    /// Newly allocated or moved owned buffer.
    Owned(PixelBuf),
}

impl<'a> PixelCow<'a> {
    /// Borrow as a [`PixelView`] (zero `Arc` refcount bump).
    #[inline]
    #[must_use]
    pub fn as_view(&self) -> PixelView<'_> {
        match self {
            Self::Borrowed(v) => PixelView {
                data: v.data,
                width: v.width,
                height: v.height,
                stride_bytes: v.stride_bytes,
                spec: v.spec.to_borrowed(),
                facts: v.facts,
                hdr: v.hdr.clone(),
                origin: v.origin,
                orientation: v.orientation,
            },
            Self::Owned(b) => b.as_view(),
        }
    }

    /// Convert into an owned [`PixelBuf`] (zero-copy if already owned, fallible copy if borrowed).
    pub fn into_owned(self) -> Result<PixelBuf, At<BufferError>> {
        match self {
            Self::Borrowed(v) => v.to_tight_buf(),
            Self::Owned(b) => Ok(b),
        }
    }
}

// ---------------------------------------------------------------------------
// BandedBuf (gigapixel-safe horizontal bands)
// ---------------------------------------------------------------------------

/// Default per-band memory cap (`16 MiB`) for [`BandedBuf`].
pub const DEFAULT_MAX_BAND_BYTES: usize = 16 * 1024 * 1024;

/// Band index and band-local row offset returned by [`BandedBuf::locate_row`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BandRowLocation {
    /// Zero-based index into [`BandedBuf::bands`].
    pub band_index: usize,
    /// Zero-based row index inside that band.
    pub local_y: u32,
}

/// Gigapixel-safe image stored as bounded-size horizontal bands.
///
/// Avoids huge single contiguous allocations while sharing the exact same
/// [`PixelSource`] / [`PixelSink`] contract as [`PixelBuf`] and [`PixelView`].
#[derive(Clone, Debug)]
pub struct BandedBuf {
    width: u32,
    height: u32,
    band_rows: u32,
    bands: Vec<PixelBuf>,
    spec: PixelFormat,
    facts: SampleFacts,
    hdr: Option<Arc<HdrMetadata>>,
    origin: Option<Arc<ColorOrigin>>,
    orientation: Orientation,
}

impl BandedBuf {
    /// Fallibly allocate a zero-filled banded image with each band targeting at most
    /// `max_band_bytes` (minimum one row per band when `0 < max_band_bytes < row_bytes`;
    /// rejects `max_band_bytes == 0`).
    ///
    /// Note: all bands are allocated upfront, so `max_band_bytes` bounds each individual
    /// contiguous allocation rather than total image residency.
    pub fn new(
        width: u32,
        height: u32,
        format: impl Into<PixelFormat>,
        max_band_bytes: usize,
    ) -> Result<Self, At<BufferError>> {
        Self::new_aligned(width, height, format, max_band_bytes, RowAlign::Tight)
    }

    /// Fallibly allocate a zero-filled banded image with explicit per-band [`RowAlign`].
    pub fn new_aligned(
        width: u32,
        height: u32,
        format: impl Into<PixelFormat>,
        max_band_bytes: usize,
        align: RowAlign,
    ) -> Result<Self, At<BufferError>> {
        let spec = format.into();
        spec.validate()
            .map_err(|e| whereat::at!(BufferError::InvalidFormat(*e.error())))?;
        if width == 0 || height == 0 || max_band_bytes == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let stride = spec
            .layout
            .aligned_stride(width, align)
            .map_err(|e| err_at(e.into()))?;
        let rows_per_band = (max_band_bytes / stride.max(1)).max(1).min(height as usize) as u32;
        let band_count = height.div_ceil(rows_per_band) as usize;
        let mut bands = Vec::new();
        bands
            .try_reserve_exact(band_count)
            .map_err(|_| err_at(BufferError::AllocationFailed))?;
        let mut y = 0u32;
        while y < height {
            let h = (height - y).min(rows_per_band);
            bands.push(PixelBuf::new_aligned(width, h, spec.clone(), align)?);
            y += h;
        }
        let facts = SampleFacts::for_zeroed_format(&spec);
        Ok(Self {
            width,
            height,
            band_rows: rows_per_band,
            bands,
            spec,
            facts,
            hdr: None,
            origin: None,
            orientation: Orientation::Identity,
        })
    }

    /// Split a contiguous [`PixelBuf`] into a [`BandedBuf`] with each band bounded by
    /// `max_band_bytes`. Moves `buf` **zero-copy** into a single band when the entire
    /// image fits within one band.
    pub fn from_buf(buf: PixelBuf, max_band_bytes: usize) -> Result<Self, At<BufferError>> {
        if max_band_bytes == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let width = buf.width();
        let height = buf.height();
        let row_bytes = buf.row_bytes();
        let rows_per_band = (max_band_bytes / row_bytes.max(1))
            .max(1)
            .min(height as usize) as u32;
        if height <= rows_per_band {
            let spec = buf.format().clone();
            let facts = buf.facts();
            let hdr = buf.hdr.clone();
            let origin = buf.origin.clone();
            let orientation = buf.orientation();
            let mut bands = Vec::new();
            bands
                .try_reserve_exact(1)
                .map_err(|_| err_at(BufferError::AllocationFailed))?;
            bands.push(buf);
            return Ok(Self {
                width,
                height,
                band_rows: height,
                bands,
                spec,
                facts,
                hdr,
                origin,
                orientation,
            });
        }
        Self::from_view(&buf.as_view(), max_band_bytes)
    }

    /// Copy a borrowed [`PixelView`] into a newly allocated [`BandedBuf`], preserving
    /// `facts`, `hdr`, `origin`, and `orientation`.
    pub fn from_view(view: &PixelView<'_>, max_band_bytes: usize) -> Result<Self, At<BufferError>> {
        if max_band_bytes == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let width = view.width();
        let height = view.height();
        let row_bytes = view.row_bytes();
        let rows_per_band = (max_band_bytes / row_bytes.max(1))
            .max(1)
            .min(height as usize) as u32;
        let band_count = height.div_ceil(rows_per_band) as usize;
        let mut bands = Vec::new();
        bands
            .try_reserve_exact(band_count)
            .map_err(|_| err_at(BufferError::AllocationFailed))?;
        let mut y = 0u32;
        while y < height {
            let h = (height - y).min(rows_per_band);
            bands.push(view.sub_rows(y, h)?.to_tight_buf()?);
            y += h;
        }
        let origin = view.origin().cloned().and_then(box_origin);
        Ok(Self {
            width,
            height,
            band_rows: rows_per_band,
            bands,
            spec: view.format().clone(),
            facts: view.facts(),
            hdr: view.hdr.clone(),
            origin,
            orientation: view.orientation(),
        })
    }

    /// Materialize this banded image into a single contiguous [`PixelBuf`].
    ///
    /// Moves the single band **zero-copy** when `self.bands().len() == 1`; otherwise
    /// fallibly gathers active rows into native-aligned storage while preserving
    /// `facts`, `hdr`, `origin`, and `orientation`.
    pub fn into_buf(mut self) -> Result<PixelBuf, At<BufferError>> {
        if self.bands.len() == 1 {
            let mut single = self.bands.pop().expect("len == 1");
            single.facts = self.facts;
            single.hdr = self.hdr;
            single.origin = self.origin;
            single.orientation = self.orientation;
            return Ok(single);
        }
        let rb = (self.width as usize)
            .checked_mul(self.spec.layout.bytes_per_pixel())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let total = rb
            .checked_mul(self.height as usize)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let storage = PixelStorage::from_rows(
            self.spec.layout.sample,
            total,
            (0..self.height).map(|y| self.row(y)),
        )
        .map_err(|e| err_at(e.into()))?;
        Ok(PixelBuf {
            storage,
            offset: 0,
            width: self.width,
            height: self.height,
            stride: rb,
            spec: self.spec,
            facts: self.facts,
            hdr: self.hdr,
            origin: self.origin,
            orientation: self.orientation,
        })
    }

    /// Number of horizontal bands.
    #[inline]
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Borrow band `index` immutably as a [`PixelView`].
    #[inline]
    #[must_use]
    pub fn band_view(&self, index: usize) -> Option<PixelView<'_>> {
        self.bands.get(index).map(PixelBuf::as_view)
    }

    /// Map global row `y` to its band index and band-local row offset, or `None` if out of bounds.
    #[inline]
    #[must_use]
    pub const fn locate_row(&self, y: u32) -> Option<BandRowLocation> {
        if y >= self.height {
            return None;
        }
        Some(BandRowLocation {
            band_index: (y / self.band_rows) as usize,
            local_y: y % self.band_rows,
        })
    }

    /// Borrow zero-copy [`PixelView`] slices covering global row range
    /// `start_y..start_y + row_count`, split at band boundaries.
    pub fn sub_row_views(
        &self,
        start_y: u32,
        row_count: u32,
    ) -> Result<Vec<PixelView<'_>>, At<BufferError>> {
        if row_count == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let end_y = start_y
            .checked_add(row_count)
            .filter(|&e| e <= self.height)
            .ok_or_else(|| err_at(BufferError::OutOfBounds))?;
        let first_band = (start_y / self.band_rows) as usize;
        let last_band = ((end_y - 1) / self.band_rows) as usize;
        let mut out = Vec::new();
        out.try_reserve_exact(last_band - first_band + 1)
            .map_err(|_| err_at(BufferError::AllocationFailed))?;
        let mut y = start_y;
        while y < end_y {
            let b = (y / self.band_rows) as usize;
            let ly = y % self.band_rows;
            let band_h = self.bands[b].height();
            let take = (end_y - y).min(band_h - ly);
            out.push(self.bands[b].as_view().sub_rows(ly, take)?);
            y += take;
        }
        Ok(out)
    }

    /// Frame width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Nominal rows per band (final band may be shorter).
    #[inline]
    #[must_use]
    pub const fn rows_per_band(&self) -> u32 {
        self.band_rows
    }

    /// Immutable slice of band buffers.
    #[inline]
    #[must_use]
    pub fn bands(&self) -> &[PixelBuf] {
        &self.bands
    }

    /// Borrow band `index` mutably as a [`PixelViewMut`] without allowing structural
    /// band replacement, automatically clearing cached [`SampleFacts`] on both the
    /// outer [`BandedBuf`] and the child band.
    #[inline]
    pub fn band_view_mut(&mut self, index: usize) -> Option<PixelViewMut<'_>> {
        self.facts = SampleFacts::NONE;
        self.bands.get_mut(index).map(PixelBuf::as_view_mut)
    }

    /// Image specification.
    #[inline]
    #[must_use]
    pub fn format(&self) -> &PixelFormat {
        &self.spec
    }

    /// Cached sample facts.
    #[inline]
    #[must_use]
    pub const fn facts(&self) -> SampleFacts {
        self.facts
    }

    /// Descriptive HDR metadata.
    #[inline]
    #[must_use]
    pub fn hdr(&self) -> HdrMetadata {
        unbox_hdr(&self.hdr)
    }

    /// Attach descriptive HDR metadata (propagated to all child bands via shared `Arc`).
    #[inline]
    #[must_use]
    pub fn with_hdr(mut self, hdr: HdrMetadata) -> Self {
        let boxed = box_hdr(hdr);
        self.hdr = boxed.clone();
        for band in &mut self.bands {
            band.hdr = boxed.clone();
        }
        self
    }

    /// Original container color provenance.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> &ColorOrigin {
        origin_ref(&self.origin)
    }

    /// Attach container color provenance (propagated to all child bands via shared `Arc`).
    #[inline]
    #[must_use]
    pub fn with_origin(mut self, origin: ColorOrigin) -> Self {
        let boxed = box_origin(origin);
        self.origin = boxed.clone();
        for band in &mut self.bands {
            band.origin = boxed.clone();
        }
        self
    }

    /// Pending EXIF display orientation attached to this banded buffer.
    #[inline]
    #[must_use]
    pub const fn orientation(&self) -> Orientation {
        self.orientation
    }

    /// Display-oriented `(width, height)` after applying [`orientation`](Self::orientation).
    #[inline]
    #[must_use]
    pub const fn display_dimensions(&self) -> (u32, u32) {
        self.orientation.output_dimensions(self.width, self.height)
    }

    /// Replace pending EXIF display orientation (propagated to all child bands).
    #[inline]
    #[must_use]
    pub fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        for band in &mut self.bands {
            band.orientation = orientation;
        }
        self
    }

    /// Set pending EXIF display orientation in place (propagated to all child bands).
    #[inline]
    pub fn set_orientation(&mut self, orientation: Orientation) {
        self.orientation = orientation;
        for band in &mut self.bands {
            band.orientation = orientation;
        }
    }

    /// Compose an additional display transform onto pending orientation in O(1).
    #[inline]
    #[must_use]
    pub fn compose_orientation(mut self, delta: Orientation) -> Self {
        let next = self.orientation.then(delta);
        self.orientation = next;
        for band in &mut self.bands {
            band.orientation = next;
        }
        self
    }

    /// Active bytes of global row `y`.
    #[inline]
    #[must_use]
    pub fn row(&self, y: u32) -> &[u8] {
        assert!(y < self.height, "row {y} out of bounds ({})", self.height);
        let b = (y / self.band_rows) as usize;
        let ly = y % self.band_rows;
        self.bands[b].as_view().row(ly)
    }

    /// Mutable active bytes of global row `y` (clears cached [`SampleFacts`] on both outer and child band).
    #[inline]
    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(y < self.height, "row {y} out of bounds ({})", self.height);
        self.facts = SampleFacts::NONE;
        let b = (y / self.band_rows) as usize;
        let ly = y % self.band_rows;
        self.bands[b].facts = SampleFacts::NONE;
        let stride = self.bands[b].stride_bytes();
        let rb = self.bands[b].row_bytes();
        let off = self.bands[b].byte_offset();
        let start = off + (ly as usize) * stride;
        &mut self.bands[b].storage.as_bytes_mut()[start..start + rb]
    }
}

// ---------------------------------------------------------------------------
// PlanarBuf / PlanarView (1..=4 planes with first-class BitPacking)
// ---------------------------------------------------------------------------

/// Owned single plane inside a [`PlanarBuf`] with validated geometry invariants.
#[derive(Clone, Debug)]
pub struct PlaneBuf {
    spec: PlaneFormat,
    width: u32,
    height: u32,
    stride: usize,
    storage: PixelStorage,
}

impl PlaneBuf {
    /// Construct a validated owned plane adopting caller-provided [`PixelStorage`] zero-copy.
    pub fn from_storage(
        storage: PixelStorage,
        width: u32,
        height: u32,
        stride_bytes: usize,
        format: PlaneFormat,
    ) -> Result<Self, At<BufferError>> {
        format.validate().map_err(|e| err_at(e.into()))?;
        if width == 0 || height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let row_bytes = (width as usize)
            .checked_mul(format.bytes_per_element())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        if stride_bytes < row_bytes {
            return Err(err_at(BufferError::StrideTooSmall));
        }
        let align = format.sample.min_alignment();
        let bytes = storage.as_bytes();
        if (bytes.as_ptr() as usize) % align != 0 || (height > 1 && stride_bytes % align != 0) {
            return Err(err_at(BufferError::AlignmentViolation));
        }
        let req = required_bytes(height, stride_bytes, row_bytes).map_err(err_at)?;
        if bytes.len() < req {
            return Err(err_at(BufferError::InsufficientData));
        }
        Ok(Self {
            spec: format,
            width,
            height,
            stride: stride_bytes,
            storage,
        })
    }

    /// Construct a validated owned plane adopting an immutable external owner (`Arc<[u8]>`,
    /// `Arc<[u16]>`, mmap, etc.) zero-copy. Returns `Err((err, storage))` untouched on failure.
    #[allow(clippy::result_large_err)]
    pub fn from_external<S: ByteStorage + Send + Sync + 'static>(
        storage: S,
        width: u32,
        height: u32,
        stride_bytes: usize,
        format: PlaneFormat,
    ) -> Result<Self, (At<BufferError>, S)> {
        let check = (|| -> Result<(), At<BufferError>> {
            format.validate().map_err(|e| err_at(e.into()))?;
            if width == 0 || height == 0 {
                return Err(err_at(BufferError::InvalidDimensions));
            }
            let row_bytes = (width as usize)
                .checked_mul(format.bytes_per_element())
                .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
            if stride_bytes < row_bytes {
                return Err(err_at(BufferError::StrideTooSmall));
            }
            let align = format.sample.min_alignment();
            let bytes = storage.as_bytes();
            if (bytes.as_ptr() as usize) % align != 0 || (height > 1 && stride_bytes % align != 0) {
                return Err(err_at(BufferError::AlignmentViolation));
            }
            let req = required_bytes(height, stride_bytes, row_bytes).map_err(err_at)?;
            if bytes.len() < req {
                return Err(err_at(BufferError::InsufficientData));
            }
            Ok(())
        })();
        if let Err(e) = check {
            return Err((e, storage));
        }
        Ok(Self {
            spec: format,
            width,
            height,
            stride: stride_bytes,
            storage: PixelStorage::from_external(storage),
        })
    }

    /// Construct a validated owned plane adopting a mutable external owner (`Box<[T]>`,
    /// custom allocator, pool lease, etc.) zero-copy. Returns `Err((err, storage))` untouched on failure.
    #[allow(clippy::result_large_err)]
    pub fn from_external_mut<S: ByteStorageMut + Send + Sync + 'static>(
        storage: S,
        width: u32,
        height: u32,
        stride_bytes: usize,
        format: PlaneFormat,
    ) -> Result<Self, (At<BufferError>, S)> {
        let check = (|| -> Result<(), At<BufferError>> {
            format.validate().map_err(|e| err_at(e.into()))?;
            if width == 0 || height == 0 {
                return Err(err_at(BufferError::InvalidDimensions));
            }
            let row_bytes = (width as usize)
                .checked_mul(format.bytes_per_element())
                .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
            if stride_bytes < row_bytes {
                return Err(err_at(BufferError::StrideTooSmall));
            }
            let align = format.sample.min_alignment();
            let bytes = storage.as_bytes();
            if (bytes.as_ptr() as usize) % align != 0 || (height > 1 && stride_bytes % align != 0) {
                return Err(err_at(BufferError::AlignmentViolation));
            }
            let req = required_bytes(height, stride_bytes, row_bytes).map_err(err_at)?;
            if bytes.len() < req {
                return Err(err_at(BufferError::InsufficientData));
            }
            Ok(())
        })();
        if let Err(e) = check {
            return Err((e, storage));
        }
        Ok(Self {
            spec: format,
            width,
            height,
            stride: stride_bytes,
            storage: PixelStorage::from_external_mut(storage),
        })
    }

    /// Consume this plane and return its backing [`PixelStorage`].
    #[inline]
    #[must_use]
    pub fn into_storage(self) -> PixelStorage {
        self.storage
    }

    /// Recover an immutable external owner `S` zero-copy; returns `Err(self)` untouched otherwise.
    #[allow(clippy::result_large_err)]
    pub fn into_external<S: 'static>(mut self) -> Result<S, Self> {
        match self.storage.into_external::<S>() {
            Ok(owner) => Ok(owner),
            Err(storage) => {
                self.storage = storage;
                Err(self)
            }
        }
    }

    /// Recover a mutable external owner `S` zero-copy; returns `Err(self)` untouched otherwise.
    #[allow(clippy::result_large_err)]
    pub fn into_external_mut<S: 'static>(mut self) -> Result<S, Self> {
        match self.storage.into_external_mut::<S>() {
            Ok(owner) => Ok(owner),
            Err(storage) => {
                self.storage = storage;
                Err(self)
            }
        }
    }

    /// Fallibly borrow as a mutable [`PlaneViewMut`], copy-on-write promoting immutable external
    /// storage without panicking on OOM.
    #[inline]
    #[doc(alias = "cow")]
    #[doc(alias = "copy_on_write")]
    pub fn try_as_view_mut(&mut self) -> Result<PlaneViewMut<'_>, At<BufferError>> {
        let rb = self.row_bytes();
        let end = required_bytes(self.height, self.stride, rb).unwrap_or(0);
        let bytes = self
            .storage
            .try_as_bytes_mut()
            .map_err(|e| err_at(e.into()))?;
        Ok(PlaneViewMut {
            spec: self.spec,
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: &mut bytes[..end],
        })
    }

    /// Plane specification.
    #[inline]
    #[must_use]
    pub const fn format(&self) -> PlaneFormat {
        self.spec
    }

    /// Plane width in elements.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Plane height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Row stride in bytes.
    #[inline]
    #[must_use]
    pub const fn stride_bytes(&self) -> usize {
        self.stride
    }

    /// Active bytes per row.
    #[inline]
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        (self.width as usize) * self.spec.bytes_per_element()
    }

    /// Backing storage reference.
    #[inline]
    #[must_use]
    pub fn storage(&self) -> &PixelStorage {
        &self.storage
    }

    /// Borrow as an immutable [`PlaneView`].
    #[inline]
    #[must_use]
    pub fn as_view(&self) -> PlaneView<'_> {
        let rb = self.row_bytes();
        let end = required_bytes(self.height, self.stride, rb).unwrap_or(0);
        PlaneView {
            spec: self.spec,
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: &self.storage.as_bytes()[..end],
        }
    }

    /// Borrow as a mutable [`PlaneViewMut`].
    #[inline]
    #[must_use]
    pub fn as_view_mut(&mut self) -> PlaneViewMut<'_> {
        let rb = self.row_bytes();
        let end = required_bytes(self.height, self.stride, rb).unwrap_or(0);
        PlaneViewMut {
            spec: self.spec,
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: &mut self.storage.as_bytes_mut()[..end],
        }
    }

    /// Immutable active bytes of plane row `y`.
    #[inline]
    #[must_use]
    pub fn row(&self, y: u32) -> &[u8] {
        assert!(
            y < self.height,
            "plane row {y} out of bounds ({})",
            self.height
        );
        let rb = self.row_bytes();
        let start = (y as usize) * self.stride;
        &self.storage.as_bytes()[start..start + rb]
    }

    /// Mutable active bytes of plane row `y`.
    #[inline]
    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(
            y < self.height,
            "plane row {y} out of bounds ({})",
            self.height
        );
        let rb = self.row_bytes();
        let start = (y as usize) * self.stride;
        &mut self.storage.as_bytes_mut()[start..start + rb]
    }

    /// Construct a validated owned plane adopting a tightly packed typed `Vec<S>` (`Vec<u8>`,
    /// `Vec<u16>`, `Vec<f32>`, etc.) **zero-copy**.
    ///
    /// Row stride is derived automatically as `width * format.components` samples (e.g. `width`
    /// for luma `Y`, `width * 2` for semi-planar `CbCr`). For padded row strides, use
    /// [`from_samples_strided`](Self::from_samples_strided).
    pub fn from_samples<S: StorageSample>(
        samples: Vec<S>,
        width: u32,
        height: u32,
        format: PlaneFormat,
    ) -> Result<Self, At<BufferError>> {
        format.validate().map_err(|e| err_at(e.into()))?;
        let tight_samples = (width as usize)
            .checked_mul(usize::from(format.components))
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        Self::from_samples_strided(samples, width, height, SampleStride(tight_samples), format)
    }

    /// Construct a validated owned plane adopting a strided typed `Vec<S>` (`Vec<u8>`,
    /// `Vec<u16>`, `Vec<f32>`, etc.) **zero-copy** with an explicit [`SampleStride`]
    /// measured in **`S` elements** (NOT bytes).
    pub fn from_samples_strided<S: StorageSample>(
        samples: Vec<S>,
        width: u32,
        height: u32,
        stride: SampleStride,
        format: PlaneFormat,
    ) -> Result<Self, At<BufferError>> {
        format.validate().map_err(|e| err_at(e.into()))?;
        if format.sample != S::SAMPLE_TYPE || format.packing.is_packed_word() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let stride_bytes = stride
            .samples()
            .checked_mul(core::mem::size_of::<S>())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let storage = PixelStorage::adopt_vec(samples, format.sample)
            .map_err(|_| err_at(BufferError::IncompatibleLayout))?;
        Self::from_storage(storage, width, height, stride_bytes, format)
    }

    /// Borrow a zero-copy rectangular crop of this single plane.
    #[inline]
    pub fn crop_view(&self, rect: PixelRect) -> Result<PlaneView<'_>, At<BufferError>> {
        self.as_view().crop(rect)
    }

    /// Borrow a zero-copy mutable rectangular crop of this single plane, fallibly
    /// copy-on-write promoting immutable external storage if needed.
    #[inline]
    pub fn crop_view_mut(&mut self, rect: PixelRect) -> Result<PlaneViewMut<'_>, At<BufferError>> {
        if rect.width == 0 || rect.height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if rect
            .x
            .checked_add(rect.width)
            .is_none_or(|e| e > self.width)
            || rect
                .y
                .checked_add(rect.height)
                .is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        self.try_as_view_mut()?.into_crop(rect)
    }

    /// Zero-copy typed slice view of row `y` (e.g. `plane.row_as::<u16>(y)`).
    pub fn row_as<S: StorageSample>(&self, y: u32) -> Result<&[S], At<BufferError>> {
        if y >= self.height {
            return Err(err_at(BufferError::OutOfBounds));
        }
        if self.spec.sample != S::SAMPLE_TYPE || self.spec.packing.is_packed_word() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        bytemuck::try_cast_slice(self.row(y)).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
        })
    }

    /// Zero-copy mutable typed slice view of row `y` (e.g. `plane.row_as_mut::<u16>(y)`).
    pub fn row_as_mut<S: StorageSample>(&mut self, y: u32) -> Result<&mut [S], At<BufferError>> {
        if y >= self.height {
            return Err(err_at(BufferError::OutOfBounds));
        }
        if self.spec.sample != S::SAMPLE_TYPE || self.spec.packing.is_packed_word() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        bytemuck::try_cast_slice_mut(self.row_mut(y)).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
        })
    }

    /// Extract the unpacked integer code for `(x, y, component)` respecting [`BitPacking`].
    ///
    /// `component` is the 0-based channel index within this plane: pass `0` for single-channel
    /// planes (`Y`, `Cb`, `Cr`, `R`, `G`, `B`, `Alpha`), or `0`/`1` for interleaved `CbCr`.
    #[must_use]
    pub fn code_u32(&self, x: u32, y: u32, component: usize) -> Option<u32> {
        self.as_view().code_u32(x, y, component)
    }
}

fn read_sample_u32(sample: SampleType, bytes: &[u8]) -> u32 {
    match sample {
        SampleType::U8 => u32::from(bytes[0]),
        SampleType::U16 => u32::from(u16::from_ne_bytes([bytes[0], bytes[1]])),
        SampleType::U32 => u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        _ => 0,
    }
}

fn validate_plane_role_set(roles: impl Iterator<Item = PlaneRole>) -> Result<(), At<BufferError>> {
    const BIT_Y: u32 = 1 << 0;
    const BIT_CB: u32 = 1 << 1;
    const BIT_CR: u32 = 1 << 2;
    const BIT_CBCR: u32 = 1 << 3;
    const BIT_CRCB: u32 = 1 << 4;
    const BIT_R: u32 = 1 << 5;
    const BIT_G: u32 = 1 << 6;
    const BIT_B: u32 = 1 << 7;
    const BIT_C: u32 = 1 << 8;
    const BIT_M: u32 = 1 << 9;
    const BIT_Y_INK: u32 = 1 << 10;
    const BIT_K: u32 = 1 << 11;
    const BIT_ALPHA: u32 = 1 << 12;
    const BIT_GAINMAP: u32 = 1 << 13;
    const BIT_DEPTH: u32 = 1 << 14;
    const BIT_MATTE: u32 = 1 << 15;
    const BIT_THERMAL: u32 = 1 << 16;

    const MASK_YCBCR: u32 = BIT_Y | BIT_CB | BIT_CR | BIT_CBCR | BIT_CRCB;
    const MASK_CHROMA: u32 = BIT_CB | BIT_CR | BIT_CBCR | BIT_CRCB;
    const MASK_RGB: u32 = BIT_R | BIT_G | BIT_B;
    const MASK_CMYK: u32 = BIT_C | BIT_M | BIT_Y_INK | BIT_K;

    let mut seen_mask: u32 = 0;
    let mut has_packed = false;
    let mut extra_ids = [0u16; PlanarBuf::MAX_PLANES];
    let mut extra_len = 0usize;

    for role in roles {
        let mask_opt: Option<u32> = match role {
            PlaneRole::Y => Some(BIT_Y),
            PlaneRole::Cb => Some(BIT_CB),
            PlaneRole::Cr => Some(BIT_CR),
            PlaneRole::CbCr => Some(BIT_CBCR),
            PlaneRole::CrCb => Some(BIT_CRCB),
            PlaneRole::R => Some(BIT_R),
            PlaneRole::G => Some(BIT_G),
            PlaneRole::B => Some(BIT_B),
            PlaneRole::Cyan => Some(BIT_C),
            PlaneRole::Magenta => Some(BIT_M),
            PlaneRole::Yellow => Some(BIT_Y_INK),
            PlaneRole::Key => Some(BIT_K),
            PlaneRole::Alpha => Some(BIT_ALPHA),
            PlaneRole::GainMap => Some(BIT_GAINMAP),
            PlaneRole::Depth => Some(BIT_DEPTH),
            PlaneRole::Matte => Some(BIT_MATTE),
            PlaneRole::Thermal => Some(BIT_THERMAL),
            PlaneRole::Packed => {
                has_packed = true;
                None
            }
            PlaneRole::Extra(id) => {
                if extra_ids[..extra_len].contains(&id) {
                    return Err(err_at(BufferError::IncompatibleLayout));
                }
                if extra_len < extra_ids.len() {
                    extra_ids[extra_len] = id;
                    extra_len += 1;
                }
                None
            }
        };
        if let Some(mask) = mask_opt {
            if (seen_mask & mask) != 0 {
                return Err(err_at(BufferError::IncompatibleLayout));
            }
            seen_mask |= mask;
        }
    }

    let has_ycbcr = (seen_mask & MASK_YCBCR) != 0;
    let has_rgb = (seen_mask & MASK_RGB) != 0;
    let has_cmyk = (seen_mask & MASK_CMYK) != 0;

    // Color families are mutually exclusive (`YCbCr`, `RGB`, `CMYK`, `Packed`).
    let families =
        u8::from(has_ycbcr) + u8::from(has_rgb) + u8::from(has_cmyk) + u8::from(has_packed);
    if families > 1 {
        return Err(err_at(BufferError::IncompatibleLayout));
    }

    // YCbCr structural invariants: no orphan Cb/Cr, no mixing planar & semi-planar chroma,
    // no CbCr + CrCb, and chroma requires luma Y.
    let has_cb = (seen_mask & BIT_CB) != 0;
    let has_cr = (seen_mask & BIT_CR) != 0;
    let has_cbcr = (seen_mask & BIT_CBCR) != 0;
    let has_crcb = (seen_mask & BIT_CRCB) != 0;
    if has_cb != has_cr
        || (has_cbcr && has_crcb)
        || ((has_cb || has_cr) && (has_cbcr || has_crcb))
        || ((seen_mask & MASK_CHROMA) != 0 && (seen_mask & BIT_Y) == 0)
    {
        return Err(err_at(BufferError::IncompatibleLayout));
    }

    // Planar RGB requires all three (R, G, B); planar CMYK requires all four (C, M, Y, K).
    if has_rgb && (seen_mask & MASK_RGB) != MASK_RGB {
        return Err(err_at(BufferError::IncompatibleLayout));
    }
    if has_cmyk && (seen_mask & MASK_CMYK) != MASK_CMYK {
        return Err(err_at(BufferError::IncompatibleLayout));
    }

    Ok(())
}

/// Multi-plane image (YCbCr 4:2:0/4:2:2/4:4:4, NV12, P010, planar RGB+A, planar CMYKA, JXL/TIFF extra planes).
#[derive(Clone, Debug)]
pub struct PlanarBuf {
    width: u32,
    height: u32,
    planes: Vec<PlaneBuf>,
    chroma_siting: ChromaSiting,
    color: Option<Arc<ColorSpace>>,
    matrix_coefficients: Option<MatrixCoefficients>,
    range: Option<SignalRange>,
    alpha: Option<AlphaMode>,
    hdr: Option<Arc<HdrMetadata>>,
    origin: Option<Arc<ColorOrigin>>,
    orientation: Orientation,
}

impl PlanarBuf {
    /// Maximum plane count supported in a single [`PlanarBuf`].
    pub const MAX_PLANES: usize = 64;

    /// Fallibly allocate a zero-filled multi-plane buffer for `plane_formats` (`1..=64` planes).
    pub fn new(
        width: u32,
        height: u32,
        plane_formats: &[PlaneFormat],
    ) -> Result<Self, At<BufferError>> {
        Self::new_aligned(width, height, plane_formats, RowAlign::Tight)
    }

    /// Fallibly allocate a zero-filled multi-plane buffer for `plane_formats` (`1..=64` planes) with
    /// explicit per-plane [`RowAlign`] (`Tight`, `Cache64`, `PixelCache64`, `Samples(n)`).
    pub fn new_aligned(
        width: u32,
        height: u32,
        plane_formats: &[PlaneFormat],
        align: RowAlign,
    ) -> Result<Self, At<BufferError>> {
        if width == 0
            || height == 0
            || plane_formats.is_empty()
            || plane_formats.len() > Self::MAX_PLANES
        {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        validate_plane_role_set(plane_formats.iter().map(|s| s.role))?;
        let mut planes = Vec::new();
        planes
            .try_reserve_exact(plane_formats.len())
            .map_err(|_| err_at(BufferError::AllocationFailed))?;
        for &ps in plane_formats {
            ps.validate().map_err(|e| err_at(e.into()))?;
            let (pw, ph) = ps.plane_dimensions_for_frame(width, height);
            let row_bytes = (pw as usize)
                .checked_mul(ps.bytes_per_element())
                .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
            let stride = ps
                .aligned_stride_for_plane_width(pw, align)
                .map_err(|e| err_at(e.into()))?;
            let total = match align {
                RowAlign::Cache64 | RowAlign::PixelCache64 => stride
                    .checked_mul(ph as usize)
                    .ok_or_else(|| err_at(BufferError::InvalidDimensions))?,
                _ => required_bytes(ph, stride, row_bytes).map_err(err_at)?,
            };
            let storage = match align {
                RowAlign::Cache64 | RowAlign::PixelCache64 => {
                    PixelStorage::zeroed_aligned64_bytes(total)
                }
                _ => PixelStorage::zeroed_bytes(ps.sample, total),
            }
            .map_err(|e| err_at(e.into()))?;
            planes.push(PlaneBuf {
                spec: ps,
                width: pw,
                height: ph,
                stride,
                storage,
            });
        }
        Ok(Self {
            width,
            height,
            planes,
            chroma_siting: ChromaSiting::Unknown,
            color: None,
            matrix_coefficients: None,
            range: None,
            alpha: None,
            hdr: None,
            origin: None,
            orientation: Orientation::Identity,
        })
    }

    /// Construct a validated [`PlanarBuf`] adopting caller-owned planes zero-copy (`1..=64` planes).
    pub fn from_planes(
        width: u32,
        height: u32,
        planes: Vec<PlaneBuf>,
    ) -> Result<Self, At<BufferError>> {
        if width == 0 || height == 0 || planes.is_empty() || planes.len() > Self::MAX_PLANES {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        validate_plane_role_set(planes.iter().map(|p| p.spec.role))?;
        for p in &planes {
            let (ew, eh) = p.spec.plane_dimensions_for_frame(width, height);
            if p.width != ew || p.height != eh {
                return Err(err_at(BufferError::InvalidDimensions));
            }
        }
        Ok(Self {
            width,
            height,
            planes,
            chroma_siting: ChromaSiting::Unknown,
            color: None,
            matrix_coefficients: None,
            range: None,
            alpha: None,
            hdr: None,
            origin: None,
            orientation: Orientation::Identity,
        })
    }

    /// Frame width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Chroma sample siting for subsampled planes.
    #[inline]
    #[must_use]
    pub const fn chroma_siting(&self) -> ChromaSiting {
        self.chroma_siting
    }

    /// Set chroma sample siting for subsampled planes.
    #[inline]
    #[must_use]
    pub const fn with_chroma_siting(mut self, siting: ChromaSiting) -> Self {
        self.chroma_siting = siting;
        self
    }

    /// Optional color interpretation (target RGB/Gray/CMYK/ICC color space of these planes).
    #[inline]
    #[must_use]
    pub fn color(&self) -> Option<&ColorSpace> {
        self.color.as_deref()
    }

    /// Attach color interpretation (primaries, transfer, range, alpha) to this planar buffer.
    #[inline]
    #[must_use]
    pub fn with_color(mut self, color: ColorSpace) -> Self {
        self.color = Some(Arc::new(color));
        self
    }

    /// Optional ITU-T H.273 [`MatrixCoefficients`] (`BT709`, `BT601`, `BT2020_NCL`, `IDENTITY`, etc.).
    #[inline]
    #[must_use]
    pub const fn matrix_coefficients(&self) -> Option<MatrixCoefficients> {
        self.matrix_coefficients
    }

    /// Attach ITU-T H.273 [`MatrixCoefficients`] for YCbCr/GBR planes.
    #[inline]
    #[must_use]
    pub const fn with_matrix_coefficients(mut self, matrix: MatrixCoefficients) -> Self {
        self.matrix_coefficients = Some(matrix);
        self
    }

    /// Explicit signal range override (`Full` or `Narrow`), if set separately from [`color`](Self::color).
    ///
    /// Prefer [`effective_range`](Self::effective_range) when reading the active range.
    #[inline]
    #[must_use]
    pub const fn range_override(&self) -> Option<SignalRange> {
        self.range
    }

    /// Effective signal range, checking [`range_override`](Self::range_override) first and falling
    /// back to [`color`](Self::color)'s range.
    #[inline]
    #[must_use]
    pub fn effective_range(&self) -> Option<SignalRange> {
        self.range
            .or_else(|| self.color.as_deref().and_then(|c| c.range()))
    }

    /// Attach an explicit signal range override (`Full` or `Narrow`).
    #[inline]
    #[must_use]
    pub const fn with_range_override(mut self, range: SignalRange) -> Self {
        self.range = Some(range);
        self
    }

    /// Explicit alpha association override for an attached alpha plane, if set separately from [`color`](Self::color).
    ///
    /// Prefer [`effective_alpha`](Self::effective_alpha) when reading the active alpha mode.
    #[inline]
    #[must_use]
    pub const fn alpha_override(&self) -> Option<AlphaMode> {
        self.alpha
    }

    /// Effective alpha association mode, checking [`alpha_override`](Self::alpha_override) first
    /// and falling back to [`color`](Self::color)'s alpha mode.
    #[inline]
    #[must_use]
    pub fn effective_alpha(&self) -> Option<AlphaMode> {
        self.alpha
            .or_else(|| self.color.as_deref().map(|c| c.alpha()))
    }

    /// Attach an explicit alpha association override (e.g. `AlphaMode::Straight` or `AlphaMode::Premultiplied`).
    #[inline]
    #[must_use]
    pub const fn with_alpha_override(mut self, alpha: AlphaMode) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Immutable slice of planes.
    #[inline]
    #[must_use]
    pub fn planes(&self) -> &[PlaneBuf] {
        &self.planes
    }

    /// Consume this planar buffer and return its owned planes zero-copy.
    #[inline]
    #[must_use]
    pub fn into_planes(self) -> Vec<PlaneBuf> {
        self.planes
    }

    /// Borrow plane `index` mutably as a [`PlaneViewMut`] without allowing structural replacement.
    #[inline]
    pub fn plane_view_mut(&mut self, index: usize) -> Option<PlaneViewMut<'_>> {
        self.planes.get_mut(index).map(PlaneBuf::as_view_mut)
    }

    /// Look up the first owned plane matching `role` (e.g. `PlaneRole::Y`, `PlaneRole::CbCr`).
    #[inline]
    #[must_use]
    pub fn plane_by_role(&self, role: PlaneRole) -> Option<&PlaneBuf> {
        self.planes.iter().find(|p| p.spec.role == role)
    }

    /// Borrow the first plane matching `role` as an immutable [`PlaneView`].
    #[inline]
    #[must_use]
    pub fn plane_view_by_role(&self, role: PlaneRole) -> Option<PlaneView<'_>> {
        self.plane_by_role(role).map(PlaneBuf::as_view)
    }

    /// Borrow the first plane matching `role` mutably as a [`PlaneViewMut`].
    #[inline]
    pub fn plane_view_mut_by_role(&mut self, role: PlaneRole) -> Option<PlaneViewMut<'_>> {
        self.planes
            .iter_mut()
            .find(|p| p.spec.role == role)
            .map(PlaneBuf::as_view_mut)
    }

    /// Descriptive HDR metadata.
    #[inline]
    #[must_use]
    pub fn hdr(&self) -> HdrMetadata {
        unbox_hdr(&self.hdr)
    }

    /// Attach descriptive HDR metadata.
    #[inline]
    #[must_use]
    pub fn with_hdr(mut self, hdr: HdrMetadata) -> Self {
        self.hdr = box_hdr(hdr);
        self
    }

    /// Original container provenance.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> &ColorOrigin {
        origin_ref(&self.origin)
    }

    /// Attach original container provenance.
    #[inline]
    #[must_use]
    pub fn with_origin(mut self, origin: ColorOrigin) -> Self {
        self.origin = box_origin(origin);
        self
    }

    /// Pending EXIF display orientation attached to this planar buffer.
    #[inline]
    #[must_use]
    pub const fn orientation(&self) -> Orientation {
        self.orientation
    }

    /// Display-oriented `(width, height)` after applying [`orientation`](Self::orientation).
    #[inline]
    #[must_use]
    pub const fn display_dimensions(&self) -> (u32, u32) {
        self.orientation.output_dimensions(self.width, self.height)
    }

    /// Replace pending EXIF display orientation.
    #[inline]
    #[must_use]
    pub const fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Set pending EXIF display orientation in place.
    #[inline]
    pub fn set_orientation(&mut self, orientation: Orientation) {
        self.orientation = orientation;
    }

    /// Compose an additional display transform onto pending orientation in O(1).
    #[inline]
    #[must_use]
    pub const fn compose_orientation(mut self, delta: Orientation) -> Self {
        self.orientation = self.orientation.then(delta);
        self
    }
}

/// Validated borrowed view of one plane inside a [`PlanarView`].
#[derive(Clone, Copy, Debug)]
pub struct PlaneView<'a> {
    spec: PlaneFormat,
    width: u32,
    height: u32,
    stride: usize,
    data: &'a [u8],
}

impl<'a> PlaneView<'a> {
    const EMPTY: Self = Self {
        spec: PlaneFormat::new(crate::PlaneRole::Y, SampleType::U8),
        width: 0,
        height: 0,
        stride: 0,
        data: &[],
    };

    /// Construct a validated borrowed plane view.
    pub fn new(
        data: &'a [u8],
        width: u32,
        height: u32,
        stride_bytes: usize,
        format: PlaneFormat,
    ) -> Result<Self, At<BufferError>> {
        format.validate().map_err(|e| err_at(e.into()))?;
        if width == 0 || height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let rb = (width as usize)
            .checked_mul(format.bytes_per_element())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        if stride_bytes < rb {
            return Err(err_at(BufferError::StrideTooSmall));
        }
        let align = format.sample.min_alignment();
        if (data.as_ptr() as usize) % align != 0 || (height > 1 && stride_bytes % align != 0) {
            return Err(err_at(BufferError::AlignmentViolation));
        }
        let req = required_bytes(height, stride_bytes, rb).map_err(err_at)?;
        if data.len() < req {
            return Err(err_at(BufferError::InsufficientData));
        }
        Ok(Self {
            spec: format,
            width,
            height,
            stride: stride_bytes,
            data: &data[..req],
        })
    }

    /// Construct a validated borrowed plane view from a tightly packed typed sample slice
    /// (`&[u8]`, `&[u16]`, `&[f32]`, etc.) **zero-copy**.
    ///
    /// Row stride is derived automatically as `width * format.components` samples (e.g. `width`
    /// for luma `Y`, `width * 2` for semi-planar `CbCr`). For padded row strides, use
    /// [`from_samples_strided`](Self::from_samples_strided).
    pub fn from_samples<S: StorageSample>(
        samples: &'a [S],
        width: u32,
        height: u32,
        format: PlaneFormat,
    ) -> Result<Self, At<BufferError>> {
        format.validate().map_err(|e| err_at(e.into()))?;
        let tight_samples = (width as usize)
            .checked_mul(usize::from(format.components))
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        Self::from_samples_strided(samples, width, height, SampleStride(tight_samples), format)
    }

    /// Construct a validated borrowed plane view from a strided typed sample slice (`&[u8]`,
    /// `&[u16]`, `&[f32]`, etc.) **zero-copy** with an explicit [`SampleStride`] measured in
    /// **`S` elements** (NOT bytes).
    pub fn from_samples_strided<S: StorageSample>(
        samples: &'a [S],
        width: u32,
        height: u32,
        stride: SampleStride,
        format: PlaneFormat,
    ) -> Result<Self, At<BufferError>> {
        format.validate().map_err(|e| err_at(e.into()))?;
        if format.sample != S::SAMPLE_TYPE || format.packing.is_packed_word() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let stride_bytes = stride
            .samples()
            .checked_mul(core::mem::size_of::<S>())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        Self::new(
            bytemuck::cast_slice(samples),
            width,
            height,
            stride_bytes,
            format,
        )
    }

    /// Plane specification.
    #[inline]
    #[must_use]
    pub const fn format(&self) -> PlaneFormat {
        self.spec
    }

    /// Plane width in elements.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Plane height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Row stride in bytes.
    #[inline]
    #[must_use]
    pub const fn stride_bytes(&self) -> usize {
        self.stride
    }

    /// Strided backing byte slice.
    #[inline]
    #[must_use]
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Active bytes of row `y`.
    #[inline]
    #[must_use]
    pub fn row(&self, y: u32) -> &'a [u8] {
        assert!(
            y < self.height,
            "plane row {y} out of bounds ({})",
            self.height
        );
        let rb = (self.width as usize) * self.spec.bytes_per_element();
        let start = (y as usize) * self.stride;
        &self.data[start..start + rb]
    }

    /// Zero-copy typed slice view of row `y` (e.g. `plane.row_as::<u16>(y)`), tied directly to `'a`.
    pub fn row_as<S: StorageSample>(&self, y: u32) -> Result<&'a [S], At<BufferError>> {
        if y >= self.height {
            return Err(err_at(BufferError::OutOfBounds));
        }
        if self.spec.sample != S::SAMPLE_TYPE || self.spec.packing.is_packed_word() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        bytemuck::try_cast_slice(self.row(y)).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
        })
    }

    /// Extract the unpacked integer code for `(x, y, component)` respecting [`BitPacking`].
    ///
    /// `component` is the 0-based channel index within this plane: pass `0` for single-channel
    /// planes (`Y`, `Cb`, `Cr`, `R`, `G`, `B`, `Alpha`), or `0`/`1` for interleaved `CbCr`.
    #[must_use]
    pub fn code_u32(&self, x: u32, y: u32, component: usize) -> Option<u32> {
        if x >= self.width || y >= self.height || component >= self.spec.components as usize {
            return None;
        }
        let row = self.row(y);
        let elem = self.spec.bytes_per_element();
        let base = (x as usize) * elem;
        let word = if self.spec.packing.is_packed_word() {
            read_sample_u32(self.spec.sample, &row[base..])
        } else {
            let sz = self.spec.sample.byte_size();
            read_sample_u32(self.spec.sample, &row[base + component * sz..])
        };
        Some(
            self.spec
                .packing
                .extract_code(self.spec.sample, word, component),
        )
    }

    /// Borrow `component` of this plane as a zero-copy [`CodePlane<'a>`] using the
    /// [`SampleEncoding`] implied by `self.format()`.
    pub fn as_code_plane(self, component: usize) -> Result<CodePlane<'a>, CodePlaneError> {
        let enc = self
            .spec
            .sample_encoding()
            .ok_or(CodePlaneError::StorageMismatch)?;
        CodePlane::from_plane_view(self, component, enc)
    }

    /// Borrow `component` of this plane as a zero-copy [`CodePlane<'a>`] with an explicit
    /// [`SampleEncoding`].
    pub fn as_code_plane_with(
        self,
        component: usize,
        encoding: SampleEncoding,
    ) -> Result<CodePlane<'a>, CodePlaneError> {
        CodePlane::from_plane_view(self, component, encoding)
    }

    /// Zero-copy rectangular crop of this single plane retaining `'a`.
    pub fn crop(&self, rect: PixelRect) -> Result<PlaneView<'a>, At<BufferError>> {
        if rect.width == 0 || rect.height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if rect
            .x
            .checked_add(rect.width)
            .is_none_or(|e| e > self.width)
            || rect
                .y
                .checked_add(rect.height)
                .is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let elem = self.spec.bytes_per_element();
        let start = (rect.y as usize) * self.stride + (rect.x as usize) * elem;
        let end = (rect.y as usize + rect.height as usize - 1) * self.stride
            + (rect.x as usize + rect.width as usize) * elem;
        Ok(PlaneView {
            spec: self.spec,
            width: rect.width,
            height: rect.height,
            stride: self.stride,
            data: &self.data[start..end],
        })
    }

    /// Zero-copy full-width horizontal strip of this single plane retaining `'a`.
    #[inline]
    pub fn sub_rows(&self, start_y: u32, row_count: u32) -> Result<PlaneView<'a>, At<BufferError>> {
        self.crop(PixelRect::new(0, start_y, self.width, row_count))
    }
}

/// Validated mutable borrowed view of one plane.
#[derive(Debug)]
pub struct PlaneViewMut<'a> {
    spec: PlaneFormat,
    width: u32,
    height: u32,
    stride: usize,
    data: &'a mut [u8],
}

impl<'a> PlaneViewMut<'a> {
    /// Construct a validated mutable borrowed plane view.
    pub fn new(
        data: &'a mut [u8],
        width: u32,
        height: u32,
        stride_bytes: usize,
        format: PlaneFormat,
    ) -> Result<Self, At<BufferError>> {
        format.validate().map_err(|e| err_at(e.into()))?;
        if width == 0 || height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let rb = (width as usize)
            .checked_mul(format.bytes_per_element())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        if stride_bytes < rb {
            return Err(err_at(BufferError::StrideTooSmall));
        }
        let align = format.sample.min_alignment();
        if (data.as_ptr() as usize) % align != 0 || (height > 1 && stride_bytes % align != 0) {
            return Err(err_at(BufferError::AlignmentViolation));
        }
        let req = required_bytes(height, stride_bytes, rb).map_err(err_at)?;
        if data.len() < req {
            return Err(err_at(BufferError::InsufficientData));
        }
        Ok(Self {
            spec: format,
            width,
            height,
            stride: stride_bytes,
            data: &mut data[..req],
        })
    }

    /// Construct a validated mutable borrowed plane view from a tightly packed typed sample
    /// slice (`&mut [u8]`, `&mut [u16]`, `&mut [f32]`, etc.) **zero-copy**.
    ///
    /// Row stride is derived automatically as `width * format.components` samples (e.g. `width`
    /// for luma `Y`, `width * 2` for semi-planar `CbCr`). For padded row strides, use
    /// [`from_samples_strided`](Self::from_samples_strided).
    pub fn from_samples<S: StorageSample>(
        samples: &'a mut [S],
        width: u32,
        height: u32,
        format: PlaneFormat,
    ) -> Result<Self, At<BufferError>> {
        format.validate().map_err(|e| err_at(e.into()))?;
        let tight_samples = (width as usize)
            .checked_mul(usize::from(format.components))
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        Self::from_samples_strided(samples, width, height, SampleStride(tight_samples), format)
    }

    /// Construct a validated mutable borrowed plane view from a strided typed sample slice
    /// (`&mut [u8]`, `&mut [u16]`, `&mut [f32]`, etc.) **zero-copy** with an explicit
    /// [`SampleStride`] measured in **`S` elements** (NOT bytes).
    pub fn from_samples_strided<S: StorageSample>(
        samples: &'a mut [S],
        width: u32,
        height: u32,
        stride: SampleStride,
        format: PlaneFormat,
    ) -> Result<Self, At<BufferError>> {
        format.validate().map_err(|e| err_at(e.into()))?;
        if format.sample != S::SAMPLE_TYPE || format.packing.is_packed_word() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let stride_bytes = stride
            .samples()
            .checked_mul(core::mem::size_of::<S>())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        Self::new(
            bytemuck::cast_slice_mut(samples),
            width,
            height,
            stride_bytes,
            format,
        )
    }

    /// Plane specification.
    #[inline]
    #[must_use]
    pub const fn format(&self) -> PlaneFormat {
        self.spec
    }

    /// Plane width in elements.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Plane height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Row stride in bytes.
    #[inline]
    #[must_use]
    pub const fn stride_bytes(&self) -> usize {
        self.stride
    }

    /// Immutable strided backing byte slice.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.data
    }

    /// Mutable strided backing byte slice.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.data
    }

    /// Consume this mutable view and return the underlying strided backing slice tied to `'a`.
    #[inline]
    #[must_use]
    pub fn into_data_mut(self) -> &'a mut [u8] {
        self.data
    }

    /// Immutable active bytes of row `y`.
    #[inline]
    #[must_use]
    pub fn row(&self, y: u32) -> &[u8] {
        assert!(
            y < self.height,
            "plane row {y} out of bounds ({})",
            self.height
        );
        let rb = (self.width as usize) * self.spec.bytes_per_element();
        let start = (y as usize) * self.stride;
        &self.data[start..start + rb]
    }

    /// Mutable active bytes of row `y`.
    #[inline]
    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(
            y < self.height,
            "plane row {y} out of bounds ({})",
            self.height
        );
        let rb = (self.width as usize) * self.spec.bytes_per_element();
        let start = (y as usize) * self.stride;
        &mut self.data[start..start + rb]
    }

    /// Zero-copy typed slice view of row `y` (e.g. `plane.row_as::<u16>(y)`).
    pub fn row_as<S: StorageSample>(&self, y: u32) -> Result<&[S], At<BufferError>> {
        if y >= self.height {
            return Err(err_at(BufferError::OutOfBounds));
        }
        if self.spec.sample != S::SAMPLE_TYPE || self.spec.packing.is_packed_word() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        bytemuck::try_cast_slice(self.row(y)).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
        })
    }

    /// Zero-copy mutable typed slice view of row `y` (e.g. `plane.row_as_mut::<u16>(y)`).
    pub fn row_as_mut<S: StorageSample>(&mut self, y: u32) -> Result<&mut [S], At<BufferError>> {
        if y >= self.height {
            return Err(err_at(BufferError::OutOfBounds));
        }
        if self.spec.sample != S::SAMPLE_TYPE || self.spec.packing.is_packed_word() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        bytemuck::try_cast_slice_mut(self.row_mut(y)).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
        })
    }

    /// Convert `component` of this mutable plane into a zero-copy [`CodePlaneMut<'a>`] using
    /// the [`SampleEncoding`] implied by `self.format()`.
    pub fn into_code_plane_mut(self, component: usize) -> Result<CodePlaneMut<'a>, CodePlaneError> {
        let enc = self
            .spec
            .sample_encoding()
            .ok_or(CodePlaneError::StorageMismatch)?;
        CodePlaneMut::from_plane_view_mut(self, component, enc)
    }

    /// Convert `component` of this mutable plane into a zero-copy [`CodePlaneMut<'a>`] with an
    /// explicit [`SampleEncoding`].
    pub fn into_code_plane_mut_with(
        self,
        component: usize,
        encoding: SampleEncoding,
    ) -> Result<CodePlaneMut<'a>, CodePlaneError> {
        CodePlaneMut::from_plane_view_mut(self, component, encoding)
    }

    /// Borrow as an immutable [`PlaneView`].
    #[inline]
    #[must_use]
    pub fn as_view(&self) -> PlaneView<'_> {
        PlaneView {
            spec: self.spec,
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: self.data,
        }
    }

    /// Downgrade into an immutable [`PlaneView<'a>`] retaining `'a`.
    #[inline]
    #[must_use]
    pub fn into_view(self) -> PlaneView<'a> {
        PlaneView {
            spec: self.spec,
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: self.data,
        }
    }

    /// Zero-copy mutable rectangular crop of this single plane.
    pub fn crop_mut(&mut self, rect: PixelRect) -> Result<PlaneViewMut<'_>, At<BufferError>> {
        if rect.width == 0 || rect.height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if rect
            .x
            .checked_add(rect.width)
            .is_none_or(|e| e > self.width)
            || rect
                .y
                .checked_add(rect.height)
                .is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let elem = self.spec.bytes_per_element();
        let start = (rect.y as usize) * self.stride + (rect.x as usize) * elem;
        let end = (rect.y as usize + rect.height as usize - 1) * self.stride
            + (rect.x as usize + rect.width as usize) * elem;
        Ok(PlaneViewMut {
            spec: self.spec,
            width: rect.width,
            height: rect.height,
            stride: self.stride,
            data: &mut self.data[start..end],
        })
    }

    /// Consume this mutable plane view and narrow it to a rectangular crop retaining `'a`.
    pub fn into_crop(self, rect: PixelRect) -> Result<PlaneViewMut<'a>, At<BufferError>> {
        if rect.width == 0 || rect.height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if rect
            .x
            .checked_add(rect.width)
            .is_none_or(|e| e > self.width)
            || rect
                .y
                .checked_add(rect.height)
                .is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let elem = self.spec.bytes_per_element();
        let start = (rect.y as usize) * self.stride + (rect.x as usize) * elem;
        let end = (rect.y as usize + rect.height as usize - 1) * self.stride
            + (rect.x as usize + rect.width as usize) * elem;
        Ok(PlaneViewMut {
            spec: self.spec,
            width: rect.width,
            height: rect.height,
            stride: self.stride,
            data: &mut self.data[start..end],
        })
    }

    /// Zero-copy mutable full-width horizontal strip of this single plane.
    #[inline]
    pub fn sub_rows_mut(
        &mut self,
        start_y: u32,
        row_count: u32,
    ) -> Result<PlaneViewMut<'_>, At<BufferError>> {
        self.crop_mut(PixelRect::new(0, start_y, self.width, row_count))
    }

    /// Consume this mutable plane view and narrow it to a full-width horizontal strip retaining `'a`.
    #[inline]
    pub fn into_sub_rows(
        self,
        start_y: u32,
        row_count: u32,
    ) -> Result<PlaneViewMut<'a>, At<BufferError>> {
        let width = self.width;
        self.into_crop(PixelRect::new(0, start_y, width, row_count))
    }
}

fn plane_crop_rect(
    spec: PlaneFormat,
    old_origin: (u32, u32),
    new_origin: (u32, u32),
    w: u32,
    h: u32,
) -> Result<PixelRect, At<BufferError>> {
    let hx = spec.subsampling.0.max(1) as u32;
    let vy = spec.subsampling.1.max(1) as u32;
    let base_cx = old_origin.0 / hx;
    let base_cy = old_origin.1 / vy;
    let rel_cx = (new_origin.0 / hx) - base_cx;
    let rel_cy = (new_origin.1 / vy) - base_cy;
    let (cw, ch) = spec
        .cropped_plane_dimensions(w, h, new_origin)
        .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
    Ok(PixelRect::new(rel_cx, rel_cy, cw, ch))
}

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
enum PlanarStorage<'a> {
    Inline {
        planes: [PlaneView<'a>; 4],
        len: usize,
    },
    Spill(Vec<PlaneView<'a>>),
}

/// Borrowed multi-plane view (zero heap allocation for `1..=4` planes; spills cleanly for `5..=64` planes).
///
/// Tracks `crop_origin` in full-resolution luma coordinates, [`ChromaSiting`], and optional
/// color interpretation (`color`, `matrix_coefficients`, `range_override`, `alpha_override`) so subsampled chroma
/// crops and YCbCr decoders carry full color authority.
#[derive(Clone, Debug)]
pub struct PlanarView<'a> {
    width: u32,
    height: u32,
    crop_origin: (u32, u32),
    chroma_siting: ChromaSiting,
    color: Option<Arc<ColorSpace>>,
    matrix_coefficients: Option<MatrixCoefficients>,
    range: Option<SignalRange>,
    alpha: Option<AlphaMode>,
    hdr: Option<Arc<HdrMetadata>>,
    origin: Option<&'a ColorOrigin>,
    orientation: Orientation,
    storage: PlanarStorage<'a>,
}

impl<'a> PlanarView<'a> {
    /// Construct a validated multi-plane borrowed view (`1..=64` planes) at crop origin `(0, 0)`.
    pub fn new(width: u32, height: u32, planes: &[PlaneView<'a>]) -> Result<Self, At<BufferError>> {
        Self::new_with_crop_origin(width, height, (0, 0), ChromaSiting::Unknown, planes)
    }

    /// Construct a validated multi-plane borrowed view with explicit parent-frame `crop_origin`
    /// and [`ChromaSiting`].
    pub fn new_with_crop_origin(
        width: u32,
        height: u32,
        crop_origin: (u32, u32),
        chroma_siting: ChromaSiting,
        planes: &[PlaneView<'a>],
    ) -> Result<Self, At<BufferError>> {
        if width == 0 || height == 0 || planes.is_empty() || planes.len() > PlanarBuf::MAX_PLANES {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        validate_plane_role_set(planes.iter().map(|p| p.spec.role))?;
        for &p in planes {
            let Some((ew, eh)) = p.spec.cropped_plane_dimensions(width, height, crop_origin) else {
                return Err(err_at(BufferError::InvalidDimensions));
            };
            if p.width != ew || p.height != eh {
                return Err(err_at(BufferError::InvalidDimensions));
            }
        }
        let storage = if planes.len() <= 4 {
            let mut arr = [PlaneView::EMPTY; 4];
            for (i, &p) in planes.iter().enumerate() {
                arr[i] = p;
            }
            PlanarStorage::Inline {
                planes: arr,
                len: planes.len(),
            }
        } else {
            let mut v = Vec::new();
            v.try_reserve_exact(planes.len())
                .map_err(|_| err_at(BufferError::AllocationFailed))?;
            v.extend_from_slice(planes);
            PlanarStorage::Spill(v)
        };
        Ok(Self {
            width,
            height,
            crop_origin,
            chroma_siting,
            color: None,
            matrix_coefficients: None,
            range: None,
            alpha: None,
            hdr: None,
            origin: None,
            orientation: Orientation::Identity,
            storage,
        })
    }

    /// Frame width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Crop origin `(x, y)` in full-resolution parent-frame coordinates.
    #[inline]
    #[must_use]
    pub const fn crop_origin(&self) -> (u32, u32) {
        self.crop_origin
    }

    /// Chroma sample siting for subsampled planes.
    #[inline]
    #[must_use]
    pub const fn chroma_siting(&self) -> ChromaSiting {
        self.chroma_siting
    }

    /// Replace chroma sample siting on this borrowed view.
    #[inline]
    #[must_use]
    pub const fn with_chroma_siting(mut self, siting: ChromaSiting) -> Self {
        self.chroma_siting = siting;
        self
    }

    /// Optional color interpretation of these planes.
    #[inline]
    #[must_use]
    pub fn color(&self) -> Option<&ColorSpace> {
        self.color.as_deref()
    }

    /// Attach color interpretation to this borrowed planar view.
    #[inline]
    #[must_use]
    pub fn with_color(mut self, color: ColorSpace) -> Self {
        self.color = Some(Arc::new(color));
        self
    }

    /// Optional ITU-T H.273 [`MatrixCoefficients`] (`BT709`, `BT601`, `BT2020_NCL`, `IDENTITY`, etc.).
    #[inline]
    #[must_use]
    pub const fn matrix_coefficients(&self) -> Option<MatrixCoefficients> {
        self.matrix_coefficients
    }

    /// Attach ITU-T H.273 [`MatrixCoefficients`].
    #[inline]
    #[must_use]
    pub const fn with_matrix_coefficients(mut self, matrix: MatrixCoefficients) -> Self {
        self.matrix_coefficients = Some(matrix);
        self
    }

    /// Explicit signal range override, if set separately from [`color`](Self::color).
    #[inline]
    #[must_use]
    pub const fn range_override(&self) -> Option<SignalRange> {
        self.range
    }

    /// Effective signal range ([`range_override`](Self::range_override) or [`color`](Self::color)'s range).
    #[inline]
    #[must_use]
    pub fn effective_range(&self) -> Option<SignalRange> {
        self.range
            .or_else(|| self.color.as_deref().and_then(|c| c.range()))
    }

    /// Attach an explicit signal range override.
    #[inline]
    #[must_use]
    pub const fn with_range_override(mut self, range: SignalRange) -> Self {
        self.range = Some(range);
        self
    }

    /// Explicit alpha association override, if set separately from [`color`](Self::color).
    #[inline]
    #[must_use]
    pub const fn alpha_override(&self) -> Option<AlphaMode> {
        self.alpha
    }

    /// Effective alpha association mode ([`alpha_override`](Self::alpha_override) or [`color`](Self::color)'s alpha).
    #[inline]
    #[must_use]
    pub fn effective_alpha(&self) -> Option<AlphaMode> {
        self.alpha
            .or_else(|| self.color.as_deref().map(|c| c.alpha()))
    }

    /// Attach an explicit alpha association override.
    #[inline]
    #[must_use]
    pub const fn with_alpha_override(mut self, alpha: AlphaMode) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Descriptive HDR container metadata attached to this planar view.
    #[inline]
    #[must_use]
    pub fn hdr(&self) -> HdrMetadata {
        unbox_hdr(&self.hdr)
    }

    /// Attach descriptive HDR container metadata to this planar view.
    #[inline]
    #[must_use]
    pub fn with_hdr(mut self, hdr: HdrMetadata) -> Self {
        self.hdr = box_hdr(hdr);
        self
    }

    /// Borrowed original container color provenance, if attached.
    #[inline]
    #[must_use]
    pub const fn origin(&self) -> Option<&'a ColorOrigin> {
        self.origin
    }

    /// Attach borrowed container color provenance to this planar view.
    #[inline]
    #[must_use]
    pub const fn with_origin(mut self, origin: &'a ColorOrigin) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Pending EXIF display orientation attached to this planar view.
    #[inline]
    #[must_use]
    pub const fn orientation(&self) -> Orientation {
        self.orientation
    }

    /// Display-oriented `(width, height)` after applying [`orientation`](Self::orientation).
    #[inline]
    #[must_use]
    pub const fn display_dimensions(&self) -> (u32, u32) {
        self.orientation.output_dimensions(self.width, self.height)
    }

    /// Replace pending EXIF display orientation on this planar view.
    #[inline]
    #[must_use]
    pub const fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Compose an additional display transform onto pending orientation in O(1).
    #[inline]
    #[must_use]
    pub const fn compose_orientation(mut self, delta: Orientation) -> Self {
        self.orientation = self.orientation.then(delta);
        self
    }

    /// Borrowed planes slice.
    #[inline]
    #[must_use]
    pub fn planes(&self) -> &[PlaneView<'a>] {
        match &self.storage {
            PlanarStorage::Inline { planes, len } => &planes[..*len],
            PlanarStorage::Spill(v) => v.as_slice(),
        }
    }

    /// Look up the first plane matching `role` (e.g. `PlaneRole::Y`, `PlaneRole::CbCr`),
    /// returning a copy tied directly to `'a`.
    #[inline]
    #[must_use]
    pub fn plane_by_role(&self, role: PlaneRole) -> Option<PlaneView<'a>> {
        self.planes().iter().copied().find(|p| p.spec.role == role)
    }

    /// Zero-copy rectangular crop of all planes preserving exact parent-frame chroma phase
    /// even for odd `(x, y)` origins on 4:2:0 / 4:2:2 subsampled frames.
    pub fn crop(&self, rect: PixelRect) -> Result<PlanarView<'a>, At<BufferError>> {
        if rect.width == 0 || rect.height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if rect
            .x
            .checked_add(rect.width)
            .is_none_or(|e| e > self.width)
            || rect
                .y
                .checked_add(rect.height)
                .is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let new_ox = self
            .crop_origin
            .0
            .checked_add(rect.x)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let new_oy = self
            .crop_origin
            .1
            .checked_add(rect.y)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let new_origin = (new_ox, new_oy);
        let src_planes = self.planes();
        let crop_one = |p: &PlaneView<'a>| -> Result<PlaneView<'a>, At<BufferError>> {
            let pr = plane_crop_rect(
                p.spec,
                self.crop_origin,
                new_origin,
                rect.width,
                rect.height,
            )?;
            p.crop(pr)
        };
        let storage = if src_planes.len() <= 4 {
            let mut arr = [PlaneView::EMPTY; 4];
            for (i, p) in src_planes.iter().enumerate() {
                arr[i] = crop_one(p)?;
            }
            PlanarStorage::Inline {
                planes: arr,
                len: src_planes.len(),
            }
        } else {
            let mut v = Vec::new();
            v.try_reserve_exact(src_planes.len())
                .map_err(|_| err_at(BufferError::AllocationFailed))?;
            for p in src_planes {
                v.push(crop_one(p)?);
            }
            PlanarStorage::Spill(v)
        };
        Ok(PlanarView {
            width: rect.width,
            height: rect.height,
            crop_origin: new_origin,
            chroma_siting: self.chroma_siting,
            color: self.color.clone(),
            matrix_coefficients: self.matrix_coefficients,
            range: self.range,
            alpha: self.alpha,
            hdr: self.hdr.clone(),
            origin: self.origin,
            orientation: self.orientation,
            storage,
        })
    }

    /// Zero-copy full-width horizontal strip of all planes preserving parent-frame chroma phase.
    #[inline]
    pub fn sub_rows(
        &self,
        start_y: u32,
        row_count: u32,
    ) -> Result<PlanarView<'a>, At<BufferError>> {
        self.crop(PixelRect::new(0, start_y, self.width, row_count))
    }
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum PlanarMutStorage<'a> {
    One([PlaneViewMut<'a>; 1]),
    Two([PlaneViewMut<'a>; 2]),
    Three([PlaneViewMut<'a>; 3]),
    Four([PlaneViewMut<'a>; 4]),
    Spill(Vec<PlaneViewMut<'a>>),
}

impl<'a> PlanarMutStorage<'a> {
    fn as_slice(&self) -> &[PlaneViewMut<'a>] {
        match self {
            Self::One(p) => p,
            Self::Two(p) => p,
            Self::Three(p) => p,
            Self::Four(p) => p,
            Self::Spill(v) => v,
        }
    }

    fn as_mut_slice(&mut self) -> &mut [PlaneViewMut<'a>] {
        match self {
            Self::One(p) => p,
            Self::Two(p) => p,
            Self::Three(p) => p,
            Self::Four(p) => p,
            Self::Spill(v) => v,
        }
    }
}

/// Mutable borrowed multi-plane view (zero heap allocation for `1..=4` disjoint planes; spills
/// cleanly for `5..=64` planes).
#[derive(Debug)]
pub struct PlanarViewMut<'a> {
    width: u32,
    height: u32,
    crop_origin: (u32, u32),
    chroma_siting: ChromaSiting,
    color: Option<Arc<ColorSpace>>,
    matrix_coefficients: Option<MatrixCoefficients>,
    range: Option<SignalRange>,
    alpha: Option<AlphaMode>,
    hdr: Option<Arc<HdrMetadata>>,
    orientation: Orientation,
    storage: PlanarMutStorage<'a>,
}

impl<'a> PlanarViewMut<'a> {
    fn validate_planes(
        width: u32,
        height: u32,
        crop_origin: (u32, u32),
        planes: &[PlaneViewMut<'a>],
    ) -> Result<(), At<BufferError>> {
        if width == 0 || height == 0 || planes.is_empty() || planes.len() > PlanarBuf::MAX_PLANES {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        validate_plane_role_set(planes.iter().map(|p| p.spec.role))?;
        for p in planes {
            let Some((ew, eh)) = p.spec.cropped_plane_dimensions(width, height, crop_origin) else {
                return Err(err_at(BufferError::InvalidDimensions));
            };
            if p.width != ew || p.height != eh {
                return Err(err_at(BufferError::InvalidDimensions));
            }
        }
        Ok(())
    }

    /// Construct a validated mutable multi-plane view from caller-owned `PlaneViewMut`s.
    pub fn new(
        width: u32,
        height: u32,
        planes: Vec<PlaneViewMut<'a>>,
    ) -> Result<Self, At<BufferError>> {
        Self::new_with_crop_origin(width, height, (0, 0), ChromaSiting::Unknown, planes)
    }

    /// Construct a validated mutable multi-plane view with explicit `crop_origin` and [`ChromaSiting`].
    pub fn new_with_crop_origin(
        width: u32,
        height: u32,
        crop_origin: (u32, u32),
        chroma_siting: ChromaSiting,
        mut planes: Vec<PlaneViewMut<'a>>,
    ) -> Result<Self, At<BufferError>> {
        Self::validate_planes(width, height, crop_origin, &planes)?;
        let storage = match planes.len() {
            1 => {
                let p0 = planes.pop().unwrap();
                PlanarMutStorage::One([p0])
            }
            2 => {
                let p1 = planes.pop().unwrap();
                let p0 = planes.pop().unwrap();
                PlanarMutStorage::Two([p0, p1])
            }
            3 => {
                let p2 = planes.pop().unwrap();
                let p1 = planes.pop().unwrap();
                let p0 = planes.pop().unwrap();
                PlanarMutStorage::Three([p0, p1, p2])
            }
            4 => {
                let p3 = planes.pop().unwrap();
                let p2 = planes.pop().unwrap();
                let p1 = planes.pop().unwrap();
                let p0 = planes.pop().unwrap();
                PlanarMutStorage::Four([p0, p1, p2, p3])
            }
            _ => PlanarMutStorage::Spill(planes),
        };
        Ok(Self {
            width,
            height,
            crop_origin,
            chroma_siting,
            color: None,
            matrix_coefficients: None,
            range: None,
            alpha: None,
            hdr: None,
            orientation: Orientation::Identity,
            storage,
        })
    }

    /// Construct a validated mutable multi-plane view from a fixed array of `1..=4` disjoint
    /// planes with **zero heap allocation**.
    pub fn from_array<const N: usize>(
        width: u32,
        height: u32,
        planes: [PlaneViewMut<'a>; N],
    ) -> Result<Self, At<BufferError>> {
        Self::from_array_with_crop_origin(width, height, (0, 0), ChromaSiting::Unknown, planes)
    }

    /// Construct a validated mutable multi-plane view from a fixed array of planes with explicit
    /// `crop_origin` and [`ChromaSiting`] (zero heap allocation for `1..=4` planes).
    pub fn from_array_with_crop_origin<const N: usize>(
        width: u32,
        height: u32,
        crop_origin: (u32, u32),
        chroma_siting: ChromaSiting,
        planes: [PlaneViewMut<'a>; N],
    ) -> Result<Self, At<BufferError>> {
        Self::validate_planes(width, height, crop_origin, &planes)?;
        let mut iter = planes.into_iter();
        let storage = match N {
            1 => PlanarMutStorage::One([iter.next().unwrap()]),
            2 => PlanarMutStorage::Two([iter.next().unwrap(), iter.next().unwrap()]),
            3 => PlanarMutStorage::Three([
                iter.next().unwrap(),
                iter.next().unwrap(),
                iter.next().unwrap(),
            ]),
            4 => PlanarMutStorage::Four([
                iter.next().unwrap(),
                iter.next().unwrap(),
                iter.next().unwrap(),
                iter.next().unwrap(),
            ]),
            _ => {
                let mut v = Vec::new();
                v.try_reserve_exact(N)
                    .map_err(|_| err_at(BufferError::AllocationFailed))?;
                v.extend(iter);
                PlanarMutStorage::Spill(v)
            }
        };
        Ok(Self {
            width,
            height,
            crop_origin,
            chroma_siting,
            color: None,
            matrix_coefficients: None,
            range: None,
            alpha: None,
            hdr: None,
            orientation: Orientation::Identity,
            storage,
        })
    }

    /// Frame width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Crop origin `(x, y)` in full-resolution parent-frame coordinates.
    #[inline]
    #[must_use]
    pub const fn crop_origin(&self) -> (u32, u32) {
        self.crop_origin
    }

    /// Chroma sample siting.
    #[inline]
    #[must_use]
    pub const fn chroma_siting(&self) -> ChromaSiting {
        self.chroma_siting
    }

    /// Replace chroma sample siting.
    #[inline]
    #[must_use]
    pub const fn with_chroma_siting(mut self, siting: ChromaSiting) -> Self {
        self.chroma_siting = siting;
        self
    }

    /// Optional color interpretation.
    #[inline]
    #[must_use]
    pub fn color(&self) -> Option<&ColorSpace> {
        self.color.as_deref()
    }

    /// Attach color interpretation.
    #[inline]
    #[must_use]
    pub fn with_color(mut self, color: ColorSpace) -> Self {
        self.color = Some(Arc::new(color));
        self
    }

    /// Optional ITU-T H.273 [`MatrixCoefficients`].
    #[inline]
    #[must_use]
    pub const fn matrix_coefficients(&self) -> Option<MatrixCoefficients> {
        self.matrix_coefficients
    }

    /// Attach ITU-T H.273 [`MatrixCoefficients`].
    #[inline]
    #[must_use]
    pub const fn with_matrix_coefficients(mut self, matrix: MatrixCoefficients) -> Self {
        self.matrix_coefficients = Some(matrix);
        self
    }

    /// Explicit signal range override, if set separately from [`color`](Self::color).
    #[inline]
    #[must_use]
    pub const fn range_override(&self) -> Option<SignalRange> {
        self.range
    }

    /// Effective signal range.
    #[inline]
    #[must_use]
    pub fn effective_range(&self) -> Option<SignalRange> {
        self.range
            .or_else(|| self.color.as_deref().and_then(|c| c.range()))
    }

    /// Attach an explicit signal range override.
    #[inline]
    #[must_use]
    pub const fn with_range_override(mut self, range: SignalRange) -> Self {
        self.range = Some(range);
        self
    }

    /// Explicit alpha association override, if set separately from [`color`](Self::color).
    #[inline]
    #[must_use]
    pub const fn alpha_override(&self) -> Option<AlphaMode> {
        self.alpha
    }

    /// Effective alpha association mode.
    #[inline]
    #[must_use]
    pub fn effective_alpha(&self) -> Option<AlphaMode> {
        self.alpha
            .or_else(|| self.color.as_deref().map(|c| c.alpha()))
    }

    /// Attach an explicit alpha association override.
    #[inline]
    #[must_use]
    pub const fn with_alpha_override(mut self, alpha: AlphaMode) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Descriptive HDR container metadata attached to this mutable planar view.
    #[inline]
    #[must_use]
    pub fn hdr(&self) -> HdrMetadata {
        unbox_hdr(&self.hdr)
    }

    /// Attach descriptive HDR container metadata.
    #[inline]
    #[must_use]
    pub fn with_hdr(mut self, hdr: HdrMetadata) -> Self {
        self.hdr = box_hdr(hdr);
        self
    }

    /// Pending EXIF display orientation attached to this mutable planar view.
    #[inline]
    #[must_use]
    pub const fn orientation(&self) -> Orientation {
        self.orientation
    }

    /// Display-oriented `(width, height)` after applying [`orientation`](Self::orientation).
    #[inline]
    #[must_use]
    pub const fn display_dimensions(&self) -> (u32, u32) {
        self.orientation.output_dimensions(self.width, self.height)
    }

    /// Replace pending EXIF display orientation.
    #[inline]
    #[must_use]
    pub const fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Set pending EXIF display orientation in place.
    #[inline]
    pub fn set_orientation(&mut self, orientation: Orientation) {
        self.orientation = orientation;
    }

    /// Compose an additional display transform onto pending orientation in O(1).
    #[inline]
    #[must_use]
    pub const fn compose_orientation(mut self, delta: Orientation) -> Self {
        self.orientation = self.orientation.then(delta);
        self
    }

    /// Immutable slice of mutable plane views.
    #[inline]
    #[must_use]
    pub fn planes(&self) -> &[PlaneViewMut<'a>] {
        self.storage.as_slice()
    }

    /// Mutable slice of all disjoint planes (allows simultaneous mutable access via `split_at_mut`).
    #[inline]
    pub fn planes_mut(&mut self) -> &mut [PlaneViewMut<'a>] {
        self.storage.as_mut_slice()
    }

    /// Mutably borrow plane `index`.
    #[inline]
    pub fn plane_mut(&mut self, index: usize) -> Option<&mut PlaneViewMut<'a>> {
        self.storage.as_mut_slice().get_mut(index)
    }

    /// Look up the first plane matching `role` immutably.
    #[inline]
    #[must_use]
    pub fn plane_by_role(&self, role: PlaneRole) -> Option<&PlaneViewMut<'a>> {
        self.storage.as_slice().iter().find(|p| p.spec.role == role)
    }

    /// Look up the first plane matching `role` mutably.
    #[inline]
    pub fn plane_by_role_mut(&mut self, role: PlaneRole) -> Option<&mut PlaneViewMut<'a>> {
        self.storage
            .as_mut_slice()
            .iter_mut()
            .find(|p| p.spec.role == role)
    }

    /// Borrow all planes immutably as a [`PlanarView`].
    #[must_use]
    pub fn as_view(&self) -> PlanarView<'_> {
        self.try_as_view()
            .expect("PlanarViewMut::as_view allocation failed for >4 planes")
    }

    /// Fallibly borrow all planes immutably as a [`PlanarView`].
    pub fn try_as_view(&self) -> Result<PlanarView<'_>, At<BufferError>> {
        let src = self.storage.as_slice();
        let storage = if src.len() <= 4 {
            let mut arr = [PlaneView::EMPTY; 4];
            for (i, p) in src.iter().enumerate() {
                arr[i] = p.as_view();
            }
            PlanarStorage::Inline {
                planes: arr,
                len: src.len(),
            }
        } else {
            let mut v = Vec::new();
            v.try_reserve_exact(src.len())
                .map_err(|_| err_at(BufferError::AllocationFailed))?;
            for p in src {
                v.push(p.as_view());
            }
            PlanarStorage::Spill(v)
        };
        Ok(PlanarView {
            width: self.width,
            height: self.height,
            crop_origin: self.crop_origin,
            chroma_siting: self.chroma_siting,
            color: self.color.clone(),
            matrix_coefficients: self.matrix_coefficients,
            range: self.range,
            alpha: self.alpha,
            hdr: self.hdr.clone(),
            origin: None,
            orientation: self.orientation,
            storage,
        })
    }

    /// Downgrade into an immutable [`PlanarView<'a>`] retaining `'a`.
    pub fn into_view(self) -> Result<PlanarView<'a>, At<BufferError>> {
        let storage = match self.storage {
            PlanarMutStorage::One([p0]) => PlanarStorage::Inline {
                planes: [
                    p0.into_view(),
                    PlaneView::EMPTY,
                    PlaneView::EMPTY,
                    PlaneView::EMPTY,
                ],
                len: 1,
            },
            PlanarMutStorage::Two([p0, p1]) => PlanarStorage::Inline {
                planes: [
                    p0.into_view(),
                    p1.into_view(),
                    PlaneView::EMPTY,
                    PlaneView::EMPTY,
                ],
                len: 2,
            },
            PlanarMutStorage::Three([p0, p1, p2]) => PlanarStorage::Inline {
                planes: [
                    p0.into_view(),
                    p1.into_view(),
                    p2.into_view(),
                    PlaneView::EMPTY,
                ],
                len: 3,
            },
            PlanarMutStorage::Four([p0, p1, p2, p3]) => PlanarStorage::Inline {
                planes: [
                    p0.into_view(),
                    p1.into_view(),
                    p2.into_view(),
                    p3.into_view(),
                ],
                len: 4,
            },
            PlanarMutStorage::Spill(v) => {
                let mut out = Vec::new();
                out.try_reserve_exact(v.len())
                    .map_err(|_| err_at(BufferError::AllocationFailed))?;
                for p in v {
                    out.push(p.into_view());
                }
                PlanarStorage::Spill(out)
            }
        };
        Ok(PlanarView {
            width: self.width,
            height: self.height,
            crop_origin: self.crop_origin,
            chroma_siting: self.chroma_siting,
            color: self.color,
            matrix_coefficients: self.matrix_coefficients,
            range: self.range,
            alpha: self.alpha,
            hdr: self.hdr,
            origin: None,
            orientation: self.orientation,
            storage,
        })
    }

    /// Zero-copy mutable rectangular crop of all planes preserving exact parent-frame chroma
    /// phase even for odd `(x, y)` origins on 4:2:0 / 4:2:2 subsampled frames (zero heap
    /// allocation for `<= 4` planes).
    pub fn crop_mut(&mut self, rect: PixelRect) -> Result<PlanarViewMut<'_>, At<BufferError>> {
        if rect.width == 0 || rect.height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if rect
            .x
            .checked_add(rect.width)
            .is_none_or(|e| e > self.width)
            || rect
                .y
                .checked_add(rect.height)
                .is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let old_origin = self.crop_origin;
        let new_ox = old_origin
            .0
            .checked_add(rect.x)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let new_oy = old_origin
            .1
            .checked_add(rect.y)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let new_origin = (new_ox, new_oy);
        let w = rect.width;
        let h = rect.height;
        let storage = match self.storage.as_mut_slice() {
            [p0] => {
                let r0 = plane_crop_rect(p0.spec, old_origin, new_origin, w, h)?;
                PlanarMutStorage::One([p0.crop_mut(r0)?])
            }
            [p0, p1] => {
                let r0 = plane_crop_rect(p0.spec, old_origin, new_origin, w, h)?;
                let r1 = plane_crop_rect(p1.spec, old_origin, new_origin, w, h)?;
                PlanarMutStorage::Two([p0.crop_mut(r0)?, p1.crop_mut(r1)?])
            }
            [p0, p1, p2] => {
                let r0 = plane_crop_rect(p0.spec, old_origin, new_origin, w, h)?;
                let r1 = plane_crop_rect(p1.spec, old_origin, new_origin, w, h)?;
                let r2 = plane_crop_rect(p2.spec, old_origin, new_origin, w, h)?;
                PlanarMutStorage::Three([p0.crop_mut(r0)?, p1.crop_mut(r1)?, p2.crop_mut(r2)?])
            }
            [p0, p1, p2, p3] => {
                let r0 = plane_crop_rect(p0.spec, old_origin, new_origin, w, h)?;
                let r1 = plane_crop_rect(p1.spec, old_origin, new_origin, w, h)?;
                let r2 = plane_crop_rect(p2.spec, old_origin, new_origin, w, h)?;
                let r3 = plane_crop_rect(p3.spec, old_origin, new_origin, w, h)?;
                PlanarMutStorage::Four([
                    p0.crop_mut(r0)?,
                    p1.crop_mut(r1)?,
                    p2.crop_mut(r2)?,
                    p3.crop_mut(r3)?,
                ])
            }
            many => {
                let mut v = Vec::new();
                v.try_reserve_exact(many.len())
                    .map_err(|_| err_at(BufferError::AllocationFailed))?;
                for p in many.iter_mut() {
                    let r = plane_crop_rect(p.spec, old_origin, new_origin, w, h)?;
                    v.push(p.crop_mut(r)?);
                }
                PlanarMutStorage::Spill(v)
            }
        };
        Ok(PlanarViewMut {
            width: w,
            height: h,
            crop_origin: new_origin,
            chroma_siting: self.chroma_siting,
            color: self.color.clone(),
            matrix_coefficients: self.matrix_coefficients,
            range: self.range,
            alpha: self.alpha,
            hdr: self.hdr.clone(),
            orientation: self.orientation,
            storage,
        })
    }

    /// Consume this mutable multi-plane view and narrow it to a rectangular crop retaining `'a`
    /// (zero heap allocation for `<= 4` planes).
    pub fn into_crop(self, rect: PixelRect) -> Result<PlanarViewMut<'a>, At<BufferError>> {
        if rect.width == 0 || rect.height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if rect
            .x
            .checked_add(rect.width)
            .is_none_or(|e| e > self.width)
            || rect
                .y
                .checked_add(rect.height)
                .is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let old_origin = self.crop_origin;
        let new_ox = old_origin
            .0
            .checked_add(rect.x)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let new_oy = old_origin
            .1
            .checked_add(rect.y)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let new_origin = (new_ox, new_oy);
        let w = rect.width;
        let h = rect.height;
        let storage = match self.storage {
            PlanarMutStorage::One([p0]) => {
                let r0 = plane_crop_rect(p0.spec, old_origin, new_origin, w, h)?;
                PlanarMutStorage::One([p0.into_crop(r0)?])
            }
            PlanarMutStorage::Two([p0, p1]) => {
                let r0 = plane_crop_rect(p0.spec, old_origin, new_origin, w, h)?;
                let r1 = plane_crop_rect(p1.spec, old_origin, new_origin, w, h)?;
                PlanarMutStorage::Two([p0.into_crop(r0)?, p1.into_crop(r1)?])
            }
            PlanarMutStorage::Three([p0, p1, p2]) => {
                let r0 = plane_crop_rect(p0.spec, old_origin, new_origin, w, h)?;
                let r1 = plane_crop_rect(p1.spec, old_origin, new_origin, w, h)?;
                let r2 = plane_crop_rect(p2.spec, old_origin, new_origin, w, h)?;
                PlanarMutStorage::Three([p0.into_crop(r0)?, p1.into_crop(r1)?, p2.into_crop(r2)?])
            }
            PlanarMutStorage::Four([p0, p1, p2, p3]) => {
                let r0 = plane_crop_rect(p0.spec, old_origin, new_origin, w, h)?;
                let r1 = plane_crop_rect(p1.spec, old_origin, new_origin, w, h)?;
                let r2 = plane_crop_rect(p2.spec, old_origin, new_origin, w, h)?;
                let r3 = plane_crop_rect(p3.spec, old_origin, new_origin, w, h)?;
                PlanarMutStorage::Four([
                    p0.into_crop(r0)?,
                    p1.into_crop(r1)?,
                    p2.into_crop(r2)?,
                    p3.into_crop(r3)?,
                ])
            }
            PlanarMutStorage::Spill(v) => {
                let mut out = Vec::new();
                out.try_reserve_exact(v.len())
                    .map_err(|_| err_at(BufferError::AllocationFailed))?;
                for p in v {
                    let r = plane_crop_rect(p.spec, old_origin, new_origin, w, h)?;
                    out.push(p.into_crop(r)?);
                }
                PlanarMutStorage::Spill(out)
            }
        };
        Ok(PlanarViewMut {
            width: w,
            height: h,
            crop_origin: new_origin,
            chroma_siting: self.chroma_siting,
            color: self.color,
            matrix_coefficients: self.matrix_coefficients,
            range: self.range,
            alpha: self.alpha,
            hdr: self.hdr,
            orientation: self.orientation,
            storage,
        })
    }

    /// Zero-copy mutable full-width horizontal strip of all planes preserving parent-frame chroma
    /// phase.
    #[inline]
    pub fn sub_rows_mut(
        &mut self,
        start_y: u32,
        row_count: u32,
    ) -> Result<PlanarViewMut<'_>, At<BufferError>> {
        self.crop_mut(PixelRect::new(0, start_y, self.width, row_count))
    }

    /// Consume this mutable multi-plane view and narrow it to a full-width horizontal strip
    /// retaining `'a`.
    #[inline]
    pub fn into_sub_rows(
        self,
        start_y: u32,
        row_count: u32,
    ) -> Result<PlanarViewMut<'a>, At<BufferError>> {
        let width = self.width;
        self.into_crop(PixelRect::new(0, start_y, width, row_count))
    }
}

impl PlanarBuf {
    /// Borrow all planes as a [`PlanarView`] (zero-allocation for `<= 4` planes).
    #[must_use]
    pub fn as_view(&self) -> PlanarView<'_> {
        self.try_as_view()
            .expect("PlanarBuf::as_view allocation failed for >4 planes")
    }

    /// Fallibly borrow all planes as a [`PlanarView`] (zero-allocation for `<= 4` planes).
    pub fn try_as_view(&self) -> Result<PlanarView<'_>, At<BufferError>> {
        let storage = if self.planes.len() <= 4 {
            let mut arr = [PlaneView::EMPTY; 4];
            for (i, p) in self.planes.iter().enumerate() {
                arr[i] = p.as_view();
            }
            PlanarStorage::Inline {
                planes: arr,
                len: self.planes.len(),
            }
        } else {
            let mut v = Vec::new();
            v.try_reserve_exact(self.planes.len())
                .map_err(|_| err_at(BufferError::AllocationFailed))?;
            for p in &self.planes {
                v.push(p.as_view());
            }
            PlanarStorage::Spill(v)
        };
        Ok(PlanarView {
            width: self.width,
            height: self.height,
            crop_origin: (0, 0),
            chroma_siting: self.chroma_siting,
            color: self.color.clone(),
            matrix_coefficients: self.matrix_coefficients,
            range: self.range,
            alpha: self.alpha,
            hdr: self.hdr.clone(),
            origin: Some(origin_ref(&self.origin)),
            orientation: self.orientation,
            storage,
        })
    }

    /// Borrow a zero-copy rectangular crop of all planes preserving exact parent-frame chroma
    /// phase (zero heap allocation for `<= 4` planes).
    #[inline]
    pub fn crop_view(&self, rect: PixelRect) -> Result<PlanarView<'_>, At<BufferError>> {
        if rect.width == 0 || rect.height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if rect
            .x
            .checked_add(rect.width)
            .is_none_or(|e| e > self.width)
            || rect
                .y
                .checked_add(rect.height)
                .is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        self.try_as_view()?.crop(rect)
    }

    /// Borrow all planes mutably as a [`PlanarViewMut`] (zero-allocation for `<= 4` planes).
    #[must_use]
    pub fn as_view_mut(&mut self) -> PlanarViewMut<'_> {
        self.try_as_view_mut()
            .expect("PlanarBuf::as_view_mut allocation failed")
    }

    /// Fallibly borrow all planes mutably as a [`PlanarViewMut`] (zero-allocation for `<= 4` planes).
    #[doc(alias = "cow")]
    #[doc(alias = "copy_on_write")]
    pub fn try_as_view_mut(&mut self) -> Result<PlanarViewMut<'_>, At<BufferError>> {
        let len = self.planes.len();
        let mut iter = self.planes.iter_mut();
        let storage = match len {
            1 => PlanarMutStorage::One([iter.next().unwrap().try_as_view_mut()?]),
            2 => PlanarMutStorage::Two([
                iter.next().unwrap().try_as_view_mut()?,
                iter.next().unwrap().try_as_view_mut()?,
            ]),
            3 => PlanarMutStorage::Three([
                iter.next().unwrap().try_as_view_mut()?,
                iter.next().unwrap().try_as_view_mut()?,
                iter.next().unwrap().try_as_view_mut()?,
            ]),
            4 => PlanarMutStorage::Four([
                iter.next().unwrap().try_as_view_mut()?,
                iter.next().unwrap().try_as_view_mut()?,
                iter.next().unwrap().try_as_view_mut()?,
                iter.next().unwrap().try_as_view_mut()?,
            ]),
            n => {
                let mut v = Vec::new();
                v.try_reserve_exact(n)
                    .map_err(|_| err_at(BufferError::AllocationFailed))?;
                for p in iter {
                    v.push(p.try_as_view_mut()?);
                }
                PlanarMutStorage::Spill(v)
            }
        };
        Ok(PlanarViewMut {
            width: self.width,
            height: self.height,
            crop_origin: (0, 0),
            chroma_siting: self.chroma_siting,
            color: self.color.clone(),
            matrix_coefficients: self.matrix_coefficients,
            range: self.range,
            alpha: self.alpha,
            hdr: self.hdr.clone(),
            orientation: self.orientation,
            storage,
        })
    }

    /// Borrow a zero-copy mutable rectangular crop of all planes preserving exact parent-frame
    /// chroma phase, fallibly copy-on-write promoting immutable external planes if needed
    /// (zero heap allocation for `<= 4` planes).
    #[inline]
    pub fn crop_view_mut(&mut self, rect: PixelRect) -> Result<PlanarViewMut<'_>, At<BufferError>> {
        if rect.width == 0 || rect.height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if rect
            .x
            .checked_add(rect.width)
            .is_none_or(|e| e > self.width)
            || rect
                .y
                .checked_add(rect.height)
                .is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        self.try_as_view_mut()?.into_crop(rect)
    }
}

// ---------------------------------------------------------------------------
// DualLayerHdrBuf & DualLayerHdrView (ISO 21496-1 / UltraHDR interchange)
// ---------------------------------------------------------------------------

fn validate_dual_layer_specs(
    base_spec: &PixelFormat,
    gain_map_spec: &PixelFormat,
    metadata: GainMapMetadata,
    alternate_color_space: Option<&crate::ColorSpace>,
) -> Result<(), At<BufferError>> {
    // Base may be RGB, Gray, or a 1/3-channel ICC-described image.
    let base_ok = match &base_spec.color {
        crate::ColorSpace::Rgb { .. } | crate::ColorSpace::Gray { .. } => true,
        crate::ColorSpace::Icc { channels, .. } => matches!(*channels, 1 | 3),
        _ => false,
    };
    if !base_ok {
        return Err(err_at(BufferError::IncompatibleLayout));
    }

    // Gain map MUST be numeric (Untagged, 1 or 3 channels, no alpha) with Full/Lsb/Msb U8/U16/F32.
    let gm_layout = gain_map_spec.layout;
    let gm_color_ok = matches!(
        gain_map_spec.color,
        crate::ColorSpace::Untagged {
            channels: 1 | 3,
            alpha: crate::AlphaMode::None,
        }
    );
    let gm_layout_ok = matches!(
        gm_layout.packing,
        BitPacking::Full | BitPacking::Lsb { .. } | BitPacking::Msb { .. }
    ) && matches!(
        gm_layout.sample,
        SampleType::U8 | SampleType::U16 | SampleType::F32
    ) && matches!(
        gm_layout.order,
        crate::ChannelOrder::Gray
            | crate::ChannelOrder::Rgb
            | crate::ChannelOrder::Generic(1)
            | crate::ChannelOrder::Generic(3)
    );
    if !gm_color_ok || !gm_layout_ok {
        return Err(err_at(BufferError::IncompatibleLayout));
    }

    if !metadata.is_valid() {
        return Err(err_at(BufferError::InvalidGainMapMetadata));
    }

    if metadata.application_space == crate::GainMapColorSpace::Alternate
        && alternate_color_space.is_none()
    {
        return Err(err_at(BufferError::IncompatibleLayout));
    }

    if let Some(alt) = alternate_color_space {
        let alt_ok = match alt {
            crate::ColorSpace::Rgb { .. } | crate::ColorSpace::Gray { .. } => true,
            crate::ColorSpace::Icc { channels, .. } => matches!(*channels, 1 | 3),
            _ => false,
        };
        if !alt_ok {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
    }
    Ok(())
}

/// Complete decomposed parts of a [`DualLayerHdrBuf`] carrier, preserving every field losslessly.
///
/// When modifying `orientation` before calling [`DualLayerHdrBuf::from_parts`], keep
/// `base.orientation()` and `gain_map.orientation()` either equal to `parts.orientation` or
/// reset to [`Orientation::Identity`] (for example, after [`DualLayerHdrBuf::into_parts`], prefer
/// [`DualLayerHdrBuf::with_orientation`] / [`DualLayerHdrBuf::compose_orientation`] for O(1) metadata
/// updates, or update `parts.base`, `parts.gain_map`, and `parts.orientation` together).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DualLayerHdrBufParts {
    /// Base rendition pixel buffer.
    pub base: PixelBuf,
    /// Numeric gain-map pixel buffer (`ColorSpace::Untagged`, 1 or 3 channels).
    pub gain_map: PixelBuf,
    /// Exact rational ISO 21496-1 metadata.
    pub metadata: GainMapMetadata,
    /// Optional alternate rendition color space.
    pub alternate_color_space: Option<crate::ColorSpace>,
    /// Shared pending EXIF orientation for the aligned pair.
    pub orientation: crate::Orientation,
}

/// Borrowed validated ISO 21496-1 / UltraHDR dual-layer HDR view (zero copy).
#[derive(Clone, Debug)]
pub struct DualLayerHdrView<'a> {
    base: PixelView<'a>,
    gain_map: PixelView<'a>,
    metadata: GainMapMetadata,
    alternate_color_space: Option<&'a crate::ColorSpace>,
    orientation: crate::Orientation,
}

impl<'a> DualLayerHdrView<'a> {
    /// Construct a validated borrowed dual-layer HDR view.
    ///
    /// Rejects misregistered layer orientations (`base.orientation() != gain_map.orientation()`)
    /// or a non-`Identity` layer orientation that contradicts `orientation`.
    pub fn new(
        mut base: PixelView<'a>,
        mut gain_map: PixelView<'a>,
        metadata: GainMapMetadata,
        alternate_color_space: Option<&'a crate::ColorSpace>,
        orientation: crate::Orientation,
    ) -> Result<Self, At<BufferError>> {
        if base.orientation() != gain_map.orientation() {
            return Err(err_at(BufferError::IncompatibleLayout).at_str(
                "DualLayerHdrView base and gain_map must have matching pending orientations",
            ));
        }
        if base.orientation() != crate::Orientation::Identity && base.orientation() != orientation {
            return Err(err_at(BufferError::IncompatibleLayout).at_str(
                "DualLayerHdrView non-Identity layer orientations must match the requested orientation",
            ));
        }
        validate_dual_layer_specs(
            base.format(),
            gain_map.format(),
            metadata,
            alternate_color_space,
        )?;
        base.orientation = orientation;
        gain_map.orientation = orientation;
        Ok(Self {
            base,
            gain_map,
            metadata,
            alternate_color_space,
            orientation,
        })
    }

    /// Borrowed base rendition view.
    #[inline]
    #[must_use]
    pub const fn base(&self) -> &PixelView<'a> {
        &self.base
    }

    /// Borrowed numeric gain-map view (`ColorSpace::Untagged`, 1 or 3 channels).
    #[inline]
    #[must_use]
    pub const fn gain_map(&self) -> &PixelView<'a> {
        &self.gain_map
    }

    /// Exact rational ISO 21496-1 reconstruction metadata.
    #[inline]
    #[must_use]
    pub const fn metadata(&self) -> GainMapMetadata {
        self.metadata
    }

    /// Alternate rendition color space, if declared (required when `application_space == Alternate`).
    #[inline]
    #[must_use]
    pub const fn alternate_color_space(&self) -> Option<&'a crate::ColorSpace> {
        self.alternate_color_space
    }

    /// Shared pending EXIF orientation for the aligned `(base, gain_map)` pair.
    #[inline]
    #[must_use]
    pub const fn orientation(&self) -> crate::Orientation {
        self.orientation
    }

    /// Replace the shared pending EXIF orientation on this borrowed view.
    #[inline]
    #[must_use]
    pub const fn with_orientation(mut self, orientation: crate::Orientation) -> Self {
        self.orientation = orientation;
        self.base.orientation = orientation;
        self.gain_map.orientation = orientation;
        self
    }

    /// Compose an additional display transform (including reflections) onto the shared pending
    /// EXIF orientation (`self.orientation.then(delta)`) in O(1) without touching or copying
    /// either pixel layer.
    #[inline]
    #[must_use]
    pub const fn compose_orientation(mut self, delta: crate::Orientation) -> Self {
        let next = self.orientation.then(delta);
        self.orientation = next;
        self.base.orientation = next;
        self.gain_map.orientation = next;
        self
    }

    /// Display-oriented `(width, height)` of the base image after applying `self.orientation()`.
    #[inline]
    #[must_use]
    pub const fn display_dimensions(&self) -> (u32, u32) {
        self.orientation
            .output_dimensions(self.base.width(), self.base.height())
    }
}

/// Container-agnostic dual-layer HDR image (base + numeric gain map + exact ISO 21496-1 metadata).
///
/// Unifies UltraHDR JPEG, AVIF `tmap`, JXL `jhgm`, and PNG gain-map HDR in one validated core type.
#[derive(Clone, Debug)]
pub struct DualLayerHdrBuf {
    base: PixelBuf,
    gain_map: PixelBuf,
    metadata: GainMapMetadata,
    alternate_color_space: Option<Box<crate::ColorSpace>>,
    orientation: crate::Orientation,
}

impl DualLayerHdrBuf {
    /// Construct a validated dual-layer HDR carrier with `GainMapColorSpace::Base`.
    ///
    /// Validates base color model (`Rgb`/`Gray`/1-or-3-ch `Icc`), numeric gain-map layout
    /// (`ColorSpace::Untagged`, 1 or 3 channels, `U8`/`U16`/`F32` with `Full`/`Lsb`/`Msb` packing),
    /// and exact rational ISO 21496-1 metadata (`gamma > 0`, `min <= max`, `H_base != H_alt`).
    /// Inherits pending [`Orientation`] from `base` (requiring `base.orientation() == gain_map.orientation()`).
    ///
    /// Rejects `metadata.application_space == GainMapColorSpace::Alternate` because no
    /// alternate color space is supplied — use
    /// [`new_with_alternate_color`](Self::new_with_alternate_color) when
    /// `application_space == Alternate`.
    pub fn new(
        base: PixelBuf,
        gain_map: PixelBuf,
        metadata: GainMapMetadata,
    ) -> Result<Self, At<BufferError>> {
        let orientation = base.orientation();
        Self::from_layers(base, gain_map, metadata, None, orientation)
    }

    /// Construct a validated dual-layer HDR carrier with an explicit alternate [`ColorSpace`]
    /// (required when `metadata.application_space == GainMapColorSpace::Alternate`).
    ///
    /// Inherits pending [`Orientation`] from `base` (rejecting misregistered
    /// `base.orientation() != gain_map.orientation()`). To attach or replace orientation metadata
    /// afterward, chain [`with_orientation`](Self::with_orientation).
    pub fn new_with_alternate_color(
        base: PixelBuf,
        gain_map: PixelBuf,
        metadata: GainMapMetadata,
        alternate_color_space: crate::ColorSpace,
    ) -> Result<Self, At<BufferError>> {
        let orientation = base.orientation();
        Self::from_layers(
            base,
            gain_map,
            metadata,
            Some(alternate_color_space),
            orientation,
        )
    }

    fn from_layers(
        mut base: PixelBuf,
        mut gain_map: PixelBuf,
        metadata: GainMapMetadata,
        alternate_color_space: Option<crate::ColorSpace>,
        orientation: crate::Orientation,
    ) -> Result<Self, At<BufferError>> {
        if base.orientation() != gain_map.orientation() {
            return Err(err_at(BufferError::IncompatibleLayout).at_str(
                "DualLayerHdrBuf base and gain_map must have matching pending orientations",
            ));
        }
        if base.orientation() != crate::Orientation::Identity && base.orientation() != orientation {
            return Err(err_at(BufferError::IncompatibleLayout).at_str(
                "DualLayerHdrBuf non-Identity layer orientations must match the requested orientation",
            ));
        }
        validate_dual_layer_specs(
            base.format(),
            gain_map.format(),
            metadata,
            alternate_color_space.as_ref(),
        )?;
        base.orientation = orientation;
        gain_map.orientation = orientation;
        Ok(Self {
            base,
            gain_map,
            metadata,
            alternate_color_space: alternate_color_space.map(Box::new),
            orientation,
        })
    }

    /// Reconstruct a validated [`DualLayerHdrBuf`] from [`DualLayerHdrBufParts`].
    ///
    /// Requires `parts.base.orientation() == parts.gain_map.orientation()`, and if non-`Identity`,
    /// that they match `parts.orientation`.
    pub fn from_parts(parts: DualLayerHdrBufParts) -> Result<Self, At<BufferError>> {
        Self::from_layers(
            parts.base,
            parts.gain_map,
            parts.metadata,
            parts.alternate_color_space,
            parts.orientation,
        )
    }

    /// Attach a validated alternate rendition color space (required when `application_space == Alternate`).
    pub fn with_alternate_color_space(
        mut self,
        alt: crate::ColorSpace,
    ) -> Result<Self, At<BufferError>> {
        validate_dual_layer_specs(
            self.base.format(),
            self.gain_map.format(),
            self.metadata,
            Some(&alt),
        )?;
        self.alternate_color_space = Some(Box::new(alt));
        Ok(self)
    }

    /// Attach a shared pending [`Orientation`] for the aligned pair.
    #[inline]
    #[must_use]
    pub const fn with_orientation(mut self, orientation: crate::Orientation) -> Self {
        self.orientation = orientation;
        self.base.orientation = orientation;
        self.gain_map.orientation = orientation;
        self
    }

    /// Compose an additional display transform (including reflections) onto the shared pending
    /// EXIF orientation (`self.orientation.then(delta)`) in O(1) without touching or rebaking
    /// either pixel layer.
    ///
    /// Keeps `base` and `gain_map` strictly registered in stored-frame coordinates for
    /// lossless EXIF passthrough workflows. To physically bake orientation into both pixel
    /// buffers, use `zenpixels_convert::gainmap::{bake_dual_layer_orientation, bake_dual_layer_orientation_after}`.
    #[inline]
    #[must_use]
    pub const fn compose_orientation(mut self, delta: crate::Orientation) -> Self {
        let next = self.orientation.then(delta);
        self.orientation = next;
        self.base.orientation = next;
        self.gain_map.orientation = next;
        self
    }

    /// Borrow as a zero-copy [`DualLayerHdrView`].
    #[inline]
    #[must_use]
    pub fn as_view(&self) -> DualLayerHdrView<'_> {
        DualLayerHdrView {
            base: self.base.as_view(),
            gain_map: self.gain_map.as_view(),
            metadata: self.metadata,
            alternate_color_space: self.alternate_color_space.as_deref(),
            orientation: self.orientation,
        }
    }

    /// Base rendition buffer.
    #[inline]
    #[must_use]
    pub fn base(&self) -> &PixelBuf {
        &self.base
    }

    /// Numeric gain-map buffer (`ColorSpace::Untagged`, 1 or 3 channels).
    #[inline]
    #[must_use]
    pub fn gain_map(&self) -> &PixelBuf {
        &self.gain_map
    }

    /// Validated exact rational ISO 21496-1 reconstruction metadata.
    #[inline]
    #[must_use]
    pub const fn metadata(&self) -> GainMapMetadata {
        self.metadata
    }

    /// Alternate rendition color space, if declared.
    #[inline]
    #[must_use]
    pub fn alternate_color_space(&self) -> Option<&crate::ColorSpace> {
        self.alternate_color_space.as_deref()
    }

    /// Shared pending EXIF orientation for the aligned pair.
    #[inline]
    #[must_use]
    pub const fn orientation(&self) -> crate::Orientation {
        self.orientation
    }

    /// Display-oriented `(width, height)` of the base image after applying `self.orientation()`.
    #[inline]
    #[must_use]
    pub const fn display_dimensions(&self) -> (u32, u32) {
        self.orientation
            .output_dimensions(self.base.width(), self.base.height())
    }

    /// Decompose into complete owned [`DualLayerHdrBufParts`], preserving `alternate_color_space`
    /// and pending `orientation` losslessly.
    #[inline]
    #[must_use]
    pub fn into_parts(self) -> DualLayerHdrBufParts {
        DualLayerHdrBufParts {
            base: self.base,
            gain_map: self.gain_map,
            metadata: self.metadata,
            alternate_color_space: self.alternate_color_space.map(|b| *b),
            orientation: self.orientation,
        }
    }
}

// ---------------------------------------------------------------------------
// Unified PixelSource / PixelSink (views, buffers, crops, and bands)
// ---------------------------------------------------------------------------

/// Uniform read interface across [`PixelView`], [`PixelBuf`], and [`BandedBuf`].
pub trait PixelSource {
    /// Frame width in pixels.
    fn source_width(&self) -> u32;
    /// Frame height in rows.
    fn source_height(&self) -> u32;
    /// Authoritative image specification.
    fn source_format(&self) -> &PixelFormat;
    /// Proven sample facts.
    fn source_facts(&self) -> SampleFacts;
    /// Descriptive HDR metadata.
    fn source_hdr(&self) -> HdrMetadata;
    /// Original container color provenance, if attached.
    fn source_origin(&self) -> Option<&ColorOrigin> {
        None
    }
    /// Pending EXIF display orientation (defaults to [`Orientation::Identity`]).
    fn source_orientation(&self) -> Orientation {
        Orientation::Identity
    }
    /// Active bytes of row `y`.
    fn source_row(&self, y: u32) -> &[u8];
}

/// Uniform write interface across [`PixelViewMut`], [`PixelBuf`], and [`BandedBuf`].
pub trait PixelSink {
    /// Frame width in pixels.
    fn sink_width(&self) -> u32;
    /// Frame height in rows.
    fn sink_height(&self) -> u32;
    /// Authoritative destination specification.
    fn sink_format(&self) -> &PixelFormat;
    /// Mutable active bytes of row `y`.
    fn sink_row_mut(&mut self, y: u32) -> &mut [u8];
    /// Propagate proven output facts & HDR metadata on owned sinks (no-op on borrowed views).
    fn sink_finish(&mut self, _facts: SampleFacts, _hdr: HdrMetadata) {}
    /// Propagate original container color provenance onto owned sinks (no-op on borrowed views).
    fn sink_set_origin(&mut self, _origin: ColorOrigin) {}
    /// Propagate pending EXIF display orientation onto sinks.
    fn sink_set_orientation(&mut self, _orientation: Orientation) {}
}

impl PixelSource for PixelView<'_> {
    #[inline]
    fn source_width(&self) -> u32 {
        self.width
    }
    #[inline]
    fn source_height(&self) -> u32 {
        self.height
    }
    #[inline]
    fn source_format(&self) -> &PixelFormat {
        self.spec.as_format()
    }
    #[inline]
    fn source_facts(&self) -> SampleFacts {
        self.facts
    }
    #[inline]
    fn source_hdr(&self) -> HdrMetadata {
        unbox_hdr(&self.hdr)
    }
    #[inline]
    fn source_origin(&self) -> Option<&ColorOrigin> {
        self.origin
    }
    #[inline]
    fn source_orientation(&self) -> Orientation {
        self.orientation
    }
    #[inline]
    fn source_row(&self, y: u32) -> &[u8] {
        self.row(y)
    }
}

impl PixelSource for PixelBuf {
    #[inline]
    fn source_width(&self) -> u32 {
        self.width
    }
    #[inline]
    fn source_height(&self) -> u32 {
        self.height
    }
    #[inline]
    fn source_format(&self) -> &PixelFormat {
        &self.spec
    }
    #[inline]
    fn source_facts(&self) -> SampleFacts {
        self.facts
    }
    #[inline]
    fn source_hdr(&self) -> HdrMetadata {
        unbox_hdr(&self.hdr)
    }
    #[inline]
    fn source_origin(&self) -> Option<&ColorOrigin> {
        Some(origin_ref(&self.origin))
    }
    #[inline]
    fn source_orientation(&self) -> Orientation {
        self.orientation
    }
    #[inline]
    fn source_row(&self, y: u32) -> &[u8] {
        assert!(y < self.height, "row {y} out of bounds ({})", self.height);
        let start = self.offset + (y as usize) * self.stride;
        &self.storage.as_bytes()[start..start + self.row_bytes()]
    }
}

impl PixelSource for BandedBuf {
    #[inline]
    fn source_width(&self) -> u32 {
        self.width
    }
    #[inline]
    fn source_height(&self) -> u32 {
        self.height
    }
    #[inline]
    fn source_format(&self) -> &PixelFormat {
        &self.spec
    }
    #[inline]
    fn source_facts(&self) -> SampleFacts {
        self.facts
    }
    #[inline]
    fn source_hdr(&self) -> HdrMetadata {
        unbox_hdr(&self.hdr)
    }
    #[inline]
    fn source_origin(&self) -> Option<&ColorOrigin> {
        Some(origin_ref(&self.origin))
    }
    #[inline]
    fn source_orientation(&self) -> Orientation {
        self.orientation
    }
    #[inline]
    fn source_row(&self, y: u32) -> &[u8] {
        self.row(y)
    }
}

impl PixelSink for PixelViewMut<'_> {
    #[inline]
    fn sink_width(&self) -> u32 {
        self.width
    }
    #[inline]
    fn sink_height(&self) -> u32 {
        self.height
    }
    #[inline]
    fn sink_format(&self) -> &PixelFormat {
        self.spec.as_format()
    }
    #[inline]
    fn sink_row_mut(&mut self, y: u32) -> &mut [u8] {
        self.row_mut(y)
    }
    #[inline]
    fn sink_set_orientation(&mut self, orientation: Orientation) {
        self.orientation = orientation;
    }
}

impl PixelSink for PixelBuf {
    #[inline]
    fn sink_width(&self) -> u32 {
        self.width
    }
    #[inline]
    fn sink_height(&self) -> u32 {
        self.height
    }
    #[inline]
    fn sink_format(&self) -> &PixelFormat {
        &self.spec
    }
    #[inline]
    fn sink_row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(y < self.height, "row {y} out of bounds ({})", self.height);
        self.make_mut()
            .expect("copy-on-write promotion of external PixelBuf failed");
        self.facts = SampleFacts::NONE;
        let rb = self.row_bytes();
        let start = self.offset + (y as usize) * self.stride;
        &mut self.storage.as_bytes_mut()[start..start + rb]
    }
    #[inline]
    fn sink_finish(&mut self, facts: SampleFacts, hdr: HdrMetadata) {
        self.facts = facts;
        self.hdr = box_hdr(hdr);
    }
    #[inline]
    fn sink_set_origin(&mut self, origin: ColorOrigin) {
        self.origin = box_origin(origin);
    }
    #[inline]
    fn sink_set_orientation(&mut self, orientation: Orientation) {
        self.orientation = orientation;
    }
}

impl PixelSink for BandedBuf {
    #[inline]
    fn sink_width(&self) -> u32 {
        self.width
    }
    #[inline]
    fn sink_height(&self) -> u32 {
        self.height
    }
    #[inline]
    fn sink_format(&self) -> &PixelFormat {
        &self.spec
    }
    #[inline]
    fn sink_row_mut(&mut self, y: u32) -> &mut [u8] {
        self.row_mut(y)
    }
    #[inline]
    fn sink_finish(&mut self, facts: SampleFacts, hdr: HdrMetadata) {
        let boxed = box_hdr(hdr);
        self.facts = facts;
        self.hdr = boxed.clone();
        for b in &mut self.bands {
            b.facts = facts;
            b.hdr = boxed.clone();
        }
    }
    #[inline]
    fn sink_set_origin(&mut self, origin: ColorOrigin) {
        let boxed = box_origin(origin);
        self.origin = boxed.clone();
        for b in &mut self.bands {
            b.origin = boxed.clone();
        }
    }
    #[inline]
    fn sink_set_orientation(&mut self, orientation: Orientation) {
        self.orientation = orientation;
        for b in &mut self.bands {
            b.orientation = orientation;
        }
    }
}
