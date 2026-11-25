//! Unparameterized pixel buffers, zero-copy strided views, bands, planes, and dual-layer HDR.
//!
//! - [`PixelBuf`] owns an alignment-exact [`PixelStorage`], optional [`HdrMetadata`],
//!   [`ColorOrigin`], and cached [`SampleFacts`] (cleared automatically on `&mut`).
//! - [`PixelView<'a>`] and [`PixelViewMut<'a>`] borrow rectangular (possibly strided
//!   or cropped) subimages with lifetimes tied directly to `'a` backing storage.
//! - Zero-copy typed pixel access is provided by methods (`row_as::<Rgb<u8>>`,
//!   `from_vec`, `try_into_vec_zero_copy`) rather than a `<P = ()>` type parameter.
//! - [`BandedBuf`] handles gigapixel images in bounded-memory horizontal bands.
//! - [`PlanarBuf`] / [`PlanarView`] handle 1..=4 planes (YCbCr, NV12, P010, planar RGB+A)
//!   including N-bit [`BitPacking`] code extraction.
//! - [`DualLayerHdr`] carries ISO 21496-1 / UltraHDR base + gain map + headroom metadata.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt;
use whereat::At;

use crate::color::{
    AlphaMode, ColorError, ColorOrigin, ColorSpace, GainMapMetadata, HdrMetadata, ImageSpec,
    SignalRange, SpecCow,
};
use crate::facts::{FloatFacts, SampleFacts};
use crate::layout::{
    BitPacking, ChannelOrder, InteropTarget, LayoutError, PlaneRole, PlaneSpec, RowAlign,
    SampleLayout, SampleType,
};
use crate::orientation::Orientation;
use crate::pixel_types::{F16Sample, GrayAlpha8, GrayAlpha16, GrayAlphaF16, GrayAlphaF32};
use crate::sample::ChromaSiting;
use crate::storage::{
    AlignedSamples, ByteStorage, ByteStorageMut, PixelStorage, StorageError, StorageSample,
};

mod sealed {
    pub trait Sealed {}
}

/// Sealed marker trait for Rust scalar, channel-array, or pixel types that carry a
/// known [`SampleLayout`] compatibility contract.
///
/// Used by [`PixelBuf::from_vec`] and [`PixelView::from_typed`] to prevent silent
/// native-endian reinterpretation across unrelated scalar types (e.g. `Vec<i32>`
/// adopted as `RGBF32`). For intentional raw byte reinterpretation, use
/// [`PixelBuf::from_raw_pod`] or [`PixelView::from_raw_pod`].
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
    InvalidSpec(ColorError),
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
        Self::InvalidSpec(e)
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
            Self::InvalidSpec(e) => e.fmt(f),
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
    spec: &ImageSpec,
) -> Result<(usize, usize), At<BufferError>> {
    spec.validate()
        .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
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
fn fill_opaque_alpha(bytes: &mut [u8], width: u32, height: u32, stride: usize, spec: &ImageSpec) {
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
    let tmp_spec = ImageSpec::new(
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
fn fill_white_pixels(bytes: &mut [u8], width: u32, height: u32, stride: usize, spec: &ImageSpec) {
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

/// Zero-copy borrowed view of an interleaved image or rectangular crop.
///
/// Every slicing method (`row`, `row_as`, `crop`, `sub_rows`,
/// `as_contiguous_bytes`) returns references or subviews tied directly to `'a`
/// rather than `&self`, so chaining (`buf.as_view().crop(...)?.row(0)`) works cleanly.
#[derive(Clone, Debug)]
pub struct PixelView<'a> {
    data: &'a [u8],
    width: u32,
    rows: u32,
    stride: usize,
    spec: SpecCow<'a>,
    facts: SampleFacts,
    hdr: HdrMetadata,
    origin: Option<&'a ColorOrigin>,
    orientation: Orientation,
}

impl<'a> PixelView<'a> {
    /// Borrow strided bytes with checked dimensions, alignment, and length.
    ///
    /// Accepts owned [`ImageSpec`], borrowed `&'a ImageSpec` (zero `Arc` refcount bump),
    /// or `Copy` [`PixelFormat`](crate::PixelFormat).
    ///
    /// Trims any trailing bytes beyond the active `(rows - 1) * stride + row_bytes`
    /// extent so zero-copy interop bridges (`try_as_imgref`) never touch trailing slop.
    pub fn new(
        data: &'a [u8],
        width: u32,
        rows: u32,
        stride: usize,
        spec: impl Into<SpecCow<'a>>,
    ) -> Result<Self, At<BufferError>> {
        let spec = spec.into();
        let (_rb, req) = validate_geometry(
            data.len(),
            data.as_ptr(),
            width,
            rows,
            stride,
            spec.as_spec(),
        )?;
        Ok(Self {
            data: &data[..req],
            width,
            rows,
            stride,
            spec,
            facts: SampleFacts::NONE,
            hdr: HdrMetadata::default(),
            origin: None,
            orientation: Orientation::Identity,
        })
    }

    /// Borrow tightly packed bytes (`stride == width * bytes_per_pixel`).
    pub fn from_packed(
        data: &'a [u8],
        width: u32,
        rows: u32,
        spec: impl Into<SpecCow<'a>>,
    ) -> Result<Self, At<BufferError>> {
        let spec = spec.into();
        let stride = (width as usize)
            .checked_mul(spec.as_spec().layout.bytes_per_pixel())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        Self::new(data, width, rows, stride, spec)
    }

    /// Borrow a tightly packed typed pixel or sample slice zero-copy via `bytemuck`.
    ///
    /// Validates that `P` is structurally compatible with `spec.layout` (rejecting
    /// e.g. `&[i32]` as `RGBF32`). For intentional raw byte reinterpretation, use
    /// [`from_raw_pod`](Self::from_raw_pod).
    pub fn from_typed<P: TypedElement>(
        pixels: &'a [P],
        width: u32,
        rows: u32,
        spec: impl Into<SpecCow<'a>>,
    ) -> Result<Self, At<BufferError>> {
        let spec = spec.into();
        if !P::matches_layout(spec.as_spec().layout) {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        Self::from_packed(bytemuck::cast_slice(pixels), width, rows, spec)
    }

    /// Borrow a tightly packed scalar sample slice (`&[u8]`, `&[u16]`, `&[f32]`, etc.) zero-copy.
    pub fn from_samples<S: StorageSample>(
        samples: &'a [S],
        width: u32,
        rows: u32,
        spec: impl Into<SpecCow<'a>>,
    ) -> Result<Self, At<BufferError>> {
        let spec = spec.into();
        if spec.as_spec().layout.sample != S::SAMPLE_TYPE {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        Self::from_packed(bytemuck::cast_slice(samples), width, rows, spec)
    }

    /// Borrow a tightly packed `rgb` pixel slice (`&[rgb::Rgb<u8>]`, etc.) zero-copy.
    #[cfg(feature = "rgb")]
    pub fn from_pixels<P: crate::interop::PixelType>(
        pixels: &'a [P],
        width: u32,
        rows: u32,
        color: crate::ColorSpace,
    ) -> Result<Self, At<BufferError>> {
        Self::from_packed(
            bytemuck::cast_slice(pixels),
            width,
            rows,
            ImageSpec::new(P::LAYOUT, color),
        )
    }

    /// Explicitly reinterpret any tightly packed `bytemuck::Pod` slice in native endianness
    /// as raw pixel bytes matching `spec`.
    pub fn from_raw_pod<P: bytemuck::Pod>(
        pixels: &'a [P],
        width: u32,
        rows: u32,
        spec: impl Into<SpecCow<'a>>,
    ) -> Result<Self, At<BufferError>> {
        Self::from_packed(bytemuck::cast_slice(pixels), width, rows, spec)
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
    pub const fn rows(&self) -> u32 {
        self.rows
    }

    /// Byte distance between consecutive row starts.
    #[inline]
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Active bytes per row (`width * bytes_per_pixel`).
    #[inline]
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        (self.width as usize) * self.spec.as_spec().layout.bytes_per_pixel()
    }

    /// Authoritative image specification (`layout` + `color`).
    #[inline]
    #[must_use]
    pub const fn spec(&self) -> &ImageSpec {
        self.spec.as_spec()
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
    pub const fn hdr(&self) -> HdrMetadata {
        self.hdr
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
        self.orientation.output_dimensions(self.width, self.rows)
    }

    /// Attach proven sample facts to this view.
    #[inline]
    #[must_use]
    pub fn with_facts(mut self, facts: SampleFacts) -> Self {
        self.facts = facts;
        self
    }

    /// Attach descriptive HDR metadata to this view.
    #[inline]
    #[must_use]
    pub fn with_hdr(mut self, hdr: HdrMetadata) -> Self {
        self.hdr = hdr;
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

    /// Compose an additional display transform onto this view's pending orientation
    /// (`self.orientation.then(delta)`) in O(1) without touching pixels.
    #[inline]
    #[must_use]
    pub const fn rotate_metadata(mut self, delta: Orientation) -> Self {
        self.orientation = self.orientation.then(delta);
        self
    }

    /// Active bytes of row `y` (excludes inter-row padding), tied to backing lifetime `'a`.
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.rows()`.
    #[inline]
    #[must_use]
    pub fn row(&self, y: u32) -> &'a [u8] {
        assert!(y < self.rows, "row {y} out of bounds (rows: {})", self.rows);
        let start = (y as usize) * self.stride;
        &self.data[start..start + self.row_bytes()]
    }

    /// Zero-copy typed slice view of row `y` (e.g. `view.row_as::<rgb::Rgb<u8>>(y)`).
    ///
    /// Validates `bytemuck::try_cast_slice` alignment and exact element divisibility.
    pub fn row_as<P: bytemuck::Pod>(&self, y: u32) -> Result<&'a [P], At<BufferError>> {
        if y >= self.rows {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let bytes = self.row(y);
        bytemuck::try_cast_slice(bytes).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
        })
    }

    /// Raw strided byte backing starting at `(0, 0)` (includes inter-row padding).
    #[inline]
    #[must_use]
    pub const fn as_strided_bytes(&self) -> &'a [u8] {
        self.data
    }

    /// Whether rows are tightly packed (`stride == row_bytes` or `rows <= 1`).
    #[inline]
    #[must_use]
    pub const fn is_contiguous(&self) -> bool {
        self.rows <= 1 || self.stride == self.row_bytes()
    }

    /// Contiguous packed bytes across all rows, or `None` if padded.
    #[inline]
    #[must_use]
    pub fn as_contiguous_bytes(&self) -> Option<&'a [u8]> {
        if self.is_contiguous() {
            let total = self.row_bytes() * (self.rows as usize);
            Some(&self.data[..total])
        } else {
            None
        }
    }

    /// Zero-copy contiguous typed slice across all rows (e.g. `&[rgb::Rgba<f32>]`).
    pub fn as_contiguous_as<P: bytemuck::Pod>(&self) -> Result<&'a [P], At<BufferError>> {
        let bytes = self
            .as_contiguous_bytes()
            .ok_or_else(|| err_at(BufferError::StrideNotPixelAligned))?;
        bytemuck::try_cast_slice(bytes).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
        })
    }

    /// Borrow contiguous bytes if tightly packed, or fallibly copy active rows into a `Vec<u8>`.
    pub fn try_contiguous_bytes(&self) -> Result<Cow<'a, [u8]>, At<BufferError>> {
        if let Some(bytes) = self.as_contiguous_bytes() {
            return Ok(Cow::Borrowed(bytes));
        }
        let rb = self.row_bytes();
        let total = rb
            .checked_mul(self.rows as usize)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let mut out = Vec::new();
        out.try_reserve_exact(total)
            .map_err(|_| err_at(BufferError::AllocationFailed))?;
        for y in 0..self.rows {
            out.extend_from_slice(self.row(y));
        }
        Ok(Cow::Owned(out))
    }

    /// Zero-copy rectangular subimage view retaining parent stride and `'a` lifetime.
    pub fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Result<PixelView<'a>, At<BufferError>> {
        if w == 0 || h == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if x.checked_add(w).is_none_or(|e| e > self.width)
            || y.checked_add(h).is_none_or(|e| e > self.rows)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let bpp = self.spec.layout.bytes_per_pixel();
        let start = (y as usize) * self.stride + (x as usize) * bpp;
        let end = (y as usize + h as usize - 1) * self.stride + (x as usize + w as usize) * bpp;
        Ok(PixelView {
            data: &self.data[start..end],
            width: w,
            rows: h,
            stride: self.stride,
            spec: self.spec.clone(),
            facts: self.facts,
            hdr: self.hdr,
            origin: self.origin,
            orientation: self.orientation,
        })
    }

    /// Zero-copy full-width horizontal strip view retaining `'a` lifetime.
    pub fn sub_rows(&self, y: u32, count: u32) -> Result<PixelView<'a>, At<BufferError>> {
        self.crop(0, y, self.width, count)
    }

    /// Fallibly copy active pixels into a freshly allocated, tightly packed [`PixelBuf`].
    pub fn try_to_tight_buf(&self) -> Result<PixelBuf, At<BufferError>> {
        let rb = self.row_bytes();
        let total = rb
            .checked_mul(self.rows as usize)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let storage = PixelStorage::try_from_rows(
            self.spec.layout.sample,
            total,
            (0..self.rows).map(|y| self.row(y)),
        )
        .map_err(|e| err_at(e.into()))?;
        Ok(PixelBuf {
            storage,
            offset: 0,
            width: self.width,
            height: self.rows,
            stride: rb,
            spec: self.spec.as_spec().clone(),
            facts: self.facts,
            hdr: self.hdr,
            origin: self.origin.cloned().unwrap_or_else(ColorOrigin::assumed),
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
        for y in 0..self.rows {
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
    rows: u32,
    stride: usize,
    spec: SpecCow<'a>,
    hdr: HdrMetadata,
    orientation: Orientation,
}

impl<'a> PixelViewMut<'a> {
    /// Borrow mutable strided bytes with checked dimensions, alignment, and length.
    ///
    /// Accepts owned [`ImageSpec`], borrowed `&'a ImageSpec` (zero `Arc` refcount bump),
    /// or `Copy` [`PixelFormat`](crate::PixelFormat).
    pub fn new(
        data: &'a mut [u8],
        width: u32,
        rows: u32,
        stride: usize,
        spec: impl Into<SpecCow<'a>>,
    ) -> Result<Self, At<BufferError>> {
        let spec = spec.into();
        let (_rb, req) = validate_geometry(
            data.len(),
            data.as_ptr(),
            width,
            rows,
            stride,
            spec.as_spec(),
        )?;
        Ok(Self {
            data: &mut data[..req],
            width,
            rows,
            stride,
            spec,
            hdr: HdrMetadata::default(),
            orientation: Orientation::Identity,
        })
    }

    /// Descriptive HDR container metadata attached to this mutable view.
    #[inline]
    #[must_use]
    pub const fn hdr(&self) -> HdrMetadata {
        self.hdr
    }

    /// Attach descriptive HDR metadata to this mutable view.
    #[inline]
    #[must_use]
    pub const fn with_hdr(mut self, hdr: HdrMetadata) -> Self {
        self.hdr = hdr;
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
        self.orientation.output_dimensions(self.width, self.rows)
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

    /// Compose an additional display transform onto this view's pending orientation
    /// (`self.orientation.then(delta)`) in O(1) without touching pixels.
    #[inline]
    #[must_use]
    pub const fn rotate_metadata(mut self, delta: Orientation) -> Self {
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
    pub const fn rows(&self) -> u32 {
        self.rows
    }

    /// Row stride in bytes.
    #[inline]
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Active bytes per row.
    #[inline]
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        (self.width as usize) * self.spec.as_spec().layout.bytes_per_pixel()
    }

    /// Authoritative image specification.
    #[inline]
    #[must_use]
    pub const fn spec(&self) -> &ImageSpec {
        self.spec.as_spec()
    }

    /// Mutable active bytes of row `y`.
    #[inline]
    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(y < self.rows, "row {y} out of bounds (rows: {})", self.rows);
        let start = (y as usize) * self.stride;
        let rb = self.row_bytes();
        &mut self.data[start..start + rb]
    }

    /// Zero-copy mutable typed slice of row `y` (e.g. `&mut [rgb::Rgba<u8>]`).
    pub fn row_as_mut<P: bytemuck::Pod>(&mut self, y: u32) -> Result<&mut [P], At<BufferError>> {
        if y >= self.rows {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let bytes = self.row_mut(y);
        bytemuck::try_cast_slice_mut(bytes).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
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
        if self.rows <= 1 || self.stride == self.row_bytes() {
            let total = self.row_bytes() * (self.rows as usize);
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
            rows: self.rows,
            stride: self.stride,
            spec: self.spec.to_borrowed(),
            facts: SampleFacts::NONE,
            hdr: self.hdr,
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
            rows: self.rows,
            stride: self.stride,
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
    pub fn crop_mut(
        &mut self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<PixelViewMut<'_>, At<BufferError>> {
        if w == 0 || h == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if x.checked_add(w).is_none_or(|e| e > self.width)
            || y.checked_add(h).is_none_or(|e| e > self.rows)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let bpp = self.spec.layout.bytes_per_pixel();
        let start = (y as usize) * self.stride + (x as usize) * bpp;
        let end = (y as usize + h as usize - 1) * self.stride + (x as usize + w as usize) * bpp;
        Ok(PixelViewMut {
            data: &mut self.data[start..end],
            width: w,
            rows: h,
            stride: self.stride,
            spec: self.spec.to_borrowed(),
            hdr: self.hdr,
            orientation: self.orientation,
        })
    }

    /// Mutable full-width strip view.
    pub fn sub_rows_mut(
        &mut self,
        y: u32,
        count: u32,
    ) -> Result<PixelViewMut<'_>, At<BufferError>> {
        self.crop_mut(0, y, self.width, count)
    }

    /// Consume this mutable view and narrow it to a rectangular crop retaining `'a`.
    pub fn into_crop(
        self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<PixelViewMut<'a>, At<BufferError>> {
        if w == 0 || h == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if x.checked_add(w).is_none_or(|e| e > self.width)
            || y.checked_add(h).is_none_or(|e| e > self.rows)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let bpp = self.spec.layout.bytes_per_pixel();
        let start = (y as usize) * self.stride + (x as usize) * bpp;
        let end = (y as usize + h as usize - 1) * self.stride + (x as usize + w as usize) * bpp;
        Ok(PixelViewMut {
            data: &mut self.data[start..end],
            width: w,
            rows: h,
            stride: self.stride,
            spec: self.spec,
            hdr: self.hdr,
            orientation: self.orientation,
        })
    }

    /// Consume this mutable view and narrow it to a full-width strip retaining `'a`.
    pub fn into_sub_rows(self, y: u32, count: u32) -> Result<PixelViewMut<'a>, At<BufferError>> {
        let width = self.width;
        self.into_crop(0, y, width, count)
    }
}

// ---------------------------------------------------------------------------
// PixelBuf (unparameterized owned buffer, fallible by default)
// ---------------------------------------------------------------------------

/// Decomposed parts returned by [`PixelBuf::into_parts`].
#[derive(Debug)]
#[non_exhaustive]
pub struct PixelBufParts {
    /// Underlying alignment-exact storage.
    pub storage: PixelStorage,
    /// Byte offset of `(0, 0)` within `storage.as_bytes()`.
    pub offset: usize,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in rows.
    pub height: u32,
    /// Byte stride between rows.
    pub stride: usize,
    /// Image specification.
    pub spec: ImageSpec,
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
    spec: ImageSpec,
    facts: SampleFacts,
    hdr: HdrMetadata,
    origin: ColorOrigin,
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
            let native = PixelStorage::try_from_rows(
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
            hdr: self.hdr,
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
            let native = PixelStorage::try_from_rows(
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
            hdr: self.hdr,
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
    /// [`AlphaMode::Undefined`] (or `Straight`), fill all pixels, and retag zero-copy with
    /// [`into_opaque_verified`](Self::into_opaque_verified) or [`assume_opaque`](Self::assume_opaque).
    pub fn new(
        width: u32,
        height: u32,
        spec: impl Into<ImageSpec>,
    ) -> Result<Self, At<BufferError>> {
        Self::new_with_fill(width, height, spec, RowAlign::Tight, BufFill::Black)
    }

    /// Fallibly allocate a tightly packed white-filled buffer (reference white on color channels
    /// and canonical maximum on any alpha slot).
    pub fn new_white(
        width: u32,
        height: u32,
        spec: impl Into<ImageSpec>,
    ) -> Result<Self, At<BufferError>> {
        Self::new_with_fill(width, height, spec, RowAlign::Tight, BufFill::White)
    }

    /// Fallibly allocate a black-filled buffer with explicit row alignment
    /// (`Tight`, `Cache64`, `PixelCache64`, `Samples(n)`).
    ///
    /// For `Cache64` and `PixelCache64`, storage is allocated as 64-byte aligned
    /// [`AlignedSamples`] so both the first pixel and every
    /// row start land on a 64-byte boundary. When `spec.color.alpha() == AlphaMode::Opaque`,
    /// alpha samples are initialized to full opacity.
    pub fn new_aligned(
        width: u32,
        height: u32,
        spec: impl Into<ImageSpec>,
        align: RowAlign,
    ) -> Result<Self, At<BufferError>> {
        Self::new_with_fill(width, height, spec, align, BufFill::Black)
    }

    /// Fallibly allocate a buffer with explicit row alignment and initial [`BufFill`]
    /// (`Black` or `White`).
    pub fn new_with_fill(
        width: u32,
        height: u32,
        spec: impl Into<ImageSpec>,
        align: RowAlign,
        fill: BufFill,
    ) -> Result<Self, At<BufferError>> {
        let spec = spec.into();
        spec.validate()
            .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
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
            RowAlign::Cache64 | RowAlign::PixelCache64 => PixelStorage::try_zeroed_aligned64(total),
            _ => PixelStorage::try_zeroed(spec.layout.sample, total),
        }
        .map_err(|e| err_at(e.into()))?;
        let facts = match fill {
            BufFill::Black => {
                if spec.color.alpha() == AlphaMode::Opaque {
                    fill_opaque_alpha(storage.as_bytes_mut(), width, height, stride, &spec);
                }
                SampleFacts::for_zeroed_spec(&spec)
            }
            BufFill::White => {
                fill_white_pixels(storage.as_bytes_mut(), width, height, stride, &spec);
                SampleFacts::for_white_spec(&spec)
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
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
            orientation: Orientation::Identity,
        })
    }

    /// Fallibly allocate raw zero-filled OS pages with **no alpha-fill pass** for maximum
    /// decoder/renderer write throughput.
    ///
    /// Refuses [`AlphaMode::Opaque`] on layouts with an alpha slot with
    /// [`BufferError::UnverifiedOpaqueAlpha`] so an opaque-tagged buffer can never silently
    /// contain zero alpha. Allocate with [`AlphaMode::Undefined`] (or `Straight`), write every
    /// pixel including full alpha, then retag zero-copy via
    /// [`into_opaque_verified`](Self::into_opaque_verified),
    /// [`into_opaque_filled`](Self::into_opaque_filled), or [`assume_opaque`](Self::assume_opaque).
    pub fn new_zeroed(
        width: u32,
        height: u32,
        spec: impl Into<ImageSpec>,
    ) -> Result<Self, At<BufferError>> {
        Self::new_zeroed_aligned(width, height, spec, RowAlign::Tight)
    }

    /// Fallibly allocate raw zero-filled aligned OS pages with **no alpha-fill pass**,
    /// refusing [`AlphaMode::Opaque`] on layouts with an alpha slot.
    pub fn new_zeroed_aligned(
        width: u32,
        height: u32,
        spec: impl Into<ImageSpec>,
        align: RowAlign,
    ) -> Result<Self, At<BufferError>> {
        let spec = spec.into();
        spec.validate()
            .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
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
            RowAlign::Cache64 | RowAlign::PixelCache64 => PixelStorage::try_zeroed_aligned64(total),
            _ => PixelStorage::try_zeroed(spec.layout.sample, total),
        }
        .map_err(|e| err_at(e.into()))?;
        let facts = SampleFacts::for_zeroed_spec(&spec);
        Ok(Self {
            storage,
            offset: 0,
            width,
            height,
            stride,
            spec,
            facts,
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
            orientation: Orientation::Identity,
        })
    }

    /// Fallibly allocate tight native storage validated upfront for a zero-copy [`InteropTarget`].
    pub fn new_for(
        width: u32,
        height: u32,
        spec: impl Into<ImageSpec>,
        target: InteropTarget,
    ) -> Result<Self, At<BufferError>> {
        let spec = spec.into();
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
    /// To recover `bytes` intact on error, use [`try_from_bytes_recover`](Self::try_from_bytes_recover).
    pub fn from_bytes(
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        spec: impl Into<ImageSpec>,
    ) -> Result<Self, At<BufferError>> {
        Self::try_from_bytes_recover(bytes, width, height, spec).map_err(|(e, _)| e)
    }

    /// Adopt tightly packed bytes (`Vec<u8>`), returning `Err((err, bytes))` with the original
    /// `Vec<u8>` untouched on validation or allocation failure.
    #[allow(clippy::result_large_err)]
    pub fn try_from_bytes_recover(
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        spec: impl Into<ImageSpec>,
    ) -> Result<Self, (At<BufferError>, Vec<u8>)> {
        let spec = spec.into();
        if let Err(e) = spec.validate() {
            return Err((whereat::at!(BufferError::InvalidSpec(*e.error())), bytes));
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
            match PixelStorage::try_from_rows(
                spec.layout.sample,
                req,
                core::iter::once(&bytes[..req]),
            ) {
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
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
            orientation: Orientation::Identity,
        })
    }

    /// Adopt a tightly packed typed `Vec<P>` (pixels like `Vec<rgb::Rgb<u16>>`, arrays `[u8; 3]`,
    /// or raw samples like `Vec<f32>`).
    ///
    /// Validates that `P` is structurally compatible with `spec.layout` (preventing silent
    /// reinterpretation such as `Vec<i32>` as `RGBF32`). For intentional raw byte
    /// reinterpretation, use [`from_raw_pod`](Self::from_raw_pod).
    ///
    /// **Zero-copy** whenever `align_of::<P>() == spec.layout.min_alignment()` and
    /// `bytemuck::allocation::try_cast_vec` succeeds (preserving exact deallocation alignment!).
    /// Otherwise fallibly copies into native-aligned [`PixelStorage`].
    ///
    /// To recover `pixels` intact on error, use [`try_from_vec_recover`](Self::try_from_vec_recover).
    pub fn from_vec<P: TypedElement>(
        pixels: Vec<P>,
        width: u32,
        height: u32,
        spec: impl Into<ImageSpec>,
    ) -> Result<Self, At<BufferError>> {
        Self::try_from_vec_recover(pixels, width, height, spec).map_err(|(e, _)| e)
    }

    /// Adopt a tightly packed typed `Vec<P>`, returning `Err((err, pixels))` with the original
    /// `Vec<P>` untouched on validation or allocation failure.
    #[allow(clippy::result_large_err)]
    pub fn try_from_vec_recover<P: TypedElement>(
        pixels: Vec<P>,
        width: u32,
        height: u32,
        spec: impl Into<ImageSpec>,
    ) -> Result<Self, (At<BufferError>, Vec<P>)> {
        let spec = spec.into();
        if !P::matches_layout(spec.layout) {
            return Err((err_at(BufferError::IncompatibleLayout), pixels));
        }
        if let Err(e) = spec.validate() {
            return Err((whereat::at!(BufferError::InvalidSpec(*e.error())), pixels));
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
        let storage = match PixelStorage::try_adopt_vec(pixels, spec.layout.sample) {
            Ok(s) => s,
            Err(pixels) => {
                let copied = {
                    let raw = bytemuck::cast_slice::<P, u8>(&pixels);
                    PixelStorage::try_from_rows(
                        spec.layout.sample,
                        req,
                        core::iter::once(&raw[..req]),
                    )
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
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
            orientation: Orientation::Identity,
        })
    }

    /// Adopt a tightly packed scalar sample `Vec<S>` (`Vec<u8>`, `Vec<u16>`, `Vec<f32>`, etc.).
    pub fn from_samples<S: StorageSample>(
        samples: Vec<S>,
        width: u32,
        height: u32,
        spec: impl Into<ImageSpec>,
    ) -> Result<Self, At<BufferError>> {
        let spec = spec.into();
        if spec.layout.sample != S::SAMPLE_TYPE {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        Self::from_raw_pod(samples, width, height, spec)
    }

    /// Adopt a tightly packed `rgb` pixel `Vec<P>` (`Vec<rgb::Rgb<u8>>`, etc.).
    #[cfg(feature = "rgb")]
    pub fn try_from_pixels<P: crate::interop::PixelType>(
        pixels: Vec<P>,
        width: u32,
        height: u32,
        color: crate::ColorSpace,
    ) -> Result<Self, At<BufferError>> {
        Self::from_raw_pod(pixels, width, height, ImageSpec::new(P::LAYOUT, color))
    }

    /// Explicitly reinterpret any tightly packed `Vec<P: bytemuck::Pod>` in native endianness
    /// as raw pixel bytes matching `spec`.
    pub fn from_raw_pod<P: bytemuck::Pod>(
        pixels: Vec<P>,
        width: u32,
        height: u32,
        spec: impl Into<ImageSpec>,
    ) -> Result<Self, At<BufferError>> {
        let spec = spec.into();
        spec.validate()
            .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
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
        let storage = match PixelStorage::try_adopt_vec(pixels, spec.layout.sample) {
            Ok(s) => s,
            Err(pixels) => {
                let raw = bytemuck::cast_slice::<P, u8>(&pixels);
                PixelStorage::try_from_rows(spec.layout.sample, req, core::iter::once(&raw[..req]))
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
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
            orientation: Orientation::Identity,
        })
    }

    /// Checked reconstruction from owned [`PixelStorage`], offset, dimensions, stride, and spec.
    pub fn from_parts(
        storage: PixelStorage,
        offset: usize,
        width: u32,
        height: u32,
        stride: usize,
        spec: impl Into<ImageSpec>,
    ) -> Result<Self, At<BufferError>> {
        let spec = spec.into();
        let bytes = storage.as_bytes();
        let slice = bytes
            .get(offset..)
            .ok_or_else(|| err_at(BufferError::InsufficientData))?;
        validate_geometry(slice.len(), slice.as_ptr(), width, height, stride, &spec)?;
        Ok(Self {
            storage,
            offset,
            width,
            height,
            stride,
            spec,
            facts: SampleFacts::NONE,
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
            orientation: Orientation::Identity,
        })
    }

    /// Checked reconstruction from a [`PixelBufParts`] struct (preserving `hdr`, `origin`,
    /// `orientation`, and `parts.facts`).
    ///
    /// On validation failure, `parts` is dropped; use
    /// [`try_from_parts_recover`](Self::try_from_parts_recover) to recover the allocation intact.
    pub fn try_from_parts(parts: PixelBufParts) -> Result<Self, At<BufferError>> {
        Self::try_from_parts_recover(parts).map_err(|(err, _)| err)
    }

    /// Checked reconstruction from [`PixelBufParts`], returning `Err((err, parts))` with the
    /// original allocation and metadata intact on validation failure.
    #[allow(clippy::result_large_err)]
    pub fn try_from_parts_recover(
        parts: PixelBufParts,
    ) -> Result<Self, (At<BufferError>, PixelBufParts)> {
        let bytes = parts.storage.as_bytes();
        let Some(slice) = bytes.get(parts.offset..) else {
            return Err((err_at(BufferError::InsufficientData), parts));
        };
        if let Err(e) = validate_geometry(
            slice.len(),
            slice.as_ptr(),
            parts.width,
            parts.height,
            parts.stride,
            &parts.spec,
        ) {
            return Err((e, parts));
        }
        Ok(Self {
            storage: parts.storage,
            offset: parts.offset,
            width: parts.width,
            height: parts.height,
            stride: parts.stride,
            spec: parts.spec,
            facts: parts.facts,
            hdr: parts.hdr,
            origin: parts.origin,
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
    /// mutable access (`as_view_mut`, `try_as_view_mut`, `try_make_mut`) transparently
    /// performs copy-on-write into owned aligned storage without mutating `storage`.
    /// Call [`try_into_external`](Self::try_into_external) to recover `S` zero-copy.
    #[allow(clippy::result_large_err)]
    pub fn from_external<S: ByteStorage + Send + Sync + 'static>(
        storage: S,
        width: u32,
        height: u32,
        stride: usize,
        spec: impl Into<ImageSpec>,
    ) -> Result<Self, (At<BufferError>, S)> {
        Self::from_external_with_offset(storage, 0, width, height, stride, spec)
    }

    /// Adopt an immutable external owned backing starting at `offset` bytes **zero-copy**.
    ///
    /// Returns `Err((err, storage))` untouched on validation failure with zero heap allocation.
    #[allow(clippy::result_large_err)]
    pub fn from_external_with_offset<S: ByteStorage + Send + Sync + 'static>(
        storage: S,
        offset: usize,
        width: u32,
        height: u32,
        stride: usize,
        spec: impl Into<ImageSpec>,
    ) -> Result<Self, (At<BufferError>, S)> {
        let spec = spec.into();
        let check = {
            let bytes = storage.as_bytes();
            match bytes.get(offset..) {
                None => Err(err_at(BufferError::InsufficientData)),
                Some(slice) => {
                    validate_geometry(slice.len(), slice.as_ptr(), width, height, stride, &spec)
                }
            }
        };
        if let Err(e) = check {
            return Err((e, storage));
        }
        Ok(Self {
            storage: PixelStorage::from_external(storage),
            offset,
            width,
            height,
            stride,
            spec,
            facts: SampleFacts::NONE,
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
            orientation: Orientation::Identity,
        })
    }

    /// Adopt a mutable external owned backing (`Box<[T]>`, custom-allocator wrapper, mutable
    /// pool lease, FFI writable buffer, etc.) **zero-copy** for both reads and in-place writes.
    ///
    /// Validates geometry and sample alignment BEFORE boxing `storage`; on error, returns
    /// `Err((err, storage))` with zero heap allocation and `storage` completely untouched.
    /// Call [`try_into_external_mut`](Self::try_into_external_mut) to recover `S` zero-copy.
    #[allow(clippy::result_large_err)]
    pub fn from_external_mut<S: ByteStorageMut + Send + Sync + 'static>(
        storage: S,
        width: u32,
        height: u32,
        stride: usize,
        spec: impl Into<ImageSpec>,
    ) -> Result<Self, (At<BufferError>, S)> {
        Self::from_external_mut_with_offset(storage, 0, width, height, stride, spec)
    }

    /// Adopt a mutable external owned backing starting at `offset` bytes **zero-copy**.
    ///
    /// Returns `Err((err, storage))` untouched on validation failure with zero heap allocation.
    #[allow(clippy::result_large_err)]
    pub fn from_external_mut_with_offset<S: ByteStorageMut + Send + Sync + 'static>(
        storage: S,
        offset: usize,
        width: u32,
        height: u32,
        stride: usize,
        spec: impl Into<ImageSpec>,
    ) -> Result<Self, (At<BufferError>, S)> {
        let spec = spec.into();
        let check = {
            let bytes = storage.as_bytes();
            match bytes.get(offset..) {
                None => Err(err_at(BufferError::InsufficientData)),
                Some(slice) => {
                    validate_geometry(slice.len(), slice.as_ptr(), width, height, stride, &spec)
                }
            }
        };
        if let Err(e) = check {
            return Err((e, storage));
        }
        Ok(Self {
            storage: PixelStorage::from_external_mut(storage),
            offset,
            width,
            height,
            stride,
            spec,
            facts: SampleFacts::NONE,
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
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
    pub fn try_into_external<S: 'static>(mut self) -> Result<S, Self> {
        match self.storage.try_into_external::<S>() {
            Ok(owner) => Ok(owner),
            Err(storage) => {
                self.storage = storage;
                Err(self)
            }
        }
    }

    /// Recover a mutable external owner `S` zero-copy; returns `Err(self)` untouched otherwise.
    #[allow(clippy::result_large_err)]
    pub fn try_into_external_mut<S: 'static>(mut self) -> Result<S, Self> {
        match self.storage.try_into_external_mut::<S>() {
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
    pub fn try_make_mut(&mut self) -> Result<(), At<BufferError>> {
        if !matches!(self.storage, PixelStorage::External(_)) {
            return Ok(());
        }
        let align = self.spec.layout.min_alignment();
        let rb = self.row_bytes();
        let req = required_bytes(self.height, self.stride, rb).map_err(err_at)?;
        if self.offset == 0 && (self.height <= 1 || self.stride % align == 0) {
            let mut aligned =
                AlignedSamples::try_zeroed_bytes(req).map_err(|e| err_at(e.into()))?;
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
                PixelStorage::try_from_rows(
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
    pub const fn stride(&self) -> usize {
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
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Authoritative image specification.
    #[inline]
    #[must_use]
    pub fn spec(&self) -> &ImageSpec {
        &self.spec
    }

    /// Replace specification metadata while preserving numeric sample type, bit packing, and
    /// `bytes_per_pixel` (e.g. relabeling color space or channel order without modifying bytes).
    ///
    /// Rejects changes to `sample` type or `packing` (use [`reinterpret`](Self::reinterpret) for
    /// deliberate raw-byte reinterpretation across sample types). Clears cached [`SampleFacts`]
    /// and resets [`ColorOrigin`] to [`ColorOrigin::assumed`] whenever `spec.color` changes.
    pub fn with_spec(mut self, spec: impl Into<ImageSpec>) -> Result<Self, At<BufferError>> {
        let spec = spec.into();
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
            self.origin = ColorOrigin::assumed();
        }
        Ok(self)
    }

    /// Replace only the color space while keeping the exact [`SampleLayout`].
    ///
    /// Resets [`ColorOrigin`] to [`ColorOrigin::assumed`] when `color` changes and clears
    /// cached [`SampleFacts`] when signal range, alpha mode, or color model changes.
    pub fn with_color(mut self, color: crate::ColorSpace) -> Result<Self, At<BufferError>> {
        let new_spec = ImageSpec::new(self.spec.layout, color);
        new_spec
            .validate()
            .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
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
            self.origin = ColorOrigin::assumed();
        }
        Ok(self)
    }

    /// Deliberately reinterpret the raw pixel bytes under a new [`ImageSpec`] with identical
    /// `bytes_per_pixel` and satisfied sample alignment (permits changing `sample` type and
    /// `packing`, e.g. `U8` ↔ `I8` or `U16` ↔ `F16`).
    ///
    /// Always clears cached [`SampleFacts`], [`ColorOrigin`], and [`HdrMetadata`].
    pub fn reinterpret(mut self, spec: impl Into<ImageSpec>) -> Result<Self, At<BufferError>> {
        let spec = spec.into();
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
        self.origin = ColorOrigin::assumed();
        self.hdr = HdrMetadata::default();
        Ok(self)
    }

    /// Verify that every active pixel's alpha slot equals its canonical maximum value, then
    /// retag this buffer as [`AlphaMode::Opaque`] with `facts.all_opaque = true` (zero copy).
    ///
    /// Returns [`BufferError::IncompatibleLayout`] if the layout has no alpha slot, or
    /// [`BufferError::UnverifiedOpaqueAlpha`] if any alpha sample is not at channel maximum.
    pub fn into_opaque_verified(mut self) -> Result<Self, At<BufferError>> {
        if !self.spec.layout.order.has_alpha_slot() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let new_spec = self.spec.clone().with_alpha(AlphaMode::Opaque);
        new_spec
            .validate()
            .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
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
            .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
        self.try_make_mut()?;
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
    /// allocate zeroed as [`AlphaMode::Undefined`], write every pixel including `A = max`,
    /// then call `assume_opaque()`.
    pub fn assume_opaque(mut self) -> Result<Self, At<BufferError>> {
        if !self.spec.layout.order.has_alpha_slot() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let new_spec = self.spec.clone().with_alpha(AlphaMode::Opaque);
        new_spec
            .validate()
            .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
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

    /// Attach proven sample facts (caller asserts validity; cleared on next `&mut` borrow).
    #[inline]
    pub fn set_facts(&mut self, facts: SampleFacts) {
        self.facts = facts;
    }

    /// Descriptive HDR container metadata.
    #[inline]
    #[must_use]
    pub const fn hdr(&self) -> HdrMetadata {
        self.hdr
    }

    /// Attach descriptive HDR container metadata.
    #[inline]
    #[must_use]
    pub fn with_hdr(mut self, hdr: HdrMetadata) -> Self {
        self.hdr = hdr;
        self
    }

    /// Original container color provenance.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> &ColorOrigin {
        &self.origin
    }

    /// Attach container color provenance.
    #[inline]
    #[must_use]
    pub fn with_origin(mut self, origin: ColorOrigin) -> Self {
        self.origin = origin;
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

    /// Compose an additional display transform onto this buffer's pending orientation
    /// (`self.orientation.then(delta)`) in O(1) without touching or copying pixels.
    #[inline]
    #[must_use]
    pub const fn rotate_metadata(mut self, delta: Orientation) -> Self {
        self.orientation = self.orientation.then(delta);
        self
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
            rows: self.height,
            stride: self.stride,
            spec: SpecCow::Borrowed(&self.spec),
            facts: self.facts,
            hdr: self.hdr,
            origin: Some(&self.origin),
            orientation: self.orientation,
        }
    }

    /// Fallibly borrow a mutable [`PixelViewMut`], clearing cached [`SampleFacts`] and
    /// copy-on-write promoting immutable external storage without panicking on OOM.
    #[inline]
    pub fn try_as_view_mut(&mut self) -> Result<PixelViewMut<'_>, At<BufferError>> {
        self.try_make_mut()?;
        self.facts = SampleFacts::NONE;
        let rb = self.row_bytes();
        let end = self.offset + required_bytes(self.height, self.stride, rb).unwrap_or(0);
        let bytes = self.storage.as_bytes_mut();
        Ok(PixelViewMut {
            data: &mut bytes[self.offset..end],
            width: self.width,
            rows: self.height,
            stride: self.stride,
            spec: SpecCow::Borrowed(&self.spec),
            hdr: self.hdr,
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

    /// Borrow a mutable horizontal strip of `count` rows starting at `y`, automatically
    /// clearing cached [`SampleFacts`].
    #[inline]
    pub fn view_rows_mut(
        &mut self,
        y: u32,
        count: u32,
    ) -> Result<PixelViewMut<'_>, At<BufferError>> {
        self.as_view_mut().into_sub_rows(y, count)
    }

    /// Borrow a rectangular cropped subimage view without copying pixels.
    #[inline]
    pub fn crop_view(
        &self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<PixelView<'_>, At<BufferError>> {
        self.as_view().crop(x, y, w, h)
    }

    /// Atomically transform this buffer's bytes and metadata in place with zero reallocation.
    ///
    /// The closure receives `(active_bytes, width, height, stride, spec, origin)` where
    /// `active_bytes` starts at pixel `(0, 0)`. If it returns
    /// `Some((new_stride, new_spec, new_origin, new_facts))`, the new spec and geometry are
    /// validated against the existing storage and adopted atomically; if it returns `None`,
    /// the buffer's metadata is unchanged.
    pub fn transform_in_place<F>(&mut self, f: F) -> Result<(), At<BufferError>>
    where
        F: FnOnce(
            &mut [u8],
            u32,
            u32,
            usize,
            &ImageSpec,
            &ColorOrigin,
        ) -> Option<(usize, ImageSpec, ColorOrigin, SampleFacts)>,
    {
        self.try_make_mut()?;
        let active = &mut self.storage.as_bytes_mut()[self.offset..];
        let Some((new_stride, new_spec, new_origin, new_facts)) = f(
            active,
            self.width,
            self.height,
            self.stride,
            &self.spec,
            &self.origin,
        ) else {
            return Ok(());
        };
        new_spec
            .validate()
            .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
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
        self.origin = new_origin;
        self.facts = new_facts;
        Ok(())
    }

    /// Compact strided rows in-place within the current allocation (`stride = row_bytes`).
    ///
    /// Preserves `offset = 0` whenever the backing storage base pointer satisfies
    /// `spec.layout.min_alignment()`, and otherwise keeps the smallest valid
    /// aligned offset (`< min_alignment()`) so sample alignment is never violated.
    pub fn into_contiguous(mut self) -> Self {
        let rb = self.row_bytes();
        let align = self.spec.layout.min_alignment();
        let ptr_rem = (self.storage.as_bytes().as_ptr() as usize) % align;
        let target_offset = if ptr_rem == 0 { 0 } else { align - ptr_rem };
        if self.offset == target_offset && (self.height <= 1 || self.stride == rb) {
            self.stride = rb;
            self.storage
                .truncate_bytes(target_offset + rb * self.height as usize);
            return self;
        }
        self.try_make_mut()
            .expect("copy-on-write promotion of external PixelBuf failed");
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
        self
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
                let new_spec = ImageSpec::new(new_layout, self.spec.color.clone());
                new_spec
                    .validate()
                    .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
                self.try_make_mut()?;
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
                let new_spec = ImageSpec::new(new_layout, new_color);
                new_spec
                    .validate()
                    .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
                self.try_make_mut()?;
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
    pub fn try_into_vec_zero_copy<P: bytemuck::Pod>(mut self) -> Result<Vec<P>, Self> {
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
        match self.storage.try_cast_into_vec::<P>() {
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
    pub fn try_into_vec<P: bytemuck::Pod>(self) -> Result<Vec<P>, At<BufferError>> {
        let elem = core::mem::size_of::<P>();
        let rb = self.row_bytes();
        let total = rb
            .checked_mul(self.height as usize)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        if elem == 0 || total % elem != 0 {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        match self.try_into_vec_zero_copy::<P>() {
            Ok(v) => Ok(v),
            Err(buf) => {
                let count = total / elem;
                if rb % elem == 0 {
                    let mut out: Vec<P> = Vec::new();
                    out.try_reserve_exact(count)
                        .map_err(|_| err_at(BufferError::AllocationFailed))?;
                    let view = buf.as_view();
                    for y in 0..view.rows() {
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
                    for y in 0..view.rows() {
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
            offset: self.offset,
            width: self.width,
            height: self.height,
            stride: self.stride,
            spec: self.spec,
            facts: self.facts,
            hdr: self.hdr,
            origin: self.origin,
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
                rows: v.rows,
                stride: v.stride,
                spec: v.spec.to_borrowed(),
                facts: v.facts,
                hdr: v.hdr,
                origin: v.origin,
                orientation: v.orientation,
            },
            Self::Owned(b) => b.as_view(),
        }
    }

    /// Convert into an owned [`PixelBuf`] (zero-copy if already owned, fallible copy if borrowed).
    pub fn try_into_owned(self) -> Result<PixelBuf, At<BufferError>> {
        match self {
            Self::Borrowed(v) => v.try_to_tight_buf(),
            Self::Owned(b) => Ok(b),
        }
    }
}

// ---------------------------------------------------------------------------
// BandedBuf (gigapixel-safe horizontal bands)
// ---------------------------------------------------------------------------

/// Default per-band memory cap (`16 MiB`) for [`BandedBuf`].
pub const DEFAULT_MAX_BAND_BYTES: usize = 16 * 1024 * 1024;

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
    spec: ImageSpec,
    facts: SampleFacts,
    hdr: HdrMetadata,
    origin: ColorOrigin,
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
        spec: impl Into<ImageSpec>,
        max_band_bytes: usize,
    ) -> Result<Self, At<BufferError>> {
        Self::new_aligned(width, height, spec, max_band_bytes, RowAlign::Tight)
    }

    /// Fallibly allocate a zero-filled banded image with explicit per-band [`RowAlign`].
    pub fn new_aligned(
        width: u32,
        height: u32,
        spec: impl Into<ImageSpec>,
        max_band_bytes: usize,
        align: RowAlign,
    ) -> Result<Self, At<BufferError>> {
        let spec = spec.into();
        spec.validate()
            .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
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
        let facts = SampleFacts::for_zeroed_spec(&spec);
        Ok(Self {
            width,
            height,
            band_rows: rows_per_band,
            bands,
            spec,
            facts,
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
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
            let spec = buf.spec().clone();
            let facts = buf.facts();
            let hdr = buf.hdr();
            let origin = buf.origin().clone();
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
        let height = view.rows();
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
            bands.push(view.sub_rows(y, h)?.try_to_tight_buf()?);
            y += h;
        }
        let origin = view.origin().cloned().unwrap_or_else(ColorOrigin::assumed);
        Ok(Self {
            width,
            height,
            band_rows: rows_per_band,
            bands,
            spec: view.spec().clone(),
            facts: view.facts(),
            hdr: view.hdr(),
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
        let storage = PixelStorage::try_from_rows(
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

    /// Map global row `y` to `(band_index, row_within_band)`, or `None` if out of bounds.
    #[inline]
    #[must_use]
    pub const fn locate(&self, y: u32) -> Option<(usize, u32)> {
        if y >= self.height {
            return None;
        }
        Some(((y / self.band_rows) as usize, y % self.band_rows))
    }

    /// Borrow zero-copy [`PixelView`] slices covering global row range `start_y..start_y + count`,
    /// split at band boundaries.
    pub fn row_range_views(
        &self,
        start_y: u32,
        count: u32,
    ) -> Result<Vec<PixelView<'_>>, At<BufferError>> {
        if count == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let end_y = start_y
            .checked_add(count)
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
    pub const fn band_rows(&self) -> u32 {
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
    pub fn spec(&self) -> &ImageSpec {
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
    pub const fn hdr(&self) -> HdrMetadata {
        self.hdr
    }

    /// Attach descriptive HDR metadata (propagated to all child bands).
    #[inline]
    #[must_use]
    pub fn with_hdr(mut self, hdr: HdrMetadata) -> Self {
        self.hdr = hdr;
        for band in &mut self.bands {
            band.hdr = hdr;
        }
        self
    }

    /// Original container color provenance.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> &ColorOrigin {
        &self.origin
    }

    /// Attach container color provenance (propagated to all child bands).
    #[inline]
    #[must_use]
    pub fn with_origin(mut self, origin: ColorOrigin) -> Self {
        self.origin = origin.clone();
        for band in &mut self.bands {
            band.origin = origin.clone();
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
    pub fn rotate_metadata(mut self, delta: Orientation) -> Self {
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
        let stride = self.bands[b].stride();
        let rb = self.bands[b].row_bytes();
        let off = self.bands[b].offset();
        let start = off + (ly as usize) * stride;
        &mut self.bands[b].storage.as_bytes_mut()[start..start + rb]
    }
}

// ---------------------------------------------------------------------------
// PlanarBuf / PlanarView (1..=4 planes with first-class BitPacking)
// ---------------------------------------------------------------------------

/// Owned single plane inside a [`PlanarBuf`] with validated geometry invariants.
#[derive(Clone, Debug)]
pub struct OwnedPlane {
    spec: PlaneSpec,
    width: u32,
    height: u32,
    stride: usize,
    storage: PixelStorage,
}

impl OwnedPlane {
    /// Construct a validated owned plane adopting caller-provided [`PixelStorage`] zero-copy.
    pub fn from_storage(
        spec: PlaneSpec,
        width: u32,
        height: u32,
        stride: usize,
        storage: PixelStorage,
    ) -> Result<Self, At<BufferError>> {
        spec.validate().map_err(|e| err_at(e.into()))?;
        if width == 0 || height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let row_bytes = (width as usize)
            .checked_mul(spec.bytes_per_element())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        if stride < row_bytes {
            return Err(err_at(BufferError::StrideTooSmall));
        }
        let align = spec.sample.min_alignment();
        let bytes = storage.as_bytes();
        if (bytes.as_ptr() as usize) % align != 0 || (height > 1 && stride % align != 0) {
            return Err(err_at(BufferError::AlignmentViolation));
        }
        let req = required_bytes(height, stride, row_bytes).map_err(err_at)?;
        if bytes.len() < req {
            return Err(err_at(BufferError::InsufficientData));
        }
        Ok(Self {
            spec,
            width,
            height,
            stride,
            storage,
        })
    }

    /// Construct a validated owned plane adopting an immutable external owner (`Arc<[u8]>`,
    /// `Arc<[u16]>`, mmap, etc.) zero-copy. Returns `Err((err, storage))` untouched on failure.
    #[allow(clippy::result_large_err)]
    pub fn from_external<S: ByteStorage + Send + Sync + 'static>(
        spec: PlaneSpec,
        width: u32,
        height: u32,
        stride: usize,
        storage: S,
    ) -> Result<Self, (At<BufferError>, S)> {
        let check = (|| -> Result<(), At<BufferError>> {
            spec.validate().map_err(|e| err_at(e.into()))?;
            if width == 0 || height == 0 {
                return Err(err_at(BufferError::InvalidDimensions));
            }
            let row_bytes = (width as usize)
                .checked_mul(spec.bytes_per_element())
                .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
            if stride < row_bytes {
                return Err(err_at(BufferError::StrideTooSmall));
            }
            let align = spec.sample.min_alignment();
            let bytes = storage.as_bytes();
            if (bytes.as_ptr() as usize) % align != 0 || (height > 1 && stride % align != 0) {
                return Err(err_at(BufferError::AlignmentViolation));
            }
            let req = required_bytes(height, stride, row_bytes).map_err(err_at)?;
            if bytes.len() < req {
                return Err(err_at(BufferError::InsufficientData));
            }
            Ok(())
        })();
        if let Err(e) = check {
            return Err((e, storage));
        }
        Ok(Self {
            spec,
            width,
            height,
            stride,
            storage: PixelStorage::from_external(storage),
        })
    }

    /// Construct a validated owned plane adopting a mutable external owner (`Box<[T]>`,
    /// custom allocator, pool lease, etc.) zero-copy. Returns `Err((err, storage))` untouched on failure.
    #[allow(clippy::result_large_err)]
    pub fn from_external_mut<S: ByteStorageMut + Send + Sync + 'static>(
        spec: PlaneSpec,
        width: u32,
        height: u32,
        stride: usize,
        storage: S,
    ) -> Result<Self, (At<BufferError>, S)> {
        let check = (|| -> Result<(), At<BufferError>> {
            spec.validate().map_err(|e| err_at(e.into()))?;
            if width == 0 || height == 0 {
                return Err(err_at(BufferError::InvalidDimensions));
            }
            let row_bytes = (width as usize)
                .checked_mul(spec.bytes_per_element())
                .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
            if stride < row_bytes {
                return Err(err_at(BufferError::StrideTooSmall));
            }
            let align = spec.sample.min_alignment();
            let bytes = storage.as_bytes();
            if (bytes.as_ptr() as usize) % align != 0 || (height > 1 && stride % align != 0) {
                return Err(err_at(BufferError::AlignmentViolation));
            }
            let req = required_bytes(height, stride, row_bytes).map_err(err_at)?;
            if bytes.len() < req {
                return Err(err_at(BufferError::InsufficientData));
            }
            Ok(())
        })();
        if let Err(e) = check {
            return Err((e, storage));
        }
        Ok(Self {
            spec,
            width,
            height,
            stride,
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
    pub fn try_into_external<S: 'static>(mut self) -> Result<S, Self> {
        match self.storage.try_into_external::<S>() {
            Ok(owner) => Ok(owner),
            Err(storage) => {
                self.storage = storage;
                Err(self)
            }
        }
    }

    /// Recover a mutable external owner `S` zero-copy; returns `Err(self)` untouched otherwise.
    #[allow(clippy::result_large_err)]
    pub fn try_into_external_mut<S: 'static>(mut self) -> Result<S, Self> {
        match self.storage.try_into_external_mut::<S>() {
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
    pub const fn spec(&self) -> PlaneSpec {
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
    pub const fn stride(&self) -> usize {
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

    /// Extract the unpacked integer code for `(x, y, component)` respecting [`BitPacking`].
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
}

fn read_sample_u32(sample: SampleType, bytes: &[u8]) -> u32 {
    match sample {
        SampleType::U8 => u32::from(bytes[0]),
        SampleType::U16 => u32::from(u16::from_ne_bytes([bytes[0], bytes[1]])),
        SampleType::U32 => u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        _ => 0,
    }
}

fn validate_unique_plane_roles(
    roles: impl Iterator<Item = PlaneRole>,
) -> Result<(), At<BufferError>> {
    let mut seen_mask: u32 = 0;
    for role in roles {
        let bit: Option<u32> = match role {
            PlaneRole::Y => Some(0),
            PlaneRole::Cb => Some(1),
            PlaneRole::Cr => Some(2),
            PlaneRole::CbCr => Some(3),
            PlaneRole::CrCb => Some(4),
            PlaneRole::R => Some(5),
            PlaneRole::G => Some(6),
            PlaneRole::B => Some(7),
            PlaneRole::Cyan => Some(8),
            PlaneRole::Magenta => Some(9),
            PlaneRole::Yellow => Some(10),
            PlaneRole::Key => Some(11),
            PlaneRole::Alpha => Some(12),
            PlaneRole::GainMap => Some(13),
            PlaneRole::Packed | PlaneRole::Extra(_) => None,
        };
        if let Some(b) = bit {
            let mask = 1u32 << b;
            if (seen_mask & mask) != 0 {
                return Err(err_at(BufferError::IncompatibleLayout));
            }
            seen_mask |= mask;
        }
    }
    Ok(())
}

/// Multi-plane image (YCbCr 4:2:0/4:2:2/4:4:4, NV12, P010, planar RGB+A, planar CMYKA, JXL/TIFF extra planes).
#[derive(Clone, Debug)]
pub struct PlanarBuf {
    width: u32,
    height: u32,
    planes: Vec<OwnedPlane>,
    chroma_siting: ChromaSiting,
    color: Option<ColorSpace>,
    ycbcr_matrix_code: Option<u8>,
    range: Option<SignalRange>,
    alpha: Option<AlphaMode>,
    hdr: HdrMetadata,
    origin: ColorOrigin,
    orientation: Orientation,
}

impl PlanarBuf {
    /// Maximum plane count supported in a single [`PlanarBuf`].
    pub const MAX_PLANES: usize = 64;

    /// Fallibly allocate a zero-filled multi-plane buffer for `specs` (`1..=64` planes).
    pub fn new(width: u32, height: u32, specs: &[PlaneSpec]) -> Result<Self, At<BufferError>> {
        if width == 0 || height == 0 || specs.is_empty() || specs.len() > Self::MAX_PLANES {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        validate_unique_plane_roles(specs.iter().map(|s| s.role))?;
        let mut planes = Vec::new();
        planes
            .try_reserve_exact(specs.len())
            .map_err(|_| err_at(BufferError::AllocationFailed))?;
        for &ps in specs {
            ps.validate().map_err(|e| err_at(e.into()))?;
            let (pw, ph) = ps.dimensions(width, height);
            let stride = (pw as usize)
                .checked_mul(ps.bytes_per_element())
                .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
            let total = required_bytes(ph, stride, stride).map_err(err_at)?;
            let storage =
                PixelStorage::try_zeroed(ps.sample, total).map_err(|e| err_at(e.into()))?;
            planes.push(OwnedPlane {
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
            ycbcr_matrix_code: None,
            range: None,
            alpha: None,
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
            orientation: Orientation::Identity,
        })
    }

    /// Construct a validated [`PlanarBuf`] adopting caller-owned planes zero-copy (`1..=64` planes).
    pub fn from_planes(
        width: u32,
        height: u32,
        planes: Vec<OwnedPlane>,
    ) -> Result<Self, At<BufferError>> {
        if width == 0 || height == 0 || planes.is_empty() || planes.len() > Self::MAX_PLANES {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        validate_unique_plane_roles(planes.iter().map(|p| p.spec.role))?;
        for p in &planes {
            let (ew, eh) = p.spec.dimensions(width, height);
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
            ycbcr_matrix_code: None,
            range: None,
            alpha: None,
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
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
        self.color.as_ref()
    }

    /// Attach color interpretation (primaries, transfer, range, alpha) to this planar buffer.
    #[inline]
    #[must_use]
    pub fn with_color(mut self, color: ColorSpace) -> Self {
        self.color = Some(color);
        self
    }

    /// Optional ITU-T H.273 `MatrixCoefficients` code (`1` = BT.709, `6` = BT.601, `9` = BT.2020 NCL, `0` = Identity).
    #[inline]
    #[must_use]
    pub const fn ycbcr_matrix_code(&self) -> Option<u8> {
        self.ycbcr_matrix_code
    }

    /// Attach ITU-T H.273 `MatrixCoefficients` code for YCbCr/GBR planes.
    #[inline]
    #[must_use]
    pub const fn with_ycbcr_matrix_code(mut self, matrix_code: u8) -> Self {
        self.ycbcr_matrix_code = Some(matrix_code);
        self
    }

    /// Explicit signal range (`Full` or `Narrow`), if set separately from [`color`](Self::color).
    #[inline]
    #[must_use]
    pub const fn range(&self) -> Option<SignalRange> {
        self.range
    }

    /// Effective signal range, checking explicit [`range`](Self::range) first and falling back
    /// to [`color`](Self::color)'s range.
    #[inline]
    #[must_use]
    pub fn effective_range(&self) -> Option<SignalRange> {
        self.range
            .or_else(|| self.color.as_ref().and_then(|c| c.range()))
    }

    /// Attach explicit signal range (`Full` or `Narrow`).
    #[inline]
    #[must_use]
    pub const fn with_range(mut self, range: SignalRange) -> Self {
        self.range = Some(range);
        self
    }

    /// Explicit alpha association mode for an attached alpha plane, if set separately from [`color`](Self::color).
    #[inline]
    #[must_use]
    pub const fn alpha(&self) -> Option<AlphaMode> {
        self.alpha
    }

    /// Effective alpha association mode, checking explicit [`alpha`](Self::alpha) first and
    /// falling back to [`color`](Self::color)'s alpha mode.
    #[inline]
    #[must_use]
    pub fn effective_alpha(&self) -> Option<AlphaMode> {
        self.alpha
            .or_else(|| self.color.as_ref().map(|c| c.alpha()))
    }

    /// Attach explicit alpha association mode (e.g. `AlphaMode::Straight` or `AlphaMode::Premultiplied`).
    #[inline]
    #[must_use]
    pub const fn with_alpha(mut self, alpha: AlphaMode) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Immutable slice of planes.
    #[inline]
    #[must_use]
    pub fn planes(&self) -> &[OwnedPlane] {
        &self.planes
    }

    /// Consume this planar buffer and return its owned planes zero-copy.
    #[inline]
    #[must_use]
    pub fn into_planes(self) -> Vec<OwnedPlane> {
        self.planes
    }

    /// Borrow plane `index` mutably as a [`PlaneViewMut`] without allowing structural replacement.
    #[inline]
    pub fn plane_view_mut(&mut self, index: usize) -> Option<PlaneViewMut<'_>> {
        self.planes.get_mut(index).map(OwnedPlane::as_view_mut)
    }

    /// Descriptive HDR metadata.
    #[inline]
    #[must_use]
    pub const fn hdr(&self) -> HdrMetadata {
        self.hdr
    }

    /// Attach descriptive HDR metadata.
    #[inline]
    #[must_use]
    pub const fn with_hdr(mut self, hdr: HdrMetadata) -> Self {
        self.hdr = hdr;
        self
    }

    /// Original container provenance.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> &ColorOrigin {
        &self.origin
    }

    /// Attach original container provenance.
    #[inline]
    #[must_use]
    pub fn with_origin(mut self, origin: ColorOrigin) -> Self {
        self.origin = origin;
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
    pub const fn rotate_metadata(mut self, delta: Orientation) -> Self {
        self.orientation = self.orientation.then(delta);
        self
    }
}

/// Validated borrowed view of one plane inside a [`PlanarView`].
#[derive(Clone, Copy, Debug)]
pub struct PlaneView<'a> {
    spec: PlaneSpec,
    width: u32,
    height: u32,
    stride: usize,
    data: &'a [u8],
}

impl<'a> PlaneView<'a> {
    const EMPTY: Self = Self {
        spec: PlaneSpec::new(crate::PlaneRole::Y, SampleType::U8),
        width: 0,
        height: 0,
        stride: 0,
        data: &[],
    };

    /// Construct a validated borrowed plane view.
    pub fn new(
        spec: PlaneSpec,
        width: u32,
        height: u32,
        stride: usize,
        data: &'a [u8],
    ) -> Result<Self, At<BufferError>> {
        spec.validate().map_err(|e| err_at(e.into()))?;
        if width == 0 || height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let rb = (width as usize)
            .checked_mul(spec.bytes_per_element())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        if stride < rb {
            return Err(err_at(BufferError::StrideTooSmall));
        }
        let align = spec.sample.min_alignment();
        if (data.as_ptr() as usize) % align != 0 || (height > 1 && stride % align != 0) {
            return Err(err_at(BufferError::AlignmentViolation));
        }
        let req = required_bytes(height, stride, rb).map_err(err_at)?;
        if data.len() < req {
            return Err(err_at(BufferError::InsufficientData));
        }
        Ok(Self {
            spec,
            width,
            height,
            stride,
            data: &data[..req],
        })
    }

    /// Plane specification.
    #[inline]
    #[must_use]
    pub const fn spec(&self) -> PlaneSpec {
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
    pub const fn stride(&self) -> usize {
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

    /// Zero-copy rectangular crop of this single plane retaining `'a`.
    pub fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Result<PlaneView<'a>, At<BufferError>> {
        if w == 0 || h == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if x.checked_add(w).is_none_or(|e| e > self.width)
            || y.checked_add(h).is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let elem = self.spec.bytes_per_element();
        let start = (y as usize) * self.stride + (x as usize) * elem;
        let end = (y as usize + h as usize - 1) * self.stride + (x as usize + w as usize) * elem;
        Ok(PlaneView {
            spec: self.spec,
            width: w,
            height: h,
            stride: self.stride,
            data: &self.data[start..end],
        })
    }
}

/// Validated mutable borrowed view of one plane.
#[derive(Debug)]
pub struct PlaneViewMut<'a> {
    spec: PlaneSpec,
    width: u32,
    height: u32,
    stride: usize,
    data: &'a mut [u8],
}

impl<'a> PlaneViewMut<'a> {
    /// Construct a validated mutable borrowed plane view.
    pub fn new(
        spec: PlaneSpec,
        width: u32,
        height: u32,
        stride: usize,
        data: &'a mut [u8],
    ) -> Result<Self, At<BufferError>> {
        spec.validate().map_err(|e| err_at(e.into()))?;
        if width == 0 || height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let rb = (width as usize)
            .checked_mul(spec.bytes_per_element())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        if stride < rb {
            return Err(err_at(BufferError::StrideTooSmall));
        }
        let align = spec.sample.min_alignment();
        if (data.as_ptr() as usize) % align != 0 || (height > 1 && stride % align != 0) {
            return Err(err_at(BufferError::AlignmentViolation));
        }
        let req = required_bytes(height, stride, rb).map_err(err_at)?;
        if data.len() < req {
            return Err(err_at(BufferError::InsufficientData));
        }
        Ok(Self {
            spec,
            width,
            height,
            stride,
            data: &mut data[..req],
        })
    }

    /// Plane specification.
    #[inline]
    #[must_use]
    pub const fn spec(&self) -> PlaneSpec {
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
    pub const fn stride(&self) -> usize {
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
    pub fn crop_mut(
        &mut self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<PlaneViewMut<'_>, At<BufferError>> {
        if w == 0 || h == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if x.checked_add(w).is_none_or(|e| e > self.width)
            || y.checked_add(h).is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let elem = self.spec.bytes_per_element();
        let start = (y as usize) * self.stride + (x as usize) * elem;
        let end = (y as usize + h as usize - 1) * self.stride + (x as usize + w as usize) * elem;
        Ok(PlaneViewMut {
            spec: self.spec,
            width: w,
            height: h,
            stride: self.stride,
            data: &mut self.data[start..end],
        })
    }

    /// Consume this mutable plane view and narrow it to a rectangular crop retaining `'a`.
    pub fn into_crop(
        self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<PlaneViewMut<'a>, At<BufferError>> {
        if w == 0 || h == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if x.checked_add(w).is_none_or(|e| e > self.width)
            || y.checked_add(h).is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let elem = self.spec.bytes_per_element();
        let start = (y as usize) * self.stride + (x as usize) * elem;
        let end = (y as usize + h as usize - 1) * self.stride + (x as usize + w as usize) * elem;
        Ok(PlaneViewMut {
            spec: self.spec,
            width: w,
            height: h,
            stride: self.stride,
            data: &mut self.data[start..end],
        })
    }
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
/// color interpretation (`color`, `ycbcr_matrix_code`, `range`, `alpha`) so subsampled chroma
/// crops and YCbCr decoders carry full color authority.
#[derive(Clone, Debug)]
pub struct PlanarView<'a> {
    width: u32,
    height: u32,
    crop_origin: (u32, u32),
    chroma_siting: ChromaSiting,
    color: Option<ColorSpace>,
    ycbcr_matrix_code: Option<u8>,
    range: Option<SignalRange>,
    alpha: Option<AlphaMode>,
    orientation: Orientation,
    storage: PlanarStorage<'a>,
}

impl<'a> PlanarView<'a> {
    /// Construct a validated multi-plane borrowed view (`1..=64` planes) at origin `(0, 0)`.
    pub fn new(width: u32, height: u32, planes: &[PlaneView<'a>]) -> Result<Self, At<BufferError>> {
        Self::new_with_origin(width, height, (0, 0), ChromaSiting::Unknown, planes)
    }

    /// Construct a validated multi-plane borrowed view with explicit parent-frame `crop_origin`
    /// and [`ChromaSiting`].
    pub fn new_with_origin(
        width: u32,
        height: u32,
        crop_origin: (u32, u32),
        chroma_siting: ChromaSiting,
        planes: &[PlaneView<'a>],
    ) -> Result<Self, At<BufferError>> {
        if width == 0 || height == 0 || planes.is_empty() || planes.len() > PlanarBuf::MAX_PLANES {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        validate_unique_plane_roles(planes.iter().map(|p| p.spec.role))?;
        for &p in planes {
            let Some((ew, eh)) = p.spec.cropped_dimensions(width, height, crop_origin) else {
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
            ycbcr_matrix_code: None,
            range: None,
            alpha: None,
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
        self.color.as_ref()
    }

    /// Attach color interpretation to this borrowed planar view.
    #[inline]
    #[must_use]
    pub fn with_color(mut self, color: ColorSpace) -> Self {
        self.color = Some(color);
        self
    }

    /// Optional ITU-T H.273 `MatrixCoefficients` code (`1` = BT.709, `6` = BT.601, `9` = BT.2020 NCL, `0` = Identity).
    #[inline]
    #[must_use]
    pub const fn ycbcr_matrix_code(&self) -> Option<u8> {
        self.ycbcr_matrix_code
    }

    /// Attach ITU-T H.273 `MatrixCoefficients` code.
    #[inline]
    #[must_use]
    pub const fn with_ycbcr_matrix_code(mut self, matrix_code: u8) -> Self {
        self.ycbcr_matrix_code = Some(matrix_code);
        self
    }

    /// Explicit signal range, if set.
    #[inline]
    #[must_use]
    pub const fn range(&self) -> Option<SignalRange> {
        self.range
    }

    /// Effective signal range (explicit [`range`](Self::range) or [`color`](Self::color)'s range).
    #[inline]
    #[must_use]
    pub fn effective_range(&self) -> Option<SignalRange> {
        self.range
            .or_else(|| self.color.as_ref().and_then(|c| c.range()))
    }

    /// Attach explicit signal range.
    #[inline]
    #[must_use]
    pub const fn with_range(mut self, range: SignalRange) -> Self {
        self.range = Some(range);
        self
    }

    /// Explicit alpha association mode, if set.
    #[inline]
    #[must_use]
    pub const fn alpha(&self) -> Option<AlphaMode> {
        self.alpha
    }

    /// Effective alpha association mode (explicit [`alpha`](Self::alpha) or [`color`](Self::color)'s alpha).
    #[inline]
    #[must_use]
    pub fn effective_alpha(&self) -> Option<AlphaMode> {
        self.alpha
            .or_else(|| self.color.as_ref().map(|c| c.alpha()))
    }

    /// Attach explicit alpha association mode.
    #[inline]
    #[must_use]
    pub const fn with_alpha(mut self, alpha: AlphaMode) -> Self {
        self.alpha = Some(alpha);
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
    pub const fn rotate_metadata(mut self, delta: Orientation) -> Self {
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

    /// Zero-copy rectangular crop of all planes preserving exact parent-frame chroma phase
    /// even for odd `(x, y)` origins on 4:2:0 / 4:2:2 subsampled frames.
    pub fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Result<PlanarView<'a>, At<BufferError>> {
        if w == 0 || h == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if x.checked_add(w).is_none_or(|e| e > self.width)
            || y.checked_add(h).is_none_or(|e| e > self.height)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let new_ox = self
            .crop_origin
            .0
            .checked_add(x)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let new_oy = self
            .crop_origin
            .1
            .checked_add(y)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let new_origin = (new_ox, new_oy);
        let src_planes = self.planes();
        let crop_one = |p: &PlaneView<'a>| -> Result<PlaneView<'a>, At<BufferError>> {
            let hx = p.spec.subsampling.0.max(1) as u32;
            let vy = p.spec.subsampling.1.max(1) as u32;
            let base_cx = self.crop_origin.0 / hx;
            let base_cy = self.crop_origin.1 / vy;
            let rel_cx = (new_ox / hx) - base_cx;
            let rel_cy = (new_oy / vy) - base_cy;
            let (cw, ch) = p
                .spec
                .cropped_dimensions(w, h, new_origin)
                .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
            p.crop(rel_cx, rel_cy, cw, ch)
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
            width: w,
            height: h,
            crop_origin: new_origin,
            chroma_siting: self.chroma_siting,
            color: self.color.clone(),
            ycbcr_matrix_code: self.ycbcr_matrix_code,
            range: self.range,
            alpha: self.alpha,
            orientation: self.orientation,
            storage,
        })
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
    color: Option<ColorSpace>,
    ycbcr_matrix_code: Option<u8>,
    range: Option<SignalRange>,
    alpha: Option<AlphaMode>,
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
        validate_unique_plane_roles(planes.iter().map(|p| p.spec.role))?;
        for p in planes {
            let Some((ew, eh)) = p.spec.cropped_dimensions(width, height, crop_origin) else {
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
        Self::new_with_origin(width, height, (0, 0), ChromaSiting::Unknown, planes)
    }

    /// Construct a validated mutable multi-plane view with explicit `crop_origin` and [`ChromaSiting`].
    pub fn new_with_origin(
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
            ycbcr_matrix_code: None,
            range: None,
            alpha: None,
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
        Self::from_array_with_origin(width, height, (0, 0), ChromaSiting::Unknown, planes)
    }

    /// Construct a validated mutable multi-plane view from a fixed array of planes with explicit
    /// `crop_origin` and [`ChromaSiting`] (zero heap allocation for `1..=4` planes).
    pub fn from_array_with_origin<const N: usize>(
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
            ycbcr_matrix_code: None,
            range: None,
            alpha: None,
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
        self.color.as_ref()
    }

    /// Attach color interpretation.
    #[inline]
    #[must_use]
    pub fn with_color(mut self, color: ColorSpace) -> Self {
        self.color = Some(color);
        self
    }

    /// Optional ITU-T H.273 `MatrixCoefficients` code.
    #[inline]
    #[must_use]
    pub const fn ycbcr_matrix_code(&self) -> Option<u8> {
        self.ycbcr_matrix_code
    }

    /// Attach ITU-T H.273 `MatrixCoefficients` code.
    #[inline]
    #[must_use]
    pub const fn with_ycbcr_matrix_code(mut self, matrix_code: u8) -> Self {
        self.ycbcr_matrix_code = Some(matrix_code);
        self
    }

    /// Explicit signal range, if set.
    #[inline]
    #[must_use]
    pub const fn range(&self) -> Option<SignalRange> {
        self.range
    }

    /// Effective signal range.
    #[inline]
    #[must_use]
    pub fn effective_range(&self) -> Option<SignalRange> {
        self.range
            .or_else(|| self.color.as_ref().and_then(|c| c.range()))
    }

    /// Attach explicit signal range.
    #[inline]
    #[must_use]
    pub const fn with_range(mut self, range: SignalRange) -> Self {
        self.range = Some(range);
        self
    }

    /// Explicit alpha association mode, if set.
    #[inline]
    #[must_use]
    pub const fn alpha(&self) -> Option<AlphaMode> {
        self.alpha
    }

    /// Effective alpha association mode.
    #[inline]
    #[must_use]
    pub fn effective_alpha(&self) -> Option<AlphaMode> {
        self.alpha
            .or_else(|| self.color.as_ref().map(|c| c.alpha()))
    }

    /// Attach explicit alpha association mode.
    #[inline]
    #[must_use]
    pub const fn with_alpha(mut self, alpha: AlphaMode) -> Self {
        self.alpha = Some(alpha);
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
    pub const fn rotate_metadata(mut self, delta: Orientation) -> Self {
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
            ycbcr_matrix_code: self.ycbcr_matrix_code,
            range: self.range,
            alpha: self.alpha,
            orientation: self.orientation,
            storage,
        })
    }

    /// Downgrade into an immutable [`PlanarView<'a>`] retaining `'a`.
    pub fn try_into_view(self) -> Result<PlanarView<'a>, At<BufferError>> {
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
            ycbcr_matrix_code: self.ycbcr_matrix_code,
            range: self.range,
            alpha: self.alpha,
            orientation: self.orientation,
            storage,
        })
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
            ycbcr_matrix_code: self.ycbcr_matrix_code,
            range: self.range,
            alpha: self.alpha,
            orientation: self.orientation,
            storage,
        })
    }

    /// Borrow all planes mutably as a [`PlanarViewMut`] (zero-allocation for `<= 4` planes).
    #[must_use]
    pub fn as_view_mut(&mut self) -> PlanarViewMut<'_> {
        self.try_as_view_mut()
            .expect("PlanarBuf::as_view_mut allocation failed")
    }

    /// Fallibly borrow all planes mutably as a [`PlanarViewMut`] (zero-allocation for `<= 4` planes).
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
            ycbcr_matrix_code: self.ycbcr_matrix_code,
            range: self.range,
            alpha: self.alpha,
            orientation: self.orientation,
            storage,
        })
    }
}

// ---------------------------------------------------------------------------
// DualLayerHdr & DualLayerHdrView (ISO 21496-1 / UltraHDR interchange)
// ---------------------------------------------------------------------------

fn validate_dual_layer_specs(
    base_spec: &ImageSpec,
    gain_map_spec: &ImageSpec,
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

/// Complete decomposed parts of a [`DualLayerHdr`] carrier, preserving every field losslessly.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DualLayerHdrParts {
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
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        if base.orientation() != crate::Orientation::Identity && base.orientation() != orientation {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        validate_dual_layer_specs(
            base.spec(),
            gain_map.spec(),
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

    /// Compose an additional display transform onto the shared pending EXIF orientation
    /// (`self.orientation.then(delta)`) in O(1) without touching or copying either pixel layer.
    #[inline]
    #[must_use]
    pub const fn rotate_metadata(mut self, delta: crate::Orientation) -> Self {
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
            .output_dimensions(self.base.width(), self.base.rows())
    }
}

/// Container-agnostic dual-layer HDR image (base + numeric gain map + exact ISO 21496-1 metadata).
///
/// Unifies UltraHDR JPEG, AVIF `tmap`, JXL `jhgm`, and PNG gain-map HDR in one validated core type.
#[derive(Clone, Debug)]
pub struct DualLayerHdr {
    base: PixelBuf,
    gain_map: PixelBuf,
    metadata: GainMapMetadata,
    alternate_color_space: Option<crate::ColorSpace>,
    orientation: crate::Orientation,
}

impl DualLayerHdr {
    /// Construct a validated dual-layer HDR carrier with `GainMapColorSpace::Base`.
    ///
    /// Validates base color model (`Rgb`/`Gray`/1-or-3-ch `Icc`), numeric gain-map layout
    /// (`ColorSpace::Untagged`, 1 or 3 channels, `U8`/`U16`/`F32` with `Full`/`Lsb`/`Msb` packing),
    /// and exact rational ISO 21496-1 metadata (`gamma > 0`, `min <= max`, `H_base != H_alt`).
    ///
    /// Rejects `metadata.application_space == GainMapColorSpace::Alternate` because no
    /// alternate color space is supplied — use [`new_with_alternate`](Self::new_with_alternate)
    /// when `application_space == Alternate`.
    pub fn new(
        base: PixelBuf,
        gain_map: PixelBuf,
        metadata: GainMapMetadata,
    ) -> Result<Self, At<BufferError>> {
        let orientation = base.orientation();
        Self::new_with_alternate(base, gain_map, metadata, None, orientation)
    }

    /// Construct a validated dual-layer HDR carrier with optional alternate color space
    /// (required when `metadata.application_space == GainMapColorSpace::Alternate`) and
    /// shared pending [`Orientation`](crate::Orientation).
    ///
    /// Rejects misregistered layer orientations (`base.orientation() != gain_map.orientation()`)
    /// or a non-`Identity` layer orientation that contradicts `orientation`.
    pub fn new_with_alternate(
        mut base: PixelBuf,
        mut gain_map: PixelBuf,
        metadata: GainMapMetadata,
        alternate_color_space: Option<crate::ColorSpace>,
        orientation: crate::Orientation,
    ) -> Result<Self, At<BufferError>> {
        if base.orientation() != gain_map.orientation() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        if base.orientation() != crate::Orientation::Identity && base.orientation() != orientation {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        validate_dual_layer_specs(
            base.spec(),
            gain_map.spec(),
            metadata,
            alternate_color_space.as_ref(),
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

    /// Reconstruct a validated [`DualLayerHdr`] from [`DualLayerHdrParts`].
    pub fn try_from_parts(parts: DualLayerHdrParts) -> Result<Self, At<BufferError>> {
        Self::new_with_alternate(
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
            self.base.spec(),
            self.gain_map.spec(),
            self.metadata,
            Some(&alt),
        )?;
        self.alternate_color_space = Some(alt);
        Ok(self)
    }

    /// Attach a shared pending [`Orientation`](crate::Orientation) for the aligned pair.
    #[inline]
    #[must_use]
    pub const fn with_orientation(mut self, orientation: crate::Orientation) -> Self {
        self.orientation = orientation;
        self.base.orientation = orientation;
        self.gain_map.orientation = orientation;
        self
    }

    /// Compose an additional display transform onto the shared pending EXIF orientation
    /// (`self.orientation.then(delta)`) in O(1) without touching or rebaking either pixel layer.
    ///
    /// Keeps `base` and `gain_map` strictly registered in stored-frame coordinates for
    /// lossless EXIF passthrough workflows. To physically bake orientation into both pixel
    /// buffers, use `zenpixels_convert::gainmap::{bake_dual_layer_orientation, rotate_dual_layer}`.
    #[inline]
    #[must_use]
    pub const fn rotate_metadata(mut self, delta: crate::Orientation) -> Self {
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
            alternate_color_space: self.alternate_color_space.as_ref(),
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
        self.alternate_color_space.as_ref()
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

    /// Decompose into complete owned [`DualLayerHdrParts`], preserving `alternate_color_space`
    /// and pending `orientation` losslessly.
    #[inline]
    #[must_use]
    pub fn into_parts(self) -> DualLayerHdrParts {
        DualLayerHdrParts {
            base: self.base,
            gain_map: self.gain_map,
            metadata: self.metadata,
            alternate_color_space: self.alternate_color_space,
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
    fn source_spec(&self) -> &ImageSpec;
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
    fn sink_spec(&self) -> &ImageSpec;
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
        self.rows
    }
    #[inline]
    fn source_spec(&self) -> &ImageSpec {
        self.spec.as_spec()
    }
    #[inline]
    fn source_facts(&self) -> SampleFacts {
        self.facts
    }
    #[inline]
    fn source_hdr(&self) -> HdrMetadata {
        self.hdr
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
    fn source_spec(&self) -> &ImageSpec {
        &self.spec
    }
    #[inline]
    fn source_facts(&self) -> SampleFacts {
        self.facts
    }
    #[inline]
    fn source_hdr(&self) -> HdrMetadata {
        self.hdr
    }
    #[inline]
    fn source_origin(&self) -> Option<&ColorOrigin> {
        Some(&self.origin)
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
    fn source_spec(&self) -> &ImageSpec {
        &self.spec
    }
    #[inline]
    fn source_facts(&self) -> SampleFacts {
        self.facts
    }
    #[inline]
    fn source_hdr(&self) -> HdrMetadata {
        self.hdr
    }
    #[inline]
    fn source_origin(&self) -> Option<&ColorOrigin> {
        Some(&self.origin)
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
        self.rows
    }
    #[inline]
    fn sink_spec(&self) -> &ImageSpec {
        self.spec.as_spec()
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
    fn sink_spec(&self) -> &ImageSpec {
        &self.spec
    }
    #[inline]
    fn sink_row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(y < self.height, "row {y} out of bounds ({})", self.height);
        self.try_make_mut()
            .expect("copy-on-write promotion of external PixelBuf failed");
        self.facts = SampleFacts::NONE;
        let rb = self.row_bytes();
        let start = self.offset + (y as usize) * self.stride;
        &mut self.storage.as_bytes_mut()[start..start + rb]
    }
    #[inline]
    fn sink_finish(&mut self, facts: SampleFacts, hdr: HdrMetadata) {
        self.facts = facts;
        self.hdr = hdr;
    }
    #[inline]
    fn sink_set_origin(&mut self, origin: ColorOrigin) {
        self.origin = origin;
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
    fn sink_spec(&self) -> &ImageSpec {
        &self.spec
    }
    #[inline]
    fn sink_row_mut(&mut self, y: u32) -> &mut [u8] {
        self.row_mut(y)
    }
    #[inline]
    fn sink_finish(&mut self, facts: SampleFacts, hdr: HdrMetadata) {
        self.facts = facts;
        self.hdr = hdr;
        for b in &mut self.bands {
            b.facts = facts;
            b.hdr = hdr;
        }
    }
    #[inline]
    fn sink_set_origin(&mut self, origin: ColorOrigin) {
        self.origin = origin.clone();
        for b in &mut self.bands {
            b.origin = origin.clone();
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
