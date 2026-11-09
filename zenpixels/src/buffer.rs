//! Opaque pixel buffer abstraction.
//!
//! Provides format-aware pixel storage that carries its own metadata:
//! [`PixelBuffer`] (owned), [`PixelSlice`] (borrowed, immutable), and
//! [`PixelSliceMut`] (borrowed, mutable).
//!
//! These types track pixel format via [`PixelDescriptor`] and color context
//! via [`ColorContext`].
//!
//! Typed variants (e.g., `PixelBuffer<Rgb<u8>>`) enforce format correctness
//! at compile time through the [`Pixel`] trait. Use `erase()` to convert
//! to a type-erased form, or `try_typed()` to go back.
//!
//! For format conversions (e.g., `convert_to`, `to_rgb8`), see the
//! `zenpixels-convert` crate which provides `PixelBufferConvertExt`.

use crate::descriptor::ChannelType;
use crate::plane::{AlignedSamples, PlaneError, PlaneSample, SampleStorage, SampleStorageMut};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

use whereat::At;
#[cfg(feature = "rgb")]
use whereat::ResultAtExt;

#[cfg(feature = "imgref")]
use imgref::ImgRef;
#[cfg(feature = "imgref")]
use imgref::ImgVec;
#[cfg(feature = "rgb")]
use rgb::alt::BGRA;
#[cfg(feature = "rgb")]
use rgb::{Gray, Rgb, Rgba};

use crate::cicp::Cicp;
use crate::color::ColorContext;
use crate::descriptor::{
    AlphaMode, ColorPrimaries, PixelDescriptor, SignalRange, TransferFunction,
};
use crate::hdr::DiffuseWhite;
use crate::pixel_types::{GrayAlpha8, GrayAlpha16, GrayAlphaF32};
use crate::sample::{ExactDepth, ExactDepthError};

// ---------------------------------------------------------------------------
// Padded pixel types (32-bit SIMD-friendly)
// ---------------------------------------------------------------------------

/// 32-bit RGB pixel with padding byte (RGBx).
///
/// Same memory layout as `Rgba<u8>` but the 4th byte is padding,
/// not alpha. Use this for SIMD-friendly 32-bit RGB processing
/// without alpha semantics.
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Rgbx {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Padding byte. Value is unspecified and should be ignored.
    pub x: u8,
}

/// 32-bit BGR pixel with padding byte (BGRx).
///
/// Same memory layout as `BGRA<u8>` but the 4th byte is padding.
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Bgrx {
    /// Blue channel.
    pub b: u8,
    /// Green channel.
    pub g: u8,
    /// Red channel.
    pub r: u8,
    /// Padding byte. Value is unspecified and should be ignored.
    pub x: u8,
}

// ---------------------------------------------------------------------------
// Pixel trait
// ---------------------------------------------------------------------------

/// Compile-time pixel format descriptor.
///
/// Implemented for pixel types to associate them with their
/// [`PixelDescriptor`]. This enables typed [`PixelSlice`] construction
/// where the type system enforces format correctness.
///
/// The trait is open (not sealed) — custom pixel types can implement it.
/// The `new_typed()` constructors include a compile-time assertion that
/// `size_of::<P>() == P::DESCRIPTOR.bytes_per_pixel()` to catch bad impls.
pub trait Pixel: bytemuck::Pod {
    /// The pixel format descriptor for this type.
    const DESCRIPTOR: PixelDescriptor;
}

impl Pixel for Rgbx {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::RGBX8;
}

impl Pixel for Bgrx {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::BGRX8;
}

#[cfg(feature = "rgb")]
impl Pixel for Rgb<u8> {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::RGB8;
}

#[cfg(feature = "rgb")]
impl Pixel for Rgba<u8> {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::RGBA8;
}

#[cfg(feature = "rgb")]
impl Pixel for Gray<u8> {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::GRAY8;
}

#[cfg(feature = "rgb")]
impl Pixel for Rgb<u16> {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::RGB16;
}

#[cfg(feature = "rgb")]
impl Pixel for Rgba<u16> {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::RGBA16;
}

#[cfg(feature = "rgb")]
impl Pixel for Gray<u16> {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::GRAY16;
}

#[cfg(feature = "rgb")]
impl Pixel for Rgb<f32> {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::RGBF32;
}

#[cfg(feature = "rgb")]
impl Pixel for Rgba<f32> {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::RGBAF32;
}

#[cfg(feature = "rgb")]
impl Pixel for Gray<f32> {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::GRAYF32;
}

#[cfg(feature = "rgb")]
impl Pixel for BGRA<u8> {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::BGRA8;
}

impl Pixel for GrayAlpha8 {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::GRAYA8;
}

impl Pixel for GrayAlpha16 {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::GRAYA16;
}

impl Pixel for GrayAlphaF32 {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::GRAYAF32;
}

// ---------------------------------------------------------------------------
// BufferError
// ---------------------------------------------------------------------------

/// Errors from pixel buffer operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferError {
    /// Data pointer or stride is not aligned for the channel type.
    ///
    /// A stride must be a multiple of the sample size, so that every row starts
    /// on a sample.
    AlignmentViolation,
    /// Data slice is too small for the given dimensions and stride.
    InsufficientData,
    /// Stride is smaller than `width * bytes_per_pixel`.
    StrideTooSmall,
    /// Stride is not a multiple of `bytes_per_pixel`, for a representation that
    /// counts its stride in pixels.
    ///
    /// Pixel views accept such a stride. Conversions that can't express one return
    /// this instead of misreading rows: an `imgref` image, for example, has no way
    /// to describe RGB8 rows 8 bytes apart.
    StrideNotPixelAligned,
    /// Width or height is zero or causes overflow.
    InvalidDimensions,
    /// Descriptor bytes_per_pixel mismatch in `reinterpret()`.
    ///
    /// The new descriptor has a different `bytes_per_pixel()` than the
    /// current one, so reinterpreting the buffer would be invalid.
    IncompatibleDescriptor,
    /// Buffer allocation failed (out of memory or overflow).
    AllocationFailed,
}

impl fmt::Display for BufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlignmentViolation => write!(f, "data is not aligned for the channel type"),
            Self::InsufficientData => {
                write!(f, "data slice is too small for the given dimensions")
            }
            Self::StrideTooSmall => write!(f, "stride is smaller than width * bytes_per_pixel"),
            Self::StrideNotPixelAligned => {
                write!(f, "stride is not a multiple of bytes_per_pixel")
            }
            Self::InvalidDimensions => write!(f, "width or height is zero or causes overflow"),
            Self::IncompatibleDescriptor => {
                write!(
                    f,
                    "descriptor conflicts with pixel layout, type, or alpha semantics"
                )
            }
            Self::AllocationFailed => write!(f, "buffer allocation failed"),
        }
    }
}

impl core::error::Error for BufferError {}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Round `val` up to the next multiple of `align` (must be a power of 2).
const fn align_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

/// Compute the byte offset needed to align `ptr` to `align`.
fn align_offset(ptr: *const u8, align: usize) -> usize {
    let addr = ptr as usize;
    align_up(addr, align) - addr
}

/// Storage a [`PixelBuffer`] can read as bytes.
///
/// Everything past this reference works on bytes, so code generic over storage
/// stops being generic here. Any [`SampleStorage`] implements it; so does
/// [`PixelStorage`], which holds one of several sample types.
pub trait ByteStorage {
    /// The samples as native-endian bytes, without copying.
    fn as_bytes(&self) -> &[u8];
}

/// Storage a [`PixelBuffer`] can write as bytes.
pub trait ByteStorageMut: ByteStorage {
    /// The samples as mutable native-endian bytes, without copying.
    fn as_bytes_mut(&mut self) -> &mut [u8];
}

impl<S: SampleStorage + ?Sized> ByteStorage for S {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self.samples())
    }
}

impl<S: SampleStorageMut + ?Sized> ByteStorageMut for S {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        bytemuck::cast_slice_mut(self.samples_mut())
    }
}

mod storage_sample {
    use super::{PixelStorage, Vec};

    /// Proves a call comes from this module, because a generic caller could otherwise
    /// reach these methods through a `StorageSample` bound.
    pub struct Private(pub(super) ());

    pub trait Sealed: Sized {
        fn into_storage(samples: Vec<Self>, _: Private) -> PixelStorage;
        fn from_storage(storage: PixelStorage, _: Private) -> Result<Vec<Self>, PixelStorage>;
        fn held_by(storage: &PixelStorage, _: Private) -> bool;
    }
}
use storage_sample::Private;

/// A sample type that [`PixelStorage`] can hold. Sealed.
///
/// Its conversions are private: use [`PixelStorage::from`] and the `PixelStorage` methods.
///
/// ```compile_fail
/// fn store<T: zenpixels::buffer::StorageSample>(samples: Vec<T>) {
///     let _ = T::into_storage(samples, Private(()));
/// }
/// ```
pub trait StorageSample: storage_sample::Sealed + bytemuck::Pod + Sized {}

/// Owned samples of the type a pixel format is made of.
///
/// This is the default storage of [`PixelBuffer`], so a decoder's `Vec<u16>` or
/// `Vec<f32>` crosses type-erased boundaries such as `DecodeOutput` and comes back
/// out without a copy. A `Vec` must free memory with the alignment it allocated, so
/// a `Vec<u8>` can never become a `Vec<u16>`, or the reverse, without copying;
/// holding the real sample type avoids that. Half-float samples use [`U16`](Self::U16).
///
/// Bytes are always available through [`ByteStorage`]. Only that reference branches
/// on the variant, so `row()` and kernels are unaffected.
#[derive(Clone)]
#[non_exhaustive]
pub enum PixelStorage {
    /// `u8` samples, or bytes adopted without knowing the sample type.
    U8(Vec<u8>),
    /// `i8` samples.
    I8(Vec<i8>),
    /// `u16` samples, including half floats.
    U16(Vec<u16>),
    /// `i16` samples.
    I16(Vec<i16>),
    /// `u32` samples.
    U32(Vec<u32>),
    /// `i32` samples.
    I32(Vec<i32>),
    /// `f32` samples.
    F32(Vec<f32>),
    /// `u64` samples.
    U64(Vec<u64>),
    /// `i64` samples.
    I64(Vec<i64>),
    /// `f64` samples.
    F64(Vec<f64>),
}

macro_rules! for_each_storage {
    ($value:expr, $vec:ident => $body:expr) => {
        match $value {
            PixelStorage::U8($vec) => $body,
            PixelStorage::I8($vec) => $body,
            PixelStorage::U16($vec) => $body,
            PixelStorage::I16($vec) => $body,
            PixelStorage::U32($vec) => $body,
            PixelStorage::I32($vec) => $body,
            PixelStorage::F32($vec) => $body,
            PixelStorage::U64($vec) => $body,
            PixelStorage::I64($vec) => $body,
            PixelStorage::F64($vec) => $body,
        }
    };
}

macro_rules! storage_samples {
    ($($sample:ty => $variant:ident),* $(,)?) => {$(
        impl storage_sample::Sealed for $sample {
            fn into_storage(samples: Vec<Self>, _: Private) -> PixelStorage {
                PixelStorage::$variant(samples)
            }
            fn from_storage(storage: PixelStorage, _: Private) -> Result<Vec<Self>, PixelStorage> {
                match storage {
                    PixelStorage::$variant(samples) => Ok(samples),
                    other => Err(other),
                }
            }
            fn held_by(storage: &PixelStorage, _: Private) -> bool {
                matches!(storage, PixelStorage::$variant(_))
            }
        }
        impl StorageSample for $sample {}
    )*};
}
storage_samples! {
    u8 => U8, i8 => I8, u16 => U16, i16 => I16, u32 => U32,
    i32 => I32, f32 => F32, u64 => U64, i64 => I64, f64 => F64,
}

impl<T: StorageSample> From<Vec<T>> for PixelStorage {
    /// Adopt the `Vec` without copying.
    fn from(samples: Vec<T>) -> Self {
        T::into_storage(samples, Private(()))
    }
}

impl PixelStorage {
    /// The `Vec`, when it holds samples of type `T`. Never copies.
    ///
    /// # Errors
    ///
    /// The storage, unchanged, when it holds another sample type.
    pub fn try_into_vec<T: StorageSample>(self) -> Result<Vec<T>, Self> {
        T::from_storage(self, Private(()))
    }

    /// Whether the storage holds samples of type `T`.
    #[cfg(feature = "image")]
    pub(crate) fn holds<T: StorageSample>(&self) -> bool {
        T::held_by(self, Private(()))
    }

    /// The samples as a `Vec` of pixels, moving the allocation.
    ///
    /// # Errors
    ///
    /// The storage, unchanged, unless the alignments match and the capacity is a
    /// whole number of pixels.
    #[cfg(feature = "imgref")]
    fn try_cast_pixels<P: bytemuck::Pod>(self) -> Result<Vec<P>, Self> {
        for_each_storage!(self, v => match bytemuck::allocation::try_cast_vec::<_, P>(v) {
            Ok(pixels) => Ok(pixels),
            Err((_, v)) => Err(v.into()),
        })
    }

    /// Storage of `channel`'s sample type holding `rows` appended in order. `rows` must
    /// be aligned for that sample type and total `bytes` bytes. Nothing is zeroed first.
    fn try_from_rows<'r>(
        channel: ChannelType,
        bytes: usize,
        rows: impl Iterator<Item = &'r [u8]>,
    ) -> Result<Self, BufferError> {
        fn gather<'r, T: StorageSample>(
            bytes: usize,
            rows: impl Iterator<Item = &'r [u8]>,
        ) -> Result<PixelStorage, BufferError> {
            let mut samples: Vec<T> = Vec::new();
            samples
                .try_reserve_exact(bytes / core::mem::size_of::<T>())
                .map_err(|_| BufferError::AllocationFailed)?;
            for row in rows {
                samples.extend_from_slice(bytemuck::cast_slice(row));
            }
            Ok(samples.into())
        }
        match channel {
            ChannelType::I8 => gather::<i8>(bytes, rows),
            ChannelType::U16 | ChannelType::F16 => gather::<u16>(bytes, rows),
            ChannelType::I16 => gather::<i16>(bytes, rows),
            ChannelType::U32 => gather::<u32>(bytes, rows),
            ChannelType::I32 => gather::<i32>(bytes, rows),
            ChannelType::F32 => gather::<f32>(bytes, rows),
            ChannelType::U64 => gather::<u64>(bytes, rows),
            ChannelType::I64 => gather::<i64>(bytes, rows),
            ChannelType::F64 => gather::<f64>(bytes, rows),
            _ => gather::<u8>(bytes, rows),
        }
    }

    /// Allocate zeroed storage of `channel`'s sample type holding `bytes` bytes.
    fn try_zeroed(channel: ChannelType, bytes: usize) -> Result<Self, BufferError> {
        // `try_zeroed_vec` goes through `alloc_zeroed`, so large buffers get untouched
        // zero pages from the OS, as `vec![0; n]` does, and it can still report failure.
        fn zeroed<T: StorageSample>(bytes: usize) -> Result<PixelStorage, BufferError> {
            let len = bytes.div_ceil(core::mem::size_of::<T>());
            let samples: Vec<T> = bytemuck::allocation::try_zeroed_vec(len)
                .map_err(|()| BufferError::AllocationFailed)?;
            Ok(samples.into())
        }
        match channel {
            ChannelType::I8 => zeroed::<i8>(bytes),
            ChannelType::U16 | ChannelType::F16 => zeroed::<u16>(bytes),
            ChannelType::I16 => zeroed::<i16>(bytes),
            ChannelType::U32 => zeroed::<u32>(bytes),
            ChannelType::I32 => zeroed::<i32>(bytes),
            ChannelType::F32 => zeroed::<f32>(bytes),
            ChannelType::U64 => zeroed::<u64>(bytes),
            ChannelType::I64 => zeroed::<i64>(bytes),
            ChannelType::F64 => zeroed::<f64>(bytes),
            _ => zeroed::<u8>(bytes),
        }
    }

    /// Drop samples past `bytes`, rounded up to a whole sample. Keeps the allocation.
    fn truncate_bytes(&mut self, bytes: usize) {
        for_each_storage!(self, v => {
            let size = size_of_element(v);
            v.truncate(bytes.div_ceil(size));
        });
    }

    /// Allocated capacity in bytes.
    #[cfg(test)]
    pub(crate) fn capacity_bytes(&self) -> usize {
        for_each_storage!(self, v => v.capacity() * size_of_element(v))
    }
}

/// Zeroed storage for `bytes` bytes of `descriptor`'s samples, first pixel aligned to
/// `align`. A sample-typed `Vec` already is unless `align` is larger, which only a
/// custom over-aligned pixel type asks for.
fn alloc_storage(
    descriptor: &PixelDescriptor,
    bytes: usize,
    align: usize,
) -> Result<(PixelStorage, u32), BufferError> {
    if align <= descriptor.min_alignment() {
        return Ok((
            PixelStorage::try_zeroed(descriptor.channel_type(), bytes)?,
            0,
        ));
    }
    let total = bytes
        .checked_add(align - 1)
        .ok_or(BufferError::InvalidDimensions)?;
    let data = PixelStorage::try_zeroed(ChannelType::U8, total)?;
    let offset = align_offset(data.as_bytes().as_ptr(), align);
    Ok((data, offset as u32))
}

fn size_of_element<T>(_: &[T]) -> usize {
    core::mem::size_of::<T>()
}

impl ByteStorage for PixelStorage {
    fn as_bytes(&self) -> &[u8] {
        for_each_storage!(self, v => bytemuck::cast_slice(v.as_slice()))
    }
}

impl ByteStorageMut for PixelStorage {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        for_each_storage!(self, v => bytemuck::cast_slice_mut(v.as_mut_slice()))
    }
}

impl fmt::Debug for PixelStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PixelStorage({} bytes)", self.as_bytes().len())
    }
}

// The enum's tag costs 8 bytes; `offset` is stored as `u32`, and that plus the
// struct's padding absorbs them, so the default buffer stays this size.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::size_of::<PixelBuffer>() == 72);

/// Validate slice parameters (shared by erased and typed constructors).
fn validate_slice(
    data_len: usize,
    data_ptr: *const u8,
    width: u32,
    rows: u32,
    stride_bytes: usize,
    descriptor: &PixelDescriptor,
) -> Result<(), BufferError> {
    descriptor.validate()?;
    if width == 0 || rows == 0 {
        return Ok(());
    }
    let bpp = descriptor.bytes_per_pixel();
    let min_stride = (width as usize)
        .checked_mul(bpp)
        .ok_or(BufferError::InvalidDimensions)?;
    if stride_bytes < min_stride {
        return Err(BufferError::StrideTooSmall);
    }
    let align = descriptor.min_alignment();
    if stride_bytes % align != 0 {
        return Err(BufferError::AlignmentViolation);
    }
    if rows > 0 {
        let required = required_bytes(rows, stride_bytes, min_stride)?;
        if data_len < required {
            return Err(BufferError::InsufficientData);
        }
    }
    if (data_ptr as usize) % align != 0 {
        return Err(BufferError::AlignmentViolation);
    }
    Ok(())
}

/// Whether a typed view of these rows has every row start aligned for `Q`. A sample-aligned
/// stride is enough for a pixel made of samples; a custom `Pixel` can ask for more.
fn rows_aligned_for<Q>(first_row: *const u8, width: u32, rows: u32, stride: usize) -> bool {
    let align = core::mem::align_of::<Q>();
    width == 0 || rows == 0 || ((first_row as usize) % align == 0 && stride % align == 0)
}

/// Minimum bytes needed: `(rows - 1) * stride + min_stride`.
fn required_bytes(rows: u32, stride: usize, min_stride: usize) -> Result<usize, BufferError> {
    if rows == 0 || min_stride == 0 {
        return Ok(0);
    }
    let preceding = (rows as usize - 1)
        .checked_mul(stride)
        .ok_or(BufferError::InvalidDimensions)?;
    preceding
        .checked_add(min_stride)
        .ok_or(BufferError::InvalidDimensions)
}

/// [`PixelBuffer::try_from_parts`]'s checks, on bytes so that they are compiled once
/// rather than once per storage type.
fn validate_parts(
    bytes: &[u8],
    offset: usize,
    width: u32,
    height: u32,
    stride_bytes: usize,
    descriptor: &PixelDescriptor,
) -> Result<(), BufferError> {
    match bytes.get(offset..) {
        Some(data) => validate_slice(
            data.len(),
            data.as_ptr(),
            width,
            height,
            stride_bytes,
            descriptor,
        ),
        None => Err(BufferError::InsufficientData),
    }
}

/// Convert `Vec<P>` to `Vec<u8>`. Zero-copy when alignment matches (u8-component
/// types), copies via `cast_slice` otherwise.
#[cfg(feature = "rgb")]
fn pixels_into_storage<P: bytemuck::Pod>(
    pixels: Vec<P>,
    channel: ChannelType,
) -> (PixelStorage, u32) {
    fn cast<P: bytemuck::Pod, T: StorageSample>(pixels: Vec<P>) -> Result<PixelStorage, Vec<P>> {
        bytemuck::allocation::try_cast_vec::<P, T>(pixels)
            .map(Into::into)
            .map_err(|(_, pixels)| pixels)
    }
    let moved = match channel {
        ChannelType::I8 => cast::<P, i8>(pixels),
        ChannelType::U16 | ChannelType::F16 => cast::<P, u16>(pixels),
        ChannelType::I16 => cast::<P, i16>(pixels),
        ChannelType::U32 => cast::<P, u32>(pixels),
        ChannelType::I32 => cast::<P, i32>(pixels),
        ChannelType::F32 => cast::<P, f32>(pixels),
        ChannelType::U64 => cast::<P, u64>(pixels),
        ChannelType::I64 => cast::<P, i64>(pixels),
        ChannelType::F64 => cast::<P, f64>(pixels),
        _ => cast::<P, u8>(pixels),
    };
    match moved {
        Ok(storage) => (storage, 0),
        Err(pixels) => {
            let (bytes, offset) = pixels_to_bytes(pixels);
            (bytes.into(), offset as u32)
        }
    }
}

/// Copy fallback for pixels whose alignment differs from their channel type's.
#[cfg(feature = "rgb")]
fn pixels_to_bytes<P: bytemuck::Pod>(pixels: Vec<P>) -> (Vec<u8>, usize) {
    match bytemuck::try_cast_vec(pixels) {
        Ok(bytes) => (bytes, 0),
        Err((_err, pixels)) => {
            let bytes = bytemuck::cast_slice::<P, u8>(&pixels);
            let align = core::mem::align_of::<P>();
            let mut data = alloc::vec![0; bytes.len().checked_add(align - 1).expect("pixel allocation overflow")];
            let offset = align_offset(data.as_ptr(), align);
            data[offset..offset + bytes.len()].copy_from_slice(bytes);
            (data, offset)
        }
    }
}

// ---------------------------------------------------------------------------
// PixelSlice (borrowed, immutable)
// ---------------------------------------------------------------------------

/// Borrowed view of pixel data.
///
/// Represents a contiguous region of pixel rows, possibly a sub-region
/// of a larger buffer. All rows share the same stride.
///
/// The type parameter `P` tracks the pixel format at compile time:
/// - `PixelSlice<'a, Rgb<u8>>` — known to be RGB8 pixels
/// - `PixelSlice<'a>` (= `PixelSlice<'a, ()>`) — type-erased, format in descriptor
///
/// Use [`new_typed()`](PixelSlice::new_typed) to create a typed slice, or
/// [`erase()`](PixelSlice::erase) / [`try_typed()`](PixelSlice::try_typed)
/// to convert between typed and erased forms.
///
/// Optionally carries [`ColorContext`] to track source color metadata
/// through the processing pipeline.
#[non_exhaustive]
pub struct PixelSlice<'a, P = ()> {
    data: &'a [u8],
    width: u32,
    rows: u32,
    stride: usize,
    descriptor: PixelDescriptor,
    color: Option<Arc<ColorContext>>,
    exact: Option<ExactDepth>,
    _pixel: PhantomData<P>,
}

impl<P> Clone for PixelSlice<'_, P> {
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            width: self.width,
            rows: self.rows,
            stride: self.stride,
            descriptor: self.descriptor,
            color: self.color.clone(),
            exact: self.exact,
            _pixel: PhantomData,
        }
    }
}

impl<'a> PixelSlice<'a> {
    /// Create a new pixel slice with validation.
    ///
    /// `stride_bytes` is the byte distance between the start of consecutive rows. It
    /// must be at least `width * bytes_per_pixel` and a multiple of the sample size.
    /// It need not be a whole number of pixels: RGB8 rows 8 bytes apart are fine.
    ///
    /// # Endianness contract
    ///
    /// Multi-byte samples (U16, F16, F32) are interpreted in **native byte
    /// order** — the bytes must already be host-endian. There is no
    /// endianness tag on [`PixelDescriptor`] (a BGR [`ChannelLayout`](crate::ChannelLayout)
    /// describes *channel* order, not byte order within a sample), so a
    /// decoder reading big-endian container data (e.g. PNG
    /// 16-bit) must byte-swap before wrapping the buffer in a slice.
    /// Handing over non-native bytes silently misinterprets every sample.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is too small, the stride is too small,
    /// or the data or stride is not aligned for the channel type.
    #[track_caller]
    pub fn new(
        data: &'a [u8],
        width: u32,
        rows: u32,
        stride_bytes: usize,
        descriptor: PixelDescriptor,
    ) -> Result<Self, At<BufferError>> {
        validate_slice(
            data.len(),
            data.as_ptr(),
            width,
            rows,
            stride_bytes,
            &descriptor,
        )
        .map_err(|e| whereat::at!(e))?;
        Ok(Self {
            data,
            width,
            rows,
            stride: stride_bytes,
            descriptor,

            color: None,
            exact: None,
            _pixel: PhantomData,
        })
    }
}

impl<'a, P> PixelSlice<'a, P> {
    /// Erase the pixel type, returning a type-erased slice.
    ///
    /// This is a zero-cost operation that just changes the type parameter.
    pub fn erase(self) -> PixelSlice<'a> {
        PixelSlice {
            data: self.data,
            width: self.width,
            rows: self.rows,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color,
            exact: self.exact,
            _pixel: PhantomData,
        }
    }

    /// Try to reinterpret as a typed pixel slice.
    ///
    /// Succeeds if the descriptors are layout-compatible (same channel type
    /// and layout) and every row start is aligned for `Q`. Transfer function and
    /// alpha mode are metadata, not layout constraints.
    ///
    /// The stride stays in bytes and need not be a whole number of `Q`s, so
    /// `bytemuck::cast_slice::<u8, Q>(slice.row(y))` works on any row.
    pub fn try_typed<Q: Pixel>(self) -> Option<PixelSlice<'a, Q>> {
        if self.descriptor.typed_view_compatible(Q::DESCRIPTOR)
            && core::mem::size_of::<Q>() == Q::DESCRIPTOR.bytes_per_pixel()
            && rows_aligned_for::<Q>(self.data.as_ptr(), self.width, self.rows, self.stride)
        {
            Some(PixelSlice {
                data: self.data,
                width: self.width,
                rows: self.rows,
                stride: self.stride,
                descriptor: self.descriptor,

                color: self.color,
                exact: self.exact,
                _pixel: PhantomData,
            })
        } else {
            None
        }
    }

    /// Replace the descriptor with a layout-compatible one.
    ///
    /// Use after a transform that changes pixel metadata without changing
    /// the buffer layout (e.g., transfer function change, alpha mode change,
    /// signal range expansion).
    ///
    /// For per-field updates, prefer the specific setters: [`with_transfer()`](Self::with_transfer),
    /// [`with_primaries()`](Self::with_primaries), [`with_signal_range()`](Self::with_signal_range),
    /// [`with_alpha_mode()`](Self::with_alpha_mode).
    ///
    /// # Panics
    ///
    /// Panics if the new descriptor is not layout-compatible (different
    /// `channel_type` or `layout`). Use [`reinterpret()`](Self::reinterpret)
    /// for genuine layout changes.
    #[inline]
    #[must_use]
    pub fn with_descriptor(mut self, descriptor: PixelDescriptor) -> Self {
        assert!(
            self.descriptor.layout_compatible(descriptor),
            "with_descriptor() cannot change physical layout ({} -> {}); \
             use reinterpret() for layout changes",
            self.descriptor,
            descriptor
        );
        descriptor.validate().expect("invalid pixel descriptor");
        if !exact_survives(self.descriptor, descriptor) {
            self.exact = None;
        }
        self.descriptor = descriptor;
        self
    }

    /// Reinterpret the buffer with a different physical layout.
    ///
    /// Unlike [`with_descriptor()`](Self::with_descriptor), this allows
    /// changing `channel_type` and `layout`. The new descriptor must have
    /// the same `bytes_per_pixel()` as the current one.
    ///
    /// Use cases: treating erased RGBA8 data as BGRA8, RGBX8 as RGBA8.
    /// Typed views reject a different channel layout/type. Call `erase()` first
    /// for a physical reinterpretation, then `try_typed()` to bind the new type.
    /// Destination alignment and descriptor validity are checked too.
    #[track_caller]
    pub fn reinterpret(mut self, descriptor: PixelDescriptor) -> Result<Self, At<BufferError>> {
        if self.descriptor.bytes_per_pixel() != descriptor.bytes_per_pixel()
            || (core::mem::size_of::<P>() != 0 && !self.descriptor.layout_compatible(descriptor))
        {
            return Err(whereat::at!(BufferError::IncompatibleDescriptor));
        }
        validate_slice(
            self.data.len(),
            self.data.as_ptr(),
            self.width,
            self.rows,
            self.stride,
            &descriptor,
        )
        .map_err(|e| whereat::at!(e))?;
        if !exact_survives(self.descriptor, descriptor) {
            self.exact = None;
        }
        self.descriptor = descriptor;
        Ok(self)
    }

    /// Return a copy with a different transfer function.
    #[inline]
    #[must_use]
    pub fn with_transfer(mut self, tf: TransferFunction) -> Self {
        self.descriptor.transfer = tf;
        self
    }

    /// Return a copy with different color primaries.
    #[inline]
    #[must_use]
    pub fn with_primaries(mut self, cp: ColorPrimaries) -> Self {
        self.descriptor.primaries = cp;
        self
    }

    /// Return a copy with a different signal range.
    #[inline]
    #[must_use]
    pub fn with_signal_range(mut self, sr: SignalRange) -> Self {
        if self.descriptor.signal_range != sr {
            self.exact = None;
        }
        self.descriptor.signal_range = sr;
        self
    }

    /// Return a copy with a different alpha mode.
    #[inline]
    #[must_use]
    pub fn with_alpha_mode(mut self, am: Option<AlphaMode>) -> Self {
        let descriptor = self.descriptor.with_alpha(am);
        descriptor.validate().expect("invalid alpha declaration");
        self.descriptor = descriptor;
        self
    }

    /// Evidence that the current samples are exact lower-depth codes, if stated.
    ///
    /// Borrowed views and clones keep it; a descriptor change that alters storage,
    /// layout or signal range drops it. See [`ExactDepth`].
    #[inline]
    pub fn exact_depth(&self) -> Option<ExactDepth> {
        self.exact
    }

    /// Attach exact-depth evidence for these samples.
    ///
    /// The statement is the caller's: only storage and range consistency are
    /// checked, without reading samples. State it where the samples were written
    /// by an exact widening, or after a scan.
    pub fn with_exact_depth(mut self, exact: ExactDepth) -> Result<Self, ExactDepthError> {
        check_exact(self.descriptor, exact)?;
        self.exact = Some(exact);
        Ok(self)
    }

    /// Image width in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Number of rows in this slice.
    #[inline]
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Byte stride between row starts.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Pixel format descriptor.
    #[inline]
    pub fn descriptor(&self) -> PixelDescriptor {
        self.descriptor
    }

    /// Source color context (ICC/CICP metadata), if set.
    #[inline]
    pub fn color_context(&self) -> Option<&Arc<ColorContext>> {
        self.color.as_ref()
    }

    /// Return a copy of this slice with a color context attached.
    #[inline]
    #[must_use]
    pub fn with_color_context(mut self, ctx: Arc<ColorContext>) -> Self {
        self.color = Some(ctx);
        self
    }

    /// Relabel the source diffuse-white anchor — the cd/m² that relative-linear
    /// `1.0` represents. **Reinterpretation only:** pixels are untouched (use a
    /// converter to change pixel values). Clones any existing [`ColorContext`],
    /// overrides its `diffuse_white`, and re-attaches it.
    #[must_use]
    pub fn with_diffuse_white(self, white: DiffuseWhite) -> Self {
        let ctx = self
            .color_context()
            .map(|c| (**c).clone())
            .unwrap_or_default()
            .with_diffuse_white(white);
        self.with_color_context(Arc::new(ctx))
    }

    /// Relabel the source [`Cicp`] colour signalling. **Reinterpretation only**
    /// — pixels are untouched. Clones any existing [`ColorContext`], sets
    /// `cicp`, and re-attaches it.
    #[must_use]
    pub fn with_cicp(self, cicp: Cicp) -> Self {
        let ctx = self
            .color_context()
            .map(|c| (**c).clone())
            .unwrap_or_default()
            .with_cicp(cicp);
        self.with_color_context(Arc::new(ctx))
    }

    /// Relabel the source ICC profile. **Reinterpretation only** — pixels are
    /// untouched. Clones any existing [`ColorContext`], sets `icc`, and
    /// re-attaches it.
    #[must_use]
    pub fn with_icc(self, icc: impl Into<Arc<[u8]>>) -> Self {
        let ctx = self
            .color_context()
            .map(|c| (**c).clone())
            .unwrap_or_default()
            .with_icc(icc);
        self.with_color_context(Arc::new(ctx))
    }

    /// Whether rows are tightly packed (no stride padding).
    ///
    /// When true, the entire pixel data is contiguous in memory and
    /// [`as_contiguous_bytes()`](Self::as_contiguous_bytes) returns `Some`.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.stride == self.width as usize * self.descriptor.bytes_per_pixel()
    }

    /// Zero-copy access to the raw pixel bytes when rows are tightly packed.
    ///
    /// Returns `Some(&[u8])` if `stride == width * bpp` (no padding),
    /// `None` if rows have stride padding.
    ///
    /// Use this to avoid `collect_contiguous_bytes()` copies when passing
    /// pixel data to FFI or other APIs that need a flat buffer.
    #[inline]
    pub fn as_contiguous_bytes(&self) -> Option<&'a [u8]> {
        if self.width == 0 || self.rows == 0 {
            return Some(&[]);
        }
        if self.is_contiguous() {
            let total = self.rows as usize * self.stride;
            Some(&self.data[..total])
        } else {
            None
        }
    }

    /// Copy the rows into a tight owned buffer of the format's own sample type, with the
    /// same color. It doesn't zero the allocation first, and the result can move to a
    /// typed owner.
    ///
    /// # Errors
    ///
    /// [`BufferError::InvalidDimensions`] when the size overflows,
    /// [`BufferError::AllocationFailed`] when the allocator refuses.
    pub fn try_to_tight_buffer(&self) -> Result<PixelBuffer, At<BufferError>> {
        let row_bytes = (self.width as usize)
            .checked_mul(self.descriptor.bytes_per_pixel())
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let total = row_bytes
            .checked_mul(self.rows as usize)
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        // A valid empty view may have no bytes at any row start, and an empty slice's
        // pointer need not be aligned for the sample type.
        let row_count = if total == 0 { 0 } else { self.rows as usize };
        let rows = (0..row_count).map(|y| {
            let start = y * self.stride;
            &self.data[start..start + row_bytes]
        });
        let data = PixelStorage::try_from_rows(self.descriptor.channel_type(), total, rows)
            .map_err(|e| whereat::at!(e))?;
        Ok(PixelBuffer {
            data,
            offset: 0,
            width: self.width,
            height: self.rows,
            stride: row_bytes,
            descriptor: self.descriptor,
            color: self.color.clone(),
            exact: self.exact,
            _pixel: PhantomData,
        })
    }

    /// Raw backing bytes including inter-row stride padding.
    ///
    /// Returns a `&[u8]` covering all rows with stride padding between
    /// them. Includes trailing padding after the last row when the
    /// backing buffer is large enough (e.g., from a full allocation),
    /// but not for sub-row views where the last row may be trimmed.
    ///
    /// Use with [`stride()`](Self::stride) for APIs that accept a byte
    /// buffer plus a stride value (GPU uploads, codec writers, etc).
    #[inline]
    pub fn as_strided_bytes(&self) -> &'a [u8] {
        if self.rows == 0 || self.width == 0 {
            return &[];
        }
        // Use rows*stride if the backing buffer is large enough (includes
        // trailing padding), otherwise trim the last row to width*bpp.
        let full = (self.rows as usize).saturating_mul(self.stride);
        if full <= self.data.len() {
            &self.data[..full]
        } else {
            let bpp = self.descriptor.bytes_per_pixel();
            let trimmed = (self.rows as usize - 1) * self.stride + self.width as usize * bpp;
            &self.data[..trimmed]
        }
    }

    /// Get the raw pixel bytes, copying only if stride padding exists.
    ///
    /// Returns `Cow::Borrowed` when rows are contiguous (zero-copy),
    /// `Cow::Owned` when stride padding must be stripped.
    pub fn contiguous_bytes(&self) -> alloc::borrow::Cow<'a, [u8]> {
        if let Some(bytes) = self.as_contiguous_bytes() {
            alloc::borrow::Cow::Borrowed(bytes)
        } else {
            let bpp = self.descriptor.bytes_per_pixel();
            let row_bytes = self.width as usize * bpp;
            let mut buf = Vec::with_capacity(row_bytes * self.rows as usize);
            for y in 0..self.rows {
                buf.extend_from_slice(self.row(y));
            }
            alloc::borrow::Cow::Owned(buf)
        }
    }

    /// Pixel bytes for row `y` (no padding, exactly `width * bpp` bytes).
    ///
    /// # Panics
    ///
    /// Panics if `y >= rows`.
    #[inline]
    pub fn row(&self, y: u32) -> &[u8] {
        assert!(
            y < self.rows,
            "row index {y} out of bounds (rows: {})",
            self.rows
        );
        if self.width == 0 {
            return &[];
        }
        let start = y as usize * self.stride;
        let len = self.width as usize * self.descriptor.bytes_per_pixel();
        &self.data[start..start + len]
    }

    /// Full stride bytes for row `y` (including any padding).
    ///
    /// # Panics
    ///
    /// Panics if `y >= rows` or if the underlying data does not contain
    /// a full stride for this row (can happen on the last row of a
    /// cropped view).
    #[inline]
    pub fn row_with_stride(&self, y: u32) -> &[u8] {
        assert!(
            y < self.rows,
            "row index {y} out of bounds (rows: {})",
            self.rows
        );
        let start = y as usize * self.stride;
        &self.data[start..start + self.stride]
    }

    /// Borrow a sub-range of rows.
    ///
    /// # Panics
    ///
    /// Panics if `y + count > rows`.
    pub fn sub_rows(&self, y: u32, count: u32) -> PixelSlice<'_, P> {
        assert!(
            y.checked_add(count).is_some_and(|end| end <= self.rows),
            "sub_rows({y}, {count}) out of bounds (rows: {})",
            self.rows
        );
        if count == 0 || self.width == 0 {
            return PixelSlice {
                data: &[],
                width: self.width,
                rows: count,
                stride: self.stride,
                descriptor: self.descriptor,

                color: self.color.clone(),
                exact: self.exact,
                _pixel: PhantomData,
            };
        }
        let bpp = self.descriptor.bytes_per_pixel();
        let start = y as usize * self.stride;
        let end = (y as usize + count as usize - 1) * self.stride + self.width as usize * bpp;
        PixelSlice {
            data: &self.data[start..end],
            width: self.width,
            rows: count,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color.clone(),
            exact: self.exact,
            _pixel: PhantomData,
        }
    }

    /// Zero-copy crop view. Adjusts the data pointer and width; stride
    /// remains the same as the parent.
    ///
    /// # Panics
    ///
    /// Panics if the crop region is out of bounds.
    pub fn crop_view(&self, x: u32, y: u32, w: u32, h: u32) -> PixelSlice<'_, P> {
        assert!(
            x.checked_add(w).is_some_and(|end| end <= self.width),
            "crop x={x} w={w} exceeds width {}",
            self.width
        );
        assert!(
            y.checked_add(h).is_some_and(|end| end <= self.rows),
            "crop y={y} h={h} exceeds rows {}",
            self.rows
        );
        if h == 0 || w == 0 {
            return PixelSlice {
                data: &[],
                width: w,
                rows: h,
                stride: self.stride,
                descriptor: self.descriptor,

                color: self.color.clone(),
                exact: self.exact,
                _pixel: PhantomData,
            };
        }
        let bpp = self.descriptor.bytes_per_pixel();
        let start = y as usize * self.stride + x as usize * bpp;
        let end = (y as usize + h as usize - 1) * self.stride + (x as usize + w as usize) * bpp;
        PixelSlice {
            data: &self.data[start..end],
            width: w,
            rows: h,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color.clone(),
            exact: self.exact,
            _pixel: PhantomData,
        }
    }
}

impl<'a, P: Pixel> PixelSlice<'a, P> {
    /// Create a typed pixel slice.
    ///
    /// `stride_pixels` is the number of pixels per row (>= width).
    /// The byte stride is `stride_pixels * size_of::<P>()`.
    ///
    /// # Compile-time safety
    ///
    /// Includes a compile-time assertion that `size_of::<P>()` matches
    /// `P::DESCRIPTOR.bytes_per_pixel()`, catching bad `Pixel` impls.
    #[track_caller]
    pub fn new_typed(
        data: &'a [u8],
        width: u32,
        rows: u32,
        stride_pixels: u32,
    ) -> Result<Self, At<BufferError>> {
        const { assert!(core::mem::size_of::<P>() == P::DESCRIPTOR.bytes_per_pixel()) }
        if width != 0 && rows != 0 && (data.as_ptr() as usize) % core::mem::align_of::<P>() != 0 {
            return Err(whereat::at!(BufferError::AlignmentViolation));
        }
        let stride_bytes = (stride_pixels as usize)
            .checked_mul(core::mem::size_of::<P>())
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        validate_slice(
            data.len(),
            data.as_ptr(),
            width,
            rows,
            stride_bytes,
            &P::DESCRIPTOR,
        )
        .map_err(|e| whereat::at!(e))?;
        Ok(Self {
            data,
            width,
            rows,
            stride: stride_bytes,
            descriptor: P::DESCRIPTOR,

            color: None,
            exact: None,
            _pixel: PhantomData,
        })
    }
}

impl<'a, P> PixelSlice<'a, P> {
    /// The pixel at `(x, y)`, or `None` when out of bounds.
    ///
    /// For one-off reads, tests and pickers. Looping over this is slow: use
    /// [`row`](Self::row) for many pixels. Wide formats, such as RGBA in `f64`, allocate.
    #[must_use]
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<crate::AnyColor> {
        if x >= self.width || y >= self.rows {
            return None;
        }
        let bytes = self.descriptor.bytes_per_pixel();
        let start = x as usize * bytes;
        let pixel = &self.row(y)[start..start + bytes];
        crate::AnyColor::try_new(self.descriptor, pixel).ok()
    }

    /// [`row`](Self::row), or `None` when `y` is out of bounds.
    #[inline]
    #[must_use]
    pub fn try_row(&self, y: u32) -> Option<&[u8]> {
        (y < self.rows).then(|| self.row(y))
    }

    /// [`row_with_stride`](Self::row_with_stride), or `None` when `y` is out of bounds.
    #[inline]
    #[must_use]
    pub fn try_row_with_stride(&self, y: u32) -> Option<&[u8]> {
        (y < self.rows).then(|| self.row_with_stride(y))
    }

    /// [`sub_rows`](Self::sub_rows), or `None` when the rows are out of bounds.
    #[must_use]
    pub fn try_sub_rows(&self, y: u32, count: u32) -> Option<PixelSlice<'_, P>> {
        y.checked_add(count)
            .is_some_and(|end| end <= self.rows)
            .then(|| self.sub_rows(y, count))
    }

    /// [`crop_view`](Self::crop_view), or `None` when the region is out of bounds.
    #[must_use]
    pub fn try_crop_view(&self, x: u32, y: u32, w: u32, h: u32) -> Option<PixelSlice<'_, P>> {
        let fits = x.checked_add(w).is_some_and(|end| end <= self.width)
            && y.checked_add(h).is_some_and(|end| end <= self.rows);
        fits.then(|| self.crop_view(x, y, w, h))
    }

    /// [`with_descriptor`](Self::with_descriptor) for a descriptor that may not fit.
    ///
    /// # Errors
    ///
    /// [`BufferError::IncompatibleDescriptor`] when it changes the physical layout
    /// or is invalid.
    pub fn try_with_descriptor(self, descriptor: PixelDescriptor) -> Result<Self, At<BufferError>> {
        if !self.descriptor.layout_compatible(descriptor) || descriptor.validate().is_err() {
            return Err(whereat::at!(BufferError::IncompatibleDescriptor));
        }
        Ok(self.with_descriptor(descriptor))
    }

    /// [`with_alpha_mode`](Self::with_alpha_mode) for a mode the layout may not allow.
    ///
    /// # Errors
    ///
    /// [`BufferError::IncompatibleDescriptor`] when the layout can't hold it.
    pub fn try_with_alpha_mode(self, am: Option<AlphaMode>) -> Result<Self, At<BufferError>> {
        if self.descriptor.with_alpha(am).validate().is_err() {
            return Err(whereat::at!(BufferError::IncompatibleDescriptor));
        }
        Ok(self.with_alpha_mode(am))
    }

    /// [`contiguous_bytes`](Self::contiguous_bytes), reporting a failed allocation.
    ///
    /// # Errors
    ///
    /// [`BufferError::AllocationFailed`] when padded rows can't be copied.
    pub fn try_contiguous_bytes(&self) -> Result<alloc::borrow::Cow<'a, [u8]>, At<BufferError>> {
        if let Some(bytes) = self.as_contiguous_bytes() {
            return Ok(alloc::borrow::Cow::Borrowed(bytes));
        }
        let row_bytes = self.width as usize * self.descriptor.bytes_per_pixel();
        let total = row_bytes
            .checked_mul(self.rows as usize)
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let mut buf = Vec::new();
        buf.try_reserve_exact(total)
            .map_err(|_| whereat::at!(BufferError::AllocationFailed))?;
        for y in 0..self.rows {
            buf.extend_from_slice(self.row(y));
        }
        Ok(alloc::borrow::Cow::Owned(buf))
    }
}

impl<P> fmt::Debug for PixelSlice<'_, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PixelSlice({}x{}, {:?} {:?})",
            self.width,
            self.rows,
            self.descriptor.layout(),
            self.descriptor.channel_type()
        )
    }
}

// ---------------------------------------------------------------------------
// PixelSliceMut (borrowed, mutable)
// ---------------------------------------------------------------------------

/// Mutable borrowed view of pixel data.
///
/// Same semantics as [`PixelSlice`] but allows writing to rows.
/// The type parameter `P` tracks pixel format at compile time.
#[non_exhaustive]
pub struct PixelSliceMut<'a, P = ()> {
    data: &'a mut [u8],
    width: u32,
    rows: u32,
    stride: usize,
    descriptor: PixelDescriptor,
    color: Option<Arc<ColorContext>>,
    _pixel: PhantomData<P>,
}

impl<'a> PixelSliceMut<'a> {
    /// Create a new mutable pixel slice with validation.
    ///
    /// `stride_bytes` is the byte distance between the start of consecutive rows. It
    /// must be at least `width * bytes_per_pixel` and a multiple of the sample size.
    /// It need not be a whole number of pixels.
    ///
    /// # Endianness contract
    ///
    /// Multi-byte samples (U16, F16, F32) are interpreted in **native byte
    /// order**; see [`PixelSlice::new`] — the same contract applies here.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is too small, the stride is too small,
    /// or the data or stride is not aligned for the channel type.
    #[track_caller]
    pub fn new(
        data: &'a mut [u8],
        width: u32,
        rows: u32,
        stride_bytes: usize,
        descriptor: PixelDescriptor,
    ) -> Result<Self, At<BufferError>> {
        validate_slice(
            data.len(),
            data.as_ptr(),
            width,
            rows,
            stride_bytes,
            &descriptor,
        )
        .map_err(|e| whereat::at!(e))?;
        Ok(Self {
            data,
            width,
            rows,
            stride: stride_bytes,
            descriptor,

            color: None,
            _pixel: PhantomData,
        })
    }
}

impl<'a, P> PixelSliceMut<'a, P> {
    /// Erase the pixel type, returning a type-erased mutable slice.
    pub fn erase(self) -> PixelSliceMut<'a> {
        PixelSliceMut {
            data: self.data,
            width: self.width,
            rows: self.rows,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color,
            _pixel: PhantomData,
        }
    }

    /// Try to reinterpret as a typed mutable pixel slice.
    ///
    /// See [`PixelSlice::try_typed()`] for the conditions.
    pub fn try_typed<Q: Pixel>(self) -> Option<PixelSliceMut<'a, Q>> {
        if self.descriptor.typed_view_compatible(Q::DESCRIPTOR)
            && core::mem::size_of::<Q>() == Q::DESCRIPTOR.bytes_per_pixel()
            && rows_aligned_for::<Q>(self.data.as_ptr(), self.width, self.rows, self.stride)
        {
            Some(PixelSliceMut {
                data: self.data,
                width: self.width,
                rows: self.rows,
                stride: self.stride,
                descriptor: self.descriptor,

                color: self.color,
                _pixel: PhantomData,
            })
        } else {
            None
        }
    }

    /// Replace the descriptor with a layout-compatible one.
    ///
    /// See [`PixelSlice::with_descriptor()`] for details.
    #[inline]
    #[must_use]
    pub fn with_descriptor(mut self, descriptor: PixelDescriptor) -> Self {
        assert!(
            self.descriptor.layout_compatible(descriptor),
            "with_descriptor() cannot change physical layout ({} -> {}); \
             use reinterpret() for layout changes",
            self.descriptor,
            descriptor
        );
        descriptor.validate().expect("invalid pixel descriptor");
        self.descriptor = descriptor;
        self
    }

    /// Reinterpret the buffer with a different physical layout.
    ///
    /// See [`PixelSlice::reinterpret()`] for details.
    #[track_caller]
    pub fn reinterpret(mut self, descriptor: PixelDescriptor) -> Result<Self, At<BufferError>> {
        if self.descriptor.bytes_per_pixel() != descriptor.bytes_per_pixel()
            || (core::mem::size_of::<P>() != 0 && !self.descriptor.layout_compatible(descriptor))
        {
            return Err(whereat::at!(BufferError::IncompatibleDescriptor));
        }
        validate_slice(
            self.data.len(),
            self.data.as_ptr(),
            self.width,
            self.rows,
            self.stride,
            &descriptor,
        )
        .map_err(|e| whereat::at!(e))?;
        self.descriptor = descriptor;
        Ok(self)
    }

    /// Return a copy with a different transfer function.
    #[inline]
    #[must_use]
    pub fn with_transfer(mut self, tf: TransferFunction) -> Self {
        self.descriptor.transfer = tf;
        self
    }

    /// Return a copy with different color primaries.
    #[inline]
    #[must_use]
    pub fn with_primaries(mut self, cp: ColorPrimaries) -> Self {
        self.descriptor.primaries = cp;
        self
    }

    /// Return a copy with a different signal range.
    #[inline]
    #[must_use]
    pub fn with_signal_range(mut self, sr: SignalRange) -> Self {
        self.descriptor.signal_range = sr;
        self
    }

    /// Return a copy with a different alpha mode.
    #[inline]
    #[must_use]
    pub fn with_alpha_mode(mut self, am: Option<AlphaMode>) -> Self {
        let descriptor = self.descriptor.with_alpha(am);
        descriptor.validate().expect("invalid alpha declaration");
        self.descriptor = descriptor;
        self
    }

    /// Image width in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Number of rows in this slice.
    #[inline]
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Byte stride between row starts.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Pixel format descriptor.
    #[inline]
    pub fn descriptor(&self) -> PixelDescriptor {
        self.descriptor
    }

    /// Source color context (ICC/CICP metadata), if set.
    #[inline]
    pub fn color_context(&self) -> Option<&Arc<ColorContext>> {
        self.color.as_ref()
    }

    /// Return a copy of this slice with a color context attached.
    #[inline]
    #[must_use]
    pub fn with_color_context(mut self, ctx: Arc<ColorContext>) -> Self {
        self.color = Some(ctx);
        self
    }

    /// Relabel the source diffuse-white anchor — the cd/m² that relative-linear
    /// `1.0` represents. **Reinterpretation only:** pixels are untouched (use a
    /// converter to change pixel values). Clones any existing [`ColorContext`],
    /// overrides its `diffuse_white`, and re-attaches it.
    #[must_use]
    pub fn with_diffuse_white(self, white: DiffuseWhite) -> Self {
        let ctx = self
            .color_context()
            .map(|c| (**c).clone())
            .unwrap_or_default()
            .with_diffuse_white(white);
        self.with_color_context(Arc::new(ctx))
    }

    /// Relabel the source [`Cicp`] colour signalling. **Reinterpretation only**
    /// — pixels are untouched. Clones any existing [`ColorContext`], sets
    /// `cicp`, and re-attaches it.
    #[must_use]
    pub fn with_cicp(self, cicp: Cicp) -> Self {
        let ctx = self
            .color_context()
            .map(|c| (**c).clone())
            .unwrap_or_default()
            .with_cicp(cicp);
        self.with_color_context(Arc::new(ctx))
    }

    /// Relabel the source ICC profile. **Reinterpretation only** — pixels are
    /// untouched. Clones any existing [`ColorContext`], sets `icc`, and
    /// re-attaches it.
    #[must_use]
    pub fn with_icc(self, icc: impl Into<Arc<[u8]>>) -> Self {
        let ctx = self
            .color_context()
            .map(|c| (**c).clone())
            .unwrap_or_default()
            .with_icc(icc);
        self.with_color_context(Arc::new(ctx))
    }

    /// Reborrow as an immutable [`PixelSlice`] (zero-copy).
    ///
    /// The returned slice borrows from `self`, so the mutable slice
    /// cannot be used while the immutable reborrow is alive.
    #[inline]
    pub fn as_pixel_slice(&self) -> PixelSlice<'_, P> {
        PixelSlice {
            data: self.data,
            width: self.width,
            rows: self.rows,
            stride: self.stride,
            descriptor: self.descriptor,
            color: self.color.clone(),
            exact: None,
            _pixel: PhantomData,
        }
    }

    /// Zero-copy access to the raw backing bytes, including any stride padding.
    ///
    /// Unlike [`PixelSlice::as_strided_bytes()`] (which clips to the image
    /// extent), this returns the full backing buffer so callers can write
    /// stride padding (zeroing, codec requirements, etc.).
    #[inline]
    pub fn as_strided_bytes(&self) -> &[u8] {
        self.data
    }

    /// Mutable access to the raw backing bytes, including any stride padding.
    ///
    /// Returns the full backing buffer so callers can write stride padding.
    #[inline]
    pub fn as_strided_bytes_mut(&mut self) -> &mut [u8] {
        self.data
    }

    /// Pixel bytes for row `y` (immutable, no padding).
    ///
    /// # Panics
    ///
    /// Panics if `y >= rows`.
    #[inline]
    pub fn row(&self, y: u32) -> &[u8] {
        assert!(
            y < self.rows,
            "row index {y} out of bounds (rows: {})",
            self.rows
        );
        if self.width == 0 {
            return &[];
        }
        let start = y as usize * self.stride;
        let len = self.width as usize * self.descriptor.bytes_per_pixel();
        &self.data[start..start + len]
    }

    /// Mutable pixel bytes for row `y` (no padding).
    ///
    /// # Panics
    ///
    /// Panics if `y >= rows`.
    #[inline]
    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(
            y < self.rows,
            "row index {y} out of bounds (rows: {})",
            self.rows
        );
        if self.width == 0 {
            return &mut [];
        }
        let start = y as usize * self.stride;
        let len = self.width as usize * self.descriptor.bytes_per_pixel();
        &mut self.data[start..start + len]
    }

    /// Borrow a mutable sub-range of rows.
    ///
    /// # Panics
    ///
    /// Panics if `y + count > rows`.
    pub fn sub_rows_mut(&mut self, y: u32, count: u32) -> PixelSliceMut<'_, P> {
        assert!(
            y.checked_add(count).is_some_and(|end| end <= self.rows),
            "sub_rows_mut({y}, {count}) out of bounds (rows: {})",
            self.rows
        );
        if count == 0 || self.width == 0 {
            return PixelSliceMut {
                data: &mut [],
                width: self.width,
                rows: count,
                stride: self.stride,
                descriptor: self.descriptor,

                color: self.color.clone(),
                _pixel: PhantomData,
            };
        }
        let bpp = self.descriptor.bytes_per_pixel();
        let start = y as usize * self.stride;
        let end = (y as usize + count as usize - 1) * self.stride + self.width as usize * bpp;
        PixelSliceMut {
            data: &mut self.data[start..end],
            width: self.width,
            rows: count,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color.clone(),
            _pixel: PhantomData,
        }
    }
}

impl<'a, P: Pixel> PixelSliceMut<'a, P> {
    /// Create a typed mutable pixel slice.
    ///
    /// `stride_pixels` is the number of pixels per row (>= width).
    /// The byte stride is `stride_pixels * size_of::<P>()`.
    #[track_caller]
    pub fn new_typed(
        data: &'a mut [u8],
        width: u32,
        rows: u32,
        stride_pixels: u32,
    ) -> Result<Self, At<BufferError>> {
        const { assert!(core::mem::size_of::<P>() == P::DESCRIPTOR.bytes_per_pixel()) }
        if width != 0 && rows != 0 && (data.as_ptr() as usize) % core::mem::align_of::<P>() != 0 {
            return Err(whereat::at!(BufferError::AlignmentViolation));
        }
        let stride_bytes = (stride_pixels as usize)
            .checked_mul(core::mem::size_of::<P>())
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        validate_slice(
            data.len(),
            data.as_ptr(),
            width,
            rows,
            stride_bytes,
            &P::DESCRIPTOR,
        )
        .map_err(|e| whereat::at!(e))?;
        Ok(Self {
            data,
            width,
            rows,
            stride: stride_bytes,
            descriptor: P::DESCRIPTOR,

            color: None,
            _pixel: PhantomData,
        })
    }
}

impl<P> PixelSliceMut<'_, P> {
    /// [`row`](Self::row), or `None` when `y` is out of bounds.
    #[inline]
    #[must_use]
    pub fn try_row(&self, y: u32) -> Option<&[u8]> {
        (y < self.rows).then(|| self.row(y))
    }

    /// [`row_mut`](Self::row_mut), or `None` when `y` is out of bounds.
    #[inline]
    #[must_use]
    pub fn try_row_mut(&mut self, y: u32) -> Option<&mut [u8]> {
        (y < self.rows).then(|| self.row_mut(y))
    }

    /// [`sub_rows_mut`](Self::sub_rows_mut), or `None` when the rows are out of bounds.
    #[must_use]
    pub fn try_sub_rows_mut(&mut self, y: u32, count: u32) -> Option<PixelSliceMut<'_, P>> {
        y.checked_add(count)
            .is_some_and(|end| end <= self.rows)
            .then(|| self.sub_rows_mut(y, count))
    }

    /// [`with_descriptor`](Self::with_descriptor) for a descriptor that may not fit.
    ///
    /// # Errors
    ///
    /// [`BufferError::IncompatibleDescriptor`] when it changes the physical layout
    /// or is invalid.
    pub fn try_with_descriptor(self, descriptor: PixelDescriptor) -> Result<Self, At<BufferError>> {
        if !self.descriptor.layout_compatible(descriptor) || descriptor.validate().is_err() {
            return Err(whereat::at!(BufferError::IncompatibleDescriptor));
        }
        Ok(self.with_descriptor(descriptor))
    }

    /// [`with_alpha_mode`](Self::with_alpha_mode) for a mode the layout may not allow.
    ///
    /// # Errors
    ///
    /// [`BufferError::IncompatibleDescriptor`] when the layout can't hold it.
    pub fn try_with_alpha_mode(self, am: Option<AlphaMode>) -> Result<Self, At<BufferError>> {
        if self.descriptor.with_alpha(am).validate().is_err() {
            return Err(whereat::at!(BufferError::IncompatibleDescriptor));
        }
        Ok(self.with_alpha_mode(am))
    }
}

impl<P> fmt::Debug for PixelSliceMut<'_, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PixelSliceMut({}x{}, {:?} {:?})",
            self.width,
            self.rows,
            self.descriptor.layout(),
            self.descriptor.channel_type()
        )
    }
}

// ---------------------------------------------------------------------------
// In-place 32-bit pixel format conversions on PixelSliceMut
// ---------------------------------------------------------------------------

/// Helper: iterate over pixel rows, calling `f` on each 4-byte pixel.
fn for_each_pixel_4bpp(
    data: &mut [u8],
    width: u32,
    rows: u32,
    stride: usize,
    mut f: impl FnMut(&mut [u8; 4]),
) {
    let row_bytes = width as usize * 4;
    if row_bytes == 0 {
        return;
    }
    for y in 0..rows as usize {
        let row_start = y * stride;
        let row = &mut data[row_start..row_start + row_bytes];
        for chunk in row.chunks_exact_mut(4) {
            let px: &mut [u8; 4] = chunk.try_into().unwrap();
            f(px);
        }
    }
}

impl<'a> PixelSliceMut<'a, Rgbx> {
    /// Byte-swap R<->B channels in place, converting to BGRX.
    #[must_use]
    pub fn swap_to_bgrx(self) -> PixelSliceMut<'a, Bgrx> {
        let width = self.width;
        let rows = self.rows;
        let stride = self.stride;

        let mut descriptor = self.descriptor;
        descriptor.format = crate::PixelFormat::Bgrx8;
        let color = self.color;
        let data = self.data;
        for_each_pixel_4bpp(data, width, rows, stride, |px| {
            px.swap(0, 2);
        });
        PixelSliceMut {
            data,
            width,
            rows,
            stride,
            descriptor,

            color,
            _pixel: PhantomData,
        }
    }
}

#[cfg(feature = "rgb")]
impl<'a> PixelSliceMut<'a, Rgbx> {}

impl<'a> PixelSliceMut<'a, Bgrx> {
    /// Byte-swap B<->R channels in place, converting to RGBX.
    #[must_use]
    pub fn swap_to_rgbx(self) -> PixelSliceMut<'a, Rgbx> {
        let width = self.width;
        let rows = self.rows;
        let stride = self.stride;

        let mut descriptor = self.descriptor;
        descriptor.format = crate::PixelFormat::Rgbx8;
        let color = self.color;
        let data = self.data;
        for_each_pixel_4bpp(data, width, rows, stride, |px| {
            px.swap(0, 2);
        });
        PixelSliceMut {
            data,
            width,
            rows,
            stride,
            descriptor,

            color,
            _pixel: PhantomData,
        }
    }
}

#[cfg(feature = "rgb")]
impl<'a> PixelSliceMut<'a, Bgrx> {}

#[cfg(feature = "rgb")]
impl<'a> PixelSliceMut<'a, Rgba<u8>> {
    /// Byte-swap R<->B channels in place, converting to BGRA.
    #[must_use]
    pub fn swap_to_bgra(self) -> PixelSliceMut<'a, BGRA<u8>> {
        let width = self.width;
        let rows = self.rows;
        let stride = self.stride;

        let mut descriptor = self.descriptor;
        descriptor.format = crate::PixelFormat::Bgra8;
        let color = self.color;
        let data = self.data;
        for_each_pixel_4bpp(data, width, rows, stride, |px| {
            px.swap(0, 2);
        });
        PixelSliceMut {
            data,
            width,
            rows,
            stride,
            descriptor,

            color,
            _pixel: PhantomData,
        }
    }
}

#[cfg(feature = "rgb")]
impl<'a> PixelSliceMut<'a, BGRA<u8>> {
    /// Byte-swap B<->R channels in place, converting to RGBA.
    #[must_use]
    pub fn swap_to_rgba(self) -> PixelSliceMut<'a, Rgba<u8>> {
        let width = self.width;
        let rows = self.rows;
        let stride = self.stride;

        let mut descriptor = self.descriptor;
        descriptor.format = crate::PixelFormat::Rgba8;
        let color = self.color;
        let data = self.data;
        for_each_pixel_4bpp(data, width, rows, stride, |px| {
            px.swap(0, 2);
        });
        PixelSliceMut {
            data,
            width,
            rows,
            stride,
            descriptor,

            color,
            _pixel: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// PixelBuffer (owned, pool-friendly)
// ---------------------------------------------------------------------------

/// Owned pixel buffer with format metadata.
///
/// Owns its samples through `S`, with an optional alignment offset so that
/// pixel rows start at the correct alignment for the channel type. The storage
/// and its layout/metadata can be recovered together with
/// [`into_parts`](Self::into_parts). Use [`into_contiguous`](Self::into_contiguous)
/// first when the receiver needs tightly packed rows.
///
/// The type parameter `P` tracks pixel format at compile time, same as
/// [`PixelSlice`].
///
/// `S` defaults to [`PixelStorage`], which owns U8, U16 or F32 samples;
/// [`PixelBuffer::try_new_aligned`] makes [`AlignedSamples`].
/// [`PixelBuffer::try_from_parts`] adopts any
/// [`SampleStorage`] without copying: a `Vec<u16>` or
/// `Vec<f32>` from another crate, a `Box<[T]>`, a borrowed slice, or a pool slot.
/// [`SampleStorageMut`] storage also gets the
/// mutable views.
#[non_exhaustive]
pub struct PixelBuffer<P = (), S = PixelStorage> {
    data: S,
    /// Byte offset from `data` start to the first pixel. `u32` keeps the default
    /// buffer at 72 bytes; `try_from_parts` refuses anything larger.
    offset: u32,
    width: u32,
    height: u32,
    stride: usize,
    descriptor: PixelDescriptor,
    color: Option<Arc<ColorContext>>,
    /// Current-sample evidence; cleared by every mutable access.
    exact: Option<ExactDepth>,
    _pixel: PhantomData<P>,
}

/// An owned allocation with the layout and color metadata of its pixels.
///
/// Returned by [`PixelBuffer::into_parts`]. Extraction moves the allocation and
/// color context without copying pixels or cloning the context. Destructure with
/// `..` to allow future metadata fields. The pixel type parameter is erased;
/// [`descriptor`](Self::descriptor) describes the physical format and semantics.
///
/// The first pixel starts at [`offset`](Self::offset), which can be nonzero even
/// after [`PixelBuffer::into_contiguous`]. Row starts are `stride_bytes` apart;
/// only `width * descriptor.bytes_per_pixel()` bytes of each row are pixels.
/// Uncompacted storage can include a prefix, row padding and trailing bytes.
///
/// This is a transfer record, not a validated pixel buffer: changing its public
/// fields does not validate the resulting layout or convert the pixels. Use
/// [`PixelBuffer::try_from_parts`] to validate and adopt the edited record.
#[derive(Debug)]
#[non_exhaustive]
pub struct PixelBufferParts<S = PixelStorage> {
    /// Original backing storage, including any alignment prefix or padding.
    pub data: S,
    /// Byte offset to the first pixel (or the empty pixel span).
    pub offset: usize,
    /// Bytes between successive row starts.
    pub stride_bytes: usize,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Physical pixel format and color semantics.
    pub descriptor: PixelDescriptor,
    /// Attached color context, moved from the buffer without an Arc clone.
    pub color_context: Option<Arc<ColorContext>>,
}

impl<S> PixelBufferParts<S> {
    /// Describe a native allocation for checked adoption without copying.
    ///
    /// Samples start at byte zero and no color context is attached. Set
    /// [`offset`](Self::offset) or [`color_context`](Self::color_context) before
    /// adoption when the native allocation has a prefix or color metadata.
    /// Construction does not validate or interpret the bytes; pass these parts
    /// to [`PixelBuffer::try_from_parts`] to check the extent and alignment.
    pub fn new(
        data: S,
        width: u32,
        height: u32,
        stride_bytes: usize,
        descriptor: PixelDescriptor,
    ) -> Self {
        Self {
            data,
            offset: 0,
            stride_bytes,
            width,
            height,
            descriptor,
            color_context: None,
        }
    }
}

/// Failed [`PixelBuffer::try_from_parts`] adoption, retaining the rejected storage.
///
/// Use [`take_parts`](Self::take_parts) to recover the allocation, or
/// `.map_err(|e| e.without_buffer())` when recovery is unnecessary. Strip the
/// buffer before boxing or storing an error to avoid retaining a large image.
/// Retaining the parts does not allocate or clone them. Capturing the validation
/// error uses the usual `whereat` diagnostic trace allocation.
#[must_use]
pub struct FromPartsError<S = PixelStorage> {
    error: At<BufferError>,
    parts: Option<PixelBufferParts<S>>,
}

impl<S> FromPartsError<S> {
    /// The validation error, including its location trace.
    pub fn error(&self) -> &At<BufferError> {
        &self.error
    }

    /// Take the rejected allocation and metadata, leaving this error intact.
    ///
    /// Returns `None` after the parts have already been taken.
    pub fn take_parts(&mut self) -> Option<PixelBufferParts<S>> {
        self.parts.take()
    }

    /// Discard any retained parts and return only the traced validation error.
    ///
    /// Prefer `.map_err(|e| e.without_buffer())?` before boxing or retaining an
    /// error if the caller does not need to recover its potentially large buffer.
    pub fn without_buffer(self) -> At<BufferError> {
        self.error
    }
}

impl<S> fmt::Debug for FromPartsError<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never dump a potentially 100 MB allocation into logs.
        f.debug_struct("FromPartsError")
            .field("error", &self.error)
            .field("has_parts", &self.parts.is_some())
            .finish()
    }
}

impl<S> fmt::Display for FromPartsError<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}

impl<S> core::error::Error for FromPartsError<S> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Borrowed-or-owned pixels with the same metadata and stride-aware view.
///
/// This is the pixel-buffer analogue of [`alloc::borrow::Cow`]: operations
/// can return a borrowed [`PixelSlice`] on their fast path and an owned
/// [`PixelBuffer`] only when conversion or mutation requires allocation.
pub enum PixelCow<'a, P = ()> {
    /// Pixels borrowed from the caller.
    Borrowed(PixelSlice<'a, P>),
    /// Independently owned pixels.
    Owned(PixelBuffer<P>),
}

impl<P> PixelCow<'_, P> {
    /// Whether this value still borrows its pixel storage.
    #[inline]
    pub const fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed(_))
    }

    /// Whether this value owns its pixel storage.
    #[inline]
    pub const fn is_owned(&self) -> bool {
        matches!(self, Self::Owned(_))
    }

    /// Borrow either representation as a uniform pixel slice.
    pub fn as_slice(&self) -> PixelSlice<'_, P> {
        match self {
            Self::Borrowed(slice) => slice.clone(),
            Self::Owned(buffer) => buffer.as_slice(),
        }
    }
}

impl<'a> PixelCow<'a> {
    /// Obtain mutable owned pixels, cloning the borrowed pixels if needed.
    pub fn to_mut(&mut self) -> &mut PixelBuffer {
        if let Self::Borrowed(slice) = self {
            let owned = pixel_slice_to_owned(slice);
            *self = Self::Owned(owned);
        }
        match self {
            Self::Owned(buffer) => buffer,
            Self::Borrowed(_) => unreachable!(),
        }
    }

    /// Consume this value, cloning the pixels only when they are borrowed.
    pub fn into_owned(self) -> PixelBuffer {
        match self {
            Self::Borrowed(slice) => pixel_slice_to_owned(&slice),
            Self::Owned(buffer) => buffer,
        }
    }
}

impl<'a, P> From<PixelSlice<'a, P>> for PixelCow<'a, P> {
    fn from(value: PixelSlice<'a, P>) -> Self {
        Self::Borrowed(value)
    }
}

impl<'a, P> From<PixelBuffer<P>> for PixelCow<'a, P> {
    fn from(value: PixelBuffer<P>) -> Self {
        Self::Owned(value)
    }
}

impl<P> fmt::Debug for PixelCow<'_, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Borrowed(slice) => f.debug_tuple("Borrowed").field(slice).finish(),
            Self::Owned(buffer) => f.debug_tuple("Owned").field(buffer).finish(),
        }
    }
}

impl<T: PlaneSample> PixelBuffer<(), AlignedSamples<T>> {
    /// Allocate a zeroed buffer with every row on a 64-byte boundary where the
    /// pixel size allows it, and the first pixel at index 0.
    ///
    /// The storage can always be viewed as the descriptor's pixel type. `T` is
    /// `u8` for any descriptor, or the descriptor's own channel type when the
    /// samples will move to an owner of that type.
    ///
    /// # Errors
    ///
    /// [`BufferError::IncompatibleDescriptor`] when `T` is neither,
    /// [`BufferError::InvalidDimensions`] when the size overflows,
    /// [`BufferError::AllocationFailed`] when the allocator refuses.
    pub fn try_new_aligned(
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
    ) -> Result<Self, At<BufferError>> {
        descriptor.validate().map_err(|e| whereat::at!(e))?;
        let sample = core::mem::size_of::<T>();
        if sample != 1 && T::CHANNEL_TYPE != descriptor.channel_type() {
            return Err(whereat::at!(BufferError::IncompatibleDescriptor));
        }
        (width as usize)
            .checked_mul(descriptor.bytes_per_pixel())
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let stride = descriptor.simd_aligned_stride(width, 64);
        let total = stride
            .checked_mul(height as usize)
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let data = AlignedSamples::try_zeroed(total / sample).map_err(|e| {
            whereat::at!(match e {
                PlaneError::AllocationFailed => BufferError::AllocationFailed,
                _ => BufferError::InvalidDimensions,
            })
        })?;
        Ok(Self {
            data,
            offset: 0,
            width,
            height,
            stride,
            descriptor,
            color: None,
            exact: None,
            _pixel: PhantomData,
        })
    }
}

impl<S: ByteStorage> PixelBuffer<(), S> {
    /// Adopt an allocation, geometry and color context without copying pixels.
    ///
    /// Validates offset, byte stride, sample alignment and the minimum visible
    /// extent, using the same storage checks as [`PixelSlice::new`]. The last row
    /// need not include trailing padding. Successful adoption does not allocate,
    /// compact or clone context; pointer, length, capacity, offset and metadata
    /// are preserved. Failure captures a `whereat` diagnostic trace.
    /// Color declarations are retained, not resolved or converted.
    ///
    /// On failure the error owns the original parts. Recover them with
    /// [`FromPartsError::take_parts`], or strip them before propagating, boxing or
    /// storing the error if recovery is unnecessary:
    ///
    /// ```
    /// use zenpixels::{BufferError, PixelBuffer, PixelDescriptor};
    /// # fn main() -> Result<(), whereat::At<BufferError>> {
    /// let parts = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB).into_parts();
    /// let buffer = PixelBuffer::try_from_parts(parts).map_err(|e| e.without_buffer())?;
    /// assert_eq!(buffer.width(), 2);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ```
    /// use zenpixels::ByteStorage;
    /// use zenpixels::{PixelBuffer, PixelDescriptor};
    /// let mut parts = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB).into_parts();
    /// parts.offset = parts.data.as_bytes().len() + 1;
    /// let mut error = PixelBuffer::try_from_parts(parts).unwrap_err();
    /// let mut parts = error.take_parts().unwrap();
    /// assert!(error.take_parts().is_none());
    /// parts.offset = 0;
    /// let buffer = PixelBuffer::try_from_parts(parts).unwrap();
    /// ```
    #[track_caller]
    #[allow(clippy::result_large_err)] // Keep rejected parts inline, without allocating an error box.
    pub fn try_from_parts(parts: PixelBufferParts<S>) -> Result<Self, FromPartsError<S>> {
        let validation = validate_parts(
            parts.data.as_bytes(),
            parts.offset,
            parts.width,
            parts.height,
            parts.stride_bytes,
            &parts.descriptor,
        );
        let stored = u32::try_from(parts.offset);
        let validation = validation.and_then(|()| {
            stored
                .map(|_| ())
                .map_err(|_| BufferError::InvalidDimensions)
        });
        if let Err(error) = validation {
            return Err(FromPartsError {
                error: whereat::at!(error),
                parts: Some(parts),
            });
        }
        let stored_offset = stored.unwrap_or_default();
        Ok(Self {
            data: parts.data,
            offset: stored_offset,
            width: parts.width,
            height: parts.height,
            stride: parts.stride_bytes,
            descriptor: parts.descriptor,
            color: parts.color_context,
            exact: None,
            _pixel: PhantomData,
        })
    }
}

impl PixelBuffer {
    /// Allocate a zero-filled buffer for the given dimensions and format.
    ///
    /// The default path is infallible for speed — see the
    /// [allocation policy](crate#allocation-policy) for the rationale and
    /// for guidance on when to reach for the `try_*` sibling.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails. Use [`try_new`](Self::try_new) for
    /// fallible allocation when handling untrusted sizes or when OOM must
    /// be recoverable.
    pub fn new(width: u32, height: u32, descriptor: PixelDescriptor) -> Self {
        Self::try_new(width, height, descriptor).expect("pixel buffer allocation failed")
    }

    /// Try to allocate a zero-filled buffer for the given dimensions and format.
    ///
    /// Returns [`BufferError::InvalidDimensions`] if the total size overflows,
    /// or [`BufferError::AllocationFailed`] if allocation fails.
    #[track_caller]
    pub fn try_new(
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
    ) -> Result<Self, At<BufferError>> {
        descriptor.validate().map_err(|e| whereat::at!(e))?;
        let stride = (width as usize)
            .checked_mul(descriptor.bytes_per_pixel())
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let total = stride
            .checked_mul(height as usize)
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let (data, offset) = alloc_storage(&descriptor, total, descriptor.min_alignment())
            .map_err(|e| whereat::at!(e))?;
        Ok(Self {
            data,
            offset,
            width,
            height,
            stride,
            descriptor,

            color: None,
            exact: None,
            _pixel: PhantomData,
        })
    }

    /// Wrap an existing `Vec<u8>` as a pixel buffer, tightly packed.
    ///
    /// The pixels start at byte 0. The `Vec` is adopted without copying when its
    /// first byte meets the format's alignment, which general-purpose allocators
    /// give in practice. A `Vec<u8>` carries no such guarantee, so on an allocator
    /// that doesn't, the pixels are copied into correctly typed storage. A `Vec` of
    /// the sample type, through [`PixelBufferParts`], never needs that.
    ///
    /// # Errors
    ///
    /// [`BufferError::InsufficientData`] when `data` holds fewer than
    /// `width * height` pixels.
    #[track_caller]
    pub fn from_vec(
        data: Vec<u8>,
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
    ) -> Result<Self, At<BufferError>> {
        descriptor.validate().map_err(|e| whereat::at!(e))?;
        let stride = (width as usize)
            .checked_mul(descriptor.bytes_per_pixel())
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let total = stride
            .checked_mul(height as usize)
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        if data.len() < total {
            return Err(whereat::at!(BufferError::InsufficientData));
        }
        let data: PixelStorage = if align_offset(data.as_ptr(), descriptor.min_alignment()) == 0 {
            data.into()
        } else {
            let mut typed = PixelStorage::try_zeroed(descriptor.channel_type(), total)
                .map_err(|e| whereat::at!(e))?;
            typed.as_bytes_mut()[..total].copy_from_slice(&data[..total]);
            typed
        };
        let offset = 0;
        Ok(Self {
            data,
            offset,
            width,
            height,
            stride,
            descriptor,

            color: None,
            exact: None,
            _pixel: PhantomData,
        })
    }
}

impl<P: Pixel> PixelBuffer<P> {
    /// Allocate a typed zero-filled buffer for the given dimensions.
    ///
    /// The descriptor is derived from `P::DESCRIPTOR`.
    ///
    /// The default path is infallible for speed — see the
    /// [allocation policy](crate#allocation-policy) for the rationale and
    /// for guidance on when to reach for the `try_*` sibling.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails. Use
    /// [`try_new_typed`](Self::try_new_typed) for fallible allocation when
    /// handling untrusted sizes or when OOM must be recoverable.
    pub fn new_typed(width: u32, height: u32) -> Self {
        Self::try_new_typed(width, height).expect("typed pixel buffer allocation failed")
    }

    /// Try to allocate a typed zero-filled buffer for the given dimensions.
    ///
    /// Returns [`BufferError::InvalidDimensions`] if the total size overflows,
    /// or [`BufferError::AllocationFailed`] if allocation fails.
    #[track_caller]
    pub fn try_new_typed(width: u32, height: u32) -> Result<Self, At<BufferError>> {
        const { assert!(core::mem::size_of::<P>() == P::DESCRIPTOR.bytes_per_pixel()) }
        let descriptor = P::DESCRIPTOR;
        descriptor.validate().map_err(|e| whereat::at!(e))?;
        let stride = (width as usize)
            .checked_mul(descriptor.bytes_per_pixel())
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let total = stride
            .checked_mul(height as usize)
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let align = descriptor.min_alignment().max(core::mem::align_of::<P>());
        let (data, offset) =
            alloc_storage(&descriptor, total, align).map_err(|e| whereat::at!(e))?;
        Ok(Self {
            data,
            offset,
            width,
            height,
            stride,
            descriptor,

            color: None,
            exact: None,
            _pixel: PhantomData,
        })
    }
}

#[cfg(feature = "rgb")]
impl<P: Pixel> PixelBuffer<P> {
    /// Construct from a typed pixel `Vec`.
    ///
    /// This selects sample storage from `P` and needs no integer inference on
    /// a `PixelStorage::from(vec![...])` call:
    ///
    /// ```
    /// use zenpixels::PixelBuffer;
    /// let pixels = vec![rgb::Rgb::new(0xa5_u8, 0, 0); 4];
    /// let buffer = PixelBuffer::from_pixels(pixels, 2, 2)?;
    /// assert_eq!(buffer.width(), 2);
    /// # Ok::<(), whereat::At<zenpixels::BufferError>>(())
    /// ```
    ///
    /// Zero-copy when `P` has alignment 1 (u8-component types like `Rgb<u8>`).
    /// Copies the data for types with higher alignment (`Rgb<u16>`, `Rgb<f32>`, etc.)
    /// because `Vec` tracks allocation alignment and `Vec<u8>` requires alignment 1.
    ///
    /// # Errors
    ///
    /// Returns [`BufferError::InvalidDimensions`] if `pixels.len() != width * height`.
    #[track_caller]
    pub fn from_pixels(pixels: Vec<P>, width: u32, height: u32) -> Result<Self, At<BufferError>> {
        const { assert!(core::mem::size_of::<P>() == P::DESCRIPTOR.bytes_per_pixel()) }
        // On 32-bit targets, `width * height` (as usize) can wrap to a small
        // value (e.g. 2^16 * 2^16 = 0), which would silently bypass the
        // length check below and accept a buffer of any size. Use
        // `checked_mul` so overflow is reported as `InvalidDimensions`
        // instead of producing a misshapen `PixelBuffer`.
        let expected = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        if pixels.len() != expected {
            return Err(whereat::at!(BufferError::InvalidDimensions));
        }
        let descriptor = P::DESCRIPTOR;
        descriptor.validate().map_err(|e| whereat::at!(e))?;
        let stride = (width as usize)
            .checked_mul(descriptor.bytes_per_pixel())
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let (data, offset) = pixels_into_storage(pixels, descriptor.channel_type());
        Ok(Self {
            data,
            offset,
            width,
            height,
            stride,
            descriptor,

            color: None,
            exact: None,
            _pixel: PhantomData,
        })
    }
}

#[cfg(feature = "imgref")]
impl<P: Pixel> PixelBuffer<P> {
    /// Construct from a typed `ImgVec`.
    ///
    /// Zero-copy when `P` has alignment 1 (u8-component types).
    /// Copies for higher-alignment types.
    ///
    /// # Panics
    ///
    /// Panics if the copy path (triggered for `P` with alignment > 1) fails
    /// to allocate. There is no fallible sibling today; if you need one,
    /// pre-convert the `ImgVec` into a `Vec<P>` and use
    /// [`try_new_typed`](Self::try_new_typed) followed by a manual copy, or
    /// file a request against the crate (see the
    /// [allocation policy](crate#allocation-policy)).
    pub fn from_imgvec(img: ImgVec<P>) -> Self {
        const { assert!(core::mem::size_of::<P>() == P::DESCRIPTOR.bytes_per_pixel()) }
        let width = u32::try_from(img.width()).expect("image width exceeds u32");
        let height = u32::try_from(img.height()).expect("image height exceeds u32");
        let stride_pixels = img.stride();
        let descriptor = P::DESCRIPTOR;
        descriptor.validate().expect("invalid pixel descriptor");
        let stride_bytes = stride_pixels
            .checked_mul(core::mem::size_of::<P>())
            .expect("image stride overflow");
        // Keep the actual backing storage and its stride; do not compact an
        // allocation only to retain the pre-compaction stride.
        let (data, offset) = pixels_into_storage(img.into_buf(), descriptor.channel_type());
        Self {
            data,
            offset,
            width,
            height,
            stride: stride_bytes,
            descriptor,

            color: None,
            exact: None,
            _pixel: PhantomData,
        }
    }
}

/// Move a typed buffer into an [`ImgVec`] without copying.
///
/// The error is the buffer, unchanged, when the first pixel is past index 0, the
/// stride is not a whole number of pixels, a dimension doesn't fit, or the allocation
/// can't become a `Vec<P>` (different alignment, or a capacity that isn't a whole
/// number of pixels).
#[cfg(feature = "imgref")]
impl<P: Pixel> TryFrom<PixelBuffer<P>> for ImgVec<P> {
    type Error = PixelBuffer<P>;
    fn try_from(buffer: PixelBuffer<P>) -> Result<Self, PixelBuffer<P>> {
        const { assert!(core::mem::size_of::<P>() == P::DESCRIPTOR.bytes_per_pixel()) }
        let size = core::mem::size_of::<P>();
        if buffer.offset != 0 || size == 0 || buffer.stride % size != 0 {
            return Err(buffer);
        }
        let (width, height) = (buffer.width as usize, buffer.height as usize);
        let stride = buffer.stride / size;
        // imgref requires a nonzero stride and a buffer covering every full row.
        let covers = stride
            .checked_mul(height)
            .is_some_and(|n| n.saturating_mul(size) <= buffer.data.as_bytes().len());
        if stride == 0 || !covers {
            return Err(buffer);
        }
        let PixelBuffer {
            data,
            offset,
            width: w,
            height: h,
            stride: stride_bytes,
            descriptor,
            color,
            exact,
            _pixel,
        } = buffer;
        match data.try_cast_pixels::<P>() {
            Ok(pixels) => Ok(imgref::Img::new_stride(pixels, width, height, stride)),
            Err(data) => Err(PixelBuffer {
                data,
                offset,
                width: w,
                height: h,
                stride: stride_bytes,
                descriptor,
                color,
                exact,
                _pixel,
            }),
        }
    }
}

#[cfg(feature = "imgref")]
impl<P: Pixel, S: ByteStorage> PixelBuffer<P, S> {
    /// Borrow the buffer as an [`ImgRef`].
    ///
    /// Zero-copy: reinterprets the raw bytes as typed pixels via
    /// [`bytemuck::cast_slice`].
    ///
    /// # Panics
    ///
    /// Panics if the stride is not a whole number of pixels, which `imgref` can't
    /// express. That can't happen for buffers created via `new_typed()`,
    /// `from_pixels()`, or `from_imgvec()`. For a buffer adopted from foreign rows,
    /// check `stride() % size_of::<P>()` first, or use
    /// [`try_as_imgref`](PixelBuffer::try_as_imgref) on the erased buffer.
    pub fn as_imgref(&self) -> ImgRef<'_, P> {
        assert_eq!(
            self.stride % core::mem::size_of::<P>(),
            0,
            "ImgRef requires a stride in whole pixels"
        );
        let total_bytes = if self.height == 0 || self.width == 0 {
            0
        } else {
            (self.height as usize - 1) * self.stride
                + self.width as usize * core::mem::size_of::<P>()
        };
        let data =
            &self.data.as_bytes()[(self.offset as usize)..(self.offset as usize) + total_bytes];
        let pixels: &[P] = if data.is_empty() {
            &[]
        } else {
            bytemuck::cast_slice(data)
        };
        // imgref requires positive stride even for zero-area views.
        let stride_px = (self.stride / core::mem::size_of::<P>()).max(1);
        imgref::Img::new_stride(pixels, self.width as usize, self.height as usize, stride_px)
    }
}

#[cfg(feature = "imgref")]
impl<P: Pixel, S: ByteStorageMut> PixelBuffer<P, S> {
    /// Borrow the buffer as a mutable [`ImgRefMut`](imgref::ImgRefMut).
    ///
    /// Zero-copy: reinterprets the raw bytes as typed pixels.
    ///
    /// # Panics
    ///
    /// Panics if the stride is not a whole number of pixels; see
    /// [`as_imgref`](Self::as_imgref).
    pub fn as_imgref_mut(&mut self) -> imgref::ImgRefMut<'_, P> {
        assert_eq!(
            self.stride % core::mem::size_of::<P>(),
            0,
            "ImgRef requires a stride in whole pixels"
        );
        // Mutable access can change any sample: end current-sample evidence.
        self.exact = None;
        let total_bytes = if self.height == 0 || self.width == 0 {
            0
        } else {
            (self.height as usize - 1) * self.stride
                + self.width as usize * core::mem::size_of::<P>()
        };
        let offset = self.offset as usize;
        let data = &mut self.data.as_bytes_mut()[offset..offset + total_bytes];
        let pixels: &mut [P] = if data.is_empty() {
            &mut []
        } else {
            bytemuck::cast_slice_mut(data)
        };
        // imgref requires positive stride even for zero-area views.
        let stride_px = (self.stride / core::mem::size_of::<P>()).max(1);
        imgref::Img::new_stride(pixels, self.width as usize, self.height as usize, stride_px)
    }
}

/// Type-erased typed pixel construction and access.
#[cfg(feature = "rgb")]
impl PixelBuffer {
    /// Zero-copy construction from typed pixels, returning an erased `PixelBuffer`.
    ///
    /// Equivalent to `PixelBuffer::<P>::from_pixels(pixels, w, h)?.into()` but
    /// avoids the intermediate typed `PixelBuffer`.
    ///
    /// # Errors
    ///
    /// Returns [`BufferError::InvalidDimensions`] if `pixels.len() != width * height`.
    #[track_caller]
    pub fn from_pixels_erased<P: Pixel>(
        pixels: Vec<P>,
        width: u32,
        height: u32,
    ) -> Result<Self, At<BufferError>> {
        PixelBuffer::<P>::from_pixels(pixels, width, height)
            .at()
            .map(PixelBuffer::from)
    }

    /// Zero-copy access to the pixel data as a typed slice.
    ///
    /// Returns `Some(&[P])` if the descriptor is layout-compatible with `P`
    /// and rows are tightly packed (no stride padding). Returns `None` otherwise.
    ///
    /// This is a convenience for the common case where you just need a flat
    /// pixel slice without imgref metadata. For padded rows, use
    /// [`try_as_imgref()`](Self::try_as_imgref) when the stride is a whole number of
    /// pixels, or cast each of [`rows`](PixelBuffer::rows).
    pub fn as_contiguous_pixels<P: Pixel>(&self) -> Option<&[P]> {
        if !self.descriptor.typed_view_compatible(P::DESCRIPTOR)
            || core::mem::size_of::<P>() != self.descriptor.bytes_per_pixel()
        {
            return None;
        }
        let pixel_size = core::mem::size_of::<P>();
        if self.width == 0 || self.height == 0 {
            return Some(&[]);
        }
        let row_bytes = self.width as usize * pixel_size;
        if pixel_size == 0 || self.stride != row_bytes {
            return None;
        }
        let total = row_bytes * self.height as usize;
        let data = &self.data.as_bytes()[(self.offset as usize)..(self.offset as usize) + total];
        if data.is_empty() {
            Some(&[])
        } else {
            bytemuck::try_cast_slice(data).ok()
        }
    }

    /// Consume the buffer and return the pixels as a typed `Vec<P>`.
    ///
    /// Returns `None` if the descriptor is not layout-compatible with `P`.
    /// Compacts stride padding in place. Reuses the allocation when `P` has
    /// alignment 1 and the allocation capacity is a multiple of its size.
    /// Other allocator layouts require one copy into a typed allocation.
    pub fn into_contiguous_pixels<P: Pixel>(self) -> Option<Vec<P>> {
        if !self.descriptor.typed_view_compatible(P::DESCRIPTOR)
            || core::mem::size_of::<P>() != self.descriptor.bytes_per_pixel()
        {
            return None;
        }
        let pixel_size = core::mem::size_of::<P>();
        if pixel_size == 0 {
            return None;
        }
        let total_pixels = self.width as usize * self.height as usize;
        if total_pixels == 0 {
            return Some(Vec::new());
        }

        let mut packed = self.into_contiguous();
        let len = total_pixels * pixel_size;
        let start = packed.offset as usize;
        if start != 0 {
            packed
                .data
                .as_bytes_mut()
                .copy_within(start..start + len, 0);
        }
        packed.data.truncate_bytes(len);
        // The cast moves the allocation when the alignments match and the capacity
        // is a whole number of pixels. Otherwise copy.
        Some(for_each_storage!(packed.data, v => {
            match bytemuck::allocation::try_cast_vec::<_, P>(v) {
                Ok(pixels) => pixels,
                Err((_, v)) => {
                    let bytes: &[u8] = bytemuck::cast_slice(v.as_slice());
                    bytes[..len]
                        .chunks_exact(pixel_size)
                        .map(bytemuck::pod_read_unaligned)
                        .collect()
                }
            }
        }))
    }
}

/// Imgref interop for type-erased PixelBuffer.
#[cfg(feature = "imgref")]
impl<S: ByteStorage> PixelBuffer<(), S> {
    /// Try to borrow the buffer as a typed [`ImgRef`].
    ///
    /// Returns `None` for incompatible pixels, insufficient alignment or a stride
    /// that is not a whole number of pixels. Row access still supports sample strides.
    #[track_caller]
    pub fn try_as_imgref<P: Pixel>(&self) -> Option<ImgRef<'_, P>> {
        let pixel_size = check_imgref::<P>(self.descriptor, self.stride).ok()?;
        let total_bytes = if self.height == 0 || self.width == 0 {
            0
        } else {
            (self.height as usize - 1) * self.stride + self.width as usize * pixel_size
        };
        let data =
            &self.data.as_bytes()[(self.offset as usize)..(self.offset as usize) + total_bytes];
        let pixels: &[P] = if data.is_empty() {
            &[]
        } else {
            bytemuck::try_cast_slice(data).ok()?
        };
        // imgref refuses a stride of 0, even for an empty image.
        let stride_px = (self.stride / pixel_size).max(1);
        Some(imgref::Img::new_stride(
            pixels,
            self.width as usize,
            self.height as usize,
            stride_px,
        ))
    }
}

#[cfg(feature = "imgref")]
impl<S: ByteStorageMut> PixelBuffer<(), S> {
    /// Try to borrow the buffer as a typed mutable [`ImgRefMut`](imgref::ImgRefMut).
    ///
    /// On refusal (`None`), exact-depth evidence is unchanged.
    ///
    /// Refusal conditions match [`try_as_imgref`](Self::try_as_imgref).
    #[track_caller]
    pub fn try_as_imgref_mut<P: Pixel>(&mut self) -> Option<imgref::ImgRefMut<'_, P>> {
        let pixel_size = check_imgref::<P>(self.descriptor, self.stride).ok()?;
        let total_bytes = if self.height == 0 || self.width == 0 {
            0
        } else {
            (self.height as usize - 1) * self.stride + self.width as usize * pixel_size
        };
        let offset = self.offset as usize;
        let bytes = self.data.as_bytes_mut();
        let data = &mut bytes[offset..offset + total_bytes];
        let pixels: &mut [P] = if data.is_empty() {
            &mut []
        } else {
            bytemuck::try_cast_slice_mut(data).ok()?
        };
        // Mutable access can change any sample: end current-sample evidence.
        self.exact = None;
        // imgref refuses a stride of 0, even for an empty image.
        let stride_px = (self.stride / pixel_size).max(1);
        Some(imgref::Img::new_stride(
            pixels,
            self.width as usize,
            self.height as usize,
            stride_px,
        ))
    }
}

/// What [`PixelBuffer::try_as_imgref`] checks before it reads any bytes. Returns the pixel size.
#[cfg(feature = "imgref")]
#[track_caller]
fn check_imgref<P: Pixel>(
    descriptor: PixelDescriptor,
    stride: usize,
) -> Result<usize, At<BufferError>> {
    let pixel_size = core::mem::size_of::<P>();
    if !descriptor.typed_view_compatible(P::DESCRIPTOR)
        || pixel_size != descriptor.bytes_per_pixel()
        || pixel_size == 0
    {
        return Err(whereat::at!(BufferError::IncompatibleDescriptor));
    }
    if stride % pixel_size != 0 {
        return Err(whereat::at!(BufferError::StrideNotPixelAligned));
    }
    Ok(pixel_size)
}

/// Everything a layout-changing in-place transform receives from
/// [`PixelBuffer::transform_in_place`]: the buffer's backing bytes (from
/// the aligned base, including stride padding) plus its current
/// description. Plain data — constructible directly (e.g. by transform
/// unit tests exercising arbitrary strides); the staleness protection
/// lives in `transform_in_place`'s atomic adoption, not here.
///
/// `#[non_exhaustive]`: construct it with [`InPlacePixels::new`] (inside
/// `transform_in_place` the buffer does this for you). Sealing lets future
/// fields — a cancellation token, a separate anchor — be added without a break.
#[non_exhaustive]
pub struct InPlacePixels<'a> {
    /// Full backing bytes from the aligned base.
    pub bytes: &'a mut [u8],
    /// Current width in pixels.
    pub width: u32,
    /// Current row count.
    pub rows: u32,
    /// Current byte stride between row starts.
    pub stride: usize,
    /// Current pixel format descriptor.
    pub descriptor: PixelDescriptor,
    /// Current color context, if any.
    pub color: Option<Arc<ColorContext>>,
}

impl<'a> InPlacePixels<'a> {
    /// Construct directly, panicking if the bytes and geometry disagree.
    ///
    /// Prefer [`try_new`](Self::try_new) when the geometry is supplied by an
    /// untrusted caller. Inside [`PixelBuffer::transform_in_place`] the buffer
    /// constructs this from its already-validated state.
    ///
    /// # Panics
    ///
    /// Panics for the same invalid geometry, undersized backing storage, or
    /// alignment errors reported by [`try_new`](Self::try_new).
    #[track_caller]
    #[must_use]
    pub fn new(
        bytes: &'a mut [u8],
        width: u32,
        rows: u32,
        stride: usize,
        descriptor: PixelDescriptor,
        color: Option<Arc<ColorContext>>,
    ) -> Self {
        Self::try_new(bytes, width, rows, stride, descriptor, color)
            .expect("invalid in-place pixel geometry")
    }

    /// Construct directly after validating storage, stride, and alignment.
    ///
    /// This applies the same invariants as [`PixelSliceMut::new`], preventing
    /// malformed transform inputs from reaching row-oriented conversion code.
    #[track_caller]
    pub fn try_new(
        bytes: &'a mut [u8],
        width: u32,
        rows: u32,
        stride: usize,
        descriptor: PixelDescriptor,
        color: Option<Arc<ColorContext>>,
    ) -> Result<Self, At<BufferError>> {
        validate_slice(
            bytes.len(),
            bytes.as_ptr(),
            width,
            rows,
            stride,
            &descriptor,
        )
        .map_err(|e| whereat::at!(e))?;
        Ok(Self::new_unchecked(
            bytes, width, rows, stride, descriptor, color,
        ))
    }

    #[inline]
    fn new_unchecked(
        bytes: &'a mut [u8],
        width: u32,
        rows: u32,
        stride: usize,
        descriptor: PixelDescriptor,
        color: Option<Arc<ColorContext>>,
    ) -> Self {
        Self {
            bytes,
            width,
            rows,
            stride,
            descriptor,
            color,
        }
    }
}

impl PixelBuffer {
    /// Run a layout-changing in-place transform over this buffer and
    /// **atomically adopt** the transform's output geometry, descriptor,
    /// and color context.
    ///
    /// This is the only supported way to rewrite a buffer into a
    /// different byte layout in place (lane drops, channel reorders,
    /// orientation bakes, load-bearing reductions — the
    /// `zenpixels-convert` in-place family is built on it). The closure
    /// receives the buffer's backing bytes (from the aligned base,
    /// including stride padding) plus the current description, rewrites
    /// the pixels, and returns a [`PixelSliceMut`] re-describing the
    /// **same memory** (built with [`PixelSliceMut::new`], so the new
    /// geometry is validated). On return, this buffer's own width /
    /// height / stride / descriptor / color context are updated in the
    /// same call — the buffer and its bytes can never disagree, which is
    /// the staleness hazard this method exists to prevent.
    ///
    /// # Panics
    ///
    /// Panics if the returned view does not start at this buffer's
    /// aligned base or does not fit inside the original allocation —
    /// both indicate a bug in the transform, not a recoverable
    /// condition (the bytes may already be partially rewritten).
    pub fn transform_in_place<F>(&mut self, f: F)
    where
        F: for<'a> FnOnce(InPlacePixels<'a>) -> PixelSliceMut<'a>,
    {
        // The transform rewrites samples in place: end current-sample evidence.
        self.exact = None;
        let total = self.storage_span_len();
        let offset = self.offset as usize;
        let avail = self.data.as_bytes().len() - offset;
        let base = self.data.as_bytes()[offset..].as_ptr() as usize;

        let out = f(InPlacePixels::new_unchecked(
            &mut self.data.as_bytes_mut()[offset..offset + total],
            self.width,
            self.height,
            self.stride,
            self.descriptor,
            self.color.clone(),
        ));

        let out_bytes = out.as_strided_bytes();
        assert_eq!(
            out_bytes.as_ptr() as usize,
            base,
            "transform_in_place: returned view must start at the buffer's aligned base"
        );
        assert!(
            out_bytes.len() <= avail,
            "transform_in_place: returned view exceeds the original allocation"
        );

        self.width = out.width();
        self.height = out.rows();
        self.stride = out.stride();
        self.descriptor = out.descriptor();
        self.color = out.color_context().cloned();
    }
}

impl<P, S: ByteStorage> PixelBuffer<P, S> {
    // Retain available final-row padding, but do not require it. Validation
    // establishes the visible extent; padding need not fit or even be representable.
    fn storage_span_len(&self) -> usize {
        self.stride
            .saturating_mul(self.height as usize)
            .min(self.data.as_bytes().len() - (self.offset as usize))
    }

    /// Erase the pixel type, returning a type-erased buffer.
    pub fn erase(self) -> PixelBuffer<(), S> {
        PixelBuffer {
            data: self.data,
            offset: self.offset,
            width: self.width,
            height: self.height,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color,
            exact: self.exact,
            _pixel: PhantomData,
        }
    }

    /// Try to reinterpret as a typed pixel buffer.
    ///
    /// See [`PixelSlice::try_typed()`] for the conditions. The stride may leave
    /// [`as_imgref`](PixelBuffer::as_imgref) unusable: it needs a whole number of pixels.
    pub fn try_typed<Q: Pixel>(self) -> Option<PixelBuffer<Q, S>> {
        if self.descriptor.typed_view_compatible(Q::DESCRIPTOR)
            && core::mem::size_of::<Q>() == Q::DESCRIPTOR.bytes_per_pixel()
            && rows_aligned_for::<Q>(
                self.data.as_bytes()[(self.offset as usize)..].as_ptr(),
                self.width,
                self.height,
                self.stride,
            )
        {
            Some(PixelBuffer {
                data: self.data,
                offset: self.offset,
                width: self.width,
                height: self.height,
                stride: self.stride,
                descriptor: self.descriptor,

                color: self.color,
                exact: self.exact,
                _pixel: PhantomData,
            })
        } else {
            None
        }
    }

    /// Replace the descriptor with a layout-compatible one.
    ///
    /// See [`PixelSlice::with_descriptor()`] for details.
    #[inline]
    #[must_use]
    pub fn with_descriptor(mut self, descriptor: PixelDescriptor) -> Self {
        assert!(
            self.descriptor.layout_compatible(descriptor),
            "with_descriptor() cannot change physical layout ({} -> {}); \
             use reinterpret() for layout changes",
            self.descriptor,
            descriptor
        );
        descriptor.validate().expect("invalid pixel descriptor");
        if !exact_survives(self.descriptor, descriptor) {
            self.exact = None;
        }
        self.descriptor = descriptor;
        self
    }

    /// Reinterpret the buffer with a different physical layout.
    ///
    /// See [`PixelSlice::reinterpret()`] for details.
    #[track_caller]
    pub fn reinterpret(mut self, descriptor: PixelDescriptor) -> Result<Self, At<BufferError>> {
        if self.descriptor.bytes_per_pixel() != descriptor.bytes_per_pixel()
            || (core::mem::size_of::<P>() != 0 && !self.descriptor.layout_compatible(descriptor))
        {
            return Err(whereat::at!(BufferError::IncompatibleDescriptor));
        }
        validate_slice(
            self.data.as_bytes().len() - (self.offset as usize),
            self.data.as_bytes()[(self.offset as usize)..].as_ptr(),
            self.width,
            self.height,
            self.stride,
            &descriptor,
        )
        .map_err(|e| whereat::at!(e))?;
        if !exact_survives(self.descriptor, descriptor) {
            self.exact = None;
        }
        self.descriptor = descriptor;
        Ok(self)
    }

    /// Return a copy with a different transfer function.
    #[inline]
    #[must_use]
    pub fn with_transfer(mut self, tf: TransferFunction) -> Self {
        self.descriptor.transfer = tf;
        self
    }

    /// Return a copy with different color primaries.
    #[inline]
    #[must_use]
    pub fn with_primaries(mut self, cp: ColorPrimaries) -> Self {
        self.descriptor.primaries = cp;
        self
    }

    /// Return a copy with a different signal range.
    #[inline]
    #[must_use]
    pub fn with_signal_range(mut self, sr: SignalRange) -> Self {
        if self.descriptor.signal_range != sr {
            self.exact = None;
        }
        self.descriptor.signal_range = sr;
        self
    }

    /// Return a copy with a different alpha mode.
    #[inline]
    #[must_use]
    pub fn with_alpha_mode(mut self, am: Option<AlphaMode>) -> Self {
        let descriptor = self.descriptor.with_alpha(am);
        descriptor.validate().expect("invalid alpha declaration");
        self.descriptor = descriptor;
        self
    }

    /// Whether this buffer carries meaningful alpha data.
    #[inline]
    pub fn has_alpha(&self) -> bool {
        self.descriptor.has_alpha()
    }

    /// Whether this buffer is grayscale (Gray or GrayAlpha layout).
    #[inline]
    pub fn is_grayscale(&self) -> bool {
        self.descriptor.is_grayscale()
    }

    /// Move out the allocation together with its layout and color metadata.
    ///
    /// Does not allocate, copy or compact pixels, or clone the color context.
    /// Preserves the Vec's pointer, length and capacity, including prefix,
    /// padding and trailing bytes. The first pixel is at `parts.data[parts.offset]`,
    /// not necessarily at the start of the allocation. The pixel type is erased.
    ///
    /// Call [`into_contiguous`](Self::into_contiguous) first if the receiver
    /// requires packed rows; otherwise pass the offset and stride onward.
    #[must_use]
    pub fn into_parts(self) -> PixelBufferParts<S> {
        PixelBufferParts {
            data: self.data,
            offset: self.offset as usize,
            stride_bytes: self.stride,
            width: self.width,
            height: self.height,
            descriptor: self.descriptor,
            color_context: self.color,
        }
    }
}

impl<P> PixelBuffer<P> {
    /// The tight pixel bytes as a `Vec<u8>`, for a receiver that needs one. Never copies.
    ///
    /// # Errors
    ///
    /// The buffer, unchanged, when its storage isn't bytes, the first pixel is past
    /// index 0, or the rows aren't tight. Use
    /// [`into_contiguous`](Self::into_contiguous) first when the rows may be padded,
    /// or [`copy_to_contiguous_bytes`](Self::copy_to_contiguous_bytes) for a copy.
    pub fn try_into_tight_u8_vec(self) -> Result<Vec<u8>, Self> {
        let row = self.width as usize * self.descriptor.bytes_per_pixel();
        let tight = self.stride == row || self.height <= 1;
        if self.offset != 0 || !tight {
            return Err(self);
        }
        let len = row * self.height as usize;
        let Self {
            data,
            offset,
            width,
            height,
            stride,
            descriptor,
            color,
            exact,
            _pixel,
        } = self;
        match data.try_into_vec::<u8>() {
            Ok(mut bytes) => {
                bytes.truncate(len);
                Ok(bytes)
            }
            Err(data) => Err(Self {
                data,
                offset,
                width,
                height,
                stride,
                descriptor,
                color,
                exact,
                _pixel,
            }),
        }
    }

    /// Pack the rows within this allocation, preserving the pixel type and metadata.
    ///
    /// The result has `stride() == width() * descriptor().bytes_per_pixel()`.
    /// Pixels retain their values and native byte order. The existing pixel
    /// offset is preserved to maintain sample alignment: contiguous rows do
    /// **not** imply that the first pixel is at allocation offset zero.
    ///
    /// Does not allocate, reallocate or clone the color context. Packing padded
    /// rows moves O(image bytes) within the existing allocation. Already packed
    /// rows need no pixel moves. The Vec's length is truncated to the offset plus
    /// the packed pixel length; its pointer and capacity are preserved. Zero-area
    /// images keep their dimensions and contain no pixel bytes.
    ///
    /// Use [`as_contiguous_bytes`](Self::as_contiguous_bytes) to borrow the packed
    /// pixels, or [`into_parts`](Self::into_parts) to move out storage and metadata:
    ///
    /// ```
    /// use zenpixels::{ByteStorage, PixelBuffer, PixelBufferParts, PixelDescriptor};
    ///
    /// let buffer = PixelBuffer::new(3, 2, PixelDescriptor::RGB8_SRGB);
    /// let PixelBufferParts {
    ///     data, offset, stride_bytes, width, height, descriptor, color_context, ..
    /// } = buffer.into_contiguous().into_parts();
    /// assert_eq!(stride_bytes, width as usize * descriptor.bytes_per_pixel());
    /// let pixels = &data.as_bytes()[offset..];
    /// assert_eq!(pixels.len(), stride_bytes * height as usize);
    /// // Move `data` and its description into the receiver; retain `offset`.
    /// ```
    #[must_use]
    pub fn into_contiguous(mut self) -> Self {
        let row_bytes = (self.width as usize)
            .checked_mul(self.descriptor.bytes_per_pixel())
            .expect("pixel buffer row size overflows usize");
        let pixel_bytes = row_bytes
            .checked_mul(self.height as usize)
            .expect("pixel buffer size overflows usize");
        let packed_end = self
            .offset_bytes()
            .checked_add(pixel_bytes)
            .expect("pixel buffer extent overflows usize");

        if pixel_bytes != 0 {
            // Check the visible extent, not stride * height: a layout transform
            // can return a view without padding after its final row.
            assert!(self.stride >= row_bytes, "pixel buffer stride is too small");
            let source_end = (self.height as usize - 1)
                .checked_mul(self.stride)
                .and_then(|n| n.checked_add(self.offset as usize))
                .and_then(|n| n.checked_add(row_bytes))
                .expect("pixel buffer extent overflows usize");
            assert!(
                source_end <= self.data.as_bytes().len(),
                "pixel buffer storage is too short"
            );
            if self.stride != row_bytes {
                // Destinations never reach a subsequent source row. copy_within
                // also handles overlap within the current row, without scratch.
                for y in 1..self.height as usize {
                    let src = (self.offset as usize) + y * self.stride;
                    let dst = (self.offset as usize) + y * row_bytes;
                    self.data
                        .as_bytes_mut()
                        .copy_within(src..src + row_bytes, dst);
                }
            }
        }
        assert!(
            packed_end <= self.data.as_bytes().len(),
            "pixel buffer storage is too short"
        );
        self.data.truncate_bytes(packed_end);
        self.stride = row_bytes;
        self
    }
}

impl<P, S: ByteStorage> PixelBuffer<P, S> {
    /// Zero-copy access to the raw pixel bytes when rows are tightly packed.
    ///
    /// Returns `Some(&[u8])` if `stride == width * bpp` (no padding),
    /// `None` if rows have stride padding.
    #[inline]
    pub fn as_contiguous_bytes(&self) -> Option<&[u8]> {
        if self.width == 0 || self.height == 0 {
            return Some(&[]);
        }
        let bpp = self.descriptor.bytes_per_pixel();
        let row_bytes = self.width as usize * bpp;
        if self.stride == row_bytes {
            let total = row_bytes * self.height as usize;
            Some(&self.data.as_bytes()[(self.offset as usize)..(self.offset as usize) + total])
        } else {
            None
        }
    }

    /// Copy pixel data to a new contiguous byte `Vec` without stride padding.
    ///
    /// Returns exactly `width * height * bytes_per_pixel` bytes in row-major order.
    /// For buffers already tightly packed (stride == width * bpp), this is a single memcpy.
    /// For padded buffers, this strips the padding row by row.
    pub fn copy_to_contiguous_bytes(&self) -> Vec<u8> {
        let bpp = self.descriptor.bytes_per_pixel();
        let row_bytes = self.width as usize * bpp;
        let total = row_bytes * self.height as usize;

        // Fast path: already contiguous
        if self.stride == row_bytes {
            let start = self.offset as usize;
            return self.data.as_bytes()[start..start + total].to_vec();
        }

        // Slow path: strip padding
        let mut out = Vec::with_capacity(total);
        let slice = self.as_slice();
        for y in 0..self.height {
            out.extend_from_slice(slice.row(y));
        }
        out
    }

    /// Image width in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Byte stride between row starts.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Pixel format descriptor.
    #[inline]
    pub fn descriptor(&self) -> PixelDescriptor {
        self.descriptor
    }

    /// Evidence that the current samples are exact lower-depth codes, if stated.
    ///
    /// Every mutable accessor ([`as_slice_mut`](Self::as_slice_mut),
    /// [`rows_mut`](Self::rows_mut), in-place transforms, typed mutable views)
    /// clears it before returning, as do descriptor changes that alter storage,
    /// layout or signal range. Views, crops and compaction keep it.
    #[inline]
    pub fn exact_depth(&self) -> Option<ExactDepth> {
        self.exact
    }

    /// State or clear exact-depth evidence for the current samples.
    ///
    /// The statement is the caller's: only storage and range consistency are
    /// checked, without reading samples. On error the evidence is unchanged.
    pub fn set_exact_depth(&mut self, exact: Option<ExactDepth>) -> Result<(), ExactDepthError> {
        if let Some(exact) = exact {
            check_exact(self.descriptor, exact)?;
        }
        self.exact = exact;
        Ok(())
    }

    /// Source color context (ICC/CICP metadata), if set.
    #[inline]
    pub fn color_context(&self) -> Option<&Arc<ColorContext>> {
        self.color.as_ref()
    }

    /// Set the color context on this buffer.
    #[inline]
    #[must_use]
    pub fn with_color_context(mut self, ctx: Arc<ColorContext>) -> Self {
        self.color = Some(ctx);
        self
    }

    /// Relabel the source diffuse-white anchor — the cd/m² that relative-linear
    /// `1.0` represents. **Reinterpretation only:** pixels are untouched (use a
    /// converter to change pixel values). Clones any existing [`ColorContext`],
    /// overrides its `diffuse_white`, and re-attaches it.
    #[must_use]
    pub fn with_diffuse_white(self, white: DiffuseWhite) -> Self {
        let ctx = self
            .color_context()
            .map(|c| (**c).clone())
            .unwrap_or_default()
            .with_diffuse_white(white);
        self.with_color_context(Arc::new(ctx))
    }

    /// Relabel the source [`Cicp`] colour signalling. **Reinterpretation only**
    /// — pixels are untouched. Clones any existing [`ColorContext`], sets
    /// `cicp`, and re-attaches it.
    #[must_use]
    pub fn with_cicp(self, cicp: Cicp) -> Self {
        let ctx = self
            .color_context()
            .map(|c| (**c).clone())
            .unwrap_or_default()
            .with_cicp(cicp);
        self.with_color_context(Arc::new(ctx))
    }

    /// Relabel the source ICC profile. **Reinterpretation only** — pixels are
    /// untouched. Clones any existing [`ColorContext`], sets `icc`, and
    /// re-attaches it.
    #[must_use]
    pub fn with_icc(self, icc: impl Into<Arc<[u8]>>) -> Self {
        let ctx = self
            .color_context()
            .map(|c| (**c).clone())
            .unwrap_or_default()
            .with_icc(icc);
        self.with_color_context(Arc::new(ctx))
    }

    /// Borrow the full buffer as an immutable [`PixelSlice`].
    pub fn as_slice(&self) -> PixelSlice<'_, P> {
        let total = self.storage_span_len();
        PixelSlice {
            data: &self.data.as_bytes()[(self.offset as usize)..(self.offset as usize) + total],
            width: self.width,
            rows: self.height,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color.clone(),
            exact: self.exact,
            _pixel: PhantomData,
        }
    }
}

impl<P, S: ByteStorageMut> PixelBuffer<P, S> {
    /// Borrow the full buffer as a mutable [`PixelSliceMut`].
    ///
    /// Note: this is a *view*; transforms that change the buffer's layout
    /// cannot be expressed through it (a re-described view would leave
    /// this buffer's own descriptor stale). For layout-changing in-place
    /// transforms, use [`PixelBuffer::transform_in_place`], which adopts
    /// the new description atomically.
    pub fn as_slice_mut(&mut self) -> PixelSliceMut<'_, P> {
        // Mutable access can change any sample: end current-sample evidence.
        self.exact = None;
        let total = self.storage_span_len();
        let offset = self.offset as usize;
        PixelSliceMut {
            data: &mut self.data.as_bytes_mut()[offset..offset + total],
            width: self.width,
            rows: self.height,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color.clone(),
            _pixel: PhantomData,
        }
    }
}

impl<P, S: ByteStorage> PixelBuffer<P, S> {
    /// [`rows`](Self::rows), or `None` when the rows are out of bounds.
    #[must_use]
    pub fn try_rows(&self, y: u32, count: u32) -> Option<PixelSlice<'_, P>> {
        y.checked_add(count)
            .is_some_and(|end| end <= self.height)
            .then(|| self.rows(y, count))
    }

    /// [`crop_view`](Self::crop_view), or `None` when the region is out of bounds.
    #[must_use]
    pub fn try_crop_view(&self, x: u32, y: u32, w: u32, h: u32) -> Option<PixelSlice<'_, P>> {
        let fits = x.checked_add(w).is_some_and(|end| end <= self.width)
            && y.checked_add(h).is_some_and(|end| end <= self.height);
        fits.then(|| self.crop_view(x, y, w, h))
    }

    /// [`with_descriptor`](Self::with_descriptor) for a descriptor that may not fit.
    ///
    /// # Errors
    ///
    /// [`BufferError::IncompatibleDescriptor`] when it changes the physical layout
    /// or is invalid.
    pub fn try_with_descriptor(self, descriptor: PixelDescriptor) -> Result<Self, At<BufferError>> {
        if !self.descriptor.layout_compatible(descriptor) || descriptor.validate().is_err() {
            return Err(whereat::at!(BufferError::IncompatibleDescriptor));
        }
        Ok(self.with_descriptor(descriptor))
    }

    /// [`with_alpha_mode`](Self::with_alpha_mode) for a mode the layout may not allow.
    ///
    /// # Errors
    ///
    /// [`BufferError::IncompatibleDescriptor`] when the layout can't hold it.
    pub fn try_with_alpha_mode(self, am: Option<AlphaMode>) -> Result<Self, At<BufferError>> {
        if self.descriptor.with_alpha(am).validate().is_err() {
            return Err(whereat::at!(BufferError::IncompatibleDescriptor));
        }
        Ok(self.with_alpha_mode(am))
    }

    /// [`copy_to_contiguous_bytes`](Self::copy_to_contiguous_bytes), reporting a
    /// failed allocation.
    ///
    /// # Errors
    ///
    /// [`BufferError::AllocationFailed`], or [`BufferError::InvalidDimensions`] when
    /// the size overflows.
    pub fn try_copy_to_contiguous_bytes(&self) -> Result<Vec<u8>, At<BufferError>> {
        let row_bytes = self.width as usize * self.descriptor.bytes_per_pixel();
        let total = row_bytes
            .checked_mul(self.height as usize)
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let mut out = Vec::new();
        out.try_reserve_exact(total)
            .map_err(|_| whereat::at!(BufferError::AllocationFailed))?;
        let slice = self.as_slice();
        for y in 0..self.height {
            out.extend_from_slice(slice.row(y));
        }
        Ok(out)
    }

    /// Borrow a range of rows as an immutable [`PixelSlice`].
    ///
    /// # Panics
    ///
    /// Panics if `y + count > height`.
    pub fn rows(&self, y: u32, count: u32) -> PixelSlice<'_, P> {
        assert!(
            y.checked_add(count).is_some_and(|end| end <= self.height),
            "rows({y}, {count}) out of bounds (height: {})",
            self.height
        );
        if count == 0 || self.width == 0 {
            return PixelSlice {
                data: &[],
                width: self.width,
                rows: count,
                stride: self.stride,
                descriptor: self.descriptor,

                color: self.color.clone(),
                exact: self.exact,
                _pixel: PhantomData,
            };
        }
        let bpp = self.descriptor.bytes_per_pixel();
        let start = (self.offset as usize) + y as usize * self.stride;
        let end = (self.offset as usize)
            + (y as usize + count as usize - 1) * self.stride
            + self.width as usize * bpp;
        PixelSlice {
            data: &self.data.as_bytes()[start..end],
            width: self.width,
            rows: count,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color.clone(),
            exact: self.exact,
            _pixel: PhantomData,
        }
    }
}

impl<P, S: ByteStorageMut> PixelBuffer<P, S> {
    /// [`rows_mut`](Self::rows_mut), or `None` when the rows are out of bounds.
    #[must_use]
    pub fn try_rows_mut(&mut self, y: u32, count: u32) -> Option<PixelSliceMut<'_, P>> {
        y.checked_add(count)
            .is_some_and(|end| end <= self.height)
            .then(|| self.rows_mut(y, count))
    }

    /// Borrow a range of rows as a mutable [`PixelSliceMut`].
    ///
    /// # Panics
    ///
    /// Panics if `y + count > height`.
    pub fn rows_mut(&mut self, y: u32, count: u32) -> PixelSliceMut<'_, P> {
        // Mutable access can change any sample: end current-sample evidence.
        self.exact = None;
        assert!(
            y.checked_add(count).is_some_and(|end| end <= self.height),
            "rows_mut({y}, {count}) out of bounds (height: {})",
            self.height
        );
        if count == 0 || self.width == 0 {
            return PixelSliceMut {
                data: &mut [],
                width: self.width,
                rows: count,
                stride: self.stride,
                descriptor: self.descriptor,

                color: self.color.clone(),
                _pixel: PhantomData,
            };
        }
        let bpp = self.descriptor.bytes_per_pixel();
        let start = (self.offset as usize) + y as usize * self.stride;
        let end = (self.offset as usize)
            + (y as usize + count as usize - 1) * self.stride
            + self.width as usize * bpp;
        PixelSliceMut {
            data: &mut self.data.as_bytes_mut()[start..end],
            width: self.width,
            rows: count,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color.clone(),
            _pixel: PhantomData,
        }
    }
}

impl<P, S: ByteStorage> PixelBuffer<P, S> {
    /// Zero-copy sub-region view (immutable).
    ///
    /// # Panics
    ///
    /// Panics if the crop region is out of bounds.
    pub fn crop_view(&self, x: u32, y: u32, w: u32, h: u32) -> PixelSlice<'_, P> {
        assert!(
            x.checked_add(w).is_some_and(|end| end <= self.width),
            "crop x={x} w={w} exceeds width {}",
            self.width
        );
        assert!(
            y.checked_add(h).is_some_and(|end| end <= self.height),
            "crop y={y} h={h} exceeds height {}",
            self.height
        );
        if h == 0 || w == 0 {
            return PixelSlice {
                data: &[],
                width: w,
                rows: h,
                stride: self.stride,
                descriptor: self.descriptor,

                color: self.color.clone(),
                exact: self.exact,
                _pixel: PhantomData,
            };
        }
        let bpp = self.descriptor.bytes_per_pixel();
        let start = (self.offset as usize) + y as usize * self.stride + x as usize * bpp;
        let end = (self.offset as usize)
            + (y as usize + h as usize - 1) * self.stride
            + (x as usize + w as usize) * bpp;
        PixelSlice {
            data: &self.data.as_bytes()[start..end],
            width: w,
            rows: h,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color.clone(),
            exact: self.exact,
            _pixel: PhantomData,
        }
    }

    /// Copy a sub-region into a new, tightly-packed [`PixelBuffer`].
    ///
    /// # Errors
    ///
    /// [`BufferError::InvalidDimensions`] when the region is out of bounds or its
    /// size overflows `usize`, and [`BufferError::AllocationFailed`].
    pub fn crop_copy(
        &self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<PixelBuffer<P>, At<BufferError>> {
        let src = self
            .try_crop_view(x, y, w, h)
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let stride = self.descriptor.aligned_stride(w);
        let align = self.descriptor.min_alignment();
        let total = stride
            .checked_mul(h as usize)
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let (data, offset) =
            alloc_storage(&self.descriptor, total, align).map_err(|e| whereat::at!(e))?;
        let mut dst = PixelBuffer {
            data,
            offset,
            width: w,
            height: h,
            stride,
            descriptor: self.descriptor,

            color: self.color.clone(),
            exact: self.exact,
            _pixel: PhantomData,
        };
        let bpp = self.descriptor.bytes_per_pixel();
        let row_bytes = w as usize * bpp;
        for row_y in 0..h {
            let src_row = src.row(row_y);
            let dst_start = dst.offset as usize + row_y as usize * dst.stride;
            dst.data.as_bytes_mut()[dst_start..dst_start + row_bytes]
                .copy_from_slice(&src_row[..row_bytes]);
        }
        Ok(dst)
    }
}

#[cfg(feature = "image")]
impl PixelBuffer {
    /// Whether the storage holds samples of type `T`.
    pub(crate) fn storage_holds<T: StorageSample>(&self) -> bool {
        self.data.holds::<T>()
    }
}

impl<P, S> PixelBuffer<P, S> {
    /// Byte index of the first pixel in the storage.
    pub(crate) fn offset_bytes(&self) -> usize {
        self.offset as usize
    }
}

impl<P, S> fmt::Debug for PixelBuffer<P, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PixelBuffer({}x{}, {:?} {:?})",
            self.width,
            self.height,
            self.descriptor.layout(),
            self.descriptor.channel_type()
        )
    }
}

/// Whether exact-depth evidence still describes samples relabeled from `old` to `new`.
fn exact_survives(old: PixelDescriptor, new: PixelDescriptor) -> bool {
    old.channel_type() == new.channel_type()
        && old.layout() == new.layout()
        && old.signal_range == new.signal_range
}

fn check_exact(descriptor: PixelDescriptor, exact: ExactDepth) -> Result<(), ExactDepthError> {
    if exact.storage() != descriptor.channel_type() || exact.range() != descriptor.signal_range {
        return Err(ExactDepthError::DescriptorMismatch);
    }
    Ok(())
}

fn pixel_slice_to_owned(slice: &PixelSlice<'_>) -> PixelBuffer {
    let mut buffer = PixelBuffer::new(slice.width(), slice.rows(), slice.descriptor());
    {
        let mut destination = buffer.as_slice_mut();
        for y in 0..slice.rows() {
            destination.row_mut(y).copy_from_slice(slice.row(y));
        }
    }
    if let Some(color) = slice.color_context() {
        buffer = buffer.with_color_context(color.clone());
    }
    // An exact copy of the same values keeps their evidence.
    buffer.exact = slice.exact_depth();
    buffer
}

// ---------------------------------------------------------------------------
// ImgRef -> PixelSlice (zero-copy From impls) -- imgref feature only
// ---------------------------------------------------------------------------

#[cfg(feature = "imgref")]
macro_rules! impl_from_imgref {
    ($pixel:ty, $descriptor:expr) => {
        impl<'a> From<ImgRef<'a, $pixel>> for PixelSlice<'a, $pixel> {
            fn from(img: ImgRef<'a, $pixel>) -> Self {
                let bytes: &[u8] = bytemuck::cast_slice(img.buf());
                let byte_stride = img.stride() * core::mem::size_of::<$pixel>();
                PixelSlice {
                    data: bytes,
                    width: img.width() as u32,
                    rows: img.height() as u32,
                    stride: byte_stride,
                    descriptor: $descriptor,

                    color: None,
                    exact: None,
                    _pixel: PhantomData,
                }
            }
        }
    };
}

// u8 types are conventionally sRGB, f32 types are conventionally linear.
// u16 types have no standard convention so use transfer-agnostic descriptors.
#[cfg(feature = "imgref")]
impl_from_imgref!(Rgb<u8>, PixelDescriptor::RGB8_SRGB);
#[cfg(feature = "imgref")]
impl_from_imgref!(Rgba<u8>, PixelDescriptor::RGBA8_SRGB);
#[cfg(feature = "imgref")]
impl_from_imgref!(Rgb<u16>, PixelDescriptor::RGB16);
#[cfg(feature = "imgref")]
impl_from_imgref!(Rgba<u16>, PixelDescriptor::RGBA16);
#[cfg(feature = "imgref")]
impl_from_imgref!(Rgb<f32>, PixelDescriptor::RGBF32_LINEAR);
#[cfg(feature = "imgref")]
impl_from_imgref!(Rgba<f32>, PixelDescriptor::RGBAF32_LINEAR);
#[cfg(feature = "imgref")]
impl_from_imgref!(Gray<u8>, PixelDescriptor::GRAY8_SRGB);
#[cfg(feature = "imgref")]
impl_from_imgref!(Gray<u16>, PixelDescriptor::GRAY16);
#[cfg(feature = "imgref")]
impl_from_imgref!(Gray<f32>, PixelDescriptor::GRAYF32_LINEAR);
#[cfg(feature = "imgref")]
impl_from_imgref!(BGRA<u8>, PixelDescriptor::BGRA8_SRGB);

// ---------------------------------------------------------------------------
// ImgRefMut -> PixelSliceMut (zero-copy From impls) -- imgref feature only
// ---------------------------------------------------------------------------

#[cfg(feature = "imgref")]
macro_rules! impl_from_imgref_mut {
    ($pixel:ty, $descriptor:expr) => {
        impl<'a> From<imgref::ImgRefMut<'a, $pixel>> for PixelSliceMut<'a, $pixel> {
            fn from(img: imgref::ImgRefMut<'a, $pixel>) -> Self {
                let width = img.width() as u32;
                let rows = img.height() as u32;
                let byte_stride = img.stride() * core::mem::size_of::<$pixel>();
                let buf = img.into_buf();
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(buf);
                PixelSliceMut {
                    data: bytes,
                    width,
                    rows,
                    stride: byte_stride,
                    descriptor: $descriptor,

                    color: None,
                    _pixel: PhantomData,
                }
            }
        }
    };
}

#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Rgb<u8>, PixelDescriptor::RGB8_SRGB);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Rgba<u8>, PixelDescriptor::RGBA8_SRGB);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Rgb<u16>, PixelDescriptor::RGB16);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Rgba<u16>, PixelDescriptor::RGBA16);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Rgb<f32>, PixelDescriptor::RGBF32_LINEAR);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Rgba<f32>, PixelDescriptor::RGBAF32_LINEAR);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Gray<u8>, PixelDescriptor::GRAY8_SRGB);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Gray<u16>, PixelDescriptor::GRAY16);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Gray<f32>, PixelDescriptor::GRAYF32_LINEAR);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(BGRA<u8>, PixelDescriptor::BGRA8_SRGB);

// ---------------------------------------------------------------------------
// Typed -> Erased blanket From impls (erase via From)
// ---------------------------------------------------------------------------

impl<'a, P: Pixel> From<PixelSlice<'a, P>> for PixelSlice<'a> {
    fn from(typed: PixelSlice<'a, P>) -> Self {
        typed.erase()
    }
}

impl<'a, P: Pixel> From<PixelSliceMut<'a, P>> for PixelSliceMut<'a> {
    fn from(typed: PixelSliceMut<'a, P>) -> Self {
        typed.erase()
    }
}

impl<P: Pixel> From<PixelBuffer<P>> for PixelBuffer {
    fn from(typed: PixelBuffer<P>) -> Self {
        typed.erase()
    }
}

// ---------------------------------------------------------------------------
// Mutable -> Immutable conversions (zero-copy reborrow / move)
// ---------------------------------------------------------------------------

/// Consume a mutable pixel slice and produce an immutable one (zero-copy).
impl<'a, P> From<PixelSliceMut<'a, P>> for PixelSlice<'a, P> {
    fn from(mut_slice: PixelSliceMut<'a, P>) -> Self {
        PixelSlice {
            data: mut_slice.data,
            width: mut_slice.width,
            rows: mut_slice.rows,
            stride: mut_slice.stride,
            descriptor: mut_slice.descriptor,
            color: mut_slice.color,
            exact: None,
            _pixel: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    /// A buffer adopted with whatever prefix makes its rows `align`-aligned, on a
    /// stride that is the LCM of `align` and the pixel size. No public constructor
    /// allocates this today; `offset` and `try_from_parts` keep it available.
    fn simd_padded(
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
        align: usize,
    ) -> PixelBuffer {
        let stride = descriptor.simd_aligned_stride(width, align);
        let total = stride * height as usize;
        let data = alloc::vec![0u8; total + align - 1];
        let mut parts: PixelBufferParts =
            PixelBufferParts::new(data.into(), width, height, stride, descriptor);
        parts.offset = align_offset(parts.data.as_bytes().as_ptr(), align);
        PixelBuffer::try_from_parts(parts).unwrap()
    }

    use super::*;
    use crate::descriptor::{ChannelLayout, ChannelType};
    use alloc::format;

    /// The buffer-level relabel builders set their field and **preserve** the
    /// rest of the `ColorContext` (clone-override-reattach), and chain.
    #[test]
    fn allocator_zeroed_storage_boundaries() {
        for size in [0, 1, 4095, 4096, 4097, 65536] {
            let storage = PixelStorage::try_zeroed(ChannelType::U8, size).unwrap();
            assert_eq!(ByteStorage::as_bytes(&storage).len(), size);
            assert!(
                ByteStorage::as_bytes(&storage)
                    .iter()
                    .all(|&byte| byte == 0)
            );
        }
        assert!(matches!(
            PixelStorage::try_zeroed(ChannelType::U8, usize::MAX),
            Err(BufferError::AllocationFailed)
        ));
    }

    #[test]
    fn relabel_builders_set_and_preserve_color_context() {
        let data = alloc::vec![0u8; 4 * 4 * 3];
        let slice = PixelSlice::new(&data, 4, 4, 12, PixelDescriptor::RGB8_SRGB)
            .unwrap()
            .with_cicp(Cicp::BT2100_PQ)
            .with_diffuse_white(DiffuseWhite::new(120.0));
        let ctx = slice.color_context().expect("context attached");
        assert_eq!(ctx.diffuse_white, Some(DiffuseWhite::new(120.0)));
        assert_eq!(
            ctx.cicp,
            Some(Cicp::BT2100_PQ),
            "cicp preserved across the second relabel"
        );

        // A third relabel (icc) preserves the two already set.
        let slice = slice.with_icc(alloc::vec![1u8, 2, 3]);
        let ctx = slice.color_context().unwrap();
        assert!(ctx.icc.is_some());
        assert_eq!(ctx.cicp, Some(Cicp::BT2100_PQ));
        assert_eq!(ctx.diffuse_white, Some(DiffuseWhite::new(120.0)));
    }

    // --- PixelBuffer allocation and row access ---

    #[test]
    fn into_contiguous_preserves_pixels_across_offsets_padding_and_empty_extents() {
        for descriptor in [
            PixelDescriptor::GRAY8_SRGB,
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::RGBA16_SRGB,
        ] {
            let bpp = descriptor.bytes_per_pixel();
            for width in [0, 1, 2, 5] {
                for height in [0, 1, 2, 5] {
                    for padding_pixels in [0, 1, 3] {
                        for prefix_samples in [0, 3] {
                            for final_padding in [false, true] {
                                let row_bytes = width as usize * bpp;
                                let stride = row_bytes + padding_pixels * bpp;
                                let visible_len = if width == 0 || height == 0 {
                                    0
                                } else {
                                    (height as usize - 1) * stride + row_bytes
                                };
                                // Reserve enough first: computing the offset must
                                // precede initialization but follow allocation.
                                let mut data = Vec::with_capacity(visible_len + stride + 128);
                                let align = descriptor.min_alignment();
                                let offset =
                                    align_offset(data.as_ptr(), align) + prefix_samples * align;
                                let len = offset
                                    + visible_len
                                    + if final_padding { stride + 7 } else { 0 };
                                data.resize(len, 0xEE);
                                let mut expected = Vec::new();
                                for y in 0..height as usize {
                                    for x in 0..row_bytes {
                                        let value = (y * 13 + x) as u8;
                                        data[offset + y * stride + x] = value;
                                        expected.push(value);
                                    }
                                }
                                let pointer = data.as_ptr();
                                let capacity = data.capacity();
                                let buffer = PixelBuffer::<()> {
                                    data: data.into(),
                                    offset: offset as u32,
                                    width,
                                    height,
                                    stride,
                                    descriptor,
                                    color: None,
                                    exact: None,
                                    _pixel: PhantomData,
                                };
                                let packed = buffer.into_contiguous();
                                assert_eq!(packed.as_contiguous_bytes(), Some(expected.as_slice()));
                                assert_eq!(packed.stride(), row_bytes);
                                assert_eq!((packed.width(), packed.height()), (width, height));
                                assert_eq!(packed.descriptor(), descriptor);
                                let parts = packed.into_parts();
                                assert_eq!(parts.data.as_bytes().as_ptr(), pointer);
                                assert_eq!(parts.data.capacity_bytes(), capacity);
                                assert_eq!(parts.offset, offset);
                                assert_eq!(parts.data.as_bytes().len(), offset + expected.len());
                                assert!(parts.data.as_bytes()[..offset].iter().all(|&v| v == 0xEE));
                                assert_eq!(parts.data.as_bytes()[offset..], expected);
                                assert!(parts.color_context.is_none());
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn into_parts_moves_the_complete_uncompacted_allocation_and_context() {
        let context = Arc::new(
            ColorContext::from_icc([1u8, 2, 3, 4]).with_diffuse_white(DiffuseWhite::new(120.0)),
        );
        let mut buffer =
            simd_padded(3, 2, PixelDescriptor::RGB8_SRGB, 64).with_color_context(context.clone());
        buffer.data.as_bytes_mut().fill(0xAB);
        let (pointer, len, capacity, offset, stride) = (
            buffer.data.as_bytes().as_ptr(),
            buffer.data.as_bytes().len(),
            buffer.data.capacity_bytes(),
            buffer.offset as usize,
            buffer.stride,
        );
        let parts = buffer.into_parts();
        assert_eq!(parts.data.as_bytes().as_ptr(), pointer);
        assert_eq!(
            (parts.data.as_bytes().len(), parts.data.capacity_bytes()),
            (len, capacity)
        );
        assert_eq!((parts.offset, parts.stride_bytes), (offset, stride));
        assert_eq!((parts.width, parts.height), (3, 2));
        assert_eq!(parts.descriptor, PixelDescriptor::RGB8_SRGB);
        assert!(parts.data.as_bytes().iter().all(|&v| v == 0xAB));
        assert!(Arc::ptr_eq(parts.color_context.as_ref().unwrap(), &context));
        assert_eq!(Arc::strong_count(&context), 2);
    }

    #[test]
    fn into_contiguous_preserves_type_and_moves_color_context() {
        let context = Arc::new(ColorContext::from_cicp(Cicp::SRGB));
        let mut buffer = simd_padded(2, 3, Rgbx::DESCRIPTOR, 16)
            .try_typed::<Rgbx>()
            .unwrap()
            .with_color_context(context.clone());
        for y in 0..3 {
            buffer.as_slice_mut().row_mut(y).fill(y as u8 + 1);
        }
        let pointer = buffer.data.as_bytes().as_ptr();
        let packed: PixelBuffer<Rgbx> = buffer.into_contiguous();
        assert_eq!(packed.data.as_bytes().as_ptr(), pointer);
        assert_eq!(packed.descriptor(), Rgbx::DESCRIPTOR);
        for y in 0..3 {
            assert_eq!(packed.as_slice().row(y), &[y as u8 + 1; 8]);
        }
        assert!(Arc::ptr_eq(packed.color_context().unwrap(), &context));
        assert_eq!(Arc::strong_count(&context), 2);
        let parts = packed.into_parts();
        assert!(Arc::ptr_eq(parts.color_context.as_ref().unwrap(), &context));
        assert_eq!(Arc::strong_count(&context), 2);
    }

    #[test]
    fn into_contiguous_is_idempotent_and_trims_trailing_storage() {
        let mut bytes = vec![1u8, 2, 3, 0xEE, 0xEE];
        bytes.reserve(128);
        let buffer = PixelBuffer::from_vec(bytes, 1, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        let (pointer, capacity) = (
            buffer.data.as_bytes().as_ptr(),
            buffer.data.capacity_bytes(),
        );
        let packed = buffer.into_contiguous().into_contiguous();
        assert_eq!(packed.as_contiguous_bytes(), Some([1, 2, 3].as_slice()));
        let parts = packed.into_parts();
        assert_eq!(parts.data.as_bytes(), [1, 2, 3]);
        assert_eq!(parts.data.as_bytes().as_ptr(), pointer);
        assert_eq!(parts.data.capacity_bytes(), capacity);
    }

    #[test]
    fn pixel_buffer_new_rgb8() {
        let buf = PixelBuffer::new(10, 5, PixelDescriptor::RGB8_SRGB);
        assert_eq!(buf.width(), 10);
        assert_eq!(buf.height(), 5);
        assert_eq!(buf.stride(), 30);
        assert_eq!(buf.descriptor(), PixelDescriptor::RGB8_SRGB);
        // All zeros
        let slice = buf.as_slice();
        assert_eq!(slice.row(0), &[0u8; 30]);
        assert_eq!(slice.row(4), &[0u8; 30]);
    }

    #[test]
    fn in_place_pixels_try_new_rejects_invalid_geometry() {
        let mut bytes = [0u8; 5];
        let error = InPlacePixels::try_new(&mut bytes, 2, 1, 5, PixelDescriptor::RGB8_SRGB, None)
            .err()
            .unwrap();
        assert_eq!(*error.error(), BufferError::StrideTooSmall);
    }

    #[test]
    fn pixel_cow_clones_only_when_mutated() {
        let bytes = [1u8, 2, 3];
        let slice = PixelSlice::new(&bytes, 1, 1, 3, PixelDescriptor::RGB8_SRGB).unwrap();
        let mut pixels = PixelCow::Borrowed(slice);

        assert!(pixels.is_borrowed());
        pixels.to_mut().as_slice_mut().row_mut(0)[0] = 9;

        assert!(pixels.is_owned());
        assert_eq!(pixels.as_slice().row(0), &[9, 2, 3]);
        assert_eq!(bytes, [1, 2, 3]);
    }

    #[test]
    fn pixel_buffer_from_vec() {
        let data = vec![0u8; 30 * 5];
        let buf = PixelBuffer::from_vec(data, 10, 5, PixelDescriptor::RGB8_SRGB).unwrap();
        assert_eq!(buf.width(), 10);
        assert_eq!(buf.height(), 5);
    }

    #[test]
    fn pixel_buffer_from_vec_too_small() {
        let data = vec![0u8; 10];
        let err = PixelBuffer::from_vec(data, 10, 5, PixelDescriptor::RGB8_SRGB);
        assert_eq!(*err.unwrap_err().error(), BufferError::InsufficientData);
    }

    #[test]
    fn pixel_buffer_into_vec_roundtrip() {
        let buf = PixelBuffer::new(4, 4, PixelDescriptor::RGBA8_SRGB);
        let v = buf.into_parts().data.try_into_vec::<u8>().unwrap();
        // Can re-wrap it
        let buf2 = PixelBuffer::from_vec(v, 4, 4, PixelDescriptor::RGBA8_SRGB).unwrap();
        assert_eq!(buf2.width(), 4);
    }

    #[test]
    fn pixel_buffer_write_and_read() {
        let mut buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);
        {
            let mut slice = buf.as_slice_mut();
            let row = slice.row_mut(0);
            row[0] = 255;
            row[1] = 128;
            row[2] = 64;
        }
        let slice = buf.as_slice();
        assert_eq!(&slice.row(0)[..3], &[255, 128, 64]);
        assert_eq!(&slice.row(1)[..3], &[0, 0, 0]);
    }

    #[test]
    fn pixel_buffer_simd_aligned() {
        let buf = simd_padded(10, 5, PixelDescriptor::RGBA8_SRGB, 64);
        assert_eq!(buf.width(), 10);
        assert_eq!(buf.height(), 5);
        // RGBA8 bpp=4, lcm(4,64)=64, raw=40 -> stride=64
        assert_eq!(buf.stride(), 64);
        // First row should be 64-byte aligned
        let slice = buf.as_slice();
        assert_eq!(slice.data.as_bytes().as_ptr() as usize % 64, 0);
    }

    // --- PixelSlice crop_view ---

    #[test]
    fn pixel_slice_crop_view() {
        // 4x4 RGB8 buffer, fill each row with row index
        let mut buf = PixelBuffer::new(4, 4, PixelDescriptor::RGB8_SRGB);
        {
            let mut slice = buf.as_slice_mut();
            for y in 0..4u32 {
                let row = slice.row_mut(y);
                for byte in row.iter_mut() {
                    *byte = y as u8;
                }
            }
        }
        // Crop 2x2 starting at (1, 1)
        let crop = buf.crop_view(1, 1, 2, 2);
        assert_eq!(crop.width(), 2);
        assert_eq!(crop.rows(), 2);
        // Row 0 of crop = row 1 of original, should be all 1s
        assert_eq!(crop.row(0), &[1, 1, 1, 1, 1, 1]);
        // Row 1 of crop = row 2 of original, should be all 2s
        assert_eq!(crop.row(1), &[2, 2, 2, 2, 2, 2]);
    }

    #[test]
    fn pixel_slice_crop_copy() {
        let mut buf = PixelBuffer::new(4, 4, PixelDescriptor::RGB8_SRGB);
        {
            let mut slice = buf.as_slice_mut();
            for y in 0..4u32 {
                let row = slice.row_mut(y);
                for (i, byte) in row.iter_mut().enumerate() {
                    *byte = (y * 100 + i as u32) as u8;
                }
            }
        }
        let cropped = buf.crop_copy(1, 1, 2, 2).unwrap();
        assert_eq!(cropped.width(), 2);
        assert_eq!(cropped.height(), 2);
        // Row 0: original row 1, pixels 1-2 -> bytes [103,104,105, 106,107,108]
        assert_eq!(cropped.as_slice().row(0), &[103, 104, 105, 106, 107, 108]);
    }

    #[test]
    fn pixel_slice_sub_rows() {
        let mut buf = PixelBuffer::new(2, 4, PixelDescriptor::GRAY8_SRGB);
        {
            let mut slice = buf.as_slice_mut();
            for y in 0..4u32 {
                let row = slice.row_mut(y);
                row[0] = y as u8 * 10;
                row[1] = y as u8 * 10 + 1;
            }
        }
        let sub = buf.rows(1, 2);
        assert_eq!(sub.rows(), 2);
        assert_eq!(sub.row(0), &[10, 11]);
        assert_eq!(sub.row(1), &[20, 21]);
    }

    // --- PixelSlice validation ---

    #[test]
    fn pixel_slice_stride_too_small() {
        let data = [0u8; 100];
        let err = PixelSlice::new(&data, 10, 1, 2, PixelDescriptor::RGB8_SRGB);
        assert_eq!(*err.unwrap_err().error(), BufferError::StrideTooSmall);
    }

    #[test]
    fn pixel_slice_insufficient_data() {
        let data = [0u8; 10];
        let err = PixelSlice::new(&data, 10, 1, 30, PixelDescriptor::RGB8_SRGB);
        assert_eq!(*err.unwrap_err().error(), BufferError::InsufficientData);
    }

    #[test]
    fn pixel_slice_zero_rows() {
        let data = [0u8; 0];
        let slice = PixelSlice::new(&data, 10, 0, 30, PixelDescriptor::RGB8_SRGB).unwrap();
        assert_eq!(slice.rows(), 0);
    }

    #[test]
    fn stride_need_not_be_whole_pixels() {
        // RGB8 bpp=3, stride=32 is not a multiple of 3, but every sample still fits.
        let data = [0u8; 128];
        let slice = PixelSlice::new(&data, 10, 2, 32, PixelDescriptor::RGB8_SRGB).unwrap();
        assert_eq!(slice.stride(), 32);
        assert_eq!(slice.row(1).len(), 30);
        assert_eq!(slice.row(1).as_ptr(), data[32..].as_ptr());
    }

    #[test]
    fn stride_must_be_whole_samples() {
        let words = [0u32; 32];
        let data: &[u8] = bytemuck::cast_slice(&words);
        for (descriptor, width, stride) in [
            (PixelDescriptor::RGB16_SRGB, 2, 13),
            (PixelDescriptor::RGBF32_LINEAR, 1, 14),
            (PixelDescriptor::RGBF32_LINEAR, 1, 15),
        ] {
            for rows in [1, 2] {
                assert_eq!(
                    *PixelSlice::new(data, width, rows, stride, descriptor)
                        .unwrap_err()
                        .error(),
                    BufferError::AlignmentViolation,
                    "{descriptor:?} stride {stride} rows {rows}"
                );
            }
        }
        assert!(PixelSlice::new(data, 2, 2, 14, PixelDescriptor::RGB16_SRGB).is_ok());
        assert!(PixelSlice::new(data, 1, 2, 16, PixelDescriptor::RGBF32_LINEAR).is_ok());
    }

    #[test]
    fn stride_pixel_aligned_accepted() {
        // RGBA8 bpp=4, stride=48 is a multiple of 4
        let data = [0u8; 256];
        let ok = PixelSlice::new(&data, 10, 2, 48, PixelDescriptor::RGBA8_SRGB);
        assert!(ok.is_ok());
        let s = ok.unwrap();
        assert_eq!(s.stride(), 48);
    }

    // --- Debug formatting ---

    #[test]
    fn debug_formats() {
        let buf = PixelBuffer::new(10, 5, PixelDescriptor::RGB8_SRGB);
        assert_eq!(format!("{buf:?}"), "PixelBuffer(10x5, Rgb U8)");

        let slice = buf.as_slice();
        assert_eq!(format!("{slice:?}"), "PixelSlice(10x5, Rgb U8)");

        let mut buf = PixelBuffer::new(3, 3, PixelDescriptor::RGBA16_SRGB);
        let slice_mut = buf.as_slice_mut();
        assert_eq!(format!("{slice_mut:?}"), "PixelSliceMut(3x3, Rgba U16)");
    }

    // --- BufferError Display ---

    #[test]
    fn buffer_error_display() {
        let msg = format!("{}", BufferError::StrideTooSmall);
        assert!(msg.contains("stride"));
    }

    // --- Edge cases ---

    #[test]
    fn bgrx8_srgb_properties() {
        let d = PixelDescriptor::BGRX8_SRGB;
        assert_eq!(d.channel_type(), ChannelType::U8);
        assert_eq!(d.layout(), ChannelLayout::Bgra);
        assert_eq!(d.alpha(), Some(AlphaMode::Undefined));
        assert_eq!(d.transfer(), TransferFunction::Srgb);
        assert_eq!(d.bytes_per_pixel(), 4);
        assert_eq!(d.min_alignment(), 1);
        // Layout-compatible with BGRA8
        assert!(d.layout_compatible(PixelDescriptor::BGRA8_SRGB));
        // BGRX has no meaningful alpha -- the fourth byte is padding
        assert!(!d.has_alpha());
        // BGRA does have meaningful alpha
        assert!(PixelDescriptor::BGRA8_SRGB.has_alpha());
        // The layout itself reports an alpha-position channel
        assert!(d.layout().has_alpha());
    }

    #[test]
    fn zero_size_buffer() {
        let buf = PixelBuffer::new(0, 0, PixelDescriptor::RGB8_SRGB);
        assert_eq!(buf.width(), 0);
        assert_eq!(buf.height(), 0);
        let slice = buf.as_slice();
        assert_eq!(slice.rows(), 0);
    }

    #[test]
    fn crop_empty() {
        let buf = PixelBuffer::new(4, 4, PixelDescriptor::RGB8_SRGB);
        let crop = buf.crop_view(0, 0, 0, 0);
        assert_eq!(crop.width(), 0);
        assert_eq!(crop.rows(), 0);
    }

    #[test]
    fn sub_rows_empty() {
        let buf = PixelBuffer::new(4, 4, PixelDescriptor::RGB8_SRGB);
        let sub = buf.rows(2, 0);
        assert_eq!(sub.rows(), 0);
    }

    // --- with_descriptor assertion ---

    #[test]
    fn with_descriptor_metadata_change_succeeds() {
        let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);
        // Changing transfer function is metadata-only -- should succeed
        let buf2 = buf.with_descriptor(PixelDescriptor::RGB8);
        assert_eq!(buf2.descriptor(), PixelDescriptor::RGB8);
    }

    #[test]
    #[should_panic(expected = "with_descriptor() cannot change physical layout")]
    fn with_descriptor_layout_change_panics() {
        let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8);
        // Trying to change from RGB8 to RGBA8 -- different layout, should panic
        let _ = buf.with_descriptor(PixelDescriptor::RGBA8);
    }

    #[test]
    fn with_descriptor_slice_assertion() {
        let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);
        let slice = buf.as_slice();
        // Metadata change OK
        let s2 = slice.with_descriptor(PixelDescriptor::RGB8);
        assert_eq!(s2.descriptor(), PixelDescriptor::RGB8);
    }

    #[test]
    #[should_panic(expected = "with_descriptor() cannot change physical layout")]
    fn with_descriptor_slice_layout_change_panics() {
        let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8);
        let slice = buf.as_slice();
        let _ = slice.with_descriptor(PixelDescriptor::RGBA8);
    }

    // --- reinterpret ---

    #[test]
    fn reinterpret_same_bpp_succeeds() {
        // RGBA8 -> BGRA8: same 4 bpp, different layout
        let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBA8);
        let buf2 = buf.reinterpret(PixelDescriptor::BGRA8).unwrap();
        assert_eq!(buf2.descriptor().layout(), ChannelLayout::Bgra);
    }

    #[test]
    fn reinterpret_different_bpp_fails() {
        // RGB8 (3 bpp) -> RGBA8 (4 bpp): different bytes_per_pixel
        let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8);
        let err = buf.reinterpret(PixelDescriptor::RGBA8);
        assert_eq!(
            *err.unwrap_err().error(),
            BufferError::IncompatibleDescriptor
        );
    }

    #[test]
    fn reinterpret_rgbx_to_rgba() {
        // RGBX8 -> RGBA8: same bpp (4), reinterpret padding as alpha
        let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBX8);
        let buf2 = buf.reinterpret(PixelDescriptor::RGBA8).unwrap();
        assert!(buf2.descriptor().has_alpha());
    }

    // --- Per-field metadata setters ---

    #[test]
    fn per_field_setters() {
        let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBA8);
        let buf = buf.with_transfer(TransferFunction::Srgb);
        assert_eq!(buf.descriptor().transfer(), TransferFunction::Srgb);
        let buf = buf.with_primaries(ColorPrimaries::DisplayP3);
        assert_eq!(buf.descriptor().primaries, ColorPrimaries::DisplayP3);
        let buf = buf.with_signal_range(SignalRange::Narrow);
        assert!(matches!(buf.descriptor().signal_range, SignalRange::Narrow));
        let buf = buf.with_alpha_mode(Some(AlphaMode::Premultiplied));
        assert_eq!(buf.descriptor().alpha(), Some(AlphaMode::Premultiplied));
    }

    // --- copy_to_contiguous_bytes ---

    #[test]
    fn copy_to_contiguous_bytes_tight() {
        let mut buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);
        {
            let mut s = buf.as_slice_mut();
            s.row_mut(0).copy_from_slice(&[1, 2, 3, 4, 5, 6]);
            s.row_mut(1).copy_from_slice(&[7, 8, 9, 10, 11, 12]);
        }
        let bytes = buf.copy_to_contiguous_bytes();
        assert_eq!(bytes, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn copy_to_contiguous_bytes_padded() {
        // Use SIMD-aligned buffer which will have stride padding for small widths
        let mut buf = simd_padded(2, 2, PixelDescriptor::RGB8_SRGB, 16);
        let stride = buf.stride();
        // Stride should be >= 6 (2 pixels * 3 bytes) and aligned to lcm(3, 16) = 48
        assert!(stride >= 6);
        {
            let mut s = buf.as_slice_mut();
            s.row_mut(0).copy_from_slice(&[1, 2, 3, 4, 5, 6]);
            s.row_mut(1).copy_from_slice(&[7, 8, 9, 10, 11, 12]);
        }
        let bytes = buf.copy_to_contiguous_bytes();
        // Should only contain the actual pixel data, no padding
        assert_eq!(bytes.len(), 12);
        assert_eq!(bytes, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    // --- BufferError Display for all variants ---

    #[test]
    fn buffer_error_display_alignment_violation() {
        let msg = format!("{}", BufferError::AlignmentViolation);
        assert_eq!(msg, "data is not aligned for the channel type");
    }

    #[test]
    fn buffer_error_display_insufficient_data() {
        let msg = format!("{}", BufferError::InsufficientData);
        assert_eq!(msg, "data slice is too small for the given dimensions");
    }

    #[test]
    fn buffer_error_display_invalid_dimensions() {
        let msg = format!("{}", BufferError::InvalidDimensions);
        assert_eq!(msg, "width or height is zero or causes overflow");
    }

    #[test]
    fn buffer_error_display_incompatible_descriptor() {
        let msg = format!("{}", BufferError::IncompatibleDescriptor);
        assert_eq!(
            msg,
            "descriptor conflicts with pixel layout, type, or alpha semantics"
        );
    }

    #[test]
    fn buffer_error_display_allocation_failed() {
        let msg = format!("{}", BufferError::AllocationFailed);
        assert_eq!(msg, "buffer allocation failed");
    }

    // --- PixelSlice::is_contiguous ---

    #[test]
    fn pixel_slice_is_contiguous_tight() {
        // Tight buffer: stride == width * bpp
        let buf = PixelBuffer::new(4, 3, PixelDescriptor::RGBA8_SRGB);
        let slice = buf.as_slice();
        // stride should be 4 * 4 = 16
        assert_eq!(slice.stride(), 16);
        assert!(slice.is_contiguous());
    }

    // --- PixelSlice::as_contiguous_bytes ---

    #[test]
    fn pixel_slice_as_contiguous_bytes_tight() {
        let mut buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBA8_SRGB);
        {
            let mut s = buf.as_slice_mut();
            s.row_mut(0).copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
            s.row_mut(1)
                .copy_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);
        }
        let slice = buf.as_slice();
        let bytes = slice.as_contiguous_bytes();
        assert!(bytes.is_some());
        assert_eq!(
            bytes.unwrap(),
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    #[test]
    fn pixel_slice_as_contiguous_bytes_padded_returns_none() {
        // SIMD-aligned buffer will have stride padding for small widths
        let buf = simd_padded(2, 2, PixelDescriptor::RGB8_SRGB, 16);
        let slice = buf.as_slice();
        // stride > width * bpp, so not contiguous
        assert!(slice.stride() > 6);
        assert!(!slice.is_contiguous());
        assert!(slice.as_contiguous_bytes().is_none());
    }

    // --- PixelSlice::as_strided_bytes ---

    #[test]
    fn pixel_slice_as_strided_bytes_tight() {
        let mut buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBA8_SRGB);
        {
            let mut s = buf.as_slice_mut();
            s.row_mut(0).copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
            s.row_mut(1)
                .copy_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);
        }
        let slice = buf.as_slice();
        let bytes = slice.as_strided_bytes();
        // Tight layout: strided bytes == contiguous bytes
        assert_eq!(
            bytes,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    #[test]
    fn pixel_slice_as_strided_bytes_padded() {
        let buf = simd_padded(2, 2, PixelDescriptor::RGB8_SRGB, 16);
        let slice = buf.as_slice();
        let stride = slice.stride();
        assert!(stride > 6, "expected padding for SIMD alignment");
        let bytes = slice.as_strided_bytes();
        // Total length includes stride padding between and after rows
        assert_eq!(bytes.len(), stride * 2);
    }

    #[test]
    fn pixel_slice_as_strided_bytes_sub_rows() {
        let buf = simd_padded(2, 4, PixelDescriptor::RGB8_SRGB, 16);
        let slice = buf.as_slice();
        let stride = slice.stride();
        let sub = slice.sub_rows(1, 2);
        let bytes = sub.as_strided_bytes();
        // sub_rows trims trailing padding on last row: (count-1)*stride + width*bpp
        let expected_len = stride + 2 * 3; // 1 stride + 6 pixel bytes
        assert_eq!(bytes.len(), expected_len);
    }

    #[test]
    fn pixel_slice_mut_as_strided_bytes() {
        let mut buf = simd_padded(2, 2, PixelDescriptor::RGB8_SRGB, 16);
        let mut slice = buf.as_slice_mut();
        let stride = slice.stride();
        // Write via as_strided_bytes_mut
        let bytes = slice.as_strided_bytes_mut();
        assert_eq!(bytes.len(), stride * 2);
        bytes[0] = 42;
        // Verify through row accessor
        assert_eq!(slice.row(0)[0], 42);
    }

    // --- PixelSlice::row_with_stride ---

    #[test]
    fn pixel_slice_row_with_stride_padded() {
        // Create a padded buffer via SIMD alignment
        let buf = simd_padded(2, 2, PixelDescriptor::RGBA8_SRGB, 64);
        let slice = buf.as_slice();
        let stride = slice.stride();
        // stride should be >= 8 (2 * 4 bpp) and aligned
        assert!(stride >= 8);
        // row_with_stride returns the full stride bytes including padding
        let full_row = slice.row_with_stride(0);
        assert_eq!(full_row.len(), stride);
        // row() returns only the pixel data (no padding)
        let pixel_row = slice.row(0);
        assert_eq!(pixel_row.len(), 8); // 2 pixels * 4 bytes
    }

    // --- PixelBuffer::has_alpha ---

    #[test]
    fn pixel_buffer_has_alpha_rgba8() {
        let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBA8_SRGB);
        assert!(buf.has_alpha());
    }

    #[test]
    fn pixel_buffer_has_alpha_rgb8_false() {
        let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);
        assert!(!buf.has_alpha());
    }

    // --- PixelBuffer::is_grayscale ---

    #[test]
    fn pixel_buffer_is_grayscale_gray8() {
        let buf = PixelBuffer::new(2, 2, PixelDescriptor::GRAY8_SRGB);
        assert!(buf.is_grayscale());
    }

    #[test]
    fn pixel_buffer_is_grayscale_rgb8_false() {
        let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);
        assert!(!buf.is_grayscale());
    }

    // --- PixelSlice::try_typed (requires "rgb" feature) ---

    #[cfg(feature = "rgb")]
    #[test]
    fn pixel_slice_try_typed_success() {
        use rgb::Rgba;

        let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBA8_SRGB);
        let slice = buf.as_slice();
        let typed: Option<PixelSlice<'_, Rgba<u8>>> = slice.try_typed();
        assert!(typed.is_some());
        let typed = typed.unwrap();
        assert_eq!(typed.width(), 2);
        assert_eq!(typed.rows(), 2);
    }

    // --- PixelBuffer::try_typed ---

    #[cfg(feature = "rgb")]
    #[test]
    fn pixel_buffer_try_typed_success() {
        use rgb::Rgba;

        let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGBA8_SRGB);
        let typed: Option<PixelBuffer<Rgba<u8>>> = buf.try_typed();
        assert!(typed.is_some());
        let typed = typed.unwrap();
        assert_eq!(typed.width(), 2);
        assert_eq!(typed.height(), 2);
    }

    #[cfg(feature = "rgb")]
    #[test]
    fn pixel_buffer_try_typed_failure_wrong_layout() {
        use rgb::Rgba;

        // RGB8 buffer cannot be typed as Rgba<u8>
        let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);
        let typed: Option<PixelBuffer<Rgba<u8>>> = buf.try_typed();
        assert!(typed.is_none());
    }

    // --- Security audit 2026-05-06: 32-bit overflow safety regressions ---
    //
    // These dimensions overflow `usize` arithmetic on 32-bit targets but are
    // representable on 64-bit, so the tests are gated to run under `cross`
    // or any other 32-bit build (e.g. i686-unknown-linux-gnu). On 64-bit the
    // overflow path is unreachable for `u32 x u32 x small-const` inputs;
    // running the suite under 32-bit exercises the actual guard.

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn from_vec_rejects_stride_height_overflow_32bit() {
        // RGB8: bpp=3. stride * height overflows 32-bit usize.
        // 0xFFFF * 3 = 0x2FFFD, * 0xFFFF heights overflows.
        let data = alloc::vec::Vec::<u8>::new();
        let err = PixelBuffer::from_vec(data, 0xFFFF, 0xFFFF, PixelDescriptor::RGB8_SRGB)
            .expect_err("expected overflow to be reported");
        assert_eq!(*err.error(), BufferError::InvalidDimensions);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    #[cfg(feature = "rgb")]
    fn from_pixels_rejects_width_height_overflow_32bit() {
        use rgb::Rgb;
        // 2^16 * 2^16 == 2^32, wraps to 0 on 32-bit. With the fix this
        // returns InvalidDimensions instead of accepting an empty vec
        // for a (65536, 65536) buffer.
        let pixels: alloc::vec::Vec<Rgb<u8>> = alloc::vec::Vec::new();
        let err = PixelBuffer::<Rgb<u8>>::from_pixels(pixels, 1 << 16, 1 << 16)
            .expect_err("expected width*height overflow to be reported");
        assert_eq!(*err.error(), BufferError::InvalidDimensions);
    }

    #[cfg(feature = "rgb")]
    #[test]
    fn from_pixels_length_mismatch_does_not_panic() {
        // On 64-bit this is the normal length-mismatch path; on 32-bit the
        // overflow guard activates first. Either way, we get a clean error
        // and never index out of bounds.
        use rgb::Rgb;
        let pixels: alloc::vec::Vec<Rgb<u8>> = alloc::vec![Rgb { r: 0, g: 0, b: 0 }; 4];
        // (3, 3) wants 9 pixels; we provided 4.
        let err = PixelBuffer::<Rgb<u8>>::from_pixels(pixels, 3, 3)
            .expect_err("length mismatch must error");
        assert_eq!(*err.error(), BufferError::InvalidDimensions);
    }

    #[test]
    fn from_vec_insufficient_data_clean_error() {
        // Standard insufficient-data path -- guards offset+total before the
        // length check post-fix. Verify the existing happy-path negative
        // still surfaces as InsufficientData (not panic) under the new
        // checked_add guard.
        let data = alloc::vec::Vec::<u8>::new();
        let err = PixelBuffer::from_vec(data, 4, 4, PixelDescriptor::RGB8_SRGB)
            .expect_err("empty data must error for non-zero dims");
        assert_eq!(*err.error(), BufferError::InsufficientData);
    }

    #[test]
    fn crop_copy_round_trip_does_not_overflow() {
        // Sanity check that crop_copy still works for normal inputs after
        // the overflow guard was added. The H1 fix only changes overflow
        // behavior (deferred slice OOB -> eager panic with named message);
        // we exercise that the happy path still returns a tightly-packed
        // buffer of the requested size.
        let buf = PixelBuffer::new(8, 4, PixelDescriptor::RGB8_SRGB);
        let cropped = buf.crop_copy(1, 1, 3, 2).unwrap();
        assert_eq!(cropped.width(), 3);
        assert_eq!(cropped.height(), 2);
        assert_eq!(cropped.stride(), 3 * 3); // tightly packed RGB8
    }
}

#[cfg(all(test, feature = "imgref"))]
mod buffer_tests {
    use super::*;
    use alloc::vec::Vec;
    use rgb::{Gray, Rgb};

    // --- ImgRef -> PixelSlice -> row access ---

    #[test]
    fn imgref_to_pixel_slice_rgb8() {
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 10,
                g: 20,
                b: 30,
            },
            Rgb {
                r: 40,
                g: 50,
                b: 60,
            },
            Rgb {
                r: 70,
                g: 80,
                b: 90,
            },
            Rgb {
                r: 100,
                g: 110,
                b: 120,
            },
        ];
        let img = imgref::Img::new(pixels.as_slice(), 2, 2);
        let slice: PixelSlice<'_, Rgb<u8>> = img.into();
        assert_eq!(slice.width(), 2);
        assert_eq!(slice.rows(), 2);
        assert_eq!(slice.row(0), &[10, 20, 30, 40, 50, 60]);
        assert_eq!(slice.row(1), &[70, 80, 90, 100, 110, 120]);
    }

    #[test]
    fn imgref_to_pixel_slice_gray16() {
        let pixels = vec![Gray::new(1000u16), Gray::new(2000u16)];
        let img = imgref::Img::new(pixels.as_slice(), 2, 1);
        let slice: PixelSlice<'_, Gray<u16>> = img.into();
        assert_eq!(slice.width(), 2);
        assert_eq!(slice.rows(), 1);
        assert_eq!(slice.descriptor(), PixelDescriptor::GRAY16);
        // Bytes should be native-endian u16
        let row = slice.row(0);
        assert_eq!(row.len(), 4);
        let v0 = u16::from_ne_bytes([row[0], row[1]]);
        let v1 = u16::from_ne_bytes([row[2], row[3]]);
        assert_eq!(v0, 1000);
        assert_eq!(v1, 2000);
    }

    // --- from_pixels_erased ---

    #[test]
    fn from_pixels_erased_matches_manual() {
        let pixels1: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 10,
                g: 20,
                b: 30,
            },
            Rgb {
                r: 40,
                g: 50,
                b: 60,
            },
        ];
        let pixels2 = pixels1.clone();

        // Manual: from_pixels + into
        let manual: PixelBuffer = PixelBuffer::from_pixels(pixels1, 2, 1).unwrap().into();

        // Erased: from_pixels_erased
        let erased = PixelBuffer::from_pixels_erased(pixels2, 2, 1).unwrap();

        assert_eq!(manual.width(), erased.width());
        assert_eq!(manual.height(), erased.height());
        assert_eq!(manual.descriptor(), erased.descriptor());
        assert_eq!(manual.as_slice().row(0), erased.as_slice().row(0));
    }

    #[test]
    fn from_pixels_erased_dimension_mismatch() {
        let pixels: Vec<Rgb<u8>> = vec![Rgb { r: 1, g: 2, b: 3 }];
        let err = PixelBuffer::from_pixels_erased(pixels, 2, 1);
        assert_eq!(*err.unwrap_err().error(), BufferError::InvalidDimensions);
    }
}
