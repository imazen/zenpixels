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

use alloc::vec::Vec;
use core::fmt;

#[cfg(feature = "rgb")]
use rgb::alt::BGRA;
#[cfg(feature = "rgb")]
use rgb::{Gray, Rgb, Rgba};

use crate::descriptor::PixelDescriptor;
#[cfg(feature = "rgb")]
use crate::pixel_types::{GrayAlpha8, GrayAlpha16, GrayAlphaF32};

mod conversions;
mod pixel_buffer;
mod slice;
mod slice_mut;
#[cfg(test)]
mod tests;

pub use pixel_buffer::*;
pub use slice::*;
pub use slice_mut::*;

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

#[cfg(feature = "rgb")]
impl Pixel for GrayAlpha8 {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::GRAYA8;
}

#[cfg(feature = "rgb")]
impl Pixel for GrayAlpha16 {
    const DESCRIPTOR: PixelDescriptor = PixelDescriptor::GRAYA16;
}

#[cfg(feature = "rgb")]
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
    /// Data pointer is not aligned for the channel type.
    AlignmentViolation,
    /// Data slice is too small for the given dimensions and stride.
    InsufficientData,
    /// Stride is smaller than `width * bytes_per_pixel`.
    StrideTooSmall,
    /// Stride is not a multiple of `bytes_per_pixel`.
    ///
    /// Every row must start on a pixel boundary. If stride is not a
    /// multiple of bpp, rows after the first will be misaligned.
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
                write!(f, "new descriptor has different bytes_per_pixel")
            }
            Self::AllocationFailed => write!(f, "buffer allocation failed"),
        }
    }
}

impl core::error::Error for BufferError {}

// ---------------------------------------------------------------------------
// Helper functions (pub(super) for sibling submodules)
// ---------------------------------------------------------------------------

/// Round `val` up to the next multiple of `align` (must be a power of 2).
pub(super) const fn align_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

/// Compute the byte offset needed to align `ptr` to `align`.
pub(super) fn align_offset(ptr: *const u8, align: usize) -> usize {
    let addr = ptr as usize;
    align_up(addr, align) - addr
}

/// Try to allocate a zeroed `Vec<u8>` of the given size.
///
/// Returns [`BufferError::AllocationFailed`] if the allocation fails.
pub(super) fn try_alloc_zeroed(size: usize) -> Result<Vec<u8>, BufferError> {
    let mut data = Vec::new();
    data.try_reserve_exact(size)
        .map_err(|_| BufferError::AllocationFailed)?;
    data.resize(size, 0);
    Ok(data)
}

/// Validate slice parameters (shared by erased and typed constructors).
pub(super) fn validate_slice(
    data_len: usize,
    data_ptr: *const u8,
    width: u32,
    rows: u32,
    stride_bytes: usize,
    descriptor: &PixelDescriptor,
) -> Result<(), BufferError> {
    let bpp = descriptor.bytes_per_pixel();
    let min_stride = (width as usize)
        .checked_mul(bpp)
        .ok_or(BufferError::InvalidDimensions)?;
    if stride_bytes < min_stride {
        return Err(BufferError::StrideTooSmall);
    }
    if bpp > 0 && !stride_bytes.is_multiple_of(bpp) {
        return Err(BufferError::StrideNotPixelAligned);
    }
    if rows > 0 {
        let required = required_bytes(rows, stride_bytes, min_stride)?;
        if data_len < required {
            return Err(BufferError::InsufficientData);
        }
    }
    let align = descriptor.min_alignment();
    if !(data_ptr as usize).is_multiple_of(align) {
        return Err(BufferError::AlignmentViolation);
    }
    Ok(())
}

/// Minimum bytes needed: `(rows - 1) * stride + min_stride`.
pub(super) fn required_bytes(
    rows: u32,
    stride: usize,
    min_stride: usize,
) -> Result<usize, BufferError> {
    let preceding = (rows as usize - 1)
        .checked_mul(stride)
        .ok_or(BufferError::InvalidDimensions)?;
    preceding
        .checked_add(min_stride)
        .ok_or(BufferError::InvalidDimensions)
}

/// Convert `Vec<P>` to `Vec<u8>`. Zero-copy when alignment matches (u8-component
/// types), copies via `cast_slice` otherwise.
#[cfg(feature = "rgb")]
pub(super) fn pixels_to_bytes<P: bytemuck::Pod>(pixels: Vec<P>) -> Vec<u8> {
    match bytemuck::try_cast_vec(pixels) {
        Ok(bytes) => bytes,
        Err((_err, pixels)) => bytemuck::cast_slice::<P, u8>(&pixels).to_vec(),
    }
}
