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

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

#[cfg(feature = "imgref")]
use imgref::ImgRef;
#[cfg(feature = "imgref")]
use imgref::ImgVec;
#[cfg(feature = "rgb")]
use rgb::alt::BGRA;
#[cfg(feature = "rgb")]
use rgb::{Gray, Rgb, Rgba};

use crate::color::ColorContext;
use crate::descriptor::{
    AlphaMode, ColorPrimaries, PixelDescriptor, SignalRange, TransferFunction,
};
#[cfg(feature = "rgb")]
use crate::pixel_types::{GrayAlpha8, GrayAlpha16, GrayAlphaF32};

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

/// Try to allocate a zeroed `Vec<u8>` of the given size.
///
/// Returns [`BufferError::AllocationFailed`] if the allocation fails.
fn try_alloc_zeroed(size: usize) -> Result<Vec<u8>, BufferError> {
    let mut data = Vec::new();
    data.try_reserve_exact(size)
        .map_err(|_| BufferError::AllocationFailed)?;
    data.resize(size, 0);
    Ok(data)
}

/// Validate slice parameters (shared by erased and typed constructors).
fn validate_slice(
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
fn required_bytes(rows: u32, stride: usize, min_stride: usize) -> Result<usize, BufferError> {
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
fn pixels_to_bytes<P: bytemuck::Pod>(pixels: Vec<P>) -> Vec<u8> {
    match bytemuck::try_cast_vec(pixels) {
        Ok(bytes) => bytes,
        Err((_err, pixels)) => bytemuck::cast_slice::<P, u8>(&pixels).to_vec(),
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
    _pixel: PhantomData<P>,
}

impl<'a> PixelSlice<'a> {
    /// Create a new pixel slice with validation.
    ///
    /// `stride_bytes` is the byte distance between the start of consecutive rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is too small, the stride is too small,
    /// or the data is not aligned for the channel type.
    pub fn new(
        data: &'a [u8],
        width: u32,
        rows: u32,
        stride_bytes: usize,
        descriptor: PixelDescriptor,
    ) -> Result<Self, BufferError> {
        validate_slice(
            data.len(),
            data.as_ptr(),
            width,
            rows,
            stride_bytes,
            &descriptor,
        )?;
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
            _pixel: PhantomData,
        }
    }

    /// Try to reinterpret as a typed pixel slice.
    ///
    /// Succeeds if the descriptors are layout-compatible (same channel type
    /// and layout). Transfer function and alpha mode are metadata, not
    /// layout constraints.
    pub fn try_typed<Q: Pixel>(self) -> Option<PixelSlice<'a, Q>> {
        if self.descriptor.layout_compatible(Q::DESCRIPTOR) {
            Some(PixelSlice {
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
        self.descriptor = descriptor;
        self
    }

    /// Reinterpret the buffer with a different physical layout.
    ///
    /// Unlike [`with_descriptor()`](Self::with_descriptor), this allows
    /// changing `channel_type` and `layout`. The new descriptor must have
    /// the same `bytes_per_pixel()` as the current one.
    ///
    /// Use cases: treating RGBA8 data as BGRA8, RGBX8 as RGBA8.
    pub fn reinterpret(mut self, descriptor: PixelDescriptor) -> Result<Self, BufferError> {
        if self.descriptor.bytes_per_pixel() != descriptor.bytes_per_pixel() {
            return Err(BufferError::IncompatibleDescriptor);
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
        self.descriptor.signal_range = sr;
        self
    }

    /// Return a copy with a different alpha mode.
    #[inline]
    #[must_use]
    pub fn with_alpha_mode(mut self, am: Option<AlphaMode>) -> Self {
        self.descriptor.alpha = am;
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
        if self.is_contiguous() {
            let total = self.rows as usize * self.stride;
            Some(&self.data[..total])
        } else {
            None
        }
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
        if self.rows == 0 {
            return &[];
        }
        // Use rows*stride if the backing buffer is large enough (includes
        // trailing padding), otherwise trim the last row to width*bpp.
        let full = self.rows as usize * self.stride;
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
        if count == 0 {
            return PixelSlice {
                data: &[],
                width: self.width,
                rows: 0,
                stride: self.stride,
                descriptor: self.descriptor,

                color: self.color.clone(),
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
    pub fn new_typed(
        data: &'a [u8],
        width: u32,
        rows: u32,
        stride_pixels: u32,
    ) -> Result<Self, BufferError> {
        const { assert!(core::mem::size_of::<P>() == P::DESCRIPTOR.bytes_per_pixel()) }
        let stride_bytes = stride_pixels as usize * core::mem::size_of::<P>();
        validate_slice(
            data.len(),
            data.as_ptr(),
            width,
            rows,
            stride_bytes,
            &P::DESCRIPTOR,
        )?;
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
    /// `stride_bytes` is the byte distance between the start of consecutive rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is too small, the stride is too small,
    /// or the data is not aligned for the channel type.
    pub fn new(
        data: &'a mut [u8],
        width: u32,
        rows: u32,
        stride_bytes: usize,
        descriptor: PixelDescriptor,
    ) -> Result<Self, BufferError> {
        validate_slice(
            data.len(),
            data.as_ptr(),
            width,
            rows,
            stride_bytes,
            &descriptor,
        )?;
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
    /// Succeeds if the descriptors are layout-compatible.
    pub fn try_typed<Q: Pixel>(self) -> Option<PixelSliceMut<'a, Q>> {
        if self.descriptor.layout_compatible(Q::DESCRIPTOR) {
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
        self.descriptor = descriptor;
        self
    }

    /// Reinterpret the buffer with a different physical layout.
    ///
    /// See [`PixelSlice::reinterpret()`] for details.
    pub fn reinterpret(mut self, descriptor: PixelDescriptor) -> Result<Self, BufferError> {
        if self.descriptor.bytes_per_pixel() != descriptor.bytes_per_pixel() {
            return Err(BufferError::IncompatibleDescriptor);
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
        self.descriptor.signal_range = sr;
        self
    }

    /// Return a copy with a different alpha mode.
    #[inline]
    #[must_use]
    pub fn with_alpha_mode(mut self, am: Option<AlphaMode>) -> Self {
        self.descriptor.alpha = am;
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

    /// Zero-copy access to the raw backing bytes, including any stride padding.
    ///
    /// See [`PixelSlice::as_strided_bytes()`] for details.
    #[inline]
    pub fn as_strided_bytes(&self) -> &[u8] {
        self.data
    }

    /// Mutable access to the raw backing bytes, including any stride padding.
    ///
    /// See [`PixelSlice::as_strided_bytes()`] for details.
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
        if count == 0 {
            return PixelSliceMut {
                data: &mut [],
                width: self.width,
                rows: 0,
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
    pub fn new_typed(
        data: &'a mut [u8],
        width: u32,
        rows: u32,
        stride_pixels: u32,
    ) -> Result<Self, BufferError> {
        const { assert!(core::mem::size_of::<P>() == P::DESCRIPTOR.bytes_per_pixel()) }
        let stride_bytes = stride_pixels as usize * core::mem::size_of::<P>();
        validate_slice(
            data.len(),
            data.as_ptr(),
            width,
            rows,
            stride_bytes,
            &P::DESCRIPTOR,
        )?;
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
            descriptor: PixelDescriptor::BGRX8_SRGB,

            color,
            _pixel: PhantomData,
        }
    }
}

#[cfg(feature = "rgb")]
impl<'a> PixelSliceMut<'a, Rgbx> {
    /// Upgrade to RGBA by setting all padding bytes to 255 (fully opaque).
    #[must_use]
    pub fn upgrade_to_rgba(self) -> PixelSliceMut<'a, Rgba<u8>> {
        let width = self.width;
        let rows = self.rows;
        let stride = self.stride;

        let color = self.color;
        let data = self.data;
        for_each_pixel_4bpp(data, width, rows, stride, |px| {
            px[3] = 255;
        });
        PixelSliceMut {
            data,
            width,
            rows,
            stride,
            descriptor: PixelDescriptor::RGBA8_SRGB,

            color,
            _pixel: PhantomData,
        }
    }
}

impl<'a> PixelSliceMut<'a, Bgrx> {
    /// Byte-swap B<->R channels in place, converting to RGBX.
    #[must_use]
    pub fn swap_to_rgbx(self) -> PixelSliceMut<'a, Rgbx> {
        let width = self.width;
        let rows = self.rows;
        let stride = self.stride;

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
            descriptor: PixelDescriptor::RGBX8_SRGB,

            color,
            _pixel: PhantomData,
        }
    }
}

#[cfg(feature = "rgb")]
impl<'a> PixelSliceMut<'a, Bgrx> {
    /// Upgrade to BGRA by setting all padding bytes to 255 (fully opaque).
    #[must_use]
    pub fn upgrade_to_bgra(self) -> PixelSliceMut<'a, BGRA<u8>> {
        let width = self.width;
        let rows = self.rows;
        let stride = self.stride;

        let color = self.color;
        let data = self.data;
        for_each_pixel_4bpp(data, width, rows, stride, |px| {
            px[3] = 255;
        });
        PixelSliceMut {
            data,
            width,
            rows,
            stride,
            descriptor: PixelDescriptor::BGRA8_SRGB,

            color,
            _pixel: PhantomData,
        }
    }
}

#[cfg(feature = "rgb")]
impl<'a> PixelSliceMut<'a, Rgba<u8>> {
    /// Matte alpha against a solid RGB background, producing RGBX.
    ///
    /// Each pixel is composited: `out = src * alpha/255 + bg * (255 - alpha)/255`.
    /// The alpha byte becomes padding.
    #[must_use]
    pub fn matte_to_rgbx(self, bg: Rgb<u8>) -> PixelSliceMut<'a, Rgbx> {
        let width = self.width;
        let rows = self.rows;
        let stride = self.stride;

        let color = self.color;
        let data = self.data;
        for_each_pixel_4bpp(data, width, rows, stride, |px| {
            let a = px[3] as u16;
            let inv_a = 255 - a;
            px[0] = ((px[0] as u16 * a + bg.r as u16 * inv_a + 127) / 255) as u8;
            px[1] = ((px[1] as u16 * a + bg.g as u16 * inv_a + 127) / 255) as u8;
            px[2] = ((px[2] as u16 * a + bg.b as u16 * inv_a + 127) / 255) as u8;
            px[3] = 0;
        });
        PixelSliceMut {
            data,
            width,
            rows,
            stride,
            descriptor: PixelDescriptor::RGBX8_SRGB,

            color,
            _pixel: PhantomData,
        }
    }

    /// Strip alpha to RGBX without compositing (just mark as padding).
    ///
    /// The alpha byte value is preserved in memory but semantically ignored.
    /// Use when you know alpha is already 255 or don't care about the values.
    #[must_use]
    pub fn strip_alpha_to_rgbx(self) -> PixelSliceMut<'a, Rgbx> {
        PixelSliceMut {
            data: self.data,
            width: self.width,
            rows: self.rows,
            stride: self.stride,
            descriptor: PixelDescriptor::RGBX8_SRGB,

            color: self.color,
            _pixel: PhantomData,
        }
    }

    /// Byte-swap R<->B channels in place, converting to BGRA.
    #[must_use]
    pub fn swap_to_bgra(self) -> PixelSliceMut<'a, BGRA<u8>> {
        let width = self.width;
        let rows = self.rows;
        let stride = self.stride;

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
            descriptor: PixelDescriptor::BGRA8_SRGB,

            color,
            _pixel: PhantomData,
        }
    }
}

#[cfg(feature = "rgb")]
impl<'a> PixelSliceMut<'a, BGRA<u8>> {
    /// Matte alpha against a solid RGB background, producing BGRX.
    ///
    /// Each pixel is composited: `out = src * alpha/255 + bg * (255 - alpha)/255`.
    /// The alpha byte becomes padding.
    #[must_use]
    pub fn matte_to_bgrx(self, bg: Rgb<u8>) -> PixelSliceMut<'a, Bgrx> {
        let width = self.width;
        let rows = self.rows;
        let stride = self.stride;

        let color = self.color;
        let data = self.data;
        for_each_pixel_4bpp(data, width, rows, stride, |px| {
            let a = px[3] as u16;
            let inv_a = 255 - a;
            // BGRA layout: [B, G, R, A]
            px[0] = ((px[0] as u16 * a + bg.b as u16 * inv_a + 127) / 255) as u8;
            px[1] = ((px[1] as u16 * a + bg.g as u16 * inv_a + 127) / 255) as u8;
            px[2] = ((px[2] as u16 * a + bg.r as u16 * inv_a + 127) / 255) as u8;
            px[3] = 0;
        });
        PixelSliceMut {
            data,
            width,
            rows,
            stride,
            descriptor: PixelDescriptor::BGRX8_SRGB,

            color,
            _pixel: PhantomData,
        }
    }

    /// Strip alpha to BGRX without compositing (just mark as padding).
    #[must_use]
    pub fn strip_alpha_to_bgrx(self) -> PixelSliceMut<'a, Bgrx> {
        PixelSliceMut {
            data: self.data,
            width: self.width,
            rows: self.rows,
            stride: self.stride,
            descriptor: PixelDescriptor::BGRX8_SRGB,

            color: self.color,
            _pixel: PhantomData,
        }
    }

    /// Byte-swap B<->R channels in place, converting to RGBA.
    #[must_use]
    pub fn swap_to_rgba(self) -> PixelSliceMut<'a, Rgba<u8>> {
        let width = self.width;
        let rows = self.rows;
        let stride = self.stride;

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
            descriptor: PixelDescriptor::RGBA8_SRGB,

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
/// Wraps a `Vec<u8>` with an optional alignment offset so that pixel
/// rows start at the correct alignment for the channel type. The
/// backing vec can be recovered with [`into_vec`](Self::into_vec) for
/// pool reuse.
///
/// The type parameter `P` tracks pixel format at compile time, same as
/// [`PixelSlice`].
#[non_exhaustive]
pub struct PixelBuffer<P = ()> {
    data: Vec<u8>,
    /// Byte offset from `data` start to the first aligned pixel.
    offset: usize,
    width: u32,
    height: u32,
    stride: usize,
    descriptor: PixelDescriptor,
    color: Option<Arc<ColorContext>>,
    _pixel: PhantomData<P>,
}

impl PixelBuffer {
    /// Allocate a zero-filled buffer for the given dimensions and format.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails. Use [`try_new`](Self::try_new) for fallible allocation.
    pub fn new(width: u32, height: u32, descriptor: PixelDescriptor) -> Self {
        Self::try_new(width, height, descriptor).expect("pixel buffer allocation failed")
    }

    /// Try to allocate a zero-filled buffer for the given dimensions and format.
    ///
    /// Returns [`BufferError::InvalidDimensions`] if the total size overflows,
    /// or [`BufferError::AllocationFailed`] if allocation fails.
    pub fn try_new(
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
    ) -> Result<Self, BufferError> {
        let stride = descriptor.aligned_stride(width);
        let total = stride
            .checked_mul(height as usize)
            .ok_or(BufferError::InvalidDimensions)?;
        let align = descriptor.min_alignment();
        let alloc_size = total
            .checked_add(align - 1)
            .ok_or(BufferError::InvalidDimensions)?;
        let data = try_alloc_zeroed(alloc_size)?;
        let offset = align_offset(data.as_ptr(), align);
        Ok(Self {
            data,
            offset,
            width,
            height,
            stride,
            descriptor,

            color: None,
            _pixel: PhantomData,
        })
    }

    /// Allocate a SIMD-aligned buffer for the given dimensions and format.
    ///
    /// Row stride is a multiple of `lcm(bpp, simd_align)`, ensuring every
    /// row start is both pixel-aligned and SIMD-aligned when the buffer
    /// itself starts at a SIMD-aligned address.
    ///
    /// `simd_align` must be a power of 2 (e.g. 16, 32, 64).
    ///
    /// # Panics
    ///
    /// Panics if allocation fails. Use [`try_new_simd_aligned`](Self::try_new_simd_aligned)
    /// for fallible allocation.
    pub fn new_simd_aligned(
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
        simd_align: usize,
    ) -> Self {
        Self::try_new_simd_aligned(width, height, descriptor, simd_align)
            .expect("pixel buffer SIMD-aligned allocation failed")
    }

    /// Try to allocate a SIMD-aligned buffer for the given dimensions and format.
    ///
    /// Returns [`BufferError::InvalidDimensions`] if the total size overflows,
    /// or [`BufferError::AllocationFailed`] if allocation fails.
    pub fn try_new_simd_aligned(
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
        simd_align: usize,
    ) -> Result<Self, BufferError> {
        let stride = descriptor.simd_aligned_stride(width, simd_align);
        let total = stride
            .checked_mul(height as usize)
            .ok_or(BufferError::InvalidDimensions)?;
        let alloc_size = total
            .checked_add(simd_align - 1)
            .ok_or(BufferError::InvalidDimensions)?;
        let data = try_alloc_zeroed(alloc_size)?;
        let offset = align_offset(data.as_ptr(), simd_align);
        Ok(Self {
            data,
            offset,
            width,
            height,
            stride,
            descriptor,

            color: None,
            _pixel: PhantomData,
        })
    }

    /// Wrap an existing `Vec<u8>` as a pixel buffer.
    ///
    /// The vec must be large enough to hold `aligned_stride(width) * height`
    /// bytes (plus any alignment offset). Stride is computed from the
    /// descriptor -- rows are assumed tightly packed.
    ///
    /// # Errors
    ///
    /// Returns [`BufferError::InsufficientData`] if the vec is too small.
    pub fn from_vec(
        data: Vec<u8>,
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
    ) -> Result<Self, BufferError> {
        let stride = descriptor.aligned_stride(width);
        let total = stride
            .checked_mul(height as usize)
            .ok_or(BufferError::InvalidDimensions)?;
        let align = descriptor.min_alignment();
        let offset = align_offset(data.as_ptr(), align);
        if data.len() < offset + total {
            return Err(BufferError::InsufficientData);
        }
        Ok(Self {
            data,
            offset,
            width,
            height,
            stride,
            descriptor,

            color: None,
            _pixel: PhantomData,
        })
    }
}

impl<P: Pixel> PixelBuffer<P> {
    /// Allocate a typed zero-filled buffer for the given dimensions.
    ///
    /// The descriptor is derived from `P::DESCRIPTOR`.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails. Use [`try_new_typed`](Self::try_new_typed)
    /// for fallible allocation.
    pub fn new_typed(width: u32, height: u32) -> Self {
        Self::try_new_typed(width, height).expect("typed pixel buffer allocation failed")
    }

    /// Try to allocate a typed zero-filled buffer for the given dimensions.
    ///
    /// Returns [`BufferError::InvalidDimensions`] if the total size overflows,
    /// or [`BufferError::AllocationFailed`] if allocation fails.
    pub fn try_new_typed(width: u32, height: u32) -> Result<Self, BufferError> {
        const { assert!(core::mem::size_of::<P>() == P::DESCRIPTOR.bytes_per_pixel()) }
        let descriptor = P::DESCRIPTOR;
        let stride = descriptor.aligned_stride(width);
        let total = stride
            .checked_mul(height as usize)
            .ok_or(BufferError::InvalidDimensions)?;
        let align = descriptor.min_alignment();
        let alloc_size = total
            .checked_add(align - 1)
            .ok_or(BufferError::InvalidDimensions)?;
        let data = try_alloc_zeroed(alloc_size)?;
        let offset = align_offset(data.as_ptr(), align);
        Ok(Self {
            data,
            offset,
            width,
            height,
            stride,
            descriptor,

            color: None,
            _pixel: PhantomData,
        })
    }
}

#[cfg(feature = "rgb")]
impl<P: Pixel> PixelBuffer<P> {
    /// Construct from a typed pixel `Vec`.
    ///
    /// Zero-copy when `P` has alignment 1 (u8-component types like `Rgb<u8>`).
    /// Copies the data for types with higher alignment (`Rgb<u16>`, `Rgb<f32>`, etc.)
    /// because `Vec` tracks allocation alignment and `Vec<u8>` requires alignment 1.
    ///
    /// # Errors
    ///
    /// Returns [`BufferError::InvalidDimensions`] if `pixels.len() != width * height`.
    pub fn from_pixels(pixels: Vec<P>, width: u32, height: u32) -> Result<Self, BufferError> {
        const { assert!(core::mem::size_of::<P>() == P::DESCRIPTOR.bytes_per_pixel()) }
        let expected = width as usize * height as usize;
        if pixels.len() != expected {
            return Err(BufferError::InvalidDimensions);
        }
        let descriptor = P::DESCRIPTOR;
        let stride = descriptor.aligned_stride(width);
        let data: Vec<u8> = pixels_to_bytes(pixels);
        Ok(Self {
            data,
            offset: 0,
            width,
            height,
            stride,
            descriptor,

            color: None,
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
    pub fn from_imgvec(img: ImgVec<P>) -> Self {
        const { assert!(core::mem::size_of::<P>() == P::DESCRIPTOR.bytes_per_pixel()) }
        let width = img.width() as u32;
        let height = img.height() as u32;
        let stride_pixels = img.stride();
        let descriptor = P::DESCRIPTOR;
        let stride_bytes = stride_pixels * core::mem::size_of::<P>();
        let (buf, ..) = img.into_contiguous_buf();
        let data: Vec<u8> = pixels_to_bytes(buf);
        Self {
            data,
            offset: 0,
            width,
            height,
            stride: stride_bytes,
            descriptor,

            color: None,
            _pixel: PhantomData,
        }
    }
}

#[cfg(feature = "imgref")]
impl<P: Pixel> PixelBuffer<P> {
    /// Borrow the buffer as an [`ImgRef`].
    ///
    /// Zero-copy: reinterprets the raw bytes as typed pixels via
    /// [`bytemuck::cast_slice`].
    ///
    /// # Panics
    ///
    /// Panics if the stride is not pixel-aligned (always succeeds for
    /// buffers created via `new_typed()`, `from_pixels()`, or `from_imgvec()`).
    pub fn as_imgref(&self) -> ImgRef<'_, P> {
        let total_bytes = if self.height == 0 {
            0
        } else {
            (self.height as usize - 1) * self.stride
                + self.width as usize * core::mem::size_of::<P>()
        };
        let data = &self.data[self.offset..self.offset + total_bytes];
        let pixels: &[P] = bytemuck::cast_slice(data);
        let stride_px = self.stride / core::mem::size_of::<P>();
        imgref::Img::new_stride(pixels, self.width as usize, self.height as usize, stride_px)
    }

    /// Borrow the buffer as a mutable [`ImgRefMut`](imgref::ImgRefMut).
    ///
    /// Zero-copy: reinterprets the raw bytes as typed pixels.
    pub fn as_imgref_mut(&mut self) -> imgref::ImgRefMut<'_, P> {
        let total_bytes = if self.height == 0 {
            0
        } else {
            (self.height as usize - 1) * self.stride
                + self.width as usize * core::mem::size_of::<P>()
        };
        let offset = self.offset;
        let data = &mut self.data[offset..offset + total_bytes];
        let pixels: &mut [P] = bytemuck::cast_slice_mut(data);
        let stride_px = self.stride / core::mem::size_of::<P>();
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
    pub fn from_pixels_erased<P: Pixel>(
        pixels: Vec<P>,
        width: u32,
        height: u32,
    ) -> Result<Self, BufferError> {
        PixelBuffer::<P>::from_pixels(pixels, width, height).map(PixelBuffer::from)
    }

    /// Zero-copy access to the pixel data as a typed slice.
    ///
    /// Returns `Some(&[P])` if the descriptor is layout-compatible with `P`
    /// and rows are tightly packed (no stride padding). Returns `None` otherwise.
    ///
    /// This is a convenience for the common case where you just need a flat
    /// pixel slice without imgref metadata. For strided access, use
    /// [`try_as_imgref()`](Self::try_as_imgref).
    pub fn as_contiguous_pixels<P: Pixel>(&self) -> Option<&[P]> {
        if !self.descriptor.layout_compatible(P::DESCRIPTOR) {
            return None;
        }
        let pixel_size = core::mem::size_of::<P>();
        let row_bytes = self.width as usize * pixel_size;
        if pixel_size == 0 || self.stride != row_bytes {
            return None;
        }
        let total = row_bytes * self.height as usize;
        let data = &self.data[self.offset..self.offset + total];
        Some(bytemuck::cast_slice(data))
    }

    /// Consume the buffer and return the pixels as a typed `Vec<P>`.
    ///
    /// Returns `None` if the descriptor is not layout-compatible with `P`.
    /// Strips stride padding if present. Zero-copy when the buffer is
    /// tightly packed and `P` has alignment 1 (u8-component types like
    /// `Rgb<u8>`, `Rgba<u8>`); copies otherwise.
    pub fn into_contiguous_pixels<P: Pixel>(self) -> Option<Vec<P>> {
        if !self.descriptor.layout_compatible(P::DESCRIPTOR) {
            return None;
        }
        let pixel_size = core::mem::size_of::<P>();
        if pixel_size == 0 {
            return None;
        }
        let row_bytes = self.width as usize * pixel_size;
        let total_pixels = self.width as usize * self.height as usize;

        if self.stride == row_bytes && self.offset == 0 {
            // Fast path: tightly packed, no offset -- try zero-copy reinterpret
            let mut data = self.data;
            data.truncate(total_pixels * pixel_size);
            match bytemuck::try_cast_vec(data) {
                Ok(pixels) => return Some(pixels),
                Err((_err, data)) => {
                    // Alignment mismatch -- copy
                    return Some(
                        bytemuck::cast_slice::<u8, P>(&data[..total_pixels * pixel_size]).to_vec(),
                    );
                }
            }
        }

        // Slow path: has offset or stride padding -- copy row by row
        let mut out = Vec::with_capacity(total_pixels);
        for y in 0..self.height as usize {
            let row_start = self.offset + y * self.stride;
            let row_data = &self.data[row_start..row_start + row_bytes];
            out.extend_from_slice(bytemuck::cast_slice(row_data));
        }
        Some(out)
    }
}

/// Imgref interop for type-erased PixelBuffer.
#[cfg(feature = "imgref")]
impl PixelBuffer {
    /// Try to borrow the buffer as a typed [`ImgRef`].
    ///
    /// Returns `None` if the descriptor is not layout-compatible with `P`.
    pub fn try_as_imgref<P: Pixel>(&self) -> Option<ImgRef<'_, P>> {
        if !self.descriptor.layout_compatible(P::DESCRIPTOR) {
            return None;
        }
        let pixel_size = core::mem::size_of::<P>();
        if pixel_size == 0 || !self.stride.is_multiple_of(pixel_size) {
            return None;
        }
        let total_bytes = if self.height == 0 {
            0
        } else {
            (self.height as usize - 1) * self.stride + self.width as usize * pixel_size
        };
        let data = &self.data[self.offset..self.offset + total_bytes];
        let pixels: &[P] = bytemuck::cast_slice(data);
        let stride_px = self.stride / pixel_size;
        Some(imgref::Img::new_stride(
            pixels,
            self.width as usize,
            self.height as usize,
            stride_px,
        ))
    }

    /// Try to borrow the buffer as a typed mutable [`ImgRefMut`](imgref::ImgRefMut).
    ///
    /// Returns `None` if the descriptor is not layout-compatible with `P`.
    pub fn try_as_imgref_mut<P: Pixel>(&mut self) -> Option<imgref::ImgRefMut<'_, P>> {
        if !self.descriptor.layout_compatible(P::DESCRIPTOR) {
            return None;
        }
        let pixel_size = core::mem::size_of::<P>();
        if pixel_size == 0 || !self.stride.is_multiple_of(pixel_size) {
            return None;
        }
        let total_bytes = if self.height == 0 {
            0
        } else {
            (self.height as usize - 1) * self.stride + self.width as usize * pixel_size
        };
        let offset = self.offset;
        let data = &mut self.data[offset..offset + total_bytes];
        let pixels: &mut [P] = bytemuck::cast_slice_mut(data);
        let stride_px = self.stride / pixel_size;
        Some(imgref::Img::new_stride(
            pixels,
            self.width as usize,
            self.height as usize,
            stride_px,
        ))
    }
}

impl<P> PixelBuffer<P> {
    /// Erase the pixel type, returning a type-erased buffer.
    pub fn erase(self) -> PixelBuffer {
        PixelBuffer {
            data: self.data,
            offset: self.offset,
            width: self.width,
            height: self.height,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color,
            _pixel: PhantomData,
        }
    }

    /// Try to reinterpret as a typed pixel buffer.
    ///
    /// Succeeds if the descriptors are layout-compatible.
    pub fn try_typed<Q: Pixel>(self) -> Option<PixelBuffer<Q>> {
        if self.descriptor.layout_compatible(Q::DESCRIPTOR) {
            Some(PixelBuffer {
                data: self.data,
                offset: self.offset,
                width: self.width,
                height: self.height,
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
        self.descriptor = descriptor;
        self
    }

    /// Reinterpret the buffer with a different physical layout.
    ///
    /// See [`PixelSlice::reinterpret()`] for details.
    pub fn reinterpret(mut self, descriptor: PixelDescriptor) -> Result<Self, BufferError> {
        if self.descriptor.bytes_per_pixel() != descriptor.bytes_per_pixel() {
            return Err(BufferError::IncompatibleDescriptor);
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
        self.descriptor.signal_range = sr;
        self
    }

    /// Return a copy with a different alpha mode.
    #[inline]
    #[must_use]
    pub fn with_alpha_mode(mut self, am: Option<AlphaMode>) -> Self {
        self.descriptor.alpha = am;
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

    /// Consume the buffer and return the backing `Vec<u8>` for pool reuse.
    pub fn into_vec(self) -> Vec<u8> {
        self.data
    }

    /// Zero-copy access to the raw pixel bytes when rows are tightly packed.
    ///
    /// Returns `Some(&[u8])` if `stride == width * bpp` (no padding),
    /// `None` if rows have stride padding.
    #[inline]
    pub fn as_contiguous_bytes(&self) -> Option<&[u8]> {
        let bpp = self.descriptor.bytes_per_pixel();
        let row_bytes = self.width as usize * bpp;
        if self.stride == row_bytes {
            let total = row_bytes * self.height as usize;
            Some(&self.data[self.offset..self.offset + total])
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
            let start = self.offset;
            return self.data[start..start + total].to_vec();
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

    /// Borrow the full buffer as an immutable [`PixelSlice`].
    pub fn as_slice(&self) -> PixelSlice<'_, P> {
        let total = self.stride * self.height as usize;
        PixelSlice {
            data: &self.data[self.offset..self.offset + total],
            width: self.width,
            rows: self.height,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color.clone(),
            _pixel: PhantomData,
        }
    }

    /// Borrow the full buffer as a mutable [`PixelSliceMut`].
    pub fn as_slice_mut(&mut self) -> PixelSliceMut<'_, P> {
        let total = self.stride * self.height as usize;
        let offset = self.offset;
        PixelSliceMut {
            data: &mut self.data[offset..offset + total],
            width: self.width,
            rows: self.height,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color.clone(),
            _pixel: PhantomData,
        }
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
        if count == 0 {
            return PixelSlice {
                data: &[],
                width: self.width,
                rows: 0,
                stride: self.stride,
                descriptor: self.descriptor,

                color: self.color.clone(),
                _pixel: PhantomData,
            };
        }
        let bpp = self.descriptor.bytes_per_pixel();
        let start = self.offset + y as usize * self.stride;
        let end = self.offset
            + (y as usize + count as usize - 1) * self.stride
            + self.width as usize * bpp;
        PixelSlice {
            data: &self.data[start..end],
            width: self.width,
            rows: count,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color.clone(),
            _pixel: PhantomData,
        }
    }

    /// Borrow a range of rows as a mutable [`PixelSliceMut`].
    ///
    /// # Panics
    ///
    /// Panics if `y + count > height`.
    pub fn rows_mut(&mut self, y: u32, count: u32) -> PixelSliceMut<'_, P> {
        assert!(
            y.checked_add(count).is_some_and(|end| end <= self.height),
            "rows_mut({y}, {count}) out of bounds (height: {})",
            self.height
        );
        if count == 0 {
            return PixelSliceMut {
                data: &mut [],
                width: self.width,
                rows: 0,
                stride: self.stride,
                descriptor: self.descriptor,

                color: self.color.clone(),
                _pixel: PhantomData,
            };
        }
        let bpp = self.descriptor.bytes_per_pixel();
        let start = self.offset + y as usize * self.stride;
        let end = self.offset
            + (y as usize + count as usize - 1) * self.stride
            + self.width as usize * bpp;
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
                _pixel: PhantomData,
            };
        }
        let bpp = self.descriptor.bytes_per_pixel();
        let start = self.offset + y as usize * self.stride + x as usize * bpp;
        let end = self.offset
            + (y as usize + h as usize - 1) * self.stride
            + (x as usize + w as usize) * bpp;
        PixelSlice {
            data: &self.data[start..end],
            width: w,
            rows: h,
            stride: self.stride,
            descriptor: self.descriptor,

            color: self.color.clone(),
            _pixel: PhantomData,
        }
    }

    /// Copy a sub-region into a new, tightly-packed [`PixelBuffer`].
    ///
    /// # Panics
    ///
    /// Panics if the crop region is out of bounds.
    pub fn crop_copy(&self, x: u32, y: u32, w: u32, h: u32) -> PixelBuffer<P> {
        let src = self.crop_view(x, y, w, h);
        let stride = self.descriptor.aligned_stride(w);
        let total = stride * h as usize;
        let align = self.descriptor.min_alignment();
        let alloc_size = total + align - 1;
        let data = vec![0u8; alloc_size];
        let offset = align_offset(data.as_ptr(), align);
        let mut dst = PixelBuffer {
            data,
            offset,
            width: w,
            height: h,
            stride,
            descriptor: self.descriptor,

            color: self.color.clone(),
            _pixel: PhantomData,
        };
        let bpp = self.descriptor.bytes_per_pixel();
        let row_bytes = w as usize * bpp;
        for row_y in 0..h {
            let src_row = src.row(row_y);
            let dst_start = dst.offset + row_y as usize * dst.stride;
            dst.data[dst_start..dst_start + row_bytes].copy_from_slice(&src_row[..row_bytes]);
        }
        dst
    }
}

impl<P> fmt::Debug for PixelBuffer<P> {
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{ChannelLayout, ChannelType};
    use alloc::format;
    use alloc::vec;

    // --- PixelBuffer allocation and row access ---

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
        assert_eq!(err.unwrap_err(), BufferError::InsufficientData);
    }

    #[test]
    fn pixel_buffer_into_vec_roundtrip() {
        let buf = PixelBuffer::new(4, 4, PixelDescriptor::RGBA8_SRGB);
        let v = buf.into_vec();
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
        let buf = PixelBuffer::new_simd_aligned(10, 5, PixelDescriptor::RGBA8_SRGB, 64);
        assert_eq!(buf.width(), 10);
        assert_eq!(buf.height(), 5);
        // RGBA8 bpp=4, lcm(4,64)=64, raw=40 -> stride=64
        assert_eq!(buf.stride(), 64);
        // First row should be 64-byte aligned
        let slice = buf.as_slice();
        assert_eq!(slice.data.as_ptr() as usize % 64, 0);
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
        let cropped = buf.crop_copy(1, 1, 2, 2);
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
        assert_eq!(err.unwrap_err(), BufferError::StrideTooSmall);
    }

    #[test]
    fn pixel_slice_insufficient_data() {
        let data = [0u8; 10];
        let err = PixelSlice::new(&data, 10, 1, 30, PixelDescriptor::RGB8_SRGB);
        assert_eq!(err.unwrap_err(), BufferError::InsufficientData);
    }

    #[test]
    fn pixel_slice_zero_rows() {
        let data = [0u8; 0];
        let slice = PixelSlice::new(&data, 10, 0, 30, PixelDescriptor::RGB8_SRGB).unwrap();
        assert_eq!(slice.rows(), 0);
    }

    #[test]
    fn stride_not_pixel_aligned_rejected() {
        // RGB8 bpp=3, stride=32 is not a multiple of 3
        let data = [0u8; 128];
        let err = PixelSlice::new(&data, 10, 1, 32, PixelDescriptor::RGB8_SRGB);
        assert_eq!(err.unwrap_err(), BufferError::StrideNotPixelAligned);

        // stride=33 IS a multiple of 3 -> accepted
        let ok = PixelSlice::new(&data, 10, 1, 33, PixelDescriptor::RGB8_SRGB);
        assert!(ok.is_ok());
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
        assert_eq!(err.unwrap_err(), BufferError::IncompatibleDescriptor);
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
        let buf = PixelBuffer::new(2, 2, PixelDescriptor::RGB8);
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
        let mut buf = PixelBuffer::new_simd_aligned(2, 2, PixelDescriptor::RGB8_SRGB, 16);
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
        assert_eq!(msg, "new descriptor has different bytes_per_pixel");
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
        let buf = PixelBuffer::new_simd_aligned(2, 2, PixelDescriptor::RGB8_SRGB, 16);
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
        let buf = PixelBuffer::new_simd_aligned(2, 2, PixelDescriptor::RGB8_SRGB, 16);
        let slice = buf.as_slice();
        let stride = slice.stride();
        assert!(stride > 6, "expected padding for SIMD alignment");
        let bytes = slice.as_strided_bytes();
        // Total length includes stride padding between and after rows
        assert_eq!(bytes.len(), stride * 2);
    }

    #[test]
    fn pixel_slice_as_strided_bytes_sub_rows() {
        let buf = PixelBuffer::new_simd_aligned(2, 4, PixelDescriptor::RGB8_SRGB, 16);
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
        let mut buf = PixelBuffer::new_simd_aligned(2, 2, PixelDescriptor::RGB8_SRGB, 16);
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
        let buf = PixelBuffer::new_simd_aligned(2, 2, PixelDescriptor::RGBA8_SRGB, 64);
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
}

#[cfg(all(test, feature = "imgref"))]
mod buffer_tests {
    use super::*;
    use alloc::vec;
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
        assert_eq!(err.unwrap_err(), BufferError::InvalidDimensions);
    }
}
