use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

use crate::color::ColorContext;
use crate::descriptor::{
    AlphaMode, ColorPrimaries, PixelDescriptor, SignalRange, TransferFunction,
};

use super::{BufferError, Pixel, validate_slice};

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
    pub(super) data: &'a [u8],
    pub(super) width: u32,
    pub(super) rows: u32,
    pub(super) stride: usize,
    pub(super) descriptor: PixelDescriptor,
    pub(super) color: Option<Arc<ColorContext>>,
    pub(super) _pixel: PhantomData<P>,
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
    pub fn with_transfer(mut self, tf: TransferFunction) -> Self {
        self.descriptor.transfer = tf;
        self
    }

    /// Return a copy with different color primaries.
    #[inline]
    pub fn with_primaries(mut self, cp: ColorPrimaries) -> Self {
        self.descriptor.primaries = cp;
        self
    }

    /// Return a copy with a different signal range.
    #[inline]
    pub fn with_signal_range(mut self, sr: SignalRange) -> Self {
        self.descriptor.signal_range = sr;
        self
    }

    /// Return a copy with a different alpha mode.
    #[inline]
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
