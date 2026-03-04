use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

#[cfg(feature = "imgref")]
use imgref::{ImgRef, ImgVec};

use crate::color::ColorContext;
use crate::descriptor::{
    AlphaMode, ColorPrimaries, PixelDescriptor, SignalRange, TransferFunction,
};

#[cfg(feature = "rgb")]
use super::pixels_to_bytes;
use super::{BufferError, Pixel, PixelSlice, PixelSliceMut, align_offset, try_alloc_zeroed};

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
    pub(super) data: Vec<u8>,
    /// Byte offset from `data` start to the first aligned pixel.
    pub(super) offset: usize,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) stride: usize,
    pub(super) descriptor: PixelDescriptor,
    pub(super) color: Option<Arc<ColorContext>>,
    pub(super) _pixel: PhantomData<P>,
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
