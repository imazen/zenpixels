use core::marker::PhantomData;

use imgref::{ImgRef, ImgVec};
use rgb::{Bgr, Bgra, Gray, Rgb, Rgba};
use whereat::At;

use crate::descriptor::{AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};
use crate::pixel_types::{F16Sample, GrayAlpha8, GrayAlpha16, GrayAlphaF16, GrayAlphaF32};

use super::pixel_storage::try_pixels_into_storage;
use super::{
    Bgrx, BufferError, ByteStorage, ByteStorageMut, Pixel, PixelBuffer, PixelSlice, PixelSliceMut,
    Rgbx, StorageSample, checked_stride_samples, err_at, rows_aligned_for, validate_slice,
};

#[track_caller]
fn check_imgvec_geom(
    width_px: usize,
    height_px: usize,
    stride_px: usize,
    pixel_size: usize,
    descriptor: PixelDescriptor,
) -> Result<(u32, u32, u32), At<BufferError>> {
    let width = u32::try_from(width_px).map_err(|_| err_at(BufferError::InvalidDimensions))?;
    let height = u32::try_from(height_px).map_err(|_| err_at(BufferError::InvalidDimensions))?;
    descriptor.validate().map_err(err_at)?;
    let stride_bytes = stride_px
        .checked_mul(pixel_size)
        .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
    let stride_samples = checked_stride_samples(stride_bytes, &descriptor).map_err(err_at)?;
    Ok((width, height, stride_samples))
}

impl<P: Pixel> PixelBuffer<P> {
    /// Construct from a typed `ImgVec`, preserving its allocation and stride.
    ///
    /// Zero-copy for all built-in sample-aligned pixel types (`u8`, `u16`,
    /// [`F16Sample`], `f32`, etc.). Copies only for custom over-aligned `Pixel`
    /// types whose alignment exceeds their channel sample alignment.
    ///
    /// # Panics
    ///
    /// Panics if dimensions overflow `u32` or if the custom over-aligned copy
    /// path fails to allocate — use [`try_from_imgvec`](Self::try_from_imgvec)
    /// to recover instead.
    pub fn from_imgvec(img: ImgVec<P>) -> Self {
        Self::try_from_imgvec(img).expect("pixel buffer from ImgVec failed")
    }

    /// Fallible [`from_imgvec`](Self::from_imgvec): reports dimension overflow
    /// or a failed over-aligned copy allocation instead of panicking.
    ///
    /// Zero-copy for all built-in sample-aligned pixel types (`u8`, `u16`,
    /// [`F16Sample`], `f32`, etc.).
    ///
    /// # Errors
    ///
    /// [`BufferError::InvalidDimensions`] when a dimension doesn't fit `u32`
    /// or the stride overflows `usize`, and
    /// [`BufferError::AllocationFailed`] when storage for a custom over-aligned
    /// `P` can't be reserved.
    #[track_caller]
    pub fn try_from_imgvec(img: ImgVec<P>) -> Result<Self, At<BufferError>> {
        const { assert!(core::mem::size_of::<P>() == P::DESCRIPTOR.bytes_per_pixel()) }
        let descriptor = P::DESCRIPTOR;
        let (width, height, stride_samples) = check_imgvec_geom(
            img.width(),
            img.height(),
            img.stride(),
            core::mem::size_of::<P>(),
            descriptor,
        )?;
        let (data, offset) = try_pixels_into_storage(img.into_buf(), descriptor.channel_type())
            .map_err(|_| err_at(BufferError::AllocationFailed))?;
        Ok(Self {
            data,
            offset,
            width,
            height,
            stride_samples,
            descriptor,
            color: None,
            exact: None,
            _pixel: PhantomData,
        })
    }

    /// Consume this buffer and return an [`ImgVec<P>`], moving the allocation
    /// zero-copy whenever possible and compacting/copying once only when needed
    /// (non-zero offset, non-pixel-multiple byte stride, or incompatible `Vec`
    /// capacity/alignment).
    ///
    /// # Panics
    ///
    /// Panics if the fallback allocation fails; use [`try_into_imgvec`](Self::try_into_imgvec)
    /// for fallible recovery.
    #[track_caller]
    pub fn into_imgvec(self) -> ImgVec<P> {
        self.try_into_imgvec()
            .expect("pixel buffer to ImgVec allocation failed")
    }

    /// Fallible [`into_imgvec`](Self::into_imgvec): moves the allocation zero-copy
    /// when [`ImgVec::try_from`] succeeds, and compacts/copies once otherwise.
    ///
    /// # Errors
    ///
    /// [`BufferError::AllocationFailed`] if the fallback copy reserve fails.
    pub fn try_into_imgvec(self) -> Result<ImgVec<P>, At<BufferError>> {
        match ImgVec::try_from(self) {
            Ok(img) => Ok(img),
            Err(buffer) => {
                let (w, h) = (buffer.width as usize, buffer.height as usize);
                let pixels = buffer
                    .erase()
                    .try_into_contiguous_pixels::<P>()?
                    .expect("typed PixelBuffer<P> is layout-compatible with P");
                Ok(imgref::Img::new_stride(pixels, w, h, w.max(1)))
            }
        }
    }
}

/// Move a typed buffer into an [`ImgVec`] without copying.
///
/// Supports zero-area (`0×H` / `W×0`) buffers and strided buffers whose final row
/// omits trailing padding (`(height - 1) * stride + width` pixels).
///
/// The error is the buffer, completely unchanged, when the first pixel is past index 0,
/// the stride is not a whole number of pixels, or the allocation can't become a
/// `Vec<P>` (different alignment, or a capacity that isn't a whole number of pixels).
impl<P: Pixel> TryFrom<PixelBuffer<P>> for ImgVec<P> {
    type Error = PixelBuffer<P>;
    fn try_from(buffer: PixelBuffer<P>) -> Result<Self, PixelBuffer<P>> {
        const { assert!(core::mem::size_of::<P>() == P::DESCRIPTOR.bytes_per_pixel()) }
        let size = core::mem::size_of::<P>();
        let stride_bytes = buffer.stride();
        if buffer.offset != 0 || size == 0 || stride_bytes % size != 0 {
            return Err(buffer);
        }
        let (width, height) = (buffer.width as usize, buffer.height as usize);
        let raw_stride_px = stride_bytes / size;
        if height > 0 && raw_stride_px < width {
            return Err(buffer);
        }
        let needed_px = if width == 0 || height == 0 {
            0
        } else {
            let Some(n) = (height - 1)
                .checked_mul(raw_stride_px)
                .and_then(|n| n.checked_add(width))
            else {
                return Err(buffer);
            };
            n
        };
        let Some(needed_bytes) = needed_px.checked_mul(size) else {
            return Err(buffer);
        };
        let byte_len = buffer.data.as_bytes().len();
        if byte_len < needed_bytes
            || buffer.data.element_align() != core::mem::align_of::<P>()
            || buffer.data.capacity_bytes() % size != 0
        {
            return Err(buffer);
        }
        let stride_px = raw_stride_px
            .max(if height == 0 { width } else { 0 })
            .max(1);
        let PixelBuffer {
            mut data,
            offset,
            width: w,
            height: h,
            stride_samples,
            descriptor,
            color,
            exact,
            _pixel,
        } = buffer;
        if byte_len % size != 0 {
            data.truncate_bytes((byte_len / size) * size);
        }
        match data.try_cast_pixels::<P>() {
            Ok(pixels) => Ok(imgref::Img::new_stride(pixels, width, height, stride_px)),
            Err(data) => Err(PixelBuffer {
                data,
                offset,
                width: w,
                height: h,
                stride_samples,
                descriptor,
                color,
                exact,
                _pixel,
            }),
        }
    }
}

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
        let stride = self.stride();
        let pixel_size = core::mem::size_of::<P>();
        assert_eq!(
            stride % pixel_size,
            0,
            "ImgRef requires a stride in whole pixels"
        );
        let (width, height) = (self.width as usize, self.height as usize);
        let total_bytes = if height == 0 || width == 0 {
            0
        } else {
            (height - 1) * stride + width * pixel_size
        };
        let data =
            &self.data.as_bytes()[(self.offset as usize)..(self.offset as usize) + total_bytes];
        let pixels: &[P] = if data.is_empty() {
            &[]
        } else {
            bytemuck::cast_slice(data)
        };
        // imgref requires positive stride >= width even for zero-area views.
        let stride_px = (stride / pixel_size)
            .max(if height == 0 { width } else { 0 })
            .max(1);
        imgref::Img::new_stride(pixels, width, height, stride_px)
    }
}

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
        let stride = self.stride();
        let pixel_size = core::mem::size_of::<P>();
        assert_eq!(
            stride % pixel_size,
            0,
            "ImgRef requires a stride in whole pixels"
        );
        // Mutable access can change any sample: end current-sample evidence.
        self.exact = None;
        let (width, height) = (self.width as usize, self.height as usize);
        let total_bytes = if height == 0 || width == 0 {
            0
        } else {
            (height - 1) * stride + width * pixel_size
        };
        let offset = self.offset as usize;
        let data = &mut self.data.as_bytes_mut()[offset..offset + total_bytes];
        let pixels: &mut [P] = if data.is_empty() {
            &mut []
        } else {
            bytemuck::cast_slice_mut(data)
        };
        // imgref requires positive stride >= width even for zero-area views.
        let stride_px = (stride / pixel_size)
            .max(if height == 0 { width } else { 0 })
            .max(1);
        imgref::Img::new_stride(pixels, width, height, stride_px)
    }
}

/// Owned `ImgVec` interop for type-erased [`PixelBuffer`].
impl PixelBuffer {
    /// Construct a type-erased [`PixelBuffer`] from a typed [`ImgVec<P>`].
    ///
    /// Zero-copy for all built-in sample-aligned pixel types (`u8`, `u16`,
    /// [`F16Sample`], `f32`, etc.).
    pub fn from_imgvec_erased<P: Pixel>(img: ImgVec<P>) -> Self {
        PixelBuffer::<P>::from_imgvec(img).erase()
    }

    /// Fallible [`from_imgvec_erased`](Self::from_imgvec_erased).
    ///
    /// # Errors
    ///
    /// Same conditions as [`PixelBuffer::try_from_imgvec`].
    pub fn try_from_imgvec_erased<P: Pixel>(img: ImgVec<P>) -> Result<Self, At<BufferError>> {
        PixelBuffer::<P>::try_from_imgvec(img).map(PixelBuffer::erase)
    }

    /// Move this buffer into an [`ImgVec<P>`] strictly zero-copy.
    ///
    /// Returns `Err(self)`, completely unchanged, when the descriptor does not
    /// match `P`, `offset != 0`, `stride` is not a whole number of pixels, or
    /// the backing `Vec` cannot be cast in place to `Vec<P>`.
    pub fn try_into_imgvec_zero_copy<P: Pixel>(self) -> Result<ImgVec<P>, Self> {
        if !self.descriptor.typed_view_compatible(P::DESCRIPTOR)
            || core::mem::size_of::<P>() != P::DESCRIPTOR.bytes_per_pixel()
            || !rows_aligned_for::<P>(
                self.data.as_bytes()[(self.offset as usize)..].as_ptr(),
                self.width,
                self.height,
                self.stride(),
            )
        {
            return Err(self);
        }
        let typed = self.try_typed::<P>().expect("checked above");
        ImgVec::try_from(typed).map_err(PixelBuffer::erase)
    }

    /// Consume this buffer and return `Some(ImgVec<P>)` if layout-compatible with `P`,
    /// moving the allocation zero-copy whenever possible and compacting/copying once
    /// only when needed.
    ///
    /// # Panics
    ///
    /// Panics if the fallback allocation fails; use [`try_into_imgvec`](Self::try_into_imgvec)
    /// for fallible recovery.
    #[track_caller]
    pub fn into_imgvec<P: Pixel>(self) -> Option<ImgVec<P>> {
        self.try_into_imgvec()
            .expect("pixel buffer to ImgVec allocation failed")
    }

    /// Fallible [`into_imgvec`](Self::into_imgvec).
    ///
    /// Returns `Ok(None)` when the descriptor is not layout-compatible with `P`,
    /// `Ok(Some(img))` on zero-copy move or single-copy compaction, and
    /// `Err(BufferError::AllocationFailed)` if the fallback copy reserve fails.
    pub fn try_into_imgvec<P: Pixel>(self) -> Result<Option<ImgVec<P>>, At<BufferError>> {
        if !self.descriptor.typed_view_compatible(P::DESCRIPTOR)
            || core::mem::size_of::<P>() != self.descriptor.bytes_per_pixel()
            || core::mem::size_of::<P>() == 0
        {
            return Ok(None);
        }
        match self.try_into_imgvec_zero_copy::<P>() {
            Ok(img) => Ok(Some(img)),
            Err(buffer) => {
                let (w, h) = (buffer.width as usize, buffer.height as usize);
                Ok(buffer
                    .try_into_contiguous_pixels::<P>()?
                    .map(|pixels| imgref::Img::new_stride(pixels, w, h, w.max(1))))
            }
        }
    }
}

/// Imgref interop for type-erased PixelBuffer.
impl<S: ByteStorage> PixelBuffer<(), S> {
    /// Try to borrow the buffer as a typed [`ImgRef`].
    ///
    /// Returns `None` for incompatible pixels, insufficient alignment or a stride
    /// that is not a whole number of pixels. For SIMD-padded 3-channel buffers whose
    /// byte stride is sample-aligned rather than pixel-aligned (such as 64-byte
    /// [`RowAlign::Cache64`](super::RowAlign::Cache64) `RGB8`), use
    /// [`try_as_sample_imgref`](Self::try_as_sample_imgref).
    #[track_caller]
    pub fn try_as_imgref<P: Pixel>(&self) -> Option<ImgRef<'_, P>> {
        let stride = self.stride();
        let pixel_size = check_imgref::<P>(self.descriptor, stride).ok()?;
        let (width, height) = (self.width as usize, self.height as usize);
        let total_bytes = if height == 0 || width == 0 {
            0
        } else {
            (height - 1) * stride + width * pixel_size
        };
        let data =
            &self.data.as_bytes()[(self.offset as usize)..(self.offset as usize) + total_bytes];
        let pixels: &[P] = if data.is_empty() {
            &[]
        } else {
            bytemuck::try_cast_slice(data).ok()?
        };
        // imgref refuses a stride of 0 or < width, even for an empty image.
        let stride_px = (stride / pixel_size)
            .max(if height == 0 { width } else { 0 })
            .max(1);
        Some(imgref::Img::new_stride(pixels, width, height, stride_px))
    }

    /// Borrow the buffer zero-copy as a sample-strided `ImgRef<'_, T>` (`width =
    /// pixel_width * channels`, `stride = byte_stride / size_of::<T>()`).
    ///
    /// Unlike [`try_as_imgref`](Self::try_as_imgref), this succeeds for **any** valid
    /// sample-aligned row stride — including 64-byte SIMD-padded 3-channel `RGB8`,
    /// `RGB16`, `RGBF16`, and `RGBF32` buffers (`RowAlign::Cache64`) whose byte stride
    /// is not a multiple of `3 * size_of::<T>()`.
    pub fn try_as_sample_imgref<T: StorageSample>(&self) -> Option<ImgRef<'_, T>> {
        self.as_slice().try_as_sample_imgref::<T>()
    }
}

impl<S: ByteStorageMut> PixelBuffer<(), S> {
    /// Try to borrow the buffer as a typed mutable [`ImgRefMut`](imgref::ImgRefMut).
    ///
    /// On refusal (`None`), exact-depth evidence is unchanged.
    ///
    /// Refusal conditions match [`try_as_imgref`](Self::try_as_imgref).
    #[track_caller]
    pub fn try_as_imgref_mut<P: Pixel>(&mut self) -> Option<imgref::ImgRefMut<'_, P>> {
        let stride = self.stride();
        let pixel_size = check_imgref::<P>(self.descriptor, stride).ok()?;
        let (width, height) = (self.width as usize, self.height as usize);
        let total_bytes = if height == 0 || width == 0 {
            0
        } else {
            (height - 1) * stride + width * pixel_size
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
        // imgref refuses a stride of 0 or < width, even for an empty image.
        let stride_px = (stride / pixel_size)
            .max(if height == 0 { width } else { 0 })
            .max(1);
        Some(imgref::Img::new_stride(pixels, width, height, stride_px))
    }

    /// Borrow the buffer zero-copy as a mutable sample-strided `ImgRefMut<'_, T>`
    /// (`width = pixel_width * channels`, `stride = byte_stride / size_of::<T>()`).
    ///
    /// On refusal (`None`), exact-depth evidence is unchanged.
    pub fn try_as_sample_imgref_mut<T: StorageSample>(
        &mut self,
    ) -> Option<imgref::ImgRefMut<'_, T>> {
        if self.descriptor.channel_type() != T::CHANNEL_TYPE {
            return None;
        }
        let sample_size = core::mem::size_of::<T>();
        let stride = self.stride();
        if sample_size == 0 || stride % sample_size != 0 {
            return None;
        }
        let (width, height) = (self.width as usize, self.height as usize);
        let sample_width = width.checked_mul(self.descriptor.channels())?;
        let total_bytes = if height == 0 || sample_width == 0 {
            0
        } else {
            (height - 1) * stride + sample_width * sample_size
        };
        let offset = self.offset as usize;
        let bytes = self.data.as_bytes_mut();
        let data = &mut bytes[offset..offset + total_bytes];
        let samples: &mut [T] = if data.is_empty() {
            &mut []
        } else {
            bytemuck::try_cast_slice_mut(data).ok()?
        };
        self.exact = None;
        let sample_stride = (stride / sample_size)
            .max(if height == 0 { sample_width } else { 0 })
            .max(1);
        Some(imgref::Img::new_stride(
            samples,
            sample_width,
            height,
            sample_stride,
        ))
    }
}

/// Symmetric `imgref` views and constructors on [`PixelSlice`] and [`PixelSliceMut`].
impl<'a, P: Pixel> PixelSlice<'a, P> {
    /// Borrow this typed slice zero-copy as an [`ImgRef<'a, P>`].
    ///
    /// # Panics
    ///
    /// Panics if `stride()` is not a whole number of pixels; use
    /// [`try_as_imgref`](Self::try_as_imgref) when the slice may have a byte-only stride.
    #[track_caller]
    pub fn as_imgref(&self) -> ImgRef<'a, P> {
        let pixel_size = core::mem::size_of::<P>();
        assert_eq!(
            self.stride % pixel_size,
            0,
            "ImgRef requires a stride in whole pixels"
        );
        self.try_as_imgref()
            .expect("typed PixelSlice rows are aligned for P")
    }

    /// Try to borrow this typed slice zero-copy as an [`ImgRef<'a, P>`], returning
    /// `None` if `stride()` is not a whole number of pixels.
    pub fn try_as_imgref(&self) -> Option<ImgRef<'a, P>> {
        let pixel_size = core::mem::size_of::<P>();
        if pixel_size == 0 || self.stride % pixel_size != 0 {
            return None;
        }
        let (width, height) = (self.width as usize, self.rows as usize);
        let total_bytes = if height == 0 || width == 0 {
            0
        } else {
            (height - 1) * self.stride + width * pixel_size
        };
        let data = &self.data[..total_bytes];
        let pixels: &'a [P] = if data.is_empty() {
            &[]
        } else {
            bytemuck::try_cast_slice(data).ok()?
        };
        let stride_px = (self.stride / pixel_size)
            .max(if height == 0 { width } else { 0 })
            .max(1);
        Some(imgref::Img::new_stride(pixels, width, height, stride_px))
    }
}

impl<'a> PixelSlice<'a> {
    /// Try to borrow this type-erased slice zero-copy as a typed [`ImgRef<'a, P>`].
    ///
    /// Returns `None` if `P` is layout-incompatible, unaligned, or `stride()` is
    /// not a multiple of `size_of::<P>()`.
    #[track_caller]
    pub fn try_as_imgref<P: Pixel>(&self) -> Option<ImgRef<'a, P>> {
        let pixel_size = check_imgref::<P>(self.descriptor, self.stride).ok()?;
        let (width, height) = (self.width as usize, self.rows as usize);
        let total_bytes = if height == 0 || width == 0 {
            0
        } else {
            (height - 1) * self.stride + width * pixel_size
        };
        let data = &self.data[..total_bytes];
        let pixels: &'a [P] = if data.is_empty() {
            &[]
        } else {
            bytemuck::try_cast_slice(data).ok()?
        };
        let stride_px = (self.stride / pixel_size)
            .max(if height == 0 { width } else { 0 })
            .max(1);
        Some(imgref::Img::new_stride(pixels, width, height, stride_px))
    }

    /// Borrow this slice zero-copy as a sample-strided `ImgRef<'a, T>` (`width =
    /// pixel_width * channels`, `stride = byte_stride / size_of::<T>()`).
    ///
    /// Succeeds for any sample-aligned stride, including 64-byte SIMD-padded 3-channel
    /// `RGB8`, `RGB16`, `RGBF16`, and `RGBF32` slices.
    pub fn try_as_sample_imgref<T: StorageSample>(&self) -> Option<ImgRef<'a, T>> {
        if self.descriptor.channel_type() != T::CHANNEL_TYPE {
            return None;
        }
        let sample_size = core::mem::size_of::<T>();
        if sample_size == 0 || self.stride % sample_size != 0 {
            return None;
        }
        let (width, height) = (self.width as usize, self.rows as usize);
        let sample_width = width.checked_mul(self.descriptor.channels())?;
        let total_bytes = if height == 0 || sample_width == 0 {
            0
        } else {
            (height - 1) * self.stride + sample_width * sample_size
        };
        let data = &self.data[..total_bytes];
        let samples: &'a [T] = if data.is_empty() {
            &[]
        } else {
            bytemuck::try_cast_slice(data).ok()?
        };
        let sample_stride = (self.stride / sample_size)
            .max(if height == 0 { sample_width } else { 0 })
            .max(1);
        Some(imgref::Img::new_stride(
            samples,
            sample_width,
            height,
            sample_stride,
        ))
    }

    /// Construct a type-erased [`PixelSlice`] zero-copy from a typed [`ImgRef<'a, P>`],
    /// using the same conventional descriptor as `PixelSlice::<P>::from(img)`.
    pub fn from_imgref<P: Pixel>(img: ImgRef<'a, P>) -> Self
    where
        PixelSlice<'a, P>: From<ImgRef<'a, P>>,
    {
        PixelSlice::<P>::from(img).erase()
    }

    /// Construct a type-erased [`PixelSlice`] zero-copy from any pixel-strided
    /// `ImgRef<'a, T>` (such as `ImgRef<'_, [u8; 3]>`, `ImgRef<'_, [u16; 4]>`, or a
    /// bare `ImgRef<'_, u8>` for grayscale) paired with an explicit `descriptor`.
    ///
    /// Requires `size_of::<T>() == descriptor.bytes_per_pixel()`. For sample-strided
    /// `ImgRef`s where `width` counts channels (`pixel_width * channels`), use
    /// [`try_from_sample_imgref_with`](Self::try_from_sample_imgref_with).
    #[track_caller]
    pub fn try_from_imgref_with<T: bytemuck::Pod>(
        img: ImgRef<'a, T>,
        descriptor: PixelDescriptor,
    ) -> Result<Self, At<BufferError>> {
        descriptor.validate().map_err(|e| whereat::at!(e))?;
        if core::mem::size_of::<T>() != descriptor.bytes_per_pixel() {
            return Err(whereat::at!(BufferError::IncompatibleDescriptor));
        }
        let width =
            u32::try_from(img.width()).map_err(|_| whereat::at!(BufferError::InvalidDimensions))?;
        let rows = u32::try_from(img.height())
            .map_err(|_| whereat::at!(BufferError::InvalidDimensions))?;
        let stride_bytes = img
            .stride()
            .checked_mul(core::mem::size_of::<T>())
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let bytes: &'a [u8] = bytemuck::cast_slice(img.buf());
        Self::new(bytes, width, rows, stride_bytes, descriptor)
    }

    /// Construct a type-erased [`PixelSlice`] zero-copy from a sample-strided
    /// `ImgRef<'a, T>` whose `width` and `stride` are measured in `T` samples
    /// (`img.width() == pixel_width * descriptor.channels()`).
    #[track_caller]
    pub fn try_from_sample_imgref_with<T: StorageSample>(
        img: ImgRef<'a, T>,
        descriptor: PixelDescriptor,
    ) -> Result<Self, At<BufferError>> {
        descriptor.validate().map_err(|e| whereat::at!(e))?;
        if descriptor.channel_type() != T::CHANNEL_TYPE {
            return Err(whereat::at!(BufferError::IncompatibleDescriptor));
        }
        let channels = descriptor.channels();
        if channels == 0 || img.width() % channels != 0 {
            return Err(whereat::at!(BufferError::InvalidDimensions));
        }
        let width = u32::try_from(img.width() / channels)
            .map_err(|_| whereat::at!(BufferError::InvalidDimensions))?;
        let rows = u32::try_from(img.height())
            .map_err(|_| whereat::at!(BufferError::InvalidDimensions))?;
        let stride_bytes = img
            .stride()
            .checked_mul(core::mem::size_of::<T>())
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let bytes: &'a [u8] = bytemuck::cast_slice(img.buf());
        Self::new(bytes, width, rows, stride_bytes, descriptor)
    }
}

impl<'a, P: Pixel> PixelSliceMut<'a, P> {
    /// Borrow this typed mutable slice zero-copy as an immutable [`ImgRef<'_, P>`].
    ///
    /// # Panics
    ///
    /// Panics if `stride()` is not a whole number of pixels.
    #[track_caller]
    pub fn as_imgref(&self) -> ImgRef<'_, P> {
        self.as_pixel_slice().as_imgref()
    }

    /// Try to borrow this typed mutable slice zero-copy as an immutable [`ImgRef<'_, P>`].
    pub fn try_as_imgref(&self) -> Option<ImgRef<'_, P>> {
        self.as_pixel_slice().try_as_imgref()
    }

    /// Reborrow this typed mutable slice zero-copy as an [`ImgRefMut<'_, P>`](imgref::ImgRefMut).
    ///
    /// # Panics
    ///
    /// Panics if `stride()` is not a whole number of pixels; use
    /// [`try_as_imgref_mut`](Self::try_as_imgref_mut) when the slice may have a byte-only stride.
    #[track_caller]
    pub fn as_imgref_mut(&mut self) -> imgref::ImgRefMut<'_, P> {
        let pixel_size = core::mem::size_of::<P>();
        assert_eq!(
            self.stride % pixel_size,
            0,
            "ImgRef requires a stride in whole pixels"
        );
        self.try_as_imgref_mut()
            .expect("typed PixelSliceMut rows are aligned for P")
    }

    /// Try to reborrow this typed mutable slice zero-copy as an
    /// [`ImgRefMut<'_, P>`](imgref::ImgRefMut), returning `None` if `stride()` is not a
    /// whole number of pixels.
    pub fn try_as_imgref_mut(&mut self) -> Option<imgref::ImgRefMut<'_, P>> {
        let pixel_size = core::mem::size_of::<P>();
        if pixel_size == 0 || self.stride % pixel_size != 0 {
            return None;
        }
        let (width, height) = (self.width as usize, self.rows as usize);
        let total_bytes = if height == 0 || width == 0 {
            0
        } else {
            (height - 1) * self.stride + width * pixel_size
        };
        let data = &mut self.data[..total_bytes];
        let pixels: &mut [P] = if data.is_empty() {
            &mut []
        } else {
            bytemuck::try_cast_slice_mut(data).ok()?
        };
        let stride_px = (self.stride / pixel_size)
            .max(if height == 0 { width } else { 0 })
            .max(1);
        Some(imgref::Img::new_stride(pixels, width, height, stride_px))
    }

    /// Consume this typed mutable slice and return an [`ImgRefMut<'a, P>`](imgref::ImgRefMut)
    /// with the underlying `'a` borrow lifetime.
    ///
    /// # Panics
    ///
    /// Panics if `stride()` is not a whole number of pixels; use
    /// [`try_into_imgref_mut`](Self::try_into_imgref_mut) to recover `self` instead.
    #[track_caller]
    pub fn into_imgref_mut(self) -> imgref::ImgRefMut<'a, P> {
        let pixel_size = core::mem::size_of::<P>();
        assert_eq!(
            self.stride % pixel_size,
            0,
            "ImgRef requires a stride in whole pixels"
        );
        match self.try_into_imgref_mut() {
            Ok(img) => img,
            Err(_) => panic!("typed PixelSliceMut rows are not aligned for P"),
        }
    }

    /// Fallibly consume this typed mutable slice into an
    /// [`ImgRefMut<'a, P>`](imgref::ImgRefMut) with lifetime `'a`, returning `Err(self)`
    /// unchanged if `stride()` is not a whole number of pixels.
    pub fn try_into_imgref_mut(self) -> Result<imgref::ImgRefMut<'a, P>, Self> {
        let pixel_size = core::mem::size_of::<P>();
        if pixel_size == 0 || self.stride % pixel_size != 0 {
            return Err(self);
        }
        let (width, height) = (self.width as usize, self.rows as usize);
        let total_bytes = if height == 0 || width == 0 {
            0
        } else {
            (height - 1) * self.stride + width * pixel_size
        };
        if total_bytes != 0 && (self.data.as_ptr() as usize) % core::mem::align_of::<P>() != 0 {
            return Err(self);
        }
        let stride_px = (self.stride / pixel_size)
            .max(if height == 0 { width } else { 0 })
            .max(1);
        let data = &mut self.data[..total_bytes];
        let pixels: &'a mut [P] = if data.is_empty() {
            &mut []
        } else {
            bytemuck::cast_slice_mut(data)
        };
        Ok(imgref::Img::new_stride(pixels, width, height, stride_px))
    }
}

impl<'a> PixelSliceMut<'a> {
    /// Try to borrow this type-erased mutable slice zero-copy as an immutable [`ImgRef<'_, P>`].
    #[track_caller]
    pub fn try_as_imgref<P: Pixel>(&self) -> Option<ImgRef<'_, P>> {
        self.as_pixel_slice().try_as_imgref::<P>()
    }

    /// Try to reborrow this type-erased mutable slice zero-copy as a typed
    /// [`ImgRefMut<'_, P>`](imgref::ImgRefMut).
    #[track_caller]
    pub fn try_as_imgref_mut<P: Pixel>(&mut self) -> Option<imgref::ImgRefMut<'_, P>> {
        let pixel_size = check_imgref::<P>(self.descriptor, self.stride).ok()?;
        let (width, height) = (self.width as usize, self.rows as usize);
        let total_bytes = if height == 0 || width == 0 {
            0
        } else {
            (height - 1) * self.stride + width * pixel_size
        };
        let data = &mut self.data[..total_bytes];
        let pixels: &mut [P] = if data.is_empty() {
            &mut []
        } else {
            bytemuck::try_cast_slice_mut(data).ok()?
        };
        let stride_px = (self.stride / pixel_size)
            .max(if height == 0 { width } else { 0 })
            .max(1);
        Some(imgref::Img::new_stride(pixels, width, height, stride_px))
    }

    /// Borrow this slice zero-copy as a sample-strided `ImgRef<'_, T>`.
    pub fn try_as_sample_imgref<T: StorageSample>(&self) -> Option<ImgRef<'_, T>> {
        self.as_pixel_slice().try_as_sample_imgref::<T>()
    }

    /// Reborrow this slice zero-copy as a mutable sample-strided
    /// `ImgRefMut<'_, T>` (`width = pixel_width * channels`, `stride = byte_stride / size_of::<T>()`).
    pub fn try_as_sample_imgref_mut<T: StorageSample>(
        &mut self,
    ) -> Option<imgref::ImgRefMut<'_, T>> {
        if self.descriptor.channel_type() != T::CHANNEL_TYPE {
            return None;
        }
        let sample_size = core::mem::size_of::<T>();
        if sample_size == 0 || self.stride % sample_size != 0 {
            return None;
        }
        let (width, height) = (self.width as usize, self.rows as usize);
        let sample_width = width.checked_mul(self.descriptor.channels())?;
        let total_bytes = if height == 0 || sample_width == 0 {
            0
        } else {
            (height - 1) * self.stride + sample_width * sample_size
        };
        let data = &mut self.data[..total_bytes];
        let samples: &mut [T] = if data.is_empty() {
            &mut []
        } else {
            bytemuck::try_cast_slice_mut(data).ok()?
        };
        let sample_stride = (self.stride / sample_size)
            .max(if height == 0 { sample_width } else { 0 })
            .max(1);
        Some(imgref::Img::new_stride(
            samples,
            sample_width,
            height,
            sample_stride,
        ))
    }

    /// Construct a type-erased [`PixelSliceMut`] zero-copy from a typed
    /// [`ImgRefMut<'a, P>`](imgref::ImgRefMut), using the same conventional descriptor as
    /// `PixelSliceMut::<P>::from(img)`.
    pub fn from_imgref_mut<P: Pixel>(img: imgref::ImgRefMut<'a, P>) -> Self
    where
        PixelSliceMut<'a, P>: From<imgref::ImgRefMut<'a, P>>,
    {
        PixelSliceMut::<P>::from(img).erase()
    }

    /// Construct a type-erased [`PixelSliceMut`] zero-copy from any pixel-strided
    /// `ImgRefMut<'a, T>` paired with an explicit `descriptor`.
    #[track_caller]
    pub fn try_from_imgref_mut_with<T: bytemuck::Pod>(
        img: imgref::ImgRefMut<'a, T>,
        descriptor: PixelDescriptor,
    ) -> Result<Self, At<BufferError>> {
        descriptor.validate().map_err(|e| whereat::at!(e))?;
        if core::mem::size_of::<T>() != descriptor.bytes_per_pixel() {
            return Err(whereat::at!(BufferError::IncompatibleDescriptor));
        }
        let width =
            u32::try_from(img.width()).map_err(|_| whereat::at!(BufferError::InvalidDimensions))?;
        let rows = u32::try_from(img.height())
            .map_err(|_| whereat::at!(BufferError::InvalidDimensions))?;
        let stride_bytes = img
            .stride()
            .checked_mul(core::mem::size_of::<T>())
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let bytes: &'a mut [u8] = bytemuck::cast_slice_mut(img.into_buf());
        Self::new(bytes, width, rows, stride_bytes, descriptor)
    }

    /// Construct a type-erased [`PixelSliceMut`] zero-copy from a sample-strided
    /// `ImgRefMut<'a, T>` (`img.width() == pixel_width * descriptor.channels()`).
    #[track_caller]
    pub fn try_from_sample_imgref_mut_with<T: StorageSample>(
        img: imgref::ImgRefMut<'a, T>,
        descriptor: PixelDescriptor,
    ) -> Result<Self, At<BufferError>> {
        descriptor.validate().map_err(|e| whereat::at!(e))?;
        if descriptor.channel_type() != T::CHANNEL_TYPE {
            return Err(whereat::at!(BufferError::IncompatibleDescriptor));
        }
        let channels = descriptor.channels();
        if channels == 0 || img.width() % channels != 0 {
            return Err(whereat::at!(BufferError::InvalidDimensions));
        }
        let width = u32::try_from(img.width() / channels)
            .map_err(|_| whereat::at!(BufferError::InvalidDimensions))?;
        let rows = u32::try_from(img.height())
            .map_err(|_| whereat::at!(BufferError::InvalidDimensions))?;
        let stride_bytes = img
            .stride()
            .checked_mul(core::mem::size_of::<T>())
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        let bytes: &'a mut [u8] = bytemuck::cast_slice_mut(img.into_buf());
        Self::new(bytes, width, rows, stride_bytes, descriptor)
    }
}

/// What [`PixelBuffer::try_as_imgref`] checks before it reads any bytes. Returns the pixel size.
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

// ---------------------------------------------------------------------------
// ImgRef -> PixelSlice (zero-copy From impls) -- imgref feature only
// ---------------------------------------------------------------------------

#[track_caller]
fn slice_from_imgref_raw<'a>(
    bytes: &'a [u8],
    width_px: usize,
    height_px: usize,
    stride_px: usize,
    pixel_size: usize,
    descriptor: PixelDescriptor,
) -> PixelSlice<'a> {
    let width = u32::try_from(width_px).expect("ImgRef width exceeds u32");
    let rows = u32::try_from(height_px).expect("ImgRef height exceeds u32");
    let byte_stride = stride_px
        .checked_mul(pixel_size)
        .expect("ImgRef byte stride overflow");
    validate_slice(
        bytes.len(),
        bytes.as_ptr(),
        width,
        rows,
        byte_stride,
        &descriptor,
    )
    .expect("ImgRef buffer is too small for its dimensions and stride");
    PixelSlice {
        data: bytes,
        width,
        rows,
        stride: byte_stride,
        descriptor,
        color: None,
        exact: None,
        _pixel: PhantomData,
    }
}

macro_rules! impl_from_imgref {
    ($pixel:ty, $descriptor:expr) => {
        impl<'a> From<ImgRef<'a, $pixel>> for PixelSlice<'a, $pixel> {
            #[inline]
            #[track_caller]
            fn from(img: ImgRef<'a, $pixel>) -> Self {
                let s = slice_from_imgref_raw(
                    bytemuck::cast_slice(img.buf()),
                    img.width(),
                    img.height(),
                    img.stride(),
                    core::mem::size_of::<$pixel>(),
                    $descriptor,
                );
                PixelSlice {
                    data: s.data,
                    width: s.width,
                    rows: s.rows,
                    stride: s.stride,
                    descriptor: s.descriptor,
                    color: s.color,
                    exact: s.exact,
                    _pixel: PhantomData,
                }
            }
        }
    };
}

// u8 types are conventionally sRGB, f16/f32 float types are conventionally linear.
// u16 types have no standard convention so use transfer-agnostic descriptors.
impl_from_imgref!(Rgb<u8>, PixelDescriptor::RGB8_SRGB);
impl_from_imgref!(Rgba<u8>, PixelDescriptor::RGBA8_SRGB);
impl_from_imgref!(Rgb<u16>, PixelDescriptor::RGB16);
impl_from_imgref!(Rgba<u16>, PixelDescriptor::RGBA16);
impl_from_imgref!(Rgb<F16Sample>, PixelDescriptor::RGBF16_LINEAR);
impl_from_imgref!(Rgba<F16Sample>, PixelDescriptor::RGBAF16_LINEAR);
impl_from_imgref!(Rgb<f32>, PixelDescriptor::RGBF32_LINEAR);
impl_from_imgref!(Rgba<f32>, PixelDescriptor::RGBAF32_LINEAR);
impl_from_imgref!(Gray<u8>, PixelDescriptor::GRAY8_SRGB);
impl_from_imgref!(Gray<u16>, PixelDescriptor::GRAY16);
impl_from_imgref!(
    Gray<F16Sample>,
    PixelDescriptor::new(
        ChannelType::F16,
        ChannelLayout::Gray,
        None,
        TransferFunction::Linear
    )
);
impl_from_imgref!(Gray<f32>, PixelDescriptor::GRAYF32_LINEAR);
impl_from_imgref!(GrayAlpha8, PixelDescriptor::GRAYA8_SRGB);
impl_from_imgref!(GrayAlpha16, PixelDescriptor::GRAYA16);
impl_from_imgref!(
    GrayAlphaF16,
    PixelDescriptor::new(
        ChannelType::F16,
        ChannelLayout::GrayAlpha,
        Some(AlphaMode::Straight),
        TransferFunction::Linear
    )
);
impl_from_imgref!(GrayAlphaF32, PixelDescriptor::GRAYAF32_LINEAR);
impl_from_imgref!(
    Bgr<u8>,
    PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Bgr,
        None,
        TransferFunction::Srgb
    )
);
impl_from_imgref!(Bgr<u16>, <Bgr<u16> as Pixel>::DESCRIPTOR);
impl_from_imgref!(
    Bgr<F16Sample>,
    <Bgr<F16Sample> as Pixel>::DESCRIPTOR.with_transfer(TransferFunction::Linear)
);
impl_from_imgref!(
    Bgr<f32>,
    <Bgr<f32> as Pixel>::DESCRIPTOR.with_transfer(TransferFunction::Linear)
);
impl_from_imgref!(Bgra<u8>, PixelDescriptor::BGRA8_SRGB);
impl_from_imgref!(Bgra<u16>, <Bgra<u16> as Pixel>::DESCRIPTOR);
impl_from_imgref!(
    Bgra<F16Sample>,
    <Bgra<F16Sample> as Pixel>::DESCRIPTOR.with_transfer(TransferFunction::Linear)
);
impl_from_imgref!(
    Bgra<f32>,
    <Bgra<f32> as Pixel>::DESCRIPTOR.with_transfer(TransferFunction::Linear)
);
impl_from_imgref!(Rgbx, PixelDescriptor::RGBX8_SRGB);
impl_from_imgref!(Bgrx, PixelDescriptor::BGRX8_SRGB);

// ---------------------------------------------------------------------------
// ImgRefMut -> PixelSliceMut (zero-copy From impls) -- imgref feature only
// ---------------------------------------------------------------------------

#[track_caller]
fn slice_mut_from_imgref_raw<'a>(
    bytes: &'a mut [u8],
    width_px: usize,
    height_px: usize,
    stride_px: usize,
    pixel_size: usize,
    descriptor: PixelDescriptor,
) -> PixelSliceMut<'a> {
    let width = u32::try_from(width_px).expect("ImgRefMut width exceeds u32");
    let rows = u32::try_from(height_px).expect("ImgRefMut height exceeds u32");
    let byte_stride = stride_px
        .checked_mul(pixel_size)
        .expect("ImgRefMut byte stride overflow");
    validate_slice(
        bytes.len(),
        bytes.as_ptr(),
        width,
        rows,
        byte_stride,
        &descriptor,
    )
    .expect("ImgRefMut buffer is too small for its dimensions and stride");
    PixelSliceMut {
        data: bytes,
        width,
        rows,
        stride: byte_stride,
        descriptor,
        color: None,
        _pixel: PhantomData,
    }
}

macro_rules! impl_from_imgref_mut {
    ($pixel:ty, $descriptor:expr) => {
        impl<'a> From<imgref::ImgRefMut<'a, $pixel>> for PixelSliceMut<'a, $pixel> {
            #[inline]
            #[track_caller]
            fn from(img: imgref::ImgRefMut<'a, $pixel>) -> Self {
                let (w, h, stride) = (img.width(), img.height(), img.stride());
                let s = slice_mut_from_imgref_raw(
                    bytemuck::cast_slice_mut(img.into_buf()),
                    w,
                    h,
                    stride,
                    core::mem::size_of::<$pixel>(),
                    $descriptor,
                );
                PixelSliceMut {
                    data: s.data,
                    width: s.width,
                    rows: s.rows,
                    stride: s.stride,
                    descriptor: s.descriptor,
                    color: s.color,
                    _pixel: PhantomData,
                }
            }
        }
    };
}

impl_from_imgref_mut!(Rgb<u8>, PixelDescriptor::RGB8_SRGB);
impl_from_imgref_mut!(Rgba<u8>, PixelDescriptor::RGBA8_SRGB);
impl_from_imgref_mut!(Rgb<u16>, PixelDescriptor::RGB16);
impl_from_imgref_mut!(Rgba<u16>, PixelDescriptor::RGBA16);
impl_from_imgref_mut!(Rgb<F16Sample>, PixelDescriptor::RGBF16_LINEAR);
impl_from_imgref_mut!(Rgba<F16Sample>, PixelDescriptor::RGBAF16_LINEAR);
impl_from_imgref_mut!(Rgb<f32>, PixelDescriptor::RGBF32_LINEAR);
impl_from_imgref_mut!(Rgba<f32>, PixelDescriptor::RGBAF32_LINEAR);
impl_from_imgref_mut!(Gray<u8>, PixelDescriptor::GRAY8_SRGB);
impl_from_imgref_mut!(Gray<u16>, PixelDescriptor::GRAY16);
impl_from_imgref_mut!(
    Gray<F16Sample>,
    PixelDescriptor::new(
        ChannelType::F16,
        ChannelLayout::Gray,
        None,
        TransferFunction::Linear
    )
);
impl_from_imgref_mut!(Gray<f32>, PixelDescriptor::GRAYF32_LINEAR);
impl_from_imgref_mut!(GrayAlpha8, PixelDescriptor::GRAYA8_SRGB);
impl_from_imgref_mut!(GrayAlpha16, PixelDescriptor::GRAYA16);
impl_from_imgref_mut!(
    GrayAlphaF16,
    PixelDescriptor::new(
        ChannelType::F16,
        ChannelLayout::GrayAlpha,
        Some(AlphaMode::Straight),
        TransferFunction::Linear
    )
);
impl_from_imgref_mut!(GrayAlphaF32, PixelDescriptor::GRAYAF32_LINEAR);
impl_from_imgref_mut!(
    Bgr<u8>,
    PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Bgr,
        None,
        TransferFunction::Srgb
    )
);
impl_from_imgref_mut!(Bgr<u16>, <Bgr<u16> as Pixel>::DESCRIPTOR);
impl_from_imgref_mut!(
    Bgr<F16Sample>,
    <Bgr<F16Sample> as Pixel>::DESCRIPTOR.with_transfer(TransferFunction::Linear)
);
impl_from_imgref_mut!(
    Bgr<f32>,
    <Bgr<f32> as Pixel>::DESCRIPTOR.with_transfer(TransferFunction::Linear)
);
impl_from_imgref_mut!(Bgra<u8>, PixelDescriptor::BGRA8_SRGB);
impl_from_imgref_mut!(Bgra<u16>, <Bgra<u16> as Pixel>::DESCRIPTOR);
impl_from_imgref_mut!(
    Bgra<F16Sample>,
    <Bgra<F16Sample> as Pixel>::DESCRIPTOR.with_transfer(TransferFunction::Linear)
);
impl_from_imgref_mut!(
    Bgra<f32>,
    <Bgra<f32> as Pixel>::DESCRIPTOR.with_transfer(TransferFunction::Linear)
);
impl_from_imgref_mut!(Rgbx, PixelDescriptor::RGBX8_SRGB);
impl_from_imgref_mut!(Bgrx, PixelDescriptor::BGRX8_SRGB);
