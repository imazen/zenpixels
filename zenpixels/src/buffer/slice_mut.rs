use alloc::sync::Arc;
use core::fmt;
use core::marker::PhantomData;

#[cfg(feature = "rgb")]
use rgb::alt::BGRA;
#[cfg(feature = "rgb")]
use rgb::{Rgb, Rgba};

use crate::color::ColorContext;
use crate::descriptor::{
    AlphaMode, ColorPrimaries, PixelDescriptor, SignalRange, TransferFunction,
};

use super::{Bgrx, BufferError, Pixel, Rgbx, validate_slice};

/// Mutable borrowed view of pixel data.
///
/// Same semantics as [`PixelSlice`](super::PixelSlice) but allows writing to rows.
/// The type parameter `P` tracks pixel format at compile time.
#[non_exhaustive]
pub struct PixelSliceMut<'a, P = ()> {
    pub(super) data: &'a mut [u8],
    pub(super) width: u32,
    pub(super) rows: u32,
    pub(super) stride: usize,
    pub(super) descriptor: PixelDescriptor,
    pub(super) color: Option<Arc<ColorContext>>,
    pub(super) _pixel: PhantomData<P>,
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
    /// See [`PixelSlice::with_descriptor()`](super::PixelSlice::with_descriptor) for details.
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
    /// See [`PixelSlice::reinterpret()`](super::PixelSlice::reinterpret) for details.
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
