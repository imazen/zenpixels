//! Pre-computed row converter.
//!
//! [`RowConverter`] wraps a [`ConvertPlan`] and provides a streaming-friendly
//! API for converting pixel rows without per-call allocation.

use zencodec_types::{PixelDescriptor, PixelSlice, PixelSliceMut};

use crate::convert::{convert_row, ConvertPlan};
use crate::error::ConvertError;

/// Pre-computed pixel format converter.
///
/// Create once, then call [`convert_row`](Self::convert_row) or
/// [`convert_slice`](Self::convert_slice) for each row/strip.
///
/// # Example
///
/// ```rust,ignore
/// use zenpixels::{RowConverter, PixelDescriptor};
///
/// let conv = RowConverter::new(
///     PixelDescriptor::RGB8_SRGB,
///     PixelDescriptor::RGBA8_SRGB,
/// )?;
///
/// for y in 0..height {
///     conv.convert_row(&src_row, &mut dst_row, width);
/// }
/// ```
pub struct RowConverter {
    plan: ConvertPlan,
}

impl RowConverter {
    /// Create a converter from `from` to `to`.
    ///
    /// Returns `Err` if no conversion path exists between the formats.
    pub fn new(from: PixelDescriptor, to: PixelDescriptor) -> Result<Self, ConvertError> {
        let plan = ConvertPlan::new(from, to)?;
        Ok(Self { plan })
    }

    /// Convert one row of `width` pixels.
    ///
    /// `src` must contain at least `width * from.bytes_per_pixel()` bytes.
    /// `dst` must contain at least `width * to.bytes_per_pixel()` bytes.
    #[inline]
    pub fn convert_row(&self, src: &[u8], dst: &mut [u8], width: u32) {
        convert_row(&self.plan, src, dst, width);
    }

    /// Convert a full [`PixelSlice`] strip into a [`PixelSliceMut`].
    ///
    /// The slices must have the same width and row count.
    pub fn convert_slice(
        &self,
        src: &PixelSlice<'_>,
        dst: &mut PixelSliceMut<'_>,
    ) -> Result<(), ConvertError> {
        if src.width() != dst.width() || src.rows() != dst.rows() {
            return Err(ConvertError::BufferSize {
                expected: dst.width() as usize * dst.rows() as usize,
                actual: src.width() as usize * src.rows() as usize,
            });
        }

        let width = src.width();
        for y in 0..src.rows() {
            let src_row = src.row(y);
            let dst_row = dst.row_mut(y);
            self.convert_row(src_row, dst_row, width);
        }

        Ok(())
    }

    /// True if the conversion is a no-op (formats are identical).
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.plan.is_identity()
    }

    /// Source pixel format.
    #[inline]
    pub fn from_descriptor(&self) -> PixelDescriptor {
        self.plan.from()
    }

    /// Target pixel format.
    #[inline]
    pub fn to_descriptor(&self) -> PixelDescriptor {
        self.plan.to()
    }

    /// Access the underlying conversion plan.
    pub fn plan(&self) -> &ConvertPlan {
        &self.plan
    }
}
