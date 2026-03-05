//! Pre-computed row converter.
//!
//! [`RowConverter`] wraps a [`ConvertPlan`] and provides a streaming-friendly
//! API for converting pixel rows without per-call allocation.

use crate::convert::{ConvertPlan, convert_row};
use crate::{ConvertError, PixelDescriptor};

/// Pre-computed pixel format converter.
///
/// Create once, then call [`convert_row`](Self::convert_row) for each row.
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
#[derive(Clone, Debug)]
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

    /// Create a converter from a pre-computed plan.
    pub fn from_plan(plan: ConvertPlan) -> Self {
        Self { plan }
    }

    /// Convert one row of `width` pixels.
    ///
    /// `src` must contain at least `width * from.bytes_per_pixel()` bytes.
    /// `dst` must contain at least `width * to.bytes_per_pixel()` bytes.
    #[inline]
    pub fn convert_row(&self, src: &[u8], dst: &mut [u8], width: u32) {
        convert_row(&self.plan, src, dst, width);
    }

    /// Convert multiple rows from a strided source buffer to a strided destination.
    ///
    /// The source and destination can have different strides.
    pub fn convert_rows(
        &self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        width: u32,
        rows: u32,
    ) -> Result<(), ConvertError> {
        for y in 0..rows {
            let src_start = y as usize * src_stride;
            let src_end = src_start + (width as usize * self.plan.from().bytes_per_pixel());
            let dst_start = y as usize * dst_stride;
            let dst_end = dst_start + (width as usize * self.plan.to().bytes_per_pixel());

            if src_end > src.len() || dst_end > dst.len() {
                return Err(ConvertError::BufferSize {
                    expected: dst_end,
                    actual: dst.len(),
                });
            }

            self.convert_row(
                &src[src_start..src_end],
                &mut dst[dst_start..dst_end],
                width,
            );
        }
        Ok(())
    }

    /// True if the conversion is a no-op (formats are identical).
    #[inline]
    #[must_use]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: converting RGB8 → Oklab F32 must not panic.
    ///
    /// The Oklab layout step requires f32 data. The plan builder must schedule
    /// depth conversion (U8 → F32) *before* the layout step (RGB → Oklab),
    /// even when both layouts have the same channel count.
    #[test]
    fn rgb8_to_oklabf32_does_not_panic() {
        let conv =
            RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::OKLABF32).unwrap();
        assert!(!conv.is_identity());

        // Convert a single pixel.
        let src = [128u8, 64, 200]; // RGB8
        let mut dst = [0u8; 12]; // Oklab F32 = 3 × f32 = 12 bytes
        conv.convert_row(&src, &mut dst, 1);

        let oklab: [f32; 3] = bytemuck::cast(dst);
        // L should be in [0, 1] for valid sRGB input.
        assert!(
            oklab[0] >= 0.0 && oklab[0] <= 1.0,
            "L out of range: {}",
            oklab[0]
        );
    }
}
