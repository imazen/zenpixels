//! Owned row bands for images that cannot be represented by one allocation.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Range;

use crate::{
    BufferError, ColorContext, Orientation, PixelBuffer, PixelDescriptor, PixelEncoding,
    PixelSlice, PixelSliceMut,
};
use whereat::At;

/// Upper bound on the default band payload.
pub const DEFAULT_MAX_BAND_BYTES: u64 = 64 * 1024 * 1024;

/// Partition of an image into bounded, independently addressable row ranges.
///
/// Geometry is independent of storage. Use this type alone to schedule a stream
/// whose total byte count exceeds addressable memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BandLayout {
    width: u32,
    height: u32,
    band_rows: u32,
}

impl BandLayout {
    /// Plan tight bands for a packed format.
    pub fn new(
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
        max_band_bytes: u64,
    ) -> Result<Self, BufferError> {
        descriptor.validate()?;
        Self::for_rows(
            width,
            height,
            width as u64 * descriptor.bytes_per_pixel() as u64,
            max_band_bytes,
            1,
        )
    }

    pub(crate) fn for_rows(
        width: u32,
        height: u32,
        row_budget: u64,
        max_band_bytes: u64,
        row_multiple: u32,
    ) -> Result<Self, BufferError> {
        if width == 0 || height == 0 || row_budget == 0 {
            return Err(BufferError::InvalidDimensions);
        }
        let cap = max_band_bytes.min(isize::MAX as u64);
        let limit = (cap / row_budget).min(u64::from(height)) as u32;
        let band_rows = if limit == height {
            height
        } else {
            limit / row_multiple * row_multiple
        };
        if band_rows == 0 {
            return Err(BufferError::InvalidDimensions);
        }
        Ok(Self {
            width,
            height,
            band_rows,
        })
    }

    /// Picture width in pixels.
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Picture height in pixels.
    pub const fn height(self) -> u32 {
        self.height
    }

    /// Picture area, including on 32-bit platforms.
    pub const fn pixel_count(self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Maximum rows in a band.
    pub const fn band_rows(self) -> u32 {
        self.band_rows
    }

    /// Number of independently stored bands.
    pub const fn band_count(self) -> u32 {
        self.height.div_ceil(self.band_rows)
    }

    /// Global row interval of one band.
    pub fn band(self, index: u32) -> Option<Range<u32>> {
        if index >= self.band_count() {
            return None;
        }
        let y = index * self.band_rows;
        Some(y..y.saturating_add(self.band_rows).min(self.height))
    }

    /// Global row intervals in image order.
    pub fn bands(self) -> impl DoubleEndedIterator<Item = Range<u32>> + ExactSizeIterator {
        (0..self.band_count()).map(move |index| self.band(index).expect("valid band index"))
    }

    /// Find the owning band and the row within that band.
    pub fn locate(self, y: u32) -> Option<(usize, u32)> {
        (y < self.height).then(|| ((y / self.band_rows) as usize, y % self.band_rows))
    }
}

/// Packed pixels in independently allocated bands with one image interpretation.
#[derive(Debug)]
pub struct BandedPixelBuffer {
    layout: BandLayout,
    descriptor: PixelDescriptor,
    color: Option<Arc<ColorContext>>,
    bands: Vec<PixelBuffer>,
}

impl BandedPixelBuffer {
    /// Allocate a zeroed image using a cap on each band's sample bytes.
    pub fn try_new(
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
        max_band_bytes: u64,
    ) -> Result<Self, At<BufferError>> {
        let layout = BandLayout::new(width, height, descriptor, max_band_bytes)
            .map_err(|e| whereat::at!(e))?;
        let mut bands = Vec::new();
        bands
            .try_reserve_exact(layout.band_count() as usize)
            .map_err(|_| whereat::at!(BufferError::AllocationFailed))?;
        for rows in layout.bands() {
            bands.push(PixelBuffer::try_new(
                width,
                rows.end - rows.start,
                descriptor,
            )?);
        }
        Ok(Self {
            layout,
            descriptor,
            color: None,
            bands,
        })
    }

    /// Set current interpretation for every borrowed band.
    pub fn with_color_context(mut self, color: Arc<ColorContext>) -> Self {
        self.color = Some(color);
        self
    }

    /// Current interpretation shared by all bands.
    pub fn encoding(&self) -> PixelEncoding<'_> {
        PixelEncoding::new(self.descriptor, self.color.as_deref())
    }

    /// Band geometry and global row addressing.
    pub const fn layout(&self) -> BandLayout {
        self.layout
    }

    /// Picture area.
    pub const fn pixel_count(&self) -> u64 {
        self.layout.pixel_count()
    }

    /// Borrow each band in image order.
    pub fn bands(&self) -> impl DoubleEndedIterator<Item = PixelSlice<'_>> + ExactSizeIterator {
        self.bands.iter().map(|band| {
            let view = band.as_slice();
            match &self.color {
                Some(color) => view.with_color_context(Arc::clone(color)),
                None => view,
            }
        })
    }

    /// Write the bands in image order. Each returned view inherits the image interpretation.
    pub fn bands_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = PixelSliceMut<'_>> + ExactSizeIterator {
        self.bands.iter_mut().map(|band| {
            let view = band.as_slice_mut();
            match &self.color {
                Some(color) => view.with_color_context(Arc::clone(color)),
                None => view,
            }
        })
    }

    /// Borrow the intersecting portions of bands for a checked row interval.
    pub fn sub_rows(&self, rows: Range<u32>) -> Option<impl Iterator<Item = PixelSlice<'_>>> {
        if rows.start > rows.end || rows.end > self.layout.height {
            return None;
        }
        let start = (rows.start / self.layout.band_rows) as usize;
        let end = if rows.is_empty() {
            start
        } else {
            rows.end.div_ceil(self.layout.band_rows) as usize
        };
        Some(
            self.bands[start..end]
                .iter()
                .enumerate()
                .map(move |(i, band)| {
                    let band_start = (start + i) as u32 * self.layout.band_rows;
                    let local_start = rows.start.saturating_sub(band_start);
                    let local_end = (rows.end - band_start).min(band.height());
                    let view = band.rows(local_start, local_end - local_start);
                    match &self.color {
                        Some(color) => view.with_color_context(Arc::clone(color)),
                        None => view,
                    }
                }),
        )
    }

    /// Borrow the active samples in global row `y`.
    pub fn row(&self, y: u32) -> Option<&[u8]> {
        let (band, local_y) = self.layout.locate(y)?;
        Some(self.bands[band].rows(local_y, 1).as_strided_bytes())
    }

    /// Write the active samples in global row `y`.
    pub fn row_mut(&mut self, y: u32) -> Option<&mut [u8]> {
        let (band, local_y) = self.layout.locate(y)?;
        Some(self.bands[band].rows_mut(local_y, 1).into_bytes())
    }

    /// Replace the image interpretation after conversion.
    pub fn set_color_context(&mut self, color: Option<Arc<ColorContext>>) {
        self.color = color;
    }
    /// Apply a row-local or row-order orientation without allocating pixel storage.
    ///
    /// Axis-swapping orientations require [Self::oriented_into].
    pub fn orient_in_place(&mut self, orientation: Orientation) -> Result<(), BufferError> {
        if orientation.swaps_axes() {
            return Err(BufferError::InvalidDimensions);
        }
        let (width, height) = (self.layout.width, self.layout.height);
        if matches!(orientation, Orientation::FlipV | Orientation::Rotate180) {
            for y in 0..height / 2 {
                let (a, ay) = self.layout.locate(y).expect("row in bounds");
                let (b, by) = self.layout.locate(height - 1 - y).expect("row in bounds");
                if a == b {
                    let mut band = self.bands[a].rows_mut(ay, by - ay + 1);
                    let row_bytes = band.stride();
                    let (first, rest) = band.as_strided_bytes_mut().split_at_mut(row_bytes);
                    let start = (by - ay - 1) as usize * row_bytes;
                    first.swap_with_slice(&mut rest[start..start + row_bytes]);
                } else {
                    let (front, back) = self.bands.split_at_mut(b);
                    let mut first = front[a].rows_mut(ay, 1);
                    let mut last = back[0].rows_mut(by, 1);
                    first.row_mut(0).swap_with_slice(last.row_mut(0));
                }
            }
        }
        if matches!(orientation, Orientation::FlipH | Orientation::Rotate180) {
            let bpp = self.descriptor.bytes_per_pixel();
            for y in 0..height {
                let row = self.row_mut(y).expect("row in bounds");
                for x in 0..(width / 2) as usize {
                    let opposite = (width as usize - 1 - x) * bpp;
                    let (front, back) = row.split_at_mut(opposite);
                    front[x * bpp..(x + 1) * bpp].swap_with_slice(&mut back[..bpp]);
                }
            }
        }
        Ok(())
    }

    /// Orient into an independent banded destination, with arbitrary band heights.
    /// Axis swaps visit bounded tiles and do not allocate temporary pixel storage.
    pub fn oriented_into(
        &self,
        orientation: Orientation,
        dst: &mut Self,
    ) -> Result<(), BufferError> {
        let (w, h) = (self.layout.width, self.layout.height);
        if self.descriptor != dst.descriptor
            || orientation.output_dimensions(w, h) != (dst.layout.width, dst.layout.height)
        {
            return Err(BufferError::InvalidDimensions);
        }
        let bpp = self.descriptor.bytes_per_pixel();
        const TILE: u32 = 32;
        for y0 in (0..h).step_by(TILE as usize) {
            for x0 in (0..w).step_by(TILE as usize) {
                for sy in y0..y0.saturating_add(TILE).min(h) {
                    let s = self.row(sy).expect("row in bounds");
                    for sx in x0..x0.saturating_add(TILE).min(w) {
                        let (dx, dy) = orientation.forward_map(sx, sy, w, h);
                        let d = dst.row_mut(dy).expect("mapped row in bounds");
                        let source = sx as usize * bpp;
                        let target = dx as usize * bpp;
                        d[target..target + bpp].copy_from_slice(&s[source..source + bpp]);
                    }
                }
            }
        }
        dst.color = self.color.clone();
        Ok(())
    }
}
