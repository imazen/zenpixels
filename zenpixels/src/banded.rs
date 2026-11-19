//! Owned row bands for images that cannot be represented by one allocation.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Range;

use crate::sample::{ExactDepth, ExactDepthError, FloatFacts, FloatFactsError, SampleEvidence};
use crate::{
    BufferError, ColorContext, Orientation, PixelBuffer, PixelDescriptor, PixelEncoding,
    PixelSlice, PixelSliceMut, RowAlign,
};
use whereat::At;

/// Upper bound on the default band payload (`64 MiB`).
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
        Self::with_align(width, height, descriptor, max_band_bytes, RowAlign::Tight)
    }

    /// Plan row-aligned bands for a packed format using exact per-row stride bytes.
    pub fn with_align(
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
        max_band_bytes: u64,
        align: RowAlign,
    ) -> Result<Self, BufferError> {
        descriptor.validate()?;
        if width == 0 || height == 0 {
            return Err(BufferError::InvalidDimensions);
        }
        let stride = match align {
            RowAlign::Tight => (width as u64)
                .checked_mul(descriptor.bytes_per_pixel() as u64)
                .ok_or(BufferError::InvalidDimensions)?,
            _ => descriptor
                .try_aligned_stride(width, align)
                .ok_or(BufferError::InvalidDimensions)? as u64,
        };
        Self::for_rows(width, height, stride, max_band_bytes, 1)
    }

    pub(crate) fn for_rows(
        width: u32,
        height: u32,
        row_budget: u64,
        max_band_bytes: u64,
        row_multiple: u32,
    ) -> Result<Self, BufferError> {
        if width == 0 || height == 0 || row_budget == 0 || row_multiple == 0 {
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

/// Packed pixels in independently allocated bands with one authoritative image interpretation.
#[derive(Debug)]
pub struct BandedPixelBuffer {
    layout: BandLayout,
    descriptor: PixelDescriptor,
    color: Option<Arc<ColorContext>>,
    evidence: SampleEvidence,
    bands: Vec<PixelBuffer>,
}

impl BandedPixelBuffer {
    /// Allocate a zeroed image using a cap on each band's sample bytes (`RowAlign::Tight`).
    #[track_caller]
    pub fn try_new(
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
        max_band_bytes: u64,
    ) -> Result<Self, At<BufferError>> {
        Self::try_new_with_align(width, height, descriptor, max_band_bytes, RowAlign::Tight)
    }

    /// Allocate a zeroed row-aligned banded image with an up-front address-space preflight.
    #[track_caller]
    pub fn try_new_with_align(
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
        max_band_bytes: u64,
        align: RowAlign,
    ) -> Result<Self, At<BufferError>> {
        let layout = BandLayout::with_align(width, height, descriptor, max_band_bytes, align)
            .map_err(|e| whereat::at!(e))?;
        let stride = descriptor
            .try_aligned_stride(width, align)
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        // Preflight aggregate owner bytes against addressable memory before allocating any band.
        let total_owner_bytes = (stride as u64)
            .checked_mul(height as u64)
            .ok_or_else(|| whereat::at!(BufferError::InvalidDimensions))?;
        if total_owner_bytes > usize::MAX as u64 {
            return Err(whereat::at!(BufferError::InvalidDimensions));
        }

        let mut bands = Vec::new();
        bands
            .try_reserve_exact(layout.band_count() as usize)
            .map_err(|_| whereat::at!(BufferError::AllocationFailed))?;
        for rows in layout.bands() {
            let h = rows.end - rows.start;
            let band = match align {
                RowAlign::Tight => PixelBuffer::try_new(width, h, descriptor)?,
                _ => PixelBuffer::try_new_with_align(width, h, descriptor, align)?,
            };
            bands.push(band);
        }
        Ok(Self {
            layout,
            descriptor,
            color: None,
            evidence: SampleEvidence::NONE,
            bands,
        })
    }

    /// Adopt an owned contiguous [`PixelBuffer`] as a single-band `BandedPixelBuffer` without copying.
    #[track_caller]
    pub fn from_contiguous(mut buffer: PixelBuffer) -> Result<Self, At<BufferError>> {
        let width = buffer.width();
        let height = buffer.height();
        let layout = BandLayout::for_rows(width, height, 1, u64::from(height), 1)
            .map_err(|e| whereat::at!(e))?;
        let descriptor = buffer.descriptor();
        let color = buffer.color_context().cloned();
        let evidence = buffer.evidence_raw();
        // Strip inner per-band metadata so `self` is the single authoritative owner.
        buffer.set_color_context(None);
        buffer.set_evidence_raw(SampleEvidence::NONE);
        Ok(Self {
            layout,
            descriptor,
            color,
            evidence,
            bands: alloc::vec![buffer],
        })
    }

    /// Convert into a contiguous [`PixelBuffer`] (zero-copy when stored as a single band).
    #[track_caller]
    pub fn try_into_contiguous(mut self) -> Result<PixelBuffer, At<BufferError>> {
        if self.bands.len() == 1 {
            let mut single = self.bands.pop().expect("single band");
            single.set_color_context(self.color);
            single.set_evidence_raw(self.evidence);
            return Ok(single);
        }
        let mut out = PixelBuffer::try_new(self.layout.width, self.layout.height, self.descriptor)?;
        {
            let mut dst = out.as_slice_mut();
            for y in 0..self.layout.height {
                dst.row_mut(y)
                    .copy_from_slice(self.row(y).expect("row in bounds"));
            }
        }
        out.set_color_context(self.color);
        out.set_evidence_raw(self.evidence);
        Ok(out)
    }

    /// Set current interpretation for every borrowed band.
    #[must_use]
    pub fn with_color_context(mut self, color: Arc<ColorContext>) -> Self {
        self.color = Some(color);
        self
    }

    /// Picture width in pixels.
    #[inline]
    pub const fn width(&self) -> u32 {
        self.layout.width
    }

    /// Picture height in pixels.
    #[inline]
    pub const fn height(&self) -> u32 {
        self.layout.height
    }

    /// Pixel format and color descriptor.
    #[inline]
    pub const fn descriptor(&self) -> PixelDescriptor {
        self.descriptor
    }

    /// Attached extended [`ColorContext`], if any.
    #[inline]
    pub fn color_context(&self) -> Option<&Arc<ColorContext>> {
        self.color.as_ref()
    }

    /// Current interpretation shared by all bands.
    #[inline]
    pub fn encoding(&self) -> PixelEncoding<'_> {
        PixelEncoding::new(self.descriptor, self.color.as_deref())
    }

    /// Declared exact-depth sample evidence, if any.
    #[inline]
    pub const fn exact_depth(&self) -> Option<ExactDepth> {
        self.evidence.exact_depth()
    }

    /// Declare or clear exact-depth sample evidence across all bands.
    pub fn set_exact_depth(&mut self, exact: Option<ExactDepth>) -> Result<(), ExactDepthError> {
        if let Some(d) = exact {
            crate::buffer::check_exact(self.descriptor, d)?;
            self.evidence = SampleEvidence::from_exact(Some(d));
        } else if !self.descriptor.channel_type().is_float() {
            self.evidence = SampleEvidence::NONE;
        }
        Ok(())
    }

    /// Declared floating-point sample facts across all bands (`FloatFacts::NONE` when unstated).
    #[inline]
    pub const fn float_facts(&self) -> FloatFacts {
        self.evidence.float_facts()
    }

    /// Attach floating-point sample facts across all bands.
    pub fn with_float_facts(mut self, facts: FloatFacts) -> Result<Self, FloatFactsError> {
        self.set_float_facts(facts)?;
        Ok(self)
    }

    /// Declare or clear floating-point sample facts across all bands.
    pub fn set_float_facts(&mut self, facts: FloatFacts) -> Result<(), FloatFactsError> {
        crate::buffer::check_float_facts(self.descriptor, facts)?;
        if self.descriptor.channel_type().is_float() {
            self.evidence = SampleEvidence::from_float(facts);
        }
        Ok(())
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
        let color = self.color.as_ref();
        let evidence = self.evidence;
        self.bands.iter().map(move |band| {
            let mut view = band.as_slice().with_evidence_raw(evidence);
            if let Some(c) = color {
                view = view.with_color_context(Arc::clone(c));
            }
            view
        })
    }

    /// Write the bands in image order (clears sample evidence).
    /// Each returned view inherits the image interpretation.
    pub fn bands_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = PixelSliceMut<'_>> + ExactSizeIterator {
        self.evidence = SampleEvidence::NONE;
        let color = self.color.as_ref();
        self.bands.iter_mut().map(move |band| {
            let mut view = band.as_slice_mut();
            if let Some(c) = color {
                view = view.with_color_context(Arc::clone(c));
            }
            view
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
        let color = self.color.as_ref();
        let evidence = self.evidence;
        Some(
            self.bands[start..end]
                .iter()
                .enumerate()
                .map(move |(i, band)| {
                    let band_start = (start + i) as u32 * self.layout.band_rows;
                    let local_start = rows.start.saturating_sub(band_start);
                    let local_end = (rows.end - band_start).min(band.height());
                    let mut view = band
                        .rows(local_start, local_end - local_start)
                        .with_evidence_raw(evidence);
                    if let Some(c) = color {
                        view = view.with_color_context(Arc::clone(c));
                    }
                    view
                }),
        )
    }

    /// Borrow the active samples in global row `y`.
    pub fn row(&self, y: u32) -> Option<&[u8]> {
        let (band, local_y) = self.layout.locate(y)?;
        Some(self.bands[band].rows(local_y, 1).as_strided_bytes())
    }

    /// Write the active samples in global row `y` (clears sample evidence).
    pub fn row_mut(&mut self, y: u32) -> Option<&mut [u8]> {
        let (band, local_y) = self.layout.locate(y)?;
        self.evidence = SampleEvidence::NONE;
        Some(self.bands[band].rows_mut(local_y, 1).into_bytes())
    }

    /// Replace the image interpretation after conversion.
    pub fn set_color_context(&mut self, color: Option<Arc<ColorContext>>) {
        self.color = color;
    }

    /// Apply a row-local or row-order orientation without allocating pixel storage.
    ///
    /// Preserves [`exact_depth`](Self::exact_depth) and [`float_facts`](Self::float_facts)
    /// because geometric permutation leaves sample values unchanged. Axis-swapping
    /// orientations require [`Self::oriented_into`].
    pub fn orient_in_place(&mut self, orientation: Orientation) -> Result<(), BufferError> {
        if orientation.swaps_axes() {
            return Err(BufferError::InvalidDimensions);
        }
        let saved_evidence = self.evidence;
        let height = self.layout.height;
        if matches!(orientation, Orientation::FlipV | Orientation::Rotate180) {
            for y in 0..height / 2 {
                let (a, ay) = self.layout.locate(y).expect("row in bounds");
                let (b, by) = self.layout.locate(height - 1 - y).expect("row in bounds");
                if a == b {
                    let mut band = self.bands[a].rows_mut(ay, by - ay + 1);
                    let row_bytes = band.stride();
                    let active_bytes =
                        (self.layout.width as usize) * self.descriptor.bytes_per_pixel();
                    let (first, rest) = band.as_strided_bytes_mut().split_at_mut(row_bytes);
                    let start = (by - ay - 1) as usize * row_bytes;
                    first[..active_bytes].swap_with_slice(&mut rest[start..start + active_bytes]);
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
                reverse_pixel_row(row, bpp);
            }
        }
        self.evidence = saved_evidence;
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
        for_each_orientation_tile(w, h, |sy, columns| {
            let s = self.row(sy).expect("row in bounds");
            for sx in columns {
                let (dx, dy) = orientation.forward_map(sx, sy, w, h);
                let d = dst.row_mut(dy).expect("mapped row in bounds");
                let source = sx as usize * bpp;
                let target = dx as usize * bpp;
                d[target..target + bpp].copy_from_slice(&s[source..source + bpp]);
            }
        });
        dst.color = self.color.clone();
        dst.evidence = self.evidence;
        Ok(())
    }
}

pub(crate) fn reverse_pixel_row(row: &mut [u8], bytes: usize) {
    let width = row.len() / bytes;
    for x in 0..width / 2 {
        let opposite = (width - 1 - x) * bytes;
        let (front, back) = row.split_at_mut(opposite);
        front[x * bytes..(x + 1) * bytes].swap_with_slice(&mut back[..bytes]);
    }
}

pub(crate) fn for_each_orientation_tile(
    width: u32,
    height: u32,
    mut row: impl FnMut(u32, Range<u32>),
) {
    const TILE: u32 = 32;
    for y0 in (0..height).step_by(TILE as usize) {
        for x0 in (0..width).step_by(TILE as usize) {
            for y in y0..y0.saturating_add(TILE).min(height) {
                row(y, x0..x0.saturating_add(TILE).min(width));
            }
        }
    }
}
