//! Unparameterized pixel buffers, zero-copy strided views, bands, planes, and dual-layer HDR.
//!
//! - [`PixelBuf`] owns an alignment-exact [`PixelStorage`], optional [`HdrMetadata`],
//!   [`ColorOrigin`], and cached [`SampleFacts`] (cleared automatically on `&mut`).
//! - [`PixelView<'a>`] and [`PixelViewMut<'a>`] borrow rectangular (possibly strided
//!   or cropped) subimages with lifetimes tied directly to `'a` backing storage.
//! - Zero-copy typed pixel access is provided by methods (`row_as::<Rgb<u8>>`,
//!   `from_vec`, `try_into_vec_zero_copy`) rather than a `<P = ()>` type parameter.
//! - [`BandedBuf`] handles gigapixel images in bounded-memory horizontal bands.
//! - [`PlanarBuf`] / [`PlanarView`] handle 1..=4 planes (YCbCr, NV12, P010, planar RGB+A)
//!   including N-bit [`BitPacking`](crate::BitPacking) code extraction.
//! - [`DualLayerHdr`] carries ISO 21496-1 / UltraHDR base + gain map + headroom metadata.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt;
use whereat::At;

use crate::color::{ColorError, ColorOrigin, GainMapMetadata, HdrMetadata, ImageSpec};
use crate::facts::{FloatFacts, SampleFacts};
use crate::layout::{InteropTarget, LayoutError, PlaneSpec, RowAlign, SampleType};
use crate::storage::{ByteStorage, ByteStorageMut, PixelStorage, StorageError};

/// 4-byte RGB + unused padding pixel (`#[repr(C)]`, `align = 1`).
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Rgbx {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Unspecified padding byte.
    pub x: u8,
}

/// 4-byte BGR + unused padding pixel (`#[repr(C)]`, `align = 1`).
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Bgrx {
    /// Blue channel.
    pub b: u8,
    /// Green channel.
    pub g: u8,
    /// Red channel.
    pub r: u8,
    /// Unspecified padding byte.
    pub x: u8,
}

/// Errors returned by pixel buffer, view, band, and plane operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferError {
    /// Image specification failed validation.
    InvalidSpec(ColorError),
    /// Layout or stride calculation error.
    InvalidLayout(LayoutError),
    /// Base pointer or row stride does not satisfy the sample or pixel alignment.
    AlignmentViolation,
    /// Provided byte or pixel slice is shorter than required for `(width, height, stride)`.
    InsufficientData,
    /// Row stride is smaller than `width * bytes_per_pixel`.
    StrideTooSmall,
    /// Stride is not a whole multiple of `size_of::<P>()` for a pixel-strided API (e.g. `imgref`).
    StrideNotPixelAligned,
    /// Zero dimension or arithmetic overflow.
    InvalidDimensions,
    /// Requested crop, band, or plane index is out of bounds.
    OutOfBounds,
    /// Typed element size does not divide the active row or buffer extent cleanly.
    IncompatibleLayout,
    /// Spec does not satisfy the requested [`InteropTarget`]'s zero-copy contract.
    InteropUnsupported,
    /// Heap allocation failed (OOM or allocation layout overflow).
    AllocationFailed,
}

impl From<ColorError> for BufferError {
    fn from(e: ColorError) -> Self {
        Self::InvalidSpec(e)
    }
}

impl From<LayoutError> for BufferError {
    fn from(e: LayoutError) -> Self {
        Self::InvalidLayout(e)
    }
}

impl From<StorageError> for BufferError {
    fn from(_: StorageError) -> Self {
        Self::AllocationFailed
    }
}

impl fmt::Display for BufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpec(e) => e.fmt(f),
            Self::InvalidLayout(e) => e.fmt(f),
            Self::AlignmentViolation => f.write_str("pointer or stride violates sample alignment"),
            Self::InsufficientData => f.write_str("buffer is smaller than required image extent"),
            Self::StrideTooSmall => f.write_str("stride is smaller than active row bytes"),
            Self::StrideNotPixelAligned => {
                f.write_str("stride is not a whole multiple of bytes_per_pixel")
            }
            Self::InvalidDimensions => f.write_str("width or height is zero or overflows"),
            Self::OutOfBounds => f.write_str("crop, row range, or plane index is out of bounds"),
            Self::IncompatibleLayout => {
                f.write_str("typed element layout is incompatible with pixel buffer")
            }
            Self::InteropUnsupported => {
                f.write_str("image spec does not satisfy requested zero-copy interop contract")
            }
            Self::AllocationFailed => f.write_str("pixel buffer allocation failed"),
        }
    }
}

impl core::error::Error for BufferError {}

#[cold]
#[inline(never)]
#[track_caller]
pub(crate) fn err_at(e: BufferError) -> At<BufferError> {
    whereat::at!(e)
}

/// Minimum bytes needed for `(rows - 1) * stride + row_bytes` (last row needs no trailing padding).
pub(crate) fn required_bytes(
    rows: u32,
    stride: usize,
    row_bytes: usize,
) -> Result<usize, BufferError> {
    if rows == 0 || row_bytes == 0 {
        return Ok(0);
    }
    let extent = u64::from(rows - 1)
        .checked_mul(stride as u64)
        .and_then(|v| v.checked_add(row_bytes as u64))
        .filter(|&b| b <= isize::MAX as u64)
        .ok_or(BufferError::InvalidDimensions)?;
    Ok(extent as usize)
}

fn validate_geometry(
    data_len: usize,
    data_ptr: *const u8,
    width: u32,
    rows: u32,
    stride: usize,
    spec: &ImageSpec,
) -> Result<usize, At<BufferError>> {
    spec.validate()
        .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
    if width == 0 || rows == 0 {
        return Err(err_at(BufferError::InvalidDimensions));
    }
    let bpp = spec.layout.bytes_per_pixel();
    let row_bytes = (width as usize)
        .checked_mul(bpp)
        .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
    if stride < row_bytes {
        return Err(err_at(BufferError::StrideTooSmall));
    }
    let align = spec.layout.min_alignment();
    if stride % align != 0 || (data_ptr as usize) % align != 0 {
        return Err(err_at(BufferError::AlignmentViolation));
    }
    let req = required_bytes(rows, stride, row_bytes).map_err(err_at)?;
    if data_len < req {
        return Err(err_at(BufferError::InsufficientData));
    }
    Ok(row_bytes)
}

// ---------------------------------------------------------------------------
// PixelView (borrowed, immutable, true `'a` lifetimes)
// ---------------------------------------------------------------------------

/// Zero-copy borrowed view of an interleaved image or rectangular crop.
///
/// Every slicing method (`row`, `row_as`, `crop`, `sub_rows`,
/// `as_contiguous_bytes`) returns references or subviews tied directly to `'a`
/// rather than `&self`, so chaining (`buf.as_view().crop(...)?.row(0)`) works cleanly.
#[derive(Clone, Debug)]
pub struct PixelView<'a> {
    data: &'a [u8],
    width: u32,
    rows: u32,
    stride: usize,
    spec: ImageSpec,
    facts: SampleFacts,
    hdr: HdrMetadata,
}

impl<'a> PixelView<'a> {
    /// Borrow strided bytes with checked dimensions, alignment, and length.
    pub fn new(
        data: &'a [u8],
        width: u32,
        rows: u32,
        stride: usize,
        spec: ImageSpec,
    ) -> Result<Self, At<BufferError>> {
        validate_geometry(data.len(), data.as_ptr(), width, rows, stride, &spec)?;
        Ok(Self {
            data,
            width,
            rows,
            stride,
            spec,
            facts: SampleFacts::NONE,
            hdr: HdrMetadata::default(),
        })
    }

    /// Borrow tightly packed bytes (`stride == width * bytes_per_pixel`).
    pub fn from_packed(
        data: &'a [u8],
        width: u32,
        rows: u32,
        spec: ImageSpec,
    ) -> Result<Self, At<BufferError>> {
        let stride = (width as usize)
            .checked_mul(spec.layout.bytes_per_pixel())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        Self::new(data, width, rows, stride, spec)
    }

    /// Borrow a tightly packed typed pixel or sample slice zero-copy via `bytemuck`.
    pub fn from_typed<P: bytemuck::Pod>(
        pixels: &'a [P],
        width: u32,
        rows: u32,
        spec: ImageSpec,
    ) -> Result<Self, At<BufferError>> {
        Self::from_packed(bytemuck::cast_slice(pixels), width, rows, spec)
    }

    /// Image width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Image height in rows.
    #[inline]
    #[must_use]
    pub const fn rows(&self) -> u32 {
        self.rows
    }

    /// Byte distance between consecutive row starts.
    #[inline]
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Active bytes per row (`width * bytes_per_pixel`).
    #[inline]
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        (self.width as usize) * self.spec.layout.bytes_per_pixel()
    }

    /// Authoritative image specification (`layout` + `color`).
    #[inline]
    #[must_use]
    pub fn spec(&self) -> &ImageSpec {
        &self.spec
    }

    /// Proven sample facts attached to this view.
    #[inline]
    #[must_use]
    pub const fn facts(&self) -> SampleFacts {
        self.facts
    }

    /// Descriptive HDR container metadata attached to this view.
    #[inline]
    #[must_use]
    pub const fn hdr(&self) -> HdrMetadata {
        self.hdr
    }

    /// Attach proven sample facts to this view.
    #[inline]
    #[must_use]
    pub fn with_facts(mut self, facts: SampleFacts) -> Self {
        self.facts = facts;
        self
    }

    /// Attach descriptive HDR metadata to this view.
    #[inline]
    #[must_use]
    pub fn with_hdr(mut self, hdr: HdrMetadata) -> Self {
        self.hdr = hdr;
        self
    }

    /// Active bytes of row `y` (excludes inter-row padding), tied to backing lifetime `'a`.
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.rows()`.
    #[inline]
    #[must_use]
    pub fn row(&self, y: u32) -> &'a [u8] {
        assert!(y < self.rows, "row {y} out of bounds (rows: {})", self.rows);
        let start = (y as usize) * self.stride;
        &self.data[start..start + self.row_bytes()]
    }

    /// Zero-copy typed slice view of row `y` (e.g. `view.row_as::<rgb::Rgb<u8>>(y)`).
    ///
    /// Validates `bytemuck::try_cast_slice` alignment and exact element divisibility.
    pub fn row_as<P: bytemuck::Pod>(&self, y: u32) -> Result<&'a [P], At<BufferError>> {
        if y >= self.rows {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let bytes = self.row(y);
        bytemuck::try_cast_slice(bytes).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
        })
    }

    /// Raw strided byte backing starting at `(0, 0)` (includes inter-row padding).
    #[inline]
    #[must_use]
    pub const fn as_strided_bytes(&self) -> &'a [u8] {
        self.data
    }

    /// Whether rows are tightly packed (`stride == row_bytes` or `rows <= 1`).
    #[inline]
    #[must_use]
    pub const fn is_contiguous(&self) -> bool {
        self.rows <= 1 || self.stride == self.row_bytes()
    }

    /// Contiguous packed bytes across all rows, or `None` if padded.
    #[inline]
    #[must_use]
    pub fn as_contiguous_bytes(&self) -> Option<&'a [u8]> {
        if self.is_contiguous() {
            let total = self.row_bytes() * (self.rows as usize);
            Some(&self.data[..total])
        } else {
            None
        }
    }

    /// Zero-copy contiguous typed slice across all rows (e.g. `&[rgb::Rgba<f32>]`).
    pub fn as_contiguous_as<P: bytemuck::Pod>(&self) -> Result<&'a [P], At<BufferError>> {
        let bytes = self
            .as_contiguous_bytes()
            .ok_or_else(|| err_at(BufferError::StrideNotPixelAligned))?;
        bytemuck::try_cast_slice(bytes).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
        })
    }

    /// Borrow contiguous bytes if tightly packed, or fallibly copy active rows into a `Vec<u8>`.
    pub fn try_contiguous_bytes(&self) -> Result<Cow<'a, [u8]>, At<BufferError>> {
        if let Some(bytes) = self.as_contiguous_bytes() {
            return Ok(Cow::Borrowed(bytes));
        }
        let rb = self.row_bytes();
        let total = rb
            .checked_mul(self.rows as usize)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let mut out = Vec::new();
        out.try_reserve_exact(total)
            .map_err(|_| err_at(BufferError::AllocationFailed))?;
        for y in 0..self.rows {
            out.extend_from_slice(self.row(y));
        }
        Ok(Cow::Owned(out))
    }

    /// Zero-copy rectangular subimage view retaining parent stride and `'a` lifetime.
    pub fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Result<PixelView<'a>, At<BufferError>> {
        if w == 0 || h == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if x.checked_add(w).is_none_or(|e| e > self.width)
            || y.checked_add(h).is_none_or(|e| e > self.rows)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let bpp = self.spec.layout.bytes_per_pixel();
        let start = (y as usize) * self.stride + (x as usize) * bpp;
        let end = (y as usize + h as usize - 1) * self.stride + (x as usize + w as usize) * bpp;
        Ok(PixelView {
            data: &self.data[start..end],
            width: w,
            rows: h,
            stride: self.stride,
            spec: self.spec.clone(),
            facts: self.facts,
            hdr: self.hdr,
        })
    }

    /// Zero-copy full-width horizontal strip view retaining `'a` lifetime.
    pub fn sub_rows(&self, y: u32, count: u32) -> Result<PixelView<'a>, At<BufferError>> {
        self.crop(0, y, self.width, count)
    }

    /// Fallibly copy active pixels into a freshly allocated, tightly packed [`PixelBuf`].
    pub fn try_to_tight_buf(&self) -> Result<PixelBuf, At<BufferError>> {
        let rb = self.row_bytes();
        let total = rb
            .checked_mul(self.rows as usize)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let storage = PixelStorage::try_from_rows(
            self.spec.layout.sample,
            total,
            (0..self.rows).map(|y| self.row(y)),
        )
        .map_err(|e| err_at(e.into()))?;
        Ok(PixelBuf {
            storage,
            offset: 0,
            width: self.width,
            height: self.rows,
            stride: rb,
            spec: self.spec.clone(),
            facts: self.facts,
            hdr: self.hdr,
            origin: ColorOrigin::assumed(),
        })
    }

    /// Measure exact [`FloatFacts`] across all active `f32` samples (returns `NONE` for non-`F32`).
    #[must_use]
    pub fn measure_float_facts(&self) -> FloatFacts {
        if self.spec.layout.sample != SampleType::F32 {
            return FloatFacts::NONE;
        }
        let mut acc = FloatFacts::UNIT_INTERVAL | FloatFacts::POSITIVE;
        for y in 0..self.rows {
            let row = self.row(y);
            if let Ok(floats) = bytemuck::try_cast_slice::<u8, f32>(row) {
                acc = acc & FloatFacts::measure_f32_slice(floats);
            } else {
                return FloatFacts::NONE;
            }
        }
        acc
    }
}

// ---------------------------------------------------------------------------
// PixelViewMut (borrowed, mutable)
// ---------------------------------------------------------------------------

/// Zero-copy mutable view of an interleaved image or rectangular crop.
#[derive(Debug)]
pub struct PixelViewMut<'a> {
    data: &'a mut [u8],
    width: u32,
    rows: u32,
    stride: usize,
    spec: ImageSpec,
    hdr: HdrMetadata,
}

impl<'a> PixelViewMut<'a> {
    /// Borrow mutable strided bytes with checked dimensions, alignment, and length.
    pub fn new(
        data: &'a mut [u8],
        width: u32,
        rows: u32,
        stride: usize,
        spec: ImageSpec,
    ) -> Result<Self, At<BufferError>> {
        validate_geometry(data.len(), data.as_ptr(), width, rows, stride, &spec)?;
        Ok(Self {
            data,
            width,
            rows,
            stride,
            spec,
            hdr: HdrMetadata::default(),
        })
    }

    /// Width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in rows.
    #[inline]
    #[must_use]
    pub const fn rows(&self) -> u32 {
        self.rows
    }

    /// Row stride in bytes.
    #[inline]
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Active bytes per row.
    #[inline]
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        (self.width as usize) * self.spec.layout.bytes_per_pixel()
    }

    /// Authoritative image specification.
    #[inline]
    #[must_use]
    pub fn spec(&self) -> &ImageSpec {
        &self.spec
    }

    /// Mutable active bytes of row `y`.
    #[inline]
    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(y < self.rows, "row {y} out of bounds (rows: {})", self.rows);
        let start = (y as usize) * self.stride;
        let rb = self.row_bytes();
        &mut self.data[start..start + rb]
    }

    /// Zero-copy mutable typed slice of row `y` (e.g. `&mut [rgb::Rgba<u8>]`).
    pub fn row_as_mut<P: bytemuck::Pod>(&mut self, y: u32) -> Result<&mut [P], At<BufferError>> {
        if y >= self.rows {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let bytes = self.row_mut(y);
        bytemuck::try_cast_slice_mut(bytes).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                err_at(BufferError::AlignmentViolation)
            }
            _ => err_at(BufferError::IncompatibleLayout),
        })
    }

    /// Mutable raw strided byte backing starting at `(0, 0)` (includes inter-row padding).
    #[inline]
    pub fn as_strided_bytes_mut(&mut self) -> &mut [u8] {
        self.data
    }

    /// Mutable contiguous bytes if rows are tightly packed.
    #[inline]
    pub fn as_contiguous_bytes_mut(&mut self) -> Option<&mut [u8]> {
        if self.rows <= 1 || self.stride == self.row_bytes() {
            let total = self.row_bytes() * (self.rows as usize);
            Some(&mut self.data[..total])
        } else {
            None
        }
    }

    /// Borrow as an immutable [`PixelView`].
    #[inline]
    #[must_use]
    pub fn as_view(&self) -> PixelView<'_> {
        PixelView {
            data: self.data,
            width: self.width,
            rows: self.rows,
            stride: self.stride,
            spec: self.spec.clone(),
            facts: SampleFacts::NONE,
            hdr: self.hdr,
        }
    }

    /// Downgrade into an immutable [`PixelView<'a>`] keeping `'a`.
    #[inline]
    #[must_use]
    pub fn into_view(self) -> PixelView<'a> {
        PixelView {
            data: self.data,
            width: self.width,
            rows: self.rows,
            stride: self.stride,
            spec: self.spec,
            facts: SampleFacts::NONE,
            hdr: self.hdr,
        }
    }

    /// Mutable rectangular subimage view.
    pub fn crop_mut(
        &mut self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<PixelViewMut<'_>, At<BufferError>> {
        if w == 0 || h == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        if x.checked_add(w).is_none_or(|e| e > self.width)
            || y.checked_add(h).is_none_or(|e| e > self.rows)
        {
            return Err(err_at(BufferError::OutOfBounds));
        }
        let bpp = self.spec.layout.bytes_per_pixel();
        let start = (y as usize) * self.stride + (x as usize) * bpp;
        let end = (y as usize + h as usize - 1) * self.stride + (x as usize + w as usize) * bpp;
        Ok(PixelViewMut {
            data: &mut self.data[start..end],
            width: w,
            rows: h,
            stride: self.stride,
            spec: self.spec.clone(),
            hdr: self.hdr,
        })
    }

    /// Mutable full-width strip view.
    pub fn sub_rows_mut(
        &mut self,
        y: u32,
        count: u32,
    ) -> Result<PixelViewMut<'_>, At<BufferError>> {
        self.crop_mut(0, y, self.width, count)
    }
}

// ---------------------------------------------------------------------------
// PixelBuf (unparameterized owned buffer, fallible by default)
// ---------------------------------------------------------------------------

/// Decomposed parts returned by [`PixelBuf::into_parts`].
#[derive(Debug)]
#[non_exhaustive]
pub struct PixelBufParts {
    /// Underlying alignment-exact storage.
    pub storage: PixelStorage,
    /// Byte offset of `(0, 0)` within `storage.as_bytes()`.
    pub offset: usize,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in rows.
    pub height: u32,
    /// Byte stride between rows.
    pub stride: usize,
    /// Image specification.
    pub spec: ImageSpec,
    /// Cached sample facts.
    pub facts: SampleFacts,
    /// Descriptive HDR metadata.
    pub hdr: HdrMetadata,
    /// Container provenance.
    pub origin: ColorOrigin,
}

/// Owned, unparameterized, alignment-exact pixel image buffer.
///
/// Every allocating constructor is fallible and returns `Result<Self, At<BufferError>>`.
/// Mutable access automatically invalidates cached [`SampleFacts`].
#[derive(Debug)]
pub struct PixelBuf {
    storage: PixelStorage,
    offset: usize,
    width: u32,
    height: u32,
    stride: usize,
    spec: ImageSpec,
    facts: SampleFacts,
    hdr: HdrMetadata,
    origin: ColorOrigin,
}

impl Clone for PixelBuf {
    fn clone(&self) -> Self {
        let align = self.spec.layout.min_alignment();
        let cloned_storage = self.storage.clone();
        // Guard against cloning an accidentally-aligned `Vec<u8>` backing on a wider sample layout:
        // if the cloned byte Vec lands misaligned, re-pack into natively aligned sample storage.
        let (storage, offset, stride) = if align > 1
            && ((cloned_storage.as_bytes().as_ptr() as usize + self.offset) % align != 0
                || (self.height > 1 && self.stride % align != 0))
        {
            let rb = self.row_bytes();
            let total = rb * (self.height as usize);
            let view = self.as_view();
            let native = PixelStorage::try_from_rows(
                self.spec.layout.sample,
                total,
                (0..self.height).map(|y| view.row(y)),
            )
            .expect("cloning PixelBuf into native-aligned storage failed");
            (native, 0, rb)
        } else {
            (cloned_storage, self.offset, self.stride)
        };
        Self {
            storage,
            offset,
            width: self.width,
            height: self.height,
            stride,
            spec: self.spec.clone(),
            facts: self.facts,
            hdr: self.hdr,
            origin: self.origin.clone(),
        }
    }
}

impl PixelBuf {
    /// Fallibly allocate a tightly packed zero-filled buffer backed by native sample storage.
    pub fn new(width: u32, height: u32, spec: ImageSpec) -> Result<Self, At<BufferError>> {
        Self::new_aligned(width, height, spec, RowAlign::Tight)
    }

    /// Fallibly allocate a zero-filled buffer with explicit row alignment (`Tight`, `Cache64`, `PixelCache64`, `Samples(n)`).
    ///
    /// For `Cache64` and `PixelCache64`, storage is allocated as 64-byte aligned
    /// [`AlignedSamples`](crate::AlignedSamples) so both the first pixel and every
    /// row start land on a 64-byte boundary.
    pub fn new_aligned(
        width: u32,
        height: u32,
        spec: ImageSpec,
        align: RowAlign,
    ) -> Result<Self, At<BufferError>> {
        spec.validate()
            .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
        if width == 0 || height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let stride = spec
            .layout
            .aligned_stride(width, align)
            .map_err(|e| err_at(e.into()))?;
        let row_bytes = (width as usize)
            .checked_mul(spec.layout.bytes_per_pixel())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let total = match align {
            RowAlign::Cache64 | RowAlign::PixelCache64 => stride
                .checked_mul(height as usize)
                .ok_or_else(|| err_at(BufferError::InvalidDimensions))?,
            _ => required_bytes(height, stride, row_bytes).map_err(err_at)?,
        };
        let storage = match align {
            RowAlign::Cache64 | RowAlign::PixelCache64 => PixelStorage::try_zeroed_aligned64(total),
            _ => PixelStorage::try_zeroed(spec.layout.sample, total),
        }
        .map_err(|e| err_at(e.into()))?;
        let facts = SampleFacts::for_zeroed(spec.layout.sample);
        Ok(Self {
            storage,
            offset: 0,
            width,
            height,
            stride,
            spec,
            facts,
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
        })
    }

    /// Fallibly allocate tight native storage validated upfront for a zero-copy [`InteropTarget`].
    pub fn new_for(
        width: u32,
        height: u32,
        spec: ImageSpec,
        target: InteropTarget,
    ) -> Result<Self, At<BufferError>> {
        if !spec.supports_interop(target) {
            return Err(err_at(BufferError::InteropUnsupported));
        }
        Self::new(width, height, spec)
    }

    /// Adopt tightly packed bytes (`Vec<u8>`), handling odd-address allocators safely.
    ///
    /// - For 1-byte sample formats (`U8`, `I8`), adopts `bytes` **zero-copy** at any address.
    /// - For wider samples (`U16`, `F16`, `F32`, `F64`), adopts zero-copy when the pointer
    ///   happens to be sample-aligned, and fallibly copies into properly aligned native
    ///   [`PixelStorage`] when the allocator returned a misaligned address (e.g. odd address).
    pub fn from_bytes(
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        spec: ImageSpec,
    ) -> Result<Self, At<BufferError>> {
        spec.validate()
            .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
        if width == 0 || height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let bpp = spec.layout.bytes_per_pixel();
        let row_bytes = (width as usize)
            .checked_mul(bpp)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let req = required_bytes(height, row_bytes, row_bytes).map_err(err_at)?;
        if bytes.len() < req {
            return Err(err_at(BufferError::InsufficientData));
        }
        let align = spec.layout.min_alignment();
        let storage = if (bytes.as_ptr() as usize) % align == 0 {
            PixelStorage::U8(bytes)
        } else {
            // Odd-allocator-safe fallible copy into natively aligned storage (first byte = first pixel).
            PixelStorage::try_from_rows(spec.layout.sample, req, core::iter::once(&bytes[..req]))
                .map_err(|e| err_at(e.into()))?
        };
        Ok(Self {
            storage,
            offset: 0,
            width,
            height,
            stride: row_bytes,
            spec,
            facts: SampleFacts::NONE,
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
        })
    }

    /// Adopt a tightly packed typed `Vec<P>` (pixels like `Vec<rgb::Rgb<u16>>` or raw samples like `Vec<f32>`).
    ///
    /// **Zero-copy** whenever `align_of::<P>() == spec.layout.min_alignment()` and
    /// `bytemuck::allocation::try_cast_vec` succeeds (preserving exact deallocation alignment!).
    /// Otherwise fallibly copies into native-aligned [`PixelStorage`].
    pub fn from_vec<P: bytemuck::Pod>(
        pixels: Vec<P>,
        width: u32,
        height: u32,
        spec: ImageSpec,
    ) -> Result<Self, At<BufferError>> {
        spec.validate()
            .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
        if width == 0 || height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let bpp = spec.layout.bytes_per_pixel();
        let row_bytes = (width as usize)
            .checked_mul(bpp)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let req = required_bytes(height, row_bytes, row_bytes).map_err(err_at)?;
        let total_bytes = pixels
            .len()
            .checked_mul(core::mem::size_of::<P>())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        if total_bytes < req {
            return Err(err_at(BufferError::InsufficientData));
        }
        let storage = match PixelStorage::try_adopt_vec(pixels, spec.layout.sample) {
            Ok(s) => s,
            Err(pixels) => {
                let raw = bytemuck::cast_slice::<P, u8>(&pixels);
                PixelStorage::try_from_rows(spec.layout.sample, req, core::iter::once(&raw[..req]))
                    .map_err(|e| err_at(e.into()))?
            }
        };
        Ok(Self {
            storage,
            offset: 0,
            width,
            height,
            stride: row_bytes,
            spec,
            facts: SampleFacts::NONE,
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
        })
    }

    /// Checked reconstruction from owned [`PixelStorage`], offset, dimensions, stride, and spec.
    pub fn from_parts(
        storage: PixelStorage,
        offset: usize,
        width: u32,
        height: u32,
        stride: usize,
        spec: ImageSpec,
    ) -> Result<Self, At<BufferError>> {
        let bytes = storage.as_bytes();
        let slice = bytes
            .get(offset..)
            .ok_or_else(|| err_at(BufferError::InsufficientData))?;
        validate_geometry(slice.len(), slice.as_ptr(), width, height, stride, &spec)?;
        Ok(Self {
            storage,
            offset,
            width,
            height,
            stride,
            spec,
            facts: SampleFacts::NONE,
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
        })
    }

    /// Width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Stride in bytes.
    #[inline]
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Active bytes per row.
    #[inline]
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        (self.width as usize) * self.spec.layout.bytes_per_pixel()
    }

    /// Leading byte offset inside backing storage.
    #[inline]
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Authoritative image specification.
    #[inline]
    #[must_use]
    pub fn spec(&self) -> &ImageSpec {
        &self.spec
    }

    /// Replace specification metadata (must keep identical `bytes_per_pixel` and valid alignment; never modifies pixel bytes).
    pub fn with_spec(mut self, spec: ImageSpec) -> Result<Self, At<BufferError>> {
        if spec.layout.bytes_per_pixel() != self.spec.layout.bytes_per_pixel() {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        let bytes = &self.storage.as_bytes()[self.offset..];
        validate_geometry(
            bytes.len(),
            bytes.as_ptr(),
            self.width,
            self.height,
            self.stride,
            &spec,
        )?;
        self.spec = spec;
        self.facts = SampleFacts::NONE;
        Ok(self)
    }

    /// Cached sample facts.
    #[inline]
    #[must_use]
    pub const fn facts(&self) -> SampleFacts {
        self.facts
    }

    /// Attach proven sample facts (caller asserts validity; cleared on next `&mut` borrow).
    #[inline]
    pub fn set_facts(&mut self, facts: SampleFacts) {
        self.facts = facts;
    }

    /// Descriptive HDR container metadata.
    #[inline]
    #[must_use]
    pub const fn hdr(&self) -> HdrMetadata {
        self.hdr
    }

    /// Attach descriptive HDR container metadata.
    #[inline]
    #[must_use]
    pub fn with_hdr(mut self, hdr: HdrMetadata) -> Self {
        self.hdr = hdr;
        self
    }

    /// Original container color provenance.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> &ColorOrigin {
        &self.origin
    }

    /// Attach container color provenance.
    #[inline]
    #[must_use]
    pub fn with_origin(mut self, origin: ColorOrigin) -> Self {
        self.origin = origin;
        self
    }

    /// Borrow an immutable [`PixelView`] of the whole image.
    #[inline]
    #[must_use]
    pub fn as_view(&self) -> PixelView<'_> {
        let bytes = self.storage.as_bytes();
        let end =
            self.offset + required_bytes(self.height, self.stride, self.row_bytes()).unwrap_or(0);
        PixelView {
            data: &bytes[self.offset..end],
            width: self.width,
            rows: self.height,
            stride: self.stride,
            spec: self.spec.clone(),
            facts: self.facts,
            hdr: self.hdr,
        }
    }

    /// Borrow a mutable [`PixelViewMut`], automatically clearing cached [`SampleFacts`].
    #[inline]
    #[must_use]
    pub fn as_view_mut(&mut self) -> PixelViewMut<'_> {
        self.facts = SampleFacts::NONE;
        let rb = self.row_bytes();
        let end = self.offset + required_bytes(self.height, self.stride, rb).unwrap_or(0);
        let bytes = self.storage.as_bytes_mut();
        PixelViewMut {
            data: &mut bytes[self.offset..end],
            width: self.width,
            rows: self.height,
            stride: self.stride,
            spec: self.spec.clone(),
            hdr: self.hdr,
        }
    }

    /// Borrow a rectangular cropped subimage view without copying pixels.
    #[inline]
    pub fn crop_view(
        &self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<PixelView<'_>, At<BufferError>> {
        self.as_view().crop(x, y, w, h)
    }

    /// Compact strided rows in-place within the current allocation (`stride = row_bytes`).
    ///
    /// Preserves `offset = 0` whenever the backing storage base pointer satisfies
    /// `spec.layout.min_alignment()`, and otherwise keeps the smallest valid
    /// aligned offset (`< min_alignment()`) so sample alignment is never violated.
    pub fn into_contiguous(mut self) -> Self {
        let rb = self.row_bytes();
        let align = self.spec.layout.min_alignment();
        let ptr_rem = (self.storage.as_bytes().as_ptr() as usize) % align;
        let target_offset = if ptr_rem == 0 { 0 } else { align - ptr_rem };
        if self.offset == target_offset && (self.height <= 1 || self.stride == rb) {
            self.stride = rb;
            self.storage
                .truncate_bytes(target_offset + rb * self.height as usize);
            return self;
        }
        let bytes = self.storage.as_bytes_mut();
        for y in 0..self.height as usize {
            let src = self.offset + y * self.stride;
            let dst = target_offset + y * rb;
            if src != dst {
                bytes.copy_within(src..src + rb, dst);
            }
        }
        self.offset = target_offset;
        self.stride = rb;
        self.storage
            .truncate_bytes(target_offset + rb * self.height as usize);
        self
    }

    /// Move the backing allocation into a tightly packed `Vec<P>` **strictly zero-copy**.
    ///
    /// Succeeds only when `offset == 0`, rows are contiguous, active byte length
    /// matches the backing storage length, `align_of::<P>()` equals the backing
    /// allocation's exact alignment, and capacity in bytes is a multiple of
    /// `size_of::<P>()`. Returns `Err(self)` untouched otherwise.
    #[allow(clippy::result_large_err)]
    pub fn try_into_vec_zero_copy<P: bytemuck::Pod>(mut self) -> Result<Vec<P>, Self> {
        let total = self.row_bytes() * (self.height as usize);
        if self.offset != 0
            || (self.height > 1 && self.stride != self.row_bytes())
            || !self.storage.can_cast_into_vec::<P>(total)
        {
            return Err(self);
        }
        // Only trim trailing length AFTER proving the zero-copy cast will succeed,
        // so `Err(self)` is ALWAYS returned completely untouched!
        if self.storage.byte_len() > total {
            self.storage.truncate_bytes(total);
        }
        match self.storage.try_cast_into_vec::<P>() {
            Ok(vec) => Ok(vec),
            Err(storage) => {
                self.storage = storage;
                Err(self)
            }
        }
    }

    /// Extract tightly packed pixels into `Vec<P>`, moving **zero-copy** when
    /// allocation alignment & capacity match, or fallibly copying active rows otherwise.
    ///
    /// Never violates deallocation alignment, never drops row tails on cross-row elements,
    /// and never panics on OOM.
    pub fn try_into_vec<P: bytemuck::Pod>(self) -> Result<Vec<P>, At<BufferError>> {
        let elem = core::mem::size_of::<P>();
        let rb = self.row_bytes();
        let total = rb
            .checked_mul(self.height as usize)
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        if elem == 0 || total % elem != 0 {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
        match self.try_into_vec_zero_copy::<P>() {
            Ok(v) => Ok(v),
            Err(buf) => {
                let count = total / elem;
                if rb % elem == 0 {
                    let mut out: Vec<P> = Vec::new();
                    out.try_reserve_exact(count)
                        .map_err(|_| err_at(BufferError::AllocationFailed))?;
                    let view = buf.as_view();
                    for y in 0..view.rows() {
                        let row = view.row(y);
                        if let Ok(typed) = bytemuck::try_cast_slice::<u8, P>(row) {
                            out.extend_from_slice(typed);
                        } else {
                            for chunk in row.chunks_exact(elem) {
                                out.push(bytemuck::pod_read_unaligned(chunk));
                            }
                        }
                    }
                    Ok(out)
                } else {
                    // Element spans row boundaries (e.g. strided RGB16 extracted as `[u16; 2]`):
                    // stream active row bytes into zeroed `Vec<P>` so tight and strided agree 100%.
                    core::alloc::Layout::array::<P>(count)
                        .map_err(|_| err_at(BufferError::AllocationFailed))?;
                    let mut out: Vec<P> = bytemuck::allocation::try_zeroed_vec(count)
                        .map_err(|()| err_at(BufferError::AllocationFailed))?;
                    let dst_bytes = bytemuck::cast_slice_mut::<P, u8>(&mut out);
                    let view = buf.as_view();
                    let mut pos = 0usize;
                    for y in 0..view.rows() {
                        let row = view.row(y);
                        dst_bytes[pos..pos + row.len()].copy_from_slice(row);
                        pos += row.len();
                    }
                    Ok(out)
                }
            }
        }
    }

    /// Decompose into owned backing storage and metadata.
    #[inline]
    #[must_use]
    pub fn into_parts(self) -> PixelBufParts {
        PixelBufParts {
            storage: self.storage,
            offset: self.offset,
            width: self.width,
            height: self.height,
            stride: self.stride,
            spec: self.spec,
            facts: self.facts,
            hdr: self.hdr,
            origin: self.origin,
        }
    }

    /// Access underlying storage reference (for diagnostics and interop checks).
    #[inline]
    #[must_use]
    pub fn storage(&self) -> &PixelStorage {
        &self.storage
    }
}

/// Borrowed-or-owned pixel image returned by zero-copy negotiation / adaptation helpers.
#[derive(Clone, Debug)]
pub enum PixelCow<'a> {
    /// Zero-copy borrowed view.
    Borrowed(PixelView<'a>),
    /// Newly allocated or moved owned buffer.
    Owned(PixelBuf),
}

impl<'a> PixelCow<'a> {
    /// Borrow as a [`PixelView`].
    #[inline]
    #[must_use]
    pub fn as_view(&self) -> PixelView<'_> {
        match self {
            Self::Borrowed(v) => v.clone(),
            Self::Owned(b) => b.as_view(),
        }
    }

    /// Convert into an owned [`PixelBuf`] (zero-copy if already owned, fallible copy if borrowed).
    pub fn try_into_owned(self) -> Result<PixelBuf, At<BufferError>> {
        match self {
            Self::Borrowed(v) => v.try_to_tight_buf(),
            Self::Owned(b) => Ok(b),
        }
    }
}

// ---------------------------------------------------------------------------
// BandedBuf (gigapixel-safe horizontal bands)
// ---------------------------------------------------------------------------

/// Default per-band memory cap (`16 MiB`) for [`BandedBuf`].
pub const DEFAULT_MAX_BAND_BYTES: usize = 16 * 1024 * 1024;

/// Gigapixel-safe image stored as bounded-size horizontal bands.
///
/// Avoids huge single contiguous allocations while sharing the exact same
/// [`PixelSource`] / [`PixelSink`] contract as [`PixelBuf`] and [`PixelView`].
#[derive(Clone, Debug)]
pub struct BandedBuf {
    width: u32,
    height: u32,
    band_rows: u32,
    bands: Vec<PixelBuf>,
    spec: ImageSpec,
    facts: SampleFacts,
    hdr: HdrMetadata,
    origin: ColorOrigin,
}

impl BandedBuf {
    /// Fallibly allocate a zero-filled banded image with each band targeting at most
    /// `max_band_bytes` (minimum one row per band when `0 < max_band_bytes < row_bytes`;
    /// rejects `max_band_bytes == 0`).
    ///
    /// Note: all bands are allocated upfront, so `max_band_bytes` bounds each individual
    /// contiguous allocation rather than total image residency.
    pub fn new(
        width: u32,
        height: u32,
        spec: ImageSpec,
        max_band_bytes: usize,
    ) -> Result<Self, At<BufferError>> {
        spec.validate()
            .map_err(|e| whereat::at!(BufferError::InvalidSpec(*e.error())))?;
        if width == 0 || height == 0 || max_band_bytes == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let row_bytes = (width as usize)
            .checked_mul(spec.layout.bytes_per_pixel())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        let rows_per_band = (max_band_bytes / row_bytes.max(1))
            .max(1)
            .min(height as usize) as u32;
        let band_count = height.div_ceil(rows_per_band) as usize;
        let mut bands = Vec::new();
        bands
            .try_reserve_exact(band_count)
            .map_err(|_| err_at(BufferError::AllocationFailed))?;
        let mut y = 0u32;
        while y < height {
            let h = (height - y).min(rows_per_band);
            bands.push(PixelBuf::new(width, h, spec.clone())?);
            y += h;
        }
        let facts = SampleFacts::for_zeroed(spec.layout.sample);
        Ok(Self {
            width,
            height,
            band_rows: rows_per_band,
            bands,
            spec,
            facts,
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
        })
    }

    /// Frame width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Nominal rows per band (final band may be shorter).
    #[inline]
    #[must_use]
    pub const fn band_rows(&self) -> u32 {
        self.band_rows
    }

    /// Immutable slice of band buffers.
    #[inline]
    #[must_use]
    pub fn bands(&self) -> &[PixelBuf] {
        &self.bands
    }

    /// Borrow band `index` mutably as a [`PixelViewMut`] without allowing structural
    /// band replacement, automatically clearing cached [`SampleFacts`] on both the
    /// outer [`BandedBuf`] and the child band.
    #[inline]
    pub fn band_view_mut(&mut self, index: usize) -> Option<PixelViewMut<'_>> {
        self.facts = SampleFacts::NONE;
        self.bands.get_mut(index).map(PixelBuf::as_view_mut)
    }

    /// Image specification.
    #[inline]
    #[must_use]
    pub fn spec(&self) -> &ImageSpec {
        &self.spec
    }

    /// Cached sample facts.
    #[inline]
    #[must_use]
    pub const fn facts(&self) -> SampleFacts {
        self.facts
    }

    /// Descriptive HDR metadata.
    #[inline]
    #[must_use]
    pub const fn hdr(&self) -> HdrMetadata {
        self.hdr
    }

    /// Attach descriptive HDR metadata.
    #[inline]
    #[must_use]
    pub fn with_hdr(mut self, hdr: HdrMetadata) -> Self {
        self.hdr = hdr;
        self
    }

    /// Original container color provenance.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> &ColorOrigin {
        &self.origin
    }

    /// Attach container color provenance.
    #[inline]
    #[must_use]
    pub fn with_origin(mut self, origin: ColorOrigin) -> Self {
        self.origin = origin;
        self
    }

    /// Active bytes of global row `y`.
    #[inline]
    #[must_use]
    pub fn row(&self, y: u32) -> &[u8] {
        assert!(y < self.height, "row {y} out of bounds ({})", self.height);
        let b = (y / self.band_rows) as usize;
        let ly = y % self.band_rows;
        self.bands[b].as_view().row(ly)
    }

    /// Mutable active bytes of global row `y` (clears cached [`SampleFacts`] on both outer and child band).
    #[inline]
    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(y < self.height, "row {y} out of bounds ({})", self.height);
        self.facts = SampleFacts::NONE;
        let b = (y / self.band_rows) as usize;
        let ly = y % self.band_rows;
        self.bands[b].facts = SampleFacts::NONE;
        let stride = self.bands[b].stride();
        let rb = self.bands[b].row_bytes();
        let off = self.bands[b].offset();
        let start = off + (ly as usize) * stride;
        &mut self.bands[b].storage.as_bytes_mut()[start..start + rb]
    }
}

// ---------------------------------------------------------------------------
// PlanarBuf / PlanarView (1..=4 planes with first-class BitPacking)
// ---------------------------------------------------------------------------

/// Owned single plane inside a [`PlanarBuf`] with validated geometry invariants.
#[derive(Clone, Debug)]
pub struct OwnedPlane {
    spec: PlaneSpec,
    width: u32,
    height: u32,
    stride: usize,
    storage: PixelStorage,
}

impl OwnedPlane {
    /// Construct a validated owned plane adopting caller-provided [`PixelStorage`] zero-copy.
    pub fn from_storage(
        spec: PlaneSpec,
        width: u32,
        height: u32,
        stride: usize,
        storage: PixelStorage,
    ) -> Result<Self, At<BufferError>> {
        spec.validate().map_err(|e| err_at(e.into()))?;
        if width == 0 || height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let row_bytes = (width as usize)
            .checked_mul(spec.bytes_per_element())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        if stride < row_bytes {
            return Err(err_at(BufferError::StrideTooSmall));
        }
        let align = spec.sample.byte_size();
        let bytes = storage.as_bytes();
        if (bytes.as_ptr() as usize) % align != 0 || (height > 1 && stride % align != 0) {
            return Err(err_at(BufferError::AlignmentViolation));
        }
        let req = required_bytes(height, stride, row_bytes).map_err(err_at)?;
        if bytes.len() < req {
            return Err(err_at(BufferError::InsufficientData));
        }
        Ok(Self {
            spec,
            width,
            height,
            stride,
            storage,
        })
    }

    /// Plane specification.
    #[inline]
    #[must_use]
    pub const fn spec(&self) -> PlaneSpec {
        self.spec
    }

    /// Plane width in elements.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Plane height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Row stride in bytes.
    #[inline]
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Active bytes per row.
    #[inline]
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        (self.width as usize) * self.spec.bytes_per_element()
    }

    /// Backing storage reference.
    #[inline]
    #[must_use]
    pub fn storage(&self) -> &PixelStorage {
        &self.storage
    }

    /// Borrow as an immutable [`PlaneView`].
    #[inline]
    #[must_use]
    pub fn as_view(&self) -> PlaneView<'_> {
        let rb = self.row_bytes();
        let end = required_bytes(self.height, self.stride, rb).unwrap_or(0);
        PlaneView {
            spec: self.spec,
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: &self.storage.as_bytes()[..end],
        }
    }

    /// Borrow as a mutable [`PlaneViewMut`].
    #[inline]
    #[must_use]
    pub fn as_view_mut(&mut self) -> PlaneViewMut<'_> {
        let rb = self.row_bytes();
        let end = required_bytes(self.height, self.stride, rb).unwrap_or(0);
        PlaneViewMut {
            spec: self.spec,
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: &mut self.storage.as_bytes_mut()[..end],
        }
    }

    /// Immutable active bytes of plane row `y`.
    #[inline]
    #[must_use]
    pub fn row(&self, y: u32) -> &[u8] {
        assert!(
            y < self.height,
            "plane row {y} out of bounds ({})",
            self.height
        );
        let rb = self.row_bytes();
        let start = (y as usize) * self.stride;
        &self.storage.as_bytes()[start..start + rb]
    }

    /// Mutable active bytes of plane row `y`.
    #[inline]
    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(
            y < self.height,
            "plane row {y} out of bounds ({})",
            self.height
        );
        let rb = self.row_bytes();
        let start = (y as usize) * self.stride;
        &mut self.storage.as_bytes_mut()[start..start + rb]
    }

    /// Extract the unpacked integer code for `(x, y, component)` respecting [`BitPacking`](crate::BitPacking).
    #[must_use]
    pub fn code_u32(&self, x: u32, y: u32, component: usize) -> Option<u32> {
        if x >= self.width || y >= self.height || component >= self.spec.components as usize {
            return None;
        }
        let row = self.row(y);
        let elem = self.spec.bytes_per_element();
        let base = (x as usize) * elem;
        let word = if self.spec.packing.is_packed_word() {
            read_sample_u32(self.spec.sample, &row[base..])
        } else {
            let sz = self.spec.sample.byte_size();
            read_sample_u32(self.spec.sample, &row[base + component * sz..])
        };
        Some(
            self.spec
                .packing
                .extract_code(self.spec.sample, word, component),
        )
    }
}

fn read_sample_u32(sample: SampleType, bytes: &[u8]) -> u32 {
    match sample {
        SampleType::U8 => u32::from(bytes[0]),
        SampleType::U16 => u32::from(u16::from_ne_bytes([bytes[0], bytes[1]])),
        SampleType::U32 => u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        _ => 0,
    }
}

/// Multi-plane image (YCbCr 4:2:0/4:2:2/4:4:4, NV12, P010, planar RGB+A, planar CMYKA, JXL/TIFF extra planes).
#[derive(Clone, Debug)]
pub struct PlanarBuf {
    width: u32,
    height: u32,
    planes: Vec<OwnedPlane>,
    hdr: HdrMetadata,
    origin: ColorOrigin,
}

impl PlanarBuf {
    /// Maximum plane count supported in a single [`PlanarBuf`].
    pub const MAX_PLANES: usize = 64;

    /// Fallibly allocate a zero-filled multi-plane buffer for `specs` (`1..=64` planes).
    pub fn new(width: u32, height: u32, specs: &[PlaneSpec]) -> Result<Self, At<BufferError>> {
        if width == 0 || height == 0 || specs.is_empty() || specs.len() > Self::MAX_PLANES {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let mut planes = Vec::new();
        planes
            .try_reserve_exact(specs.len())
            .map_err(|_| err_at(BufferError::AllocationFailed))?;
        for &ps in specs {
            ps.validate().map_err(|e| err_at(e.into()))?;
            let (pw, ph) = ps.dimensions(width, height);
            let stride = (pw as usize)
                .checked_mul(ps.bytes_per_element())
                .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
            let total = required_bytes(ph, stride, stride).map_err(err_at)?;
            let storage =
                PixelStorage::try_zeroed(ps.sample, total).map_err(|e| err_at(e.into()))?;
            planes.push(OwnedPlane {
                spec: ps,
                width: pw,
                height: ph,
                stride,
                storage,
            });
        }
        Ok(Self {
            width,
            height,
            planes,
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
        })
    }

    /// Construct a validated [`PlanarBuf`] adopting caller-owned planes zero-copy (`1..=64` planes).
    pub fn from_planes(
        width: u32,
        height: u32,
        planes: Vec<OwnedPlane>,
    ) -> Result<Self, At<BufferError>> {
        if width == 0 || height == 0 || planes.is_empty() || planes.len() > Self::MAX_PLANES {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        for p in &planes {
            let (ew, eh) = p.spec.dimensions(width, height);
            if p.width != ew || p.height != eh {
                return Err(err_at(BufferError::InvalidDimensions));
            }
        }
        Ok(Self {
            width,
            height,
            planes,
            hdr: HdrMetadata::default(),
            origin: ColorOrigin::assumed(),
        })
    }

    /// Frame width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Immutable slice of planes.
    #[inline]
    #[must_use]
    pub fn planes(&self) -> &[OwnedPlane] {
        &self.planes
    }

    /// Borrow plane `index` mutably as a [`PlaneViewMut`] without allowing structural replacement.
    #[inline]
    pub fn plane_view_mut(&mut self, index: usize) -> Option<PlaneViewMut<'_>> {
        self.planes.get_mut(index).map(OwnedPlane::as_view_mut)
    }

    /// Descriptive HDR metadata.
    #[inline]
    #[must_use]
    pub const fn hdr(&self) -> HdrMetadata {
        self.hdr
    }

    /// Original container provenance.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> &ColorOrigin {
        &self.origin
    }
}

/// Validated borrowed view of one plane inside a [`PlanarView`].
#[derive(Clone, Copy, Debug)]
pub struct PlaneView<'a> {
    spec: PlaneSpec,
    width: u32,
    height: u32,
    stride: usize,
    data: &'a [u8],
}

impl<'a> PlaneView<'a> {
    const EMPTY: Self = Self {
        spec: PlaneSpec::new(crate::PlaneRole::Y, SampleType::U8),
        width: 0,
        height: 0,
        stride: 0,
        data: &[],
    };

    /// Construct a validated borrowed plane view.
    pub fn new(
        spec: PlaneSpec,
        width: u32,
        height: u32,
        stride: usize,
        data: &'a [u8],
    ) -> Result<Self, At<BufferError>> {
        spec.validate().map_err(|e| err_at(e.into()))?;
        if width == 0 || height == 0 {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        let rb = (width as usize)
            .checked_mul(spec.bytes_per_element())
            .ok_or_else(|| err_at(BufferError::InvalidDimensions))?;
        if stride < rb {
            return Err(err_at(BufferError::StrideTooSmall));
        }
        let align = spec.sample.byte_size();
        if (data.as_ptr() as usize) % align != 0 || (height > 1 && stride % align != 0) {
            return Err(err_at(BufferError::AlignmentViolation));
        }
        let req = required_bytes(height, stride, rb).map_err(err_at)?;
        if data.len() < req {
            return Err(err_at(BufferError::InsufficientData));
        }
        Ok(Self {
            spec,
            width,
            height,
            stride,
            data: &data[..req],
        })
    }

    /// Plane specification.
    #[inline]
    #[must_use]
    pub const fn spec(&self) -> PlaneSpec {
        self.spec
    }

    /// Plane width in elements.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Plane height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Row stride in bytes.
    #[inline]
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Strided backing byte slice.
    #[inline]
    #[must_use]
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Active bytes of row `y`.
    #[inline]
    #[must_use]
    pub fn row(&self, y: u32) -> &'a [u8] {
        assert!(
            y < self.height,
            "plane row {y} out of bounds ({})",
            self.height
        );
        let rb = (self.width as usize) * self.spec.bytes_per_element();
        let start = (y as usize) * self.stride;
        &self.data[start..start + rb]
    }
}

/// Validated mutable borrowed view of one plane.
#[derive(Debug)]
pub struct PlaneViewMut<'a> {
    spec: PlaneSpec,
    width: u32,
    height: u32,
    stride: usize,
    data: &'a mut [u8],
}

impl<'a> PlaneViewMut<'a> {
    /// Plane specification.
    #[inline]
    #[must_use]
    pub const fn spec(&self) -> PlaneSpec {
        self.spec
    }

    /// Plane width in elements.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Plane height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Row stride in bytes.
    #[inline]
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Mutable active bytes of row `y`.
    #[inline]
    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(
            y < self.height,
            "plane row {y} out of bounds ({})",
            self.height
        );
        let rb = (self.width as usize) * self.spec.bytes_per_element();
        let start = (y as usize) * self.stride;
        &mut self.data[start..start + rb]
    }
}

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
enum PlanarStorage<'a> {
    Inline {
        planes: [PlaneView<'a>; 4],
        len: usize,
    },
    Spill(Vec<PlaneView<'a>>),
}

/// Borrowed multi-plane view (zero heap allocation for `1..=4` planes; spills cleanly for `5..=64` planes).
#[derive(Clone, Debug)]
pub struct PlanarView<'a> {
    width: u32,
    height: u32,
    storage: PlanarStorage<'a>,
}

impl<'a> PlanarView<'a> {
    /// Construct a validated multi-plane borrowed view (`1..=64` planes).
    pub fn new(width: u32, height: u32, planes: &[PlaneView<'a>]) -> Result<Self, At<BufferError>> {
        if width == 0 || height == 0 || planes.is_empty() || planes.len() > PlanarBuf::MAX_PLANES {
            return Err(err_at(BufferError::InvalidDimensions));
        }
        for &p in planes {
            let (ew, eh) = p.spec.dimensions(width, height);
            if p.width != ew || p.height != eh {
                return Err(err_at(BufferError::InvalidDimensions));
            }
        }
        let storage = if planes.len() <= 4 {
            let mut arr = [PlaneView::EMPTY; 4];
            for (i, &p) in planes.iter().enumerate() {
                arr[i] = p;
            }
            PlanarStorage::Inline {
                planes: arr,
                len: planes.len(),
            }
        } else {
            let mut v = Vec::new();
            v.try_reserve_exact(planes.len())
                .map_err(|_| err_at(BufferError::AllocationFailed))?;
            v.extend_from_slice(planes);
            PlanarStorage::Spill(v)
        };
        Ok(Self {
            width,
            height,
            storage,
        })
    }

    /// Frame width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Borrowed planes slice.
    #[inline]
    #[must_use]
    pub fn planes(&self) -> &[PlaneView<'a>] {
        match &self.storage {
            PlanarStorage::Inline { planes, len } => &planes[..*len],
            PlanarStorage::Spill(v) => v.as_slice(),
        }
    }
}

impl PlanarBuf {
    /// Borrow all planes as a [`PlanarView`] (zero-allocation for `<= 4` planes).
    #[must_use]
    pub fn as_view(&self) -> PlanarView<'_> {
        if self.planes.len() <= 4 {
            let mut arr = [PlaneView::EMPTY; 4];
            for (i, p) in self.planes.iter().enumerate() {
                arr[i] = p.as_view();
            }
            PlanarView {
                width: self.width,
                height: self.height,
                storage: PlanarStorage::Inline {
                    planes: arr,
                    len: self.planes.len(),
                },
            }
        } else {
            let v: Vec<PlaneView<'_>> = self.planes.iter().map(OwnedPlane::as_view).collect();
            PlanarView {
                width: self.width,
                height: self.height,
                storage: PlanarStorage::Spill(v),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DualLayerHdr & DualLayerHdrView (ISO 21496-1 / UltraHDR interchange)
// ---------------------------------------------------------------------------

fn validate_dual_layer_specs(
    base_spec: &ImageSpec,
    gain_map_spec: &ImageSpec,
    metadata: GainMapMetadata,
    alternate_color_space: Option<&crate::ColorSpace>,
) -> Result<(), At<BufferError>> {
    // Base may be RGB, Gray, or a 1/3-channel ICC-described image.
    let base_ok = match &base_spec.color {
        crate::ColorSpace::Rgb { .. } | crate::ColorSpace::Gray { .. } => true,
        crate::ColorSpace::Icc { channels, .. } => matches!(*channels, 1 | 3),
        _ => false,
    };
    if !base_ok {
        return Err(err_at(BufferError::IncompatibleLayout));
    }

    // Gain map MUST be numeric (Untagged, 1 or 3 channels, no alpha) with Full/Lsb/Msb U8/U16/F32.
    let gm_layout = gain_map_spec.layout;
    let gm_color_ok = matches!(
        gain_map_spec.color,
        crate::ColorSpace::Untagged {
            channels: 1 | 3,
            alpha: crate::AlphaMode::None,
        }
    );
    let gm_layout_ok = !gm_layout.packing.is_packed_word()
        && matches!(
            gm_layout.sample,
            SampleType::U8 | SampleType::U16 | SampleType::F32
        )
        && matches!(
            gm_layout.order,
            crate::ChannelOrder::Gray
                | crate::ChannelOrder::Rgb
                | crate::ChannelOrder::Generic(1)
                | crate::ChannelOrder::Generic(3)
        );
    if !gm_color_ok || !gm_layout_ok {
        return Err(err_at(BufferError::IncompatibleLayout));
    }

    if !metadata.is_valid() {
        return Err(err_at(BufferError::InvalidDimensions));
    }

    if let Some(alt) = alternate_color_space {
        let alt_ok = match alt {
            crate::ColorSpace::Rgb { .. } | crate::ColorSpace::Gray { .. } => true,
            crate::ColorSpace::Icc { channels, .. } => matches!(*channels, 1 | 3),
            _ => false,
        };
        if !alt_ok {
            return Err(err_at(BufferError::IncompatibleLayout));
        }
    }
    Ok(())
}

/// Borrowed validated ISO 21496-1 / UltraHDR dual-layer HDR view (zero copy).
#[derive(Clone, Debug)]
pub struct DualLayerHdrView<'a> {
    base: PixelView<'a>,
    gain_map: PixelView<'a>,
    metadata: GainMapMetadata,
    alternate_color_space: Option<&'a crate::ColorSpace>,
    orientation: crate::Orientation,
}

impl<'a> DualLayerHdrView<'a> {
    /// Construct a validated borrowed dual-layer HDR view.
    pub fn new(
        base: PixelView<'a>,
        gain_map: PixelView<'a>,
        metadata: GainMapMetadata,
        alternate_color_space: Option<&'a crate::ColorSpace>,
        orientation: crate::Orientation,
    ) -> Result<Self, At<BufferError>> {
        validate_dual_layer_specs(
            base.spec(),
            gain_map.spec(),
            metadata,
            alternate_color_space,
        )?;
        Ok(Self {
            base,
            gain_map,
            metadata,
            alternate_color_space,
            orientation,
        })
    }

    /// Borrowed base rendition view.
    #[inline]
    #[must_use]
    pub const fn base(&self) -> &PixelView<'a> {
        &self.base
    }

    /// Borrowed numeric gain-map view (`ColorSpace::Untagged`, 1 or 3 channels).
    #[inline]
    #[must_use]
    pub const fn gain_map(&self) -> &PixelView<'a> {
        &self.gain_map
    }

    /// Exact rational ISO 21496-1 reconstruction metadata.
    #[inline]
    #[must_use]
    pub const fn metadata(&self) -> GainMapMetadata {
        self.metadata
    }

    /// Alternate rendition color space, if declared (required when `application_space == Alternate`).
    #[inline]
    #[must_use]
    pub const fn alternate_color_space(&self) -> Option<&'a crate::ColorSpace> {
        self.alternate_color_space
    }

    /// Shared pending EXIF orientation for the aligned `(base, gain_map)` pair.
    #[inline]
    #[must_use]
    pub const fn orientation(&self) -> crate::Orientation {
        self.orientation
    }
}

/// Container-agnostic dual-layer HDR image (base + numeric gain map + exact ISO 21496-1 metadata).
///
/// Unifies UltraHDR JPEG, AVIF `tmap`, JXL `jhgm`, and PNG gain-map HDR in one validated core type.
#[derive(Clone, Debug)]
pub struct DualLayerHdr {
    base: PixelBuf,
    gain_map: PixelBuf,
    metadata: GainMapMetadata,
    alternate_color_space: Option<crate::ColorSpace>,
    orientation: crate::Orientation,
}

impl DualLayerHdr {
    /// Construct a validated dual-layer HDR carrier.
    ///
    /// Validates base color model (`Rgb`/`Gray`/1-or-3-ch `Icc`), numeric gain-map layout
    /// (`ColorSpace::Untagged`, 1 or 3 channels, `U8`/`U16`/`F32` with `Full`/`Lsb`/`Msb` packing),
    /// and exact rational ISO 21496-1 metadata (`gamma > 0`, `min <= max`, `H_base != H_alt`).
    pub fn new(
        base: PixelBuf,
        gain_map: PixelBuf,
        metadata: GainMapMetadata,
    ) -> Result<Self, At<BufferError>> {
        validate_dual_layer_specs(base.spec(), gain_map.spec(), metadata, None)?;
        Ok(Self {
            base,
            gain_map,
            metadata,
            alternate_color_space: None,
            orientation: crate::Orientation::Identity,
        })
    }

    /// Attach a validated alternate rendition color space (required when `application_space == Alternate`).
    pub fn with_alternate_color_space(
        mut self,
        alt: crate::ColorSpace,
    ) -> Result<Self, At<BufferError>> {
        validate_dual_layer_specs(
            self.base.spec(),
            self.gain_map.spec(),
            self.metadata,
            Some(&alt),
        )?;
        self.alternate_color_space = Some(alt);
        Ok(self)
    }

    /// Attach a shared pending [`Orientation`](crate::Orientation) for the aligned pair.
    #[inline]
    #[must_use]
    pub const fn with_orientation(mut self, orientation: crate::Orientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Borrow as a zero-copy [`DualLayerHdrView`].
    #[inline]
    #[must_use]
    pub fn as_view(&self) -> DualLayerHdrView<'_> {
        DualLayerHdrView {
            base: self.base.as_view(),
            gain_map: self.gain_map.as_view(),
            metadata: self.metadata,
            alternate_color_space: self.alternate_color_space.as_ref(),
            orientation: self.orientation,
        }
    }

    /// Base rendition buffer.
    #[inline]
    #[must_use]
    pub fn base(&self) -> &PixelBuf {
        &self.base
    }

    /// Numeric gain-map buffer (`ColorSpace::Untagged`, 1 or 3 channels).
    #[inline]
    #[must_use]
    pub fn gain_map(&self) -> &PixelBuf {
        &self.gain_map
    }

    /// Validated exact rational ISO 21496-1 reconstruction metadata.
    #[inline]
    #[must_use]
    pub const fn metadata(&self) -> GainMapMetadata {
        self.metadata
    }

    /// Alternate rendition color space, if declared.
    #[inline]
    #[must_use]
    pub fn alternate_color_space(&self) -> Option<&crate::ColorSpace> {
        self.alternate_color_space.as_ref()
    }

    /// Shared pending EXIF orientation for the aligned pair.
    #[inline]
    #[must_use]
    pub const fn orientation(&self) -> crate::Orientation {
        self.orientation
    }

    /// Decompose into owned `(base, gain_map, metadata)`.
    #[inline]
    #[must_use]
    pub fn into_parts(self) -> (PixelBuf, PixelBuf, GainMapMetadata) {
        (self.base, self.gain_map, self.metadata)
    }
}

// ---------------------------------------------------------------------------
// Unified PixelSource / PixelSink (views, buffers, crops, and bands)
// ---------------------------------------------------------------------------

/// Uniform read interface across [`PixelView`], [`PixelBuf`], and [`BandedBuf`].
pub trait PixelSource {
    /// Frame width in pixels.
    fn source_width(&self) -> u32;
    /// Frame height in rows.
    fn source_height(&self) -> u32;
    /// Authoritative image specification.
    fn source_spec(&self) -> &ImageSpec;
    /// Proven sample facts.
    fn source_facts(&self) -> SampleFacts;
    /// Descriptive HDR metadata.
    fn source_hdr(&self) -> HdrMetadata;
    /// Active bytes of row `y`.
    fn source_row(&self, y: u32) -> &[u8];
}

/// Uniform write interface across [`PixelViewMut`], [`PixelBuf`], and [`BandedBuf`].
pub trait PixelSink {
    /// Frame width in pixels.
    fn sink_width(&self) -> u32;
    /// Frame height in rows.
    fn sink_height(&self) -> u32;
    /// Authoritative destination specification.
    fn sink_spec(&self) -> &ImageSpec;
    /// Mutable active bytes of row `y`.
    fn sink_row_mut(&mut self, y: u32) -> &mut [u8];
    /// Propagate proven output facts & HDR metadata on owned sinks (no-op on borrowed views).
    fn sink_finish(&mut self, _facts: SampleFacts, _hdr: HdrMetadata) {}
}

impl PixelSource for PixelView<'_> {
    #[inline]
    fn source_width(&self) -> u32 {
        self.width
    }
    #[inline]
    fn source_height(&self) -> u32 {
        self.rows
    }
    #[inline]
    fn source_spec(&self) -> &ImageSpec {
        &self.spec
    }
    #[inline]
    fn source_facts(&self) -> SampleFacts {
        self.facts
    }
    #[inline]
    fn source_hdr(&self) -> HdrMetadata {
        self.hdr
    }
    #[inline]
    fn source_row(&self, y: u32) -> &[u8] {
        self.row(y)
    }
}

impl PixelSource for PixelBuf {
    #[inline]
    fn source_width(&self) -> u32 {
        self.width
    }
    #[inline]
    fn source_height(&self) -> u32 {
        self.height
    }
    #[inline]
    fn source_spec(&self) -> &ImageSpec {
        &self.spec
    }
    #[inline]
    fn source_facts(&self) -> SampleFacts {
        self.facts
    }
    #[inline]
    fn source_hdr(&self) -> HdrMetadata {
        self.hdr
    }
    #[inline]
    fn source_row(&self, y: u32) -> &[u8] {
        assert!(y < self.height, "row {y} out of bounds ({})", self.height);
        let start = self.offset + (y as usize) * self.stride;
        &self.storage.as_bytes()[start..start + self.row_bytes()]
    }
}

impl PixelSource for BandedBuf {
    #[inline]
    fn source_width(&self) -> u32 {
        self.width
    }
    #[inline]
    fn source_height(&self) -> u32 {
        self.height
    }
    #[inline]
    fn source_spec(&self) -> &ImageSpec {
        &self.spec
    }
    #[inline]
    fn source_facts(&self) -> SampleFacts {
        self.facts
    }
    #[inline]
    fn source_hdr(&self) -> HdrMetadata {
        self.hdr
    }
    #[inline]
    fn source_row(&self, y: u32) -> &[u8] {
        self.row(y)
    }
}

impl PixelSink for PixelViewMut<'_> {
    #[inline]
    fn sink_width(&self) -> u32 {
        self.width
    }
    #[inline]
    fn sink_height(&self) -> u32 {
        self.rows
    }
    #[inline]
    fn sink_spec(&self) -> &ImageSpec {
        &self.spec
    }
    #[inline]
    fn sink_row_mut(&mut self, y: u32) -> &mut [u8] {
        self.row_mut(y)
    }
}

impl PixelSink for PixelBuf {
    #[inline]
    fn sink_width(&self) -> u32 {
        self.width
    }
    #[inline]
    fn sink_height(&self) -> u32 {
        self.height
    }
    #[inline]
    fn sink_spec(&self) -> &ImageSpec {
        &self.spec
    }
    #[inline]
    fn sink_row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(y < self.height, "row {y} out of bounds ({})", self.height);
        self.facts = SampleFacts::NONE;
        let rb = self.row_bytes();
        let start = self.offset + (y as usize) * self.stride;
        &mut self.storage.as_bytes_mut()[start..start + rb]
    }
    #[inline]
    fn sink_finish(&mut self, facts: SampleFacts, hdr: HdrMetadata) {
        self.facts = facts;
        self.hdr = hdr;
    }
}

impl PixelSink for BandedBuf {
    #[inline]
    fn sink_width(&self) -> u32 {
        self.width
    }
    #[inline]
    fn sink_height(&self) -> u32 {
        self.height
    }
    #[inline]
    fn sink_spec(&self) -> &ImageSpec {
        &self.spec
    }
    #[inline]
    fn sink_row_mut(&mut self, y: u32) -> &mut [u8] {
        self.row_mut(y)
    }
    #[inline]
    fn sink_finish(&mut self, facts: SampleFacts, hdr: HdrMetadata) {
        self.facts = facts;
        self.hdr = hdr;
        for b in &mut self.bands {
            b.facts = facts;
            b.hdr = hdr;
        }
    }
}
