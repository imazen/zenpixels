//! Strided planes and the layout of a planar picture.
//!
//! [`Plane`] is a two-dimensional array of samples with a row stride,
//! generic over who owns the samples: [`PlaneRef`] borrows them, [`PlaneMut`]
//! borrows them mutably, [`PlaneBuf`] owns a `Vec`, [`AlignedPlane`] owns
//! [`AlignedSamples`], and any other owner
//! (shared or accounted storage) implements [`SampleStorage`]. Rows are
//! contiguous `&[T]` slices, so kernels read them without per-sample
//! dispatch. Samples are native-endian: an adapter for big-endian samples
//! (FFmpeg's `*BE` formats) swaps them into a plane; a plane never holds
//! byte-swapped samples. A plane says nothing about what its samples mean.
//!
//! [`PlanarLayout`] says what they mean: each plane's [`PlaneRole`] (which
//! component, or for semi-planar chroma which two interleaved components), its
//! [`PlaneEncoding`] (integer codes per [`SampleEncoding`], or scaled floats),
//! and the picture's chroma subsampling and siting. Each plane has its own
//! encoding, so alpha or extra planes can differ in depth from color.
//!
//! Matrix, range, primaries, transfer and ICC are not part of the layout: they
//! travel with the enclosing descriptor (for example `zenpixels::Cicp`), as for
//! packed pixels. Range applies to integer code planes; a float plane's scale
//! and offset already say where its nominal values lie.
//!
//! ```
//! use zenpixels::{ChannelType, plane::*, sample::*};
//!
//! // NV12: a luma plane and one plane of interleaved Cb, Cr.
//! let u8_codes = PlaneEncoding::Code(SampleEncoding::new(ChannelType::U8, 8, 0).unwrap());
//! let layout = PlanarLayout::new(
//!     vec![
//!         PlaneSpec::new(PlaneRole::Luma, u8_codes),
//!         PlaneSpec::new(PlaneRole::CbCr, u8_codes),
//!     ],
//!     ChromaSubsampling::Yuv420,
//!     ChromaSiting::Left,
//! )
//! .unwrap();
//! assert_eq!(layout.plane_size(1, 5, 3), Some((3, 2)));
//!
//! let y = AlignedPlane::<u8>::alloc(5, 3).unwrap();
//! let uv = AlignedPlane::<u8>::alloc(3 * PlaneRole::CbCr.components(), 2).unwrap();
//! layout
//!     .check(&[AnyPlane::U8(y.view()), AnyPlane::U8(uv.view())], 5, 3)
//!     .unwrap();
//! ```

use crate::ChannelType;
use crate::pixel_types::F16Sample;
use crate::sample::{ChromaSiting, ChromaSubsampling, SampleEncoding};
use crate::storage::ROW_ALIGN;
pub use crate::storage::{
    AlignedSamples, PlaneError, PlaneSample, RowAlign, SampleStorage, SampleStorageMut,
};
use alloc::vec::Vec;
use core::fmt;

macro_rules! plane_vec_methods {
    ($($alloc:ident)?) => {
        impl<T: PlaneSample $(, $alloc: core::alloc::Allocator)?> Plane<Vec<T $(, $alloc)?>> {
            /// Give up the `Vec`, for an owner that takes a plain `Vec` and a stride
            /// (an `ImgVec`, yuvutils-rs' owned planes). Read
            /// [`width`](Plane::width), [`height`](Plane::height) and
            /// [`stride`](Plane::stride) first.
            ///
            /// # Errors
            ///
            /// The plane itself, unchanged, when its first row isn't at index 0.
            /// Today's constructors always put it there, but the first row may start
            /// later in a future aligned form, so this stays fallible. Borrow it with
            /// [`view`](Plane::view) instead, or copy it.
            pub fn try_into_vec(self) -> Result<Vec<T $(, $alloc)?>, Self> {
                if self.geom.start == 0 {
                    Ok(self.buf)
                } else {
                    Err(self)
                }
            }
        }
    };
}
#[cfg(not(feature = "allocator_api"))]
plane_vec_methods!();
#[cfg(feature = "allocator_api")]
#[rustversion::before(1.100)]
plane_vec_methods!();
#[cfg(feature = "allocator_api")]
#[rustversion::since(1.100)]
plane_vec_methods!(A);

/// Type-erased geometry of a [`Plane`], compiled once rather than per `(Sample, Owner)`.
#[derive(Clone, Copy)]
struct PlaneGeom {
    start: usize,
    width: usize,
    height: usize,
    stride: usize,
}

impl PlaneGeom {
    fn new_checked(
        buf_len: usize,
        start: usize,
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, PlaneError> {
        let end = if width == 0 || height == 0 {
            start
        } else {
            if stride < width {
                return Err(PlaneError::ShortStride);
            }
            (height - 1)
                .checked_mul(stride)
                .and_then(|n| n.checked_add(width))
                .and_then(|n| n.checked_add(start))
                .ok_or(PlaneError::GeometryOverflow)?
        };
        if end > buf_len {
            return Err(PlaneError::ShortBuffer);
        }
        Ok(Self {
            start,
            width,
            height,
            stride,
        })
    }

    #[inline]
    fn row_start(self, y: usize) -> usize {
        if self.width == 0 {
            self.start
        } else {
            self.start + y * self.stride
        }
    }

    #[inline]
    fn strided_end(self) -> usize {
        if self.width == 0 || self.height == 0 {
            self.start
        } else {
            self.start + (self.height - 1) * self.stride + self.width
        }
    }

    fn sub_rect(
        self,
        buf_len: usize,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Result<Self, PlaneError> {
        let fits = |o: usize, n: usize, max: usize| o.checked_add(n).is_some_and(|e| e <= max);
        if !fits(x, width, self.width) || !fits(y, height, self.height) {
            return Err(PlaneError::OutsidePlane);
        }
        let start = if width == 0 || height == 0 {
            self.start
        } else {
            self.start + y * self.stride + x
        };
        Self::new_checked(buf_len, start, width, height, self.stride)
    }

    fn split_at_row(self, buf_len: usize, y: usize) -> Result<(usize, Self, Self), PlaneError> {
        if y > self.height {
            return Err(PlaneError::OutsidePlane);
        }
        let at = if y == 0 || self.width == 0 {
            self.start
        } else if y == self.height {
            buf_len
        } else {
            self.start + y * self.stride
        };
        Ok((
            at,
            Self {
                start: self.start,
                width: self.width,
                height: y,
                stride: self.stride,
            },
            Self {
                start: 0,
                width: self.width,
                height: self.height - y,
                stride: self.stride,
            },
        ))
    }

    fn fmt_debug(self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plane")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

fn aligned_plane_stride(
    width: usize,
    height: usize,
    size: usize,
    align: RowAlign,
) -> Result<(usize, usize), PlaneError> {
    let stride = match align {
        RowAlign::Tight => {
            width
                .checked_mul(size)
                .ok_or(PlaneError::GeometryOverflow)?;
            width
        }
        RowAlign::Cache64 | RowAlign::PixelCache64 => {
            let row_bytes = width
                .checked_mul(size)
                .and_then(|b| b.checked_next_multiple_of(ROW_ALIGN))
                .ok_or(PlaneError::GeometryOverflow)?;
            let stride_bytes = if row_bytes != 0 && row_bytes % 4096 == 0 {
                row_bytes
                    .checked_add(ROW_ALIGN)
                    .ok_or(PlaneError::GeometryOverflow)?
            } else {
                row_bytes
            };
            stride_bytes / size
        }
        RowAlign::Samples(n) => {
            let step = usize::from(n.max(1));
            let stride = width
                .checked_next_multiple_of(step)
                .ok_or(PlaneError::GeometryOverflow)?;
            stride
                .checked_mul(size)
                .ok_or(PlaneError::GeometryOverflow)?;
            stride
        }
    };
    let len = stride
        .checked_mul(height)
        .ok_or(PlaneError::GeometryOverflow)?;
    Ok((stride, len))
}

/// A strided two-dimensional array of samples, generic over its owner.
///
/// `width` and `stride` count samples, not bytes, so a stride always holds whole
/// samples. A plane holding two interleaved components (see
/// [`PlaneRole::CbCr`]) is `components * positions` samples wide. The last row
/// need not have padding after it.
#[derive(Clone, Copy)]
pub struct Plane<C> {
    buf: C,
    geom: PlaneGeom,
}

/// A borrowed plane.
pub type PlaneRef<'a, T> = Plane<&'a [T]>;
/// A mutably borrowed plane.
pub type PlaneMut<'a, T> = Plane<&'a mut [T]>;
/// An owned plane.
pub type PlaneBuf<T> = Plane<Vec<T>>;
/// A plane that owns [`AlignedSamples`]; see [`Plane::alloc`].
pub type AlignedPlane<T> = Plane<AlignedSamples<T>>;

impl<C> fmt::Debug for Plane<C> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.geom.fmt_debug(f)
    }
}

impl<C> Plane<C> {
    /// The index of row 0 in the storage.
    #[cfg(feature = "image")]
    #[inline]
    pub(crate) fn samples_start(&self) -> usize {
        self.geom.start
    }

    /// Recover the original storage owner without copying samples, discarding
    /// plane geometry. Read dimensions/stride first if the next owner needs them.
    #[inline]
    pub fn into_storage(self) -> C {
        self.buf
    }
}

impl<C: SampleStorage> Plane<C> {
    /// Adopt storage holding `height` rows of `width` samples, `stride` samples
    /// apart, starting at the first sample. No samples are copied or read.
    ///
    /// # Errors
    ///
    /// [`PlaneError::ShortStride`] when a nonempty plane's stride is shorter
    /// than its width, [`PlaneError::ShortBuffer`] when the storage ends
    /// before the last active sample, [`PlaneError::GeometryOverflow`] when the
    /// extent overflows.
    #[inline]
    pub fn new(buf: C, width: usize, height: usize, stride: usize) -> Result<Self, PlaneError> {
        Self::with_start(buf, 0, width, height, stride)
    }

    #[inline]
    fn with_start(
        buf: C,
        start: usize,
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, PlaneError> {
        let geom = PlaneGeom::new_checked(buf.samples().len(), start, width, height, stride)?;
        Ok(Self { buf, geom })
    }

    /// Samples per row.
    #[inline]
    pub fn width(&self) -> usize {
        self.geom.width
    }

    /// Rows.
    #[inline]
    pub fn height(&self) -> usize {
        self.geom.height
    }

    /// Samples from one row start to the next.
    #[inline]
    pub fn stride(&self) -> usize {
        self.geom.stride
    }

    /// Row `y`, without padding; `None` past the last row.
    #[inline]
    pub fn row(&self, y: usize) -> Option<&[C::Sample]> {
        if y >= self.geom.height {
            return None;
        }
        let s = self.geom.row_start(y);
        Some(&self.buf.samples()[s..s + self.geom.width])
    }

    /// Every row, top to bottom, without padding.
    #[inline]
    pub fn rows(&self) -> impl DoubleEndedIterator<Item = &[C::Sample]> + ExactSizeIterator {
        let geom = self.geom;
        let samples = self.buf.samples();
        (0..geom.height).map(move |y| {
            let s = geom.row_start(y);
            &samples[s..s + geom.width]
        })
    }

    /// The samples from the first active sample through the last one, including
    /// padding between rows but not after the last row. Empty for an empty
    /// plane. For handing the plane to an API that takes a slice and a stride.
    #[inline]
    pub fn strided_samples(&self) -> &[C::Sample] {
        &self.buf.samples()[self.geom.start..self.geom.strided_end()]
    }

    /// Borrow this plane.
    #[inline]
    pub fn view(&self) -> PlaneRef<'_, C::Sample> {
        Plane {
            buf: self.strided_samples(),
            geom: PlaneGeom {
                start: 0,
                ..self.geom
            },
        }
    }

    /// Borrow the `width` x `height` samples at column `x`, row `y`.
    ///
    /// # Errors
    ///
    /// [`PlaneError::OutsidePlane`] when the rectangle leaves the plane.
    #[inline]
    pub fn sub(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Result<PlaneRef<'_, C::Sample>, PlaneError> {
        let samples = self.buf.samples();
        let geom = self.geom.sub_rect(samples.len(), x, y, width, height)?;
        Ok(Plane { buf: samples, geom })
    }
}

impl<C: SampleStorageMut> Plane<C> {
    /// Row `y`, without padding; `None` past the last row.
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> Option<&mut [C::Sample]> {
        let geom = self.geom;
        if y >= geom.height {
            return None;
        }
        let s = geom.row_start(y);
        Some(&mut self.buf.samples_mut()[s..s + geom.width])
    }

    /// Every row, top to bottom, without padding.
    #[inline]
    pub fn rows_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = &mut [C::Sample]> + ExactSizeIterator {
        let geom = self.geom;
        let end = geom.strided_end();
        let samples = &mut self.buf.samples_mut()[geom.start..end];
        RowsMut {
            samples,
            width: geom.width,
            stride: geom.stride,
            rows: geom.height,
        }
    }

    /// Mutably borrow this plane.
    #[inline]
    pub fn view_mut(&mut self) -> PlaneMut<'_, C::Sample> {
        let geom = self.geom;
        let end = geom.strided_end();
        Plane {
            buf: &mut self.buf.samples_mut()[geom.start..end],
            geom: PlaneGeom { start: 0, ..geom },
        }
    }

    /// Mutable samples from the first active sample through the last one,
    /// including padding between rows (excluding trailing padding past the
    /// last active row).
    #[inline]
    pub fn strided_samples_mut(&mut self) -> &mut [C::Sample] {
        let end = self.geom.strided_end();
        &mut self.buf.samples_mut()[self.geom.start..end]
    }

    /// Mutably borrow the `width` x `height` samples at column `x`, row `y`.
    ///
    /// # Errors
    ///
    /// [`PlaneError::OutsidePlane`] when the rectangle leaves the plane.
    #[inline]
    pub fn sub_mut(
        &mut self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Result<PlaneMut<'_, C::Sample>, PlaneError> {
        let geom = self.geom;
        let samples = self.buf.samples_mut();
        let sub_geom = geom.sub_rect(samples.len(), x, y, width, height)?;
        Ok(Plane {
            buf: samples,
            geom: sub_geom,
        })
    }
}

impl<T: PlaneSample> Plane<&mut [T]> {
    /// Split into rows `0..y` and rows `y..`, for filling strips or
    /// handing disjoint rows to threads.
    ///
    /// # Errors
    ///
    /// [`PlaneError::OutsidePlane`] when `y` is past the last row.
    #[inline]
    pub fn split_at_row(self, y: usize) -> Result<(Self, Self), PlaneError> {
        let (at, top_geom, bottom_geom) = self.geom.split_at_row(self.buf.len(), y)?;
        let (top, bottom) = self.buf.split_at_mut(at);
        Ok((
            Plane {
                buf: top,
                geom: top_geom,
            },
            Plane {
                buf: bottom,
                geom: bottom_geom,
            },
        ))
    }
}

impl<'a, T: PlaneSample> Plane<&'a [T]> {
    /// [`Plane::strided_samples`] for the borrowed lifetime, so an adapter can
    /// hand the samples to another crate's borrowed frame type (slice from the
    /// first sample, stride in samples) without copying.
    #[inline]
    pub fn into_strided_samples(self) -> &'a [T] {
        &self.buf[self.geom.start..self.geom.strided_end()]
    }
}

impl<'a, T: PlaneSample> Plane<&'a mut [T]> {
    /// The mutable samples from the first active sample through the last one,
    /// for the borrowed lifetime; see [`into_strided_samples`](Plane::into_strided_samples).
    #[inline]
    pub fn into_strided_samples_mut(self) -> &'a mut [T] {
        &mut self.buf[self.geom.start..self.geom.strided_end()]
    }
}

/// Converting to imgref: dimensions past `u32::MAX`, which imgref cannot hold,
/// are [`PlaneError::GeometryOverflow`]. An empty plane's zero stride becomes 1,
/// which imgref requires.
#[cfg(feature = "imgref")]
fn imgref_geometry(geom: PlaneGeom) -> Result<usize, PlaneError> {
    if u32::try_from(geom.width).is_err() || u32::try_from(geom.height).is_err() {
        return Err(PlaneError::GeometryOverflow);
    }
    Ok(geom
        .stride
        .max(if geom.height == 0 { geom.width } else { 0 })
        .max(1))
}

/// Borrow an [`imgref::ImgRef`] of samples as a plane, without copying.
///
/// # Errors
///
/// A [`PlaneError`] when the image's buffer is shorter than its geometry.
#[cfg(feature = "imgref")]
impl<'a, T: PlaneSample> TryFrom<imgref::ImgRef<'a, T>> for PlaneRef<'a, T> {
    type Error = PlaneError;
    #[inline]
    fn try_from(img: imgref::ImgRef<'a, T>) -> Result<Self, PlaneError> {
        Plane::new(img.buf(), img.width(), img.height(), img.stride())
    }
}

/// Borrow an [`imgref::ImgRefMut`] of samples as a plane, without copying.
#[cfg(feature = "imgref")]
impl<'a, T: PlaneSample> TryFrom<imgref::ImgRefMut<'a, T>> for PlaneMut<'a, T> {
    type Error = PlaneError;
    #[inline]
    fn try_from(img: imgref::ImgRefMut<'a, T>) -> Result<Self, PlaneError> {
        let (width, height, stride) = (img.width(), img.height(), img.stride());
        Plane::new(img.into_buf(), width, height, stride)
    }
}

/// Adopt an [`imgref::ImgVec`]'s `Vec` as a plane, without copying.
#[cfg(feature = "imgref")]
impl<T: PlaneSample> TryFrom<imgref::ImgVec<T>> for PlaneBuf<T> {
    type Error = PlaneError;
    #[inline]
    fn try_from(img: imgref::ImgVec<T>) -> Result<Self, PlaneError> {
        let (width, height, stride) = (img.width(), img.height(), img.stride());
        Plane::new(img.into_buf(), width, height, stride)
    }
}

/// Borrow a plane as an [`imgref::ImgRef`], without copying.
#[cfg(feature = "imgref")]
impl<'a, T: PlaneSample> TryFrom<PlaneRef<'a, T>> for imgref::ImgRef<'a, T> {
    type Error = PlaneError;
    #[inline]
    fn try_from(plane: PlaneRef<'a, T>) -> Result<Self, PlaneError> {
        let stride = imgref_geometry(plane.geom)?;
        let (width, height) = (plane.geom.width, plane.geom.height);
        Ok(imgref::Img::new_stride(
            plane.into_strided_samples(),
            width,
            height,
            stride,
        ))
    }
}

/// Borrow a mutable plane as an [`imgref::ImgRefMut`], without copying.
#[cfg(feature = "imgref")]
impl<'a, T: PlaneSample> TryFrom<PlaneMut<'a, T>> for imgref::ImgRefMut<'a, T> {
    type Error = PlaneError;
    #[inline]
    fn try_from(plane: PlaneMut<'a, T>) -> Result<Self, PlaneError> {
        let stride = imgref_geometry(plane.geom)?;
        let (width, height) = (plane.geom.width, plane.geom.height);
        Ok(imgref::Img::new_stride(
            plane.into_strided_samples_mut(),
            width,
            height,
            stride,
        ))
    }
}

struct RowsMut<'a, T> {
    samples: &'a mut [T],
    width: usize,
    stride: usize,
    rows: usize,
}

impl<'a, T> Iterator for RowsMut<'a, T> {
    type Item = &'a mut [T];
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.rows == 0 {
            return None;
        }
        self.rows -= 1;
        if self.width == 0 {
            return Some(&mut []);
        }
        let samples = core::mem::take(&mut self.samples);
        if self.rows == 0 {
            return Some(&mut samples[..self.width]);
        }
        let (row, rest) = samples.split_at_mut(self.stride);
        self.samples = rest;
        Some(&mut row[..self.width])
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.rows, Some(self.rows))
    }
}

impl<T> DoubleEndedIterator for RowsMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.rows == 0 {
            return None;
        }
        self.rows -= 1;
        if self.width == 0 {
            return Some(&mut []);
        }
        let samples = core::mem::take(&mut self.samples);
        let (rest, last) = samples.split_at_mut(self.rows * self.stride);
        self.samples = rest;
        Some(&mut last[..self.width])
    }
}

impl<T> ExactSizeIterator for RowsMut<'_, T> {}

impl<T: PlaneSample> Plane<AlignedSamples<T>> {
    /// Allocate a zeroed `width` x `height` plane for SIMD kernels.
    ///
    /// Equivalent to `Self::alloc_aligned(width, height, RowAlign::Cache64)`:
    /// every row starts on a 64-byte boundary, with row 0 at index 0.
    ///
    /// # Errors
    ///
    /// [`PlaneError::GeometryOverflow`] when the size overflows,
    /// [`PlaneError::AllocationFailed`] when the allocator refuses.
    #[inline]
    pub fn alloc(width: usize, height: usize) -> Result<Self, PlaneError> {
        Self::alloc_aligned(width, height, RowAlign::Cache64)
    }

    /// Allocate a zeroed `width` x `height` plane with explicit [`RowAlign`] row
    /// alignment, with row 0 on a 64-byte boundary at index 0.
    ///
    /// # Errors
    ///
    /// [`PlaneError::GeometryOverflow`] when the size overflows,
    /// [`PlaneError::AllocationFailed`] when the allocator refuses.
    #[inline]
    pub fn alloc_aligned(width: usize, height: usize, align: RowAlign) -> Result<Self, PlaneError> {
        let (stride, len) = aligned_plane_stride(width, height, core::mem::size_of::<T>(), align)?;
        Self::with_start(AlignedSamples::try_zeroed(len)?, 0, width, height, stride)
    }
}

/// Hand a plane's `Vec` to an [`imgref::ImgVec`] without copying.
///
/// The error is the plane itself, unchanged, when it starts past index 0 (see
/// [`PlaneBuf::try_into_vec`]) or holds dimensions past `u32::MAX`.
#[cfg(feature = "imgref")]
impl<T: PlaneSample> TryFrom<PlaneBuf<T>> for imgref::ImgVec<T> {
    type Error = PlaneBuf<T>;
    #[inline]
    fn try_from(plane: PlaneBuf<T>) -> Result<Self, PlaneBuf<T>> {
        let Ok(stride) = imgref_geometry(plane.geom) else {
            return Err(plane);
        };
        let (width, height) = (plane.geom.width, plane.geom.height);
        let buf = plane.try_into_vec()?;
        Ok(imgref::Img::new_stride(buf, width, height, stride))
    }
}

/// A read-only borrowed plane with runtime-selected native sample type.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum AnyPlane<'a> {
    /// Native `u8` samples.
    U8(PlaneRef<'a, u8>),
    /// Native `i8` samples.
    I8(PlaneRef<'a, i8>),
    /// Native `u16` samples.
    U16(PlaneRef<'a, u16>),
    /// Native `i16` samples.
    I16(PlaneRef<'a, i16>),
    /// Native `f16` (`F16Sample`) samples.
    F16(PlaneRef<'a, F16Sample>),
    /// Native `u32` samples.
    U32(PlaneRef<'a, u32>),
    /// Native `i32` samples.
    I32(PlaneRef<'a, i32>),
    /// Native `f32` samples.
    F32(PlaneRef<'a, f32>),
    /// Native `u64` samples.
    U64(PlaneRef<'a, u64>),
    /// Native `i64` samples.
    I64(PlaneRef<'a, i64>),
    /// Native `f64` samples.
    F64(PlaneRef<'a, f64>),
}

impl AnyPlane<'_> {
    /// Samples per row.
    pub fn width(&self) -> usize {
        match self {
            Self::U8(p) => p.width(),
            Self::I8(p) => p.width(),
            Self::U16(p) => p.width(),
            Self::I16(p) => p.width(),
            Self::F16(p) => p.width(),
            Self::U32(p) => p.width(),
            Self::I32(p) => p.width(),
            Self::F32(p) => p.width(),
            Self::U64(p) => p.width(),
            Self::I64(p) => p.width(),
            Self::F64(p) => p.width(),
        }
    }
    /// Rows.
    pub fn height(&self) -> usize {
        match self {
            Self::U8(p) => p.height(),
            Self::I8(p) => p.height(),
            Self::U16(p) => p.height(),
            Self::I16(p) => p.height(),
            Self::F16(p) => p.height(),
            Self::U32(p) => p.height(),
            Self::I32(p) => p.height(),
            Self::F32(p) => p.height(),
            Self::U64(p) => p.height(),
            Self::I64(p) => p.height(),
            Self::F64(p) => p.height(),
        }
    }
    /// Native storage type.
    pub fn channel_type(&self) -> ChannelType {
        match self {
            Self::U8(_) => ChannelType::U8,
            Self::I8(_) => ChannelType::I8,
            Self::U16(_) => ChannelType::U16,
            Self::I16(_) => ChannelType::I16,
            Self::F16(_) => ChannelType::F16,
            Self::U32(_) => ChannelType::U32,
            Self::I32(_) => ChannelType::I32,
            Self::F32(_) => ChannelType::F32,
            Self::U64(_) => ChannelType::U64,
            Self::I64(_) => ChannelType::I64,
            Self::F64(_) => ChannelType::F64,
        }
    }
    /// Bytes between row starts, with checked sample-to-byte multiplication.
    pub fn stride_bytes(&self) -> Result<usize, PlaneError> {
        let stride = match self {
            Self::U8(p) => p.stride(),
            Self::I8(p) => p.stride(),
            Self::U16(p) => p.stride(),
            Self::I16(p) => p.stride(),
            Self::F16(p) => p.stride(),
            Self::U32(p) => p.stride(),
            Self::I32(p) => p.stride(),
            Self::F32(p) => p.stride(),
            Self::U64(p) => p.stride(),
            Self::I64(p) => p.stride(),
            Self::F64(p) => p.stride(),
        };
        stride
            .checked_mul(self.channel_type().byte_size())
            .ok_or(PlaneError::GeometryOverflow)
    }
    /// Native-endian samples and intervening padding, without final-row padding.
    pub fn as_strided_bytes(&self) -> &[u8] {
        match self {
            Self::U8(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::I8(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::U16(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::I16(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::F16(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::U32(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::I32(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::F32(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::U64(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::I64(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::F64(p) => bytemuck::cast_slice(p.strided_samples()),
        }
    }
}

/// A writable borrowed plane with runtime-selected native sample type.
#[derive(Debug)]
#[non_exhaustive]
pub enum AnyPlaneMut<'a> {
    /// Native `u8` samples.
    U8(PlaneMut<'a, u8>),
    /// Native `i8` samples.
    I8(PlaneMut<'a, i8>),
    /// Native `u16` samples.
    U16(PlaneMut<'a, u16>),
    /// Native `i16` samples.
    I16(PlaneMut<'a, i16>),
    /// Native `f16` (`F16Sample`) samples.
    F16(PlaneMut<'a, F16Sample>),
    /// Native `u32` samples.
    U32(PlaneMut<'a, u32>),
    /// Native `i32` samples.
    I32(PlaneMut<'a, i32>),
    /// Native `f32` samples.
    F32(PlaneMut<'a, f32>),
    /// Native `u64` samples.
    U64(PlaneMut<'a, u64>),
    /// Native `i64` samples.
    I64(PlaneMut<'a, i64>),
    /// Native `f64` samples.
    F64(PlaneMut<'a, f64>),
}

impl<'a> AnyPlaneMut<'a> {
    /// Samples per row.
    pub fn width(&self) -> usize {
        self.as_ref().width()
    }
    /// Rows.
    pub fn height(&self) -> usize {
        self.as_ref().height()
    }
    /// Native storage type.
    pub fn channel_type(&self) -> ChannelType {
        self.as_ref().channel_type()
    }
    /// Bytes between row starts, with checked sample-to-byte multiplication.
    pub fn stride_bytes(&self) -> Result<usize, PlaneError> {
        self.as_ref().stride_bytes()
    }
    /// Native-endian samples and intervening padding, without final-row padding.
    pub fn as_strided_bytes(&self) -> &[u8] {
        match self {
            Self::U8(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::I8(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::U16(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::I16(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::F16(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::U32(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::I32(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::F32(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::U64(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::I64(p) => bytemuck::cast_slice(p.strided_samples()),
            Self::F64(p) => bytemuck::cast_slice(p.strided_samples()),
        }
    }
    /// Mutable native-endian samples and intervening padding, without final-row padding.
    pub fn as_strided_bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Self::U8(p) => bytemuck::cast_slice_mut(p.strided_samples_mut()),
            Self::I8(p) => bytemuck::cast_slice_mut(p.strided_samples_mut()),
            Self::U16(p) => bytemuck::cast_slice_mut(p.strided_samples_mut()),
            Self::I16(p) => bytemuck::cast_slice_mut(p.strided_samples_mut()),
            Self::F16(p) => bytemuck::cast_slice_mut(p.strided_samples_mut()),
            Self::U32(p) => bytemuck::cast_slice_mut(p.strided_samples_mut()),
            Self::I32(p) => bytemuck::cast_slice_mut(p.strided_samples_mut()),
            Self::F32(p) => bytemuck::cast_slice_mut(p.strided_samples_mut()),
            Self::U64(p) => bytemuck::cast_slice_mut(p.strided_samples_mut()),
            Self::I64(p) => bytemuck::cast_slice_mut(p.strided_samples_mut()),
            Self::F64(p) => bytemuck::cast_slice_mut(p.strided_samples_mut()),
        }
    }
    /// Borrow without allocating or copying samples.
    pub fn as_ref(&self) -> AnyPlane<'_> {
        match self {
            Self::U8(p) => AnyPlane::U8(p.view()),
            Self::I8(p) => AnyPlane::I8(p.view()),
            Self::U16(p) => AnyPlane::U16(p.view()),
            Self::I16(p) => AnyPlane::I16(p.view()),
            Self::F16(p) => AnyPlane::F16(p.view()),
            Self::U32(p) => AnyPlane::U32(p.view()),
            Self::I32(p) => AnyPlane::I32(p.view()),
            Self::F32(p) => AnyPlane::F32(p.view()),
            Self::U64(p) => AnyPlane::U64(p.view()),
            Self::I64(p) => AnyPlane::I64(p.view()),
            Self::F64(p) => AnyPlane::F64(p.view()),
        }
    }
    /// Reborrow writable samples without replacing the original descriptor.
    pub fn as_mut(&mut self) -> AnyPlaneMut<'_> {
        match self {
            Self::U8(p) => AnyPlaneMut::U8(p.view_mut()),
            Self::I8(p) => AnyPlaneMut::I8(p.view_mut()),
            Self::U16(p) => AnyPlaneMut::U16(p.view_mut()),
            Self::I16(p) => AnyPlaneMut::I16(p.view_mut()),
            Self::F16(p) => AnyPlaneMut::F16(p.view_mut()),
            Self::U32(p) => AnyPlaneMut::U32(p.view_mut()),
            Self::I32(p) => AnyPlaneMut::I32(p.view_mut()),
            Self::F32(p) => AnyPlaneMut::F32(p.view_mut()),
            Self::U64(p) => AnyPlaneMut::U64(p.view_mut()),
            Self::I64(p) => AnyPlaneMut::I64(p.view_mut()),
            Self::F64(p) => AnyPlaneMut::F64(p.view_mut()),
        }
    }
}

/// A Vec-owned plane with runtime-selected native sample type.
#[derive(Debug)]
#[non_exhaustive]
pub enum AnyPlaneBuf {
    /// Native `u8` samples.
    U8(PlaneBuf<u8>),
    /// Native `i8` samples.
    I8(PlaneBuf<i8>),
    /// Native `u16` samples.
    U16(PlaneBuf<u16>),
    /// Native `i16` samples.
    I16(PlaneBuf<i16>),
    /// Native `f16` (`F16Sample`) samples.
    F16(PlaneBuf<F16Sample>),
    /// Native `u32` samples.
    U32(PlaneBuf<u32>),
    /// Native `i32` samples.
    I32(PlaneBuf<i32>),
    /// Native `f32` samples.
    F32(PlaneBuf<f32>),
    /// Native `u64` samples.
    U64(PlaneBuf<u64>),
    /// Native `i64` samples.
    I64(PlaneBuf<i64>),
    /// Native `f64` samples.
    F64(PlaneBuf<f64>),
}

impl AnyPlaneBuf {
    /// Borrow without allocating or copying samples.
    pub fn as_ref(&self) -> AnyPlane<'_> {
        match self {
            Self::U8(p) => AnyPlane::U8(p.view()),
            Self::I8(p) => AnyPlane::I8(p.view()),
            Self::U16(p) => AnyPlane::U16(p.view()),
            Self::I16(p) => AnyPlane::I16(p.view()),
            Self::F16(p) => AnyPlane::F16(p.view()),
            Self::U32(p) => AnyPlane::U32(p.view()),
            Self::I32(p) => AnyPlane::I32(p.view()),
            Self::F32(p) => AnyPlane::F32(p.view()),
            Self::U64(p) => AnyPlane::U64(p.view()),
            Self::I64(p) => AnyPlane::I64(p.view()),
            Self::F64(p) => AnyPlane::F64(p.view()),
        }
    }
    /// Reborrow writable samples without replacing the original descriptor.
    pub fn as_mut(&mut self) -> AnyPlaneMut<'_> {
        match self {
            Self::U8(p) => AnyPlaneMut::U8(p.view_mut()),
            Self::I8(p) => AnyPlaneMut::I8(p.view_mut()),
            Self::U16(p) => AnyPlaneMut::U16(p.view_mut()),
            Self::I16(p) => AnyPlaneMut::I16(p.view_mut()),
            Self::F16(p) => AnyPlaneMut::F16(p.view_mut()),
            Self::U32(p) => AnyPlaneMut::U32(p.view_mut()),
            Self::I32(p) => AnyPlaneMut::I32(p.view_mut()),
            Self::F32(p) => AnyPlaneMut::F32(p.view_mut()),
            Self::U64(p) => AnyPlaneMut::U64(p.view_mut()),
            Self::I64(p) => AnyPlaneMut::I64(p.view_mut()),
            Self::F64(p) => AnyPlaneMut::F64(p.view_mut()),
        }
    }
}

/// A plane owner or view that can lend native samples and geometry.
///
/// Implement this for a runtime-selected owner such as a codec's accounted
/// plane enum. Typed [`Plane<C>`] implements it for every [`SampleStorage`].
/// The geometry and native sample type must remain stable while borrowed by a
/// picture; interior mutation must not replace or resize the backing storage.
/// This trait carries no color interpretation and does not allocate.
pub trait PlaneStorage {
    /// Borrow this plane's native samples and geometry.
    fn as_plane(&self) -> AnyPlane<'_>;
}

/// Exclusive sample access to a [`PlaneStorage`]. Read-only/shared owners need
/// not implement this. The writable view must have the same geometry and sample
/// type as `as_plane`; mutation changes samples, not the storage descriptor.
pub trait PlaneStorageMut: PlaneStorage {
    /// Reborrow writable samples, preserving the owner's descriptor.
    fn as_plane_mut(&mut self) -> AnyPlaneMut<'_>;
}

impl<C: SampleStorage> PlaneStorage for Plane<C> {
    fn as_plane(&self) -> AnyPlane<'_> {
        <C::Sample as crate::buffer::storage_sample::Sealed>::plane_view(
            self.view(),
            crate::buffer::storage_sample::Private(()),
        )
    }
}
impl<C: SampleStorageMut> PlaneStorageMut for Plane<C> {
    fn as_plane_mut(&mut self) -> AnyPlaneMut<'_> {
        <C::Sample as crate::buffer::storage_sample::Sealed>::plane_view_mut(
            self.view_mut(),
            crate::buffer::storage_sample::Private(()),
        )
    }
}
impl PlaneStorage for AnyPlane<'_> {
    fn as_plane(&self) -> AnyPlane<'_> {
        *self
    }
}
impl PlaneStorage for AnyPlaneMut<'_> {
    fn as_plane(&self) -> AnyPlane<'_> {
        self.as_ref()
    }
}
impl PlaneStorageMut for AnyPlaneMut<'_> {
    fn as_plane_mut(&mut self) -> AnyPlaneMut<'_> {
        self.as_mut()
    }
}
impl PlaneStorage for AnyPlaneBuf {
    fn as_plane(&self) -> AnyPlane<'_> {
        self.as_ref()
    }
}
impl PlaneStorageMut for AnyPlaneBuf {
    fn as_plane_mut(&mut self) -> AnyPlaneMut<'_> {
        self.as_mut()
    }
}

// Erase the collection owner only for shared picture views. Writable views
// retain their concrete plane-owner type and iterate its slice directly.
trait PlaneCollection: Sync {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<AnyPlane<'_>>;
}
impl<P: PlaneStorage + Sync> PlaneCollection for Vec<P> {
    fn len(&self) -> usize {
        self.as_slice().len()
    }
    fn get(&self, index: usize) -> Option<AnyPlane<'_>> {
        self.as_slice().get(index).map(PlaneStorage::as_plane)
    }
}
impl<P: PlaneStorage + Sync> PlaneCollection for &mut [P] {
    fn len(&self) -> usize {
        <[P]>::len(self)
    }
    fn get(&self, index: usize) -> Option<AnyPlane<'_>> {
        <[P]>::get(self, index).map(PlaneStorage::as_plane)
    }
}
impl fmt::Debug for dyn PlaneCollection + '_ {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries((0..self.len()).map(|i| self.get(i)))
            .finish()
    }
}

// Views borrow the descriptor owner, rather than allocating a second descriptor
// list or imposing a fixed plane-count limit. Mutable storage is only exposed
// through reborrows: replacing a returned descriptor cannot invalidate its owner.
#[derive(Clone, Copy, Debug)]
enum Planes<'a> {
    Borrowed(&'a [AnyPlane<'a>]),
    Owner(&'a dyn PlaneCollection),
}
impl<'a> Planes<'a> {
    fn len(self) -> usize {
        match self {
            Self::Borrowed(p) => p.len(),
            Self::Owner(p) => p.len(),
        }
    }
    fn get(self, index: usize) -> Option<AnyPlane<'a>> {
        match self {
            Self::Borrowed(p) => p.get(index).copied(),
            Self::Owner(p) => p.get(index),
        }
    }
    fn check(
        self,
        layout: &PlanarLayout,
        width: u32,
        height: u32,
        alpha: Option<crate::AlphaMode>,
    ) -> Result<(), LayoutError> {
        if self.len() != layout.planes().len() {
            return Err(LayoutError::PlaneCount {
                expected: layout.planes().len(),
                actual: self.len(),
            });
        }
        for i in 0..self.len() {
            layout.check_plane(i, &self.get(i).expect("in bounds"), width, height)?;
        }
        if layout.roles().any(PlaneRole::is_alpha) != alpha.is_some()
            || alpha == Some(crate::AlphaMode::Undefined)
        {
            return Err(LayoutError::AlphaInterpretation);
        }
        Ok(())
    }
}

/// What a plane holds.
///
/// A gray or monochrome picture is a `Luma` plane. Semi-planar chroma (NV12,
/// P010, NV21) is one plane holding two components per position, in the order
/// the variant names. Planar CMYK holds `Cyan`, `Magenta`, `Yellow` and `Black`
/// planes (ink polarity belongs to [`PlanarLayout::is_inverted_cmyk`], not to
/// separate role variants). Extra channels (`Alpha`, `Depth`, `SelectionMask`,
/// `Thermal`, `SpotColor`, `Cfa`, `Auxiliary`) may accompany any color family
/// or form an extra-only `Mono` layout; callers should query
/// [`PlanarLayout::plane_size`] for every plane's dimensions and use
/// [`PlaneRole::is_alpha`] / [`PlaneRole::is_cmyk`] rather than hardcoding
/// variant equality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlaneRole {
    /// Y, or gray.
    Luma,
    /// Cb (U).
    Cb,
    /// Cr (V).
    Cr,
    /// Cb and Cr interleaved, Cb first (NV12, P010).
    CbCr,
    /// Cr and Cb interleaved, Cr first (NV21).
    CrCb,
    /// R.
    Red,
    /// G.
    Green,
    /// B.
    Blue,
    /// Alpha.
    Alpha,
    /// Cyan ink component in planar CMYK.
    Cyan,
    /// Magenta ink component in planar CMYK.
    Magenta,
    /// Yellow ink component in planar CMYK.
    Yellow,
    /// Black (key) ink component in planar CMYK.
    Black,
    /// Scene or camera depth map extra channel.
    Depth,
    /// Selection or segmentation mask extra channel.
    SelectionMask,
    /// Thermal / infrared radiance extra channel.
    Thermal,
    /// Spot-color ink extra channel (`0..=255`).
    SpotColor(u8),
    /// Bayer / color-filter-array sensor sub-channel (`0..=255`).
    Cfa(u8),
    /// Codec-generic auxiliary extra channel (`0..=255`, e.g. JPEG XL `Optional` / `Unknown`).
    Auxiliary(u8),
}

// Explicit Hash preserves the `isize` discriminant width of the original unit
// variants (`0..=8`) so `portable_v1_golden_hashes` remains bit-for-bit stable
// across compilers and enum representation changes.
impl core::hash::Hash for PlaneRole {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        let (tag, index): (isize, Option<u8>) = match *self {
            Self::Luma => (0, None),
            Self::Cb => (1, None),
            Self::Cr => (2, None),
            Self::CbCr => (3, None),
            Self::CrCb => (4, None),
            Self::Red => (5, None),
            Self::Green => (6, None),
            Self::Blue => (7, None),
            Self::Alpha => (8, None),
            Self::Cyan => (9, None),
            Self::Magenta => (10, None),
            Self::Yellow => (11, None),
            Self::Black => (12, None),
            Self::Depth => (13, None),
            Self::SelectionMask => (14, None),
            Self::Thermal => (15, None),
            Self::SpotColor(i) => (16, Some(i)),
            Self::Cfa(i) => (17, Some(i)),
            Self::Auxiliary(i) => (18, Some(i)),
        };
        state.write_isize(tag);
        if let Some(i) = index {
            state.write_u8(i);
        }
    }
}

impl PlaneRole {
    /// Samples stored per position: 2 for interleaved chroma, else 1. A plane
    /// holding this role is `positions * components()` samples wide.
    pub const fn components(self) -> usize {
        match self {
            Self::CbCr | Self::CrCb => 2,
            _ => 1,
        }
    }

    /// True when this role is a primary color component (`Luma`/chroma, `RGB`, or `CMYK`).
    pub const fn is_color(self) -> bool {
        !self.is_extra()
    }

    /// True when this role is an alpha plane (today [`Self::Alpha`]; future
    /// minor releases may add indexed secondary alpha variants).
    ///
    /// Prefer `role.is_alpha()` over `role == PlaneRole::Alpha`.
    pub const fn is_alpha(self) -> bool {
        matches!(self, Self::Alpha)
    }

    /// True when this role is one of the four planar CMYK ink components.
    ///
    /// Prefer `role.is_cmyk()` combined with [`PlanarLayout::is_inverted_cmyk`]
    /// rather than expecting polarity-specific role variants.
    pub const fn is_cmyk(self) -> bool {
        matches!(
            self,
            Self::Cyan | Self::Magenta | Self::Yellow | Self::Black
        )
    }

    /// True when this role is an extra channel (`Alpha`, `Depth`, `SelectionMask`,
    /// `Thermal`, `SpotColor`, `Cfa`, or `Auxiliary`).
    pub const fn is_extra(self) -> bool {
        matches!(
            self,
            Self::Alpha
                | Self::Depth
                | Self::SelectionMask
                | Self::Thermal
                | Self::SpotColor(_)
                | Self::Cfa(_)
                | Self::Auxiliary(_)
        )
    }

    const fn is_chroma(self) -> bool {
        matches!(self, Self::Cb | Self::Cr | Self::CbCr | Self::CrCb)
    }

    const fn is_rgb(self) -> bool {
        matches!(self, Self::Red | Self::Green | Self::Blue)
    }
}

/// A float encoding is not finite, or its scale is zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FloatEncodingError {
    /// Scale or offset is NaN or infinite.
    NotFinite,
    /// Scale is zero.
    ZeroScale,
}

impl fmt::Display for FloatEncodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NotFinite => "float plane scale and offset must be finite",
            Self::ZeroScale => "float plane scale must not be zero",
        })
    }
}
impl core::error::Error for FloatEncodingError {}

/// How float samples relate to nominal component values.
///
/// `stored = nominal * scale + offset`. Nominal luma, red, green, blue and
/// alpha run from 0 to 1, and nominal Cb and Cr from -0.5 to 0.5, all full
/// range; stored values may lie outside. For example, zenjpeg's centered
/// planes are `scale = 255` with `offset = -128` for Y and `offset = 0` for
/// Cb and Cr. A decoder that keeps limited-range levels in floats states them
/// here too, so a float plane never needs a separate range.
#[derive(Clone, Copy, Debug)]
pub struct FloatEncoding {
    scale: f32,
    offset: f32,
}

impl FloatEncoding {
    /// Stored values are the nominal values.
    pub const NOMINAL: Self = Self {
        scale: 1.0,
        offset: 0.0,
    };

    /// Stored values are `nominal * scale + offset`.
    ///
    /// # Errors
    ///
    /// [`FloatEncodingError::NotFinite`] for a NaN or infinite value,
    /// [`FloatEncodingError::ZeroScale`] for a zero scale.
    pub fn new(scale: f32, offset: f32) -> Result<Self, FloatEncodingError> {
        if !scale.is_finite() || !offset.is_finite() {
            return Err(FloatEncodingError::NotFinite);
        }
        if scale == 0.0 {
            return Err(FloatEncodingError::ZeroScale);
        }
        // One representation of zero, so equal encodings compare equal.
        let offset = if offset == 0.0 { 0.0 } else { offset };
        Ok(Self { scale, offset })
    }

    /// Stored units per nominal unit.
    pub const fn scale(self) -> f32 {
        self.scale
    }

    /// The stored value of nominal zero.
    pub const fn offset(self) -> f32 {
        self.offset
    }
}

impl PartialEq for FloatEncoding {
    fn eq(&self, other: &Self) -> bool {
        self.scale.to_bits() == other.scale.to_bits()
            && self.offset.to_bits() == other.offset.to_bits()
    }
}
impl Eq for FloatEncoding {}
impl core::hash::Hash for FloatEncoding {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.scale.to_bits().hash(state);
        self.offset.to_bits().hash(state);
    }
}

/// How a plane's samples encode component values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PlaneEncoding {
    /// Integer codes in native signed or unsigned storage; range comes from
    /// the enclosing signal. Storage support does not imply numerical conversion.
    Code(SampleEncoding),
    /// Floating-point samples (`f32`); the encoding describes values, not the
    /// sample width.
    Float(FloatEncoding),
    /// Native F64 samples, with the same explicit affine interpretation.
    /// Scale and offset are described by [`FloatEncoding`]; samples are not narrowed.
    Float64(FloatEncoding),
    /// Native F16 ([`F16Sample`]) samples, with the same explicit affine
    /// interpretation (`scale` and `offset` in [`FloatEncoding`]).
    Float16(FloatEncoding),
}

impl PlaneEncoding {
    /// The sample type this encoding is stored in.
    pub const fn channel_type(self) -> ChannelType {
        match self {
            Self::Code(e) => e.storage(),
            Self::Float(_) => ChannelType::F32,
            Self::Float64(_) => ChannelType::F64,
            Self::Float16(_) => ChannelType::F16,
        }
    }
}

/// One plane of a [`PlanarLayout`]: what it holds and how.
///
/// Extra-channel planes (`Alpha`, `Depth`, `SelectionMask`, `Thermal`,
/// `SpotColor`, `Cfa`, `Auxiliary`) may optionally carry power-of-two horizontal
/// and vertical downsampling shifts (`0..=3`, i.e. 1× to 1/8× resolution,
/// rounded up at odd image edges) via [`PlaneSpec::with_subsampling_shift`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneSpec {
    role: PlaneRole,
    encoding: PlaneEncoding,
    h_shift: u8,
    v_shift: u8,
}

// Explicit Hash preserves exact v1 golden hashes for unshifted (`0, 0`) specs
// while including non-zero subsampling shifts when present.
impl core::hash::Hash for PlaneSpec {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.role.hash(state);
        self.encoding.hash(state);
        if self.h_shift != 0 || self.v_shift != 0 {
            state.write_u8(self.h_shift);
            state.write_u8(self.v_shift);
        }
    }
}

impl PlaneSpec {
    /// A full-resolution (`h_shift = 0, v_shift = 0`) plane holding `role`,
    /// encoded as `encoding`. Chroma planes derive their subsampling from the
    /// enclosing [`PlanarLayout`].
    pub const fn new(role: PlaneRole, encoding: PlaneEncoding) -> Self {
        Self {
            role,
            encoding,
            h_shift: 0,
            v_shift: 0,
        }
    }

    /// Attach power-of-two horizontal and vertical subsampling shifts (`0..=3`,
    /// meaning `width.div_ceil(1 << h_shift)` by `height.div_ceil(1 << v_shift)`)
    /// to an extra-channel plane (`role.is_extra()`, e.g. HEIC/AVIF half-res
    /// depth/gain-map/alpha or JPEG XL `dim_shift`).
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::Subsampling`] if `h_shift > 3`, `v_shift > 3`, or
    /// either shift is non-zero on a primary color plane (`Luma`, chroma, `RGB`,
    /// or `CMYK`, whose subsampling belongs to [`PlanarLayout::subsampling`]).
    pub const fn with_subsampling_shift(
        mut self,
        h_shift: u8,
        v_shift: u8,
    ) -> Result<Self, LayoutError> {
        if h_shift > 3 || v_shift > 3 {
            return Err(LayoutError::Subsampling);
        }
        if (h_shift != 0 || v_shift != 0) && !self.role.is_extra() {
            return Err(LayoutError::Subsampling);
        }
        self.h_shift = h_shift;
        self.v_shift = v_shift;
        Ok(self)
    }

    /// What the plane holds.
    pub const fn role(self) -> PlaneRole {
        self.role
    }

    /// How its samples encode values.
    pub const fn encoding(self) -> PlaneEncoding {
        self.encoding
    }

    /// Horizontal power-of-two subsampling shift (`0..=3`) for extra-channel
    /// planes; always `0` for primary color and chroma planes.
    pub const fn h_shift(self) -> u8 {
        self.h_shift
    }

    /// Vertical power-of-two subsampling shift (`0..=3`) for extra-channel
    /// planes; always `0` for primary color and chroma planes.
    pub const fn v_shift(self) -> u8 {
        self.v_shift
    }
}

/// A layout is inconsistent, or planes do not match it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LayoutError {
    /// A real Alpha plane requires Straight, Premultiplied or Opaque; otherwise
    /// the interpretation must be absent. Undefined is padding, not alpha.
    AlphaInterpretation,
    /// The layout has no planes.
    Empty,
    /// A component appears in more than one plane.
    DuplicateRole(PlaneRole),
    /// Primary color families (YCbCr, RGB, CMYK) are mixed.
    MixedFamilies,
    /// A component the others require is missing (Luma with chroma, one of
    /// R, G, B, or one of C, M, Y, K).
    MissingRole(PlaneRole),
    /// Subsampling disagrees with the planes: chroma planes need a
    /// subsampling other than `Mono`, luma-only and extra-only layouts need
    /// `Mono`, planar RGB or CMYK needs `Yuv444`, and per-plane subsampling
    /// shifts (`0..=3`) apply only to extra-channel planes.
    Subsampling,
    /// Inverted CMYK ink polarity was requested on a non-CMYK layout.
    InkPolarity,
    /// The number of planes differs from the layout.
    PlaneCount {
        /// Planes in the layout.
        expected: usize,
        /// Planes supplied.
        actual: usize,
    },
    /// A plane index is outside this layout.
    PlaneIndex {
        /// Requested plane.
        index: usize,
        /// Planes in the layout.
        count: usize,
    },
    /// A plane's sample type differs from its encoding.
    SampleType {
        /// Plane index.
        index: usize,
    },
    /// A plane's size differs from what the layout gives for the picture.
    Dimensions {
        /// Plane index.
        index: usize,
        /// Samples per row and rows the layout requires.
        expected: (usize, usize),
        /// Samples per row and rows supplied.
        actual: (usize, usize),
    },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlphaInterpretation => {
                f.write_str("planar Alpha requires Straight, Premultiplied or Opaque; without Alpha, interpretation must be absent")
            }
            Self::Empty => f.write_str("planar layout has no planes"),
            Self::DuplicateRole(r) => write!(f, "planar layout holds {r:?} twice"),
            Self::MixedFamilies => {
                f.write_str("planar layout mixes incompatible color families (YCbCr, RGB, CMYK)")
            }
            Self::MissingRole(r) => write!(f, "planar layout lacks {r:?}"),
            Self::Subsampling => f.write_str("chroma or extra-plane subsampling disagrees with the planes"),
            Self::InkPolarity => {
                f.write_str("inverted CMYK polarity requires a planar CMYK layout")
            }
            Self::PlaneCount { expected, actual } => {
                write!(f, "layout has {expected} planes, {actual} supplied")
            }
            Self::PlaneIndex { index, count } => {
                write!(f, "plane {index} is outside a {count}-plane layout")
            }
            Self::SampleType { index } => {
                write!(f, "plane {index} sample type differs from its encoding")
            }
            Self::Dimensions {
                index,
                expected,
                actual,
            } => write!(
                f,
                "plane {index} is {}x{} samples, layout requires {}x{}",
                actual.0, actual.1, expected.0, expected.1
            ),
        }
    }
}
impl core::error::Error for LayoutError {}

/// The planes of a planar picture, with chroma subsampling and siting.
///
/// Plane order is the order of [`PlanarLayout::planes`]; it is not implied by
/// the roles. Color signaling travels with the enclosing descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanarLayout {
    planes: Vec<PlaneSpec>,
    subsampling: ChromaSubsampling,
    siting: ChromaSiting,
    inverted_cmyk: bool,
}

// Explicit Hash preserves exact v1 golden hashes for conventional layouts while
// distinguishing inverted-CMYK polarity when set.
impl core::hash::Hash for PlanarLayout {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.planes.hash(state);
        self.subsampling.hash(state);
        self.siting.hash(state);
        if self.inverted_cmyk {
            state.write_u8(1);
        }
    }
}

impl PlanarLayout {
    /// Check and adopt a plane list.
    ///
    /// Primary color families are mutually exclusive:
    /// - **YCbCr / Gray**: `Luma`, optionally both `Cb` and `Cr` (separately or
    ///   in one `CbCr`/`CrCb` plane).
    /// - **RGB**: `Red`, `Green`, and `Blue` with `Yuv444`.
    /// - **CMYK**: `Cyan`, `Magenta`, `Yellow`, and `Black` with `Yuv444`
    ///   (defaults to conventional ink polarity; chain
    ///   [`PlanarLayout::with_inverted_cmyk`] for inverted `0 = full ink` CMYK).
    ///
    /// Extra roles (`Alpha`, `Depth`, `SelectionMask`, `Thermal`,
    /// `SpotColor(i)`, `Cfa(i)`, `Auxiliary(i)`) may accompany any primary
    /// family or stand alone with `Mono` subsampling, and may carry per-plane
    /// subsampling shifts via [`PlaneSpec::with_subsampling_shift`].
    ///
    /// **Forward-compatibility notes for `0.3.x` consumers**:
    /// - Always query [`PlanarLayout::plane_size`] for every plane rather than
    ///   assuming non-chroma / extra planes match `(width, height)`.
    /// - Today [`PlanarLayout::new`] rejects duplicate roles, but future minor
    ///   constructors may permit multiple extra channels (e.g. secondary alphas,
    ///   masks, or depth planes); use [`PlaneRole::is_alpha`] and iterate
    ///   [`PlanarLayout::planes`] rather than assuming at most one extra channel.
    /// - Planar CMYK ink polarity belongs to [`PlanarLayout::is_inverted_cmyk`],
    ///   keeping [`PlaneRole`] purely semantic.
    ///
    /// # Errors
    ///
    /// A [`LayoutError`] naming the first inconsistency.
    pub fn new(
        planes: Vec<PlaneSpec>,
        subsampling: ChromaSubsampling,
        siting: ChromaSiting,
    ) -> Result<Self, LayoutError> {
        if planes.is_empty() {
            return Err(LayoutError::Empty);
        }
        let has = |r: PlaneRole| planes.iter().any(|p| p.role == r);
        let (mut cb, mut cr) = (false, false);
        for (i, p) in planes.iter().enumerate() {
            if planes[..i].iter().any(|q| q.role == p.role) {
                return Err(LayoutError::DuplicateRole(p.role));
            }
            let (holds_cb, holds_cr) = match p.role {
                PlaneRole::Cb => (true, false),
                PlaneRole::Cr => (false, true),
                PlaneRole::CbCr | PlaneRole::CrCb => (true, true),
                _ => (false, false),
            };
            if (holds_cb && cb) || (holds_cr && cr) {
                return Err(LayoutError::DuplicateRole(p.role));
            }
            cb |= holds_cb;
            cr |= holds_cr;
        }
        let rgb = planes.iter().any(|p| p.role.is_rgb());
        let cmyk = planes.iter().any(|p| p.role.is_cmyk());
        let ycbcr = has(PlaneRole::Luma) || cb || cr;
        if u8::from(rgb) + u8::from(cmyk) + u8::from(ycbcr) > 1 {
            return Err(LayoutError::MixedFamilies);
        }
        if rgb {
            for r in [PlaneRole::Red, PlaneRole::Green, PlaneRole::Blue] {
                if !has(r) {
                    return Err(LayoutError::MissingRole(r));
                }
            }
            if subsampling != ChromaSubsampling::Yuv444 {
                return Err(LayoutError::Subsampling);
            }
        } else if cmyk {
            for r in [
                PlaneRole::Cyan,
                PlaneRole::Magenta,
                PlaneRole::Yellow,
                PlaneRole::Black,
            ] {
                if !has(r) {
                    return Err(LayoutError::MissingRole(r));
                }
            }
            if subsampling != ChromaSubsampling::Yuv444 {
                return Err(LayoutError::Subsampling);
            }
        } else if ycbcr {
            if !has(PlaneRole::Luma) {
                return Err(LayoutError::MissingRole(PlaneRole::Luma));
            }
            if cb != cr {
                let missing = if cb { PlaneRole::Cr } else { PlaneRole::Cb };
                return Err(LayoutError::MissingRole(missing));
            }
            if cb == (subsampling == ChromaSubsampling::Mono) {
                return Err(LayoutError::Subsampling);
            }
        } else if subsampling != ChromaSubsampling::Mono {
            // Extra-only layout (e.g. lone Alpha or Depth/Mask planes) has no chroma.
            return Err(LayoutError::Subsampling);
        }
        Ok(Self {
            planes,
            subsampling,
            siting,
            inverted_cmyk: false,
        })
    }

    /// Set whether a planar CMYK layout uses inverted ink polarity
    /// (`0 = full ink`, max/`1.0` = no ink, as in JPEG XL and Adobe APP14 CMYK)
    /// instead of conventional ICC ink polarity (`0 = no ink`).
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::InkPolarity`] if `inverted` is `true` on a layout
    /// whose primary color family is not planar CMYK.
    pub fn with_inverted_cmyk(mut self, inverted: bool) -> Result<Self, LayoutError> {
        if inverted && !self.planes.iter().any(|p| p.role.is_cmyk()) {
            return Err(LayoutError::InkPolarity);
        }
        self.inverted_cmyk = inverted;
        Ok(self)
    }

    /// The planes, in storage order.
    pub fn planes(&self) -> &[PlaneSpec] {
        &self.planes
    }

    /// Roles, in storage order.
    pub fn roles(&self) -> impl ExactSizeIterator<Item = PlaneRole> + Clone + '_ {
        self.planes().iter().map(|p| p.role)
    }

    /// Chroma subsampling; `Mono` without chroma, `Yuv444` for planar RGB.
    pub fn subsampling(&self) -> ChromaSubsampling {
        self.subsampling
    }

    /// Chroma siting.
    pub fn siting(&self) -> ChromaSiting {
        self.siting
    }

    /// Whether planar CMYK ink planes use inverted polarity (`0 = full ink`,
    /// max/1.0 = no ink, as in JPEG XL / Adobe APP14 CMYK) rather than
    /// conventional ICC ink polarity (`0 = no ink`).
    ///
    /// Defaults to `false`; configure with [`PlanarLayout::with_inverted_cmyk`].
    pub const fn is_inverted_cmyk(&self) -> bool {
        self.inverted_cmyk
    }

    /// Positions (not samples) per row and rows of plane `index` in a
    /// `width` x `height` picture; `None` for an index past the last plane.
    /// Subsampled chroma and shifted extra-channel planes are rounded up at odd
    /// edges; multiply the width by [`PlaneRole::components`] for samples per row.
    pub fn plane_size(&self, index: usize, width: u32, height: u32) -> Option<(u32, u32)> {
        let spec = self.planes().get(index)?;
        if spec.role.is_chroma() {
            self.subsampling.chroma_size(width, height)
        } else {
            Some((
                width.div_ceil(1u32 << spec.h_shift),
                height.div_ceil(1u32 << spec.v_shift),
            ))
        }
    }

    /// Check that `planes` hold this layout for a `width` x `height` picture:
    /// one plane per spec, in order, each with its encoding's sample type and
    /// exactly the layout's samples per row and rows.
    ///
    /// # Errors
    ///
    /// A [`LayoutError`] naming the first plane that does not match.
    pub fn check(
        &self,
        planes: &[AnyPlane<'_>],
        width: u32,
        height: u32,
    ) -> Result<(), LayoutError> {
        if planes.len() != self.planes.len() {
            return Err(LayoutError::PlaneCount {
                expected: self.planes.len(),
                actual: planes.len(),
            });
        }
        for (index, plane) in planes.iter().enumerate() {
            self.check_plane(index, plane, width, height)?;
        }
        Ok(())
    }

    /// Validate one plane's type and dimensions without assembling every view.
    pub fn check_plane(
        &self,
        index: usize,
        plane: &AnyPlane<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), LayoutError> {
        let spec = self.planes().get(index).ok_or(LayoutError::PlaneIndex {
            index,
            count: self.planes.len(),
        })?;
        if plane.channel_type() != spec.encoding.channel_type() {
            return Err(LayoutError::SampleType { index });
        }
        let (w, h) = self
            .plane_size(index, width, height)
            .expect("validated plane index");
        let expected_width = usize::try_from(w)
            .ok()
            .and_then(|w| w.checked_mul(spec.role.components()));
        let expected = (expected_width.unwrap_or(usize::MAX), h as usize);
        let actual = (plane.width(), plane.height());
        if expected_width.is_none() || actual != expected {
            return Err(LayoutError::Dimensions {
                index,
                expected,
                actual,
            });
        }
        Ok(())
    }
}

/// A borrowed, color-described planar picture using standard plane roles,
/// encodings, strides and subsampling. Construction checks storage and layout;
/// it does not convert, normalize or infer color. Empty pictures are allowed
/// here; consumers decide their minimum geometry.
///
/// Integer payload depth and placement use [`SampleEncoding`]; read supported
/// native components through [`crate::sample::CodePlane`]. A representable
/// encoding does not establish support for reading or processing that storage.
/// Color signaling is carried verbatim: unknown CICP, authoritative ICC and
/// missing luminance anchors are not normalized or replaced. In particular,
/// an HLG tag does not select [`crate::hdr::HlgRendering`]; a metric or converter
/// supplies any required display conditions in its own configuration.
#[derive(Clone, Copy, Debug)]
pub struct PlanarSlice<'a> {
    planes: Planes<'a>,
    layout: &'a PlanarLayout,
    width: u32,
    height: u32,
    color: &'a crate::ColorContext,
    alpha: Option<crate::AlphaMode>,
}
impl<'a> PlanarSlice<'a> {
    /// Borrow a complete picture, preserving independently strided planes.
    /// `color` describes these samples, including matrix/range for YCbCr and
    /// an explicit luminance anchor when relative and absolute light interact.
    /// `alpha` is required exactly when the layout contains an alpha plane.
    /// A real alpha plane requires Straight, Premultiplied or Opaque;
    /// Undefined describes padding and is not an alpha-plane interpretation.
    ///
    /// # Errors
    /// Returns a layout, storage, geometry or alpha-interpretation mismatch.
    pub fn new(
        planes: &'a [AnyPlane<'a>],
        layout: &'a PlanarLayout,
        width: u32,
        height: u32,
        color: &'a crate::ColorContext,
        alpha: Option<crate::AlphaMode>,
    ) -> Result<Self, LayoutError> {
        let planes = Planes::Borrowed(planes);
        planes.check(layout, width, height, alpha)?;
        Ok(Self {
            planes,
            layout,
            width,
            height,
            color,
            alpha,
        })
    }
    /// Borrowed native storage, in layout order. No allocation or pixel copy.
    pub fn planes(
        self,
    ) -> impl ExactSizeIterator<Item = AnyPlane<'a>> + DoubleEndedIterator + Clone {
        (0..self.planes.len()).map(move |i| self.planes.get(i).expect("in bounds"))
    }
    /// Borrow one plane by its layout index; `None` past the last plane.
    pub fn plane(self, index: usize) -> Option<AnyPlane<'a>> {
        self.planes.get(index)
    }
    /// Role, sample encoding, subsampling and siting of each plane.
    pub const fn layout(self) -> &'a PlanarLayout {
        self.layout
    }
    /// Complete image width, before any chroma subsampling.
    pub const fn width(self) -> u32 {
        self.width
    }
    /// Complete image height.
    pub const fn height(self) -> u32 {
        self.height
    }
    /// Color and luminance signaling for the actual samples.
    pub const fn color_context(self) -> &'a crate::ColorContext {
        self.color
    }
    /// Declared alpha interpretation.
    pub const fn alpha_mode(self) -> Option<crate::AlphaMode> {
        self.alpha
    }
}

/// A writable planar picture over concrete plane owners or borrowed views.
///
/// Each plane keeps its own stride. Access reborrows samples without exposing
/// replaceable descriptors. The owner type `P` retains any backing-sample
/// lifetime independently of the short picture borrow `'a`.
#[derive(Debug)]
pub struct PlanarSliceMut<'a, P = AnyPlaneMut<'a>> {
    planes: &'a mut [P],
    layout: &'a PlanarLayout,
    width: u32,
    height: u32,
    color: &'a crate::ColorContext,
    alpha: Option<crate::AlphaMode>,
}
impl<'a, P: PlaneStorageMut + Sync> PlanarSliceMut<'a, P> {
    /// Check and borrow writable plane owners or views without allocation.
    /// Validation matches [`PlanarSlice::new`]. No color is inferred.
    pub fn new(
        planes: &'a mut [P],
        layout: &'a PlanarLayout,
        width: u32,
        height: u32,
        color: &'a crate::ColorContext,
        alpha: Option<crate::AlphaMode>,
    ) -> Result<Self, LayoutError> {
        Planes::Owner(&planes).check(layout, width, height, alpha)?;
        Ok(Self {
            planes,
            layout,
            width,
            height,
            color,
            alpha,
        })
    }
    /// Read the complete picture without allocation or sample copies.
    pub fn as_slice(&self) -> PlanarSlice<'_> {
        PlanarSlice {
            planes: Planes::Owner(&self.planes),
            layout: self.layout,
            width: self.width,
            height: self.height,
            color: self.color,
            alpha: self.alpha,
        }
    }
    /// Reborrow one writable plane without exposing a replaceable descriptor.
    pub fn plane_mut(&mut self, index: usize) -> Option<AnyPlaneMut<'_>> {
        self.planes
            .get_mut(index)
            .map(PlaneStorageMut::as_plane_mut)
    }
    /// Lend disjoint writable planes simultaneously, without allocating.
    pub fn planes_mut(&mut self) -> impl ExactSizeIterator<Item = AnyPlaneMut<'_>> {
        self.planes.iter_mut().map(PlaneStorageMut::as_plane_mut)
    }
}

/// An owned planar picture, including layout and current color interpretation.
///
/// The default owner is [`AnyPlaneBuf`], an enum of native typed vectors.
/// Use `PlanarBuffer<Plane<C>>` for typed aligned, pooled or shared storage,
/// or implement [`PlaneStorage`] for a codec's heterogeneous owner enum.
/// Borrowing creates neither sample nor descriptor allocations. Layout remains
/// Vec-backed; there is no fixed-capacity descriptor array. Frame timing belongs
/// to the codec or media layer, not this pixel container. Owners must be `Sync`
/// to lend the shared, thread-safe [`PlanarSlice`]. `Send` is not required.
///
/// ```
/// use zenpixels::{ChannelType, Cicp, ColorContext, plane::*, sample::*};
///
/// // A decoder can move its existing Y/Cb/Cr allocations into these planes.
/// let encoding = PlaneEncoding::Code(SampleEncoding::new(ChannelType::U8, 8, 0)?);
/// let layout = PlanarLayout::new(
///     [PlaneRole::Luma, PlaneRole::Cb, PlaneRole::Cr]
///         .map(|role| PlaneSpec::new(role, encoding)).to_vec(),
///     ChromaSubsampling::Yuv420, ChromaSiting::Center,
/// )?;
/// let mut picture = PlanarBuffer::new(vec![
///     AnyPlaneBuf::U8(PlaneBuf::new(vec![16; 15], 5, 3, 5)?),
///     AnyPlaneBuf::U8(PlaneBuf::new(vec![128; 6], 3, 2, 3)?),
///     AnyPlaneBuf::U8(PlaneBuf::new(vec![128; 6], 3, 2, 3)?),
/// ], layout, 5, 3, ColorContext::from_cicp(Cicp::new(1, 13, 6, true)), None)?;
/// {
///     let mut writable = picture.as_slice_mut();
///     if let Some(AnyPlaneMut::U8(mut y)) = writable.plane_mut(0) {
///         y.row_mut(0).unwrap().fill(32);
///     }
/// }
/// assert_eq!(picture.as_slice().planes().len(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct PlanarBuffer<P = AnyPlaneBuf> {
    planes: Vec<P>,
    layout: PlanarLayout,
    width: u32,
    height: u32,
    color: crate::ColorContext,
    alpha: Option<crate::AlphaMode>,
}
impl<P: PlaneStorage + Sync> PlanarBuffer<P> {
    /// Check and adopt existing allocations and interpretation. On failure,
    /// the supplied owned values are dropped. This constructor allocates nothing.
    /// Validation matches [`PlanarSlice::new`], including subsampled odd edges.
    pub fn new(
        planes: Vec<P>,
        layout: PlanarLayout,
        width: u32,
        height: u32,
        color: crate::ColorContext,
        alpha: Option<crate::AlphaMode>,
    ) -> Result<Self, LayoutError> {
        Self::try_from_parts(PlanarBufferParts::new(
            planes, layout, width, height, color, alpha,
        ))
        .map_err(PlanarFromPartsError::without_buffer)
    }

    /// Validate and adopt plane owners, retaining all storage and interpretation
    /// on failure. No allocation or sample copy occurs on either path.
    #[allow(clippy::result_large_err)] // Retain rejected owners inline, without allocating an error box.
    pub fn try_from_parts(parts: PlanarBufferParts<P>) -> Result<Self, PlanarFromPartsError<P>> {
        if let Err(error) = Planes::Owner(&parts.planes).check(
            &parts.layout,
            parts.width,
            parts.height,
            parts.alpha,
        ) {
            return Err(PlanarFromPartsError {
                error,
                parts: Some(parts),
            });
        }
        let PlanarBufferParts {
            planes,
            layout,
            width,
            height,
            color,
            alpha,
        } = parts;
        Ok(Self {
            planes,
            layout,
            width,
            height,
            color,
            alpha,
        })
    }

    /// Move every owner and its interpretation out without cloning or copying.
    pub fn into_parts(self) -> PlanarBufferParts<P> {
        PlanarBufferParts::new(
            self.planes,
            self.layout,
            self.width,
            self.height,
            self.color,
            self.alpha,
        )
    }

    /// Read the complete picture without allocating or copying samples.
    pub fn as_slice(&self) -> PlanarSlice<'_> {
        PlanarSlice {
            planes: Planes::Owner(&self.planes),
            layout: &self.layout,
            width: self.width,
            height: self.height,
            color: &self.color,
            alpha: self.alpha,
        }
    }

    /// Write samples without allocating, copying, or exposing replaceable
    /// descriptors. Interpretation is retained verbatim; mutation does not
    /// perform a color conversion or establish precision evidence.
    pub fn as_slice_mut(&mut self) -> PlanarSliceMut<'_, P>
    where
        P: PlaneStorageMut,
    {
        PlanarSliceMut {
            planes: &mut self.planes,
            layout: &self.layout,
            width: self.width,
            height: self.height,
            color: &self.color,
            alpha: self.alpha,
        }
    }

    /// Recover the plane allocations, discarding the picture interpretation.
    /// No samples are copied. Read any needed interpretation before consuming.
    pub fn into_planes(self) -> Vec<P> {
        self.planes
    }
}

/// Plane owners and complete picture interpretation, transferred together.
/// This record is unvalidated; use [`PlanarBuffer::try_from_parts`] after editing.
#[derive(Debug)]
#[non_exhaustive]
pub struct PlanarBufferParts<P = AnyPlaneBuf> {
    /// Plane owners in layout order.
    pub planes: Vec<P>,
    /// Roles, sample encodings, chroma subsampling and siting.
    pub layout: PlanarLayout,
    /// Complete image width, before subsampling.
    pub width: u32,
    /// Complete image height.
    pub height: u32,
    /// Current sample interpretation, retained verbatim.
    pub color: crate::ColorContext,
    /// Alpha interpretation, present exactly when an Alpha role exists.
    pub alpha: Option<crate::AlphaMode>,
}
impl<P> PlanarBufferParts<P> {
    /// Assemble an unvalidated transfer record without allocation or copying.
    pub fn new(
        planes: Vec<P>,
        layout: PlanarLayout,
        width: u32,
        height: u32,
        color: crate::ColorContext,
        alpha: Option<crate::AlphaMode>,
    ) -> Self {
        Self {
            planes,
            layout,
            width,
            height,
            color,
            alpha,
        }
    }
}

/// Failed planar adoption retaining all original owners and interpretation.
/// Use `take_parts()` for recovery, or `without_buffer()` before retaining an
/// error when its potentially large image storage is no longer needed.
#[must_use]
pub struct PlanarFromPartsError<P = AnyPlaneBuf> {
    error: LayoutError,
    parts: Option<PlanarBufferParts<P>>,
}
impl<P> PlanarFromPartsError<P> {
    /// The validation failure.
    pub fn error(&self) -> &LayoutError {
        &self.error
    }
    /// Recover the original transfer record; `None` after the first call.
    pub fn take_parts(&mut self) -> Option<PlanarBufferParts<P>> {
        self.parts.take()
    }
    /// Discard the retained owners and return only the validation failure.
    pub fn without_buffer(self) -> LayoutError {
        self.error
    }
}
impl<P> fmt::Debug for PlanarFromPartsError<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlanarFromPartsError")
            .field("error", &self.error)
            .field("has_parts", &self.parts.is_some())
            .finish()
    }
}
impl<P> fmt::Display for PlanarFromPartsError<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}
impl<P> core::error::Error for PlanarFromPartsError<P> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.error)
    }
}
