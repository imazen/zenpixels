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
use crate::sample::{ChromaSiting, ChromaSubsampling, SampleEncoding};
use alloc::{boxed::Box, vec::Vec};
use core::fmt;

/// The shared native storage sample contract. This is the same sealed trait
/// used by [`crate::buffer::PixelStorage`], not a separate plane-only roster.
pub use crate::buffer::StorageSample as PlaneSample;

/// Read access to the samples of a plane's owner.
///
/// Implement this for an owner type (for example accounted shared storage) to
/// use it as `Plane<YourOwner>`. `samples` must return the same slice on every
/// call: geometry is checked once, at construction, and an owner whose samples
/// shrink afterwards makes row access panic.
///
/// Slices, references, boxes and `Vec` implement it, and the trait stays
/// dyn-compatible, so an owner can also be hidden as
/// `Plane<Box<dyn SampleStorage<Sample = T> + Send + Sync>>`. Planes of one
/// picture normally share one owner type per sample type; borrowed views
/// ([`Plane::view`], [`AnyPlane`]) erase the owner at no cost.
pub trait SampleStorage {
    /// The sample type.
    type Sample: PlaneSample;
    /// The samples, including any padding.
    fn samples(&self) -> &[Self::Sample];
}

/// Write access to the samples of a plane's owner.
pub trait SampleStorageMut: SampleStorage {
    /// The samples, including any padding.
    fn samples_mut(&mut self) -> &mut [Self::Sample];
}

impl<T: PlaneSample> SampleStorage for [T] {
    type Sample = T;
    fn samples(&self) -> &[T] {
        self
    }
}
impl<T: PlaneSample> SampleStorageMut for [T] {
    fn samples_mut(&mut self) -> &mut [T] {
        self
    }
}
impl<S: SampleStorage + ?Sized> SampleStorage for &S {
    type Sample = S::Sample;
    fn samples(&self) -> &[S::Sample] {
        (**self).samples()
    }
}
impl<S: SampleStorage + ?Sized> SampleStorage for &mut S {
    type Sample = S::Sample;
    fn samples(&self) -> &[S::Sample] {
        (**self).samples()
    }
}
impl<S: SampleStorageMut + ?Sized> SampleStorageMut for &mut S {
    fn samples_mut(&mut self) -> &mut [S::Sample] {
        (**self).samples_mut()
    }
}
// `Vec` and `Box` storage. With the `allocator_api` feature on Rust 1.100 or newer
// these cover every allocator, which includes the global one, so existing uses
// keep compiling; before 1.100 the feature changes nothing.
macro_rules! owned_storage {
    ($($alloc:ident)?) => {
        impl<S: SampleStorage + ?Sized $(, $alloc: core::alloc::Allocator)?> SampleStorage
            for Box<S $(, $alloc)?>
        {
            type Sample = S::Sample;
            fn samples(&self) -> &[S::Sample] {
                (**self).samples()
            }
        }
        impl<S: SampleStorageMut + ?Sized $(, $alloc: core::alloc::Allocator)?> SampleStorageMut
            for Box<S $(, $alloc)?>
        {
            fn samples_mut(&mut self) -> &mut [S::Sample] {
                (**self).samples_mut()
            }
        }
        impl<T: PlaneSample $(, $alloc: core::alloc::Allocator)?> SampleStorage
            for Vec<T $(, $alloc)?>
        {
            type Sample = T;
            fn samples(&self) -> &[T] {
                self
            }
        }
        impl<T: PlaneSample $(, $alloc: core::alloc::Allocator)?> SampleStorageMut
            for Vec<T $(, $alloc)?>
        {
            fn samples_mut(&mut self) -> &mut [T] {
                self
            }
        }
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
                if self.start == 0 {
                    Ok(self.buf)
                } else {
                    Err(self)
                }
            }
        }
    };
}
#[cfg(not(feature = "allocator_api"))]
owned_storage!();
#[cfg(feature = "allocator_api")]
#[rustversion::before(1.100)]
owned_storage!();
#[cfg(feature = "allocator_api")]
#[rustversion::since(1.100)]
owned_storage!(A);

/// One allocation unit of [`AlignedSamples`]. Its alignment is what keeps the
/// buffer aligned without an offset, a custom allocator, or `unsafe`: `Vec`
/// allocates and frees it with this layout.
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C, align(64))]
struct Chunk([u8; 64]);

/// Owned samples whose first element is on a 64-byte boundary, on any allocator.
///
/// 64 bytes covers every interleaved pixel type, so the samples can always be
/// viewed as `&[Rgb<u16>]`, `&[Rgba<f32>]` and the like with `bytemuck`, and index
/// 0 is always the first sample. Clones stay aligned. It derefs to `[T]`, so it
/// also serves as an `image::ImageBuffer` container.
///
/// The price is that it cannot become a `Vec<T>` without a copy: a `Vec` has to
/// free memory with the layout it allocated. Adopt a `Vec<T>` directly when the
/// next owner needs one.
#[derive(Clone)]
pub struct AlignedSamples<T> {
    chunks: Vec<Chunk>,
    len: usize,
    _sample: core::marker::PhantomData<T>,
}

impl<T: PlaneSample> AlignedSamples<T> {
    /// Allocate `len` zeroed samples, fallibly.
    ///
    /// # Errors
    ///
    /// [`PlaneError::GeometryOverflow`] when the size overflows,
    /// [`PlaneError::AllocationFailed`] when the allocator refuses.
    pub fn try_zeroed(len: usize) -> Result<Self, PlaneError> {
        let chunk_count = len
            .checked_mul(core::mem::size_of::<T>())
            .ok_or(PlaneError::GeometryOverflow)?
            .div_ceil(core::mem::size_of::<Chunk>());
        // Goes through `alloc_zeroed`, so large buffers get untouched zero pages.
        let chunks: Vec<Chunk> = bytemuck::allocation::try_zeroed_vec(chunk_count)
            .map_err(|()| PlaneError::AllocationFailed)?;
        Ok(Self {
            chunks,
            len,
            _sample: core::marker::PhantomData,
        })
    }
}

impl<T: PlaneSample> core::ops::Deref for AlignedSamples<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &bytemuck::cast_slice(&self.chunks)[..self.len]
    }
}

impl<T: PlaneSample> core::ops::DerefMut for AlignedSamples<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut bytemuck::cast_slice_mut(&mut self.chunks)[..self.len]
    }
}

impl<T> fmt::Debug for AlignedSamples<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AlignedSamples(len: {})", self.len)
    }
}

impl<T: PlaneSample> SampleStorage for AlignedSamples<T> {
    type Sample = T;
    fn samples(&self) -> &[T] {
        self
    }
}

impl<T: PlaneSample> SampleStorageMut for AlignedSamples<T> {
    fn samples_mut(&mut self) -> &mut [T] {
        self
    }
}

/// A plane's geometry does not fit its storage, or a request leaves the plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlaneError {
    /// A nonempty plane's stride is shorter than its row.
    ShortStride,
    /// Storage ends before the last active sample.
    ShortBuffer,
    /// Computing the plane's extent overflows addressable storage.
    GeometryOverflow,
    /// A requested rectangle or row extends beyond the plane.
    OutsidePlane,
    /// Allocating plane storage failed.
    AllocationFailed,
}

impl fmt::Display for PlaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ShortStride => "plane stride is shorter than its row",
            Self::ShortBuffer => "plane storage ends before the last active sample",
            Self::GeometryOverflow => "plane geometry overflows addressable storage",
            Self::OutsidePlane => "request extends beyond the plane",
            Self::AllocationFailed => "plane allocation failed",
        })
    }
}
impl core::error::Error for PlaneError {}

/// A strided two-dimensional array of samples, generic over its owner.
///
/// `width` and `stride` count samples, not bytes, so a stride always holds whole
/// samples. A plane holding two interleaved components (see
/// [`PlaneRole::CbCr`]) is `components * positions` samples wide. The last row
/// need not have padding after it.
#[derive(Clone, Copy)]
pub struct Plane<C> {
    buf: C,
    start: usize,
    width: usize,
    height: usize,
    stride: usize,
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plane")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

/// Samples from the first active sample through the last one.
fn extent(start: usize, width: usize, height: usize, stride: usize) -> Result<usize, PlaneError> {
    if width == 0 || height == 0 {
        return Ok(start);
    }
    if stride < width {
        return Err(PlaneError::ShortStride);
    }
    (height - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(width))
        .and_then(|n| n.checked_add(start))
        .ok_or(PlaneError::GeometryOverflow)
}

impl<C> Plane<C> {
    /// The index of row 0 in the storage.
    #[cfg(feature = "image")]
    pub(crate) fn samples_start(&self) -> usize {
        self.start
    }
}

impl<C> Plane<C> {
    /// Recover the original storage owner without copying samples, discarding
    /// plane geometry. Read dimensions/stride first if the next owner needs them.
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
    pub fn new(buf: C, width: usize, height: usize, stride: usize) -> Result<Self, PlaneError> {
        Self::with_start(buf, 0, width, height, stride)
    }

    fn with_start(
        buf: C,
        start: usize,
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, PlaneError> {
        if extent(start, width, height, stride)? > buf.samples().len() {
            return Err(PlaneError::ShortBuffer);
        }
        Ok(Self {
            buf,
            start,
            width,
            height,
            stride,
        })
    }

    /// Samples per row.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Rows.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Samples from one row start to the next.
    pub fn stride(&self) -> usize {
        self.stride
    }

    // First sample of row `y`. An empty row starts at `start`, which is
    // within the storage even when `y * stride` is not.
    fn row_start(&self, y: usize) -> usize {
        if self.width == 0 {
            self.start
        } else {
            self.start + y * self.stride
        }
    }

    /// Row `y`, without padding; `None` past the last row.
    pub fn row(&self, y: usize) -> Option<&[C::Sample]> {
        if y >= self.height {
            return None;
        }
        let s = self.row_start(y);
        Some(&self.buf.samples()[s..s + self.width])
    }

    /// Every row, top to bottom, without padding.
    pub fn rows(&self) -> impl DoubleEndedIterator<Item = &[C::Sample]> + ExactSizeIterator {
        (0..self.height).map(move |y| {
            let s = self.row_start(y);
            &self.buf.samples()[s..s + self.width]
        })
    }

    /// The samples from the first active sample through the last one, including
    /// padding between rows but not after the last row. Empty for an empty
    /// plane. For handing the plane to an API that takes a slice and a stride.
    pub fn strided_samples(&self) -> &[C::Sample] {
        let end = extent(self.start, self.width, self.height, self.stride)
            .expect("checked at construction");
        &self.buf.samples()[self.start..end]
    }

    /// Borrow this plane.
    pub fn view(&self) -> PlaneRef<'_, C::Sample> {
        Plane {
            buf: self.strided_samples(),
            start: 0,
            width: self.width,
            height: self.height,
            stride: self.stride,
        }
    }

    /// Borrow the `width` x `height` samples at column `x`, row `y`.
    ///
    /// # Errors
    ///
    /// [`PlaneError::OutsidePlane`] when the rectangle leaves the plane.
    pub fn sub(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Result<PlaneRef<'_, C::Sample>, PlaneError> {
        let (start, width, height) = self.sub_rect(x, y, width, height)?;
        Plane::with_start(self.buf.samples(), start, width, height, self.stride)
    }

    fn sub_rect(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Result<(usize, usize, usize), PlaneError> {
        let fits = |o: usize, n: usize, max: usize| o.checked_add(n).is_some_and(|e| e <= max);
        if !fits(x, width, self.width) || !fits(y, height, self.height) {
            return Err(PlaneError::OutsidePlane);
        }
        if width == 0 || height == 0 {
            return Ok((self.start, width, height));
        }
        Ok((self.start + y * self.stride + x, width, height))
    }
}

impl<C: SampleStorageMut> Plane<C> {
    /// Row `y`, without padding; `None` past the last row.
    pub fn row_mut(&mut self, y: usize) -> Option<&mut [C::Sample]> {
        if y >= self.height {
            return None;
        }
        let s = self.row_start(y);
        Some(&mut self.buf.samples_mut()[s..s + self.width])
    }

    /// Every row, top to bottom, without padding.
    pub fn rows_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = &mut [C::Sample]> + ExactSizeIterator {
        let (width, height, stride) = (self.width, self.height, self.stride);
        let end = extent(self.start, width, height, stride).expect("checked at construction");
        let samples = &mut self.buf.samples_mut()[self.start..end];
        RowsMut {
            samples,
            width,
            stride,
            rows: height,
        }
    }

    /// Mutably borrow this plane.
    pub fn view_mut(&mut self) -> PlaneMut<'_, C::Sample> {
        let (width, height, stride) = (self.width, self.height, self.stride);
        let end = extent(self.start, width, height, stride).expect("checked at construction");
        Plane {
            buf: &mut self.buf.samples_mut()[self.start..end],
            start: 0,
            width,
            height,
            stride,
        }
    }

    /// Mutably borrow the `width` x `height` samples at column `x`, row `y`.
    ///
    /// # Errors
    ///
    /// [`PlaneError::OutsidePlane`] when the rectangle leaves the plane.
    pub fn sub_mut(
        &mut self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Result<PlaneMut<'_, C::Sample>, PlaneError> {
        let (start, width, height) = self.sub_rect(x, y, width, height)?;
        let stride = self.stride;
        Plane::with_start(self.buf.samples_mut(), start, width, height, stride)
    }
}

impl<T: PlaneSample> Plane<&mut [T]> {
    /// Split into rows `0..y` and rows `y..`, for filling strips or
    /// handing disjoint rows to threads.
    ///
    /// # Errors
    ///
    /// [`PlaneError::OutsidePlane`] when `y` is past the last row.
    pub fn split_at_row(self, y: usize) -> Result<(Self, Self), PlaneError> {
        if y > self.height {
            return Err(PlaneError::OutsidePlane);
        }
        let Plane {
            buf,
            start,
            width,
            height,
            stride,
        } = self;
        let at = if y == 0 || width == 0 {
            start
        } else if y == height {
            // The final row need not include its trailing stride padding.
            buf.len()
        } else {
            start + y * stride
        };
        let (top, bottom) = buf.split_at_mut(at);
        Ok((
            Plane {
                buf: top,
                start,
                width,
                height: y,
                stride,
            },
            Plane {
                buf: bottom,
                start: 0,
                width,
                height: height - y,
                stride,
            },
        ))
    }
}

impl<'a, T: PlaneSample> Plane<&'a [T]> {
    /// [`Plane::strided_samples`] for the borrowed lifetime, so an adapter can
    /// hand the samples to another crate's borrowed frame type (slice from the
    /// first sample, stride in samples) without copying.
    pub fn into_strided_samples(self) -> &'a [T] {
        let end = extent(self.start, self.width, self.height, self.stride)
            .expect("checked at construction");
        &self.buf[self.start..end]
    }
}

impl<'a, T: PlaneSample> Plane<&'a mut [T]> {
    /// The mutable samples from the first active sample through the last one,
    /// for the borrowed lifetime; see [`into_strided_samples`](Plane::into_strided_samples).
    pub fn into_strided_samples_mut(self) -> &'a mut [T] {
        let end = extent(self.start, self.width, self.height, self.stride)
            .expect("checked at construction");
        &mut self.buf[self.start..end]
    }
}

/// Converting to imgref: dimensions past `u32::MAX`, which imgref cannot hold,
/// are [`PlaneError::GeometryOverflow`]. An empty plane's zero stride becomes 1,
/// which imgref requires.
#[cfg(feature = "imgref")]
fn imgref_geometry(width: usize, height: usize, stride: usize) -> Result<usize, PlaneError> {
    if u32::try_from(width).is_err() || u32::try_from(height).is_err() {
        return Err(PlaneError::GeometryOverflow);
    }
    Ok(stride.max(1))
}

/// Borrow an [`imgref::ImgRef`] of samples as a plane, without copying.
///
/// # Errors
///
/// A [`PlaneError`] when the image's buffer is shorter than its geometry.
#[cfg(feature = "imgref")]
impl<'a, T: PlaneSample> TryFrom<imgref::ImgRef<'a, T>> for PlaneRef<'a, T> {
    type Error = PlaneError;
    fn try_from(img: imgref::ImgRef<'a, T>) -> Result<Self, PlaneError> {
        Plane::new(img.buf(), img.width(), img.height(), img.stride())
    }
}

/// Borrow an [`imgref::ImgRefMut`] of samples as a plane, without copying.
#[cfg(feature = "imgref")]
impl<'a, T: PlaneSample> TryFrom<imgref::ImgRefMut<'a, T>> for PlaneMut<'a, T> {
    type Error = PlaneError;
    fn try_from(img: imgref::ImgRefMut<'a, T>) -> Result<Self, PlaneError> {
        let (width, height, stride) = (img.width(), img.height(), img.stride());
        Plane::new(img.into_buf(), width, height, stride)
    }
}

/// Adopt an [`imgref::ImgVec`]'s `Vec` as a plane, without copying.
#[cfg(feature = "imgref")]
impl<T: PlaneSample> TryFrom<imgref::ImgVec<T>> for PlaneBuf<T> {
    type Error = PlaneError;
    fn try_from(img: imgref::ImgVec<T>) -> Result<Self, PlaneError> {
        let (width, height, stride) = (img.width(), img.height(), img.stride());
        Plane::new(img.into_buf(), width, height, stride)
    }
}

/// Borrow a plane as an [`imgref::ImgRef`], without copying.
#[cfg(feature = "imgref")]
impl<'a, T: PlaneSample> TryFrom<PlaneRef<'a, T>> for imgref::ImgRef<'a, T> {
    type Error = PlaneError;
    fn try_from(plane: PlaneRef<'a, T>) -> Result<Self, PlaneError> {
        let stride = imgref_geometry(plane.width, plane.height, plane.stride)?;
        let (width, height) = (plane.width, plane.height);
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
    fn try_from(plane: PlaneMut<'a, T>) -> Result<Self, PlaneError> {
        let stride = imgref_geometry(plane.width, plane.height, plane.stride)?;
        let (width, height) = (plane.width, plane.height);
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

/// Row starts on 64 bytes, the widest SIMD vector and the x86/Neoverse line.
const ROW_ALIGN: usize = 64;

/// Row-start alignment policy for [`Plane::alloc_aligned`] and
/// [`crate::PixelBuffer::try_new_aligned`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RowAlign {
    /// Tightly packed rows (`stride == width` samples, or `width * bytes_per_pixel` bytes).
    #[default]
    Tight,
    /// 64-byte row alignment (widest SIMD vector and x86/ARM cache line), plus
    /// a 64-byte anti-aliasing bump when the aligned row is a non-zero multiple
    /// of 4096 bytes (`benchmarks/stride_aliasing_2026-11-01.md`).
    ///
    /// For 3-channel packed formats such as `RGB8` (3 B/px) or `RGB16` (6 B/px),
    /// byte stride is a multiple of 64 (and of `size_of::<T>()`), not necessarily
    /// a multiple of `bytes_per_pixel`.
    Cache64,
    /// Whole-pixel 64-byte row alignment (`lcm(bytes_per_pixel, 64)`), plus one
    /// alignment step when the aligned row is a non-zero multiple of 4096 bytes.
    ///
    /// Every row starts on both a 64-byte boundary and a whole-pixel boundary,
    /// satisfying whole-pixel stride consumers such as `imgref`. On a single-sample
    /// [`Plane`], this equals [`Self::Cache64`].
    PixelCache64,
    /// Align row stride to a multiple of `N` samples (`0` is treated as `1`,
    /// i.e. tight), with no 4096-byte bump.
    ///
    /// Use [`Self::from_bytes`] or [`Self::try_from_bytes`] to translate a
    /// hardware byte-pitch requirement (such as `wgpu`'s 256-byte row alignment).
    Samples(u16),
}

const _: () = assert!(core::mem::size_of::<RowAlign>() == 4);

impl RowAlign {
    /// Convert a hardware byte-pitch requirement into a sample-count alignment
    /// for sample type `T`.
    ///
    /// # Panics
    ///
    /// Panics (at compile time in `const` context) if `bytes == 0` or `bytes`
    /// is not a multiple of `size_of::<T>()`.
    #[inline]
    pub const fn from_bytes<T: PlaneSample>(bytes: u16) -> Self {
        match Self::try_from_bytes(bytes, core::mem::size_of::<T>()) {
            Some(align) => align,
            None => panic!("RowAlign::from_bytes requires a non-zero multiple of size_of::<T>()"),
        }
    }

    /// Fallibly convert a byte-pitch alignment into `RowAlign::Samples` for a
    /// given `sample_bytes` width (`1`, `2`, `4`, or `8`).
    ///
    /// Returns `None` if `bytes == 0`, `sample_bytes == 0`, or `bytes` is not
    /// divisible by `sample_bytes`.
    #[inline]
    pub const fn try_from_bytes(bytes: u16, sample_bytes: usize) -> Option<Self> {
        if bytes == 0 || sample_bytes == 0 || (bytes as usize) % sample_bytes != 0 {
            None
        } else {
            Some(Self::Samples((bytes as usize / sample_bytes) as u16))
        }
    }
}

impl<T: PlaneSample> Plane<AlignedSamples<T>> {
    /// Allocate a zeroed `width` x `height` plane for SIMD kernels.
    ///
    /// Equivalent to `Self::alloc_aligned(width, height, RowAlign::Cache64)`:
    /// every row starts on a 64-byte boundary, with row 0 at index 0. The stride
    /// is the row rounded up to 64 bytes, plus 64 more when that is a multiple of
    /// 4096: such strides put a column's rows in one cache set, which cost
    /// cache-resident 12-row vertical filters 19-48% on Zen 4
    /// (`benchmarks/stride_aliasing_2026-11-01.md` in the zenpixels repo). The
    /// allocation is fallible, and clones keep the alignment.
    ///
    /// # Errors
    ///
    /// [`PlaneError::GeometryOverflow`] when the size overflows,
    /// [`PlaneError::AllocationFailed`] when the allocator refuses.
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
    pub fn alloc_aligned(width: usize, height: usize, align: RowAlign) -> Result<Self, PlaneError> {
        let size = core::mem::size_of::<T>();
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
    fn try_from(plane: PlaneBuf<T>) -> Result<Self, PlaneBuf<T>> {
        let Ok(stride) = imgref_geometry(plane.width, plane.height, plane.stride) else {
            return Err(plane);
        };
        let (width, height) = (plane.width, plane.height);
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

impl AnyPlaneMut<'_> {
    /// Borrow without allocating or copying samples.
    pub fn as_ref(&self) -> AnyPlane<'_> {
        match self {
            Self::U8(p) => AnyPlane::U8(p.view()),
            Self::I8(p) => AnyPlane::I8(p.view()),
            Self::U16(p) => AnyPlane::U16(p.view()),
            Self::I16(p) => AnyPlane::I16(p.view()),
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
        if layout.roles().any(|r| r == PlaneRole::Alpha) != alpha.is_some()
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
/// the variant names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
}

impl PlaneEncoding {
    /// The sample type this encoding is stored in.
    pub const fn channel_type(self) -> ChannelType {
        match self {
            Self::Code(e) => e.storage(),
            Self::Float(_) => ChannelType::F32,
            Self::Float64(_) => ChannelType::F64,
        }
    }
}

/// One plane of a [`PlanarLayout`]: what it holds and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaneSpec {
    role: PlaneRole,
    encoding: PlaneEncoding,
}

impl PlaneSpec {
    /// A plane holding `role`, encoded as `encoding`.
    pub const fn new(role: PlaneRole, encoding: PlaneEncoding) -> Self {
        Self { role, encoding }
    }

    /// What the plane holds.
    pub const fn role(self) -> PlaneRole {
        self.role
    }

    /// How its samples encode values.
    pub const fn encoding(self) -> PlaneEncoding {
        self.encoding
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
    /// YCbCr and RGB components are mixed.
    MixedFamilies,
    /// A component the others require is missing (Luma with chroma, or one
    /// of R, G, B).
    MissingRole(PlaneRole),
    /// Subsampling disagrees with the planes: chroma planes need a
    /// subsampling other than `Mono`, a luma-only layout needs `Mono`, and
    /// planar RGB needs `Yuv444`.
    Subsampling,
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
            Self::MixedFamilies => f.write_str("planar layout mixes YCbCr and RGB components"),
            Self::MissingRole(r) => write!(f, "planar layout lacks {r:?}"),
            Self::Subsampling => f.write_str("chroma subsampling disagrees with the planes"),
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlanarLayout {
    planes: Vec<PlaneSpec>,
    subsampling: ChromaSubsampling,
    siting: ChromaSiting,
}

impl PlanarLayout {
    /// Check and adopt a plane list.
    ///
    /// Each component appears once. YCbCr layouts hold Luma, optionally both
    /// Cb and Cr (separately or in one interleaved plane), and optionally
    /// Alpha; RGB layouts hold Red, Green, Blue and optionally Alpha; an alpha-only
    /// layout holds Alpha alone, with `Mono` subsampling.
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
        let ycbcr = has(PlaneRole::Luma) || cb || cr;
        if rgb && ycbcr {
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
        } else {
            // A lone alpha plane is a valid image (`image`'s `A8`); anything else needs luma.
            let alpha_only = planes.len() == 1 && has(PlaneRole::Alpha);
            if !has(PlaneRole::Luma) && !alpha_only {
                return Err(LayoutError::MissingRole(PlaneRole::Luma));
            }
            if cb != cr {
                let missing = if cb { PlaneRole::Cr } else { PlaneRole::Cb };
                return Err(LayoutError::MissingRole(missing));
            }
            if cb == (subsampling == ChromaSubsampling::Mono) {
                return Err(LayoutError::Subsampling);
            }
        }
        Ok(Self {
            planes,
            subsampling,
            siting,
        })
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

    /// Positions (not samples) per row and rows of plane `index` in a
    /// `width` x `height` picture; `None` for an index past the last plane.
    /// Chroma planes are rounded up; multiply the width by
    /// [`PlaneRole::components`] for samples per row.
    pub fn plane_size(&self, index: usize, width: u32, height: u32) -> Option<(u32, u32)> {
        let spec = self.planes().get(index)?;
        if spec.role.is_chroma() {
            self.subsampling.chroma_size(width, height)
        } else {
            Some((width, height))
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
