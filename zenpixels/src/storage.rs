//! Unconditional sample-owner traits, 64-byte aligned owner, and row-stride alignment policy.
//!
//! Shared by packed [`crate::PixelBuffer`] (always available) and planar
//! `zenpixels::plane` (when `feature = "planar"` is enabled).

use alloc::{boxed::Box, vec::Vec};
use core::fmt;

/// The shared native storage sample contract. This is the same sealed trait
/// used by [`crate::buffer::PixelStorage`], not a separate plane-only roster.
pub use crate::buffer::StorageSample as PlaneSample;

/// Read access to the samples of a buffer or plane owner.
///
/// Implement this for an owner type (for example accounted shared storage) to
/// use it as `PixelBuffer<(), YourOwner>` or `Plane<YourOwner>`. `samples` must
/// return the same slice on every call: geometry is checked once, at construction.
///
/// Slices, references, boxes and `Vec` implement it, and the trait stays
/// dyn-compatible.
pub trait SampleStorage {
    /// The sample type.
    type Sample: PlaneSample;
    /// The samples, including any padding.
    fn samples(&self) -> &[Self::Sample];
}

/// Write access to the samples of a buffer or plane owner.
pub trait SampleStorageMut: SampleStorage {
    /// The samples, including any padding.
    fn samples_mut(&mut self) -> &mut [Self::Sample];
}

impl<T: PlaneSample> SampleStorage for [T] {
    type Sample = T;
    #[inline]
    fn samples(&self) -> &[T] {
        self
    }
}
impl<T: PlaneSample> SampleStorageMut for [T] {
    #[inline]
    fn samples_mut(&mut self) -> &mut [T] {
        self
    }
}
impl<S: SampleStorage + ?Sized> SampleStorage for &S {
    type Sample = S::Sample;
    #[inline]
    fn samples(&self) -> &[S::Sample] {
        (**self).samples()
    }
}
impl<S: SampleStorage + ?Sized> SampleStorage for &mut S {
    type Sample = S::Sample;
    #[inline]
    fn samples(&self) -> &[S::Sample] {
        (**self).samples()
    }
}
impl<S: SampleStorageMut + ?Sized> SampleStorageMut for &mut S {
    #[inline]
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
            #[inline]
            fn samples(&self) -> &[S::Sample] {
                (**self).samples()
            }
        }
        impl<S: SampleStorageMut + ?Sized $(, $alloc: core::alloc::Allocator)?> SampleStorageMut
            for Box<S $(, $alloc)?>
        {
            #[inline]
            fn samples_mut(&mut self) -> &mut [S::Sample] {
                (**self).samples_mut()
            }
        }
        impl<T: PlaneSample $(, $alloc: core::alloc::Allocator)?> SampleStorage
            for Vec<T $(, $alloc)?>
        {
            type Sample = T;
            #[inline]
            fn samples(&self) -> &[T] {
                self
            }
        }
        impl<T: PlaneSample $(, $alloc: core::alloc::Allocator)?> SampleStorageMut
            for Vec<T $(, $alloc)?>
        {
            #[inline]
            fn samples_mut(&mut self) -> &mut [T] {
                self
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
pub(crate) struct Chunk([u8; 64]);

fn try_zeroed_chunks(len: usize, sample_size: usize) -> Result<Vec<Chunk>, PlaneError> {
    let chunk_count = len
        .checked_mul(sample_size)
        .ok_or(PlaneError::GeometryOverflow)?
        .div_ceil(core::mem::size_of::<Chunk>());
    bytemuck::allocation::try_zeroed_vec(chunk_count).map_err(|()| PlaneError::AllocationFailed)
}

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
    #[inline]
    pub fn try_zeroed(len: usize) -> Result<Self, PlaneError> {
        let chunks = try_zeroed_chunks(len, core::mem::size_of::<T>())?;
        Ok(Self {
            chunks,
            len,
            _sample: core::marker::PhantomData,
        })
    }
}

impl<T: PlaneSample> core::ops::Deref for AlignedSamples<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        &bytemuck::cast_slice(&self.chunks)[..self.len]
    }
}

impl<T: PlaneSample> core::ops::DerefMut for AlignedSamples<T> {
    #[inline]
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
    #[inline]
    fn samples(&self) -> &[T] {
        self
    }
}

impl<T: PlaneSample> SampleStorageMut for AlignedSamples<T> {
    #[inline]
    fn samples_mut(&mut self) -> &mut [T] {
        self
    }
}

/// A plane or aligned-owner geometry does not fit its storage, or a request leaves the bounds.
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

/// Row starts on 64 bytes, the widest SIMD vector and the x86/Neoverse line.
#[cfg(feature = "planar")]
pub(crate) const ROW_ALIGN: usize = 64;

/// Row-start alignment policy for [`crate::PixelBuffer::try_new_aligned`],
/// [`crate::PixelBuffer::try_new_with_align`], and `Plane::alloc_aligned`.
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
    /// plane, this equals [`Self::Cache64`].
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
