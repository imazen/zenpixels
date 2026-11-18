use alloc::vec::Vec;
use core::fmt;

use crate::descriptor::{ChannelType, PixelDescriptor};
use crate::pixel_types::F16Sample;
use crate::storage::{SampleStorage, SampleStorageMut};

use super::{BufferError, align_offset};

/// Storage a [`PixelBuffer`](super::PixelBuffer) can read as bytes.
///
/// Everything past this reference works on bytes, so code generic over storage
/// stops being generic here. Any [`SampleStorage`] implements it; so does
/// [`PixelStorage`], which holds one of several sample types.
pub trait ByteStorage {
    /// The samples as native-endian bytes, without copying.
    fn as_bytes(&self) -> &[u8];
}

/// Storage a [`PixelBuffer`](super::PixelBuffer) can write as bytes.
pub trait ByteStorageMut: ByteStorage {
    /// The samples as mutable native-endian bytes, without copying.
    fn as_bytes_mut(&mut self) -> &mut [u8];
}

impl<S: SampleStorage + ?Sized> ByteStorage for S {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self.samples())
    }
}

impl<S: SampleStorageMut + ?Sized> ByteStorageMut for S {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        bytemuck::cast_slice_mut(self.samples_mut())
    }
}

pub(crate) mod storage_sample {
    use super::{PixelStorage, Vec};

    /// Proves a call comes from this module, because a generic caller could otherwise
    /// reach these methods through a `StorageSample` bound.
    pub struct Private(pub(crate) ());

    pub trait Sealed: Sized {
        fn into_storage(samples: Vec<Self>, _: Private) -> PixelStorage;
        fn from_storage(storage: PixelStorage, _: Private) -> Result<Vec<Self>, PixelStorage>;
        #[cfg(feature = "image")]
        fn held_by(storage: &PixelStorage, _: Private) -> bool;
        #[cfg(feature = "planar")]
        fn plane_view(
            plane: crate::plane::PlaneRef<'_, Self>,
            _: Private,
        ) -> crate::plane::AnyPlane<'_>;
        #[cfg(feature = "planar")]
        fn plane_view_mut(
            plane: crate::plane::PlaneMut<'_, Self>,
            _: Private,
        ) -> crate::plane::AnyPlaneMut<'_>;
    }
}
use storage_sample::Private;

/// A sample type that [`PixelStorage`] can hold. Sealed.
///
/// Its conversions are private: use [`PixelStorage::from`] and the `PixelStorage` methods.
///
/// ```compile_fail
/// fn store<T: zenpixels::buffer::StorageSample>(samples: Vec<T>) {
///     let _ = T::into_storage(samples, Private(()));
/// }
/// ```
pub trait StorageSample:
    storage_sample::Sealed
    + bytemuck::Pod
    + Default
    + PartialEq
    + core::fmt::Debug
    + Send
    + Sync
    + 'static
{
    /// Native sample type shared by typed storage, planes and runtime owners.
    const CHANNEL_TYPE: ChannelType;
}

/// Owned samples of the type a pixel format is made of.
///
/// This is the default storage of [`PixelBuffer`](super::PixelBuffer), so a decoder's `Vec<u16>`,
/// `Vec<F16Sample>`, or `Vec<f32>` crosses type-erased boundaries such as
/// `DecodeOutput` and comes back out without a copy. A `Vec` must free memory
/// with the alignment it allocated, so a `Vec<u8>` can never become a `Vec<u16>`,
/// or the reverse, without copying; holding the real sample type avoids that.
///
/// Bytes are always available through [`ByteStorage`]. Only that reference branches
/// on the variant, so `row()` and kernels are unaffected.
#[non_exhaustive]
pub enum PixelStorage {
    /// `u8` samples, or bytes adopted without knowing the sample type.
    U8(Vec<u8>),
    /// `i8` samples.
    I8(Vec<i8>),
    /// `u16` samples.
    U16(Vec<u16>),
    /// `i16` samples.
    I16(Vec<i16>),
    /// `f16` (IEEE 754 half-precision float) samples.
    F16(Vec<F16Sample>),
    /// `u32` samples.
    U32(Vec<u32>),
    /// `i32` samples.
    I32(Vec<i32>),
    /// `f32` samples.
    F32(Vec<f32>),
    /// `u64` samples.
    U64(Vec<u64>),
    /// `i64` samples.
    I64(Vec<i64>),
    /// `f64` samples.
    F64(Vec<f64>),
}

macro_rules! for_each_storage {
    ($value:expr, $vec:ident => $body:expr) => {
        match $value {
            crate::buffer::PixelStorage::U8($vec) => $body,
            crate::buffer::PixelStorage::I8($vec) => $body,
            crate::buffer::PixelStorage::U16($vec) => $body,
            crate::buffer::PixelStorage::I16($vec) => $body,
            crate::buffer::PixelStorage::F16($vec) => $body,
            crate::buffer::PixelStorage::U32($vec) => $body,
            crate::buffer::PixelStorage::I32($vec) => $body,
            crate::buffer::PixelStorage::F32($vec) => $body,
            crate::buffer::PixelStorage::U64($vec) => $body,
            crate::buffer::PixelStorage::I64($vec) => $body,
            crate::buffer::PixelStorage::F64($vec) => $body,
        }
    };
}
#[cfg(any(feature = "rgb", feature = "imgref"))]
pub(super) use for_each_storage;

impl Clone for PixelStorage {
    fn clone(&self) -> Self {
        let bytes = self.as_bytes();
        Self::try_from_rows_dyn(
            self.channel_type(),
            bytes.len(),
            &mut core::iter::once(bytes),
        )
        .expect("pixel storage clone allocation failed")
    }
}

macro_rules! storage_samples {
    ($($sample:ty => $variant:ident),* $(,)?) => {$(
        impl storage_sample::Sealed for $sample {
            fn into_storage(samples: Vec<Self>, _: Private) -> PixelStorage {
                PixelStorage::$variant(samples)
            }
            fn from_storage(storage: PixelStorage, _: Private) -> Result<Vec<Self>, PixelStorage> {
                match storage {
                    PixelStorage::$variant(samples) => Ok(samples),
                    other => Err(other),
                }
            }
            #[cfg(feature = "image")]
            fn held_by(storage: &PixelStorage, _: Private) -> bool {
                matches!(storage, PixelStorage::$variant(_))
            }
            #[cfg(feature = "planar")]
            fn plane_view(plane: crate::plane::PlaneRef<'_, Self>, _: Private) -> crate::plane::AnyPlane<'_> {
                crate::plane::AnyPlane::$variant(plane)
            }
            #[cfg(feature = "planar")]
            fn plane_view_mut(plane: crate::plane::PlaneMut<'_, Self>, _: Private) -> crate::plane::AnyPlaneMut<'_> {
                crate::plane::AnyPlaneMut::$variant(plane)
            }
        }
        impl StorageSample for $sample {
            const CHANNEL_TYPE: ChannelType = ChannelType::$variant;
        }
    )*};
}
storage_samples! {
    u8 => U8, i8 => I8, u16 => U16, i16 => I16, F16Sample => F16, u32 => U32,
    i32 => I32, f32 => F32, u64 => U64, i64 => I64, f64 => F64,
}

impl<T: StorageSample> From<Vec<T>> for PixelStorage {
    /// Adopt the `Vec` without copying.
    fn from(samples: Vec<T>) -> Self {
        T::into_storage(samples, Private(()))
    }
}

#[inline]
fn cast_same_align<A: bytemuck::NoUninit, B: bytemuck::AnyBitPattern>(v: Vec<A>) -> Vec<B> {
    match bytemuck::allocation::try_cast_vec(v) {
        Ok(out) => out,
        Err(_) => panic!("same-size same-align sample cast failed"),
    }
}

fn gather_u8(bytes: usize, rows: &mut dyn Iterator<Item = &[u8]>) -> Result<Vec<u8>, BufferError> {
    let mut samples: Vec<u8> = Vec::new();
    samples
        .try_reserve_exact(bytes)
        .map_err(|_| BufferError::AllocationFailed)?;
    for row in rows {
        samples.extend_from_slice(row);
    }
    Ok(samples)
}

fn gather_u16(
    bytes: usize,
    rows: &mut dyn Iterator<Item = &[u8]>,
) -> Result<Vec<u16>, BufferError> {
    let mut samples: Vec<u16> = Vec::new();
    samples
        .try_reserve_exact(bytes / 2)
        .map_err(|_| BufferError::AllocationFailed)?;
    for row in rows {
        samples.extend_from_slice(bytemuck::cast_slice(row));
    }
    Ok(samples)
}

fn gather_u32(
    bytes: usize,
    rows: &mut dyn Iterator<Item = &[u8]>,
) -> Result<Vec<u32>, BufferError> {
    let mut samples: Vec<u32> = Vec::new();
    samples
        .try_reserve_exact(bytes / 4)
        .map_err(|_| BufferError::AllocationFailed)?;
    for row in rows {
        samples.extend_from_slice(bytemuck::cast_slice(row));
    }
    Ok(samples)
}

fn gather_u64(
    bytes: usize,
    rows: &mut dyn Iterator<Item = &[u8]>,
) -> Result<Vec<u64>, BufferError> {
    let mut samples: Vec<u64> = Vec::new();
    samples
        .try_reserve_exact(bytes / 8)
        .map_err(|_| BufferError::AllocationFailed)?;
    for row in rows {
        samples.extend_from_slice(bytemuck::cast_slice(row));
    }
    Ok(samples)
}

fn gather_f64(
    bytes: usize,
    rows: &mut dyn Iterator<Item = &[u8]>,
) -> Result<Vec<f64>, BufferError> {
    if core::mem::align_of::<u64>() == core::mem::align_of::<f64>() {
        return Ok(cast_same_align(gather_u64(bytes, rows)?));
    }
    let mut samples: Vec<f64> = Vec::new();
    samples
        .try_reserve_exact(bytes / 8)
        .map_err(|_| BufferError::AllocationFailed)?;
    for row in rows {
        samples.extend_from_slice(bytemuck::cast_slice(row));
    }
    Ok(samples)
}

fn zeroed_u8(bytes: usize) -> Result<Vec<u8>, BufferError> {
    bytemuck::allocation::try_zeroed_vec(bytes).map_err(|()| BufferError::AllocationFailed)
}

fn zeroed_u16(bytes: usize) -> Result<Vec<u16>, BufferError> {
    bytemuck::allocation::try_zeroed_vec(bytes.div_ceil(2))
        .map_err(|()| BufferError::AllocationFailed)
}

fn zeroed_u32(bytes: usize) -> Result<Vec<u32>, BufferError> {
    bytemuck::allocation::try_zeroed_vec(bytes.div_ceil(4))
        .map_err(|()| BufferError::AllocationFailed)
}

fn zeroed_u64(bytes: usize) -> Result<Vec<u64>, BufferError> {
    bytemuck::allocation::try_zeroed_vec(bytes.div_ceil(8))
        .map_err(|()| BufferError::AllocationFailed)
}

fn zeroed_f64(bytes: usize) -> Result<Vec<f64>, BufferError> {
    if core::mem::align_of::<u64>() == core::mem::align_of::<f64>() {
        return Ok(cast_same_align(zeroed_u64(bytes)?));
    }
    bytemuck::allocation::try_zeroed_vec(bytes.div_ceil(8))
        .map_err(|()| BufferError::AllocationFailed)
}

impl PixelStorage {
    fn channel_type(&self) -> ChannelType {
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

    /// The `Vec`, when it holds samples of type `T`. Never copies.
    ///
    /// # Errors
    ///
    /// The storage, unchanged, when it holds another sample type.
    pub fn try_into_vec<T: StorageSample>(self) -> Result<Vec<T>, Self> {
        T::from_storage(self, Private(()))
    }

    /// Whether the storage holds samples of type `T`.
    #[cfg(feature = "image")]
    pub(crate) fn holds<T: StorageSample>(&self) -> bool {
        T::held_by(self, Private(()))
    }

    /// The samples as a `Vec` of pixels, moving the allocation.
    ///
    /// # Errors
    ///
    /// The storage, unchanged, unless the alignments match and the capacity is a
    /// whole number of pixels.
    #[cfg(feature = "imgref")]
    pub(super) fn try_cast_pixels<P: bytemuck::Pod>(self) -> Result<Vec<P>, Self> {
        use bytemuck::allocation::try_cast_vec;
        match core::mem::align_of::<P>() {
            1 => match self {
                Self::U8(v) => try_cast_vec::<u8, P>(v).map_err(|(_, v)| Self::U8(v)),
                Self::I8(v) => try_cast_vec::<u8, P>(cast_same_align(v))
                    .map_err(|(_, v)| Self::I8(cast_same_align(v))),
                other => Err(other),
            },
            2 => match self {
                Self::U16(v) => try_cast_vec::<u16, P>(v).map_err(|(_, v)| Self::U16(v)),
                Self::I16(v) => try_cast_vec::<u16, P>(cast_same_align(v))
                    .map_err(|(_, v)| Self::I16(cast_same_align(v))),
                Self::F16(v) => try_cast_vec::<u16, P>(cast_same_align(v))
                    .map_err(|(_, v)| Self::F16(cast_same_align(v))),
                other => Err(other),
            },
            4 => match self {
                Self::U32(v) => try_cast_vec::<u32, P>(v).map_err(|(_, v)| Self::U32(v)),
                Self::I32(v) => try_cast_vec::<u32, P>(cast_same_align(v))
                    .map_err(|(_, v)| Self::I32(cast_same_align(v))),
                Self::F32(v) => try_cast_vec::<u32, P>(cast_same_align(v))
                    .map_err(|(_, v)| Self::F32(cast_same_align(v))),
                other => Err(other),
            },
            _ => {
                for_each_storage!(self, v => match try_cast_vec::<_, P>(v) {
                    Ok(pixels) => Ok(pixels),
                    Err((_, v)) => Err(v.into()),
                })
            }
        }
    }

    /// Storage of `channel`'s sample type holding `rows` appended in order. `rows` must
    /// be aligned for that sample type and total `bytes` bytes. Nothing is zeroed first.
    #[inline]
    pub(super) fn try_from_rows<'r>(
        channel: ChannelType,
        bytes: usize,
        mut rows: impl Iterator<Item = &'r [u8]>,
    ) -> Result<Self, BufferError> {
        Self::try_from_rows_dyn(channel, bytes, &mut rows)
    }

    fn try_from_rows_dyn(
        channel: ChannelType,
        bytes: usize,
        rows: &mut dyn Iterator<Item = &[u8]>,
    ) -> Result<Self, BufferError> {
        match channel {
            ChannelType::I8 => Ok(Self::I8(cast_same_align(gather_u8(bytes, rows)?))),
            ChannelType::U16 => Ok(Self::U16(gather_u16(bytes, rows)?)),
            ChannelType::I16 => Ok(Self::I16(cast_same_align(gather_u16(bytes, rows)?))),
            ChannelType::F16 => Ok(Self::F16(cast_same_align(gather_u16(bytes, rows)?))),
            ChannelType::U32 => Ok(Self::U32(gather_u32(bytes, rows)?)),
            ChannelType::I32 => Ok(Self::I32(cast_same_align(gather_u32(bytes, rows)?))),
            ChannelType::F32 => Ok(Self::F32(cast_same_align(gather_u32(bytes, rows)?))),
            ChannelType::U64 => Ok(Self::U64(gather_u64(bytes, rows)?)),
            ChannelType::I64 => Ok(Self::I64(cast_same_align(gather_u64(bytes, rows)?))),
            ChannelType::F64 => Ok(Self::F64(gather_f64(bytes, rows)?)),
            _ => Ok(Self::U8(gather_u8(bytes, rows)?)),
        }
    }

    /// Allocate zeroed storage of `channel`'s sample type holding `bytes` bytes.
    pub(super) fn try_zeroed(channel: ChannelType, bytes: usize) -> Result<Self, BufferError> {
        if bytes > isize::MAX as usize {
            return Err(BufferError::AllocationFailed);
        }
        match channel {
            ChannelType::I8 => Ok(Self::I8(cast_same_align(zeroed_u8(bytes)?))),
            ChannelType::U16 => Ok(Self::U16(zeroed_u16(bytes)?)),
            ChannelType::I16 => Ok(Self::I16(cast_same_align(zeroed_u16(bytes)?))),
            ChannelType::F16 => Ok(Self::F16(cast_same_align(zeroed_u16(bytes)?))),
            ChannelType::U32 => Ok(Self::U32(zeroed_u32(bytes)?)),
            ChannelType::I32 => Ok(Self::I32(cast_same_align(zeroed_u32(bytes)?))),
            ChannelType::F32 => Ok(Self::F32(cast_same_align(zeroed_u32(bytes)?))),
            ChannelType::U64 => Ok(Self::U64(zeroed_u64(bytes)?)),
            ChannelType::I64 => Ok(Self::I64(cast_same_align(zeroed_u64(bytes)?))),
            ChannelType::F64 => Ok(Self::F64(zeroed_f64(bytes)?)),
            _ => Ok(Self::U8(zeroed_u8(bytes)?)),
        }
    }

    /// Drop samples past `bytes`, rounded up to a whole sample. Keeps the allocation.
    pub(super) fn truncate_bytes(&mut self, bytes: usize) {
        for_each_storage!(self, v => {
            let size = size_of_element(v);
            v.truncate(bytes.div_ceil(size));
        });
    }

    /// Allocated capacity in bytes.
    #[cfg(any(test, feature = "imgref"))]
    pub(crate) fn capacity_bytes(&self) -> usize {
        for_each_storage!(self, v => v.capacity() * size_of_element(v))
    }

    /// Alignment of the stored sample type.
    #[cfg(feature = "imgref")]
    pub(super) fn element_align(&self) -> usize {
        for_each_storage!(self, v => align_of_element(v))
    }
}

/// Zeroed storage for `bytes` bytes of `descriptor`'s samples, first pixel aligned to
/// `align`. A sample-typed `Vec` already is unless `align` is larger, which only a
/// custom over-aligned pixel type asks for.
pub(super) fn alloc_storage(
    descriptor: &PixelDescriptor,
    bytes: usize,
    align: usize,
) -> Result<(PixelStorage, u32), BufferError> {
    if align <= descriptor.min_alignment() {
        return Ok((
            PixelStorage::try_zeroed(descriptor.channel_type(), bytes)?,
            0,
        ));
    }
    let total = bytes
        .checked_add(align - 1)
        .ok_or(BufferError::InvalidDimensions)?;
    let data = PixelStorage::try_zeroed(ChannelType::U8, total)?;
    let offset = align_offset(data.as_bytes().as_ptr(), align);
    Ok((data, offset as u32))
}

fn size_of_element<T>(_: &[T]) -> usize {
    core::mem::size_of::<T>()
}

#[cfg(feature = "imgref")]
fn align_of_element<T>(_: &[T]) -> usize {
    core::mem::align_of::<T>()
}

impl ByteStorage for PixelStorage {
    fn as_bytes(&self) -> &[u8] {
        for_each_storage!(self, v => bytemuck::cast_slice(v.as_slice()))
    }
}

impl ByteStorageMut for PixelStorage {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        for_each_storage!(self, v => bytemuck::cast_slice_mut(v.as_mut_slice()))
    }
}

impl fmt::Debug for PixelStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PixelStorage({} bytes)", self.as_bytes().len())
    }
}

/// Convert `Vec<P>` to `PixelStorage`. Zero-copy when sample alignment and capacity match,
/// fallibly copies via `pixels_to_bytes` otherwise.
#[cfg(feature = "rgb")]
pub(super) fn try_pixels_into_storage<P: bytemuck::Pod>(
    pixels: Vec<P>,
    channel: ChannelType,
) -> Result<(PixelStorage, u32), Vec<P>> {
    use bytemuck::allocation::try_cast_vec;
    let moved = match core::mem::align_of::<P>() {
        1 => match channel {
            ChannelType::I8 => try_cast_vec::<P, u8>(pixels)
                .map(|v| PixelStorage::I8(cast_same_align(v)))
                .map_err(|(_, p)| p),
            ChannelType::U8 => try_cast_vec::<P, u8>(pixels)
                .map(PixelStorage::U8)
                .map_err(|(_, p)| p),
            _ => Err(pixels),
        },
        2 => match channel {
            ChannelType::U16 => try_cast_vec::<P, u16>(pixels)
                .map(PixelStorage::U16)
                .map_err(|(_, p)| p),
            ChannelType::I16 => try_cast_vec::<P, u16>(pixels)
                .map(|v| PixelStorage::I16(cast_same_align(v)))
                .map_err(|(_, p)| p),
            ChannelType::F16 => try_cast_vec::<P, u16>(pixels)
                .map(|v| PixelStorage::F16(cast_same_align(v)))
                .map_err(|(_, p)| p),
            _ => Err(pixels),
        },
        4 => match channel {
            ChannelType::U32 => try_cast_vec::<P, u32>(pixels)
                .map(PixelStorage::U32)
                .map_err(|(_, p)| p),
            ChannelType::I32 => try_cast_vec::<P, u32>(pixels)
                .map(|v| PixelStorage::I32(cast_same_align(v)))
                .map_err(|(_, p)| p),
            ChannelType::F32 => try_cast_vec::<P, u32>(pixels)
                .map(|v| PixelStorage::F32(cast_same_align(v)))
                .map_err(|(_, p)| p),
            _ => Err(pixels),
        },
        _ => match channel {
            ChannelType::U64 => try_cast_vec::<P, u64>(pixels)
                .map(PixelStorage::U64)
                .map_err(|(_, p)| p),
            ChannelType::I64 => try_cast_vec::<P, u64>(pixels)
                .map(|v| PixelStorage::I64(cast_same_align(v)))
                .map_err(|(_, p)| p),
            ChannelType::F64 => try_cast_vec::<P, f64>(pixels)
                .map(PixelStorage::F64)
                .map_err(|(_, p)| p),
            _ => Err(pixels),
        },
    };
    match moved {
        Ok(storage) => Ok((storage, 0)),
        Err(pixels) => {
            let (bytes, offset) = pixels_to_bytes(pixels)?;
            Ok((bytes.into(), offset as u32))
        }
    }
}

#[cfg(feature = "rgb")]
fn pixels_to_bytes<P: bytemuck::Pod>(pixels: Vec<P>) -> Result<(Vec<u8>, usize), Vec<P>> {
    match bytemuck::try_cast_vec(pixels) {
        Ok(bytes) => Ok((bytes, 0)),
        Err((_err, pixels)) => {
            let bytes = bytemuck::cast_slice::<P, u8>(&pixels);
            let align = core::mem::align_of::<P>();
            let Some(total) = bytes.len().checked_add(align - 1) else {
                return Err(pixels);
            };
            let Ok(mut data) = bytemuck::allocation::try_zeroed_vec::<u8>(total) else {
                return Err(pixels);
            };
            let offset = align_offset(data.as_ptr(), align);
            data[offset..offset + bytes.len()].copy_from_slice(bytes);
            Ok((data, offset))
        }
    }
}
