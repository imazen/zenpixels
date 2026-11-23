//! Alignment-exact, `#![forbid(unsafe_code)]` owned pixel storage.
//!
//! # Allocation / deallocation alignment rules (`bytemuck` & `GlobalAlloc`)
//!
//! Rust's [`core::alloc::GlobalAlloc::dealloc`] requires every allocation to be
//! freed with the **exact same `Layout` (byte capacity and alignment)** that
//! allocated it. Therefore:
//!
//! - A `Vec<u8>` (`align = 1`) can **never** become a `Vec<u16>` (`align = 2`)
//!   or `Vec<Rgb<u16>>` (`align = 2`) without copying, even if glibc happens to
//!   return an even pointer.
//! - Conversely, `Vec<Rgb<u16>>` (`align = 2, size = 6`) **can** move zero-copy
//!   to and from `Vec<u16>` (`align = 2, size = 2`) via
//!   [`bytemuck::allocation::try_cast_vec`] whenever both length and capacity in
//!   bytes are multiples of `size_of::<Rgb<u16>>()` (`6`).
//! - [`AlignedSamples`] allocates `Vec<Chunk>` with `#[repr(C, align(64))]`, so
//!   its base address is always 64-byte aligned (divisible by `1, 2, 4, 8, 16,
//!   32, 64` for zero-copy slice borrowing via `bytemuck::cast_slice`), and it
//!   always deallocates with `align = 64` (never UB-cast to `Vec<u8>`).
//! - On 32-bit targets where `align_of::<u64>()` may differ from
//!   `align_of::<f64>()`, `F64` checks exact alignment equality before casting.

use alloc::vec::Vec;
use core::fmt;

use crate::layout::SampleType;
use crate::pixel_types::F16Sample;

/// 64-byte aligned allocation unit for [`AlignedSamples`].
///
/// Using `Vec<Chunk>` guarantees 64-byte base alignment and matching 64-byte
/// deallocation layout under `#![forbid(unsafe_code)]`.
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C, align(64))]
struct Chunk([u8; 64]);

/// Owned byte storage whose first byte is guaranteed 64-byte aligned on every allocator.
///
/// Zero-copy slice borrowing works for every sample and pixel type (`&[u8]`,
/// `&[u16]`, `&[f32]`, `&[f64]`, `&[Rgb<u16>]`, `&[Rgba<f32>]`, …) because 64
/// is a multiple of every natural alignment. Cannot convert to `Vec<T>` without
/// copying because its deallocation alignment is 64.
#[derive(Clone)]
pub struct AlignedSamples {
    chunks: Vec<Chunk>,
    byte_len: usize,
}

impl AlignedSamples {
    /// Fallibly allocate `byte_len` zeroed bytes on a 64-byte boundary.
    pub fn try_zeroed_bytes(byte_len: usize) -> Result<Self, StorageError> {
        if byte_len > isize::MAX as usize {
            return Err(StorageError::AllocationFailed);
        }
        let count = byte_len.div_ceil(core::mem::size_of::<Chunk>());
        core::alloc::Layout::array::<Chunk>(count).map_err(|_| StorageError::AllocationFailed)?;
        let chunks = bytemuck::allocation::try_zeroed_vec(count)
            .map_err(|()| StorageError::AllocationFailed)?;
        Ok(Self { chunks, byte_len })
    }

    /// Immutable byte view (guaranteed 64-byte aligned base pointer).
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &bytemuck::cast_slice(&self.chunks)[..self.byte_len]
    }

    /// Mutable byte view (guaranteed 64-byte aligned base pointer).
    #[inline]
    #[must_use]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut bytemuck::cast_slice_mut(&mut self.chunks)[..self.byte_len]
    }

    /// Active byte length.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.byte_len
    }

    /// Whether the buffer is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.byte_len == 0
    }

    /// Truncate active byte length without reallocating.
    #[inline]
    pub fn truncate(&mut self, byte_len: usize) {
        if byte_len < self.byte_len {
            self.byte_len = byte_len;
        }
    }
}

impl fmt::Debug for AlignedSamples {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AlignedSamples({} bytes, align=64)", self.byte_len)
    }
}

/// Low-level storage allocation error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageError {
    /// Allocator refused or byte length exceeded `isize::MAX`.
    AllocationFailed,
}

/// Read access to native-endian pixel bytes without copying.
pub trait ByteStorage {
    /// Underlying bytes in native endianness.
    fn as_bytes(&self) -> &[u8];
}

/// Mutable access to native-endian pixel bytes without copying.
pub trait ByteStorageMut: ByteStorage {
    /// Mutable underlying bytes in native endianness.
    fn as_bytes_mut(&mut self) -> &mut [u8];
}

impl ByteStorage for AlignedSamples {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl ByteStorageMut for AlignedSamples {
    #[inline]
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.as_bytes_mut()
    }
}

mod sealed {
    use super::*;
    pub trait Sealed: Sized {
        fn into_storage(v: Vec<Self>) -> PixelStorage;
        fn from_storage(s: PixelStorage) -> Result<Vec<Self>, PixelStorage>;
    }
}

/// Sealed trait implemented by the 11 native sample types stored in [`PixelStorage`].
pub trait StorageSample:
    sealed::Sealed + bytemuck::Pod + Default + PartialEq + fmt::Debug + Send + Sync + 'static
{
    /// Corresponding [`SampleType`] variant.
    const SAMPLE_TYPE: SampleType;
}

/// Alignment-exact owned pixel backing storage.
///
/// Stores the true sample-typed `Vec<T>` (or 64-byte [`AlignedSamples`]) so
/// `Vec<u16>`, `Vec<Rgb<u16>>`, `Vec<f32>`, `Vec<Rgba<f32>>`, etc. cross
/// type-erased [`PixelBuf`](crate::PixelBuf) boundaries **zero-copy** while
/// strictly preserving Rust's allocation/deallocation alignment contract.
#[non_exhaustive]
pub enum PixelStorage {
    /// `u8` backing (`align = 1`).
    U8(Vec<u8>),
    /// `i8` backing (`align = 1`).
    I8(Vec<i8>),
    /// `u16` backing (`align = 2`).
    U16(Vec<u16>),
    /// `i16` backing (`align = 2`).
    I16(Vec<i16>),
    /// `F16Sample` backing (`align = 2`).
    F16(Vec<F16Sample>),
    /// `u32` backing (`align = 4`).
    U32(Vec<u32>),
    /// `i32` backing (`align = 4`).
    I32(Vec<i32>),
    /// `f32` backing (`align = 4`).
    F32(Vec<f32>),
    /// `u64` backing (`align = align_of::<u64>()`).
    U64(Vec<u64>),
    /// `i64` backing (`align = align_of::<i64>()`).
    I64(Vec<i64>),
    /// `f64` backing (`align = align_of::<f64>()`).
    F64(Vec<f64>),
    /// 64-byte SIMD-aligned backing (`align = 64`).
    Aligned64(AlignedSamples),
}

macro_rules! impl_storage_samples {
    ($($ty:ty => $var:ident),* $(,)?) => {$(
        impl sealed::Sealed for $ty {
            #[inline]
            fn into_storage(v: Vec<Self>) -> PixelStorage {
                PixelStorage::$var(v)
            }
            #[inline]
            fn from_storage(s: PixelStorage) -> Result<Vec<Self>, PixelStorage> {
                match s {
                    PixelStorage::$var(v) => Ok(v),
                    other => Err(other),
                }
            }
        }
        impl StorageSample for $ty {
            const SAMPLE_TYPE: SampleType = SampleType::$var;
        }
        impl From<Vec<$ty>> for PixelStorage {
            #[inline]
            fn from(v: Vec<$ty>) -> Self {
                PixelStorage::$var(v)
            }
        }
    )*};
}

impl_storage_samples! {
    u8 => U8, i8 => I8,
    u16 => U16, i16 => I16, F16Sample => F16,
    u32 => U32, i32 => I32, f32 => F32,
    u64 => U64, i64 => I64, f64 => F64,
}

#[inline]
fn cast_same_layout<A: bytemuck::NoUninit, B: bytemuck::AnyBitPattern>(v: Vec<A>) -> Vec<B> {
    match bytemuck::allocation::try_cast_vec(v) {
        Ok(out) => out,
        Err(_) => panic!("same-size same-alignment Vec cast failed"),
    }
}

fn zeroed_vec<T: bytemuck::Zeroable + bytemuck::Pod>(bytes: usize) -> Result<Vec<T>, StorageError> {
    let sz = core::mem::size_of::<T>();
    let count = bytes.div_ceil(sz);
    core::alloc::Layout::array::<T>(count).map_err(|_| StorageError::AllocationFailed)?;
    bytemuck::allocation::try_zeroed_vec(count).map_err(|()| StorageError::AllocationFailed)
}

fn gather_typed<T: bytemuck::Zeroable + bytemuck::Pod>(
    total_bytes: usize,
    rows: &mut dyn Iterator<Item = &[u8]>,
) -> Result<Vec<T>, StorageError> {
    if total_bytes > isize::MAX as usize {
        return Err(StorageError::AllocationFailed);
    }
    let mut out = zeroed_vec::<T>(total_bytes)?;
    let dst = bytemuck::cast_slice_mut::<T, u8>(&mut out);
    let mut pos = 0usize;
    for row in rows {
        let end = pos
            .checked_add(row.len())
            .ok_or(StorageError::AllocationFailed)?;
        if end > total_bytes || end > dst.len() {
            return Err(StorageError::AllocationFailed);
        }
        dst[pos..end].copy_from_slice(row);
        pos = end;
    }
    Ok(out)
}

impl PixelStorage {
    /// Allocate zeroed storage with exact alignment for `sample`, holding at least `bytes` bytes.
    pub fn try_zeroed(sample: SampleType, bytes: usize) -> Result<Self, StorageError> {
        if bytes > isize::MAX as usize {
            return Err(StorageError::AllocationFailed);
        }
        match sample {
            SampleType::U8 => Ok(Self::U8(zeroed_vec(bytes)?)),
            SampleType::I8 => Ok(Self::I8(cast_same_layout(zeroed_vec::<u8>(bytes)?))),
            SampleType::U16 => Ok(Self::U16(zeroed_vec(bytes)?)),
            SampleType::I16 => Ok(Self::I16(cast_same_layout(zeroed_vec::<u16>(bytes)?))),
            SampleType::F16 => Ok(Self::F16(cast_same_layout(zeroed_vec::<u16>(bytes)?))),
            SampleType::U32 => Ok(Self::U32(zeroed_vec(bytes)?)),
            SampleType::I32 => Ok(Self::I32(cast_same_layout(zeroed_vec::<u32>(bytes)?))),
            SampleType::F32 => Ok(Self::F32(cast_same_layout(zeroed_vec::<u32>(bytes)?))),
            SampleType::U64 => Ok(Self::U64(zeroed_vec(bytes)?)),
            SampleType::I64 => Ok(Self::I64(cast_same_layout(zeroed_vec::<u64>(bytes)?))),
            SampleType::F64 => {
                if core::mem::align_of::<u64>() == core::mem::align_of::<f64>() {
                    Ok(Self::F64(cast_same_layout(zeroed_vec::<u64>(bytes)?)))
                } else {
                    Ok(Self::F64(zeroed_vec(bytes)?))
                }
            }
        }
    }

    /// Allocate 64-byte SIMD-aligned zeroed storage.
    #[inline]
    pub fn try_zeroed_aligned64(bytes: usize) -> Result<Self, StorageError> {
        Ok(Self::Aligned64(AlignedSamples::try_zeroed_bytes(bytes)?))
    }

    /// Gather row slices into fresh tightly packed storage of `sample`'s native alignment.
    pub fn try_from_rows<'r>(
        sample: SampleType,
        total_bytes: usize,
        mut rows: impl Iterator<Item = &'r [u8]>,
    ) -> Result<Self, StorageError> {
        Self::try_from_rows_dyn(sample, total_bytes, &mut rows)
    }

    fn try_from_rows_dyn(
        sample: SampleType,
        total_bytes: usize,
        rows: &mut dyn Iterator<Item = &[u8]>,
    ) -> Result<Self, StorageError> {
        match sample {
            SampleType::U8 => Ok(Self::U8(gather_typed(total_bytes, rows)?)),
            SampleType::I8 => Ok(Self::I8(cast_same_layout(gather_typed::<u8>(
                total_bytes,
                rows,
            )?))),
            SampleType::U16 => Ok(Self::U16(gather_typed(total_bytes, rows)?)),
            SampleType::I16 => Ok(Self::I16(cast_same_layout(gather_typed::<u16>(
                total_bytes,
                rows,
            )?))),
            SampleType::F16 => Ok(Self::F16(cast_same_layout(gather_typed::<u16>(
                total_bytes,
                rows,
            )?))),
            SampleType::U32 => Ok(Self::U32(gather_typed(total_bytes, rows)?)),
            SampleType::I32 => Ok(Self::I32(cast_same_layout(gather_typed::<u32>(
                total_bytes,
                rows,
            )?))),
            SampleType::F32 => Ok(Self::F32(cast_same_layout(gather_typed::<u32>(
                total_bytes,
                rows,
            )?))),
            SampleType::U64 => Ok(Self::U64(gather_typed(total_bytes, rows)?)),
            SampleType::I64 => Ok(Self::I64(cast_same_layout(gather_typed::<u64>(
                total_bytes,
                rows,
            )?))),
            SampleType::F64 => {
                if core::mem::align_of::<u64>() == core::mem::align_of::<f64>() {
                    Ok(Self::F64(cast_same_layout(gather_typed::<u64>(
                        total_bytes,
                        rows,
                    )?)))
                } else {
                    Ok(Self::F64(gather_typed(total_bytes, rows)?))
                }
            }
        }
    }

    /// Move a typed `Vec<P>` into native [`PixelStorage`] **zero-copy** when
    /// `align_of::<P>()` and capacity match `sample`'s backing layout.
    ///
    /// On mismatch (e.g., an over-aligned pixel type), returns `Err(pixels)`
    /// untouched without allocating.
    pub fn try_adopt_vec<P: bytemuck::Pod>(
        pixels: Vec<P>,
        sample: SampleType,
    ) -> Result<Self, Vec<P>> {
        use bytemuck::allocation::try_cast_vec;
        match sample {
            SampleType::U8 => try_cast_vec::<P, u8>(pixels)
                .map(Self::U8)
                .map_err(|(_, v)| v),
            SampleType::I8 => try_cast_vec::<P, i8>(pixels)
                .map(Self::I8)
                .map_err(|(_, v)| v),
            SampleType::U16 => try_cast_vec::<P, u16>(pixels)
                .map(Self::U16)
                .map_err(|(_, v)| v),
            SampleType::I16 => try_cast_vec::<P, i16>(pixels)
                .map(Self::I16)
                .map_err(|(_, v)| v),
            SampleType::F16 => try_cast_vec::<P, F16Sample>(pixels)
                .map(Self::F16)
                .map_err(|(_, v)| v),
            SampleType::U32 => try_cast_vec::<P, u32>(pixels)
                .map(Self::U32)
                .map_err(|(_, v)| v),
            SampleType::I32 => try_cast_vec::<P, i32>(pixels)
                .map(Self::I32)
                .map_err(|(_, v)| v),
            SampleType::F32 => try_cast_vec::<P, f32>(pixels)
                .map(Self::F32)
                .map_err(|(_, v)| v),
            SampleType::U64 => try_cast_vec::<P, u64>(pixels)
                .map(Self::U64)
                .map_err(|(_, v)| v),
            SampleType::I64 => try_cast_vec::<P, i64>(pixels)
                .map(Self::I64)
                .map_err(|(_, v)| v),
            SampleType::F64 => try_cast_vec::<P, f64>(pixels)
                .map(Self::F64)
                .map_err(|(_, v)| v),
        }
    }

    /// Move this storage into `Vec<P>` **zero-copy** when `align_of::<P>()`
    /// matches the backing allocation's exact alignment AND both active byte
    /// length and byte capacity are exact multiples of `size_of::<P>()`.
    ///
    /// Returns `Err(self)` untouched otherwise (never UB, never copies).
    pub fn try_cast_into_vec<P: bytemuck::Pod>(self) -> Result<Vec<P>, Self> {
        use bytemuck::allocation::try_cast_vec;
        match self {
            Self::U8(v) => try_cast_vec(v).map_err(|(_, v)| Self::U8(v)),
            Self::I8(v) => try_cast_vec(v).map_err(|(_, v)| Self::I8(v)),
            Self::U16(v) => try_cast_vec(v).map_err(|(_, v)| Self::U16(v)),
            Self::I16(v) => try_cast_vec(v).map_err(|(_, v)| Self::I16(v)),
            Self::F16(v) => try_cast_vec(v).map_err(|(_, v)| Self::F16(v)),
            Self::U32(v) => try_cast_vec(v).map_err(|(_, v)| Self::U32(v)),
            Self::I32(v) => try_cast_vec(v).map_err(|(_, v)| Self::I32(v)),
            Self::F32(v) => try_cast_vec(v).map_err(|(_, v)| Self::F32(v)),
            Self::U64(v) => try_cast_vec(v).map_err(|(_, v)| Self::U64(v)),
            Self::I64(v) => try_cast_vec(v).map_err(|(_, v)| Self::I64(v)),
            Self::F64(v) => try_cast_vec(v).map_err(|(_, v)| Self::F64(v)),
            // Aligned64 has deallocation alignment 64; cannot become Vec<P> zero-copy unless align_of::<P>() == 64.
            Self::Aligned64(a) => Err(Self::Aligned64(a)),
        }
    }

    /// Extract the exact sample-typed `Vec<T>` zero-copy when the variant matches.
    #[inline]
    pub fn try_into_sample_vec<T: StorageSample>(self) -> Result<Vec<T>, Self> {
        T::from_storage(self)
    }

    /// Element alignment of the underlying heap allocation.
    #[inline]
    #[must_use]
    pub const fn alloc_alignment(&self) -> usize {
        match self {
            Self::U8(_) | Self::I8(_) => core::mem::align_of::<u8>(),
            Self::U16(_) | Self::I16(_) | Self::F16(_) => core::mem::align_of::<u16>(),
            Self::U32(_) | Self::I32(_) => core::mem::align_of::<u32>(),
            Self::F32(_) => core::mem::align_of::<f32>(),
            Self::U64(_) | Self::I64(_) => core::mem::align_of::<u64>(),
            Self::F64(_) => core::mem::align_of::<f64>(),
            Self::Aligned64(_) => 64,
        }
    }

    /// Byte capacity of the underlying allocation.
    #[inline]
    #[must_use]
    pub fn byte_capacity(&self) -> usize {
        match self {
            Self::U8(v) => v.capacity(),
            Self::I8(v) => v.capacity(),
            Self::U16(v) => v.capacity() * 2,
            Self::I16(v) => v.capacity() * 2,
            Self::F16(v) => v.capacity() * 2,
            Self::U32(v) => v.capacity() * 4,
            Self::I32(v) => v.capacity() * 4,
            Self::F32(v) => v.capacity() * 4,
            Self::U64(v) => v.capacity() * 8,
            Self::I64(v) => v.capacity() * 8,
            Self::F64(v) => v.capacity() * 8,
            Self::Aligned64(a) => a.byte_len,
        }
    }

    /// Whether truncating to `target_bytes` and calling `try_cast_into_vec::<P>`
    /// will succeed zero-copy without violating deallocation alignment or capacity.
    #[inline]
    #[must_use]
    pub fn can_cast_into_vec<P: bytemuck::Pod>(&self, target_bytes: usize) -> bool {
        let p_size = core::mem::size_of::<P>();
        let p_align = core::mem::align_of::<P>();
        if p_size == 0
            || matches!(self, Self::Aligned64(_))
            || self.alloc_alignment() != p_align
            || target_bytes > self.byte_len()
            || target_bytes % p_size != 0
            || self.byte_capacity() % p_size != 0
        {
            return false;
        }
        // `truncate_bytes` rounds up to sample unit boundaries, so `target_bytes` must be sample-aligned.
        let unit = match self {
            Self::U8(_) | Self::I8(_) => 1,
            Self::U16(_) | Self::I16(_) | Self::F16(_) => 2,
            Self::U32(_) | Self::I32(_) | Self::F32(_) => 4,
            Self::U64(_) | Self::I64(_) | Self::F64(_) => 8,
            Self::Aligned64(_) => 64,
        };
        target_bytes % unit == 0
    }

    /// Byte length of the stored allocation.
    #[inline]
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.as_bytes().len()
    }

    /// Truncate active byte count in place (rounded up to whole sample units).
    pub fn truncate_bytes(&mut self, bytes: usize) {
        match self {
            Self::U8(v) => v.truncate(bytes),
            Self::I8(v) => v.truncate(bytes),
            Self::U16(v) => v.truncate(bytes.div_ceil(2)),
            Self::I16(v) => v.truncate(bytes.div_ceil(2)),
            Self::F16(v) => v.truncate(bytes.div_ceil(2)),
            Self::U32(v) => v.truncate(bytes.div_ceil(4)),
            Self::I32(v) => v.truncate(bytes.div_ceil(4)),
            Self::F32(v) => v.truncate(bytes.div_ceil(4)),
            Self::U64(v) => v.truncate(bytes.div_ceil(8)),
            Self::I64(v) => v.truncate(bytes.div_ceil(8)),
            Self::F64(v) => v.truncate(bytes.div_ceil(8)),
            Self::Aligned64(a) => a.truncate(bytes),
        }
    }
}

impl Clone for PixelStorage {
    fn clone(&self) -> Self {
        match self {
            Self::U8(v) => Self::U8(v.clone()),
            Self::I8(v) => Self::I8(v.clone()),
            Self::U16(v) => Self::U16(v.clone()),
            Self::I16(v) => Self::I16(v.clone()),
            Self::F16(v) => Self::F16(v.clone()),
            Self::U32(v) => Self::U32(v.clone()),
            Self::I32(v) => Self::I32(v.clone()),
            Self::F32(v) => Self::F32(v.clone()),
            Self::U64(v) => Self::U64(v.clone()),
            Self::I64(v) => Self::I64(v.clone()),
            Self::F64(v) => Self::F64(v.clone()),
            Self::Aligned64(a) => Self::Aligned64(a.clone()),
        }
    }
}

impl ByteStorage for PixelStorage {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        match self {
            Self::U8(v) => v,
            Self::I8(v) => bytemuck::cast_slice(v),
            Self::U16(v) => bytemuck::cast_slice(v),
            Self::I16(v) => bytemuck::cast_slice(v),
            Self::F16(v) => bytemuck::cast_slice(v),
            Self::U32(v) => bytemuck::cast_slice(v),
            Self::I32(v) => bytemuck::cast_slice(v),
            Self::F32(v) => bytemuck::cast_slice(v),
            Self::U64(v) => bytemuck::cast_slice(v),
            Self::I64(v) => bytemuck::cast_slice(v),
            Self::F64(v) => bytemuck::cast_slice(v),
            Self::Aligned64(a) => a.as_bytes(),
        }
    }
}

impl ByteStorageMut for PixelStorage {
    #[inline]
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Self::U8(v) => v,
            Self::I8(v) => bytemuck::cast_slice_mut(v),
            Self::U16(v) => bytemuck::cast_slice_mut(v),
            Self::I16(v) => bytemuck::cast_slice_mut(v),
            Self::F16(v) => bytemuck::cast_slice_mut(v),
            Self::U32(v) => bytemuck::cast_slice_mut(v),
            Self::I32(v) => bytemuck::cast_slice_mut(v),
            Self::F32(v) => bytemuck::cast_slice_mut(v),
            Self::U64(v) => bytemuck::cast_slice_mut(v),
            Self::I64(v) => bytemuck::cast_slice_mut(v),
            Self::F64(v) => bytemuck::cast_slice_mut(v),
            Self::Aligned64(a) => a.as_bytes_mut(),
        }
    }
}

impl fmt::Debug for PixelStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PixelStorage({} bytes, align={})",
            self.byte_len(),
            self.alloc_alignment()
        )
    }
}
