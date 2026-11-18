use alloc::vec::Vec;

use super::{
    AlignedPlane, AnyPlaneBuf, LayoutError, PlanarBuffer, PlanarLayout, PlanarSlice,
    PlanarSliceMut, Plane, PlaneBuf, PlaneError, PlaneMut, PlaneRef, PlaneSample,
};
use crate::banded::BandLayout;
use crate::sample::ChromaSubsampling;
use crate::{AlphaMode, BufferError, ChannelType, ColorContext, F16Sample, RowAlign};

/// Single native sample plane allocated in independent row bands.
#[derive(Debug)]
pub struct BandedPlane<T: PlaneSample> {
    layout: BandLayout,
    bands: Vec<AlignedPlane<T>>,
}

impl<T: PlaneSample> BandedPlane<T> {
    /// Plan geometry for tight rows with a 64-byte-aligned allocation per band.
    pub fn band_layout(
        width: u32,
        height: u32,
        max_band_bytes: u64,
    ) -> Result<BandLayout, PlaneError> {
        let cap = max_band_bytes.min(isize::MAX as u64) / 64 * 64;
        BandLayout::for_rows(
            width,
            height,
            u64::from(width) * core::mem::size_of::<T>() as u64,
            cap,
            1,
        )
        .map_err(|_| PlaneError::GeometryOverflow)
    }

    /// Allocate a zeroed, banded plane.
    pub fn try_new(width: u32, height: u32, max_band_bytes: u64) -> Result<Self, PlaneError> {
        let layout = Self::band_layout(width, height, max_band_bytes)?;
        let mut bands = Vec::new();
        bands
            .try_reserve_exact(layout.band_count() as usize)
            .map_err(|_| PlaneError::AllocationFailed)?;
        for range in layout.bands() {
            bands.push(AlignedPlane::alloc_aligned(
                width as usize,
                (range.end - range.start) as usize,
                RowAlign::Tight,
            )?);
        }
        Ok(Self { layout, bands })
    }

    /// Band geometry and global row addressing.
    pub const fn layout(&self) -> BandLayout {
        self.layout
    }

    /// Borrow each band.
    pub fn bands(&self) -> impl ExactSizeIterator<Item = PlaneRef<'_, T>> + '_ {
        self.bands.iter().map(Plane::view)
    }

    /// Lend disjoint writable bands.
    pub fn bands_mut(&mut self) -> impl ExactSizeIterator<Item = PlaneMut<'_, T>> + '_ {
        self.bands.iter_mut().map(Plane::view_mut)
    }

    /// Borrow a global row.
    pub fn row(&self, y: u32) -> Option<&[T]> {
        let (band, row) = self.layout.locate(y)?;
        self.bands[band].row(row as usize)
    }

    /// Write a global row.
    pub fn row_mut(&mut self, y: u32) -> Option<&mut [T]> {
        let (band, row) = self.layout.locate(y)?;
        self.bands[band].row_mut(row as usize)
    }
}

/// Planning, allocation or layout failure for a banded planar picture.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanarBandError {
    /// No band can fit the configured geometry and capacity.
    Geometry(BufferError),
    /// A plane allocation failed.
    Storage(PlaneError),
    /// The plane layout and interpretation disagree.
    Layout(LayoutError),
}

impl core::fmt::Display for PlanarBandError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Geometry(e) => e.fmt(f),
            Self::Storage(e) => e.fmt(f),
            Self::Layout(e) => e.fmt(f),
        }
    }
}

impl core::error::Error for PlanarBandError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Geometry(e) => Some(e),
            Self::Storage(e) => Some(e),
            Self::Layout(e) => Some(e),
        }
    }
}

/// Owned planar picture with consistent band boundaries across subsampled planes.
#[derive(Debug)]
pub struct BandedPlanarBuffer {
    geometry: BandLayout,
    bands: Vec<PlanarBuffer>,
}

impl BandedPlanarBuffer {
    /// Plan bands without allocating. The byte cap counts all planes in one band.
    pub fn band_layout(
        layout: &PlanarLayout,
        width: u32,
        height: u32,
        max_band_bytes: u64,
    ) -> Result<BandLayout, PlanarBandError> {
        let mut multiple = 1;
        for spec in layout.planes() {
            let shift = if spec.role().is_chroma() {
                u8::from(matches!(
                    layout.subsampling(),
                    ChromaSubsampling::Yuv420 | ChromaSubsampling::Yuv440
                ))
            } else {
                spec.v_shift()
            };
            multiple = multiple.max(1u32 << shift);
        }
        // Compute exact bytes in an aligned group of luma rows. The final band
        // may be shorter and round its plane extents up at the image edge.
        let bytes_for = |rows| {
            layout
                .planes()
                .iter()
                .enumerate()
                .try_fold(0u64, |sum, (i, spec)| {
                    let (w, h) = layout.plane_size(i, width, rows)?;
                    u64::from(w)
                        .checked_mul(u64::from(h))?
                        .checked_mul(spec.role().components() as u64)?
                        .checked_mul(spec.encoding().channel_type().byte_size() as u64)?
                        .checked_add(sum)
                })
                .ok_or(PlanarBandError::Geometry(BufferError::InvalidDimensions))
        };
        let cap = max_band_bytes.min(isize::MAX as u64);
        if bytes_for(height).is_ok_and(|bytes| bytes <= cap) {
            return BandLayout::for_rows(width, height, 1, u64::from(height), 1)
                .map_err(PlanarBandError::Geometry);
        }
        let group_bytes = bytes_for(multiple)?;
        let rows_per_band = (cap / group_bytes).min(u64::from(height / multiple)) as u32 * multiple;
        BandLayout::for_rows(width, height, 1, u64::from(rows_per_band), multiple)
            .map_err(PlanarBandError::Geometry)
    }

    /// Allocate typed, tight planes per band. Every view uses the same layout,
    /// color interpretation and alpha mode.
    pub fn try_new(
        layout: &PlanarLayout,
        width: u32,
        height: u32,
        color: ColorContext,
        alpha: Option<AlphaMode>,
        max_band_bytes: u64,
    ) -> Result<Self, PlanarBandError> {
        let geometry = Self::band_layout(layout, width, height, max_band_bytes)?;
        let mut bands = Vec::new();
        bands
            .try_reserve_exact(geometry.band_count() as usize)
            .map_err(|_| PlanarBandError::Storage(PlaneError::AllocationFailed))?;
        for range in geometry.bands() {
            let rows = range.end - range.start;
            let mut planes = Vec::new();
            planes
                .try_reserve_exact(layout.planes().len())
                .map_err(|_| PlanarBandError::Storage(PlaneError::AllocationFailed))?;
            for (i, spec) in layout.planes().iter().enumerate() {
                let (w, h) = layout
                    .plane_size(i, width, rows)
                    .expect("valid plane index");
                let w = (w as usize)
                    .checked_mul(spec.role().components())
                    .ok_or(PlanarBandError::Storage(PlaneError::GeometryOverflow))?;
                planes.push(
                    allocate(spec.encoding().channel_type(), w, h as usize)
                        .map_err(PlanarBandError::Storage)?,
                );
            }
            bands.push(
                PlanarBuffer::new(planes, layout.clone(), width, rows, color.clone(), alpha)
                    .map_err(PlanarBandError::Layout)?,
            );
        }
        Ok(Self { geometry, bands })
    }

    /// Luma-space geometry of the bands.
    pub const fn geometry(&self) -> BandLayout {
        self.geometry
    }

    /// Borrow each band in luma-space order.
    pub fn bands(&self) -> impl ExactSizeIterator<Item = PlanarSlice<'_>> + '_ {
        self.bands.iter().map(PlanarBuffer::as_slice)
    }

    /// Write the validated planar bands in luma-space order.
    pub fn bands_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = PlanarSliceMut<'_, AnyPlaneBuf>> + '_ {
        self.bands.iter_mut().map(PlanarBuffer::as_slice_mut)
    }
}

fn allocate(channel: ChannelType, width: usize, height: usize) -> Result<AnyPlaneBuf, PlaneError> {
    fn plane<T: PlaneSample>(width: usize, height: usize) -> Result<PlaneBuf<T>, PlaneError> {
        let n = width
            .checked_mul(height)
            .ok_or(PlaneError::GeometryOverflow)?;
        core::alloc::Layout::array::<T>(n).map_err(|_| PlaneError::GeometryOverflow)?;
        let data =
            bytemuck::allocation::try_zeroed_vec(n).map_err(|()| PlaneError::AllocationFailed)?;
        Plane::new(data, width, height, width)
    }
    macro_rules! typed {
        ($($v:ident => $ty:ty),* $(,)?) => {
            match channel {
                $(ChannelType::$v => plane::<$ty>(width, height).map(AnyPlaneBuf::$v),)*
                _ => Err(PlaneError::GeometryOverflow),
            }
        };
    }
    typed!(U8 => u8, I8 => i8, U16 => u16, I16 => i16, F16 => F16Sample, U32 => u32,
        I32 => i32, F32 => f32, U64 => u64, I64 => i64, F64 => f64)
}
