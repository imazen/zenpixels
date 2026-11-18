use alloc::vec::Vec;

use super::{
    AlignedPlane, AnyPlaneBuf, LayoutError, PlanarBuffer, PlanarLayout, PlanarSlice,
    PlanarSliceMut, Plane, PlaneBuf, PlaneError, PlaneMut, PlaneRef, PlaneSample, PlaneSpec,
};
use crate::banded::BandLayout;
use crate::sample::{ChromaSiting, ChromaSubsampling};
use crate::{AlphaMode, BufferError, ChannelType, ColorContext, F16Sample, Orientation, RowAlign};

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

    /// Apply identity, a flip, or a half-turn without allocating another band.
    pub fn orient_in_place(&mut self, orientation: Orientation) -> Result<(), PlaneError> {
        if orientation.swaps_axes() {
            return Err(PlaneError::OutsidePlane);
        }
        let height = self.layout.height();
        if matches!(orientation, Orientation::FlipV | Orientation::Rotate180) {
            for y in 0..height / 2 {
                let (a, ay) = self.layout.locate(y).expect("row in bounds");
                let (b, by) = self.layout.locate(height - 1 - y).expect("row in bounds");
                if a == b {
                    let (mut first, mut last) =
                        self.bands[a].view_mut().split_at_row(by as usize)?;
                    first
                        .row_mut(ay as usize)
                        .unwrap()
                        .swap_with_slice(last.row_mut(0).unwrap());
                } else {
                    let (before, after) = self.bands.split_at_mut(b);
                    before[a]
                        .row_mut(ay as usize)
                        .unwrap()
                        .swap_with_slice(after[0].row_mut(by as usize).unwrap());
                }
            }
        }
        if matches!(orientation, Orientation::FlipH | Orientation::Rotate180) {
            for plane in &mut self.bands {
                for row in plane.rows_mut() {
                    row.reverse();
                }
            }
        }
        Ok(())
    }

    /// Copy an oriented plane into already allocated bands. Axis swaps visit
    /// bounded tiles and can cross independently chosen band boundaries.
    pub fn oriented_into(
        &self,
        orientation: Orientation,
        destination: &mut Self,
    ) -> Result<(), PlaneError> {
        let (width, height) = (self.layout.width(), self.layout.height());
        if orientation.output_dimensions(width, height)
            != (destination.layout.width(), destination.layout.height())
        {
            return Err(PlaneError::OutsidePlane);
        }
        crate::banded::for_each_orientation_tile(width, height, |y, columns| {
            let row = self.row(y).expect("row in bounds");
            for x in columns {
                let (dx, dy) = orientation.forward_map(x, y, width, height);
                destination.row_mut(dy).unwrap()[dx as usize] = row[x as usize];
            }
        });
        Ok(())
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
    /// The output layout cannot express the oriented sampling lattice.
    OrientationNeedsResampling,
}

impl core::fmt::Display for PlanarBandError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Geometry(e) => e.fmt(f),
            Self::Storage(e) => e.fmt(f),
            Self::Layout(e) => e.fmt(f),
            Self::OrientationNeedsResampling => {
                f.write_str("orientation requires a different sampling lattice")
            }
        }
    }
}

impl core::error::Error for PlanarBandError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Geometry(e) => Some(e),
            Self::Storage(e) => Some(e),
            Self::Layout(e) => Some(e),
            Self::OrientationNeedsResampling => None,
        }
    }
}

/// Owned planar picture with consistent band boundaries across subsampled planes.
#[derive(Debug)]
pub struct BandedPlanarBuffer {
    geometry: BandLayout,
    bands: Vec<PlanarBuffer>,
}

macro_rules! plane_row_bytes {
    ($owner:expr, $row:expr, $method:ident, $cast:ident) => {
        match $owner {
            AnyPlaneBuf::U8(plane) => plane.$method($row).map(bytemuck::$cast),
            AnyPlaneBuf::I8(plane) => plane.$method($row).map(bytemuck::$cast),
            AnyPlaneBuf::U16(plane) => plane.$method($row).map(bytemuck::$cast),
            AnyPlaneBuf::I16(plane) => plane.$method($row).map(bytemuck::$cast),
            AnyPlaneBuf::F16(plane) => plane.$method($row).map(bytemuck::$cast),
            AnyPlaneBuf::U32(plane) => plane.$method($row).map(bytemuck::$cast),
            AnyPlaneBuf::I32(plane) => plane.$method($row).map(bytemuck::$cast),
            AnyPlaneBuf::F32(plane) => plane.$method($row).map(bytemuck::$cast),
            AnyPlaneBuf::U64(plane) => plane.$method($row).map(bytemuck::$cast),
            AnyPlaneBuf::I64(plane) => plane.$method($row).map(bytemuck::$cast),
            AnyPlaneBuf::F64(plane) => plane.$method($row).map(bytemuck::$cast),
        }
    };
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

    /// Active bytes of one row in a plane's own subsampled coordinate system.
    /// Interleaved components such as NV12 Cb/Cr remain interleaved.
    pub fn plane_row(&self, plane: usize, y: u32) -> Option<&[u8]> {
        let (band, row) = self.plane_location(plane, y)?;
        plane_row_bytes!(&self.bands[band].planes[plane], row, row, cast_slice)
    }

    /// Mutably borrow a global row in one plane's coordinate system.
    pub fn plane_row_mut(&mut self, plane: usize, y: u32) -> Option<&mut [u8]> {
        let (band, row) = self.plane_location(plane, y)?;
        plane_row_bytes!(
            &mut self.bands[band].planes[plane],
            row,
            row_mut,
            cast_slice_mut
        )
    }

    /// Orient native samples into an independent banded picture.
    ///
    /// The destination layout must represent the rotated or reflected sampling
    /// lattice. A flip can require resampling subsampled chroma at an odd edge.
    /// Validation finishes before any output row is written.
    pub fn oriented_into(
        &self,
        orientation: Orientation,
        destination: &mut Self,
    ) -> Result<(), PlanarBandError> {
        self.check_orientation(orientation, destination)?;
        let first = &self.bands[0];
        for (index, spec) in first.layout.planes().iter().enumerate() {
            let (width, height) = first
                .layout
                .plane_size(index, self.geometry.width(), self.geometry.height())
                .unwrap();
            let bytes = group_bytes(spec);
            crate::banded::for_each_orientation_tile(width, height, |y, columns| {
                let input = self.plane_row(index, y).unwrap();
                for x in columns {
                    let (dx, dy) = orientation.forward_map(x, y, width, height);
                    let output = destination.plane_row_mut(index, dy).unwrap();
                    let source = x as usize * bytes;
                    let target = dx as usize * bytes;
                    output[target..target + bytes].copy_from_slice(&input[source..source + bytes]);
                }
            });
        }
        for band in &mut destination.bands {
            band.color = first.color.clone();
        }
        Ok(())
    }

    /// Apply identity, a flip, or a half-turn in the existing sampling lattice
    /// without allocating plane storage.
    pub fn orient_in_place(&mut self, orientation: Orientation) -> Result<(), PlanarBandError> {
        if orientation.swaps_axes() {
            return Err(PlanarBandError::OrientationNeedsResampling);
        }
        self.check_orientation(orientation, self)?;
        let (flip_x, flip_y) = flipped_axes(orientation);
        for index in 0..self.bands[0].layout.planes().len() {
            let layout = &self.bands[0].layout;
            let (width, height) = layout
                .plane_size(index, self.geometry.width(), self.geometry.height())
                .unwrap();
            let bytes = group_bytes(&layout.planes()[index]);
            if flip_y {
                let row_bytes = width as usize * bytes;
                for y in 0..height / 2 {
                    let (a, ay) = self.plane_location(index, y).unwrap();
                    let (b, by) = self.plane_location(index, height - 1 - y).unwrap();
                    if a == b {
                        let mut plane = self.bands[a].planes[index].as_mut();
                        let rows = plane.as_strided_bytes_mut();
                        let (first, last) = rows.split_at_mut(by * row_bytes);
                        let first = &mut first[ay * row_bytes..(ay + 1) * row_bytes];
                        first.swap_with_slice(&mut last[..row_bytes]);
                    } else {
                        let (front, back) = self.bands.split_at_mut(b);
                        let first: &mut [u8] = plane_row_bytes!(
                            &mut front[a].planes[index],
                            ay,
                            row_mut,
                            cast_slice_mut
                        )
                        .unwrap();
                        let last: &mut [u8] = plane_row_bytes!(
                            &mut back[0].planes[index],
                            by,
                            row_mut,
                            cast_slice_mut
                        )
                        .unwrap();
                        first.swap_with_slice(last);
                    }
                }
            }
            if flip_x {
                for y in 0..height {
                    let row = self.plane_row_mut(index, y).unwrap();
                    crate::banded::reverse_pixel_row(row, bytes);
                }
            }
        }
        Ok(())
    }

    fn check_orientation(
        &self,
        orientation: Orientation,
        destination: &Self,
    ) -> Result<(), PlanarBandError> {
        use PlanarBandError::{Geometry, OrientationNeedsResampling};
        let (sw, sh) = (self.geometry.width(), self.geometry.height());
        let (dw, dh) = (destination.geometry.width(), destination.geometry.height());
        if orientation.output_dimensions(sw, sh) != (dw, dh) {
            return Err(Geometry(BufferError::InvalidDimensions));
        }
        let source = &self.bands[0];
        let target = &destination.bands[0];
        if source.alpha != target.alpha
            || source.layout.planes().len() != target.layout.planes().len()
        {
            return Err(OrientationNeedsResampling);
        }
        let (flip_x, flip_y) = flipped_axes(orientation);
        for (i, (src, dst)) in source
            .layout
            .planes()
            .iter()
            .zip(target.layout.planes())
            .enumerate()
        {
            let (pw, ph) = source.layout.plane_size(i, sw, sh).unwrap();
            let (qw, qh) = target.layout.plane_size(i, dw, dh).unwrap();
            if src.role() != dst.role()
                || src.encoding() != dst.encoding()
                || orientation.output_dimensions(pw, ph) != (qw, qh)
            {
                return Err(OrientationNeedsResampling);
            }
            let mut axes = [
                (sw, pw, grid(&source.layout, src, false)),
                (sh, ph, grid(&source.layout, src, true)),
            ];
            if orientation.swaps_axes() {
                axes.swap(0, 1);
            }
            for (axis, flip) in [flip_x, flip_y].into_iter().enumerate() {
                let (extent, samples, (step, offset)) = axes[axis];
                let (target_step, target_offset) = grid(&target.layout, dst, axis == 1);
                let offset = if flip {
                    let offset = offset.ok_or(OrientationNeedsResampling)?;
                    Some(2 * i64::from(extent - 1) - 2 * i64::from(samples - 1) * step - offset)
                } else {
                    offset
                };
                if step != target_step || offset != target_offset {
                    return Err(OrientationNeedsResampling);
                }
            }
        }
        Ok(())
    }

    fn plane_location(&self, plane: usize, y: u32) -> Option<(usize, usize)> {
        let first = self.bands.first()?;
        let (_, total) =
            first
                .layout
                .plane_size(plane, self.geometry.width(), self.geometry.height())?;
        if y >= total {
            return None;
        }
        let rows = first.as_slice().plane(plane)?.height();
        Some((y as usize / rows, y as usize % rows))
    }
}

fn group_bytes(spec: &PlaneSpec) -> usize {
    spec.role().components() * spec.encoding().channel_type().byte_size()
}

fn flipped_axes(orientation: Orientation) -> (bool, bool) {
    (
        matches!(
            orientation,
            Orientation::FlipH
                | Orientation::Rotate180
                | Orientation::Rotate90
                | Orientation::Transverse
        ),
        matches!(
            orientation,
            Orientation::FlipV
                | Orientation::Rotate180
                | Orientation::Rotate270
                | Orientation::Transverse
        ),
    )
}

// Step in luma pixels and the first sample location in half-luma units.
fn grid(layout: &PlanarLayout, spec: &PlaneSpec, vertical: bool) -> (i64, Option<i64>) {
    if !spec.role().is_chroma() {
        let step = 1
            << if vertical {
                spec.v_shift()
            } else {
                spec.h_shift()
            };
        return (step, (step == 1).then_some(0));
    }
    let subsampled = match layout.subsampling() {
        ChromaSubsampling::Yuv420 => true,
        ChromaSubsampling::Yuv422 => !vertical,
        ChromaSubsampling::Yuv440 => vertical,
        ChromaSubsampling::Mono | ChromaSubsampling::Yuv444 => false,
    };
    if !subsampled {
        return (1, Some(0));
    }
    let offset = match layout.siting() {
        ChromaSiting::Center => Some(1),
        ChromaSiting::Left => Some(i64::from(vertical)),
        ChromaSiting::TopLeft => Some(0),
        ChromaSiting::Unknown => None,
    };
    (2, offset)
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
