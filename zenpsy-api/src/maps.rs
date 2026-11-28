//! Borrowed native maps; placement does not imply local causal attribution.
use crate::{ErrorKind, MetricError, MetricResult, ScoreSpec, fail};
use zenpixels::plane::AnyPlane;

/// Placement of a regular native map grid in input-pixel coordinates.
/// Cells may cover only a cropped interior or extend beyond the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct MapGrid {
    /// Input image width, not the number of map cells.
    input_width: u32,
    /// Input image height.
    input_height: u32,
    /// Left edge of cell column zero, possibly negative.
    origin_x: i64,
    /// Top edge of cell row zero, possibly negative.
    origin_y: i64,
    /// Cell width and horizontal step in input pixels; positive.
    cell_width: u32,
    /// Cell height and vertical step in input pixels; positive.
    cell_height: u32,
}
impl MapGrid {
    /// One cell per input pixel; empty input geometry is refused.
    pub fn new(input_width: u32, input_height: u32) -> MetricResult<Self> {
        if input_width == 0 || input_height == 0 {
            return fail(
                MetricError::new(ErrorKind::InvalidConfig).with_message("empty map geometry"),
            );
        }
        Ok(Self {
            input_width,
            input_height,
            origin_x: 0,
            origin_y: 0,
            cell_width: 1,
            cell_height: 1,
        })
    }
    /// Set native cell placement without resizing, padding or resampling values.
    pub fn with_cells(mut self, x: i64, y: i64, width: u32, height: u32) -> MetricResult<Self> {
        if width == 0 || height == 0 {
            return fail(
                MetricError::new(ErrorKind::InvalidConfig).with_message("zero map cell step"),
            );
        }
        self.origin_x = x;
        self.origin_y = y;
        self.cell_width = width;
        self.cell_height = height;
        Ok(self)
    }
    /// Input image width.
    pub const fn input_width(self) -> u32 {
        self.input_width
    }
    /// Input image height.
    pub const fn input_height(self) -> u32 {
        self.input_height
    }
    /// Left edge of the first cell.
    pub const fn origin_x(self) -> i64 {
        self.origin_x
    }
    /// Top edge of the first cell.
    pub const fn origin_y(self) -> i64 {
        self.origin_y
    }
    /// Positive cell width and horizontal step.
    pub const fn cell_width(self) -> u32 {
        self.cell_width
    }
    /// Positive cell height and vertical step.
    pub const fn cell_height(self) -> u32 {
        self.cell_height
    }
    /// Whether the map has one cell per input pixel starting at `(0, 0)`.
    pub const fn is_pixel_aligned(self, map_width: usize, map_height: usize) -> bool {
        self.origin_x == 0
            && self.origin_y == 0
            && self.cell_width == 1
            && self.cell_height == 1
            && map_width == self.input_width as usize
            && map_height == self.input_height as usize
    }
    /// Input-pixel half-open bounds `(x0, y0, x1, y1)` covered by cell `(col, row)`.
    pub const fn cell_bounds(self, col: usize, row: usize) -> (i64, i64, i64, i64) {
        let col_i64 = if col > i64::MAX as usize {
            i64::MAX
        } else {
            col as i64
        };
        let row_i64 = if row > i64::MAX as usize {
            i64::MAX
        } else {
            row as i64
        };
        let x0 = self
            .origin_x
            .saturating_add(col_i64.saturating_mul(self.cell_width as i64));
        let y0 = self
            .origin_y
            .saturating_add(row_i64.saturating_mul(self.cell_height as i64));
        let x1 = x0.saturating_add(self.cell_width as i64);
        let y1 = y0.saturating_add(self.cell_height as i64);
        (x0, y0, x1, y1)
    }
}

/// A borrowed native numeric map, including reduced grids and strided rows.
/// The assessment owns its storage and descriptors. Undefined native samples
/// (including NaN) are preserved; scalar validity rules do not apply to samples.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct MapView<'a> {
    /// Stable map key, ordering and native unit. Unordered diagnostic features
    /// need no quality direction or perfect value.
    pub spec: &'a ScoreSpec,
    /// Original samples, with active width, height and sample stride.
    pub values: AnyPlane<'a>,
    /// Cell placement in the input image, independent of filter support.
    pub grid: MapGrid,
}
impl<'a> MapView<'a> {
    /// Borrow an existing map without copying its samples or metadata.
    pub const fn new(spec: &'a ScoreSpec, values: AnyPlane<'a>, grid: MapGrid) -> Self {
        Self { spec, values, grid }
    }
    /// Borrow the native plane directly when samples are stored as `f32`.
    pub const fn as_f32(&self) -> Option<zenpixels::plane::PlaneRef<'a, f32>> {
        match self.values {
            AnyPlane::F32(plane) => Some(plane),
            _ => None,
        }
    }
    /// Read cell `(col, row)` as `f64`, regardless of native `u8`, `u16` or `f32` storage.
    pub fn sample_f64(&self, col: usize, row: usize) -> Option<f64> {
        match self.values {
            AnyPlane::F32(plane) => plane.row(row)?.get(col).map(|&v| f64::from(v)),
            AnyPlane::U8(plane) => plane.row(row)?.get(col).map(|&v| f64::from(v)),
            AnyPlane::U16(plane) => plane.row(row)?.get(col).map(|&v| f64::from(v)),
            _ => None,
        }
    }
    /// Convert cell `(col, row)` into a nonnegative lower-is-better distortion via [`ScoreSpec::loss`].
    pub fn loss_at(&self, col: usize, row: usize) -> Option<f64> {
        self.spec.loss(self.sample_f64(col, row)?)
    }
}
