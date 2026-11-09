//! Borrowed native maps; placement does not imply local causal attribution.
use crate::ScoreSpec;
use zenpixels::plane::AnyPlane;

/// Placement of a regular native map grid in input-pixel coordinates.
/// Cells may cover only a cropped interior or extend beyond the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct MapGrid {
    /// Input image width, not the number of map cells.
    pub input_width: u32,
    /// Input image height.
    pub input_height: u32,
    /// Left edge of cell column zero, possibly negative.
    pub origin_x: i64,
    /// Top edge of cell row zero, possibly negative.
    pub origin_y: i64,
    /// Cell width and horizontal step in input pixels; positive.
    pub cell_width: u32,
    /// Cell height and vertical step in input pixels; positive.
    pub cell_height: u32,
}
impl MapGrid {
    /// One cell per input pixel. Checked scoring validates nonempty geometry.
    pub const fn new(input_width: u32, input_height: u32) -> Self {
        Self {
            input_width,
            input_height,
            origin_x: 0,
            origin_y: 0,
            cell_width: 1,
            cell_height: 1,
        }
    }
    /// Set native cell placement without resizing, padding or resampling values.
    pub const fn with_cells(mut self, x: i64, y: i64, width: u32, height: u32) -> Self {
        self.origin_x = x;
        self.origin_y = y;
        self.cell_width = width;
        self.cell_height = height;
        self
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
}
