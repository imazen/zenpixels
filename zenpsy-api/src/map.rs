//! Per-pixel maps: what the values mean ([`MapPolarity`]), which input
//! pixels each value describes ([`MapGeometry`]), how to ask for one
//! ([`MapRequest`]) and how to fold one into a scalar ([`MapFold`]).

use alloc::format;
use alloc::vec::Vec;

use zenpixels::{PixelBuffer, PixelCow, PixelDescriptor, PixelSlice};

use crate::error::{ErrorKind, MetricError, MetricResult, fail};

/// What a map's values mean: direction and semantics.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MapPolarity {
    /// Higher = more similar (FSIM, HaarPSI, SSIM-family, GMSD `gms`,
    /// VSI). `1` = identical.
    Similarity,
    /// Higher = more visible error (butteraugli, cvvdp). `0` = identical.
    Difference,
    /// Probability the difference is detected (HDR-VDP `P_map`), `[0, 1]`.
    DetectionProbability,
    /// A native intermediate with no similarity/difference reading against
    /// the reference: pooling weights (IW-SSIM information content),
    /// information carried by one channel (VIF's numerator or denominator),
    /// a no-reference measurement of one image (CAMBI banding visibility).
    /// The map's name and its metric's docs give its meaning and direction;
    /// identity does not fix its value.
    Feature,
}

impl MapPolarity {
    /// `true` when larger values mean a worse match. `false` for
    /// [`MapPolarity::Feature`], which has no match direction.
    pub const fn higher_is_worse(self) -> bool {
        matches!(
            self,
            MapPolarity::Difference | MapPolarity::DetectionProbability
        )
    }
}

/// A fold of a map into one scalar. Typed, so there is no string grammar
/// to parse; `Custom` carries metric-defined folds.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MapFold<'a> {
    /// `Σv / n`.
    Mean,
    /// Population standard deviation (GMSD's headline over `gms`).
    Std,
    /// Largest value.
    Max,
    /// Smallest value.
    Min,
    /// `(Σ|v|^p / n)^(1/p)`; `p = ∞` is `max |v|`. Butteraugli's combine
    /// is `p = 12`.
    PNorm {
        /// Exponent, `> 0`.
        p: f64,
    },
    /// Fraction of values `≥ threshold` (HDR-VDP's visible fraction is
    /// `0.5` over `P_map`).
    FractionAbove {
        /// Inclusion threshold.
        threshold: f64,
    },
    /// A metric-defined fold, by name.
    Custom(&'a str),
}

/// How a subsampled map combines the native cells it covers.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MapReduce {
    /// Box mean.
    #[default]
    Mean,
    /// Largest value (keeps peak errors visible in a difference map).
    Max,
    /// Smallest value (keeps worst similarity visible).
    Min,
}

/// Which map [`crate::EvalOpts`] asks for.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MapRequest {
    /// The metric's own resolution, the map its reference publishes
    /// (GMSD's `w/2 × h/2`, MS-SSIM's `(w−10) × (h−10)` interior).
    #[default]
    Native,
    /// The native map reduced over `factor × factor` blocks of cells.
    /// Edge blocks reduce what remains.
    Subsampled {
        /// Block size in native cells, `≥ 1`.
        factor: u32,
        /// How each block combines.
        reduce: MapReduce,
    },
}

impl MapRequest {
    /// `Subsampled { factor, reduce: Mean }`.
    pub const fn subsampled(factor: u32) -> Self {
        MapRequest::Subsampled {
            factor,
            reduce: MapReduce::Mean,
        }
    }
}

/// How far beyond its own cell a map value's inputs reach.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Support {
    /// The value depends on its own cell plus `halo_*` pixels on each
    /// side (filter radii, pyramid footprints).
    Local {
        /// Extra pixels left and right.
        halo_x: u32,
        /// Extra pixels above and below.
        halo_y: u32,
    },
    /// The value can depend on any pixel (FFT-domain filters, global
    /// normalization).
    Global,
}

/// Which input pixels each map cell describes.
///
/// Cell `(i, j)` *owns* the input rectangle starting at
/// `(origin_x + i·cell_width, origin_y + j·cell_height)` of size
/// `cell_width × cell_height`, clipped to the image: the pixels to paint
/// when overlaying the map. [`Support`] says which further pixels its value
/// depends on. The origin may be negative when cell 0 is centered before
/// pixel 0, and cells may not cover the whole image (MS-SSIM's valid
/// interior).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct MapGeometry {
    /// Input image width.
    pub input_width: u32,
    /// Input image height.
    pub input_height: u32,
    /// Left edge of cell column 0, in input pixels.
    pub origin_x: i64,
    /// Top edge of cell row 0, in input pixels.
    pub origin_y: i64,
    /// Cell width and horizontal step, `≥ 1`.
    pub cell_width: u32,
    /// Cell height and vertical step, `≥ 1`.
    pub cell_height: u32,
    /// Map width in cells.
    pub cells_x: u32,
    /// Map height in cells.
    pub cells_y: u32,
    /// Dependence beyond the owned cell.
    pub support: Support,
}

/// A clipped input rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct PixelRect {
    /// Left.
    pub x: u32,
    /// Top.
    pub y: u32,
    /// Width, `≥ 1`.
    pub width: u32,
    /// Height, `≥ 1`.
    pub height: u32,
}

impl MapGeometry {
    /// One cell per input pixel, depending on that pixel alone.
    pub const fn identity(width: u32, height: u32) -> Self {
        Self::grid(width, height, 0, 0, 1, 1, width, height)
    }

    /// A regular grid with no halo; add one with [`Self::with_support`].
    #[allow(clippy::too_many_arguments)]
    pub const fn grid(
        input_width: u32,
        input_height: u32,
        origin_x: i64,
        origin_y: i64,
        cell_width: u32,
        cell_height: u32,
        cells_x: u32,
        cells_y: u32,
    ) -> Self {
        Self {
            input_width,
            input_height,
            origin_x,
            origin_y,
            cell_width,
            cell_height,
            cells_x,
            cells_y,
            support: Support::Local {
                halo_x: 0,
                halo_y: 0,
            },
        }
    }

    /// Set the support.
    #[must_use]
    pub const fn with_support(mut self, support: Support) -> Self {
        self.support = support;
        self
    }

    /// Shorthand for a local halo.
    #[must_use]
    pub const fn with_halo(self, halo_x: u32, halo_y: u32) -> Self {
        self.with_support(Support::Local { halo_x, halo_y })
    }

    /// The geometry after [`MapRequest::Subsampled`] by `factor`: each cell
    /// owns the `factor × factor` block of native cells it reduces, counts
    /// round up, support is unchanged. `factor` 0 is treated as 1.
    pub const fn subsampled(self, factor: u32) -> Self {
        let f = if factor == 0 { 1 } else { factor };
        Self {
            cell_width: self.cell_width.saturating_mul(f),
            cell_height: self.cell_height.saturating_mul(f),
            cells_x: self.cells_x.div_ceil(f),
            cells_y: self.cells_y.div_ceil(f),
            ..self
        }
    }

    fn clip(&self, x0: i64, y0: i64, x1: i64, y1: i64) -> Option<PixelRect> {
        let (x1, y1) = (
            x1.min(i64::from(self.input_width)),
            y1.min(i64::from(self.input_height)),
        );
        let (x0, y0) = (x0.max(0), y0.max(0));
        (x1 > x0 && y1 > y0).then(|| PixelRect {
            x: x0 as u32,
            y: y0 as u32,
            width: (x1 - x0) as u32,
            height: (y1 - y0) as u32,
        })
    }

    fn own(&self, i: u32, j: u32) -> Option<(i64, i64, i64, i64)> {
        if i >= self.cells_x || j >= self.cells_y {
            return None;
        }
        let x0 = self.origin_x + i64::from(i) * i64::from(self.cell_width);
        let y0 = self.origin_y + i64::from(j) * i64::from(self.cell_height);
        Some((
            x0,
            y0,
            x0 + i64::from(self.cell_width),
            y0 + i64::from(self.cell_height),
        ))
    }

    /// The input pixels cell `(i, j)` owns, clipped; `None` when out of
    /// range or wholly outside the image.
    pub fn cell_rect(&self, i: u32, j: u32) -> Option<PixelRect> {
        let (x0, y0, x1, y1) = self.own(i, j)?;
        self.clip(x0, y0, x1, y1)
    }

    /// The input pixels cell `(i, j)`'s value depends on, clipped: the
    /// owned cell grown by the halo, or the whole image for
    /// [`Support::Global`].
    pub fn support_rect(&self, i: u32, j: u32) -> Option<PixelRect> {
        let (x0, y0, x1, y1) = self.own(i, j)?;
        match self.support {
            Support::Local { halo_x, halo_y } => {
                let (hx, hy) = (i64::from(halo_x), i64::from(halo_y));
                self.clip(x0 - hx, y0 - hy, x1 + hx, y1 + hy)
            }
            Support::Global => self.clip(
                0,
                0,
                i64::from(self.input_width),
                i64::from(self.input_height),
            ),
        }
    }

    /// The cell that owns input pixel `(x, y)`, if any.
    pub fn cell_at(&self, x: u32, y: u32) -> Option<(u32, u32)> {
        if self.cell_width == 0 || self.cell_height == 0 {
            return None;
        }
        let dx = i64::from(x) - self.origin_x;
        let dy = i64::from(y) - self.origin_y;
        if dx < 0 || dy < 0 {
            return None;
        }
        let i = dx / i64::from(self.cell_width);
        let j = dy / i64::from(self.cell_height);
        (i < i64::from(self.cells_x) && j < i64::from(self.cells_y)).then_some((i as u32, j as u32))
    }
}

/// One map an evaluation publishes.
#[non_exhaustive]
#[derive(Debug)]
pub struct MapView<'a> {
    /// Stable map name within the metric (`"similarity"`, `"p_det"`).
    pub name: &'static str,
    /// `GRAYF32_LINEAR` values, `cells_x × cells_y`.
    pub pixels: PixelCow<'a>,
    /// What the values mean.
    pub polarity: MapPolarity,
    /// Which input pixels each value describes.
    pub geometry: MapGeometry,
    /// The request this map answers.
    pub served: MapRequest,
}

impl<'a> MapView<'a> {
    /// A native-resolution map.
    ///
    /// # Errors
    /// [`ErrorKind::Internal`] when `pixels` isn't `GRAYF32_LINEAR` or its
    /// size isn't the geometry's cell count — a metric bug, not bad input.
    #[track_caller]
    pub fn new(
        name: &'static str,
        pixels: impl Into<PixelCow<'a>>,
        polarity: MapPolarity,
        geometry: MapGeometry,
    ) -> MetricResult<Self> {
        let pixels = pixels.into();
        let sl = pixels.as_slice();
        if sl.descriptor() != PixelDescriptor::GRAYF32_LINEAR {
            return fail(
                MetricError::new(ErrorKind::Internal)
                    .with_message(format!("map {name}: descriptor {:?}", sl.descriptor())),
            );
        }
        if (sl.width(), sl.rows()) != (geometry.cells_x, geometry.cells_y) {
            return fail(MetricError::new(ErrorKind::Internal).with_message(format!(
                "map {name}: {}x{} values, geometry says {}x{}",
                sl.width(),
                sl.rows(),
                geometry.cells_x,
                geometry.cells_y
            )));
        }
        Ok(Self {
            name,
            pixels,
            polarity,
            geometry,
            served: MapRequest::Native,
        })
    }

    /// Map width in cells.
    pub fn width(&self) -> u32 {
        self.geometry.cells_x
    }

    /// Map height in cells.
    pub fn height(&self) -> u32 {
        self.geometry.cells_y
    }

    /// The values, row-major and tightly packed: borrowed when the
    /// storage already is, copied otherwise.
    pub fn values(&self) -> alloc::borrow::Cow<'_, [f32]> {
        f32_values(&self.pixels.as_slice())
    }

    /// An owned copy that no longer borrows the evaluation.
    pub fn into_owned(self) -> MapView<'static> {
        MapView {
            name: self.name,
            pixels: PixelCow::Owned(self.pixels.into_owned()),
            polarity: self.polarity,
            geometry: self.geometry,
            served: self.served,
        }
    }
}

/// Read a `GRAYF32_LINEAR` slice as packed `f32`s without panicking on
/// stride or alignment.
fn f32_values<'s>(sl: &PixelSlice<'s>) -> alloc::borrow::Cow<'s, [f32]> {
    let (w, h) = (sl.width() as usize, sl.rows() as usize);
    if let Some(bytes) = sl.as_contiguous_bytes().and_then(|b| b.get(..w * h * 4))
        && let Ok(v) = bytemuck::try_cast_slice::<u8, f32>(bytes)
    {
        return alloc::borrow::Cow::Borrowed(v);
    }
    let all = sl.as_strided_bytes();
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let row = &all[y * sl.stride()..y * sl.stride() + w * 4];
        out.extend(
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_ne_bytes(*c)),
        );
    }
    alloc::borrow::Cow::Owned(out)
}

/// Fold a map. NaN anywhere yields NaN (a map with a NaN is not a
/// number to rank by).
///
/// # Errors
/// [`ErrorKind::InvalidInput`] for an empty map;
/// [`ErrorKind::InvalidConfig`] for `PNorm` with `p ≤ 0` or NaN;
/// [`ErrorKind::UnsupportedRequest`] for `Custom` (metrics that define
/// custom folds handle them before calling this).
#[track_caller]
pub fn map_fold(map: &MapView<'_>, fold: MapFold<'_>) -> MetricResult<f64> {
    fold_values(&map.values(), fold)
}

/// [`map_fold`] over raw values.
///
/// # Errors
/// As [`map_fold`].
#[track_caller]
pub fn fold_values(vals: &[f32], fold: MapFold<'_>) -> MetricResult<f64> {
    if vals.is_empty() {
        return fail(MetricError::new(ErrorKind::InvalidInput).with_message("empty map"));
    }
    if let MapFold::Custom(name) = fold {
        return fail(
            MetricError::new(ErrorKind::UnsupportedRequest)
                .with_message(format!("custom map fold {name:?}")),
        );
    }
    if let MapFold::PNorm { p } = fold
        && (p.is_nan() || p <= 0.0)
    {
        return fail(
            MetricError::new(ErrorKind::InvalidConfig).with_message(format!("p-norm exponent {p}")),
        );
    }
    if vals.iter().any(|v| v.is_nan()) {
        return Ok(f64::NAN);
    }
    let n = vals.len() as f64;
    let mean = || vals.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    Ok(match fold {
        MapFold::Mean => mean(),
        MapFold::Max => vals
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            .into(),
        MapFold::Min => vals.iter().copied().fold(f32::INFINITY, f32::min).into(),
        MapFold::Std => {
            let m = mean();
            let var = vals
                .iter()
                .map(|&v| {
                    let d = f64::from(v) - m;
                    d * d
                })
                .sum::<f64>()
                / n;
            libm::sqrt(var)
        }
        MapFold::PNorm { p } if p.is_infinite() => {
            vals.iter().map(|v| f64::from(v.abs())).fold(0.0, f64::max)
        }
        MapFold::PNorm { p } => {
            let sum: f64 = vals.iter().map(|&v| libm::pow(f64::from(v).abs(), p)).sum();
            libm::pow(sum / n, p.recip())
        }
        MapFold::FractionAbove { threshold } => {
            vals.iter().filter(|&&v| f64::from(v) >= threshold).count() as f64 / n
        }
        MapFold::Custom(_) => unreachable!("handled above"),
    })
}

/// Reduce a packed `w × h` map over `factor × factor` blocks. Returns
/// `(values, w_out, h_out)` with `w_out = ⌈w/factor⌉`. `factor` 0 is
/// treated as 1. NaN propagates through `Mean`; `Max`/`Min` follow
/// `f32::max`/`min` (NaN-ignoring).
///
/// # Errors
/// [`ErrorKind::Internal`] when `map.len() != w·h`.
#[track_caller]
pub fn map_subsample_f32(
    map: &[f32],
    w: u32,
    h: u32,
    factor: u32,
    reduce: MapReduce,
) -> MetricResult<(Vec<f32>, u32, u32)> {
    let f = factor.max(1) as usize;
    let (w, h) = (w as usize, h as usize);
    if map.len() != w * h {
        return fail(
            MetricError::new(ErrorKind::Internal)
                .with_message(format!("map len {} != {w}x{h}", map.len())),
        );
    }
    let (wo, ho) = (w.div_ceil(f), h.div_ceil(f));
    let mut out = Vec::with_capacity(wo * ho);
    for oy in 0..ho {
        let (y0, y1) = (oy * f, (oy * f + f).min(h));
        for ox in 0..wo {
            let (x0, x1) = (ox * f, (ox * f + f).min(w));
            let block = (y0..y1).flat_map(|y| map[y * w + x0..y * w + x1].iter().copied());
            let v = match reduce {
                MapReduce::Mean => {
                    let (s, c) = block.fold((0.0f64, 0u32), |(s, c), v| (s + f64::from(v), c + 1));
                    (s / f64::from(c)) as f32
                }
                MapReduce::Max => block.fold(f32::NEG_INFINITY, f32::max),
                MapReduce::Min => block.fold(f32::INFINITY, f32::min),
            };
            out.push(v);
        }
    }
    Ok((out, wo as u32, ho as u32))
}

/// View packed `w × h` values as a `GRAYF32_LINEAR` slice.
///
/// # Errors
/// [`ErrorKind::Internal`] when `data.len() != w·h`.
#[track_caller]
pub fn gray_f32_slice(data: &[f32], w: u32, h: u32) -> MetricResult<PixelSlice<'_>> {
    if data.len() != w as usize * h as usize {
        return fail(
            MetricError::new(ErrorKind::Internal)
                .with_message(format!("map len {} != {w}x{h}", data.len())),
        );
    }
    PixelSlice::new(
        bytemuck::cast_slice(data),
        w,
        h,
        w as usize * 4,
        PixelDescriptor::GRAYF32_LINEAR,
    )
    .map_err(|e| e.map_error(|b| MetricError::from_detail(ErrorKind::Internal, b)))
}

/// Move packed `w × h` values into an owned `GRAYF32_LINEAR` buffer
/// (aligned by construction; no reinterpreting casts).
///
/// # Errors
/// [`ErrorKind::Internal`] when `data.len() != w·h`;
/// [`ErrorKind::Resource`] when the allocation fails.
#[track_caller]
pub fn gray_f32_buffer(data: &[f32], w: u32, h: u32) -> MetricResult<PixelBuffer> {
    if data.len() != w as usize * h as usize {
        return fail(
            MetricError::new(ErrorKind::Internal)
                .with_message(format!("map len {} != {w}x{h}", data.len())),
        );
    }
    let mut buf = PixelBuffer::try_new(w, h, PixelDescriptor::GRAYF32_LINEAR)
        .map_err(|e| e.map_error(|b| MetricError::from_detail(ErrorKind::Resource, b)))?;
    let mut dst = buf.as_slice_mut();
    for y in 0..h {
        let src = &data[y as usize * w as usize..(y as usize + 1) * w as usize];
        for (o, &v) in dst.row_mut(y).as_chunks_mut::<4>().0.iter_mut().zip(src) {
            o.copy_from_slice(&v.to_ne_bytes());
        }
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn view(vals: &[f32], w: u32, h: u32) -> MapView<'static> {
        MapView::new(
            "m",
            gray_f32_buffer(vals, w, h).unwrap(),
            MapPolarity::Difference,
            MapGeometry::identity(w, h),
        )
        .unwrap()
    }

    #[test]
    fn polarity_directions() {
        assert!(MapPolarity::Difference.higher_is_worse());
        assert!(MapPolarity::DetectionProbability.higher_is_worse());
        assert!(!MapPolarity::Similarity.higher_is_worse());
        assert!(!MapPolarity::Feature.higher_is_worse());
    }

    #[test]
    fn folds() {
        let m = view(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!(map_fold(&m, MapFold::Mean).unwrap(), 2.5);
        assert_eq!(map_fold(&m, MapFold::Max).unwrap(), 4.0);
        assert_eq!(map_fold(&m, MapFold::Min).unwrap(), 1.0);
        assert_eq!(
            map_fold(&m, MapFold::PNorm { p: f64::INFINITY }).unwrap(),
            4.0
        );
        assert!((map_fold(&m, MapFold::PNorm { p: 1.0 }).unwrap() - 2.5).abs() < 1e-12);
        assert_eq!(
            map_fold(&m, MapFold::FractionAbove { threshold: 3.0 }).unwrap(),
            0.5
        );
        assert!((map_fold(&m, MapFold::Std).unwrap() - 1.25f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn fold_errors_and_nan() {
        let m = view(&[1.0, f32::NAN], 2, 1);
        assert!(map_fold(&m, MapFold::Max).unwrap().is_nan());
        let k = |f| map_fold(&view(&[1.0], 1, 1), f).unwrap_err().error().kind();
        assert_eq!(k(MapFold::PNorm { p: 0.0 }), ErrorKind::InvalidConfig);
        assert_eq!(k(MapFold::PNorm { p: f64::NAN }), ErrorKind::InvalidConfig);
        assert_eq!(k(MapFold::Custom("c_max")), ErrorKind::UnsupportedRequest);
        assert_eq!(
            fold_values(&[], MapFold::Mean).unwrap_err().error().kind(),
            ErrorKind::InvalidInput
        );
    }

    #[test]
    fn strided_map_folds_every_row() {
        // 2x2 map in rows of 3 floats (one padding float per row).
        let backing = [1.0f32, 2.0, 99.0, 3.0, 4.0, 99.0];
        let sl = PixelSlice::new(
            bytemuck::cast_slice(&backing),
            2,
            2,
            12,
            PixelDescriptor::GRAYF32_LINEAR,
        )
        .unwrap();
        let m = MapView::new(
            "m",
            sl,
            MapPolarity::Difference,
            MapGeometry::identity(2, 2),
        )
        .unwrap();
        assert_eq!(&*m.values(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(map_fold(&m, MapFold::Max).unwrap(), 4.0);
    }

    #[test]
    fn view_rejects_geometry_mismatch() {
        let b = gray_f32_buffer(&[0.0; 6], 3, 2).unwrap();
        let e =
            MapView::new("m", b, MapPolarity::Similarity, MapGeometry::identity(2, 3)).unwrap_err();
        assert_eq!(e.error().kind(), ErrorKind::Internal);
    }

    #[test]
    fn subsample_reduces_and_clips_edges() {
        let m: Vec<f32> = (0..15).map(|v| v as f32).collect(); // 5x3
        let (mean, w, h) = map_subsample_f32(&m, 5, 3, 2, MapReduce::Mean).unwrap();
        assert_eq!((w, h), (3, 2));
        assert_eq!(mean, vec![3.0, 5.0, 6.5, 10.5, 12.5, 14.0]);
        let (max, ..) = map_subsample_f32(&m, 5, 3, 2, MapReduce::Max).unwrap();
        assert_eq!(max, vec![6.0, 8.0, 9.0, 11.0, 13.0, 14.0]);
        assert!(map_subsample_f32(&m, 4, 3, 2, MapReduce::Mean).is_err());
    }

    #[test]
    fn geometry_cells_support_and_subsampling() {
        // GMSD-like: 2x2 cells from pixel 0, 3 cells across a 7-wide image,
        // a 3x3 filter on the half-res grid (halo 2 input pixels).
        let g = MapGeometry::grid(7, 4, 0, 0, 2, 2, 3, 2).with_halo(2, 2);
        assert_eq!(
            g.cell_rect(2, 1),
            Some(PixelRect {
                x: 4,
                y: 2,
                width: 2,
                height: 2
            })
        );
        assert_eq!(
            g.support_rect(0, 0),
            Some(PixelRect {
                x: 0,
                y: 0,
                width: 4,
                height: 4
            })
        );
        assert_eq!(g.cell_at(6, 0), None);
        assert_eq!(g.cell_at(5, 3), Some((2, 1)));
        let s = g.subsampled(2);
        assert_eq!((s.cells_x, s.cells_y, s.cell_width), (2, 1, 4));
        assert_eq!(
            s.cell_rect(1, 0),
            Some(PixelRect {
                x: 4,
                y: 0,
                width: 3,
                height: 4
            })
        );
        assert_eq!(s.support, g.support);
        // Negative origin clips at the image edge.
        let c = MapGeometry::grid(10, 10, -1, -1, 3, 3, 4, 4);
        assert_eq!(
            c.cell_rect(0, 0),
            Some(PixelRect {
                x: 0,
                y: 0,
                width: 2,
                height: 2
            })
        );
        let gl = c.with_support(Support::Global);
        assert_eq!(
            gl.support_rect(3, 3),
            Some(PixelRect {
                x: 0,
                y: 0,
                width: 10,
                height: 10
            })
        );
        assert_eq!(
            MapGeometry::identity(4, 4).subsampled(0),
            MapGeometry::identity(4, 4)
        );
    }
}
