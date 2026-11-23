//! Band-generic packed pixel source and sink traits (`PackedSource`, `PackedSink`)
//! and unified `orient_into`.
//!
//! Each trait is sealed to `zenpixels`'s packed carriers (`PixelSlice`, `PixelSliceMut`,
//! `PixelBuffer`, and `BandedPixelBuffer`) and lowers at `#[inline]` entry to a lightweight
//! internal enum (`SourceView` / `SinkView`). Consequently:
//!
//! - Contiguous `(PixelSlice / &PixelBuffer, PixelSliceMut / &mut PixelBuffer)` calls
//!   constant-fold straight to the single-slice execution path with zero band-iterator
//!   overhead and zero per-carrier monomorphization of row kernels.
//! - Multi-band and mixed `(Single <-> Banded)` calls stream across band intersections
//!   with zero whole-frame intermediate allocation.
//! - Owned destinations (`&mut PixelBuffer`, `&mut BandedPixelBuffer`) receive updated
//!   [`ColorContext`](zenpixels::ColorContext) and sample evidence (`ExactDepth`,
//!   `FloatFacts`) on success. Borrowed mutable views (`&mut PixelSliceMut`) receive
//!   updated [`ColorContext`](zenpixels::ColorContext) only (sample evidence lives on
//!   owned buffers). Both remain untouched when preflight validation fails.

use alloc::sync::Arc;
use whereat::At;
use zenpixels::sample::{ExactDepth, FloatFacts};
use zenpixels::{
    BandedPixelBuffer, ByteStorage, ByteStorageMut, ColorContext, Orientation, PixelBuffer,
    PixelDescriptor, PixelEncoding, PixelSlice, PixelSliceMut,
};

use crate::{ConvertError, OutputPlan, RowConverter};

pub(crate) mod private {
    use super::*;

    #[derive(Clone)]
    pub enum SourceView<'a> {
        Single(PixelSlice<'a>),
        Banded(&'a BandedPixelBuffer),
    }

    impl<'a> SourceView<'a> {
        #[inline]
        pub fn width(&self) -> u32 {
            match self {
                Self::Single(s) => s.width(),
                Self::Banded(b) => b.width(),
            }
        }

        #[inline]
        pub fn height(&self) -> u32 {
            match self {
                Self::Single(s) => s.rows(),
                Self::Banded(b) => b.height(),
            }
        }

        #[inline]
        pub fn descriptor(&self) -> PixelDescriptor {
            match self {
                Self::Single(s) => s.descriptor(),
                Self::Banded(b) => b.descriptor(),
            }
        }

        #[inline]
        pub fn color_context(&self) -> Option<&Arc<ColorContext>> {
            match self {
                Self::Single(s) => s.color_context(),
                Self::Banded(b) => b.color_context(),
            }
        }

        #[inline]
        pub fn encoding(&self) -> PixelEncoding<'_> {
            match self {
                Self::Single(s) => s.encoding(),
                Self::Banded(b) => b.encoding(),
            }
        }

        #[inline]
        pub fn exact_depth(&self) -> Option<ExactDepth> {
            match self {
                Self::Single(s) => s.exact_depth(),
                Self::Banded(b) => b.exact_depth(),
            }
        }

        #[inline]
        pub fn float_facts(&self) -> FloatFacts {
            match self {
                Self::Single(s) => s.float_facts(),
                Self::Banded(b) => b.float_facts(),
            }
        }

        #[inline]
        pub fn row(&self, y: u32) -> &[u8] {
            match self {
                Self::Single(s) => s.row(y),
                Self::Banded(b) => b.row(y).expect("row in bounds"),
            }
        }
    }

    pub enum SinkView<'a> {
        Single(PixelSliceMut<'a>),
        Banded(&'a mut BandedPixelBuffer),
    }

    #[derive(Default)]
    pub struct SinkStamp {
        pub wants_stamp: bool,
        pub apply: bool,
        pub color: Option<Arc<ColorContext>>,
        pub exact_depth: Option<ExactDepth>,
        pub float_facts: FloatFacts,
    }

    pub trait SealedSource: Sized {
        fn with_source<R>(self, f: impl FnOnce(SourceView<'_>) -> R) -> R;
    }

    pub trait SealedSink: Sized {
        fn sink_meta(&self) -> (u32, u32, PixelDescriptor);
        fn with_sink<R>(self, f: impl FnOnce(SinkView<'_>, &mut SinkStamp) -> R) -> R;
    }
}

/// Sealed read-only packed pixel carrier accepted by [`RowConverter::convert`],
/// [`OutputPlan::convert_into`], and [`orient_into`].
///
/// Implemented for [`PixelSlice`], `&PixelSlice`, `&PixelBuffer<P, S>`, and
/// `&BandedPixelBuffer`.
pub trait PackedSource: private::SealedSource {}

/// Sealed writable packed pixel carrier accepted by [`RowConverter::convert`],
/// [`OutputPlan::convert_into`], and [`orient_into`].
///
/// Implemented for [`PixelSliceMut`], `&mut PixelSliceMut`, `&mut PixelBuffer<P, S>`,
/// and `&mut BandedPixelBuffer`.
pub trait PackedSink: private::SealedSink {}

impl<P> private::SealedSource for PixelSlice<'_, P> {
    #[inline]
    fn with_source<R>(self, f: impl FnOnce(private::SourceView<'_>) -> R) -> R {
        f(private::SourceView::Single(self.erase()))
    }
}
impl<P> PackedSource for PixelSlice<'_, P> {}

impl<P> private::SealedSource for &PixelSlice<'_, P> {
    #[inline]
    fn with_source<R>(self, f: impl FnOnce(private::SourceView<'_>) -> R) -> R {
        f(private::SourceView::Single((*self).clone().erase()))
    }
}
impl<P> PackedSource for &PixelSlice<'_, P> {}

impl<P, S: ByteStorage> private::SealedSource for &PixelBuffer<P, S> {
    #[inline]
    fn with_source<R>(self, f: impl FnOnce(private::SourceView<'_>) -> R) -> R {
        f(private::SourceView::Single(self.as_slice().erase()))
    }
}
impl<P, S: ByteStorage> PackedSource for &PixelBuffer<P, S> {}

impl private::SealedSource for &BandedPixelBuffer {
    #[inline]
    fn with_source<R>(self, f: impl FnOnce(private::SourceView<'_>) -> R) -> R {
        f(private::SourceView::Banded(self))
    }
}
impl PackedSource for &BandedPixelBuffer {}

impl<P> private::SealedSink for PixelSliceMut<'_, P> {
    #[inline]
    fn sink_meta(&self) -> (u32, u32, PixelDescriptor) {
        (self.width(), self.rows(), self.descriptor())
    }

    #[inline]
    fn with_sink<R>(
        self,
        f: impl FnOnce(private::SinkView<'_>, &mut private::SinkStamp) -> R,
    ) -> R {
        let mut stamp = private::SinkStamp::default();
        f(private::SinkView::Single(self.erase()), &mut stamp)
    }
}
impl<P> PackedSink for PixelSliceMut<'_, P> {}

impl<P> private::SealedSink for &mut PixelSliceMut<'_, P> {
    #[inline]
    fn sink_meta(&self) -> (u32, u32, PixelDescriptor) {
        (self.width(), self.rows(), self.descriptor())
    }

    #[inline]
    fn with_sink<R>(
        self,
        f: impl FnOnce(private::SinkView<'_>, &mut private::SinkStamp) -> R,
    ) -> R {
        let mut stamp = private::SinkStamp {
            wants_stamp: true,
            ..Default::default()
        };
        let rows = self.rows();
        let out = {
            let sub = self.sub_rows_mut(0, rows).erase();
            f(private::SinkView::Single(sub), &mut stamp)
        };
        if stamp.apply {
            self.set_color_context(stamp.color);
        }
        out
    }
}
impl<P> PackedSink for &mut PixelSliceMut<'_, P> {}

impl<P, S: ByteStorageMut> private::SealedSink for &mut PixelBuffer<P, S> {
    #[inline]
    fn sink_meta(&self) -> (u32, u32, PixelDescriptor) {
        (self.width(), self.height(), self.descriptor())
    }

    #[inline]
    fn with_sink<R>(
        self,
        f: impl FnOnce(private::SinkView<'_>, &mut private::SinkStamp) -> R,
    ) -> R {
        let mut stamp = private::SinkStamp {
            wants_stamp: true,
            ..Default::default()
        };
        let out = f(
            private::SinkView::Single(self.as_slice_mut().erase()),
            &mut stamp,
        );
        if stamp.apply {
            self.set_color_context(stamp.color);
            let _ = self.set_exact_depth(stamp.exact_depth);
            let _ = self.set_float_facts(stamp.float_facts);
        }
        out
    }
}
impl<P, S: ByteStorageMut> PackedSink for &mut PixelBuffer<P, S> {}

impl private::SealedSink for &mut BandedPixelBuffer {
    #[inline]
    fn sink_meta(&self) -> (u32, u32, PixelDescriptor) {
        (self.width(), self.height(), self.descriptor())
    }

    #[inline]
    fn with_sink<R>(
        self,
        f: impl FnOnce(private::SinkView<'_>, &mut private::SinkStamp) -> R,
    ) -> R {
        let mut stamp = private::SinkStamp {
            wants_stamp: true,
            ..Default::default()
        };
        let out = f(private::SinkView::Banded(self), &mut stamp);
        if stamp.apply {
            self.set_color_context(stamp.color);
            let _ = self.set_exact_depth(stamp.exact_depth);
            let _ = self.set_float_facts(stamp.float_facts);
        }
        out
    }
}
impl PackedSink for &mut BandedPixelBuffer {}

// ---------------------------------------------------------------------------
// Unified RowConverter::convert execution
// ---------------------------------------------------------------------------

#[track_caller]
#[inline]
pub(crate) fn convert_packed(
    worker: &mut RowConverter,
    src: impl PackedSource,
    dst: impl PackedSink,
) -> Result<(), At<ConvertError>> {
    let dst_meta = dst.sink_meta();
    src.with_source(|src_view| {
        preflight_worker(worker, &src_view, dst_meta)?;
        dst.with_sink(|dst_view, stamp| convert_packed_inner(worker, src_view, dst_view, stamp))
    })
}

#[track_caller]
fn preflight_worker(
    worker: &RowConverter,
    src: &private::SourceView<'_>,
    (dst_w, dst_h, dst_desc): (u32, u32, PixelDescriptor),
) -> Result<(), At<ConvertError>> {
    let from = worker.from_descriptor();
    let to = worker.to_descriptor();
    if src.descriptor() != from {
        return Err(whereat::at!(ConvertError::NoPath {
            from: src.descriptor(),
            to,
        }));
    }
    if src.encoding() != worker.expected_source_encoding() {
        return Err(whereat::at!(ConvertError::SourceInterpretationChanged));
    }
    if !src.float_facts().contains(worker.input_float_facts()) {
        return Err(whereat::at!(ConvertError::FloatFactsUnsatisfied {
            required: worker.input_float_facts(),
            actual: src.float_facts(),
        }));
    }
    if dst_desc != to {
        return Err(whereat::at!(ConvertError::NoPath { from, to: dst_desc }));
    }
    if src.width() != dst_w || src.height() != dst_h {
        return Err(whereat::at!(ConvertError::Buffer(
            zenpixels::BufferError::InvalidDimensions
        )));
    }
    if worker
        .prepared_capacity()
        .is_some_and(|cap| src.width() > cap)
    {
        return Err(whereat::at!(ConvertError::InvalidWidth(src.width())));
    }
    Ok(())
}

#[track_caller]
fn convert_packed_inner(
    worker: &mut RowConverter,
    src: private::SourceView<'_>,
    dst: private::SinkView<'_>,
    stamp: &mut private::SinkStamp,
) -> Result<(), At<ConvertError>> {
    use private::{SinkView, SourceView};

    let width = src.width();
    let height = src.height();
    let preserved_exact = if worker.is_identity() && src.descriptor() == worker.to_descriptor() {
        src.exact_depth()
    } else {
        None
    };

    if width > 0 && height > 0 {
        match (src, dst) {
            (SourceView::Single(s), SinkView::Single(mut d)) => {
                let d_stride = d.stride();
                worker.convert_raw_rows(
                    s.as_strided_bytes(),
                    s.stride(),
                    d.as_strided_bytes_mut(),
                    d_stride,
                    width,
                    height,
                )?;
            }
            (SourceView::Banded(s), SinkView::Banded(d)) => {
                let mut s_iter = s.bands();
                let mut d_iter = d.bands_mut();
                let mut cur_s = s_iter.next().map(|b| (b, 0u32));
                let mut cur_d = d_iter.next().map(|b| (b, 0u32));
                let mut y = 0u32;
                while y < height {
                    let (sb, sy) = cur_s.as_ref().expect("source bands cover height");
                    let (db, dy) = cur_d.as_mut().expect("destination bands cover height");
                    let chunk = (sb.rows() - *sy).min(db.rows() - *dy);
                    let s_sub = sb.sub_rows(*sy, chunk);
                    let mut d_sub = db.sub_rows_mut(*dy, chunk);
                    let d_stride = d_sub.stride();
                    worker.convert_raw_rows(
                        s_sub.as_strided_bytes(),
                        s_sub.stride(),
                        d_sub.as_strided_bytes_mut(),
                        d_stride,
                        width,
                        chunk,
                    )?;
                    y += chunk;
                    if *sy + chunk == sb.rows() {
                        cur_s = s_iter.next().map(|b| (b, 0));
                    } else {
                        cur_s.as_mut().unwrap().1 += chunk;
                    }
                    if *dy + chunk == db.rows() {
                        cur_d = d_iter.next().map(|b| (b, 0));
                    } else {
                        *dy += chunk;
                    }
                }
            }
            (SourceView::Single(s), SinkView::Banded(d)) => {
                let mut y = 0u32;
                for mut db in d.bands_mut() {
                    let chunk = db.rows();
                    let s_sub = s.sub_rows(y, chunk);
                    let d_stride = db.stride();
                    worker.convert_raw_rows(
                        s_sub.as_strided_bytes(),
                        s_sub.stride(),
                        db.as_strided_bytes_mut(),
                        d_stride,
                        width,
                        chunk,
                    )?;
                    y += chunk;
                }
            }
            (SourceView::Banded(s), SinkView::Single(mut d)) => {
                let mut y = 0u32;
                for sb in s.bands() {
                    let chunk = sb.rows();
                    let mut d_sub = d.sub_rows_mut(y, chunk);
                    let d_stride = d_sub.stride();
                    worker.convert_raw_rows(
                        sb.as_strided_bytes(),
                        sb.stride(),
                        d_sub.as_strided_bytes_mut(),
                        d_stride,
                        width,
                        chunk,
                    )?;
                    y += chunk;
                }
            }
        }
    }

    if stamp.wants_stamp {
        stamp.apply = true;
        stamp.color = worker.output_color_arc().cloned();
        stamp.exact_depth = preserved_exact;
        stamp.float_facts = worker.output_float_facts();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Unified OutputPlan::convert_into execution
// ---------------------------------------------------------------------------

#[track_caller]
#[inline]
pub(crate) fn convert_output_packed(
    plan: &mut OutputPlan,
    src: impl PackedSource,
    dst: impl PackedSink,
) -> Result<(), At<ConvertError>> {
    let (dst_w, dst_h, dst_desc) = dst.sink_meta();
    src.with_source(|src_view| {
        plan.validate_source_view(&src_view)?;
        if dst_desc != plan.output_descriptor() {
            return Err(whereat::at!(ConvertError::NoPath {
                from: plan.output_descriptor(),
                to: dst_desc,
            }));
        }
        if src_view.width() != dst_w || src_view.height() != dst_h {
            return Err(whereat::at!(ConvertError::Buffer(
                zenpixels::BufferError::InvalidDimensions
            )));
        }
        dst.with_sink(|dst_view, stamp| convert_output_inner(plan, src_view, dst_view, stamp))
    })
}

#[track_caller]
fn convert_output_inner(
    plan: &mut OutputPlan,
    src: private::SourceView<'_>,
    dst: private::SinkView<'_>,
    stamp: &mut private::SinkStamp,
) -> Result<(), At<ConvertError>> {
    use private::{SinkView, SourceView};

    let preserved_exact = if !plan.has_converter() && src.descriptor() == plan.output_descriptor() {
        src.exact_depth()
    } else {
        None
    };

    if let Some(worker) = plan.converter_mut() {
        convert_packed_inner(worker, src, dst, stamp)?;
    } else {
        let width = src.width();
        let height = src.height();
        if width > 0 && height > 0 {
            match (src, dst) {
                (SourceView::Single(s), SinkView::Single(mut d)) => {
                    for y in 0..height {
                        d.row_mut(y).copy_from_slice(s.row(y));
                    }
                }
                (SourceView::Banded(s), SinkView::Banded(d)) => {
                    let mut oy = 0u32;
                    for mut db in d.bands_mut() {
                        for ry in 0..db.rows() {
                            db.row_mut(ry)
                                .copy_from_slice(s.row(oy + ry).expect("row in bounds"));
                        }
                        oy += db.rows();
                    }
                }
                (SourceView::Single(s), SinkView::Banded(d)) => {
                    let mut oy = 0u32;
                    for mut db in d.bands_mut() {
                        for ry in 0..db.rows() {
                            db.row_mut(ry).copy_from_slice(s.row(oy + ry));
                        }
                        oy += db.rows();
                    }
                }
                (SourceView::Banded(s), SinkView::Single(mut d)) => {
                    let mut oy = 0u32;
                    for sb in s.bands() {
                        for ry in 0..sb.rows() {
                            d.row_mut(oy + ry).copy_from_slice(sb.row(ry));
                        }
                        oy += sb.rows();
                    }
                }
            }
        }
    }

    if stamp.wants_stamp {
        stamp.apply = true;
        stamp.color = Some(Arc::clone(plan.output_color_context()));
        stamp.exact_depth = preserved_exact;
        stamp.float_facts = plan.output_float_facts();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Unified orient_into across any PackedSource / PackedSink pair
// ---------------------------------------------------------------------------

/// Physically apply `orientation` from any [`PackedSource`] (`PixelSlice`, `&PixelBuffer`,
/// or `&BandedPixelBuffer`) into any pre-allocated [`PackedSink`] (`PixelSliceMut`,
/// `&mut PixelBuffer`, or `&mut BandedPixelBuffer`) without intermediate heap allocation.
///
/// Axis-swapping orientations (`Transpose`, `Rotate90`, `Rotate270`, `Transverse`) dispatch
/// each `(source_band, destination_band)` intersection rectangle through the cache-blocked
/// SIMD transpose kernels. When `src` and `dst` have identical [`PixelDescriptor`]s,
/// `dst` inherits `src`'s [`ColorContext`], [`ExactDepth`], and [`FloatFacts`].
#[track_caller]
#[inline]
pub fn orient_into(
    src: impl PackedSource,
    orientation: Orientation,
    dst: impl PackedSink,
) -> Result<(), At<ConvertError>> {
    let (dst_w, dst_h, dst_desc) = dst.sink_meta();
    src.with_source(|src_view| {
        let w = src_view.width();
        let h = src_view.height();
        let desc = src_view.descriptor();
        let bpp = desc.bytes_per_pixel();
        let dst_bpp = dst_desc.bytes_per_pixel();
        let (ow, oh) = orientation.output_dimensions(w, h);

        if dst_w != ow || dst_h != oh || dst_bpp != bpp {
            return Err(whereat::at!(ConvertError::BufferSize {
                expected: (ow as usize)
                    .saturating_mul(oh as usize)
                    .saturating_mul(bpp),
                actual: (dst_w as usize)
                    .saturating_mul(dst_h as usize)
                    .saturating_mul(dst_bpp),
            }));
        }
        if dst_desc.pixel_format() != desc.pixel_format() {
            return Err(whereat::at!(ConvertError::NoPath {
                from: desc,
                to: dst_desc,
            }));
        }

        dst.with_sink(|dst_view, stamp| {
            orient_packed_inner(src_view, orientation, dst_view, dst_desc, stamp)
        })
    })
}

#[track_caller]
fn orient_packed_inner(
    src: private::SourceView<'_>,
    orientation: Orientation,
    dst: private::SinkView<'_>,
    dst_desc: PixelDescriptor,
    stamp: &mut private::SinkStamp,
) -> Result<(), At<ConvertError>> {
    use private::{SinkView, SourceView};

    let w = src.width();
    let h = src.height();
    let desc = src.descriptor();
    let bpp = desc.bytes_per_pixel();
    let same_desc = dst_desc == desc;
    let src_color = if stamp.wants_stamp && same_desc {
        src.color_context().cloned()
    } else {
        None
    };
    let src_exact = src.exact_depth();
    let src_facts = src.float_facts();

    match (src, dst) {
        (SourceView::Single(s), SinkView::Single(d)) => {
            crate::orient::apply_orientation_into(s, orientation, d)
                .map_err(|e| whereat::at!(e))?;
        }
        (SourceView::Banded(s), SinkView::Banded(d)) => {
            crate::orient::apply_orientation_banded_into(s, orientation, d)
                .map_err(|e| whereat::at!(e))?;
        }
        (src_mixed, mut dst_mixed) => {
            if w > 0 && h > 0 && bpp > 0 {
                orient_mixed_bands(&src_mixed, orientation, &mut dst_mixed, dst_desc)
                    .map_err(|e| whereat::at!(e))?;
            }
        }
    }

    if stamp.wants_stamp {
        stamp.apply = true;
        if same_desc {
            stamp.color = src_color;
            stamp.exact_depth = src_exact;
            stamp.float_facts = src_facts;
        } else {
            stamp.color = None;
            stamp.exact_depth = None;
            stamp.float_facts = FloatFacts::NONE;
        }
    }
    Ok(())
}

fn orient_mixed_bands(
    src: &private::SourceView<'_>,
    orientation: Orientation,
    dst: &mut private::SinkView<'_>,
    dst_desc: PixelDescriptor,
) -> Result<(), ConvertError> {
    use private::{SinkView, SourceView};

    let w = src.width();
    let h = src.height();
    let desc = src.descriptor();
    let bpp = desc.bytes_per_pixel();

    match orientation {
        Orientation::Identity
        | Orientation::FlipV
        | Orientation::FlipH
        | Orientation::Rotate180 => {
            let write_band = |slice: &mut PixelSliceMut<'_>, oy: u32| {
                for ry in 0..slice.rows() {
                    let sy = match orientation {
                        Orientation::Identity | Orientation::FlipH => oy + ry,
                        Orientation::FlipV | Orientation::Rotate180 => h - 1 - (oy + ry),
                        _ => unreachable!(),
                    };
                    let s_row = src.row(sy);
                    let d_row = slice.row_mut(ry);
                    if matches!(orientation, Orientation::FlipH | Orientation::Rotate180) {
                        crate::orient::reverse_row(s_row, d_row, w as usize, bpp);
                    } else {
                        d_row.copy_from_slice(s_row);
                    }
                }
            };
            match dst {
                SinkView::Single(d) => write_band(d, 0),
                SinkView::Banded(b) => {
                    let mut oy = 0u32;
                    for mut d_slice in b.bands_mut() {
                        let rows = d_slice.rows();
                        write_band(&mut d_slice, oy);
                        oy += rows;
                    }
                }
            }
        }
        Orientation::Transpose
        | Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Transverse => {
            let transpose_into_band = |d_slice: &mut PixelSliceMut<'_>,
                                       oy0: u32|
             -> Result<(), ConvertError> {
                let dh = d_slice.rows();
                let oy1 = oy0 + dh;
                let sx0 = match orientation {
                    Orientation::Transpose | Orientation::Rotate90 => oy0,
                    Orientation::Rotate270 | Orientation::Transverse => w - oy1,
                    _ => unreachable!(),
                };
                let d_stride = d_slice.stride();
                let d_bytes = d_slice.as_strided_bytes_mut();

                let mut transpose_from_band =
                    |s_slice: &PixelSlice<'_>, sy0: u32| -> Result<u32, ConvertError> {
                        let sh = s_slice.rows();
                        let sy1 = sy0 + sh;
                        let ox0 = match orientation {
                            Orientation::Transpose | Orientation::Rotate270 => sy0,
                            Orientation::Rotate90 | Orientation::Transverse => h - sy1,
                            _ => unreachable!(),
                        };
                        let s_stride = s_slice.stride();
                        let s_bytes = s_slice.as_strided_bytes();
                        let s_off = (sx0 as usize) * bpp;
                        let d_off = (ox0 as usize) * bpp;
                        let src_rect = PixelSlice::new(&s_bytes[s_off..], dh, sh, s_stride, desc)
                            .map_err(|e| ConvertError::Buffer(*e.error()))?;
                        let dst_rect =
                            PixelSliceMut::new(&mut d_bytes[d_off..], sh, dh, d_stride, dst_desc)
                                .map_err(|e| ConvertError::Buffer(*e.error()))?;
                        crate::orient::apply_orientation_into(src_rect, orientation, dst_rect)?;
                        Ok(sy1)
                    };

                match src {
                    SourceView::Single(s) => {
                        transpose_from_band(s, 0)?;
                    }
                    SourceView::Banded(sb) => {
                        let mut sy0 = 0u32;
                        for s_slice in sb.bands() {
                            sy0 = transpose_from_band(&s_slice, sy0)?;
                        }
                    }
                }
                Ok(())
            };

            match dst {
                SinkView::Single(d) => transpose_into_band(d, 0)?,
                SinkView::Banded(b) => {
                    let mut oy0 = 0u32;
                    for mut d_slice in b.bands_mut() {
                        let dh = d_slice.rows();
                        transpose_into_band(&mut d_slice, oy0)?;
                        oy0 += dh;
                    }
                }
            }
        }
        _ => {
            for sy in 0..h {
                let s_row = src.row(sy);
                for sx in 0..w {
                    let (ox, oy) = orientation.forward_map(sx, sy, w, h);
                    let s_px = &s_row[sx as usize * bpp..(sx as usize + 1) * bpp];
                    let d_row = match dst {
                        SinkView::Single(d) => d.row_mut(oy),
                        SinkView::Banded(b) => b.row_mut(oy).expect("row in bounds"),
                    };
                    d_row[ox as usize * bpp..(ox as usize + 1) * bpp].copy_from_slice(s_px);
                }
            }
        }
    }
    Ok(())
}
