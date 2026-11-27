//! Spec-aware content analysis and bit-exact lossless format compaction.
//!
//! Answers: *"Which parts of this buffer's declared [`PixelFormat`] are actually
//! required by its pixel values?"* — non-opaque alpha, chroma (`R != G` or `G != B`),
//! and 16-bit precision beyond byte-replicated 8-bit (`(v >> 8) != (v & 0xFF)` on
//! full-range `U16`) — and optionally compacts the pixel buffer either into a fresh
//! [`PixelBuf`] ([`compact_to_buf`]) or **in place** with zero allocation
//! ([`compact_in_place`]).
//!
//! Every reduction this module performs is **bit-exact invertible**:
//! - Dropping an all-max alpha lane (or undefined `RGBX`/`BGRX` padding).
//! - Collapsing `R == G == B` to `Gray`/`GrayAlpha` (respecting ICC class
//!   authority: an RGB-class ICC requires a qualified GRAY-class replacement).
//! - Narrowing bit-replicated full-range `U16` (`v * 257`) to `U8` (narrow-range
//!   `U16` is kept at 16-bit because its anchors scale by 256, not 257).

use alloc::sync::Arc;
use whereat::At;
use zenpixels::{
    AlphaMode, ChannelOrder, ColorOrigin, ColorSpace, InPlaceTransformOutput, PixelBuf,
    PixelFormat, PixelView, Primaries, SampleFacts, SampleLayout, SampleType, SignalRange,
    Transfer,
};

use crate::error::{ConvertError, err_at};
use crate::icc_profiles::{SynthesizedIcc, synthesize_gray_icc_for_cicp};
use crate::scan::{self, FusedRequest};

/// Which channels and precision of a buffer's declared [`PixelFormat`] are required
/// to represent its current pixel values without loss.
///
/// Each field is `Option<bool>`:
/// - `Some(false)` — measured redundant (safe to drop/narrow without loss).
/// - `Some(true)` — measured required (keep).
/// - `None` — predicate did not apply / was not assessed (keep conservatively).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompactionReport {
    /// `Some(true)` if any alpha sample is below channel-max; `Some(false)` if
    /// all-max, `Opaque`, `Undefined` padding, or no alpha channel.
    pub requires_alpha: Option<bool>,
    /// `Some(true)` if any pixel has `R != G` or `G != B`; `Some(false)` if all
    /// neutral or already grayscale.
    pub requires_chroma: Option<bool>,
    /// `Some(true)` if any active full-range `U16` sample is not byte-replicated
    /// (`(v >> 8) != (v & 0xFF)`); `Some(false)` if bit-replicated (`v * 257`) or
    /// already `U8`. Narrow-range `U16` reports `None` and stays 16-bit.
    pub requires_u16_precision: Option<bool>,
}

impl CompactionReport {
    /// Whether at least one predicate ran or answered structurally.
    #[inline]
    #[must_use]
    pub const fn any_analyzed(&self) -> bool {
        self.requires_alpha.is_some()
            || self.requires_chroma.is_some()
            || self.requires_u16_precision.is_some()
    }

    /// Compute the narrowest candidate [`PixelFormat`] justified by this report (metadata only;
    /// does not inspect or transform pixels).
    ///
    /// Order of reduction:
    /// 1. Full-range `U16 -> U8` when `requires_u16_precision == Some(false)`.
    /// 2. Alpha/padding drop (`RGBA/BGRA/RGBX/BGRX -> RGB`, `GrayAlpha -> Gray`)
    ///    when `requires_alpha == Some(false)`.
    /// 3. Chroma drop (`RGB/BGR -> Gray`, `RGBA/BGRA/RGBX/BGRX -> GrayAlpha/Gray`)
    ///    when `requires_chroma == Some(false)` (only on structured D65 `Rgb` / `Gray`;
    ///    `ColorSpace::Icc` chroma collapse is gated separately by ICC class swapping).
    #[must_use]
    pub fn candidate_format(&self, src: &PixelFormat) -> PixelFormat {
        if !src.layout.packing.is_full() {
            return src.clone();
        }
        let mut sample = src.layout.sample;
        let mut order = src.layout.order;
        let mut color = src.color.clone();

        // 1. Full-range U16 -> U8 narrowing.
        if matches!(self.requires_u16_precision, Some(false))
            && sample == SampleType::U16
            && color.range().unwrap_or(SignalRange::Full) == SignalRange::Full
        {
            sample = SampleType::U8;
        }

        // 2. Alpha / padding drop.
        if matches!(self.requires_alpha, Some(false)) {
            let next_order = match order {
                ChannelOrder::Rgba
                | ChannelOrder::Bgra
                | ChannelOrder::Rgbx
                | ChannelOrder::Bgrx => ChannelOrder::Rgb,
                ChannelOrder::GrayAlpha => ChannelOrder::Gray,
                other => other,
            };
            if next_order != order {
                order = next_order;
                color = color.with_alpha(AlphaMode::None);
            }
        }

        // 3. Chroma drop (for structured D65 ColorSpace::Rgb; ColorSpace::Icc requires explicit swap,
        // and non-D65 / Unknown RGB primaries cannot collapse to D65 Gray without shifting white point).
        if matches!(self.requires_chroma, Some(false))
            && let ColorSpace::Rgb {
                primaries,
                transfer,
                range,
                alpha,
                white,
            } = color
            && primaries.white_point() == Some(Primaries::WHITE_D65)
        {
            let (next_order, next_alpha) = match order {
                ChannelOrder::Rgb | ChannelOrder::Bgr => (ChannelOrder::Gray, AlphaMode::None),
                ChannelOrder::Rgbx | ChannelOrder::Bgrx => (ChannelOrder::Gray, AlphaMode::None),
                ChannelOrder::Rgba | ChannelOrder::Bgra => (ChannelOrder::GrayAlpha, alpha),
                other => (other, alpha),
            };
            if next_order != order {
                order = next_order;
                color = ColorSpace::Gray {
                    transfer,
                    range,
                    alpha: next_alpha,
                    white,
                };
            }
        }

        let candidate = PixelFormat::new(SampleLayout::new(sample, order), color);
        if candidate.validate().is_ok() {
            candidate
        } else {
            src.clone()
        }
    }
}

/// How [`compact_in_place`] handles redundant alpha or padding lanes.
///
/// Depth (`U16 → U8`) and chroma (`RGB → Gray`) compaction still run in both modes whenever
/// justified by the scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AlphaCompaction {
    /// Upgrade provably opaque `Straight`/`Premultiplied` alpha to [`AlphaMode::Opaque`]
    /// in metadata without restructuring alpha/padding lanes (0 byte movement when depth and
    /// chroma are unchanged).
    RetagOpaque,
    /// Physically drop redundant alpha or undefined padding lanes in place
    /// (`RGBA`/`BGRA`/`RGBX`/`BGRX` → `RGB`, `GrayAlpha` → `Gray`).
    DropUnusedLanes,
}

enum GraySignalPlan {
    Carry,
    Swap { gray_icc: Option<Arc<[u8]>> },
    Suppress,
}

fn is_d65_white(wp: (f32, f32)) -> bool {
    const D65: (f32, f32) = Primaries::WHITE_D65;
    (wp.0 - D65.0).abs() <= 1e-4 && (wp.1 - D65.1).abs() <= 1e-4
}

fn plan_chroma_collapse_for_icc(icc: &[u8]) -> GraySignalPlan {
    let (cicp, primaries) = if let Some(id) = zenpixels::icc::IccIdentification::from_icc(icc)
        .filter(|id| id.valid_use == zenpixels::icc::IdentificationUse::MatrixTrcSubstitution)
    {
        let Some(c) = id.to_cicp() else {
            return GraySignalPlan::Suppress;
        };
        (c, Some(id.primaries))
    } else if let Some(tag) = zenpixels::CicpTag::from_icc(icc)
        .filter(|t| t.cicp().matrix_coefficients == 0 && t.cicp().full_range)
    {
        (tag.cicp(), tag.primaries())
    } else {
        return GraySignalPlan::Suppress;
    };
    if !primaries
        .and_then(Primaries::white_point)
        .is_some_and(is_d65_white)
    {
        return GraySignalPlan::Suppress;
    }
    match synthesize_gray_icc_for_cicp(cicp) {
        SynthesizedIcc::Profile(bytes) => GraySignalPlan::Swap {
            gray_icc: Some(Arc::from(bytes.as_ref())),
        },
        SynthesizedIcc::NotNeeded => GraySignalPlan::Swap { gray_icc: None },
        _ => GraySignalPlan::Suppress,
    }
}

fn plan_chroma_collapse(spec: &PixelFormat) -> GraySignalPlan {
    match &spec.color {
        ColorSpace::Icc {
            profile,
            color_channels: 3,
            ..
        } => plan_chroma_collapse_for_icc(profile),
        ColorSpace::Rgb { primaries, .. } => {
            if primaries.white_point() == Some(Primaries::WHITE_D65) {
                GraySignalPlan::Carry
            } else {
                GraySignalPlan::Suppress
            }
        }
        ColorSpace::Gray { .. } => GraySignalPlan::Carry,
        _ => GraySignalPlan::Suppress,
    }
}

fn apply_icc_chroma_collapse(
    src: &PixelFormat,
    report: &CompactionReport,
    plan: &GraySignalPlan,
) -> PixelFormat {
    let base = report.candidate_format(src);
    if !matches!(report.requires_chroma, Some(false)) {
        return base;
    }
    let ColorSpace::Icc {
        color_channels: 3,
        alpha,
        intent,
        ..
    } = base.color
    else {
        return base;
    };
    let GraySignalPlan::Swap { gray_icc, .. } = plan else {
        return base;
    };
    let (next_order, next_alpha) = match base.layout.order {
        ChannelOrder::Rgb | ChannelOrder::Bgr | ChannelOrder::Rgbx | ChannelOrder::Bgrx => {
            (ChannelOrder::Gray, AlphaMode::None)
        }
        ChannelOrder::Rgba | ChannelOrder::Bgra => (ChannelOrder::GrayAlpha, alpha),
        other => (other, alpha),
    };
    let next_color = match gray_icc {
        Some(bytes) => ColorSpace::Icc {
            profile: Arc::clone(bytes),
            color_channels: 1,
            alpha: next_alpha,
            intent,
        },
        None => ColorSpace::Gray {
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: next_alpha,
            white: None,
        },
    };
    let candidate = PixelFormat::new(
        SampleLayout::new(base.layout.sample, next_order),
        next_color,
    );
    if candidate.validate().is_ok() {
        candidate
    } else {
        base
    }
}

/// Measure which dimensions of `view`'s spec are actually required by its pixel values.
#[must_use]
pub fn analyze_compaction(view: &PixelView<'_>) -> CompactionReport {
    let spec = view.format();
    if !spec.layout.packing.is_full() {
        return CompactionReport::default();
    }
    let order = spec.layout.order;
    let sample = spec.layout.sample;

    let alpha_structural: Option<Option<bool>> = if order.has_padding_slot() {
        Some(Some(false))
    } else if order.has_alpha_slot() {
        match spec.color.alpha() {
            AlphaMode::Undefined | AlphaMode::Opaque => Some(Some(false)),
            _ => None,
        }
    } else {
        None
    };
    let scan_alpha = alpha_structural.is_none();

    // Fused U16 single-pass scan (opacity + chroma + bit-replication, ignoring RGBX padding).
    if sample == SampleType::U16
        && matches!(
            order,
            ChannelOrder::Rgba
                | ChannelOrder::Bgra
                | ChannelOrder::Rgbx
                | ChannelOrder::Bgrx
                | ChannelOrder::Rgb
                | ChannelOrder::Bgr
                | ChannelOrder::GrayAlpha
                | ChannelOrder::Gray
        )
    {
        let has_alpha = order.has_alpha_slot();
        let has_chroma = order.is_rgb_family();
        let padding = order.has_padding_slot();
        let is_narrow = spec.color.range() == Some(SignalRange::Narrow);
        let ch = order.channels();
        let mut flags = [false, false, is_narrow];
        for y in 0..view.height() {
            let row = view.row(y);
            if let Ok(s16) = bytemuck::try_cast_slice::<u8, u16>(row) {
                flags =
                    scan::fused_u16(s16, ch, has_alpha && scan_alpha, has_chroma, padding, flags);
            } else {
                for px in row.chunks_exact(ch * 2) {
                    if has_alpha && scan_alpha && !flags[0] {
                        let a = u16::from_ne_bytes([px[(ch - 1) * 2], px[(ch - 1) * 2 + 1]]);
                        if a != u16::MAX {
                            flags[0] = true;
                        }
                    }
                    if has_chroma && !flags[1] {
                        let s0 = u16::from_ne_bytes([px[0], px[1]]);
                        let s1 = u16::from_ne_bytes([px[2], px[3]]);
                        let s2 = u16::from_ne_bytes([px[4], px[5]]);
                        if s0 != s1 || s1 != s2 {
                            flags[1] = true;
                        }
                    }
                    if !flags[2] {
                        let data_ch = if padding { ch - 1 } else { ch };
                        for c in 0..data_ch {
                            let v = u16::from_ne_bytes([px[c * 2], px[c * 2 + 1]]);
                            if (v >> 8) != (v & 0xFF) {
                                flags[2] = true;
                                break;
                            }
                        }
                    }
                }
            }
            if (!has_alpha || !scan_alpha || flags[0]) && (!has_chroma || flags[1]) && flags[2] {
                break;
            }
        }
        return CompactionReport {
            requires_alpha: Some(flags[0]),
            requires_chroma: Some(flags[1]),
            requires_u16_precision: if is_narrow { None } else { Some(flags[2]) },
        };
    }

    let (mut requires_alpha, requires_chroma) = match (order, sample) {
        (
            ChannelOrder::Rgba | ChannelOrder::Bgra | ChannelOrder::Rgbx | ChannelOrder::Bgrx,
            SampleType::U8,
        ) => {
            let fused = fused_rgba8_over_rows(
                view,
                FusedRequest {
                    check_opaque: scan_alpha,
                    check_grayscale: true,
                },
            );
            (Some(!fused.is_opaque), Some(!fused.is_grayscale))
        }
        (ChannelOrder::Rgb | ChannelOrder::Bgr, SampleType::U8) => {
            let gray = if let Some(b) = view.as_contiguous_bytes() {
                scan::is_grayscale_rgb8(b)
            } else {
                (0..view.height()).all(|y| scan::is_grayscale_rgb8(view.row(y)))
            };
            (Some(false), Some(!gray))
        }
        (ChannelOrder::GrayAlpha, SampleType::U8) => {
            let op = if !scan_alpha {
                true
            } else if let Some(b) = view.as_contiguous_bytes() {
                scan::is_opaque_ga8(b)
            } else {
                (0..view.height()).all(|y| scan::is_opaque_ga8(view.row(y)))
            };
            (Some(!op), Some(false))
        }
        (ChannelOrder::Gray, _) => (Some(false), Some(false)),
        (
            ChannelOrder::Rgba | ChannelOrder::Bgra | ChannelOrder::Rgbx | ChannelOrder::Bgrx,
            SampleType::F32,
        ) => {
            let op = if !scan_alpha {
                true
            } else {
                rows_all_f32(view, scan::is_opaque_rgba_f32)
            };
            let gr = rows_all_f32(view, scan::is_grayscale_rgba_f32);
            (Some(!op), Some(!gr))
        }
        (ChannelOrder::Rgb | ChannelOrder::Bgr, SampleType::F32) => (
            Some(false),
            Some(!rows_all_f32(view, scan::is_grayscale_rgb_f32)),
        ),
        (ChannelOrder::GrayAlpha, SampleType::F32) => {
            let op = if !scan_alpha {
                true
            } else {
                rows_all_f32(view, scan::is_opaque_ga_f32)
            };
            (Some(!op), Some(false))
        }
        _ => (None, None),
    };

    if let Some(structural) = alpha_structural
        && requires_alpha.is_some()
    {
        requires_alpha = structural;
    }

    let requires_u16_precision = match sample {
        SampleType::U8 => Some(false),
        SampleType::U16 => Some(!rows_all_u16(view, scan::bit_replication_lossless_u16)),
        _ => None,
    };

    CompactionReport {
        requires_alpha,
        requires_chroma,
        requires_u16_precision,
    }
}

/// Analyze `view` and compact into a newly allocated [`PixelBuf`] if a narrower bit-exact
/// representation exists (`Ok(None)` if already minimal), preserving allocation errors.
#[doc(alias = "load_bearing")]
#[doc(alias = "shrink")]
pub fn compact_to_buf(view: &PixelView<'_>) -> Result<Option<PixelBuf>, At<ConvertError>> {
    let src = view.format().clone();
    let mut report = analyze_compaction(view);
    let plan = plan_chroma_collapse(&src);
    if matches!(plan, GraySignalPlan::Suppress) {
        report.requires_chroma = None;
    }
    let target = apply_icc_chroma_collapse(&src, &report, &plan);
    if target == src {
        return Ok(None);
    }
    let mut out = PixelBuf::new(view.width(), view.height(), target.clone())
        .map_err(|e| err_at((*e.error()).into()))?;
    if transform_into(view, &src, &target, &mut out).is_none() {
        return Ok(None);
    }
    let out_origin = view.origin().cloned().unwrap_or_else(ColorOrigin::assumed);
    Ok(Some(
        out.with_origin(out_origin)
            .with_hdr(view.hdr())
            .with_orientation(view.orientation()),
    ))
}

/// Analyze and compact `buf` **in place** without reallocating owned mutable storage
/// (copy-on-write promotes immutable external storage, propagating allocation errors instead
/// of swallowing them).
///
/// Depth (`U16 → U8`) and chroma (`RGB → Gray`) compaction run whenever justified;
/// `alpha_strategy` controls whether redundant alpha/padding lanes are retagged in metadata
/// ([`AlphaCompaction::RetagOpaque`]) or physically dropped in place
/// ([`AlphaCompaction::DropUnusedLanes`]).
///
/// Returns `Ok(true)` if any metadata or layout compaction occurred, `Ok(false)` if `buf`
/// was already minimal, or `Err(ConvertError::Buffer(_))` on CoW allocation failure.
#[doc(alias = "load_bearing")]
#[doc(alias = "shrink")]
pub fn compact_in_place(
    buf: &mut PixelBuf,
    alpha_strategy: AlphaCompaction,
) -> Result<bool, At<ConvertError>> {
    let drop_alpha_lanes = alpha_strategy == AlphaCompaction::DropUnusedLanes;
    let mut compacted = false;
    buf.transform_in_place(|bytes, input| {
        let width = input.width;
        let rows = input.height;
        let in_stride = input.stride_bytes;
        let src = input.format;
        let orig = input.origin;
        if width == 0 || rows == 0 || !src.layout.packing.is_full() {
            return None;
        }
        let mut report = {
            let view = PixelView::new(bytes, width, rows, in_stride, src).ok()?;
            analyze_compaction(&view)
        };
        let plan = plan_chroma_collapse(src);
        if matches!(plan, GraySignalPlan::Suppress) {
            report.requires_chroma = None;
        }

        let alpha_droppable =
            matches!(report.requires_alpha, Some(false)) && src.layout.order.has_alpha_slot();
        if !drop_alpha_lanes {
            report.requires_alpha = None;
        }

        let mut target = apply_icc_chroma_collapse(src, &report, &plan);
        if !drop_alpha_lanes
            && alpha_droppable
            && target.layout.order.has_alpha_slot()
            && !matches!(src.color.alpha(), AlphaMode::Undefined | AlphaMode::Opaque)
        {
            target = target.with_alpha(AlphaMode::Opaque);
        }

        if &target == src {
            return None;
        }

        // Equal-bpp target: metadata-only alpha retag (0 data movement).
        let in_bpp = src.layout.bytes_per_pixel();
        let out_bpp = target.layout.bytes_per_pixel();
        if out_bpp == in_bpp {
            let retagged = src.clone().with_alpha(target.color.alpha());
            let mut facts = SampleFacts::NONE;
            if matches!(retagged.color.alpha(), AlphaMode::Opaque) {
                facts.all_opaque = true;
                facts.binary_alpha = true;
            }
            compacted = true;
            return Some(InPlaceTransformOutput {
                stride_bytes: in_stride,
                format: retagged,
                origin: orig.clone(),
                facts,
            });
        }
        if out_bpp > in_bpp {
            return None;
        }

        let narrow16 =
            src.layout.sample == SampleType::U16 && target.layout.sample == SampleType::U8;
        let map = selection_map(src.layout.order, target.layout.order)?;
        let out_stride = (width as usize) * out_bpp;

        compact_rows_in_place(
            bytes,
            width as usize,
            rows as usize,
            in_stride,
            in_bpp,
            out_bpp,
            src.layout.order,
            target.layout.order,
            map,
            narrow16,
        );

        compacted = true;
        Some(InPlaceTransformOutput {
            stride_bytes: out_stride,
            format: target,
            origin: orig.clone(),
            facts: SampleFacts::NONE,
        })
    })
    .map_err(|e| err_at((*e.error()).into()))?;
    Ok(compacted)
}

// ── Row helpers & byte-exact compaction kernels ────────────────────────

fn fused_rgba8_over_rows(view: &PixelView<'_>, request: FusedRequest) -> scan::FusedResult {
    if let Some(bytes) = view.as_contiguous_bytes() {
        return scan::fused_predicates_rgba8_cg(bytes, request);
    }
    let mut req = request;
    let mut total = scan::FusedResult {
        is_opaque: req.check_opaque,
        is_grayscale: req.check_grayscale,
    };
    for y in 0..view.height() {
        if !req.check_opaque && !req.check_grayscale {
            break;
        }
        let r = scan::fused_predicates_rgba8_cg(view.row(y), req);
        if req.check_opaque && !r.is_opaque {
            total.is_opaque = false;
            req.check_opaque = false;
        }
        if req.check_grayscale && !r.is_grayscale {
            total.is_grayscale = false;
            req.check_grayscale = false;
        }
    }
    total
}

fn rows_all_u16(view: &PixelView<'_>, pred: fn(&[u16]) -> bool) -> bool {
    if let Some(bytes) = view.as_contiguous_bytes()
        && let Ok(s) = bytemuck::try_cast_slice::<u8, u16>(bytes)
    {
        return pred(s);
    }
    for y in 0..view.height() {
        let row = view.row(y);
        if let Ok(s) = bytemuck::try_cast_slice::<u8, u16>(row) {
            if !pred(s) {
                return false;
            }
        } else {
            for chunk in row.as_chunks::<2>().0 {
                let v = u16::from_ne_bytes(*chunk);
                if (v >> 8) != (v & 0xFF) {
                    return false;
                }
            }
        }
    }
    true
}

fn rows_all_f32(view: &PixelView<'_>, pred: fn(&[f32]) -> bool) -> bool {
    if let Some(bytes) = view.as_contiguous_bytes()
        && let Ok(s) = bytemuck::try_cast_slice::<u8, f32>(bytes)
    {
        return pred(s);
    }
    for y in 0..view.height() {
        let row = view.row(y);
        if let Ok(s) = bytemuck::try_cast_slice::<u8, f32>(row) {
            if !pred(s) {
                return false;
            }
        } else {
            return false;
        }
    }
    true
}

fn selection_map(src: ChannelOrder, dst: ChannelOrder) -> Option<&'static [usize]> {
    static IDENTITY: [usize; 4] = [0, 1, 2, 3];
    Some(match (src, dst) {
        _ if src == dst => &IDENTITY[..src.channels()],
        (ChannelOrder::Rgba | ChannelOrder::Rgbx, ChannelOrder::Rgb) => &[0, 1, 2],
        (ChannelOrder::Bgra | ChannelOrder::Bgrx, ChannelOrder::Rgb) => &[2, 1, 0],
        (ChannelOrder::Rgba | ChannelOrder::Bgra, ChannelOrder::GrayAlpha) => &[0, 3],
        (
            ChannelOrder::Rgba
            | ChannelOrder::Bgra
            | ChannelOrder::Rgbx
            | ChannelOrder::Bgrx
            | ChannelOrder::Rgb
            | ChannelOrder::Bgr
            | ChannelOrder::GrayAlpha,
            ChannelOrder::Gray,
        ) => &[0],
        _ => return None,
    })
}

fn transform_into(
    view: &PixelView<'_>,
    src: &PixelFormat,
    dst: &PixelFormat,
    out: &mut PixelBuf,
) -> Option<()> {
    let narrow16 = src.layout.sample == SampleType::U16 && dst.layout.sample == SampleType::U8;
    if !narrow16 && src.layout.sample != dst.layout.sample {
        return None;
    }
    let in_ch = src.layout.channels();
    let map = selection_map(src.layout.order, dst.layout.order)?;
    let mut out_view = out.as_view_mut();
    for y in 0..view.height() {
        let row_in = view.row(y);
        let row_out = out_view.row_mut(y);
        if narrow16 {
            select_row_u16_to_u8(row_in, row_out, in_ch, map);
        } else {
            match (
                dst.layout.sample.byte_size(),
                src.layout.order,
                dst.layout.order,
            ) {
                (1, ChannelOrder::Rgba | ChannelOrder::Rgbx, ChannelOrder::Rgb) => {
                    garb::bytes::rgba_to_rgb(row_in, row_out).ok()?;
                }
                (1, ChannelOrder::Bgra | ChannelOrder::Bgrx, ChannelOrder::Rgb) => {
                    garb::bytes::bgra_to_rgb(row_in, row_out).ok()?;
                }
                (1, ..) => select_row::<1>(row_in, row_out, in_ch, map),
                (2, ..) => select_row::<2>(row_in, row_out, in_ch, map),
                (4, ..) => select_row::<4>(row_in, row_out, in_ch, map),
                _ => return None,
            }
        }
    }
    Some(())
}

#[allow(clippy::too_many_arguments)]
fn compact_rows_in_place(
    data: &mut [u8],
    width: usize,
    rows: usize,
    in_stride: usize,
    in_bpp: usize,
    out_bpp: usize,
    src_order: ChannelOrder,
    dst_order: ChannelOrder,
    map: &[usize],
    narrow16: bool,
) {
    let out_stride = width * out_bpp;
    let in_ch = src_order.channels();
    let elem = if narrow16 { 2 } else { in_bpp / in_ch };
    let row_in_len = width * in_bpp;
    for y in 0..rows {
        let src_start = y * in_stride;
        let dst_start = y * out_stride;
        let dst_end = dst_start + out_stride;
        if dst_end <= src_start {
            let (head, tail) = data.split_at_mut(src_start);
            let row_in = &tail[..row_in_len];
            let row_out = &mut head[dst_start..dst_end];
            if narrow16 {
                select_row_u16_to_u8(row_in, row_out, in_ch, map);
            } else {
                match (elem, src_order, dst_order) {
                    (1, ChannelOrder::Rgba | ChannelOrder::Rgbx, ChannelOrder::Rgb) => {
                        if garb::bytes::rgba_to_rgb(row_in, row_out).is_err() {
                            select_row::<1>(row_in, row_out, in_ch, map);
                        }
                    }
                    (1, ChannelOrder::Bgra | ChannelOrder::Bgrx, ChannelOrder::Rgb) => {
                        if garb::bytes::bgra_to_rgb(row_in, row_out).is_err() {
                            select_row::<1>(row_in, row_out, in_ch, map);
                        }
                    }
                    (1, ..) => select_row::<1>(row_in, row_out, in_ch, map),
                    (2, ..) => select_row::<2>(row_in, row_out, in_ch, map),
                    _ => select_row::<4>(row_in, row_out, in_ch, map),
                }
            }
        } else {
            for x in 0..width {
                let s = src_start + x * in_bpp;
                let mut tmp = [0u8; 16];
                tmp[..in_bpp].copy_from_slice(&data[s..s + in_bpp]);
                let d = dst_start + x * out_bpp;
                if narrow16 {
                    for (k, &c) in map.iter().enumerate() {
                        data[d + k] = tmp[c * 2];
                    }
                } else {
                    for (k, &c) in map.iter().enumerate() {
                        data[d + k * elem..d + (k + 1) * elem]
                            .copy_from_slice(&tmp[c * elem..(c + 1) * elem]);
                    }
                }
            }
        }
    }
}

#[inline]
fn select_row<const E: usize>(row_in: &[u8], row_out: &mut [u8], in_ch: usize, map: &[usize]) {
    let out_px = map.len() * E;
    let in_px = in_ch * E;
    for (dst, src) in row_out
        .chunks_exact_mut(out_px)
        .zip(row_in.chunks_exact(in_px))
    {
        for (k, &c) in map.iter().enumerate() {
            dst[k * E..(k + 1) * E].copy_from_slice(&src[c * E..c * E + E]);
        }
    }
}

#[inline]
fn select_row_u16_to_u8(row_in: &[u8], row_out: &mut [u8], in_ch: usize, map: &[usize]) {
    let in_px = in_ch * 2;
    for (dst, src) in row_out
        .chunks_exact_mut(map.len())
        .zip(row_in.chunks_exact(in_px))
    {
        for (k, &c) in map.iter().enumerate() {
            dst[k] = src[c * 2];
        }
    }
}
