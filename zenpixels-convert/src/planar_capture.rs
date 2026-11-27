//! Forward packed-to-planar capture ([`PlanarCaptureSpec`], [`PlanarCapturePlan`],
//! [`CaptureScratch`]): strided interleaved pixels → multi-plane YCbCr / YCgCo /
//! Identity GBR / planar RGB / monochrome [`PlanarViewMut`] (`U8`, `U16` 8/10/12/16-bit
//! including `P010` / `U10_LSB`, `U32`, and `F32` planar RGB).
//!
//! Two-stage execution guarantees identical color, transfer, HDR white/peak, HLG OOTF,
//! alpha matte, gamut, and CMS behavior with [`RowPlan`](crate::RowPlan):
//! 1. Stage 1 converts source rows into full-range `f32` in the destination's primaries,
//!    transfer, diffuse white, and alpha mode using a compiled [`RowPlan`](crate::RowPlan)
//!    over an O(width) ring buffer of at most `2 * max_vertical_subsampling + 1` rows
//!    (5 rows for 4:2:0 — ~230 KiB at 4K instead of a full frame buffer).
//! 2. Stage 2 evaluates exact `f64` ITU-T H.273 matrix encoding, phase-exact chroma
//!    downsampling honoring [`PlanarCaptureSpec::chroma_siting`] and
//!    [`PlanarCaptureSpec::visible_rect`] crop origin (using local integer coordinates
//!    immune to large-offset float rounding), final `f64` quantization (`Round` /
//!    `Truncate` / `OrderedDither`), and [`BitPacking::pack_code`](zenpixels::BitPacking::pack_code).

use alloc::boxed::Box;
use alloc::vec::Vec;
use enough::{Stop, Unstoppable};
use whereat::At;
use zenpixels::{
    AlphaMode, ChromaSiting, Cicp, ColorSpace, DiffuseWhite, HdrMetadata, Nits, PixelFormat,
    PixelRect, PixelView, PlanarBuf, PlanarViewMut, PlaneFormat, PlaneRole, Primaries,
    SampleLayout, SampleType, SignalRange, Transfer,
};

use crate::error::{ConvertError, err_at};
use crate::plan::{
    RowPlan, color_primaries, color_transfer, is_gamut_narrowing, is_scene_linear_hdr_candidate,
    try_substitute_icc,
};
use crate::policy::{ConvertPolicy, DepthPolicy, GamutPolicy, RangePolicy};
use crate::ycbcr::YcbcrMatrix;

/// 4×4 Bayer ordered-dither integer thresholds (`0..=15`, normalized as `(b + 0.5) / 16`).
const BAYER_4X4_INT: [i128; 16] = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];

/// 4×4 Bayer ordered-dither threshold matrix normalized to `[0.5/16, 15.5/16]`.
const BAYER_4X4: [f64; 16] = [
    0.5 / 16.0,
    8.5 / 16.0,
    2.5 / 16.0,
    10.5 / 16.0,
    12.5 / 16.0,
    4.5 / 16.0,
    14.5 / 16.0,
    6.5 / 16.0,
    3.5 / 16.0,
    11.5 / 16.0,
    1.5 / 16.0,
    9.5 / 16.0,
    15.5 / 16.0,
    7.5 / 16.0,
    13.5 / 16.0,
    5.5 / 16.0,
];

/// Chroma downsampling filter for subsampled YCbCr capture (`4:2:0`, `4:2:2`, `4:4:0`, `4:1:1`).
///
/// Ignored for full-resolution (`1×1`) planar RGB, 4:4:4 YCbCr/GBR, and monochrome targets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ChromaDownsample {
    /// Phase-exact separable triangle downsampler honoring [`PlanarCaptureSpec::chroma_siting`]
    /// and [`PlanarCaptureSpec::visible_rect`] crop origin.
    ///
    /// Produces exact `[1, 2, 1]/4` co-sited filtering for `Left`/`TopLeft` 2× axes,
    /// `[1, 3, 3, 1]/8` dual-bilinear filtering for `Center` 2× axes, and exact
    /// dyadic triangle taps for 4× axes, with zero phase shift and edge clamping.
    ///
    /// Refuses [`ChromaSiting::Unknown`] on any subsampled axis unless
    /// [`PlanarCaptureOptions::unknown_siting_fallback`] is set.
    #[default]
    SitingAware,
    /// Unweighted box average over each chroma cell's luma pixels, honoring crop origin
    /// and clamping at visible edges. Works even when `chroma_siting == ChromaSiting::Unknown`.
    BoxAverage,
    /// Nearest luma sample at each chroma sample center, honoring [`PlanarCaptureSpec::chroma_siting`]
    /// and [`PlanarCaptureSpec::visible_rect`] crop origin.
    ///
    /// Refuses [`ChromaSiting::Unknown`] on any subsampled axis unless
    /// [`PlanarCaptureOptions::unknown_siting_fallback`] is set.
    Nearest,
}

/// Configuration for forward planar capture ([`PlanarCapturePlan`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct PlanarCaptureOptions {
    /// Chroma downsampling kernel (`SitingAware` default).
    pub downsample: ChromaDownsample,
    /// Optional fallback [`ChromaSiting`] used when a siting-dependent downsample
    /// (`SitingAware` or `Nearest`) runs on a subsampled target whose
    /// `spec.chroma_siting == ChromaSiting::Unknown`.
    pub unknown_siting_fallback: Option<ChromaSiting>,
}

impl PlanarCaptureOptions {
    /// Phase-exact siting-aware chroma downsampling (`ChromaDownsample::SitingAware`).
    #[inline]
    #[must_use]
    pub const fn siting_aware() -> Self {
        Self {
            downsample: ChromaDownsample::SitingAware,
            unknown_siting_fallback: None,
        }
    }

    /// Cell box-average chroma downsampling (`ChromaDownsample::BoxAverage`).
    #[inline]
    #[must_use]
    pub const fn box_average() -> Self {
        Self {
            downsample: ChromaDownsample::BoxAverage,
            unknown_siting_fallback: None,
        }
    }

    /// Nearest-sample chroma downsampling (`ChromaDownsample::Nearest`).
    #[inline]
    #[must_use]
    pub const fn nearest() -> Self {
        Self {
            downsample: ChromaDownsample::Nearest,
            unknown_siting_fallback: None,
        }
    }

    /// Select the chroma downsampling kernel.
    #[inline]
    #[must_use]
    pub const fn with_downsample(mut self, downsample: ChromaDownsample) -> Self {
        self.downsample = downsample;
        self
    }

    /// Provide an explicit fallback [`ChromaSiting`] when the target declares `ChromaSiting::Unknown`.
    #[inline]
    #[must_use]
    pub const fn with_unknown_siting_fallback(mut self, siting: ChromaSiting) -> Self {
        self.unknown_siting_fallback = Some(siting);
        self
    }
}

/// Target planar geometry, plane layout, chroma siting, and true coded CICP signal
/// for [`PlanarCapturePlan`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct PlanarCaptureSpec {
    /// Full coded frame dimensions `(width, height)` in luma pixels.
    pub coded_extent: (u32, u32),
    /// Visible active rectangle within `coded_extent`; its `(x, y)` origin establishes
    /// the subsampling phase against the coded grid, and `(width, height)` must match
    /// the source [`PixelView`] and destination [`PlanarViewMut`] dimensions.
    pub visible_rect: PixelRect,
    /// Target plane storage specifications (roles, sample types, bit packing, subsampling).
    pub planes: Box<[PlaneFormat]>,
    /// Target chroma sample siting.
    pub chroma_siting: ChromaSiting,
    /// True coded destination CICP signal (primaries, transfer, matrix coefficients, full/narrow range).
    pub signal: Cicp,
    /// Target alpha association when `planes` includes [`PlaneRole::Alpha`] (default `Straight`).
    pub alpha: AlphaMode,
    /// Optional relative-linear `1.0` diffuse-white anchor for the destination color space.
    pub diffuse_white: Option<DiffuseWhite>,
}

impl PlanarCaptureSpec {
    /// Construct a planar capture specification with explicit `coded_extent` and `visible_rect`.
    #[must_use]
    pub fn new(
        coded_extent: (u32, u32),
        visible_rect: PixelRect,
        planes: impl Into<Box<[PlaneFormat]>>,
        chroma_siting: ChromaSiting,
        signal: Cicp,
    ) -> Self {
        Self {
            coded_extent,
            visible_rect,
            planes: planes.into(),
            chroma_siting,
            signal,
            alpha: AlphaMode::Straight,
            diffuse_white: None,
        }
    }

    /// Construct a full-frame planar capture specification (`visible_rect` covers `(0, 0, width, height)`).
    #[must_use]
    pub fn full_frame(
        width: u32,
        height: u32,
        planes: impl Into<Box<[PlaneFormat]>>,
        chroma_siting: ChromaSiting,
        signal: Cicp,
    ) -> Self {
        Self::new(
            (width, height),
            PixelRect::new(0, 0, width, height),
            planes,
            chroma_siting,
            signal,
        )
    }

    /// Set alpha association mode for an optional [`PlaneRole::Alpha`] plane.
    #[inline]
    #[must_use]
    pub const fn with_alpha(mut self, alpha: AlphaMode) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set explicit diffuse-white anchor for the destination color space.
    #[inline]
    #[must_use]
    pub const fn with_diffuse_white(mut self, white: DiffuseWhite) -> Self {
        self.diffuse_white = Some(white);
        self
    }
}

#[derive(Debug)]
enum ScratchBacking<'a> {
    Borrowed(&'a mut [f64]),
    Owned(Vec<f64>),
}

/// Reusable caller-owned or budget-borrowed `f64` scratch workspace for [`PlanarCapturePlan::execute`].
///
/// Use [`CaptureScratch::from_f64_slice`] with a caller-allocated or `StorageBudget`-backed
/// slice for zero-heap capture on native conversion routes (a custom external `CmsTransform`
/// invoked inside Stage 1 may still allocate unless its [`CmsQuote::apply_scratch_bytes`](crate::CmsQuote::apply_scratch_bytes)
/// is `Some(0)`), or [`CaptureScratch::owned`] to reuse one fallibly allocated buffer across frames.
#[derive(Debug)]
pub struct CaptureScratch<'a> {
    backing: ScratchBacking<'a>,
}

impl Default for CaptureScratch<'static> {
    fn default() -> Self {
        Self::owned()
    }
}

impl<'a> CaptureScratch<'a> {
    /// Borrow a caller-provided `&mut [f64]` workspace (never heap-allocates; fails with
    /// [`ConvertError::DimensionMismatch`] if shorter than [`PlanarCapturePlan::required_scratch_f64s`]).
    #[inline]
    #[must_use]
    pub fn from_f64_slice(slice: &'a mut [f64]) -> Self {
        Self {
            backing: ScratchBacking::Borrowed(slice),
        }
    }

    /// Number of `f64` elements currently available without reallocating.
    #[inline]
    #[must_use]
    pub fn len_f64s(&self) -> usize {
        match &self.backing {
            ScratchBacking::Borrowed(s) => s.len(),
            ScratchBacking::Owned(v) => v.len(),
        }
    }

    fn acquire(&mut self, needed: usize) -> Result<&mut [f64], At<ConvertError>> {
        match &mut self.backing {
            ScratchBacking::Borrowed(s) => {
                if s.len() < needed {
                    return Err(err_at(ConvertError::DimensionMismatch));
                }
                Ok(&mut s[..needed])
            }
            ScratchBacking::Owned(v) => {
                if v.len() < needed {
                    let extra = needed - v.len();
                    v.try_reserve_exact(extra)
                        .map_err(|_| err_at(ConvertError::AllocationFailed))?;
                    v.resize(needed, 0.0);
                }
                Ok(&mut v[..needed])
            }
        }
    }
}

impl CaptureScratch<'static> {
    /// Create an empty owned scratch buffer that grows fallibly on first use and reuses capacity.
    #[inline]
    #[must_use]
    pub const fn owned() -> Self {
        Self {
            backing: ScratchBacking::Owned(Vec::new()),
        }
    }

    /// Pre-allocate an owned scratch buffer for at least `len_f64s` `f64` elements.
    pub fn with_capacity_f64s(len_f64s: usize) -> Result<Self, At<ConvertError>> {
        let mut v = Vec::new();
        v.try_reserve_exact(len_f64s)
            .map_err(|_| err_at(ConvertError::AllocationFailed))?;
        v.resize(len_f64s, 0.0);
        Ok(Self {
            backing: ScratchBacking::Owned(v),
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum TargetLayout {
    Monochrome {
        y_idx: usize,
        alpha_idx: Option<usize>,
    },
    PlanarRgb {
        r_idx: usize,
        g_idx: usize,
        b_idx: usize,
        alpha_idx: Option<usize>,
    },
    YcbcrPlanar {
        y_idx: usize,
        cb_idx: usize,
        cr_idx: usize,
        alpha_idx: Option<usize>,
    },
    YcbcrSemi {
        y_idx: usize,
        uv_idx: usize,
        swapped: bool,
        alpha_idx: Option<usize>,
    },
}

#[derive(Clone, Copy, Debug)]
enum ForwardMatrix {
    Identity,
    YCgCo,
    Ncl {
        kr: f64,
        kg: f64,
        kb: f64,
        inv_2_1_minus_kb: f64,
        inv_2_1_minus_kr: f64,
        /// Exact `(kr_num, kg_num, kb_num, den)` integer rational coefficients for known
        /// ITU-T H.273 NCL matrices so integer-RGB inputs round exact half-ties without
        /// IEEE-754 ULP drift.
        exact_rational: Option<(i128, i128, i128, i128)>,
    },
}

#[derive(Clone, Copy, Debug)]
struct DirectIntRgb {
    is_u16: bool,
    channels: usize,
    r_idx: usize,
    g_idx: usize,
    b_idx: usize,
    a_idx: Option<usize>,
}

/// Prepared, reusable, thread-safe forward planar capture plan.
///
/// Built via [`PlanarCapturePlan::build`] or [`PlanarCapturePlan::build_with_options`],
/// capturing the validated target [`PlanarCaptureSpec`], compiled color/transfer [`RowPlan`],
/// exact `f64` / rational `i128` YCbCr/GBR forward matrix, and chroma downsampling geometry.
#[derive(Clone, Debug)]
pub struct PlanarCapturePlan {
    spec: PlanarCaptureSpec,
    options: PlanarCaptureOptions,
    row_plan: RowPlan,
    direct_int_rgb: Option<DirectIntRgb>,
    inter_channels: usize,
    layout: TargetLayout,
    matrix: ForwardMatrix,
    range: SignalRange,
    depth_policy: DepthPolicy,
    resolved_siting: ChromaSiting,
    any_subsampled: bool,
    ring_rows: usize,
    dst_color: ColorSpace,
    dst_can_represent_hdr: bool,
}

fn exact_ncl_rational(ym: YcbcrMatrix) -> Option<(i128, i128, i128, i128)> {
    match ym {
        YcbcrMatrix::Bt709 => Some((2126, 7152, 722, 10_000)),
        YcbcrMatrix::Fcc => Some((3000, 5900, 1100, 10_000)),
        YcbcrMatrix::Bt470Bg | YcbcrMatrix::Bt601 => Some((2990, 5870, 1140, 10_000)),
        YcbcrMatrix::Smpte240M => Some((2120, 7010, 870, 10_000)),
        YcbcrMatrix::Bt2020 => Some((2627, 6780, 593, 10_000)),
        YcbcrMatrix::ChromaDerivedNcl | YcbcrMatrix::Identity | YcbcrMatrix::YCgCo => None,
    }
}

fn exact_ncl_kr_kb_f64(ym: YcbcrMatrix, primaries: Option<Primaries>) -> Option<(f64, f64)> {
    match ym {
        YcbcrMatrix::Bt709 => Some((0.2126, 0.0722)),
        YcbcrMatrix::Fcc => Some((0.30, 0.11)),
        YcbcrMatrix::Bt470Bg | YcbcrMatrix::Bt601 => Some((0.299, 0.114)),
        YcbcrMatrix::Smpte240M => Some((0.212, 0.087)),
        YcbcrMatrix::Bt2020 => Some((0.2627, 0.0593)),
        YcbcrMatrix::ChromaDerivedNcl => {
            let (kr, kb) = ym.kr_kb_for(primaries)?;
            Some((f64::from(kr), f64::from(kb)))
        }
        YcbcrMatrix::Identity | YcbcrMatrix::YCgCo => None,
    }
}

impl PlanarCapturePlan {
    /// Compile a forward planar capture plan with default [`PlanarCaptureOptions::siting_aware()`]
    /// and no pre-resolved HDR peak.
    pub fn build(
        src: &PixelFormat,
        dst: PlanarCaptureSpec,
        policy: &ConvertPolicy,
    ) -> Result<Self, At<ConvertError>> {
        Self::build_with_options(src, dst, policy, PlanarCaptureOptions::default(), None)
    }

    /// Compile a forward planar capture plan with explicit [`PlanarCaptureOptions`] and
    /// optional resolved HDR source peak luminance (for HDR → SDR tone mapping).
    pub fn build_with_options(
        src: &PixelFormat,
        dst: PlanarCaptureSpec,
        policy: &ConvertPolicy,
        options: PlanarCaptureOptions,
        resolved_source_peak: Option<Nits>,
    ) -> Result<Self, At<ConvertError>> {
        src.validate().map_err(|e| err_at((*e.error()).into()))?;

        // 1. Validate coded_extent & visible_rect containment.
        let (cw, ch) = dst.coded_extent;
        let vis = dst.visible_rect;
        if cw == 0 || ch == 0 || vis.width == 0 || vis.height == 0 {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        let end_x = vis
            .x
            .checked_add(vis.width)
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        let end_y = vis
            .y
            .checked_add(vis.height)
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        if end_x > cw || end_y > ch {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        if dst.planes.is_empty() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }

        // 2. Parse & validate destination plane formats.
        let (layout, has_alpha, is_mono, allow_f32_planes, any_subsampled, max_vy, min_int_bits) =
            parse_target_planes(&dst.planes, vis)?;

        // 3. Resolve destination range, transfer, primaries, matrix, and chroma siting.
        let dst_range = if dst.signal.full_range {
            SignalRange::Full
        } else {
            SignalRange::Narrow
        };
        // Floating-point planar RGB carries normalized full-range values; refuse narrow F32.
        if dst_range == SignalRange::Narrow
            && dst.planes.iter().any(|p| p.sample == SampleType::F32)
        {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        let src_range = src.color.range().unwrap_or(SignalRange::Full);
        if src_range != dst_range && policy.range == RangePolicy::Forbid {
            return Err(err_at(ConvertError::RangeConversionForbidden));
        }

        let transfer = Transfer::from_cicp(dst.signal.transfer_characteristics)
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        let primaries = if is_mono {
            Primaries::Bt709
        } else {
            Primaries::from_cicp(dst.signal.color_primaries)
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?
        };

        let dst_alpha = if has_alpha {
            match dst.alpha {
                AlphaMode::Straight | AlphaMode::Premultiplied | AlphaMode::Opaque => dst.alpha,
                _ => return Err(err_at(ConvertError::UnsupportedRoute)),
            }
        } else {
            AlphaMode::None
        };

        let matrix = match layout {
            TargetLayout::Monochrome { .. } | TargetLayout::PlanarRgb { .. } => {
                if matches!(layout, TargetLayout::PlanarRgb { .. })
                    && dst.signal.matrix_coefficients != 0
                {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
                ForwardMatrix::Identity
            }
            TargetLayout::YcbcrPlanar { .. } | TargetLayout::YcbcrSemi { .. } => {
                let ym = YcbcrMatrix::from_cicp_code(dst.signal.matrix_coefficients)
                    .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
                match ym {
                    YcbcrMatrix::Identity => {
                        if matches!(layout, TargetLayout::YcbcrSemi { .. }) || any_subsampled {
                            return Err(err_at(ConvertError::UnsupportedRoute));
                        }
                        ForwardMatrix::Identity
                    }
                    YcbcrMatrix::YCgCo => ForwardMatrix::YCgCo,
                    _ => {
                        let p_opt = if primaries == Primaries::Unknown {
                            None
                        } else {
                            Some(primaries)
                        };
                        let (kr, kb) = exact_ncl_kr_kb_f64(ym, p_opt)
                            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
                        let kg = 1.0 - kr - kb;
                        if kg <= 1e-9 || (1.0 - kr).abs() <= 1e-9 || (1.0 - kb).abs() <= 1e-9 {
                            return Err(err_at(ConvertError::UnsupportedRoute));
                        }
                        ForwardMatrix::Ncl {
                            kr,
                            kg,
                            kb,
                            inv_2_1_minus_kb: 1.0 / (2.0 * (1.0 - kb)),
                            inv_2_1_minus_kr: 1.0 / (2.0 * (1.0 - kr)),
                            exact_rational: exact_ncl_rational(ym),
                        }
                    }
                }
            }
        };

        let needs_siting = any_subsampled
            && matches!(
                options.downsample,
                ChromaDownsample::SitingAware | ChromaDownsample::Nearest
            );
        let resolved_siting = if needs_siting {
            if dst.chroma_siting != ChromaSiting::Unknown {
                dst.chroma_siting
            } else if let Some(fb) = options.unknown_siting_fallback
                && fb != ChromaSiting::Unknown
            {
                fb
            } else {
                return Err(err_at(ConvertError::UnsupportedRoute));
            }
        } else if dst.chroma_siting != ChromaSiting::Unknown {
            dst.chroma_siting
        } else {
            options
                .unknown_siting_fallback
                .unwrap_or(ChromaSiting::Unknown)
        };

        // 4. Enforce final-plane depth & gamut policies that Stage 1 (targeting F32) wouldn't see.
        let targets_bounded_int =
            !allow_f32_planes || dst.planes.iter().any(|p| p.sample != SampleType::F32);
        if targets_bounded_int {
            let src_bits = src_effective_bits(src);
            let depth_reduces = src.layout.sample.is_float() || min_int_bits < src_bits;
            if depth_reduces && policy.depth == DepthPolicy::Forbid {
                return Err(err_at(ConvertError::DepthReductionForbidden));
            }
        }

        // 5. Build full-range F32 intermediate PixelFormat and compile Stage 1 RowPlan.
        let (inter_layout, inter_color, dst_color) = if is_mono {
            let layout_inter = if has_alpha {
                SampleLayout::GRAYAF32
            } else {
                SampleLayout::GRAYF32
            };
            let inter_cs = ColorSpace::Gray {
                transfer,
                range: SignalRange::Full,
                alpha: dst_alpha,
                white: dst.diffuse_white,
            };
            let tagged_cs = ColorSpace::Gray {
                transfer,
                range: dst_range,
                alpha: dst_alpha,
                white: dst.diffuse_white,
            };
            (layout_inter, inter_cs, tagged_cs)
        } else {
            let layout_inter = if has_alpha {
                SampleLayout::RGBAF32
            } else {
                SampleLayout::RGBF32
            };
            let inter_cs = ColorSpace::Rgb {
                primaries,
                transfer,
                range: SignalRange::Full,
                alpha: dst_alpha,
                white: dst.diffuse_white,
            };
            let tagged_cs = ColorSpace::Rgb {
                primaries,
                transfer,
                range: dst_range,
                alpha: dst_alpha,
                white: dst.diffuse_white,
            };
            (layout_inter, inter_cs, tagged_cs)
        };
        let inter_fmt = PixelFormat::new(inter_layout, inter_color);

        if targets_bounded_int {
            let src_p = color_primaries(&src.color);
            let dst_p = color_primaries(&inter_fmt.color);
            let cross_profile_cms =
                matches!(src.color, ColorSpace::Icc { .. } | ColorSpace::Cmyk { .. })
                    && try_substitute_icc(src, policy).is_none();
            let gamut_narrows = is_gamut_narrowing(src_p, dst_p) || cross_profile_cms;
            let src_tf = color_transfer(&src.color);
            let dst_tf = color_transfer(&inter_fmt.color);
            let is_hdr_to_sdr = (src_tf.is_hdr() && !dst_tf.is_hdr())
                || (is_scene_linear_hdr_candidate(src, &inter_fmt)
                    && !policy.source_facts.float.is_unit_interval());
            let unproven_float_to_int =
                src.layout.sample.is_float() && !policy.source_facts.float.is_unit_interval();
            if (policy.gamut == GamutPolicy::Forbid
                && (gamut_narrows || is_hdr_to_sdr || unproven_float_to_int))
                || (policy.gamut == GamutPolicy::PreserveExtended
                    && policy.depth == DepthPolicy::Forbid
                    && (gamut_narrows || is_hdr_to_sdr || unproven_float_to_int))
            {
                return Err(err_at(ConvertError::GamutLossForbidden));
            }
        }

        // Stage 1 converts to full-range F32; we already checked `policy.range` against `dst_range`.
        let mut stage1_policy = policy.clone();
        stage1_policy.range = RangePolicy::Convert;
        // Ordered dither applies at final integer plane quantization in Stage 2, not into F32.
        if stage1_policy.depth == DepthPolicy::OrderedDither {
            stage1_policy.depth = DepthPolicy::Round;
        }
        let row_plan = RowPlan::build(src, &inter_fmt, &stage1_policy, resolved_source_peak)?;
        let direct_int_rgb = if is_mono {
            None
        } else {
            detect_direct_int_rgb(src, &inter_fmt, has_alpha)
        };

        let ring_rows = if any_subsampled && max_vy > 1 {
            match options.downsample {
                ChromaDownsample::SitingAware => usize::from(max_vy) * 2 + 1,
                ChromaDownsample::BoxAverage => usize::from(max_vy) + 1,
                ChromaDownsample::Nearest => 2,
            }
        } else {
            1
        };

        let dst_can_represent_hdr =
            transfer.is_hdr() || (transfer.is_linear() && !targets_bounded_int);

        Ok(Self {
            spec: dst,
            options,
            row_plan,
            direct_int_rgb,
            inter_channels: inter_layout.channels(),
            layout,
            matrix,
            range: dst_range,
            depth_policy: policy.depth,
            resolved_siting,
            any_subsampled,
            ring_rows,
            dst_color,
            dst_can_represent_hdr,
        })
    }

    /// Validated destination [`PlanarCaptureSpec`].
    #[inline]
    #[must_use]
    pub fn spec(&self) -> &PlanarCaptureSpec {
        &self.spec
    }

    /// Expected source [`PixelFormat`].
    #[inline]
    #[must_use]
    pub fn source_format(&self) -> &PixelFormat {
        self.row_plan.source_format()
    }

    /// Active [`PlanarCaptureOptions`].
    #[inline]
    #[must_use]
    pub const fn options(&self) -> PlanarCaptureOptions {
        self.options
    }

    /// Effective [`ChromaSiting`] applied during capture (reflecting `unknown_siting_fallback`
    /// when `spec.chroma_siting` was [`ChromaSiting::Unknown`]).
    #[inline]
    #[must_use]
    pub const fn resolved_siting(&self) -> ChromaSiting {
        self.resolved_siting
    }

    /// Number of `f64` elements required in [`CaptureScratch`] for [`execute`](Self::execute).
    #[inline]
    #[must_use]
    pub fn required_scratch_f64s(&self) -> usize {
        (self.spec.visible_rect.width as usize)
            .saturating_mul(self.inter_channels)
            .saturating_mul(self.ring_rows)
    }

    /// Bytes required in [`CaptureScratch`] for [`execute`](Self::execute).
    #[inline]
    #[must_use]
    pub fn required_scratch_bytes(&self) -> usize {
        self.required_scratch_f64s()
            .saturating_mul(core::mem::size_of::<f64>())
    }

    /// Execute the capture plan into a caller-provided strided [`PlanarViewMut`] using
    /// caller-owned [`CaptureScratch`] (zero heap allocation on native routes when `scratch`
    /// is pre-sized or borrowed via [`CaptureScratch::from_f64_slice`]).
    ///
    /// For subsampled targets, `dst.chroma_siting()` must match [`resolved_siting`](Self::resolved_siting)
    /// so borrowed destinations never report an inconsistent or missing chroma siting.
    pub fn execute(
        &self,
        src: PixelView<'_>,
        dst: &mut PlanarViewMut<'_>,
        scratch: &mut CaptureScratch<'_>,
    ) -> Result<(), At<ConvertError>> {
        self.execute_with_stop(src, dst, scratch, &Unstoppable)
    }

    /// Execute the capture plan with cooperative [`Stop`] cancellation checked per row.
    pub fn execute_with_stop(
        &self,
        src: PixelView<'_>,
        dst: &mut PlanarViewMut<'_>,
        scratch: &mut CaptureScratch<'_>,
        stop: &dyn Stop,
    ) -> Result<(), At<ConvertError>> {
        let vis = self.spec.visible_rect;
        if src.width() != vis.width || src.height() != vis.height {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        if src.format() != self.row_plan.source_format() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        if dst.width() != vis.width
            || dst.height() != vis.height
            || dst.crop_origin() != (vis.x, vis.y)
        {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        if self.any_subsampled {
            if dst.chroma_siting() != self.resolved_siting {
                return Err(err_at(ConvertError::UnsupportedRoute));
            }
        } else if dst.chroma_siting() != ChromaSiting::Unknown
            && self.resolved_siting != ChromaSiting::Unknown
            && dst.chroma_siting() != self.resolved_siting
        {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        if dst.planes().len() != self.spec.planes.len() {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        for (actual, expected) in dst.planes().iter().zip(self.spec.planes.iter()) {
            if actual.format() != *expected {
                return Err(err_at(ConvertError::UnsupportedRoute));
            }
            let (ew, eh) = expected
                .cropped_plane_dimensions(vis)
                .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
            if actual.width() != ew || actual.height() != eh {
                return Err(err_at(ConvertError::DimensionMismatch));
            }
        }

        let w = vis.width as usize;
        let row_floats = w
            .checked_mul(self.inter_channels)
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        let needed = row_floats
            .checked_mul(self.ring_rows)
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        let ring = scratch.acquire(needed)?;

        match self.layout {
            TargetLayout::Monochrome { y_idx, alpha_idx } => {
                self.exec_monochrome(&src, dst, ring, row_floats, y_idx, alpha_idx, stop)?;
            }
            TargetLayout::PlanarRgb {
                r_idx,
                g_idx,
                b_idx,
                alpha_idx,
            } => {
                self.exec_planar_rgb(
                    &src, dst, ring, row_floats, r_idx, g_idx, b_idx, alpha_idx, stop,
                )?;
            }
            TargetLayout::YcbcrPlanar {
                y_idx,
                cb_idx,
                cr_idx,
                alpha_idx,
            } => {
                self.exec_ycbcr(
                    &src,
                    dst,
                    ring,
                    row_floats,
                    y_idx,
                    YcbcrDstIndices::Planar { cb_idx, cr_idx },
                    alpha_idx,
                    stop,
                )?;
            }
            TargetLayout::YcbcrSemi {
                y_idx,
                uv_idx,
                swapped,
                alpha_idx,
            } => {
                self.exec_ycbcr(
                    &src,
                    dst,
                    ring,
                    row_floats,
                    y_idx,
                    YcbcrDstIndices::Semi { uv_idx, swapped },
                    alpha_idx,
                    stop,
                )?;
            }
        }

        dst.set_orientation(src.orientation());
        Ok(())
    }

    /// Allocate a tagged [`PlanarBuf`] covering the zero-origin [`visible_rect`](PlanarCaptureSpec::visible_rect)
    /// (does not allocate extra `coded_extent` right/bottom padding; requires `visible_rect.(x, y) == (0, 0)`).
    ///
    /// For non-zero crop origins or padded coded frames, construct a [`PlanarViewMut`] over your
    /// own backing planes and call [`execute`](Self::execute).
    pub fn capture_to_buf(
        &self,
        src: PixelView<'_>,
        scratch: &mut CaptureScratch<'_>,
    ) -> Result<PlanarBuf, At<ConvertError>> {
        self.capture_to_buf_with_stop(src, scratch, &Unstoppable)
    }

    /// Allocate a tagged [`PlanarBuf`] covering the zero-origin [`visible_rect`](PlanarCaptureSpec::visible_rect)
    /// with cooperative cancellation.
    pub fn capture_to_buf_with_stop(
        &self,
        src: PixelView<'_>,
        scratch: &mut CaptureScratch<'_>,
        stop: &dyn Stop,
    ) -> Result<PlanarBuf, At<ConvertError>> {
        let vis = self.spec.visible_rect;
        if vis.x != 0 || vis.y != 0 {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        let hdr = if self.dst_can_represent_hdr {
            src.hdr()
        } else {
            HdrMetadata::default()
        };
        let mut out = PlanarBuf::new(vis.width, vis.height, &self.spec.planes)
            .map_err(|e| err_at((*e.error()).into()))?
            .with_chroma_siting(self.resolved_siting)
            .with_color(self.dst_color.clone())
            .with_matrix_coefficients(self.spec.signal.matrix())
            .with_range_override(self.range)
            .with_hdr(hdr)
            .with_orientation(src.orientation());
        if self.dst_color.alpha() != AlphaMode::None {
            out = out.with_alpha_override(self.dst_color.alpha());
        }
        if let Some(origin) = src.origin() {
            out = out.with_origin(origin.clone());
        }
        {
            let mut view_mut = out
                .try_cow_view_mut()
                .map_err(|e| err_at((*e.error()).into()))?;
            self.execute_with_stop(src, &mut view_mut, scratch, stop)?;
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn exec_monochrome(
        &self,
        src: &PixelView<'_>,
        dst: &mut PlanarViewMut<'_>,
        ring: &mut [f64],
        row_floats: usize,
        y_idx: usize,
        alpha_idx: Option<usize>,
        stop: &dyn Stop,
    ) -> Result<(), At<ConvertError>> {
        let vis = self.spec.visible_rect;
        let y_spec = self.spec.planes[y_idx];
        let a_spec = alpha_idx.map(|i| self.spec.planes[i]);
        let (y_off, y_span, y_lo, y_hi) = luma_encode_params(y_spec, self.range)?;
        let a_params = a_spec.map(alpha_encode_params);
        let inter_ch = self.inter_channels;
        let f32_bytes = row_floats * core::mem::size_of::<f32>();

        let row_buf = &mut ring[..row_floats];
        for y in 0..vis.height {
            stop.check().map_err(|r| err_at(r.into()))?;
            self.row_plan.convert_row_at(
                y,
                src.row(y),
                &mut bytemuck::cast_slice_mut::<f64, u8>(row_buf)[..f32_bytes],
                vis.width,
            )?;
            let row_f32: &[f32] = &bytemuck::cast_slice::<f64, f32>(row_buf)[..row_floats];
            let planes = dst.planes_mut();
            let y_row = planes[y_idx].row_mut(y);
            for x in 0..vis.width as usize {
                let v = f64::from(row_f32[x * inter_ch]).clamp(0.0, 1.0);
                let code = quantize_code(
                    y_off + v * y_span,
                    y_lo,
                    y_hi,
                    self.depth_policy,
                    vis.x as usize + x,
                    vis.y as usize + y as usize,
                );
                write_plane_code(y_spec, y_row, x, 0, code);
            }
            if let (Some(ai), Some((a_s, a_max))) = (alpha_idx, a_params) {
                let a_row = planes[ai].row_mut(y);
                for x in 0..vis.width as usize {
                    let a = row_f32[x * inter_ch + 1];
                    write_alpha_sample(
                        a_s,
                        a_max,
                        a_row,
                        x,
                        a,
                        self.depth_policy,
                        vis.x as usize + x,
                        vis.y as usize + y as usize,
                    );
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn exec_planar_rgb(
        &self,
        src: &PixelView<'_>,
        dst: &mut PlanarViewMut<'_>,
        ring: &mut [f64],
        row_floats: usize,
        r_idx: usize,
        g_idx: usize,
        b_idx: usize,
        alpha_idx: Option<usize>,
        stop: &dyn Stop,
    ) -> Result<(), At<ConvertError>> {
        let vis = self.spec.visible_rect;
        let r_spec = self.spec.planes[r_idx];
        let g_spec = self.spec.planes[g_idx];
        let b_spec = self.spec.planes[b_idx];
        let a_spec = alpha_idx.map(|i| self.spec.planes[i]);
        let r_p = luma_encode_params(r_spec, self.range)?;
        let g_p = luma_encode_params(g_spec, self.range)?;
        let b_p = luma_encode_params(b_spec, self.range)?;
        let r_pi = luma_encode_params_int(r_spec, self.range)?;
        let g_pi = luma_encode_params_int(g_spec, self.range)?;
        let b_pi = luma_encode_params_int(b_spec, self.range)?;
        let a_p = a_spec.map(alpha_encode_params);
        let inter_ch = self.inter_channels;
        let f32_bytes = row_floats * core::mem::size_of::<f32>();

        let row_buf = &mut ring[..row_floats];
        for y in 0..vis.height {
            stop.check().map_err(|r| err_at(r.into()))?;
            let src_row = src.row(y);
            let gy = vis.y as usize + y as usize;
            if let Some(direct) = self.direct_int_rgb {
                let planes = dst.planes_mut();
                for x in 0..vis.width as usize {
                    let gx = vis.x as usize + x;
                    let (ri, gi, bi, ai_v, max_in) = read_direct_rgb_int(direct, src_row, x);
                    write_rgb_plane_sample_int(
                        r_spec,
                        r_pi,
                        planes[r_idx].row_mut(y),
                        x,
                        ri,
                        max_in,
                        self.depth_policy,
                        gx,
                        gy,
                    );
                    write_rgb_plane_sample_int(
                        g_spec,
                        g_pi,
                        planes[g_idx].row_mut(y),
                        x,
                        gi,
                        max_in,
                        self.depth_policy,
                        gx,
                        gy,
                    );
                    write_rgb_plane_sample_int(
                        b_spec,
                        b_pi,
                        planes[b_idx].row_mut(y),
                        x,
                        bi,
                        max_in,
                        self.depth_policy,
                        gx,
                        gy,
                    );
                    if let (Some(ai), Some((a_s, a_max))) = (alpha_idx, a_p) {
                        write_alpha_sample_int(
                            a_s,
                            a_max,
                            planes[ai].row_mut(y),
                            x,
                            ai_v,
                            max_in,
                            self.depth_policy,
                            gx,
                            gy,
                        );
                    }
                }
                continue;
            }

            self.row_plan.convert_row_at(
                y,
                src_row,
                &mut bytemuck::cast_slice_mut::<f64, u8>(row_buf)[..f32_bytes],
                vis.width,
            )?;
            let row_f32: &[f32] = &bytemuck::cast_slice::<f64, f32>(row_buf)[..row_floats];
            let planes = dst.planes_mut();
            {
                let r_row = planes[r_idx].row_mut(y);
                for x in 0..vis.width as usize {
                    write_rgb_plane_sample(
                        r_spec,
                        r_p,
                        r_row,
                        x,
                        row_f32[x * inter_ch],
                        self.depth_policy,
                        vis.x as usize + x,
                        gy,
                    );
                }
            }
            {
                let g_row = planes[g_idx].row_mut(y);
                for x in 0..vis.width as usize {
                    write_rgb_plane_sample(
                        g_spec,
                        g_p,
                        g_row,
                        x,
                        row_f32[x * inter_ch + 1],
                        self.depth_policy,
                        vis.x as usize + x,
                        gy,
                    );
                }
            }
            {
                let b_row = planes[b_idx].row_mut(y);
                for x in 0..vis.width as usize {
                    write_rgb_plane_sample(
                        b_spec,
                        b_p,
                        b_row,
                        x,
                        row_f32[x * inter_ch + 2],
                        self.depth_policy,
                        vis.x as usize + x,
                        gy,
                    );
                }
            }
            if let (Some(ai), Some((a_s, a_max))) = (alpha_idx, a_p) {
                let a_row = planes[ai].row_mut(y);
                for x in 0..vis.width as usize {
                    write_alpha_sample(
                        a_s,
                        a_max,
                        a_row,
                        x,
                        row_f32[x * inter_ch + 3],
                        self.depth_policy,
                        vis.x as usize + x,
                        gy,
                    );
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn exec_ycbcr(
        &self,
        src: &PixelView<'_>,
        dst: &mut PlanarViewMut<'_>,
        ring: &mut [f64],
        row_floats: usize,
        y_idx: usize,
        chroma_dst: YcbcrDstIndices,
        alpha_idx: Option<usize>,
        stop: &dyn Stop,
    ) -> Result<(), At<ConvertError>> {
        let vis = self.spec.visible_rect;
        let y_spec = self.spec.planes[y_idx];
        let (cb_spec, cr_spec, cb_comp, cr_comp) = match chroma_dst {
            YcbcrDstIndices::Planar { cb_idx, cr_idx } => (
                self.spec.planes[cb_idx],
                self.spec.planes[cr_idx],
                0usize,
                0usize,
            ),
            YcbcrDstIndices::Semi { uv_idx, swapped } => {
                let s = self.spec.planes[uv_idx];
                (s, s, usize::from(swapped), usize::from(!swapped))
            }
        };

        let (y_off, y_span, y_lo, y_hi) = luma_encode_params(y_spec, self.range)?;
        let (cb_mid, cb_span, cb_lo, cb_hi) = match self.matrix {
            ForwardMatrix::Identity => luma_encode_params(cb_spec, self.range)?,
            _ => chroma_encode_params(cb_spec, cb_comp, self.range)?,
        };
        let (cr_mid, cr_span, cr_lo, cr_hi) = match self.matrix {
            ForwardMatrix::Identity => luma_encode_params(cr_spec, self.range)?,
            _ => chroma_encode_params(cr_spec, cr_comp, self.range)?,
        };
        let (y_off_i, y_span_i, _, _) = luma_encode_params_int(y_spec, self.range)?;
        let (cb_mid_i, cb_span_i, _, _) = match self.matrix {
            ForwardMatrix::Identity => luma_encode_params_int(cb_spec, self.range)?,
            _ => chroma_encode_params_int(cb_spec, cb_comp, self.range)?,
        };
        let (cr_mid_i, cr_span_i, _, _) = match self.matrix {
            ForwardMatrix::Identity => luma_encode_params_int(cr_spec, self.range)?,
            _ => chroma_encode_params_int(cr_spec, cr_comp, self.range)?,
        };
        let a_params = alpha_idx.map(|i| alpha_encode_params(self.spec.planes[i]));

        let (cb_hx, cb_vy) = (cb_spec.subsampling.0.max(1), cb_spec.subsampling.1.max(1));
        let (cr_hx, cr_vy) = (cr_spec.subsampling.0.max(1), cr_spec.subsampling.1.max(1));
        let full_res_chroma = cb_hx == 1 && cb_vy == 1 && cr_hx == 1 && cr_vy == 1;

        let (cb_w, cb_h) = cb_spec
            .cropped_plane_dimensions(vis)
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        let (cr_w, cr_h) = cr_spec
            .cropped_plane_dimensions(vis)
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;

        let cb_ox0 = vis.x / u32::from(cb_hx);
        let cb_oy0 = vis.y / u32::from(cb_vy);
        let cr_ox0 = vis.x / u32::from(cr_hx);
        let cr_oy0 = vis.y / u32::from(cr_vy);

        let (cb_dx, cb_dy) = self
            .resolved_siting
            .offset_in_luma_pixels_for_divisors(cb_hx, cb_vy)
            .unwrap_or((0.0, 0.0));
        let (cr_dx, cr_dy) = self
            .resolved_siting
            .offset_in_luma_pixels_for_divisors(cr_hx, cr_vy)
            .unwrap_or((0.0, 0.0));

        let inter_ch = self.inter_channels;
        let f32_bytes = row_floats * core::mem::size_of::<f32>();
        let ring_rows = self.ring_rows;
        let mut next_cb_y = 0u32;
        let mut next_cr_y = 0u32;
        let semi_shared = matches!(chroma_dst, YcbcrDstIndices::Semi { .. });

        let exact_chroma_denoms = self
            .direct_int_rgb
            .and_then(|d| chroma_int_denoms(self.matrix, if d.is_u16 { 65_535 } else { 255 }));

        for y in 0..vis.height {
            stop.check().map_err(|r| err_at(r.into()))?;
            let slot = (y as usize) % ring_rows;
            let slot_slice = &mut ring[slot * row_floats..(slot + 1) * row_floats];
            let src_row = src.row(y);
            if self.direct_int_rgb.is_none() {
                self.row_plan.convert_row_at(
                    y,
                    src_row,
                    &mut bytemuck::cast_slice_mut::<f64, u8>(slot_slice)[..f32_bytes],
                    vis.width,
                )?;
            }

            let planes = dst.planes_mut();
            let gy = vis.y as usize + y as usize;

            if full_res_chroma {
                if let Some(direct) = self.direct_int_rgb {
                    for x in 0..vis.width as usize {
                        let gx = vis.x as usize + x;
                        let (ri, gi, bi, ai_v, max_in) = read_direct_rgb_int(direct, src_row, x);
                        let (y_code, cb_code, cr_code) = match encode_ycbcr_exact_int(
                            self.matrix,
                            ri,
                            gi,
                            bi,
                            max_in,
                            (y_off_i, y_span_i, y_lo, y_hi),
                            (cb_mid_i, cb_span_i, cb_lo, cb_hi),
                            (cr_mid_i, cr_span_i, cr_lo, cr_hi),
                            self.depth_policy,
                            gx,
                            gy,
                        ) {
                            Some(codes) => codes,
                            None => {
                                let inv = 1.0 / (max_in as f64);
                                let (yn, cbn, crn) = apply_forward_matrix_f64(
                                    self.matrix,
                                    (ri as f64) * inv,
                                    (gi as f64) * inv,
                                    (bi as f64) * inv,
                                );
                                (
                                    quantize_code(
                                        y_off + yn * y_span,
                                        y_lo,
                                        y_hi,
                                        self.depth_policy,
                                        gx,
                                        gy,
                                    ),
                                    quantize_code(
                                        cb_mid + cbn * cb_span,
                                        cb_lo,
                                        cb_hi,
                                        self.depth_policy,
                                        gx,
                                        gy,
                                    ),
                                    quantize_code(
                                        cr_mid + crn * cr_span,
                                        cr_lo,
                                        cr_hi,
                                        self.depth_policy,
                                        gx,
                                        gy,
                                    ),
                                )
                            }
                        };
                        write_plane_code(y_spec, planes[y_idx].row_mut(y), x, 0, y_code);
                        match chroma_dst {
                            YcbcrDstIndices::Planar { cb_idx, cr_idx } => {
                                write_plane_code(cb_spec, planes[cb_idx].row_mut(y), x, 0, cb_code);
                                write_plane_code(cr_spec, planes[cr_idx].row_mut(y), x, 0, cr_code);
                            }
                            YcbcrDstIndices::Semi { uv_idx, swapped } => {
                                let (c0, c1) = if swapped {
                                    (cr_code, cb_code)
                                } else {
                                    (cb_code, cr_code)
                                };
                                write_semi_plane_pair(
                                    cb_spec,
                                    planes[uv_idx].row_mut(y),
                                    x,
                                    c0,
                                    c1,
                                );
                            }
                        }
                        if let (Some(ai), Some((a_s, a_max))) = (alpha_idx, a_params) {
                            write_alpha_sample_int(
                                a_s,
                                a_max,
                                planes[ai].row_mut(y),
                                x,
                                ai_v,
                                max_in,
                                self.depth_policy,
                                gx,
                                gy,
                            );
                        }
                    }
                } else {
                    let row_f32: &[f32] =
                        &bytemuck::cast_slice::<f64, f32>(slot_slice)[..row_floats];
                    for x in 0..vis.width as usize {
                        let base = x * inter_ch;
                        let r = f64::from(row_f32[base]).clamp(0.0, 1.0);
                        let g = f64::from(row_f32[base + 1]).clamp(0.0, 1.0);
                        let b = f64::from(row_f32[base + 2]).clamp(0.0, 1.0);
                        let (yn, cbn, crn) = apply_forward_matrix_f64(self.matrix, r, g, b);
                        let gx = vis.x as usize + x;
                        let y_code = quantize_code(
                            y_off + yn * y_span,
                            y_lo,
                            y_hi,
                            self.depth_policy,
                            gx,
                            gy,
                        );
                        let cb_code = quantize_code(
                            cb_mid + cbn * cb_span,
                            cb_lo,
                            cb_hi,
                            self.depth_policy,
                            gx,
                            gy,
                        );
                        let cr_code = quantize_code(
                            cr_mid + crn * cr_span,
                            cr_lo,
                            cr_hi,
                            self.depth_policy,
                            gx,
                            gy,
                        );
                        write_plane_code(y_spec, planes[y_idx].row_mut(y), x, 0, y_code);
                        match chroma_dst {
                            YcbcrDstIndices::Planar { cb_idx, cr_idx } => {
                                write_plane_code(cb_spec, planes[cb_idx].row_mut(y), x, 0, cb_code);
                                write_plane_code(cr_spec, planes[cr_idx].row_mut(y), x, 0, cr_code);
                            }
                            YcbcrDstIndices::Semi { uv_idx, swapped } => {
                                let (c0, c1) = if swapped {
                                    (cr_code, cb_code)
                                } else {
                                    (cb_code, cr_code)
                                };
                                write_semi_plane_pair(
                                    cb_spec,
                                    planes[uv_idx].row_mut(y),
                                    x,
                                    c0,
                                    c1,
                                );
                            }
                        }
                        if let (Some(ai), Some((a_s, a_max))) = (alpha_idx, a_params) {
                            write_alpha_sample(
                                a_s,
                                a_max,
                                planes[ai].row_mut(y),
                                x,
                                row_f32[base + 3],
                                self.depth_policy,
                                gx,
                                gy,
                            );
                        }
                    }
                }
                continue;
            }

            // Subsampled chroma path: exact rational/f64 luma write + store full precision
            // in the f64 ring buffer. When `exact_chroma_denoms` is Some, store exact integer
            // chroma numerators (|numer| <= 6.55e8 << 2^53, 100% lossless in f64) so integer
            // tap kernels accumulate and quantize in exact i128 rational arithmetic with zero
            // tie drift!
            if let Some(direct) = self.direct_int_rgb {
                {
                    let y_row = planes[y_idx].row_mut(y);
                    for x in 0..vis.width as usize {
                        let base = x * inter_ch;
                        let gx = vis.x as usize + x;
                        let (ri, gi, bi, ai_v, max_in) = read_direct_rgb_int(direct, src_row, x);
                        if let Some((cb_n, cr_n)) = exact_chroma_numerators(self.matrix, ri, gi, bi)
                        {
                            slot_slice[base] = 0.0;
                            slot_slice[base + 1] = cb_n as f64;
                            slot_slice[base + 2] = cr_n as f64;
                        } else {
                            let (yn, cbn, crn) =
                                apply_forward_matrix_int_to_f64(self.matrix, ri, gi, bi, max_in);
                            slot_slice[base] = yn;
                            slot_slice[base + 1] = cbn;
                            slot_slice[base + 2] = crn;
                        }
                        if inter_ch > 3 {
                            slot_slice[base + 3] = (ai_v as f64) / (max_in as f64);
                        }
                        let y_code = match encode_luma_exact_int(
                            self.matrix,
                            ri,
                            gi,
                            bi,
                            max_in,
                            (y_off_i, y_span_i, y_lo, y_hi),
                            self.depth_policy,
                            gx,
                            gy,
                        ) {
                            Some(code) => code,
                            None => quantize_code(
                                y_off + slot_slice[base] * y_span,
                                y_lo,
                                y_hi,
                                self.depth_policy,
                                gx,
                                gy,
                            ),
                        };
                        write_plane_code(y_spec, y_row, x, 0, y_code);
                    }
                }
                if let (Some(ai), Some((a_s, a_max))) = (alpha_idx, a_params) {
                    let a_row = planes[ai].row_mut(y);
                    for x in 0..vis.width as usize {
                        let (_, _, _, ai_v, max_in) = read_direct_rgb_int(direct, src_row, x);
                        write_alpha_sample_int(
                            a_s,
                            a_max,
                            a_row,
                            x,
                            ai_v,
                            max_in,
                            self.depth_policy,
                            vis.x as usize + x,
                            gy,
                        );
                    }
                }
            } else {
                // In-place reverse widening from Stage 1's f32 prefix [0..row_floats*4B]
                // into full f64 [0..row_floats*8B].
                let y_row = planes[y_idx].row_mut(y);
                for x in (0..vis.width as usize).rev() {
                    let base = x * inter_ch;
                    let (r, g, b, a_f) = {
                        let f32s = bytemuck::cast_slice::<f64, f32>(slot_slice);
                        let r = f64::from(f32s[base]).clamp(0.0, 1.0);
                        let g = f64::from(f32s[base + 1]).clamp(0.0, 1.0);
                        let b = f64::from(f32s[base + 2]).clamp(0.0, 1.0);
                        let a_f = if inter_ch > 3 {
                            f64::from(f32s[base + 3])
                        } else {
                            1.0
                        };
                        (r, g, b, a_f)
                    };
                    let (yn, cbn, crn) = apply_forward_matrix_f64(self.matrix, r, g, b);
                    slot_slice[base] = yn;
                    slot_slice[base + 1] = cbn;
                    slot_slice[base + 2] = crn;
                    if inter_ch > 3 {
                        slot_slice[base + 3] = a_f;
                    }
                    let y_code = quantize_code(
                        y_off + yn * y_span,
                        y_lo,
                        y_hi,
                        self.depth_policy,
                        vis.x as usize + x,
                        gy,
                    );
                    write_plane_code(y_spec, y_row, x, 0, y_code);
                }
                if let (Some(ai), Some((a_s, a_max))) = (alpha_idx, a_params) {
                    let a_row = planes[ai].row_mut(y);
                    for x in 0..vis.width as usize {
                        write_alpha_sample_f64(
                            a_s,
                            a_max,
                            a_row,
                            x,
                            slot_slice[x * inter_ch + 3],
                            self.depth_policy,
                            vis.x as usize + x,
                            gy,
                        );
                    }
                }
            }

            let is_last_row = y + 1 == vis.height;

            if semi_shared {
                let YcbcrDstIndices::Semi { uv_idx, swapped } = chroma_dst else {
                    unreachable!()
                };
                while next_cb_y < cb_h {
                    let vtaps = axis_downsample_taps(
                        cb_oy0 + next_cb_y,
                        cb_vy,
                        cb_dy,
                        vis.y,
                        vis.height,
                        self.options.downsample,
                    );
                    if !is_last_row && vtaps.max_idx > y {
                        break;
                    }
                    let uv_row = planes[uv_idx].row_mut(next_cb_y);
                    let gcy = (cb_oy0 + next_cb_y) as usize;
                    for cx in 0..cb_w {
                        let htaps = axis_downsample_taps(
                            cb_ox0 + cx,
                            cb_hx,
                            cb_dx,
                            vis.x,
                            vis.width,
                            self.options.downsample,
                        );
                        let gcx = (cb_ox0 + cx) as usize;
                        let (cb_code, cr_code) = if let (
                            Some((cb_base_den, cr_base_den)),
                            Some((vw, vsum)),
                            Some((hw, hsum)),
                        ) =
                            (exact_chroma_denoms, vtaps.int_weights, htaps.int_weights)
                        {
                            let (acc_cb, acc_cr) = downsample_2ch_ring_int(
                                ring, ring_rows, row_floats, inter_ch, &vtaps, &vw, &htaps, &hw,
                            );
                            let w_den = i128::from(vsum) * i128::from(hsum);
                            let cb_den = cb_base_den * w_den;
                            let cr_den = cr_base_den * w_den;
                            let cb_code = quantize_rational(
                                cb_mid_i * cb_den + acc_cb * cb_span_i,
                                cb_den,
                                cb_lo,
                                cb_hi,
                                self.depth_policy,
                                gcx,
                                gcy,
                            );
                            let cr_code = quantize_rational(
                                cr_mid_i * cr_den + acc_cr * cr_span_i,
                                cr_den,
                                cr_lo,
                                cr_hi,
                                self.depth_policy,
                                gcx,
                                gcy,
                            );
                            (cb_code, cr_code)
                        } else {
                            let (mut cbv, mut crv) = downsample_2ch_ring(
                                ring, ring_rows, row_floats, inter_ch, &vtaps, &htaps,
                            );
                            if let Some((cb_base_den, cr_base_den)) = exact_chroma_denoms {
                                cbv /= cb_base_den as f64;
                                crv /= cr_base_den as f64;
                            }
                            (
                                quantize_code(
                                    cb_mid + cbv * cb_span,
                                    cb_lo,
                                    cb_hi,
                                    self.depth_policy,
                                    gcx,
                                    gcy,
                                ),
                                quantize_code(
                                    cr_mid + crv * cr_span,
                                    cr_lo,
                                    cr_hi,
                                    self.depth_policy,
                                    gcx,
                                    gcy,
                                ),
                            )
                        };
                        let (c0, c1) = if swapped {
                            (cr_code, cb_code)
                        } else {
                            (cb_code, cr_code)
                        };
                        write_semi_plane_pair(cb_spec, uv_row, cx as usize, c0, c1);
                    }
                    next_cb_y += 1;
                }
            } else {
                let YcbcrDstIndices::Planar { cb_idx, cr_idx } = chroma_dst else {
                    unreachable!()
                };
                while next_cb_y < cb_h {
                    let vtaps = axis_downsample_taps(
                        cb_oy0 + next_cb_y,
                        cb_vy,
                        cb_dy,
                        vis.y,
                        vis.height,
                        self.options.downsample,
                    );
                    if !is_last_row && vtaps.max_idx > y {
                        break;
                    }
                    let cb_row = planes[cb_idx].row_mut(next_cb_y);
                    let gcy = (cb_oy0 + next_cb_y) as usize;
                    for cx in 0..cb_w {
                        let htaps = axis_downsample_taps(
                            cb_ox0 + cx,
                            cb_hx,
                            cb_dx,
                            vis.x,
                            vis.width,
                            self.options.downsample,
                        );
                        let gcx = (cb_ox0 + cx) as usize;
                        let cb_code =
                            if let (Some((cb_base_den, _)), Some((vw, vsum)), Some((hw, hsum))) =
                                (exact_chroma_denoms, vtaps.int_weights, htaps.int_weights)
                            {
                                let acc_cb = downsample_1ch_ring_int(
                                    ring, ring_rows, row_floats, inter_ch, 1, &vtaps, &vw, &htaps,
                                    &hw,
                                );
                                let cb_den = cb_base_den * i128::from(vsum) * i128::from(hsum);
                                quantize_rational(
                                    cb_mid_i * cb_den + acc_cb * cb_span_i,
                                    cb_den,
                                    cb_lo,
                                    cb_hi,
                                    self.depth_policy,
                                    gcx,
                                    gcy,
                                )
                            } else {
                                let mut cbv = downsample_1ch_ring(
                                    ring, ring_rows, row_floats, inter_ch, 1, &vtaps, &htaps,
                                );
                                if let Some((cb_base_den, _)) = exact_chroma_denoms {
                                    cbv /= cb_base_den as f64;
                                }
                                quantize_code(
                                    cb_mid + cbv * cb_span,
                                    cb_lo,
                                    cb_hi,
                                    self.depth_policy,
                                    gcx,
                                    gcy,
                                )
                            };
                        write_plane_code(cb_spec, cb_row, cx as usize, 0, cb_code);
                    }
                    next_cb_y += 1;
                }
                while next_cr_y < cr_h {
                    let vtaps = axis_downsample_taps(
                        cr_oy0 + next_cr_y,
                        cr_vy,
                        cr_dy,
                        vis.y,
                        vis.height,
                        self.options.downsample,
                    );
                    if !is_last_row && vtaps.max_idx > y {
                        break;
                    }
                    let cr_row = planes[cr_idx].row_mut(next_cr_y);
                    let gcy = (cr_oy0 + next_cr_y) as usize;
                    for cx in 0..cr_w {
                        let htaps = axis_downsample_taps(
                            cr_ox0 + cx,
                            cr_hx,
                            cr_dx,
                            vis.x,
                            vis.width,
                            self.options.downsample,
                        );
                        let gcx = (cr_ox0 + cx) as usize;
                        let cr_code =
                            if let (Some((_, cr_base_den)), Some((vw, vsum)), Some((hw, hsum))) =
                                (exact_chroma_denoms, vtaps.int_weights, htaps.int_weights)
                            {
                                let acc_cr = downsample_1ch_ring_int(
                                    ring, ring_rows, row_floats, inter_ch, 2, &vtaps, &vw, &htaps,
                                    &hw,
                                );
                                let cr_den = cr_base_den * i128::from(vsum) * i128::from(hsum);
                                quantize_rational(
                                    cr_mid_i * cr_den + acc_cr * cr_span_i,
                                    cr_den,
                                    cr_lo,
                                    cr_hi,
                                    self.depth_policy,
                                    gcx,
                                    gcy,
                                )
                            } else {
                                let mut crv = downsample_1ch_ring(
                                    ring, ring_rows, row_floats, inter_ch, 2, &vtaps, &htaps,
                                );
                                if let Some((_, cr_base_den)) = exact_chroma_denoms {
                                    crv /= cr_base_den as f64;
                                }
                                quantize_code(
                                    cr_mid + crv * cr_span,
                                    cr_lo,
                                    cr_hi,
                                    self.depth_policy,
                                    gcx,
                                    gcy,
                                )
                            };
                        write_plane_code(cr_spec, cr_row, cx as usize, 0, cr_code);
                    }
                    next_cr_y += 1;
                }
            }
        }

        Ok(())
    }
}

#[inline(always)]
fn apply_forward_matrix_f64(matrix: ForwardMatrix, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    match matrix {
        ForwardMatrix::Identity => (g, b, r),
        ForwardMatrix::YCgCo => {
            let co = 0.5 * (r - b);
            let t = b + co;
            let cg = 0.5 * (g - t);
            let yv = t + cg;
            (yv, cg, co)
        }
        ForwardMatrix::Ncl {
            kr,
            kg,
            kb,
            inv_2_1_minus_kb,
            inv_2_1_minus_kr,
            ..
        } => {
            let yv = kr * r + kg * g + kb * b;
            let cbv = ((b - yv) * inv_2_1_minus_kb).clamp(-0.5, 0.5);
            let crv = ((r - yv) * inv_2_1_minus_kr).clamp(-0.5, 0.5);
            (yv, cbv, crv)
        }
    }
}

#[derive(Clone, Copy)]
enum YcbcrDstIndices {
    Planar { cb_idx: usize, cr_idx: usize },
    Semi { uv_idx: usize, swapped: bool },
}

/// Up to 8 taps for 1D downsampling (covers factors 1, 2, and 4 with triangle/box kernels).
#[derive(Clone, Copy)]
struct AxisTaps {
    indices: [u32; 8],
    weights: [f64; 8],
    /// Exact integer `(weights, weight_sum)` when kernel weights are exact rational fractions
    /// (all standard half-integer H.273 chroma sitings, `BoxAverage`, `Nearest`, and `1×`).
    int_weights: Option<([i64; 8], i64)>,
    len: usize,
    max_idx: u32,
}

#[inline]
fn axis_downsample_taps(
    global_chroma_idx: u32,
    factor: u8,
    offset_in_cell: f32,
    vis_origin: u32,
    vis_len: u32,
    mode: ChromaDownsample,
) -> AxisTaps {
    let f = u32::from(factor.max(1));
    let max_local = vis_len.saturating_sub(1);
    if f == 1 {
        let idx = global_chroma_idx.saturating_sub(vis_origin).min(max_local);
        return AxisTaps {
            indices: [idx, 0, 0, 0, 0, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            int_weights: Some(([1, 0, 0, 0, 0, 0, 0, 0], 1)),
            len: 1,
            max_idx: idx,
        };
    }

    // Compute sample center in *local* visible coordinates using exact i64 integer difference
    // BEFORE adding the fractional siting offset in f64 — immune to f32 precision loss even
    // at coded crop origins >= 2^24!
    let base_local: i64 = i64::from(global_chroma_idx) * i64::from(f) - i64::from(vis_origin);
    let clamp_local = |rel: i64| -> u32 {
        if rel <= 0 {
            0
        } else if rel >= i64::from(max_local) {
            max_local
        } else {
            rel as u32
        }
    };

    match mode {
        ChromaDownsample::Nearest => {
            let center_local = (base_local as f64) + f64::from(offset_in_cell);
            let rel = libm::floor(center_local + 0.5) as i64;
            let idx = clamp_local(rel);
            AxisTaps {
                indices: [idx, 0, 0, 0, 0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                int_weights: Some(([1, 0, 0, 0, 0, 0, 0, 0], 1)),
                len: 1,
                max_idx: idx,
            }
        }
        ChromaDownsample::BoxAverage => {
            let mut indices = [0u32; 8];
            let mut weights = [0.0f64; 8];
            let mut iw = [0i64; 8];
            let count = (f as usize).min(8);
            let w = 1.0 / (count as f64);
            let mut max_idx = 0u32;
            for i in 0..count {
                let idx = clamp_local(base_local + i as i64);
                indices[i] = idx;
                weights[i] = w;
                iw[i] = 1;
                if idx > max_idx {
                    max_idx = idx;
                }
            }
            AxisTaps {
                indices,
                weights,
                int_weights: Some((iw, count as i64)),
                len: count,
                max_idx,
            }
        }
        ChromaDownsample::SitingAware => {
            // Continuous triangle kernel of half-width `f` centered at `base_local + offset`.
            // Exact `[1, 2, 1]/4` for co-sited 2×, `[1, 3, 3, 1]/8` for centered 2×,
            // `[1, 2, 3, 4, 3, 2, 1]/16` for co-sited 4×, `[1, 3, 5, 7, 7, 5, 3, 1]/32` for centered 4×.
            let offset_2x = f64::from(offset_in_cell) * 2.0;
            let offset_2x_round = libm::round(offset_2x);
            let is_half_int = libm::fabs(offset_2x - offset_2x_round) == 0.0;
            let center_2x: i64 = base_local * 2 + (offset_2x_round as i64);
            let radius_2x: i64 = i64::from(f) * 2;

            let center_local = (base_local as f64) + f64::from(offset_in_cell);
            let radius = f64::from(f);
            let k_start = libm::floor(center_local - radius) as i64 + 1;
            let k_end = libm::ceil(center_local + radius) as i64 - 1;
            let mut indices = [0u32; 8];
            let mut weights = [0.0f64; 8];
            let mut iw = [0i64; 8];
            let mut iw_sum = 0i64;
            let mut len = 0usize;
            let mut sum = 0.0f64;
            let mut max_idx = 0u32;
            let mut k = k_start;
            while k <= k_end && len < 8 {
                let dist = libm::fabs((k as f64) - center_local);
                let w = 1.0 - dist / radius;
                if w > 0.0 {
                    let idx = clamp_local(k);
                    indices[len] = idx;
                    weights[len] = w;
                    sum += w;
                    if is_half_int {
                        let wi = radius_2x - (2 * k - center_2x).abs();
                        iw[len] = wi;
                        iw_sum += wi;
                    }
                    if idx > max_idx {
                        max_idx = idx;
                    }
                    len += 1;
                }
                k += 1;
            }
            if len == 0 || sum <= 0.0 {
                let idx = clamp_local(libm::floor(center_local + 0.5) as i64);
                return AxisTaps {
                    indices: [idx, 0, 0, 0, 0, 0, 0, 0],
                    weights: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    int_weights: Some(([1, 0, 0, 0, 0, 0, 0, 0], 1)),
                    len: 1,
                    max_idx: idx,
                };
            }
            let inv_sum = 1.0 / sum;
            for w in &mut weights[..len] {
                *w *= inv_sum;
            }
            AxisTaps {
                indices,
                weights,
                int_weights: if is_half_int && iw_sum > 0 {
                    Some((iw, iw_sum))
                } else {
                    None
                },
                len,
                max_idx,
            }
        }
    }
}

#[inline(always)]
fn chroma_int_denoms(matrix: ForwardMatrix, max_in: i128) -> Option<(i128, i128)> {
    match matrix {
        ForwardMatrix::YCgCo => Some((4 * max_in, 2 * max_in)),
        ForwardMatrix::Ncl {
            exact_rational: Some((kr, _, kb, k_den)),
            ..
        } => Some((2 * (k_den - kb) * max_in, 2 * (k_den - kr) * max_in)),
        _ => None,
    }
}

#[inline(always)]
fn exact_chroma_numerators(
    matrix: ForwardMatrix,
    r: i128,
    g: i128,
    b: i128,
) -> Option<(i128, i128)> {
    match matrix {
        ForwardMatrix::YCgCo => Some((2 * g - r - b, r - b)),
        ForwardMatrix::Ncl {
            exact_rational: Some((kr, kg, kb, k_den)),
            ..
        } => {
            let y_sum = kr * r + kg * g + kb * b;
            Some((k_den * b - y_sum, k_den * r - y_sum))
        }
        _ => None,
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn downsample_1ch_ring_int(
    ring: &[f64],
    ring_rows: usize,
    row_floats: usize,
    inter_ch: usize,
    ch_offset: usize,
    vtaps: &AxisTaps,
    vw: &[i64; 8],
    htaps: &AxisTaps,
    hw: &[i64; 8],
) -> i128 {
    let mut acc = 0i128;
    for (&vidx, &w_v) in vtaps.indices[..vtaps.len].iter().zip(&vw[..vtaps.len]) {
        let wy = i128::from(w_v);
        let row_slot = (vidx as usize) % ring_rows;
        let row_base = row_slot * row_floats + ch_offset;
        let mut h_acc = 0i128;
        for (&hidx, &w_h) in htaps.indices[..htaps.len].iter().zip(&hw[..htaps.len]) {
            let x = hidx as usize;
            h_acc += (ring[row_base + x * inter_ch] as i128) * i128::from(w_h);
        }
        acc += h_acc * wy;
    }
    acc
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn downsample_2ch_ring_int(
    ring: &[f64],
    ring_rows: usize,
    row_floats: usize,
    inter_ch: usize,
    vtaps: &AxisTaps,
    vw: &[i64; 8],
    htaps: &AxisTaps,
    hw: &[i64; 8],
) -> (i128, i128) {
    let mut acc_cb = 0i128;
    let mut acc_cr = 0i128;
    for (&vidx, &w_v) in vtaps.indices[..vtaps.len].iter().zip(&vw[..vtaps.len]) {
        let wy = i128::from(w_v);
        let row_slot = (vidx as usize) % ring_rows;
        let row_base = row_slot * row_floats;
        let mut h_cb = 0i128;
        let mut h_cr = 0i128;
        for (&hidx, &w_h) in htaps.indices[..htaps.len].iter().zip(&hw[..htaps.len]) {
            let off = row_base + (hidx as usize) * inter_ch;
            let wx = i128::from(w_h);
            h_cb += (ring[off + 1] as i128) * wx;
            h_cr += (ring[off + 2] as i128) * wx;
        }
        acc_cb += h_cb * wy;
        acc_cr += h_cr * wy;
    }
    (acc_cb, acc_cr)
}

#[inline]
fn downsample_1ch_ring(
    ring: &[f64],
    ring_rows: usize,
    row_floats: usize,
    inter_ch: usize,
    ch_offset: usize,
    vtaps: &AxisTaps,
    htaps: &AxisTaps,
) -> f64 {
    let mut acc = 0.0f64;
    for vi in 0..vtaps.len {
        let wy = vtaps.weights[vi];
        let row_slot = (vtaps.indices[vi] as usize) % ring_rows;
        let row_base = row_slot * row_floats + ch_offset;
        let mut h_acc = 0.0f64;
        for hi in 0..htaps.len {
            let x = htaps.indices[hi] as usize;
            h_acc += ring[row_base + x * inter_ch] * htaps.weights[hi];
        }
        acc += h_acc * wy;
    }
    acc
}

#[inline]
fn downsample_2ch_ring(
    ring: &[f64],
    ring_rows: usize,
    row_floats: usize,
    inter_ch: usize,
    vtaps: &AxisTaps,
    htaps: &AxisTaps,
) -> (f64, f64) {
    let mut acc_cb = 0.0f64;
    let mut acc_cr = 0.0f64;
    for vi in 0..vtaps.len {
        let wy = vtaps.weights[vi];
        let row_slot = (vtaps.indices[vi] as usize) % ring_rows;
        let row_base = row_slot * row_floats;
        let mut h_cb = 0.0f64;
        let mut h_cr = 0.0f64;
        for hi in 0..htaps.len {
            let off = row_base + (htaps.indices[hi] as usize) * inter_ch;
            let wx = htaps.weights[hi];
            h_cb += ring[off + 1] * wx;
            h_cr += ring[off + 2] * wx;
        }
        acc_cb += h_cb * wy;
        acc_cr += h_cr * wy;
    }
    (acc_cb, acc_cr)
}

#[allow(clippy::type_complexity)]
fn parse_target_planes(
    planes: &[PlaneFormat],
    vis: PixelRect,
) -> Result<(TargetLayout, bool, bool, bool, bool, u8, u8), At<ConvertError>> {
    let mut y_idx = None;
    let mut cb_idx = None;
    let mut cr_idx = None;
    let mut semi_idx = None;
    let mut r_idx = None;
    let mut g_idx = None;
    let mut b_idx = None;
    let mut alpha_idx = None;

    for (i, &pf) in planes.iter().enumerate() {
        pf.validate().map_err(|e| err_at(e.into()))?;
        let (pw, ph) = pf
            .cropped_plane_dimensions(vis)
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        if pw == 0 || ph == 0 {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        if pf.subsampling.0 > 4 || pf.subsampling.1 > 4 {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        match pf.role {
            PlaneRole::Y => {
                if y_idx.replace(i).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            PlaneRole::Cb => {
                if cb_idx.replace(i).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            PlaneRole::Cr => {
                if cr_idx.replace(i).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            PlaneRole::CbCr | PlaneRole::CrCb => {
                if semi_idx.replace(i).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            PlaneRole::R => {
                if r_idx.replace(i).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            PlaneRole::G => {
                if g_idx.replace(i).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            PlaneRole::B => {
                if b_idx.replace(i).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            PlaneRole::Alpha => {
                if alpha_idx.replace(i).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            _ => return Err(err_at(ConvertError::UnsupportedRoute)),
        }
    }

    let check_plane = |pf: PlaneFormat,
                       expected_comp: u8,
                       full_res: bool,
                       allow_f32: bool|
     -> Result<u8, At<ConvertError>> {
        let sample_ok = matches!(
            pf.sample,
            SampleType::U8 | SampleType::U16 | SampleType::U32
        ) || (allow_f32 && pf.sample == SampleType::F32 && pf.packing.is_full());
        if !sample_ok || pf.components != expected_comp {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        if full_res && pf.subsampling != (1, 1) {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        if pf.sample == SampleType::F32 {
            Ok(32)
        } else {
            let max_code = pf.packing.max_code(pf.sample, 0);
            if max_code == 0 {
                return Err(err_at(ConvertError::UnsupportedRoute));
            }
            Ok((32 - max_code.leading_zeros()) as u8)
        }
    };

    if let (Some(ri), Some(gi), Some(bi)) = (r_idx, g_idx, b_idx) {
        if y_idx.is_some() || cb_idx.is_some() || cr_idx.is_some() || semi_idx.is_some() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        let mut min_bits = check_plane(planes[ri], 1, true, true)?;
        min_bits = min_bits.min(check_plane(planes[gi], 1, true, true)?);
        min_bits = min_bits.min(check_plane(planes[bi], 1, true, true)?);
        if let Some(ai) = alpha_idx {
            min_bits = min_bits.min(check_plane(planes[ai], 1, true, true)?);
        }
        return Ok((
            TargetLayout::PlanarRgb {
                r_idx: ri,
                g_idx: gi,
                b_idx: bi,
                alpha_idx,
            },
            alpha_idx.is_some(),
            false,
            true,
            false,
            1,
            min_bits,
        ));
    }
    if r_idx.is_some() || g_idx.is_some() || b_idx.is_some() {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }

    let yi = y_idx.ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
    let mut min_bits = check_plane(planes[yi], 1, true, false)?;
    if let Some(ai) = alpha_idx {
        min_bits = min_bits.min(check_plane(planes[ai], 1, true, false)?);
    }

    if let Some(si) = semi_idx {
        if cb_idx.is_some() || cr_idx.is_some() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        let pf = planes[si];
        min_bits = min_bits.min(check_plane(pf, 2, false, false)?);
        let any_sub = pf.subsampling.0 > 1 || pf.subsampling.1 > 1;
        Ok((
            TargetLayout::YcbcrSemi {
                y_idx: yi,
                uv_idx: si,
                swapped: pf.role == PlaneRole::CrCb,
                alpha_idx,
            },
            alpha_idx.is_some(),
            false,
            false,
            any_sub,
            pf.subsampling.1,
            min_bits,
        ))
    } else if cb_idx.is_none() && cr_idx.is_none() {
        Ok((
            TargetLayout::Monochrome {
                y_idx: yi,
                alpha_idx,
            },
            alpha_idx.is_some(),
            true,
            false,
            false,
            1,
            min_bits,
        ))
    } else {
        let cbi = cb_idx.ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        let cri = cr_idx.ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        min_bits = min_bits.min(check_plane(planes[cbi], 1, false, false)?);
        min_bits = min_bits.min(check_plane(planes[cri], 1, false, false)?);
        let cb_sub = planes[cbi].subsampling;
        let cr_sub = planes[cri].subsampling;
        let any_sub = cb_sub.0 > 1 || cb_sub.1 > 1 || cr_sub.0 > 1 || cr_sub.1 > 1;
        let max_vy = cb_sub.1.max(cr_sub.1);
        Ok((
            TargetLayout::YcbcrPlanar {
                y_idx: yi,
                cb_idx: cbi,
                cr_idx: cri,
                alpha_idx,
            },
            alpha_idx.is_some(),
            false,
            false,
            any_sub,
            max_vy,
            min_bits,
        ))
    }
}

fn src_effective_bits(src: &PixelFormat) -> u8 {
    if !src.layout.packing.is_full() {
        let mc = src.layout.packing.max_code(src.layout.sample, 0);
        return (32 - mc.leading_zeros()) as u8;
    }
    src.layout.sample.bits()
}

/// Returns `(offset, span, min_code, max_code)` in exact `f64` for luma / planar-RGB / Identity-GBR.
fn luma_encode_params(
    spec: PlaneFormat,
    range: SignalRange,
) -> Result<(f64, f64, u32, u32), At<ConvertError>> {
    if spec.sample == SampleType::F32 {
        if range != SignalRange::Full {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        return Ok((0.0, 1.0, 0, 1));
    }
    let max_u32 = spec.packing.max_code(spec.sample, 0).max(1);
    let bits = (32 - max_u32.leading_zeros()) as i32;
    match range {
        SignalRange::Full => Ok((0.0, f64::from(max_u32), 0, max_u32)),
        SignalRange::Narrow => {
            if bits < 8 {
                return Err(err_at(ConvertError::UnsupportedRoute));
            }
            let shift = (bits - 8) as u32;
            let lo = 16u32 << shift;
            let span = 219u32 << shift;
            let hi = (lo + span).min(max_u32);
            Ok((f64::from(lo), f64::from(span), lo, hi))
        }
        _ => Err(err_at(ConvertError::UnsupportedRoute)),
    }
}

/// Returns `(mid, span, min_code, max_code)` in exact `f64` for Cb/Cr/Cg/Co chroma encoding.
fn chroma_encode_params(
    spec: PlaneFormat,
    comp: usize,
    range: SignalRange,
) -> Result<(f64, f64, u32, u32), At<ConvertError>> {
    let max_u32 = spec.packing.max_code(spec.sample, comp).max(1);
    let bits = (32 - max_u32.leading_zeros()) as i32;
    match range {
        SignalRange::Full => {
            let mid = f64::from(1u32 << (bits - 1).max(0));
            Ok((mid, f64::from(max_u32), 0, max_u32))
        }
        SignalRange::Narrow => {
            if bits < 8 {
                return Err(err_at(ConvertError::UnsupportedRoute));
            }
            let shift = (bits - 8) as u32;
            let mid = f64::from(128u32 << shift);
            let span = f64::from(224u32 << shift);
            let lo = 16u32 << shift;
            let hi = (240u32 << shift).min(max_u32);
            Ok((mid, span, lo, hi))
        }
        _ => Err(err_at(ConvertError::UnsupportedRoute)),
    }
}

fn alpha_encode_params(spec: PlaneFormat) -> (PlaneFormat, u32) {
    if spec.sample == SampleType::F32 {
        (spec, 1)
    } else {
        (spec, spec.packing.max_code(spec.sample, 0).max(1))
    }
}

#[inline(always)]
fn quantize_code(
    val: f64,
    lo: u32,
    hi: u32,
    depth_policy: DepthPolicy,
    gx: usize,
    gy: usize,
) -> u32 {
    let clamped = val.clamp(f64::from(lo), f64::from(hi));
    let q = match depth_policy {
        DepthPolicy::Truncate => libm::floor(clamped) as u32,
        DepthPolicy::OrderedDither => {
            let t = BAYER_4X4[((gy & 3) << 2) | (gx & 3)];
            libm::floor(clamped + t) as u32
        }
        _ => libm::floor(clamped + 0.5) as u32,
    };
    q.clamp(lo, hi)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn write_rgb_plane_sample(
    spec: PlaneFormat,
    params: (f64, f64, u32, u32),
    row: &mut [u8],
    x: usize,
    v: f32,
    depth_policy: DepthPolicy,
    gx: usize,
    gy: usize,
) {
    if spec.sample == SampleType::F32 {
        let off = x * 4;
        row[off..off + 4].copy_from_slice(&v.to_ne_bytes());
    } else {
        let (off, span, lo, hi) = params;
        let code = quantize_code(
            off + f64::from(v).clamp(0.0, 1.0) * span,
            lo,
            hi,
            depth_policy,
            gx,
            gy,
        );
        write_plane_code(spec, row, x, 0, code);
    }
}

fn detect_direct_int_rgb(
    src: &PixelFormat,
    inter_fmt: &PixelFormat,
    has_alpha: bool,
) -> Option<DirectIntRgb> {
    if !src.layout.packing.is_full() {
        return None;
    }
    let is_u16 = match src.layout.sample {
        SampleType::U8 => false,
        SampleType::U16 => true,
        _ => return None,
    };
    let (
        ColorSpace::Rgb {
            primaries: sp,
            transfer: st,
            alpha: sa,
            range: SignalRange::Full,
            white: sw,
        },
        ColorSpace::Rgb {
            primaries: dp,
            transfer: dt,
            alpha: da,
            range: SignalRange::Full,
            white: dw,
        },
    ) = (&src.color, &inter_fmt.color)
    else {
        return None;
    };
    if sp != dp || st != dt || sw != dw {
        return None;
    }
    if has_alpha {
        if sa != da {
            return None;
        }
    } else if !matches!(
        sa,
        AlphaMode::None | AlphaMode::Opaque | AlphaMode::Undefined
    ) {
        return None;
    }
    let (channels, r_idx, g_idx, b_idx, a_idx) = match src.layout.order {
        zenpixels::ChannelOrder::Rgb => (3, 0, 1, 2, None),
        zenpixels::ChannelOrder::Bgr => (3, 2, 1, 0, None),
        zenpixels::ChannelOrder::Rgba => (4, 0, 1, 2, Some(3)),
        zenpixels::ChannelOrder::Bgra => (4, 2, 1, 0, Some(3)),
        zenpixels::ChannelOrder::Rgbx => (4, 0, 1, 2, None),
        zenpixels::ChannelOrder::Bgrx => (4, 2, 1, 0, None),
        _ => return None,
    };
    if has_alpha && a_idx.is_none() {
        return None;
    }
    Some(DirectIntRgb {
        is_u16,
        channels,
        r_idx,
        g_idx,
        b_idx,
        a_idx,
    })
}

#[inline(always)]
fn read_direct_rgb_int(d: DirectIntRgb, row: &[u8], x: usize) -> (i128, i128, i128, i128, i128) {
    let base = x * d.channels;
    if d.is_u16 {
        let read_ch = |ch: usize| -> i128 {
            let off = (base + ch) * 2;
            i128::from(u16::from_ne_bytes([row[off], row[off + 1]]))
        };
        let a = d.a_idx.map_or(65_535, read_ch);
        (
            read_ch(d.r_idx),
            read_ch(d.g_idx),
            read_ch(d.b_idx),
            a,
            65_535,
        )
    } else {
        let read_ch = |ch: usize| -> i128 { i128::from(row[base + ch]) };
        let a = d.a_idx.map_or(255, read_ch);
        (read_ch(d.r_idx), read_ch(d.g_idx), read_ch(d.b_idx), a, 255)
    }
}

/// Exact integer `(offset, span, min_code, max_code)` for luma / planar-RGB / Identity-GBR.
fn luma_encode_params_int(
    spec: PlaneFormat,
    range: SignalRange,
) -> Result<(i128, i128, u32, u32), At<ConvertError>> {
    if spec.sample == SampleType::F32 {
        if range != SignalRange::Full {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        return Ok((0, 1, 0, 1));
    }
    let max_u32 = spec.packing.max_code(spec.sample, 0).max(1);
    let bits = (32 - max_u32.leading_zeros()) as i32;
    match range {
        SignalRange::Full => Ok((0, i128::from(max_u32), 0, max_u32)),
        SignalRange::Narrow => {
            if bits < 8 {
                return Err(err_at(ConvertError::UnsupportedRoute));
            }
            let shift = (bits - 8) as u32;
            let lo = 16u32 << shift;
            let span = 219u32 << shift;
            let hi = (lo + span).min(max_u32);
            Ok((i128::from(lo), i128::from(span), lo, hi))
        }
        _ => Err(err_at(ConvertError::UnsupportedRoute)),
    }
}

/// Exact integer `(mid, span, min_code, max_code)` for Cb/Cr/Cg/Co chroma encoding.
fn chroma_encode_params_int(
    spec: PlaneFormat,
    comp: usize,
    range: SignalRange,
) -> Result<(i128, i128, u32, u32), At<ConvertError>> {
    let max_u32 = spec.packing.max_code(spec.sample, comp).max(1);
    let bits = (32 - max_u32.leading_zeros()) as i32;
    match range {
        SignalRange::Full => {
            let mid = 1u32 << (bits - 1).max(0);
            Ok((i128::from(mid), i128::from(max_u32), 0, max_u32))
        }
        SignalRange::Narrow => {
            if bits < 8 {
                return Err(err_at(ConvertError::UnsupportedRoute));
            }
            let shift = (bits - 8) as u32;
            let mid = 128u32 << shift;
            let span = 224u32 << shift;
            let lo = 16u32 << shift;
            let hi = (240u32 << shift).min(max_u32);
            Ok((i128::from(mid), i128::from(span), lo, hi))
        }
        _ => Err(err_at(ConvertError::UnsupportedRoute)),
    }
}

/// Quantize exact rational `num / den` (`den > 0`) under `depth_policy`, clamped to `[lo, hi]`.
#[inline(always)]
fn quantize_rational(
    num: i128,
    den: i128,
    lo: u32,
    hi: u32,
    depth_policy: DepthPolicy,
    gx: usize,
    gy: usize,
) -> u32 {
    let lo_i = i128::from(lo);
    let hi_i = i128::from(hi);
    // Clamp rational before rounding so values outside [lo, hi] never shift dither/round results.
    let clamped_num = num.clamp(lo_i * den, hi_i * den);
    let q = match depth_policy {
        DepthPolicy::Truncate => clamped_num.div_euclid(den),
        DepthPolicy::OrderedDither => {
            let b = BAYER_4X4_INT[((gy & 3) << 2) | (gx & 3)];
            (32 * clamped_num + (2 * b + 1) * den).div_euclid(32 * den)
        }
        _ => (2 * clamped_num + den).div_euclid(2 * den),
    };
    q.clamp(lo_i, hi_i) as u32
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn encode_luma_exact_int(
    matrix: ForwardMatrix,
    r: i128,
    g: i128,
    b: i128,
    max_in: i128,
    (y_off, y_span, y_lo, y_hi): (i128, i128, u32, u32),
    depth_policy: DepthPolicy,
    gx: usize,
    gy: usize,
) -> Option<u32> {
    match matrix {
        ForwardMatrix::Identity => Some(quantize_rational(
            y_off * max_in + g * y_span,
            max_in,
            y_lo,
            y_hi,
            depth_policy,
            gx,
            gy,
        )),
        ForwardMatrix::YCgCo => {
            let den = 4 * max_in;
            let num = y_off * den + (r + 2 * g + b) * y_span;
            Some(quantize_rational(
                num,
                den,
                y_lo,
                y_hi,
                depth_policy,
                gx,
                gy,
            ))
        }
        ForwardMatrix::Ncl {
            exact_rational: Some((kr, kg, kb, k_den)),
            ..
        } => {
            let y_sum = kr * r + kg * g + kb * b;
            let den = k_den * max_in;
            let num = y_off * den + y_sum * y_span;
            Some(quantize_rational(
                num,
                den,
                y_lo,
                y_hi,
                depth_policy,
                gx,
                gy,
            ))
        }
        ForwardMatrix::Ncl {
            exact_rational: None,
            ..
        } => None,
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn encode_ycbcr_exact_int(
    matrix: ForwardMatrix,
    r: i128,
    g: i128,
    b: i128,
    max_in: i128,
    (y_off, y_span, y_lo, y_hi): (i128, i128, u32, u32),
    (cb_mid, cb_span, cb_lo, cb_hi): (i128, i128, u32, u32),
    (cr_mid, cr_span, cr_lo, cr_hi): (i128, i128, u32, u32),
    depth_policy: DepthPolicy,
    gx: usize,
    gy: usize,
) -> Option<(u32, u32, u32)> {
    match matrix {
        ForwardMatrix::Identity => Some((
            quantize_rational(
                y_off * max_in + g * y_span,
                max_in,
                y_lo,
                y_hi,
                depth_policy,
                gx,
                gy,
            ),
            quantize_rational(
                cb_mid * max_in + b * cb_span,
                max_in,
                cb_lo,
                cb_hi,
                depth_policy,
                gx,
                gy,
            ),
            quantize_rational(
                cr_mid * max_in + r * cr_span,
                max_in,
                cr_lo,
                cr_hi,
                depth_policy,
                gx,
                gy,
            ),
        )),
        ForwardMatrix::YCgCo => {
            let den4 = 4 * max_in;
            let den2 = 2 * max_in;
            let y_code = quantize_rational(
                y_off * den4 + (r + 2 * g + b) * y_span,
                den4,
                y_lo,
                y_hi,
                depth_policy,
                gx,
                gy,
            );
            let cg_code = quantize_rational(
                cb_mid * den4 + (2 * g - r - b) * cb_span,
                den4,
                cb_lo,
                cb_hi,
                depth_policy,
                gx,
                gy,
            );
            let co_code = quantize_rational(
                cr_mid * den2 + (r - b) * cr_span,
                den2,
                cr_lo,
                cr_hi,
                depth_policy,
                gx,
                gy,
            );
            Some((y_code, cg_code, co_code))
        }
        ForwardMatrix::Ncl {
            exact_rational: Some((kr, kg, kb, k_den)),
            ..
        } => {
            let y_sum = kr * r + kg * g + kb * b;
            let y_den = k_den * max_in;
            let y_code = quantize_rational(
                y_off * y_den + y_sum * y_span,
                y_den,
                y_lo,
                y_hi,
                depth_policy,
                gx,
                gy,
            );
            let cb_den = 2 * (k_den - kb) * max_in;
            let cb_code = quantize_rational(
                cb_mid * cb_den + (k_den * b - y_sum) * cb_span,
                cb_den,
                cb_lo,
                cb_hi,
                depth_policy,
                gx,
                gy,
            );
            let cr_den = 2 * (k_den - kr) * max_in;
            let cr_code = quantize_rational(
                cr_mid * cr_den + (k_den * r - y_sum) * cr_span,
                cr_den,
                cr_lo,
                cr_hi,
                depth_policy,
                gx,
                gy,
            );
            Some((y_code, cb_code, cr_code))
        }
        ForwardMatrix::Ncl {
            exact_rational: None,
            ..
        } => None,
    }
}

#[inline(always)]
fn apply_forward_matrix_int_to_f64(
    matrix: ForwardMatrix,
    r: i128,
    g: i128,
    b: i128,
    max_in: i128,
) -> (f64, f64, f64) {
    match matrix {
        ForwardMatrix::Identity => {
            let inv = 1.0 / (max_in as f64);
            ((g as f64) * inv, (b as f64) * inv, (r as f64) * inv)
        }
        ForwardMatrix::YCgCo => {
            let inv4 = 1.0 / ((4 * max_in) as f64);
            let inv2 = 1.0 / ((2 * max_in) as f64);
            (
                ((r + 2 * g + b) as f64) * inv4,
                ((2 * g - r - b) as f64) * inv4,
                ((r - b) as f64) * inv2,
            )
        }
        ForwardMatrix::Ncl {
            exact_rational: Some((kr, kg, kb, k_den)),
            ..
        } => {
            let y_sum = kr * r + kg * g + kb * b;
            let yn = (y_sum as f64) / ((k_den * max_in) as f64);
            let cbn = ((k_den * b - y_sum) as f64) / ((2 * (k_den - kb) * max_in) as f64);
            let crn = ((k_den * r - y_sum) as f64) / ((2 * (k_den - kr) * max_in) as f64);
            (yn, cbn.clamp(-0.5, 0.5), crn.clamp(-0.5, 0.5))
        }
        ForwardMatrix::Ncl {
            exact_rational: None,
            ..
        } => {
            let inv = 1.0 / (max_in as f64);
            apply_forward_matrix_f64(matrix, (r as f64) * inv, (g as f64) * inv, (b as f64) * inv)
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn write_rgb_plane_sample_int(
    spec: PlaneFormat,
    params: (i128, i128, u32, u32),
    row: &mut [u8],
    x: usize,
    v: i128,
    max_in: i128,
    depth_policy: DepthPolicy,
    gx: usize,
    gy: usize,
) {
    if spec.sample == SampleType::F32 {
        let off = x * 4;
        let f = ((v as f64) / (max_in as f64)) as f32;
        row[off..off + 4].copy_from_slice(&f.to_ne_bytes());
    } else {
        let (off, span, lo, hi) = params;
        let code = quantize_rational(
            off * max_in + v * span,
            max_in,
            lo,
            hi,
            depth_policy,
            gx,
            gy,
        );
        write_plane_code(spec, row, x, 0, code);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn write_alpha_sample_int(
    spec: PlaneFormat,
    max_code: u32,
    row: &mut [u8],
    x: usize,
    a: i128,
    max_in: i128,
    depth_policy: DepthPolicy,
    gx: usize,
    gy: usize,
) {
    if spec.sample == SampleType::F32 {
        let off = x * 4;
        let f = ((a as f64) / (max_in as f64)) as f32;
        row[off..off + 4].copy_from_slice(&f.to_ne_bytes());
    } else {
        let code = quantize_rational(
            a * i128::from(max_code),
            max_in,
            0,
            max_code,
            depth_policy,
            gx,
            gy,
        );
        write_plane_code(spec, row, x, 0, code);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn write_alpha_sample_f64(
    spec: PlaneFormat,
    max_code: u32,
    row: &mut [u8],
    x: usize,
    a: f64,
    depth_policy: DepthPolicy,
    gx: usize,
    gy: usize,
) {
    if spec.sample == SampleType::F32 {
        let off = x * 4;
        row[off..off + 4].copy_from_slice(&(a as f32).to_ne_bytes());
    } else {
        let code = quantize_code(
            a.clamp(0.0, 1.0) * f64::from(max_code),
            0,
            max_code,
            depth_policy,
            gx,
            gy,
        );
        write_plane_code(spec, row, x, 0, code);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn write_alpha_sample(
    spec: PlaneFormat,
    max_code: u32,
    row: &mut [u8],
    x: usize,
    a: f32,
    depth_policy: DepthPolicy,
    gx: usize,
    gy: usize,
) {
    if spec.sample == SampleType::F32 {
        let off = x * 4;
        row[off..off + 4].copy_from_slice(&a.to_ne_bytes());
    } else {
        let code = quantize_code(
            f64::from(a).clamp(0.0, 1.0) * f64::from(max_code),
            0,
            max_code,
            depth_policy,
            gx,
            gy,
        );
        write_plane_code(spec, row, x, 0, code);
    }
}

#[inline(always)]
fn write_plane_code(spec: PlaneFormat, row: &mut [u8], x: usize, comp: usize, code: u32) {
    let sz = spec.sample.byte_size();
    let elem = spec.bytes_per_element();
    let off = if spec.packing.is_packed_word() {
        x * elem
    } else {
        x * elem + comp * sz
    };
    let word = spec.packing.pack_code(spec.sample, comp, code);
    match spec.sample {
        SampleType::U8 => row[off] = word as u8,
        SampleType::U16 => row[off..off + 2].copy_from_slice(&(word as u16).to_ne_bytes()),
        SampleType::U32 => row[off..off + 4].copy_from_slice(&word.to_ne_bytes()),
        _ => {}
    }
}

#[inline(always)]
fn write_semi_plane_pair(spec: PlaneFormat, row: &mut [u8], x: usize, c0: u32, c1: u32) {
    let elem = spec.bytes_per_element();
    if spec.packing.is_packed_word() {
        let off = x * elem;
        let word =
            spec.packing.pack_code(spec.sample, 0, c0) | spec.packing.pack_code(spec.sample, 1, c1);
        match spec.sample {
            SampleType::U8 => row[off] = word as u8,
            SampleType::U16 => row[off..off + 2].copy_from_slice(&(word as u16).to_ne_bytes()),
            SampleType::U32 => row[off..off + 4].copy_from_slice(&word.to_ne_bytes()),
            _ => {}
        }
    } else {
        write_plane_code(spec, row, x, 0, c0);
        write_plane_code(spec, row, x, 1, c1);
    }
}
