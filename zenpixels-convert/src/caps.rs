//! Compile-time capability bitmasking for dead-code elimination of unused
//! conversion kernels.
//!
//! [`ConverterProfile`] binds kernel function pointers at plan-build time under
//! a `const CAPS: u64` bitmask from this module and returns an ordinary
//! non-generic [`ConvertPlan`] (which prepares into an
//! ordinary [`RowConverter`](crate::RowConverter)).
//!
//! Because `CAPS` is a compile-time constant, LLVM constant-folds disabled arms
//! out of the step binder and dead-code-eliminates every unreferenced kernel
//! — even when default Cargo features or multiple codecs are linked into the
//! same binary. Multiple [`ConverterProfile`] instantiations share one
//! non-generic planner and one row loop, linking only the union of their
//! enabled leaf kernels.

use crate::{CandidateTarget, ConversionSpec, ConvertError, ConvertPlan, SelectedConversion};
use whereat::At;
use zenpixels::PixelEncoding;

// ── Channel depths ───────────────────────────────────────────────────────────

/// 8-bit unsigned integer (`U8`) layout, alpha, and depth kernels.
pub const U8: u64 = 1 << 0;
/// 16-bit unsigned integer (`U16`) layout, alpha, and depth kernels.
pub const U16: u64 = 1 << 1;
/// 16-bit IEEE half-float (`F16`) layout, alpha, and depth kernels.
pub const F16: u64 = 1 << 2;
/// 32-bit IEEE float (`F32`) layout, alpha, and depth kernels.
pub const F32: u64 = 1 << 3;

// ── Layout & alpha families ──────────────────────────────────────────────────

/// RGB / RGBA / BGR / BGRA byte swizzle and alpha add/drop kernels.
pub const LAYOUT: u64 = 1 << 4;
/// Gray / GrayAlpha expansion and luma-collapse kernels.
pub const GRAY: u64 = 1 << 5;
/// Straight ↔ premultiplied alpha conversion kernels.
pub const PREMUL: u64 = 1 << 6;
/// Linear-light alpha matte compositing (`AlphaPolicy::CompositeOnto`).
pub const MATTE: u64 = 1 << 7;
/// Truncating depth-narrowing kernels (`DepthPolicy::Truncate`).
pub const DEPTH_TRUNCATE: u64 = 1 << 8;

// ── Transfer functions ───────────────────────────────────────────────────────

/// Clamped sRGB transfer kernels (U8↔F32, F32↔F32, and sRGB SDR U16→U8 LUT).
pub const TF_SRGB: u64 = 1 << 9;
/// Sign-preserving extended-range sRGB / Gamma 2.2 F32 kernels (`GamutPolicy::PreserveExtended`).
pub const TF_EXTENDED: u64 = 1 << 10;
/// BT.709 transfer kernels (and BT.709 SDR U16→U8 LUT).
pub const TF_BT709: u64 = 1 << 11;
/// Gamma 2.2 (Adobe RGB 1998) transfer kernels.
pub const TF_GAMMA22: u64 = 1 << 12;
/// PQ (SMPTE ST 2084) transfer kernels (also requires Cargo feature `hdr-transfers`).
pub const TF_PQ: u64 = 1 << 13;
/// HLG (ARIB STD-B67 + BT.2100 display rendering) kernels (also requires `hdr-transfers`).
pub const TF_HLG: u64 = 1 << 14;

// ── Gamut & color models ─────────────────────────────────────────────────────

/// Linear F32 3×3 gamut matrix kernels (`GamutMatrixRgbF32` / `GamutMatrixRgbaF32`).
pub const GAMUT_MATRIX: u64 = 1 << 15;
/// Fused sRGB U8 / U16 / F32 gamut-matrix + transfer-LUT kernels (`fast_gamut`).
pub const GAMUT_FUSED: u64 = 1 << 16;
/// Linear RGB(A) ↔ Oklab(A) kernels (also requires Cargo feature `oklab`).
pub const OKLAB: u64 = 1 << 17;
/// OKLch soft chroma compression (`GamutPolicy::SoftCompress`, requires `oklab`).
pub const SOFT_COMPRESS: u64 = 1 << 18;
/// HDR→SDR tone-mapping step (also requires Cargo feature `hdr-experimental`).
pub const TONE_MAP: u64 = 1 << 19;

// ── Perceptual & cylindrical families (also require Cargo feature `perceptual`)

/// Polar ↔ Cartesian coordinate steps (`Oklch ↔ Oklab`, `Lch ↔ Lab`, `Jzczhz ↔ Jzazbz`).
pub const PERCEPTUAL_POLAR: u64 = 1 << 20;
/// Cylindrical `HSL` / `HSV` ↔ RGB steps.
pub const PERCEPTUAL_HSL_HSV: u64 = 1 << 21;
/// CIE 1976 D50 `L*C*h°` ↔ linear RGB steps.
pub const PERCEPTUAL_LCH: u64 = 1 << 22;
/// Safdar et al. `Jzazbz` ↔ linear / PQ RGB steps.
pub const PERCEPTUAL_JZAZBZ: u64 = 1 << 23;
/// Integer `U8` / `U16` encoding/decoding for perceptual spaces.
pub const PERCEPTUAL_INT: u64 = 1 << 24;

// ── Specialization ───────────────────────────────────────────────────────────

/// Specialized `FloatFacts` fast paths (`NO_NAN`, `UNIT_CLEAN`, `FiniteNonNeg`).
///
/// When cleared, float kernels bind their untrusted (full-check) path to avoid
/// emitting multiple SIMD monomorphizations per kernel.
pub const FLOAT_FACTS_SPEC: u64 = 1 << 25;

// ── Profile & CMS resolution ─────────────────────────────────────────────────

/// `ColorContext` / ICC / CICP profile resolution and `PluggableCms` /
/// `ZenCmsLite` dispatch.
///
/// When cleared, [`ConverterProfile`] operates strictly on
/// [`PixelDescriptor`](zenpixels::PixelDescriptor) values (refusing attached
/// ICC / CICP / parametric profiles or CMS dispatch with
/// [`ConvertError::NeedsCms`]), allowing LLVM to dead-strip ICC fingerprint
/// hash tables, `ColorProfileSource::resolve`, and CMS routing.
pub const PROFILES: u64 = 1 << 26;

// ── Curated presets ──────────────────────────────────────────────────────────

/// All conversion capabilities enabled (matches [`ConvertPlan::build`]).
pub const ALL: u64 = !0u64;

/// Minimal U8 layout-only profile: RGB / RGBA / BGR / BGRA swizzle and alpha
/// add/drop on plain [`PixelDescriptor`](zenpixels::PixelDescriptor) values
/// (no ICC/CMS profile resolution, depth/transfer/gamut, or float planners).
pub const LAYOUT_U8: u64 = U8 | LAYOUT;

/// Web SDR image pipeline: U8 + F32, layout, gray, premul, matte, clamped sRGB
/// transfer, fused/matrix gamut conversion, and ICC/CICP profile resolution
/// (no U16, F16, BT.709, Gamma 2.2, PQ/HLG, Oklab, soft-compress, or perceptual).
///
/// Clear [`PROFILES`] (`SDR_WEB & !PROFILES`) for descriptor-only builds that
/// do not need ICC profile identification or CMS plugins.
pub const SDR_WEB: u64 =
    U8 | F32 | LAYOUT | GRAY | PREMUL | MATTE | TF_SRGB | GAMUT_MATRIX | GAMUT_FUSED | PROFILES;

/// Typical JPEG codec profile: U8 + F32, layout, gray, sRGB / BT.709 / Gamma 2.2,
/// gamut conversion, and ICC/CICP profile resolution.
pub const CODEC_JPEG: u64 = U8
    | F32
    | LAYOUT
    | GRAY
    | TF_SRGB
    | TF_BT709
    | TF_GAMMA22
    | GAMUT_MATRIX
    | GAMUT_FUSED
    | PROFILES;

/// Typical PNG codec profile: U8 + U16 + F32, layout, gray, premul, matte,
/// sRGB / BT.709 / PQ / HLG transfers, gamut conversion, and ICC/CICP profile
/// resolution.
pub const CODEC_PNG: u64 = U8
    | U16
    | F32
    | LAYOUT
    | GRAY
    | PREMUL
    | MATTE
    | TF_SRGB
    | TF_BT709
    | TF_PQ
    | TF_HLG
    | GAMUT_MATRIX
    | GAMUT_FUSED
    | PROFILES;

/// Full SDR profile across all channel depths (`U8`/`U16`/`F16`/`F32`) and SDR
/// transfers (including extended-range, `FloatFacts` specialization, and
/// ICC/CICP profile resolution), without HDR transfers, Oklab, tone mapping,
/// or closed-form perceptual spaces.
pub const SDR_ALL: u64 = U8
    | U16
    | F16
    | F32
    | LAYOUT
    | GRAY
    | PREMUL
    | MATTE
    | DEPTH_TRUNCATE
    | TF_SRGB
    | TF_EXTENDED
    | TF_BT709
    | TF_GAMMA22
    | GAMUT_MATRIX
    | GAMUT_FUSED
    | FLOAT_FACTS_SPEC
    | PROFILES;

/// Full HDR + Oklab profile (everything except closed-form perceptual polar /
/// HSL / HSV / Lch / Jzazbz spaces).
pub const HDR: u64 = SDR_ALL | TF_PQ | TF_HLG | OKLAB | SOFT_COMPRESS | TONE_MAP;

/// All closed-form perceptual / cylindrical spaces (`Oklch`, `Lch`, `Jzazbz`,
/// `Jzczhz`, `HSL`, `HSV`, including integer encoding/decoding).
pub const PERCEPTUAL: u64 =
    PERCEPTUAL_POLAR | PERCEPTUAL_HSL_HSV | PERCEPTUAL_LCH | PERCEPTUAL_JZAZBZ | PERCEPTUAL_INT;

/// Whether `CAPS` can only ever execute same-depth, integer-only, same-color
/// layout/gray/premul conversions without profiles, transfers, gamut, or matte.
pub(crate) const fn is_layout_only(caps: u64) -> bool {
    const NON_LAYOUT: u64 = PROFILES
        | MATTE
        | DEPTH_TRUNCATE
        | TF_SRGB
        | TF_EXTENDED
        | TF_BT709
        | TF_GAMMA22
        | TF_PQ
        | TF_HLG
        | GAMUT_MATRIX
        | GAMUT_FUSED
        | OKLAB
        | SOFT_COMPRESS
        | TONE_MAP
        | PERCEPTUAL
        | FLOAT_FACTS_SPEC
        | F16
        | F32;
    const INT_DEPTHS: u64 = U8 | U16;
    (caps & NON_LAYOUT) == 0 && (caps & INT_DEPTHS).count_ones() <= 1
}

/// Non-generic function-pointer table of optional planner sub-trees.
///
/// Built at compile time by [`PlannerHooks::for_caps`]. Passing a `&PlannerHooks`
/// into the shared non-generic planner core avoids duplicating core planner
/// code across multiple [`ConverterProfile`] instantiations in a multi-codec
/// binary while still eliminating symbol references to every disabled
/// sub-planner (ICC/CMS, HLG rendering, HDR pivot, tone mapping, Oklab,
/// soft-compress, perceptual) at const-eval time.
type PlanWithOptsFn = fn(
    zenpixels::PixelDescriptor,
    zenpixels::PixelDescriptor,
    &crate::ConvertOptions,
    &PlannerHooks,
) -> Result<crate::NativePlan, At<ConvertError>>;

#[cfg(feature = "bt2446a")]
type Bt2446AFn = fn(
    zenpixels::PixelDescriptor,
    zenpixels::PixelDescriptor,
    crate::HdrConfig,
) -> Result<crate::NativePlan, At<ConvertError>>;

#[cfg(feature = "oklab")]
type OklabPivotFn = fn(
    zenpixels::PixelDescriptor,
    zenpixels::PixelDescriptor,
) -> Option<zenpixels::PixelDescriptor>;

#[cfg(feature = "perceptual")]
type PerceptualPlanFn = fn(
    zenpixels::PixelDescriptor,
    zenpixels::PixelDescriptor,
    &PlannerHooks,
) -> Result<Option<crate::NativePlan>, At<ConvertError>>;

#[derive(Clone, Copy)]
pub(crate) struct PlannerHooks {
    pub(crate) profiles: Option<crate::plan::ProfileHooks>,
    pub(crate) hdr_pivot: Option<crate::converter::HdrPivotFn>,
    pub(crate) hlg_render: Option<PlanWithOptsFn>,
    #[cfg(feature = "bt2446a")]
    pub(crate) bt2446a: Option<Bt2446AFn>,
    #[cfg(feature = "hdr-experimental")]
    pub(crate) explicit_hdr: Option<crate::plan::ExplicitHdrFn>,
    #[cfg(feature = "oklab")]
    pub(crate) soft_compress: Option<PlanWithOptsFn>,
    #[cfg(feature = "oklab")]
    pub(crate) oklab_pivot: Option<OklabPivotFn>,
    #[cfg(feature = "perceptual")]
    pub(crate) perceptual: Option<PerceptualPlanFn>,
}

impl PlannerHooks {
    pub(crate) const ALL_HOOKS: Self = Self::for_caps::<ALL>();

    pub(crate) const fn for_caps<const CAPS: u64>() -> Self {
        Self::for_caps_inner::<CAPS>(true)
    }

    /// Variant used when both endpoints are proven at the callsite to carry no
    /// `ColorContext` (`PixelEncoding::from(PixelDescriptor)`): keeps CMS
    /// dispatch when `PROFILES` is set, but omits `ColorContext`/ICC/HDR-pivot
    /// source-resolution hooks so descriptor-only callers dead-strip them even
    /// under `caps::ALL`.
    pub(crate) const fn for_caps_descriptors<const CAPS: u64>() -> Self {
        Self::for_caps_inner::<CAPS>(false)
    }

    const fn for_caps_inner<const CAPS: u64>(with_contexts: bool) -> Self {
        Self {
            profiles: if (CAPS & PROFILES) != 0 {
                if with_contexts {
                    Some(crate::plan::PROFILE_HOOKS)
                } else {
                    Some(crate::plan::DESCRIPTOR_CMS_HOOKS)
                }
            } else {
                None
            },
            hdr_pivot: if with_contexts && (CAPS & PROFILES) != 0 && (CAPS & (TF_PQ | TF_HLG)) != 0
            {
                Some(crate::converter::RowRecipe::new_with_sources_hdr_pivot_hooks)
            } else {
                None
            },
            hlg_render: {
                #[cfg(feature = "hdr-transfers")]
                {
                    if (CAPS & TF_HLG) != 0 {
                        Some(crate::NativePlan::plan_hlg_rendering_core)
                    } else {
                        None
                    }
                }
                #[cfg(not(feature = "hdr-transfers"))]
                {
                    None
                }
            },
            #[cfg(feature = "bt2446a")]
            bt2446a: if (CAPS & TONE_MAP) != 0 {
                Some(crate::NativePlan::new_with_hdr_config_unbound)
            } else {
                None
            },
            #[cfg(feature = "hdr-experimental")]
            explicit_hdr: if (CAPS & TONE_MAP) != 0 {
                Some(crate::plan::compile_explicit_hdr)
            } else {
                None
            },
            #[cfg(feature = "oklab")]
            soft_compress: if (CAPS & SOFT_COMPRESS) != 0 {
                Some(crate::NativePlan::new_soft_compressed_core)
            } else {
                None
            },
            #[cfg(feature = "oklab")]
            oklab_pivot: if (CAPS & OKLAB) != 0 {
                Some(crate::convert::oklab_pivot_pub)
            } else {
                None
            },
            #[cfg(feature = "perceptual")]
            perceptual: if (CAPS & PERCEPTUAL) != 0 {
                Some(crate::convert::plan_perceptual_core)
            } else {
                None
            },
        }
    }
}

/// Compile-time capability-masked planner producing an ordinary [`ConvertPlan`].
///
/// Pass a `const CAPS: u64` mask from [`crate::caps`] (or combine flags with `|`).
/// Any planned step whose required capability bits are cleared returns
/// [`ConvertError::NoPath`] and its kernel code is dead-code-eliminated.
///
/// The returned [`ConvertPlan`] and [`RowConverter`](crate::RowConverter) are
/// non-generic, so multiple profiles in one binary link only the union of their
/// enabled leaf kernels.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConverterProfile<const CAPS: u64>;

impl<const CAPS: u64> ConverterProfile<CAPS> {
    /// Compile a conversion route restricted to the capabilities in `CAPS`.
    #[inline(always)]
    #[track_caller]
    pub fn build(
        source: PixelEncoding<'_>,
        target: PixelEncoding<'_>,
        spec: &ConversionSpec<'_>,
    ) -> Result<ConvertPlan, At<ConvertError>> {
        ConvertPlan::build_with_caps::<CAPS>(source, target, spec)
    }

    /// Select an admitted route restricted to the capabilities in `CAPS`.
    ///
    /// Candidates whose plans require capabilities outside `CAPS` are skipped
    /// just like other unsupported routes.
    #[inline(always)]
    #[track_caller]
    pub fn select(
        source: PixelEncoding<'_>,
        targets: &[CandidateTarget<'_>],
        spec: &ConversionSpec<'_>,
    ) -> Result<SelectedConversion, At<ConvertError>> {
        ConvertPlan::select_with_caps::<CAPS>(source, targets, spec)
    }
}
