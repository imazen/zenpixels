//! Compile-time capability bitmasking for dead-code elimination of unused
//! conversion kernels.
//!
//! [`ConverterProfile`] binds kernel function pointers at plan-build time under
//! a `const CAPS: u64` bitmask from this module and returns an ordinary
//! non-generic [`ConvertPlan`](crate::ConvertPlan) (which prepares into an
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

// ── Curated presets ──────────────────────────────────────────────────────────

/// All conversion capabilities enabled (matches [`ConvertPlan::build`]).
pub const ALL: u64 = !0u64;

/// Minimal U8 layout-only profile: RGB / RGBA / BGR / BGRA swizzle and alpha add/drop.
pub const LAYOUT_U8: u64 = U8 | LAYOUT;

/// Web SDR image pipeline: U8 + F32, layout, gray, premul, matte, clamped sRGB
/// transfer, and fused/matrix gamut conversion (no U16, F16, BT.709, Gamma 2.2,
/// PQ/HLG, Oklab, soft-compress, or perceptual).
pub const SDR_WEB: u64 =
    U8 | F32 | LAYOUT | GRAY | PREMUL | MATTE | TF_SRGB | GAMUT_MATRIX | GAMUT_FUSED;

/// Typical JPEG codec profile: U8 + F32, layout, gray, sRGB / BT.709 / Gamma 2.2,
/// and gamut conversion.
pub const CODEC_JPEG: u64 =
    U8 | F32 | LAYOUT | GRAY | TF_SRGB | TF_BT709 | TF_GAMMA22 | GAMUT_MATRIX | GAMUT_FUSED;

/// Typical PNG codec profile: U8 + U16 + F32, layout, gray, premul, matte,
/// sRGB / BT.709 / PQ / HLG transfers, and gamut conversion.
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
    | GAMUT_FUSED;

/// Full SDR profile across all channel depths (`U8`/`U16`/`F16`/`F32`) and SDR
/// transfers (including extended-range and `FloatFacts` specialization), without
/// HDR transfers, Oklab, tone mapping, or closed-form perceptual spaces.
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
    | FLOAT_FACTS_SPEC;

/// Full HDR + Oklab profile (everything except closed-form perceptual polar /
/// HSL / HSV / Lch / Jzazbz spaces).
pub const HDR: u64 = SDR_ALL | TF_PQ | TF_HLG | OKLAB | SOFT_COMPRESS | TONE_MAP;

/// All closed-form perceptual / cylindrical spaces (`Oklch`, `Lch`, `Jzazbz`,
/// `Jzczhz`, `HSL`, `HSV`, including integer encoding/decoding).
pub const PERCEPTUAL: u64 =
    PERCEPTUAL_POLAR | PERCEPTUAL_HSL_HSV | PERCEPTUAL_LCH | PERCEPTUAL_JZAZBZ | PERCEPTUAL_INT;

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
