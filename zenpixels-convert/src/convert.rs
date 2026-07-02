//! Row-level pixel conversion kernels.
//!
//! Each kernel converts one row of `width` pixels from a source format to
//! a destination format. Individual step kernels are pure functions with
//! no allocation. Multi-step plans use [`ConvertScratch`] ping-pong
//! buffers to avoid per-row heap allocation in streaming loops.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::min;

use crate::policy::{AlphaPolicy, ConvertOptions, DepthPolicy, LumaCoefficients};
use crate::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, ConvertError, PixelDescriptor,
    TransferFunction,
};
use whereat::{At, ResultAtExt};

/// HDR→SDR tone-mapping configuration for
/// [`ConvertPlan::new_with_hdr_config`].
///
/// Bundles the source-peak luminance (mandatory — the curve is
/// parameterized by it), target-peak luminance (typically 100 cd/m² for
/// SDR), and the OKLch soft chroma-compression knee (production default `0.96`).
#[cfg(feature = "hdr-experimental")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrConfig {
    /// HDR source peak luminance in cd/m². The BT.2446-A curve treats
    /// `1.0` source-normalized as this peak. Typical values: 1000 (HDR10,
    /// Apple HDR), 4000 (HDR10+ reference), 10000 (PQ peak).
    pub source_peak_nits: f32,
    /// SDR target peak luminance in cd/m². Typical value: 100 (BT.709 /
    /// sRGB diffuse-white peak; BT.1886 reference).
    pub target_peak_nits: f32,
    /// Fraction of max chroma where OKLch soft compression kicks in
    /// (`0.0`–`1.0`). `0.96` (the default) compresses only the outermost
    /// 4 % of the gamut; lower values bring the compression in earlier.
    /// Ignored when the target primaries are BT.2020 (wide-gamut output
    /// mode emits no `SoftCompressOklch` step).
    ///
    /// The default was calibrated against the 76-sample imazen-26
    /// gain-mapped corpus on 2026-06-23: `0.96` is the largest knee
    /// (least desaturation) where the corpus-p90 fraction of pre-clamp
    /// out-of-gamut pixels stays under 0.1 %. Findings:
    /// [`zentone/benchmarks/softcompress_knee_findings_2026-06-23.md`](https://github.com/imazen/zentone).
    pub gamut_knee: f32,
}

#[cfg(feature = "hdr-experimental")]
impl Default for HdrConfig {
    /// Pipeline defaults: `target_peak_nits = 100.0` (SDR reference white)
    /// and `gamut_knee = 0.96` (empirically calibrated against the
    /// imazen-26 gain-mapped HDR corpus, 2026-06-23). `source_peak_nits`
    /// has no default — callers must set it explicitly. Returns
    /// `source_peak_nits = 0.0`; override before passing to
    /// [`ConvertPlan::new_with_hdr_config`].
    fn default() -> Self {
        Self {
            source_peak_nits: 0.0,
            target_peak_nits: 100.0,
            gamut_knee: 0.96,
        }
    }
}

/// True when `(from.transfer, to.transfer)` describes an HDR→SDR
/// transition that requires the BT.2446-A tone map step.
///
/// PQ / HLG source to an SDR-encoded target (`Srgb` / `Bt709` /
/// `Gamma22`). The `Linear` target is **not** considered SDR here —
/// decoding a PQ buffer to relative-linear F32 preserves the data
/// losslessly (the value is just in a different transfer-function
/// representation), and the caller may downstream apply their own
/// tone mapping or carry the wide dynamic range through. HLG↔PQ is
/// handled by the dedicated refusal upstream (different luminance
/// domains, no straight tone-map path).
#[cfg(feature = "hdr-experimental")]
fn is_hdr_to_sdr(from: TransferFunction, to: TransferFunction) -> bool {
    let src_is_hdr = matches!(from, TransferFunction::Pq | TransferFunction::Hlg);
    let dst_is_sdr_encoded = matches!(
        to,
        TransferFunction::Srgb | TransferFunction::Bt709 | TransferFunction::Gamma22
    );
    src_is_hdr && dst_is_sdr_encoded
}

/// Pre-computed conversion plan.
///
/// Stores the chain of steps needed to convert from one format to another.
/// Created once, applied to every row.
#[derive(Clone, Debug)]
pub struct ConvertPlan {
    pub(crate) from: PixelDescriptor,
    pub(crate) to: PixelDescriptor,
    pub(crate) steps: Vec<ConvertStep>,
    /// Relative-linear → PQ-absolute scale = `diffuse_white_nits / 10000`,
    /// applied by the PQ kernels (encode multiplies pre-OETF, decode divides
    /// post-EOTF). `1.0` is the unsignaled default and means "treat linear as
    /// already PQ-absolute (1.0 = 10000 cd/m²)" — i.e. exactly the prior
    /// behavior, so plans built without an anchor are byte-for-byte unchanged.
    /// Set via [`with_pq_anchor`](Self::with_pq_anchor). HLG steps ignore it
    /// (scene-referred — different anchoring, out of scope here).
    pub(crate) pq_anchor_scale: f32,
}

/// Selects which fused TF + matrix + TF kernel a [`ConvertStep::Fused`]
/// dispatches to. Each variant is one (source-TF, source-depth, dest-depth,
/// dest-TF, channel-shape) shape that the planner can peephole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FusedKind {
    /// `SrgbU8 → matrix → SrgbU8`, 3-channel RGB.
    SrgbU8GamutRgb,
    /// `SrgbU8 → matrix → SrgbU8`, 4-channel RGBA (alpha passthrough).
    SrgbU8GamutRgba,
    /// `SrgbU16 → matrix → SrgbU16`, 3-channel RGB via 65K-entry LUTs.
    SrgbU16GamutRgb,
    /// `SrgbU8 → matrix → LinearF32`, 3-channel RGB (cross-depth);
    /// output preserves extended range (no clamp).
    SrgbU8ToLinearF32Rgb,
    /// `LinearF32 → matrix → SrgbU8`, 3-channel RGB (cross-depth);
    /// always clamps since u8 can't represent out-of-gamut values.
    LinearF32ToSrgbU8Rgb,
}

impl FusedKind {
    /// The historical per-variant name kept stable for the
    /// `__trace_ops` recorder + `tests/plan_validation.rs` (which still
    /// asserts on `s.contains("FusedSrgb")`).
    #[inline]
    #[allow(dead_code)] // used only when `__trace_ops` feature is enabled
    pub(crate) const fn variant_name(self) -> &'static str {
        match self {
            Self::SrgbU8GamutRgb => "FusedSrgbU8GamutRgb",
            Self::SrgbU8GamutRgba => "FusedSrgbU8GamutRgba",
            Self::SrgbU16GamutRgb => "FusedSrgbU16GamutRgb",
            Self::SrgbU8ToLinearF32Rgb => "FusedSrgbU8ToLinearF32Rgb",
            Self::LinearF32ToSrgbU8Rgb => "FusedLinearF32ToSrgbU8Rgb",
        }
    }
}

/// A single conversion step.
///
/// Not `Copy` — some variants (e.g., `ExternalTransform`) carry an
/// `Arc`. Peephole rewrites must use `.clone()` or index assignment with
/// pattern matching instead of `*step` dereferences.
#[derive(Clone, Debug)]
pub(crate) enum ConvertStep {
    /// No-op (identity).
    Identity,
    /// BGRA → RGBA byte swizzle (or vice versa).
    SwizzleBgraRgba,
    /// Fused RGB → BGRA: byte swap + add opaque alpha in a single SIMD pass.
    /// Equivalent to `[AddAlpha, SwizzleBgraRgba]` but writes the destination
    /// once instead of twice.
    RgbToBgra,
    /// Add alpha channel (3ch → 4ch), filling with opaque.
    AddAlpha,
    /// Drop alpha channel (4ch → 3ch).
    DropAlpha,
    /// Composite onto solid matte color, then drop alpha (4ch → 3ch).
    ///
    /// Blends in linear light using the source descriptor's transfer
    /// function: pixel RGB is EOTF'd per source TF, alpha-blended against
    /// the pre-linearized matte, then OETF'd back to source TF. Alpha is
    /// treated as linear regardless of color-channel TF. The matte
    /// `(r, g, b)` is always interpreted as sRGB u8 (CSS-style background).
    ///
    /// Implemented uniformly across U8/U16/F32/F16 via per-TF
    /// monomorphization; sRGB integer paths use LUT-based EOTF/OETF.
    MatteComposite { r: u8, g: u8, b: u8 },
    /// Gray → RGB (replicate gray to all 3 channels).
    GrayToRgb,
    /// Gray → RGBA (replicate + opaque alpha).
    GrayToRgba,
    /// RGB → Gray (Y' encoded luma — coefficients applied to encoded bytes).
    ///
    /// The semantic is BT.709/BT.601/etc. Y' (encoded luma), NOT linear-light
    /// luminance L. This is fast, exactly round-trips for `R==G==B` inputs,
    /// and matches what JPEG/video pipelines compute. Linear-light luminance
    /// would require linearize → weight → encode and is not currently
    /// surfaced; document any future linear-L pathway as a separate variant.
    ///
    /// Coefficients are resolved from `ConvertOptions::luma` at plan build
    /// time (`new_explicit`). Default for plans built via `Self::new`
    /// without options is `LumaCoefficients::Bt709`.
    RgbToGray { coefficients: LumaCoefficients },
    /// RGBA → Gray, drop alpha. See [`RgbToGray`](Self::RgbToGray) for
    /// semantic and coefficient resolution.
    RgbaToGray { coefficients: LumaCoefficients },
    /// GrayAlpha → RGBA (replicate gray, keep alpha).
    GrayAlphaToRgba,
    /// GrayAlpha → RGB (replicate gray, drop alpha).
    GrayAlphaToRgb,
    /// Gray → GrayAlpha (add opaque alpha).
    GrayToGrayAlpha,
    /// GrayAlpha → Gray (drop alpha).
    GrayAlphaToGray,
    /// sRGB u8 → linear f32 (per channel, EOTF).
    SrgbU8ToLinearF32,
    /// Linear f32 → sRGB u8 (per channel, OETF).
    LinearF32ToSrgbU8,
    /// Naive u8 → f32 (v / 255.0, no gamma).
    NaiveU8ToF32,
    /// Naive f32 → u8 (clamp * 255 + 0.5, no gamma).
    NaiveF32ToU8,
    /// u16 → u8 ((v * 255 + 32768) >> 16).
    U16ToU8,
    /// u8 → u16 (v * 257).
    U8ToU16,
    /// u16 → f32 (v / 65535.0).
    U16ToF32,
    /// f32 → u16 (clamp * 65535 + 0.5).
    F32ToU16,
    /// f16 → f32 (IEEE 754 half-precision unpack, no TF).
    F16ToF32,
    /// f32 → f16 (round-to-nearest-even, no TF).
    F32ToF16,
    /// PQ (SMPTE ST 2084) u16 → linear f32 (EOTF).
    PqU16ToLinearF32,
    /// Linear f32 → PQ u16 (inverse EOTF / OETF).
    LinearF32ToPqU16,
    /// PQ f32 `[0,1]` → linear f32 (EOTF, no depth change).
    PqF32ToLinearF32,
    /// Linear f32 → PQ f32 `[0,1]` (OETF, no depth change).
    LinearF32ToPqF32,
    /// HLG (ARIB STD-B67) u16 → linear f32 (EOTF).
    HlgU16ToLinearF32,
    /// Linear f32 → HLG u16 (OETF).
    LinearF32ToHlgU16,
    /// HLG f32 `[0,1]` → linear f32 (EOTF, no depth change).
    HlgF32ToLinearF32,
    /// Linear f32 → HLG f32 `[0,1]` (OETF, no depth change).
    LinearF32ToHlgF32,
    /// sRGB f32 `[0,1]` → linear f32 (EOTF, no depth change). Clamps input.
    SrgbF32ToLinearF32,
    /// Linear f32 → sRGB f32 `[0,1]` (OETF, no depth change). Clamps output.
    LinearF32ToSrgbF32,
    /// sRGB f32 → linear f32 (EOTF, sign-preserving extended range).
    /// Emitted when `ConvertOptions::clip_out_of_gamut == false`.
    SrgbF32ToLinearF32Extended,
    /// Linear f32 → sRGB f32 (OETF, sign-preserving extended range).
    LinearF32ToSrgbF32Extended,
    /// BT.709 f32 `[0,1]` → linear f32 (EOTF, no depth change).
    Bt709F32ToLinearF32,
    /// Linear f32 → BT.709 f32 `[0,1]` (OETF, no depth change).
    LinearF32ToBt709F32,
    /// Gamma 2.2 (Adobe RGB 1998) f32 `[0,1]` → linear f32 (EOTF, no depth change).
    /// Uses the Adobe RGB 1998 canonical exponent 563/256 ≈ 2.19921875.
    Gamma22F32ToLinearF32,
    /// Linear f32 → Gamma 2.2 (Adobe RGB 1998) f32 `[0,1]` (OETF, no depth change).
    LinearF32ToGamma22F32,
    /// Straight → Premultiplied alpha.
    StraightToPremul,
    /// Premultiplied → Straight alpha.
    PremulToStraight,
    /// Linear RGB f32 → Oklab f32 (3-channel color model change).
    LinearRgbToOklab,
    /// Oklab f32 → Linear RGB f32 (3-channel color model change).
    OklabToLinearRgb,
    /// Linear RGBA f32 → Oklaba f32 (4-channel, alpha preserved).
    LinearRgbaToOklaba,
    /// Oklaba f32 → Linear RGBA f32 (4-channel, alpha preserved).
    OklabaToLinearRgba,
    /// Apply a 3×3 gamut matrix to linear RGB f32 (3 channels per pixel).
    ///
    /// Used for color primaries conversion (e.g., BT.709 ↔ Display P3 ↔ BT.2020).
    /// Data must be in linear light. The matrix is row-major `[[f32; 3]; 3]`
    /// flattened to `[f32; 9]`.
    GamutMatrixRgbF32([f32; 9]),
    /// Apply a 3×3 gamut matrix to linear RGBA f32 (4 channels, alpha passthrough).
    GamutMatrixRgbaF32([f32; 9]),
    /// Fused TF + 3×3 gamut + TF in one pass. Carries the matrix flattened
    /// row-major to `[f32; 9]`, plus a [`FusedKind`] tag selecting which
    /// linearize → matrix → encode shape to dispatch. Replaces the 3-step
    /// sequence `[<lin>, GamutMatrix*F32, <enc>]` whenever the planner can
    /// peephole it. See [`FusedKind`] for the supported shapes.
    Fused { kind: FusedKind, matrix: [f32; 9] },
    /// BT.2446 Method A HDR→SDR tone-map on linear-light f32 RGB in BT.2020
    /// primaries. The plan builder ensures this step sees BT.2020 linear-light
    /// input via preceding gamut-matrix steps; a following gamut-matrix step
    /// (BT.2020 → target.primaries) handles the destination primaries.
    /// Input is source-normalized (`1.0 = source_peak_nits`); output is
    /// target-normalized (`1.0 = target_peak_nits`). RGB-only — alpha is
    /// handled at descriptor-layout level (the planner pairs this with the
    /// appropriate RGB/RGBA carrier).
    ///
    /// Gated behind `hdr-experimental` at the kernel side.
    #[cfg(feature = "hdr-experimental")]
    ToneMapBt2446A {
        source_peak_nits: f32,
        target_peak_nits: f32,
    },
    /// OKLch soft chroma compression on linear-light f32 RGB in
    /// `primaries`. Pulls residual out-of-gamut excursions back into the
    /// target unit cube using a hue-preserving rational knee curve.
    /// Skipped (no step emitted) when the target is BT.2020 — the
    /// wide-gamut output mode.
    ///
    /// Gated behind `hdr-experimental` at the kernel side.
    #[cfg(feature = "hdr-experimental")]
    SoftCompressOklch {
        primaries: ColorPrimaries,
        knee: f32,
    },
}

impl ConvertStep {
    /// The stable variant name used by the `__trace_ops` recorder. Kept as
    /// a `const fn` on `ConvertStep` (no `strum`/proc-macro dep) so the
    /// recorder has one source of truth — historically there were several
    /// independent 60-arm matches that drifted easily.
    #[inline]
    #[allow(dead_code)] // used only when `__trace_ops` feature is enabled
    pub(crate) const fn variant_name(&self) -> &'static str {
        match self {
            Self::Identity => "Identity",
            Self::SwizzleBgraRgba => "SwizzleBgraRgba",
            Self::RgbToBgra => "RgbToBgra",
            Self::AddAlpha => "AddAlpha",
            Self::DropAlpha => "DropAlpha",
            Self::MatteComposite { .. } => "MatteComposite",
            Self::GrayToRgb => "GrayToRgb",
            Self::GrayToRgba => "GrayToRgba",
            Self::RgbToGray { .. } => "RgbToGray",
            Self::RgbaToGray { .. } => "RgbaToGray",
            Self::GrayAlphaToRgba => "GrayAlphaToRgba",
            Self::GrayAlphaToRgb => "GrayAlphaToRgb",
            Self::GrayToGrayAlpha => "GrayToGrayAlpha",
            Self::GrayAlphaToGray => "GrayAlphaToGray",
            Self::SrgbU8ToLinearF32 => "SrgbU8ToLinearF32",
            Self::LinearF32ToSrgbU8 => "LinearF32ToSrgbU8",
            Self::NaiveU8ToF32 => "NaiveU8ToF32",
            Self::NaiveF32ToU8 => "NaiveF32ToU8",
            Self::U16ToU8 => "U16ToU8",
            Self::U8ToU16 => "U8ToU16",
            Self::U16ToF32 => "U16ToF32",
            Self::F32ToU16 => "F32ToU16",
            Self::F16ToF32 => "F16ToF32",
            Self::F32ToF16 => "F32ToF16",
            Self::PqU16ToLinearF32 => "PqU16ToLinearF32",
            Self::LinearF32ToPqU16 => "LinearF32ToPqU16",
            Self::PqF32ToLinearF32 => "PqF32ToLinearF32",
            Self::LinearF32ToPqF32 => "LinearF32ToPqF32",
            Self::HlgU16ToLinearF32 => "HlgU16ToLinearF32",
            Self::LinearF32ToHlgU16 => "LinearF32ToHlgU16",
            Self::HlgF32ToLinearF32 => "HlgF32ToLinearF32",
            Self::LinearF32ToHlgF32 => "LinearF32ToHlgF32",
            Self::SrgbF32ToLinearF32 => "SrgbF32ToLinearF32",
            Self::LinearF32ToSrgbF32 => "LinearF32ToSrgbF32",
            Self::SrgbF32ToLinearF32Extended => "SrgbF32ToLinearF32Extended",
            Self::LinearF32ToSrgbF32Extended => "LinearF32ToSrgbF32Extended",
            Self::Bt709F32ToLinearF32 => "Bt709F32ToLinearF32",
            Self::LinearF32ToBt709F32 => "LinearF32ToBt709F32",
            Self::Gamma22F32ToLinearF32 => "Gamma22F32ToLinearF32",
            Self::LinearF32ToGamma22F32 => "LinearF32ToGamma22F32",
            Self::StraightToPremul => "StraightToPremul",
            Self::PremulToStraight => "PremulToStraight",
            Self::LinearRgbToOklab => "LinearRgbToOklab",
            Self::OklabToLinearRgb => "OklabToLinearRgb",
            Self::LinearRgbaToOklaba => "LinearRgbaToOklaba",
            Self::OklabaToLinearRgba => "OklabaToLinearRgba",
            Self::GamutMatrixRgbF32(_) => "GamutMatrixRgbF32",
            Self::GamutMatrixRgbaF32(_) => "GamutMatrixRgbaF32",
            Self::Fused { kind, .. } => kind.variant_name(),
            #[cfg(feature = "hdr-experimental")]
            Self::ToneMapBt2446A { .. } => "ToneMapBt2446A",
            #[cfg(feature = "hdr-experimental")]
            Self::SoftCompressOklch { .. } => "SoftCompressOklch",
        }
    }
}

/// Color models that zenpixels-convert's built-in kernels resolve natively.
///
/// Anything outside this set is a device-dependent / CMS-only path —
/// CMYK today, Lab / XYZ / spot inks if/when those land as
/// [`crate::ColorModel`] variants. See [`requires_cms`].
#[inline]
fn native_color_model(m: crate::ColorModel) -> bool {
    // `Gray`, `Rgb` and `Oklab` are the colorimetric spaces the built-in
    // kernels handle end-to-end (gamut matrices, transfer LUTs, fused
    // matluts, polyfit decoders, the OKLab gamut-compression path, …).
    // `YCbCr` is also colorimetric-equivalent to RGB once the matrix has
    // been applied, but no kernel here consumes raw `YCbCr` pixels: every
    // entry point that touches YCbCr first lifts it into RGB via the
    // decoder's own coefficient pair, so the planner never sees a
    // `YCbCr` color model on either side. CMYK is the only non-native
    // model that currently reaches the planner.
    matches!(
        m,
        crate::ColorModel::Gray | crate::ColorModel::Rgb | crate::ColorModel::Oklab
    )
}

/// True when the `(from, to)` pair cannot be handled by the built-in
/// kernels and must dispatch through a color management plugin.
///
/// Today this fires when either side's [`color_model`](PixelDescriptor::color_model)
/// is outside the native set (currently just CMYK; future variants —
/// Lab / XYZ / spot inks — will plug in here). The companion
/// [`ConvertError::NeedsCms`] is what entry points return when this is
/// true and no `cms` was passed.
///
/// Useful to schedulers: a caller doing batch encode/decode can probe
/// `requires_cms` once per source/target pair and decide whether to
/// attach a CMS plugin (e.g. `&MoxCms`) for that batch.
///
/// [`color_model`]: zenpixels::PixelDescriptor::color_model
pub fn requires_cms(from: &PixelDescriptor, to: &PixelDescriptor) -> bool {
    !native_color_model(from.color_model()) || !native_color_model(to.color_model())
}

impl ConvertPlan {
    /// Assemble a plan with the default (no-anchor) PQ scale. The single place
    /// `pq_anchor_scale` is defaulted, so every construction path starts at the
    /// behavior-preserving `1.0`.
    fn build(from: PixelDescriptor, to: PixelDescriptor, steps: Vec<ConvertStep>) -> Self {
        Self {
            from,
            to,
            steps,
            pq_anchor_scale: 1.0,
        }
    }

    /// Anchor this plan's **PQ** steps to an absolute-luminance white point —
    /// the cd/m² that relative-linear `1.0` represents (e.g.
    /// [`DiffuseWhite::BT2408`](zenpixels::hdr::DiffuseWhite::BT2408) = 203).
    ///
    /// The PQ kernels then scale by `nits / 10000` across the relative-linear ↔
    /// PQ-absolute boundary (encode multiplies before the OETF, decode divides
    /// after the EOTF), so a relative-linear buffer maps to PQ at the right
    /// brightness without the caller pre-scaling. A decode+encode pair in one
    /// plan shares the scale and round-trips exactly. The BT.2408 default (203)
    /// reproduces the byte-parity-verified pre-scale that `quantize_to` used to
    /// do by hand. HLG steps are unaffected (scene-referred anchoring differs).
    #[must_use]
    pub(crate) fn with_pq_anchor(mut self, anchor: zenpixels::hdr::DiffuseWhite) -> Self {
        // `diffuse_white_nits` / `PQ_PEAK_NITS` makes the unit explicit: the scale
        // is the fraction of PQ's 10000 cd/m² peak that relative-linear 1.0 sits at.
        let diffuse_white_nits = f64::from(anchor.nits());
        const PQ_PEAK_NITS: f64 = 10_000.0;
        self.pq_anchor_scale = (diffuse_white_nits / PQ_PEAK_NITS) as f32;
        self
    }

    /// Create a conversion plan from `from` to `to`.
    ///
    /// Returns `Err` if no conversion path exists. A
    /// [`SignalRange`](zenpixels::SignalRange) mismatch always refuses
    /// ([`ConvertError::NoPath`]): there are no Narrow↔Full conversion
    /// kernels, and relabeling without rescaling would corrupt pixels — see
    /// the signal-range notes on the [crate docs](crate#step-3-convert).
    ///
    /// CMYK (and any other non-native color model) returns
    /// [`ConvertError::NeedsCms`] so the caller can re-issue via
    /// [`RowConverter::new_explicit_with_cms`](crate::RowConverter::new_explicit_with_cms)
    /// with a [`PluggableCms`](crate::cms::PluggableCms) backend attached.
    /// `ConvertPlan` itself never dispatches through CMS — wire the call
    /// through `RowConverter` for that.
    #[track_caller]
    pub fn new(from: PixelDescriptor, to: PixelDescriptor) -> Result<Self, At<ConvertError>> {
        if requires_cms(&from, &to) {
            return Err(whereat::at!(ConvertError::NeedsCms { from, to }));
        }
        if from == to {
            return Ok(Self::build(from, to, vec![ConvertStep::Identity]));
        }

        // Refuse signal-range crossings: no Narrow↔Full steps exist (no
        // expand/contract kernels), and a plan built from the *other*
        // descriptor differences would emit the source's range-coded values
        // under the target's range label — mislabeled pixels (lifted blacks
        // when narrow data is labeled full, crushed when full data is later
        // expanded as narrow), not a conversion. Until range kernels land,
        // range is preserved verbatim or the conversion fails loudly.
        // Same-range plans (including Narrow→Narrow) are unaffected.
        if from.signal_range != to.signal_range {
            return Err(whereat::at!(ConvertError::NoPath { from, to }));
        }

        // Refuse HLG↔PQ. HLG is scene-referred — these kernels apply only its
        // OETF, with no OOTF and no `Lw`/peak — while PQ is absolute display
        // light (cd/m²). Routing one to the other through the shared "linear"
        // intermediate conflates the two luminance domains by orders of
        // magnitude (scene-normalized `[0,1]` vs absolute [0,10000 cd/m²]): a
        // deterministic but grossly **wrong** result. Until the OOTF +
        // `(diffuse_white, Lw)` threading lands (#45 S2), fail loudly rather than
        // emit wrong pixels — the same posture as the signal-range refusal.
        // HLG↔SDR/linear are *not* refused here: those stay within a normalized
        // domain (endpoint-correct), missing only the mid-tone OOTF gamma.
        if matches!(
            (from.transfer(), to.transfer()),
            (TransferFunction::Hlg, TransferFunction::Pq)
                | (TransferFunction::Pq, TransferFunction::Hlg)
        ) {
            return Err(whereat::at!(ConvertError::NoPath { from, to }));
        }

        // Refuse HDR → SDR through the plain entry point. The plain plan
        // builder has no source-peak luminance to thread into the BT.2446-A
        // curve, and silently routing through `Pq U16 → Linear F32 →
        // Linear F32 (no tone map) → target` would produce semantically
        // wrong pixels (any HDR sample above SDR diffuse-white saturates
        // to 1.0). Force the caller to use the tone-mapped entry point.
        // HLG↔PQ already refused above; this catches HDR→{Linear, Srgb,
        // Bt709, Gamma22}. Under `hdr-experimental` only — without it
        // the variant doesn't exist and the historic pass-through
        // behavior is preserved as a deliberate semi-compatibility
        // shim for legacy non-HDR builds.
        #[cfg(feature = "hdr-experimental")]
        if is_hdr_to_sdr(from.transfer(), to.transfer()) {
            return Err(whereat::at!(ConvertError::HdrSourceRequiresPeak {
                from,
                to,
            }));
        }

        let mut steps = Vec::with_capacity(3);

        // Step 1: Layout conversion (within same depth class).
        // Step 2: Depth conversion.
        // Step 3: Alpha mode conversion.
        //
        // For cross-depth conversions, we convert layout at the source depth
        // first, then change depth. This minimizes the number of channels
        // we need to depth-convert.

        let need_depth_change = from.channel_type() != to.channel_type();
        let need_layout_change = from.layout() != to.layout();
        let need_alpha_change =
            from.alpha() != to.alpha() && from.alpha().is_some() && to.alpha().is_some();

        // Depth/TF steps are needed when depth changes, or when transfer
        // functions differ (at any depth — integer TF changes route through
        // an F32 linear intermediate, handled in `depth_steps`).
        let need_depth_or_tf = need_depth_change || from.transfer() != to.transfer();

        // If we need to change depth AND layout, plan the optimal order.
        if need_layout_change {
            // When going to fewer channels, convert layout first (less depth work).
            // When going to more channels, convert depth first (less layout work).
            //
            // Exception: Oklab layout steps require f32 data. When the source
            // is integer (U8/U16) and the layout change involves Oklab, we must
            // convert depth first regardless of channel count.
            let src_ch = from.layout().channels();
            let dst_ch = to.layout().channels();
            let involves_oklab =
                matches!(from.layout(), ChannelLayout::Oklab | ChannelLayout::OklabA)
                    || matches!(to.layout(), ChannelLayout::Oklab | ChannelLayout::OklabA);

            // Oklab conversion requires known primaries for the RGB→LMS matrix.
            if involves_oklab && from.primaries == ColorPrimaries::Unknown {
                return Err(whereat::at!(ConvertError::NoPath { from, to }));
            }

            let depth_first = need_depth_or_tf
                && (dst_ch > src_ch || (involves_oklab && from.channel_type() != ChannelType::F32));

            if depth_first {
                // Depth first, then layout.
                steps.extend(
                    depth_steps(
                        from.channel_type(),
                        to.channel_type(),
                        from.transfer(),
                        to.transfer(),
                    )
                    .map_err(|e| whereat::at!(e))?,
                );
                steps.extend(layout_steps(from.layout(), to.layout()));
            } else {
                // Layout first, then depth.
                steps.extend(layout_steps(from.layout(), to.layout()));
                if need_depth_or_tf {
                    steps.extend(
                        depth_steps(
                            from.channel_type(),
                            to.channel_type(),
                            from.transfer(),
                            to.transfer(),
                        )
                        .map_err(|e| whereat::at!(e))?,
                    );
                }
            }
        } else if need_depth_or_tf {
            steps.extend(
                depth_steps(
                    from.channel_type(),
                    to.channel_type(),
                    from.transfer(),
                    to.transfer(),
                )
                .map_err(|e| whereat::at!(e))?,
            );
        }

        // Alpha mode conversion (if both have alpha and modes differ).
        if need_alpha_change {
            match (from.alpha(), to.alpha()) {
                (Some(AlphaMode::Straight), Some(AlphaMode::Premultiplied)) => {
                    steps.push(ConvertStep::StraightToPremul);
                }
                (Some(AlphaMode::Premultiplied), Some(AlphaMode::Straight)) => {
                    steps.push(ConvertStep::PremulToStraight);
                }
                _ => {}
            }
        }

        // Primaries conversion: if source and destination have different known
        // primaries, inject a gamut matrix in linear f32 space.
        let need_primaries = from.primaries != to.primaries
            && from.primaries != ColorPrimaries::Unknown
            && to.primaries != ColorPrimaries::Unknown;

        if need_primaries
            && let Some(matrix) = crate::gamut::conversion_matrix(from.primaries, to.primaries)
        {
            // Flatten the 3×3 matrix for storage in the step enum.
            let flat = [
                matrix[0][0],
                matrix[0][1],
                matrix[0][2],
                matrix[1][0],
                matrix[1][1],
                matrix[1][2],
                matrix[2][0],
                matrix[2][1],
                matrix[2][2],
            ];

            // The gamut matrix must be applied in linear f32 space.
            // Check if the existing steps already go through linear f32.
            let mut goes_through_linear = false;
            {
                let mut desc = from;
                for step in &steps {
                    desc = intermediate_desc(desc, step);
                    if desc.channel_type() == ChannelType::F32
                        && desc.transfer() == TransferFunction::Linear
                    {
                        goes_through_linear = true;
                    }
                }
            }

            if goes_through_linear {
                // Insert the gamut matrix right after the first step that
                // produces linear f32. All subsequent steps encode to the
                // target format.
                let mut insert_pos = 0;
                let mut desc = from;
                for (i, step) in steps.iter().enumerate() {
                    desc = intermediate_desc(desc, step);
                    if desc.channel_type() == ChannelType::F32
                        && desc.transfer() == TransferFunction::Linear
                    {
                        insert_pos = i + 1;
                        break;
                    }
                }
                let gamut_step = if desc.layout().has_alpha() {
                    ConvertStep::GamutMatrixRgbaF32(flat)
                } else {
                    ConvertStep::GamutMatrixRgbF32(flat)
                };
                steps.insert(insert_pos, gamut_step);
            } else {
                // No existing linear f32 step — we must add linearize → gamut → delinearize.
                // Determine layout for the gamut step.
                let has_alpha = from.layout().has_alpha() || to.layout().has_alpha();
                // Use the layout at the current point in the plan.
                let mut desc = from;
                for step in &steps {
                    desc = intermediate_desc(desc, step);
                }
                let gamut_step = if desc.layout().has_alpha() || has_alpha {
                    ConvertStep::GamutMatrixRgbaF32(flat)
                } else {
                    ConvertStep::GamutMatrixRgbF32(flat)
                };

                // Insert linearize → gamut → encode-to-target-tf at the end,
                // before any alpha mode steps.
                let linearize = match desc.transfer() {
                    TransferFunction::Srgb => ConvertStep::SrgbF32ToLinearF32,
                    TransferFunction::Bt709 => ConvertStep::Bt709F32ToLinearF32,
                    TransferFunction::Pq => ConvertStep::PqF32ToLinearF32,
                    TransferFunction::Hlg => ConvertStep::HlgF32ToLinearF32,
                    TransferFunction::Gamma22 => ConvertStep::Gamma22F32ToLinearF32,
                    TransferFunction::Linear => ConvertStep::Identity,
                    _ => ConvertStep::SrgbF32ToLinearF32, // assume sRGB for Unknown
                };
                let to_target_tf = match to.transfer() {
                    TransferFunction::Srgb => ConvertStep::LinearF32ToSrgbF32,
                    TransferFunction::Bt709 => ConvertStep::LinearF32ToBt709F32,
                    TransferFunction::Pq => ConvertStep::LinearF32ToPqF32,
                    TransferFunction::Hlg => ConvertStep::LinearF32ToHlgF32,
                    TransferFunction::Gamma22 => ConvertStep::LinearF32ToGamma22F32,
                    TransferFunction::Linear => ConvertStep::Identity,
                    _ => ConvertStep::LinearF32ToSrgbF32, // assume sRGB for Unknown
                };

                // Need to be in f32 first. If current is integer, add naive conversion.
                let mut gamut_steps = Vec::new();
                // Direct fused-step emissions for common cases.
                if desc.channel_type() == ChannelType::U16
                    && desc.transfer() == TransferFunction::Srgb
                    && to.channel_type() == ChannelType::U16
                    && to.transfer() == TransferFunction::Srgb
                    && !desc.layout().has_alpha()
                    && !to.layout().has_alpha()
                {
                    // u16 sRGB → u16 sRGB RGB: single-step matlut.
                    gamut_steps.push(ConvertStep::Fused {
                        kind: FusedKind::SrgbU16GamutRgb,
                        matrix: flat,
                    });
                    steps.extend(gamut_steps);
                    if steps.is_empty() {
                        steps.push(ConvertStep::Identity);
                    }
                    fuse_matlut_patterns(&mut steps);
                    return Ok(Self::build(from, to, steps));
                }
                if desc.channel_type() == ChannelType::U8
                    && matches!(desc.transfer(), TransferFunction::Srgb)
                    && to.channel_type() == ChannelType::F32
                    && to.transfer() == TransferFunction::Linear
                    && !desc.layout().has_alpha()
                    && !to.layout().has_alpha()
                {
                    // u8 sRGB → linear f32 RGB: cross-depth matlut.
                    gamut_steps.push(ConvertStep::Fused {
                        kind: FusedKind::SrgbU8ToLinearF32Rgb,
                        matrix: flat,
                    });
                    steps.extend(gamut_steps);
                    if steps.is_empty() {
                        steps.push(ConvertStep::Identity);
                    }
                    fuse_matlut_patterns(&mut steps);
                    return Ok(Self::build(from, to, steps));
                }
                if desc.channel_type() == ChannelType::F32
                    && desc.transfer() == TransferFunction::Linear
                    && to.channel_type() == ChannelType::U8
                    && to.transfer() == TransferFunction::Srgb
                    && !desc.layout().has_alpha()
                    && !to.layout().has_alpha()
                {
                    // linear f32 → u8 sRGB RGB: cross-depth matlut.
                    gamut_steps.push(ConvertStep::Fused {
                        kind: FusedKind::LinearF32ToSrgbU8Rgb,
                        matrix: flat,
                    });
                    steps.extend(gamut_steps);
                    if steps.is_empty() {
                        steps.push(ConvertStep::Identity);
                    }
                    fuse_matlut_patterns(&mut steps);
                    return Ok(Self::build(from, to, steps));
                }
                if desc.channel_type() != ChannelType::F32 {
                    // Use the fused sRGB u8→linear f32 if applicable.
                    if desc.channel_type() == ChannelType::U8
                        && matches!(
                            desc.transfer(),
                            TransferFunction::Srgb
                                | TransferFunction::Bt709
                                | TransferFunction::Unknown
                        )
                    {
                        gamut_steps.push(ConvertStep::SrgbU8ToLinearF32);
                        // Already linear, skip separate linearize.
                        gamut_steps.push(gamut_step);
                        gamut_steps.push(ConvertStep::LinearF32ToSrgbU8);
                    } else if desc.channel_type() == ChannelType::U16
                        && desc.transfer() == TransferFunction::Pq
                    {
                        gamut_steps.push(ConvertStep::PqU16ToLinearF32);
                        gamut_steps.push(gamut_step);
                        gamut_steps.push(ConvertStep::LinearF32ToPqU16);
                    } else if desc.channel_type() == ChannelType::U16
                        && desc.transfer() == TransferFunction::Hlg
                    {
                        gamut_steps.push(ConvertStep::HlgU16ToLinearF32);
                        gamut_steps.push(gamut_step);
                        gamut_steps.push(ConvertStep::LinearF32ToHlgU16);
                    } else {
                        // Generic: naive to f32, linearize, gamut, delinearize, naive back
                        gamut_steps.push(ConvertStep::NaiveU8ToF32);
                        if !matches!(linearize, ConvertStep::Identity) {
                            gamut_steps.push(linearize);
                        }
                        gamut_steps.push(gamut_step);
                        if !matches!(to_target_tf, ConvertStep::Identity) {
                            gamut_steps.push(to_target_tf);
                        }
                        gamut_steps.push(ConvertStep::NaiveF32ToU8);
                    }
                } else {
                    // Already f32, just linearize → gamut → encode
                    if !matches!(linearize, ConvertStep::Identity) {
                        gamut_steps.push(linearize);
                    }
                    gamut_steps.push(gamut_step);
                    if !matches!(to_target_tf, ConvertStep::Identity) {
                        gamut_steps.push(to_target_tf);
                    }
                }

                steps.extend(gamut_steps);
            }
        }

        if steps.is_empty() {
            // Transfer-only difference or alpha-mode-only: identity path.
            steps.push(ConvertStep::Identity);
        }

        // Peephole fusion: collapse common 3-step patterns into single fused
        // kernels that avoid scratch-buffer round-trips.
        fuse_matlut_patterns(&mut steps);

        Ok(Self::build(from, to, steps))
    }

    /// Create an HDR→SDR conversion plan with the given source-peak
    /// luminance.
    ///
    /// Equivalent to [`ConvertPlan::new_with_hdr_config`] called with
    /// `HdrConfig { source_peak_nits, target_peak_nits: 100.0, gamut_knee: 0.96 }`.
    ///
    /// The plan inserts a [`Bt2446A`](crate::hdr::Bt2446A) tone-map step
    /// (and an OKLch soft-compress step for non-BT.2020 targets) into the
    /// usual transfer / depth / gamut chain. Non-HDR conversions go through
    /// the same path as [`ConvertPlan::new`].
    ///
    /// # Errors
    ///
    /// Same as [`ConvertPlan::new`] for non-HDR conversions. For HDR
    /// sources the rejection list shrinks by one ([`HdrSourceRequiresPeak`]
    /// is no longer raised — peak is supplied).
    ///
    /// [`HdrSourceRequiresPeak`]: ConvertError::HdrSourceRequiresPeak
    ///
    /// # Panics
    ///
    /// Same panics as [`ConvertPlan::new`] (CMYK descriptors).
    #[cfg(feature = "hdr-experimental")]
    #[track_caller]
    pub fn new_with_hdr_peak(
        from: PixelDescriptor,
        to: PixelDescriptor,
        source_peak_nits: f32,
    ) -> Result<Self, At<ConvertError>> {
        Self::new_with_hdr_config(
            from,
            to,
            HdrConfig {
                source_peak_nits,
                ..HdrConfig::default()
            },
        )
    }

    /// Create an HDR→SDR conversion plan with full knob control.
    ///
    /// On HDR→SDR conversions (`Pq` / `Hlg` source → SDR target, OR a
    /// `Linear` source where the caller declares HDR semantics via this
    /// constructor) inserts:
    ///
    /// 1. HDR transfer decode (PQ/HLG → linear) — same kernels as
    ///    [`ConvertPlan::new`]. Skipped when the source is already
    ///    `Linear`.
    /// 2. Source primaries → BT.2020 matrix (skipped when source is BT.2020).
    /// 3. `ToneMapBt2446A` step (the BT.2446 Method A curve operating in
    ///    BT.2020 RGB).
    /// 4. BT.2020 → target primaries matrix (skipped when target is BT.2020).
    /// 5. `SoftCompressOklch` step (skipped when target is BT.2020 —
    ///    wide-gamut output mode preserves chroma).
    /// 6. Linear → target transfer encode + any depth conversion (sRGB u8,
    ///    BT.1886 f32, etc.) — same kernels as [`ConvertPlan::new`].
    ///
    /// For sources that are obviously SDR (`Srgb` / `Bt709` / `Gamma22`)
    /// the `hdr` argument is ignored and this returns the same plan
    /// [`ConvertPlan::new`] would build — no tone-map gets injected into
    /// a path that doesn't need one.
    ///
    // ToneMapBt2446A / SoftCompressOklch are crate-internal `ConvertStep` variants
    // — referenced by name in the prose above; explicit links would point at
    // private items.
    ///
    /// # Errors
    ///
    /// Same as [`ConvertPlan::new`] for non-HDR conversions. For HDR
    /// sources the rejection list shrinks by one ([`HdrSourceRequiresPeak`]
    /// is no longer raised — peak is supplied via `hdr.source_peak_nits`).
    ///
    /// [`HdrSourceRequiresPeak`]: ConvertError::HdrSourceRequiresPeak
    ///
    /// CMYK (and any other non-native color model) returns
    /// [`ConvertError::NeedsCms`] — same posture as
    /// [`ConvertPlan::new`]. HDR tone-mapping is RGB-only; a CMS is the
    /// right tool for CMYK↔RGB even on the HDR construction path.
    #[cfg(feature = "hdr-experimental")]
    #[track_caller]
    pub fn new_with_hdr_config(
        from: PixelDescriptor,
        to: PixelDescriptor,
        hdr: HdrConfig,
    ) -> Result<Self, At<ConvertError>> {
        if requires_cms(&from, &to) {
            return Err(whereat::at!(ConvertError::NeedsCms { from, to }));
        }
        // SDR source paths take the regular plan path — calling the
        // HDR-aware constructor on (e.g.) sRGB → sRGB shouldn't force a
        // tone map. The HDR pipeline runs for PQ/HLG sources AND for
        // `Linear` sources (the caller may have a Linear-tagged
        // gain-map-reconstructed HDR buffer; the constructor's name is
        // the opt-in signal).
        let src_is_sdr_encoded = matches!(
            from.transfer(),
            TransferFunction::Srgb | TransferFunction::Bt709 | TransferFunction::Gamma22
        );
        if src_is_sdr_encoded {
            return Self::new(from, to);
        }
        // Note: do NOT early-return on `from == to`. The HDR-aware
        // constructor is the caller's opt-in signal that the source carries
        // HDR semantics — even when the source and target descriptors are
        // byte-identical (e.g. both `RGBF32_LINEAR`), the tone-map +
        // gamut-compress chain still needs to run. Identity bytes-out
        // would silently skip the HDR work the constructor was called to
        // perform.

        // Same signal-range posture as `new` — Narrow↔Full crossings refuse
        // because no kernels exist yet.
        if from.signal_range != to.signal_range {
            return Err(whereat::at!(ConvertError::NoPath { from, to }));
        }

        // The pipeline: src → linear-F32-in-source-primaries → (source→BT.2020)
        // → ToneMap → (BT.2020→target) → SoftCompress → target-encode.
        // The intermediate descriptor between steps is linear-light F32 in
        // some primaries, with the source's layout (RGB or RGBA) carried
        // through (alpha passthrough at every step). We let the existing
        // depth_steps build the decode side, then append our HDR steps,
        // then let the existing encode chain finish.
        let mut steps: Vec<ConvertStep> = Vec::with_capacity(8);

        // ---- (a) Decode source transfer → F32 linear. Reuse `depth_steps`
        // with intermediate target = (F32, source.layout(), source.alpha(),
        // Linear) — this emits the right PQ/HLG/cross-depth kernels.
        let after_decode = PixelDescriptor::new(
            ChannelType::F32,
            from.layout(),
            from.alpha(),
            TransferFunction::Linear,
        );
        steps.extend(
            depth_steps(
                from.channel_type(),
                ChannelType::F32,
                from.transfer(),
                TransferFunction::Linear,
            )
            .map_err(|e| whereat::at!(e))?,
        );

        // ---- (b) Source primaries → BT.2020 (skip when source IS BT.2020).
        if from.primaries != ColorPrimaries::Bt2020
            && let Some(matrix) =
                crate::gamut::conversion_matrix(from.primaries, ColorPrimaries::Bt2020)
        {
            let flat = [
                matrix[0][0],
                matrix[0][1],
                matrix[0][2],
                matrix[1][0],
                matrix[1][1],
                matrix[1][2],
                matrix[2][0],
                matrix[2][1],
                matrix[2][2],
            ];
            let step = if after_decode.layout().has_alpha() {
                ConvertStep::GamutMatrixRgbaF32(flat)
            } else {
                ConvertStep::GamutMatrixRgbF32(flat)
            };
            steps.push(step);
        }

        // ---- (c) BT.2446 Method A tone map (BT.2020 HDR → BT.2020 SDR).
        steps.push(ConvertStep::ToneMapBt2446A {
            source_peak_nits: hdr.source_peak_nits,
            target_peak_nits: hdr.target_peak_nits,
        });

        // ---- (d) BT.2020 → target primaries (skip when target IS BT.2020).
        if to.primaries != ColorPrimaries::Bt2020
            && to.primaries != ColorPrimaries::Unknown
            && let Some(matrix) =
                crate::gamut::conversion_matrix(ColorPrimaries::Bt2020, to.primaries)
        {
            let flat = [
                matrix[0][0],
                matrix[0][1],
                matrix[0][2],
                matrix[1][0],
                matrix[1][1],
                matrix[1][2],
                matrix[2][0],
                matrix[2][1],
                matrix[2][2],
            ];
            let step = if after_decode.layout().has_alpha() {
                ConvertStep::GamutMatrixRgbaF32(flat)
            } else {
                ConvertStep::GamutMatrixRgbF32(flat)
            };
            steps.push(step);
        }

        // ---- (e) OKLch soft chroma compression (skip when target IS BT.2020;
        // wide-gamut output mode preserves chroma).
        if to.primaries != ColorPrimaries::Bt2020 && to.primaries != ColorPrimaries::Unknown {
            steps.push(ConvertStep::SoftCompressOklch {
                primaries: to.primaries,
                knee: hdr.gamut_knee,
            });
        }

        // ---- (f) Layout conversion (e.g., RGBA→RGB DropAlpha), if any.
        // After the HDR steps we're still in (F32, from.layout(), from.alpha(),
        // Linear) carrying target.primaries (last gamut matrix updated them).
        if from.layout() != to.layout() {
            steps.extend(layout_steps(from.layout(), to.layout()));
        }

        // ---- (g) Linear F32 → target transfer + depth. Re-use depth_steps
        // for the F32 → target.channel_type leg with the encode TF.
        let need_depth_or_tf_encode =
            to.channel_type() != ChannelType::F32 || to.transfer() != TransferFunction::Linear;
        if need_depth_or_tf_encode {
            steps.extend(
                depth_steps(
                    ChannelType::F32,
                    to.channel_type(),
                    TransferFunction::Linear,
                    to.transfer(),
                )
                .map_err(|e| whereat::at!(e))?,
            );
        }

        // ---- (h) Alpha mode (Straight↔Premultiplied).
        if from.alpha() != to.alpha() && from.alpha().is_some() && to.alpha().is_some() {
            match (from.alpha(), to.alpha()) {
                (Some(AlphaMode::Straight), Some(AlphaMode::Premultiplied)) => {
                    steps.push(ConvertStep::StraightToPremul);
                }
                (Some(AlphaMode::Premultiplied), Some(AlphaMode::Straight)) => {
                    steps.push(ConvertStep::PremulToStraight);
                }
                _ => {}
            }
        }

        if steps.is_empty() {
            steps.push(ConvertStep::Identity);
        }

        Ok(Self::build(from, to, steps))
    }

    /// Create a conversion plan with explicit policy enforcement.
    ///
    /// Validates that the planned conversion steps are allowed by the given
    /// policies before creating the plan. Returns an error if a forbidden
    /// operation would be required.
    ///
    /// CMYK (and any other non-native color model) returns
    /// [`ConvertError::NeedsCms`] — same posture as
    /// [`ConvertPlan::new`]. To dispatch CMYK ↔ RGB through a CMS, build
    /// the converter via
    /// [`RowConverter::new_explicit_with_cms`](crate::RowConverter::new_explicit_with_cms)
    /// with a [`PluggableCms`](crate::cms::PluggableCms) plugin attached.
    #[track_caller]
    pub fn new_explicit(
        from: PixelDescriptor,
        to: PixelDescriptor,
        options: &ConvertOptions,
    ) -> Result<Self, At<ConvertError>> {
        if requires_cms(&from, &to) {
            return Err(whereat::at!(ConvertError::NeedsCms { from, to }));
        }
        // Check alpha removal policy.
        let drops_alpha = from.alpha().is_some() && to.alpha().is_none();
        if drops_alpha && options.alpha_policy == AlphaPolicy::Forbid {
            return Err(whereat::at!(ConvertError::AlphaRemovalForbidden));
        }

        // Check depth reduction policy. Compare by precision bits, not byte
        // size — F16 and U16 are both 2 bytes but F16 carries only ~11 bits of
        // precision vs U16's 16, so a U16→F16 hop IS a precision reduction and
        // must be policy-gated.
        let reduces_depth = crate::negotiate::channel_bits(from.channel_type())
            > crate::negotiate::channel_bits(to.channel_type());
        if reduces_depth && options.depth_policy == DepthPolicy::Forbid {
            return Err(whereat::at!(ConvertError::DepthReductionForbidden));
        }

        // Check RGB→Gray requires luma coefficients.
        let src_is_rgb = matches!(
            from.layout(),
            ChannelLayout::Rgb | ChannelLayout::Rgba | ChannelLayout::Bgra
        );
        let dst_is_gray = matches!(to.layout(), ChannelLayout::Gray | ChannelLayout::GrayAlpha);
        if src_is_rgb && dst_is_gray && options.luma.is_none() {
            return Err(whereat::at!(ConvertError::RgbToGray));
        }

        let mut plan = Self::new(from, to).at()?;

        // Replace DropAlpha with MatteComposite when policy is CompositeOnto.
        //
        // The `matte_composite` kernel uses the straight-alpha over operator
        // `fg*a + bg*(1-a)`, linearizing the sRGB matte and pixel RGB
        // per-pixel using the source TF (kernel-side TF dispatch via the
        // `MatteTf` trait). Alpha stays as-is (alpha is always linear,
        // regardless of color-channel TF).
        //
        // One planner-side caveat handled here:
        //
        // **Premultiplied source.** If the source is premultiplied (our
        // library's convention is encoded-space premul, per Canvas 2D),
        // the straight kernel would multiply by `a` twice:
        // `straight*a² + bg*(1-a)`. Fix: insert `PremulToStraight` before
        // `MatteComposite`.
        //
        // We deliberately do NOT wrap with `SrgbF32ToLinearF32` /
        // `LinearF32ToSrgbF32` to handle non-linear data: those steps
        // linearize alpha too, which breaks the blend math.
        if drops_alpha && let AlphaPolicy::CompositeOnto { r, g, b } = options.alpha_policy {
            let src_is_premul = from.alpha() == Some(AlphaMode::Premultiplied);
            let mut idx = 0;
            while idx < plan.steps.len() {
                if matches!(plan.steps[idx], ConvertStep::DropAlpha) {
                    plan.steps[idx] = ConvertStep::MatteComposite { r, g, b };
                    if src_is_premul {
                        plan.steps.insert(idx, ConvertStep::PremulToStraight);
                        idx += 1;
                    }
                }
                idx += 1;
            }
        }

        // When the caller opts out of clipping, swap pure-f32 sRGB transfer
        // steps for their sign-preserving extended-range counterparts.
        // Fused u8/u16 matlut steps are unaffected (integer I/O can't
        // represent extended range anyway).
        if !options.clip_out_of_gamut {
            for step in &mut plan.steps {
                match step {
                    ConvertStep::SrgbF32ToLinearF32 => {
                        *step = ConvertStep::SrgbF32ToLinearF32Extended;
                    }
                    ConvertStep::LinearF32ToSrgbF32 => {
                        *step = ConvertStep::LinearF32ToSrgbF32Extended;
                    }
                    _ => {}
                }
            }
        }

        // Resolve luma coefficients on RgbToGray / RgbaToGray steps. The
        // None case was rejected above (line 636), so unwrap is safe here.
        // `layout_steps` constructs these variants with a Bt709 placeholder
        // because it has no access to options; we replace with the user's
        // explicit choice (or the permissive default of Bt709) here.
        let user_luma = options.luma.unwrap_or(LumaCoefficients::Bt709);
        for step in &mut plan.steps {
            match step {
                ConvertStep::RgbToGray { coefficients }
                | ConvertStep::RgbaToGray { coefficients } => {
                    *coefficients = user_luma;
                }
                _ => {}
            }
        }

        Ok(plan)
    }

    /// Create a shell plan that records from/to but has no conversion steps.
    ///
    /// Used when an external CMS transform handles the conversion — the
    /// plan exists only for `from()`/`to()` metadata; the actual row
    /// work is driven by the external transform stored on `RowConverter`.
    pub(crate) fn identity(from: PixelDescriptor, to: PixelDescriptor) -> Self {
        Self::build(from, to, vec![ConvertStep::Identity])
    }

    /// Compose two plans into one: apply `self` then `other`.
    ///
    /// The composed plan executes both conversions in a single `convert_row`
    /// call, using one intermediate buffer instead of two. Adjacent inverse
    /// steps are cancelled (e.g., `SrgbU8ToLinearF32` + `LinearF32ToSrgbU8`
    /// → identity).
    ///
    /// Returns `None` if `self.to` != `other.from` (incompatible plans).
    pub fn compose(&self, other: &Self) -> Option<Self> {
        if self.to != other.from {
            return None;
        }

        let mut steps = self.steps.clone();

        // Append other's steps, skipping its Identity if present.
        for step in &other.steps {
            if matches!(step, ConvertStep::Identity) {
                continue;
            }
            steps.push(step.clone());
        }

        // Peephole: cancel adjacent inverse pairs.
        let mut changed = true;
        while changed {
            changed = false;
            let mut i = 0;
            while i + 1 < steps.len() {
                if are_inverse(&steps[i], &steps[i + 1]) {
                    steps.remove(i + 1);
                    steps.remove(i);
                    changed = true;
                    // Don't advance — check the new adjacent pair.
                } else {
                    i += 1;
                }
            }
        }

        // If everything cancelled, produce identity.
        if steps.is_empty() {
            steps.push(ConvertStep::Identity);
        }

        // Remove leading/trailing Identity if there are real steps.
        if steps.len() > 1 {
            steps.retain(|s| !matches!(s, ConvertStep::Identity));
            if steps.is_empty() {
                steps.push(ConvertStep::Identity);
            }
        }

        // Composition runs at plan-build time, before any anchor is attached
        // (`with_pq_anchor` is applied to the finished plan), so both inputs
        // carry the default scale; the merged plan does too.
        Some(Self::build(self.from, other.to, steps))
    }

    /// True if conversion is a no-op.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.steps.len() == 1 && matches!(self.steps[0], ConvertStep::Identity)
    }

    /// Maximum bytes-per-pixel across all intermediate formats in the plan.
    ///
    /// Used to pre-allocate scratch buffers for streaming conversion.
    pub(crate) fn max_intermediate_bpp(&self) -> usize {
        let mut desc = self.from;
        let mut max_bpp = desc.bytes_per_pixel();
        for step in &self.steps {
            desc = intermediate_desc(desc, step);
            max_bpp = max_bpp.max(desc.bytes_per_pixel());
        }
        max_bpp
    }

    /// Crate-internal view of the planned step list — exposed for the
    /// estimate-API code under `crate::estimate`. NOT public:
    /// `ConvertStep` itself is `pub(crate)`.
    pub(crate) fn steps(&self) -> &[ConvertStep] {
        &self.steps
    }

    /// Source descriptor.
    pub fn from(&self) -> PixelDescriptor {
        self.from
    }

    /// Target descriptor.
    pub fn to(&self) -> PixelDescriptor {
        self.to
    }

    /// Estimate resources for executing this plan on `image` under the
    /// given [`ComputeEnvironment`](crate::estimate::ComputeEnvironment).
    /// Returns a [`ResourceEstimate`](crate::estimate::ResourceEstimate)
    /// whose type shape matches `zencodec::estimate::ResourceEstimate` so
    /// codec-side encode/decode estimates can be wired through a multi-
    /// stage pipeline at the codec boundary with a trivial conversion.
    ///
    /// Calibrated from `benches/t1_layout`, `t2_depth`, `t3_tf_fused`,
    /// `t4_tf_f32`, `t5_alpha`, `t6_oklab`, `t7_gamut` steady-state
    /// throughput; best-effort, ±30 % on the reference machine
    /// (Ryzen 9 7950X, AVX2). Real wall time varies with contention,
    /// frequency scaling, and CPU model. Identity at 0×0 returns a
    /// zero-cost estimate.
    ///
    /// `peak_memory_bytes_est` is the destination buffer plus row-sized
    /// ping-pong scratch (multi-step plans). It does NOT include the
    /// caller's persistent state. `intermediate_buffer_count` reports the
    /// number of full-image intermediate buffers held simultaneously
    /// (0 for identity / single-step plans; 2 for multi-step plans using
    /// ping-pong scratch) so schedulers can distinguish 1-giant-buffer plans
    /// from N-medium-buffer plans for paging-pressure decisions.
    /// `wall_ms` is divided down by `compute.cores()` via the plan's
    /// internal threading-bottleneck model (see the [`estimate`] module
    /// docs): any SERIAL step forces the whole plan SERIAL; otherwise the
    /// smallest per-step knee — `rows / 64` clamped to `[1, 16]` — caps
    /// the useful thread count.
    ///
    /// [`estimate`]: crate::estimate
    ///
    /// `compute.simd_tier()` applies a coarse per-tier wall-time
    /// multiplier on top of the AVX2 baseline (see the `estimate`
    /// module docs for the per-tier ratios; TODO per-tier calibration).
    ///
    /// Cheap to call — walks the plan's steps once and does no
    /// allocation. Safe to call repeatedly per-frame in throttled
    /// pipelines.
    ///
    /// For a quick estimate using [`ComputeEnvironment::new()`](crate::estimate::ComputeEnvironment::new)
    /// defaults on a `width × height` image, see [`estimate`](Self::estimate).
    ///
    /// # Example
    ///
    /// ```rust
    /// use zenpixels::PixelDescriptor;
    /// use zenpixels_convert::{ComputeEnvironment, ConvertPlan, ImageCharacteristics};
    ///
    /// let plan = ConvertPlan::new(
    ///     PixelDescriptor::RGB8_SRGB,
    ///     PixelDescriptor::RGBA8_SRGB,
    /// ).unwrap();
    /// let image = ImageCharacteristics::new(1920, 1080, PixelDescriptor::RGB8_SRGB);
    /// let compute = ComputeEnvironment::new().with_cores(8);
    /// let est = plan.estimate_in(&image, &compute);
    /// assert!(est.peak_memory_bytes_est().unwrap_or(0) > 0);
    /// // wall_ms is `Some(_)` once the plan has measurable work (it can
    /// // round to 0 ms for trivial plans, but the field is populated).
    /// assert!(est.wall_ms().is_some());
    /// ```
    #[must_use]
    pub fn estimate_in(
        &self,
        image: &crate::estimate::ImageCharacteristics,
        compute: &crate::estimate::ComputeEnvironment,
    ) -> crate::estimate::ResourceEstimate {
        crate::estimate::estimate_plan(self, image, compute)
    }

    /// Shortcut: estimate with [`ComputeEnvironment::new()`](crate::estimate::ComputeEnvironment::new)
    /// defaults (single core, unknown RAM, unspecified SIMD tier) on a
    /// `width × height` image. Builds the
    /// [`ImageCharacteristics`](crate::estimate::ImageCharacteristics) from
    /// the plan's `from()` descriptor and calls [`estimate_in`](Self::estimate_in).
    /// Use [`estimate_in`](Self::estimate_in) directly when the caller has
    /// a populated compute environment (e.g.
    /// `available_parallelism() + archmage tier`).
    ///
    /// # Example
    ///
    /// ```rust
    /// use zenpixels::PixelDescriptor;
    /// use zenpixels_convert::ConvertPlan;
    ///
    /// let plan = ConvertPlan::new(
    ///     PixelDescriptor::RGB8_SRGB,
    ///     PixelDescriptor::RGBA8_SRGB,
    /// ).unwrap();
    /// let est = plan.estimate(1920, 1080);
    /// assert!(est.peak_memory_bytes_est().unwrap_or(0) > 0);
    /// assert!(est.wall_ms().is_some());
    /// ```
    #[must_use]
    pub fn estimate(&self, width: u32, height: u32) -> crate::estimate::ResourceEstimate {
        let image = crate::estimate::ImageCharacteristics::new(width, height, self.from());
        let compute = crate::estimate::ComputeEnvironment::new();
        self.estimate_in(&image, &compute)
    }
}

/// Bridge for the [`crate::estimate`] module: mirror of
/// [`intermediate_desc`] without making that function public.
pub(crate) fn intermediate_desc_for_estimate(
    current: PixelDescriptor,
    step: &ConvertStep,
) -> PixelDescriptor {
    intermediate_desc(current, step)
}

/// Determine the layout conversion step(s).
///
/// Some layout conversions require two steps (e.g., BGRA -> RGB needs
/// swizzle + drop alpha). Returns up to 2 steps.
fn layout_steps(from: ChannelLayout, to: ChannelLayout) -> Vec<ConvertStep> {
    if from == to {
        return Vec::new();
    }
    match (from, to) {
        (ChannelLayout::Bgra, ChannelLayout::Rgba) | (ChannelLayout::Rgba, ChannelLayout::Bgra) => {
            vec![ConvertStep::SwizzleBgraRgba]
        }
        (ChannelLayout::Rgb, ChannelLayout::Rgba) => vec![ConvertStep::AddAlpha],
        (ChannelLayout::Rgb, ChannelLayout::Bgra) => {
            // Single fused SIMD pass (garb::bytes::rgb_to_bgra). For non-u8
            // channel types `apply_step_u8` falls back to AddAlpha+Swizzle.
            vec![ConvertStep::RgbToBgra]
        }
        (ChannelLayout::Rgba, ChannelLayout::Rgb) => vec![ConvertStep::DropAlpha],
        (ChannelLayout::Bgra, ChannelLayout::Rgb) => {
            // BGRA -> RGBA -> RGB: swizzle then drop alpha.
            vec![ConvertStep::SwizzleBgraRgba, ConvertStep::DropAlpha]
        }
        (ChannelLayout::Gray, ChannelLayout::Rgb) => vec![ConvertStep::GrayToRgb],
        (ChannelLayout::Gray, ChannelLayout::Rgba) => vec![ConvertStep::GrayToRgba],
        (ChannelLayout::Gray, ChannelLayout::Bgra) => {
            // Gray -> RGBA -> BGRA: expand then swizzle.
            vec![ConvertStep::GrayToRgba, ConvertStep::SwizzleBgraRgba]
        }
        (ChannelLayout::Rgb, ChannelLayout::Gray) => vec![ConvertStep::RgbToGray {
            coefficients: LumaCoefficients::Bt709,
        }],
        (ChannelLayout::Rgba, ChannelLayout::Gray) => vec![ConvertStep::RgbaToGray {
            coefficients: LumaCoefficients::Bt709,
        }],
        (ChannelLayout::Bgra, ChannelLayout::Gray) => {
            // BGRA -> RGBA -> Gray: swizzle then to gray.
            vec![
                ConvertStep::SwizzleBgraRgba,
                ConvertStep::RgbaToGray {
                    coefficients: LumaCoefficients::Bt709,
                },
            ]
        }
        (ChannelLayout::GrayAlpha, ChannelLayout::Rgba) => vec![ConvertStep::GrayAlphaToRgba],
        (ChannelLayout::GrayAlpha, ChannelLayout::Bgra) => {
            // GrayAlpha -> RGBA -> BGRA: expand then swizzle.
            vec![ConvertStep::GrayAlphaToRgba, ConvertStep::SwizzleBgraRgba]
        }
        (ChannelLayout::GrayAlpha, ChannelLayout::Rgb) => vec![ConvertStep::GrayAlphaToRgb],
        (ChannelLayout::Gray, ChannelLayout::GrayAlpha) => vec![ConvertStep::GrayToGrayAlpha],
        (ChannelLayout::GrayAlpha, ChannelLayout::Gray) => vec![ConvertStep::GrayAlphaToGray],

        // Oklab ↔ RGB conversions (via linear RGB).
        (ChannelLayout::Rgb, ChannelLayout::Oklab) => vec![ConvertStep::LinearRgbToOklab],
        (ChannelLayout::Oklab, ChannelLayout::Rgb) => vec![ConvertStep::OklabToLinearRgb],
        (ChannelLayout::Rgba, ChannelLayout::OklabA) => vec![ConvertStep::LinearRgbaToOklaba],
        (ChannelLayout::OklabA, ChannelLayout::Rgba) => vec![ConvertStep::OklabaToLinearRgba],

        // Oklab ↔ RGB with alpha add/drop.
        (ChannelLayout::Rgb, ChannelLayout::OklabA) => {
            vec![ConvertStep::AddAlpha, ConvertStep::LinearRgbaToOklaba]
        }
        (ChannelLayout::OklabA, ChannelLayout::Rgb) => {
            vec![ConvertStep::OklabaToLinearRgba, ConvertStep::DropAlpha]
        }
        (ChannelLayout::Oklab, ChannelLayout::Rgba) => {
            vec![ConvertStep::OklabToLinearRgb, ConvertStep::AddAlpha]
        }
        (ChannelLayout::Rgba, ChannelLayout::Oklab) => {
            vec![ConvertStep::DropAlpha, ConvertStep::LinearRgbToOklab]
        }

        // Oklab ↔ BGRA (swizzle to/from RGBA, then Oklab).
        (ChannelLayout::Bgra, ChannelLayout::OklabA) => {
            vec![
                ConvertStep::SwizzleBgraRgba,
                ConvertStep::LinearRgbaToOklaba,
            ]
        }
        (ChannelLayout::OklabA, ChannelLayout::Bgra) => {
            vec![
                ConvertStep::OklabaToLinearRgba,
                ConvertStep::SwizzleBgraRgba,
            ]
        }
        (ChannelLayout::Bgra, ChannelLayout::Oklab) => {
            vec![
                ConvertStep::SwizzleBgraRgba,
                ConvertStep::DropAlpha,
                ConvertStep::LinearRgbToOklab,
            ]
        }
        (ChannelLayout::Oklab, ChannelLayout::Bgra) => {
            vec![
                ConvertStep::OklabToLinearRgb,
                ConvertStep::AddAlpha,
                ConvertStep::SwizzleBgraRgba,
            ]
        }

        // Gray ↔ Oklab (expand gray to RGB first).
        (ChannelLayout::Gray, ChannelLayout::Oklab) => {
            vec![ConvertStep::GrayToRgb, ConvertStep::LinearRgbToOklab]
        }
        (ChannelLayout::Oklab, ChannelLayout::Gray) => {
            vec![
                ConvertStep::OklabToLinearRgb,
                ConvertStep::RgbToGray {
                    coefficients: LumaCoefficients::Bt709,
                },
            ]
        }
        (ChannelLayout::Gray, ChannelLayout::OklabA) => {
            vec![ConvertStep::GrayToRgba, ConvertStep::LinearRgbaToOklaba]
        }
        (ChannelLayout::OklabA, ChannelLayout::Gray) => {
            vec![
                ConvertStep::OklabaToLinearRgba,
                ConvertStep::RgbaToGray {
                    coefficients: LumaCoefficients::Bt709,
                },
            ]
        }
        (ChannelLayout::GrayAlpha, ChannelLayout::OklabA) => {
            vec![
                ConvertStep::GrayAlphaToRgba,
                ConvertStep::LinearRgbaToOklaba,
            ]
        }
        (ChannelLayout::OklabA, ChannelLayout::GrayAlpha) => {
            // Drop alpha from OklabA→Oklab, convert to RGB, then to GrayAlpha.
            // Alpha is lost; this is inherently lossy.
            vec![
                ConvertStep::OklabaToLinearRgba,
                ConvertStep::RgbaToGray {
                    coefficients: LumaCoefficients::Bt709,
                },
                ConvertStep::GrayToGrayAlpha,
            ]
        }
        (ChannelLayout::GrayAlpha, ChannelLayout::Oklab) => {
            vec![ConvertStep::GrayAlphaToRgb, ConvertStep::LinearRgbToOklab]
        }
        (ChannelLayout::Oklab, ChannelLayout::GrayAlpha) => {
            vec![
                ConvertStep::OklabToLinearRgb,
                ConvertStep::RgbToGray {
                    coefficients: LumaCoefficients::Bt709,
                },
                ConvertStep::GrayToGrayAlpha,
            ]
        }

        // Oklab ↔ alpha variants.
        (ChannelLayout::Oklab, ChannelLayout::OklabA) => vec![ConvertStep::AddAlpha],
        (ChannelLayout::OklabA, ChannelLayout::Oklab) => vec![ConvertStep::DropAlpha],

        _ => Vec::new(), // Unsupported layout conversion.
    }
}

/// F32→F32 linearize step for a transfer function, or `None` if the TF is
/// already linear (or Unknown — caller decides how to handle Unknown).
fn f32_linearize_step(tf: TransferFunction) -> Option<ConvertStep> {
    match tf {
        TransferFunction::Linear => None,
        TransferFunction::Srgb => Some(ConvertStep::SrgbF32ToLinearF32),
        TransferFunction::Bt709 => Some(ConvertStep::Bt709F32ToLinearF32),
        TransferFunction::Pq => Some(ConvertStep::PqF32ToLinearF32),
        TransferFunction::Hlg => Some(ConvertStep::HlgF32ToLinearF32),
        TransferFunction::Gamma22 => Some(ConvertStep::Gamma22F32ToLinearF32),
        TransferFunction::Unknown => None,
        _ => None,
    }
}

/// F32→F32 OETF step for a transfer function, or `None` if the TF is linear
/// (or Unknown).
fn f32_encode_step(tf: TransferFunction) -> Option<ConvertStep> {
    match tf {
        TransferFunction::Linear => None,
        TransferFunction::Srgb => Some(ConvertStep::LinearF32ToSrgbF32),
        TransferFunction::Bt709 => Some(ConvertStep::LinearF32ToBt709F32),
        TransferFunction::Pq => Some(ConvertStep::LinearF32ToPqF32),
        TransferFunction::Hlg => Some(ConvertStep::LinearF32ToHlgF32),
        TransferFunction::Gamma22 => Some(ConvertStep::LinearF32ToGamma22F32),
        TransferFunction::Unknown => None,
        _ => None,
    }
}

/// F32→F32 TF-change steps: linearize (if not already linear) then encode
/// (if target is not linear).
///
/// Returns empty when `from == to`, or when either side is `Unknown` — when
/// one side's TF is unknown we can't mechanically compute a correct
/// conversion, so we preserve bytes as-is. Addressing the Unknown ambiguity
/// via explicit opt-in API is tracked as issue #19 `[C]`/`[D]` (deprecate-and-add).
fn f32_tf_pair_steps(from: TransferFunction, to: TransferFunction) -> Vec<ConvertStep> {
    if from == to || from == TransferFunction::Unknown || to == TransferFunction::Unknown {
        return Vec::new();
    }
    let mut steps = Vec::with_capacity(2);
    if let Some(s) = f32_linearize_step(from) {
        steps.push(s);
    }
    if let Some(s) = f32_encode_step(to) {
        steps.push(s);
    }
    steps
}

/// Depth conversion step into F32 for any non-F32 channel type (U8, U16, F16).
/// Panics for F32 (caller must check); CMYK is rejected upstream by
/// [`requires_cms`] before any plan steps are picked.
fn to_f32_step(ct: ChannelType) -> ConvertStep {
    match ct {
        ChannelType::U8 => ConvertStep::NaiveU8ToF32,
        ChannelType::U16 => ConvertStep::U16ToF32,
        ChannelType::F16 => ConvertStep::F16ToF32,
        _ => unreachable!("to_f32_step called with F32 or unsupported channel type"),
    }
}

/// F32→depth step for any non-F32 channel type.
fn f32_to_depth_step(ct: ChannelType) -> ConvertStep {
    match ct {
        ChannelType::U8 => ConvertStep::NaiveF32ToU8,
        ChannelType::U16 => ConvertStep::F32ToU16,
        ChannelType::F16 => ConvertStep::F32ToF16,
        _ => unreachable!("f32_to_depth_step called with F32 or unsupported channel type"),
    }
}

/// Determine the depth conversion step(s), considering transfer functions.
///
/// Returns one or more steps. Multi-step conversions route through an F32
/// linear intermediate (e.g. PQ U16 → sRGB U8 goes PQ U16 → Linear F32 →
/// sRGB U8), and same-depth integer TF changes route through an F32 linear
/// intermediate too: passing integer bytes through unchanged under a new
/// TF label produces wrong pixels.
fn depth_steps(
    from: ChannelType,
    to: ChannelType,
    from_tf: TransferFunction,
    to_tf: TransferFunction,
) -> Result<Vec<ConvertStep>, ConvertError> {
    if from == to && from_tf == to_tf {
        return Ok(Vec::new());
    }

    // Same depth, F32: apply EOTF/OETF in place.
    if from == to && from == ChannelType::F32 {
        return Ok(f32_tf_pair_steps(from_tf, to_tf));
    }

    // Same depth, non-F32 (U8/U16/F16): TF change requires re-encoding. Route
    // through F32 linear intermediate — passing bytes through labeled as a
    // different TF produces wrong pixels.
    //
    // Exception: if either TF is Unknown, we don't know the correct conversion.
    // Preserve bytes exactly (no F32 round-trip — that would introduce U8/U16
    // rounding error for no semantic benefit). Addressed properly by issue
    // #19 [C]/[D] via opt-in deprecate-and-add.
    if from == to && from != ChannelType::F32 {
        if from_tf == TransferFunction::Unknown || to_tf == TransferFunction::Unknown {
            return Ok(Vec::new());
        }
        let mut steps = Vec::with_capacity(4);
        steps.push(to_f32_step(from));
        steps.extend(f32_tf_pair_steps(from_tf, to_tf));
        steps.push(f32_to_depth_step(to));
        return Ok(steps);
    }

    match (from, to) {
        (ChannelType::U8, ChannelType::F32) => {
            // Fused sRGB EOTF kernel — sRGB only. BT.709 uses a different EOTF
            // (~17% linear-light error at mid-gray if we routed it through the
            // sRGB kernel) and must compose through the F32 BT.709 EOTF step.
            if from_tf == TransferFunction::Srgb && to_tf == TransferFunction::Linear {
                Ok(vec![ConvertStep::SrgbU8ToLinearF32])
            } else if from_tf == to_tf {
                Ok(vec![ConvertStep::NaiveU8ToF32])
            } else {
                // Cross-depth + cross-TF: linearize/encode after the U8→F32 scale.
                // Previously dropped the TF math and returned bytes labeled with
                // the target TF — silent wrong pixels for any TF pair other than
                // {Srgb,Bt709}→Linear.
                let mut steps = Vec::with_capacity(3);
                steps.push(ConvertStep::NaiveU8ToF32);
                steps.extend(f32_tf_pair_steps(from_tf, to_tf));
                Ok(steps)
            }
        }
        (ChannelType::F32, ChannelType::U8) => {
            // Fused sRGB OETF kernel — sRGB only (same reason as above).
            if from_tf == TransferFunction::Linear && to_tf == TransferFunction::Srgb {
                Ok(vec![ConvertStep::LinearF32ToSrgbU8])
            } else if from_tf == to_tf {
                Ok(vec![ConvertStep::NaiveF32ToU8])
            } else {
                // Linearize/encode in F32 first, then compress to U8.
                let mut steps = f32_tf_pair_steps(from_tf, to_tf);
                steps.push(ConvertStep::NaiveF32ToU8);
                Ok(steps)
            }
        }
        (ChannelType::U16, ChannelType::F32) => {
            // PQ/HLG U16 → Linear F32: apply EOTF during conversion.
            match (from_tf, to_tf) {
                (TransferFunction::Pq, TransferFunction::Linear) => {
                    Ok(vec![ConvertStep::PqU16ToLinearF32])
                }
                (TransferFunction::Hlg, TransferFunction::Linear) => {
                    Ok(vec![ConvertStep::HlgU16ToLinearF32])
                }
                (a, b) if a == b => Ok(vec![ConvertStep::U16ToF32]),
                _ => {
                    let mut steps = Vec::with_capacity(3);
                    steps.push(ConvertStep::U16ToF32);
                    steps.extend(f32_tf_pair_steps(from_tf, to_tf));
                    Ok(steps)
                }
            }
        }
        (ChannelType::F32, ChannelType::U16) => {
            // Linear F32 → PQ/HLG U16: apply OETF during conversion.
            match (from_tf, to_tf) {
                (TransferFunction::Linear, TransferFunction::Pq) => {
                    Ok(vec![ConvertStep::LinearF32ToPqU16])
                }
                (TransferFunction::Linear, TransferFunction::Hlg) => {
                    Ok(vec![ConvertStep::LinearF32ToHlgU16])
                }
                (a, b) if a == b => Ok(vec![ConvertStep::F32ToU16]),
                _ => {
                    let mut steps = f32_tf_pair_steps(from_tf, to_tf);
                    steps.push(ConvertStep::F32ToU16);
                    Ok(steps)
                }
            }
        }
        (ChannelType::U16, ChannelType::U8) => {
            // HDR U16 → SDR U8: go through linear F32 with proper EOTF → OETF.
            if from_tf == TransferFunction::Pq && to_tf == TransferFunction::Srgb {
                Ok(vec![
                    ConvertStep::PqU16ToLinearF32,
                    ConvertStep::LinearF32ToSrgbU8,
                ])
            } else if from_tf == TransferFunction::Hlg && to_tf == TransferFunction::Srgb {
                Ok(vec![
                    ConvertStep::HlgU16ToLinearF32,
                    ConvertStep::LinearF32ToSrgbU8,
                ])
            } else if from_tf == to_tf {
                Ok(vec![ConvertStep::U16ToU8])
            } else {
                let mut steps = Vec::with_capacity(4);
                steps.push(ConvertStep::U16ToF32);
                steps.extend(f32_tf_pair_steps(from_tf, to_tf));
                steps.push(ConvertStep::NaiveF32ToU8);
                Ok(steps)
            }
        }
        (ChannelType::U8, ChannelType::U16) => {
            if from_tf == to_tf {
                Ok(vec![ConvertStep::U8ToU16])
            } else {
                let mut steps = Vec::with_capacity(4);
                steps.push(ConvertStep::NaiveU8ToF32);
                steps.extend(f32_tf_pair_steps(from_tf, to_tf));
                steps.push(ConvertStep::F32ToU16);
                Ok(steps)
            }
        }
        // F16 paths route through F32. No fused TF kernels yet — these are
        // optimization targets for a future pass.
        (ChannelType::F16, ChannelType::F32) => {
            let mut steps = Vec::with_capacity(3);
            steps.push(ConvertStep::F16ToF32);
            if from_tf != to_tf {
                steps.extend(f32_tf_pair_steps(from_tf, to_tf));
            }
            Ok(steps)
        }
        (ChannelType::F32, ChannelType::F16) => {
            let mut steps = Vec::with_capacity(3);
            if from_tf != to_tf {
                steps.extend(f32_tf_pair_steps(from_tf, to_tf));
            }
            steps.push(ConvertStep::F32ToF16);
            Ok(steps)
        }
        (ChannelType::F16, ChannelType::U8) => {
            let mut steps = Vec::with_capacity(4);
            steps.push(ConvertStep::F16ToF32);
            if from_tf == TransferFunction::Linear && to_tf == TransferFunction::Srgb {
                steps.push(ConvertStep::LinearF32ToSrgbU8);
            } else if from_tf == to_tf {
                steps.push(ConvertStep::NaiveF32ToU8);
            } else {
                steps.extend(f32_tf_pair_steps(from_tf, to_tf));
                steps.push(ConvertStep::NaiveF32ToU8);
            }
            Ok(steps)
        }
        (ChannelType::U8, ChannelType::F16) => {
            let mut steps = Vec::with_capacity(4);
            if from_tf == TransferFunction::Srgb && to_tf == TransferFunction::Linear {
                steps.push(ConvertStep::SrgbU8ToLinearF32);
            } else if from_tf == to_tf {
                steps.push(ConvertStep::NaiveU8ToF32);
            } else {
                steps.push(ConvertStep::NaiveU8ToF32);
                steps.extend(f32_tf_pair_steps(from_tf, to_tf));
            }
            steps.push(ConvertStep::F32ToF16);
            Ok(steps)
        }
        (ChannelType::F16, ChannelType::U16) => {
            let mut steps = Vec::with_capacity(4);
            steps.push(ConvertStep::F16ToF32);
            if from_tf == TransferFunction::Linear && to_tf == TransferFunction::Pq {
                steps.push(ConvertStep::LinearF32ToPqU16);
            } else if from_tf == TransferFunction::Linear && to_tf == TransferFunction::Hlg {
                steps.push(ConvertStep::LinearF32ToHlgU16);
            } else if from_tf == to_tf {
                steps.push(ConvertStep::F32ToU16);
            } else {
                steps.extend(f32_tf_pair_steps(from_tf, to_tf));
                steps.push(ConvertStep::F32ToU16);
            }
            Ok(steps)
        }
        (ChannelType::U16, ChannelType::F16) => {
            let mut steps = Vec::with_capacity(4);
            if from_tf == TransferFunction::Pq && to_tf == TransferFunction::Linear {
                steps.push(ConvertStep::PqU16ToLinearF32);
            } else if from_tf == TransferFunction::Hlg && to_tf == TransferFunction::Linear {
                steps.push(ConvertStep::HlgU16ToLinearF32);
            } else if from_tf == to_tf {
                steps.push(ConvertStep::U16ToF32);
            } else {
                steps.push(ConvertStep::U16ToF32);
                steps.extend(f32_tf_pair_steps(from_tf, to_tf));
            }
            steps.push(ConvertStep::F32ToF16);
            Ok(steps)
        }
        _ => Err(ConvertError::NoPath {
            from: PixelDescriptor::new(from, ChannelLayout::Rgb, None, from_tf),
            to: PixelDescriptor::new(to, ChannelLayout::Rgb, None, to_tf),
        }),
    }
}

// ---------------------------------------------------------------------------
// Row conversion kernels
// ---------------------------------------------------------------------------

/// Pre-allocated scratch buffer for multi-step row conversions.
///
/// Eliminates per-row heap allocation by reusing two ping-pong halves
/// of a single buffer across calls. Create once per [`ConvertPlan`],
/// then pass to `convert_row_buffered` for each row.
pub(crate) struct ConvertScratch {
    /// Single allocation split into two halves via `split_at_mut`.
    /// Stored as `Vec<u32>` to guarantee 4-byte alignment, which lets
    /// garb and bytemuck use fast aligned paths instead of unaligned fallbacks.
    buf: Vec<u32>,
    /// Row-persistent scratch for the HDR tone-map kernels (RGB strip +
    /// cached `SoftCompress` gamut LUT). Empty placeholder without the
    /// `hdr-experimental` feature.
    hdr: convert_kernels::HdrKernelScratch,
}

impl ConvertScratch {
    /// Create empty scratch (buffer grows on first use).
    pub(crate) fn new() -> Self {
        Self {
            buf: Vec::new(),
            hdr: convert_kernels::HdrKernelScratch::default(),
        }
    }

    /// Ensure the buffer is large enough for two halves of the max
    /// intermediate format at the given width.
    fn ensure_capacity(&mut self, plan: &ConvertPlan, width: u32) {
        let half_bytes = (width as usize) * plan.max_intermediate_bpp();
        let total_u32 = (half_bytes * 2).div_ceil(4);
        if self.buf.len() < total_u32 {
            self.buf.resize(total_u32, 0);
        }
    }
}

impl core::fmt::Debug for ConvertScratch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ConvertScratch")
            .field("capacity", &self.buf.capacity())
            .finish()
    }
}

/// Convert one row of `width` pixels using a pre-computed plan.
///
/// `src` and `dst` must be sized for `width` pixels in their respective formats.
/// For multi-step plans, an internal scratch buffer is allocated per call.
/// Prefer [`RowConverter`](crate::RowConverter) in hot loops (reuses scratch buffers).
pub fn convert_row(plan: &ConvertPlan, src: &[u8], dst: &mut [u8], width: u32) {
    // Allocating fallback for one-off calls: the scratch starts empty and
    // only grows if the plan actually needs it (multi-step ping-pong or an
    // HDR tone-map kernel); identity and other single-step plans stay
    // allocation-free.
    let mut scratch = ConvertScratch::new();
    convert_row_buffered(plan, src, dst, width, &mut scratch);
}

/// Convert one row of `width` pixels, reusing pre-allocated scratch buffers.
///
/// For multi-step plans this avoids per-row heap allocation by ping-ponging
/// between two halves of a scratch buffer. Single-step plans bypass scratch.
pub(crate) fn convert_row_buffered(
    plan: &ConvertPlan,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    scratch: &mut ConvertScratch,
) {
    if plan.is_identity() {
        let len = min(src.len(), dst.len());
        dst[..len].copy_from_slice(&src[..len]);
        return;
    }

    if plan.steps.len() == 1 {
        apply_step_u8(
            &plan.steps[0],
            src,
            dst,
            width,
            plan.from,
            plan.to,
            plan.pq_anchor_scale,
            &mut scratch.hdr,
        );
        return;
    }

    scratch.ensure_capacity(plan, width);

    // Destructure so the ping-pong halves and the HDR kernel scratch are
    // disjoint mutable borrows across the step loop.
    let ConvertScratch { buf, hdr } = scratch;
    let buf_bytes: &mut [u8] = bytemuck::cast_slice_mut(buf.as_mut_slice());
    let half = buf_bytes.len() / 2;
    let (buf_a, buf_b) = buf_bytes.split_at_mut(half);

    let num_steps = plan.steps.len();
    let mut current_desc = plan.from;

    for (i, step) in plan.steps.iter().enumerate() {
        let is_last = i == num_steps - 1;
        let next_desc = if is_last {
            plan.to
        } else {
            intermediate_desc(current_desc, step)
        };

        let next_len = (width as usize) * next_desc.bytes_per_pixel();
        let curr_len = (width as usize) * current_desc.bytes_per_pixel();

        // Ping-pong: even steps read src/buf_b and write buf_a;
        // odd steps read buf_a and write buf_b. Each branch only
        // borrows each half in one mode, satisfying the borrow checker.
        if i % 2 == 0 {
            let input = if i == 0 { src } else { &buf_b[..curr_len] };
            if is_last {
                apply_step_u8(
                    step,
                    input,
                    dst,
                    width,
                    current_desc,
                    next_desc,
                    plan.pq_anchor_scale,
                    &mut *hdr,
                );
            } else {
                apply_step_u8(
                    step,
                    input,
                    &mut buf_a[..next_len],
                    width,
                    current_desc,
                    next_desc,
                    plan.pq_anchor_scale,
                    &mut *hdr,
                );
            }
        } else {
            let input = &buf_a[..curr_len];
            if is_last {
                apply_step_u8(
                    step,
                    input,
                    dst,
                    width,
                    current_desc,
                    next_desc,
                    plan.pq_anchor_scale,
                    &mut *hdr,
                );
            } else {
                apply_step_u8(
                    step,
                    input,
                    &mut buf_b[..next_len],
                    width,
                    current_desc,
                    next_desc,
                    plan.pq_anchor_scale,
                    &mut *hdr,
                );
            }
        }

        current_desc = next_desc;
    }
}

/// Check if two steps are inverses that cancel each other.
/// Collapse `[SrgbU8ToLinearF32, GamutMatrix*F32(m), LinearF32ToSrgbU8]`
/// into a single fused matlut step. Mutates in place.
fn fuse_matlut_patterns(steps: &mut Vec<ConvertStep>) {
    let mut i = 0;
    while i + 2 < steps.len() {
        let rewrite = match (&steps[i], &steps[i + 1], &steps[i + 2]) {
            (
                ConvertStep::SrgbU8ToLinearF32,
                ConvertStep::GamutMatrixRgbF32(m),
                ConvertStep::LinearF32ToSrgbU8,
            ) => Some(ConvertStep::Fused {
                kind: FusedKind::SrgbU8GamutRgb,
                matrix: *m,
            }),
            (
                ConvertStep::SrgbU8ToLinearF32,
                ConvertStep::GamutMatrixRgbaF32(m),
                ConvertStep::LinearF32ToSrgbU8,
            ) => Some(ConvertStep::Fused {
                kind: FusedKind::SrgbU8GamutRgba,
                matrix: *m,
            }),
            _ => None,
        };
        if let Some(fused) = rewrite {
            steps[i] = fused;
            steps.drain(i + 1..i + 3);
            continue;
        }
        i += 1;
    }
}

fn are_inverse(a: &ConvertStep, b: &ConvertStep) -> bool {
    matches!(
        (a, b),
        // Self-inverse
        (ConvertStep::SwizzleBgraRgba, ConvertStep::SwizzleBgraRgba)
        // Layout inverses (lossless for opaque data)
        | (ConvertStep::AddAlpha, ConvertStep::DropAlpha)
        // Transfer function f32↔f32 (exact inverses in float)
        | (ConvertStep::SrgbF32ToLinearF32, ConvertStep::LinearF32ToSrgbF32)
        | (ConvertStep::LinearF32ToSrgbF32, ConvertStep::SrgbF32ToLinearF32)
        | (ConvertStep::PqF32ToLinearF32, ConvertStep::LinearF32ToPqF32)
        | (ConvertStep::LinearF32ToPqF32, ConvertStep::PqF32ToLinearF32)
        | (ConvertStep::HlgF32ToLinearF32, ConvertStep::LinearF32ToHlgF32)
        | (ConvertStep::LinearF32ToHlgF32, ConvertStep::HlgF32ToLinearF32)
        | (ConvertStep::Bt709F32ToLinearF32, ConvertStep::LinearF32ToBt709F32)
        | (ConvertStep::LinearF32ToBt709F32, ConvertStep::Bt709F32ToLinearF32)
        | (ConvertStep::Gamma22F32ToLinearF32, ConvertStep::LinearF32ToGamma22F32)
        | (ConvertStep::LinearF32ToGamma22F32, ConvertStep::Gamma22F32ToLinearF32)
        // Alpha mode (exact inverses in float)
        | (ConvertStep::StraightToPremul, ConvertStep::PremulToStraight)
        | (ConvertStep::PremulToStraight, ConvertStep::StraightToPremul)
        // Color model (exact inverses in float)
        | (ConvertStep::LinearRgbToOklab, ConvertStep::OklabToLinearRgb)
        | (ConvertStep::OklabToLinearRgb, ConvertStep::LinearRgbToOklab)
        | (ConvertStep::LinearRgbaToOklaba, ConvertStep::OklabaToLinearRgba)
        | (ConvertStep::OklabaToLinearRgba, ConvertStep::LinearRgbaToOklaba)
        // Cross-depth pairs (near-lossless for same depth class)
        | (ConvertStep::NaiveU8ToF32, ConvertStep::NaiveF32ToU8)
        | (ConvertStep::NaiveF32ToU8, ConvertStep::NaiveU8ToF32)
        | (ConvertStep::U8ToU16, ConvertStep::U16ToU8)
        | (ConvertStep::U16ToU8, ConvertStep::U8ToU16)
        | (ConvertStep::U16ToF32, ConvertStep::F32ToU16)
        | (ConvertStep::F32ToU16, ConvertStep::U16ToF32)
        | (ConvertStep::F16ToF32, ConvertStep::F32ToF16)
        | (ConvertStep::F32ToF16, ConvertStep::F16ToF32)
        // Cross-depth with transfer (near-lossless roundtrip)
        | (ConvertStep::SrgbU8ToLinearF32, ConvertStep::LinearF32ToSrgbU8)
        | (ConvertStep::LinearF32ToSrgbU8, ConvertStep::SrgbU8ToLinearF32)
        | (ConvertStep::PqU16ToLinearF32, ConvertStep::LinearF32ToPqU16)
        | (ConvertStep::LinearF32ToPqU16, ConvertStep::PqU16ToLinearF32)
        | (ConvertStep::HlgU16ToLinearF32, ConvertStep::LinearF32ToHlgU16)
        | (ConvertStep::LinearF32ToHlgU16, ConvertStep::HlgU16ToLinearF32)
        // Extended-range sRGB f32 pairs
        | (ConvertStep::SrgbF32ToLinearF32Extended, ConvertStep::LinearF32ToSrgbF32Extended)
        | (ConvertStep::LinearF32ToSrgbF32Extended, ConvertStep::SrgbF32ToLinearF32Extended)
    )
}

/// Compute the descriptor after applying one step.
fn intermediate_desc(current: PixelDescriptor, step: &ConvertStep) -> PixelDescriptor {
    match step {
        ConvertStep::Identity => current,
        ConvertStep::SwizzleBgraRgba => {
            let new_layout = match current.layout() {
                ChannelLayout::Bgra => ChannelLayout::Rgba,
                ChannelLayout::Rgba => ChannelLayout::Bgra,
                other => other,
            };
            PixelDescriptor::new(
                current.channel_type(),
                new_layout,
                current.alpha(),
                current.transfer(),
            )
        }
        ConvertStep::AddAlpha => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            current.transfer(),
        ),
        ConvertStep::RgbToBgra => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Bgra,
            Some(AlphaMode::Straight),
            current.transfer(),
        ),
        ConvertStep::DropAlpha | ConvertStep::MatteComposite { .. } => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Rgb,
            None,
            current.transfer(),
        ),
        ConvertStep::GrayToRgb => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Rgb,
            None,
            current.transfer(),
        ),
        ConvertStep::GrayToRgba => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            current.transfer(),
        ),
        ConvertStep::RgbToGray { .. } | ConvertStep::RgbaToGray { .. } => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Gray,
            None,
            current.transfer(),
        ),
        ConvertStep::GrayAlphaToRgba => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Rgba,
            current.alpha(),
            current.transfer(),
        ),
        ConvertStep::GrayAlphaToRgb => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Rgb,
            None,
            current.transfer(),
        ),
        ConvertStep::GrayToGrayAlpha => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            current.transfer(),
        ),
        ConvertStep::GrayAlphaToGray => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Gray,
            None,
            current.transfer(),
        ),
        ConvertStep::SrgbU8ToLinearF32
        | ConvertStep::NaiveU8ToF32
        | ConvertStep::U16ToF32
        | ConvertStep::PqU16ToLinearF32
        | ConvertStep::HlgU16ToLinearF32
        | ConvertStep::PqF32ToLinearF32
        | ConvertStep::HlgF32ToLinearF32
        | ConvertStep::SrgbF32ToLinearF32
        | ConvertStep::SrgbF32ToLinearF32Extended
        | ConvertStep::Bt709F32ToLinearF32
        | ConvertStep::Gamma22F32ToLinearF32 => PixelDescriptor::new(
            ChannelType::F32,
            current.layout(),
            current.alpha(),
            TransferFunction::Linear,
        ),
        ConvertStep::LinearF32ToSrgbU8 | ConvertStep::NaiveF32ToU8 | ConvertStep::U16ToU8 => {
            PixelDescriptor::new(
                ChannelType::U8,
                current.layout(),
                current.alpha(),
                TransferFunction::Srgb,
            )
        }
        ConvertStep::U8ToU16 => PixelDescriptor::new(
            ChannelType::U16,
            current.layout(),
            current.alpha(),
            current.transfer(),
        ),
        ConvertStep::F32ToU16 | ConvertStep::LinearF32ToPqU16 | ConvertStep::LinearF32ToHlgU16 => {
            let tf = match step {
                ConvertStep::LinearF32ToPqU16 => TransferFunction::Pq,
                ConvertStep::LinearF32ToHlgU16 => TransferFunction::Hlg,
                _ => current.transfer(),
            };
            PixelDescriptor::new(ChannelType::U16, current.layout(), current.alpha(), tf)
        }
        ConvertStep::LinearF32ToPqF32 => PixelDescriptor::new(
            ChannelType::F32,
            current.layout(),
            current.alpha(),
            TransferFunction::Pq,
        ),
        ConvertStep::LinearF32ToHlgF32 => PixelDescriptor::new(
            ChannelType::F32,
            current.layout(),
            current.alpha(),
            TransferFunction::Hlg,
        ),
        ConvertStep::LinearF32ToSrgbF32 | ConvertStep::LinearF32ToSrgbF32Extended => {
            PixelDescriptor::new(
                ChannelType::F32,
                current.layout(),
                current.alpha(),
                TransferFunction::Srgb,
            )
        }
        ConvertStep::LinearF32ToBt709F32 => PixelDescriptor::new(
            ChannelType::F32,
            current.layout(),
            current.alpha(),
            TransferFunction::Bt709,
        ),
        ConvertStep::LinearF32ToGamma22F32 => PixelDescriptor::new(
            ChannelType::F32,
            current.layout(),
            current.alpha(),
            TransferFunction::Gamma22,
        ),
        ConvertStep::StraightToPremul => PixelDescriptor::new(
            current.channel_type(),
            current.layout(),
            Some(AlphaMode::Premultiplied),
            current.transfer(),
        ),
        ConvertStep::PremulToStraight => PixelDescriptor::new(
            current.channel_type(),
            current.layout(),
            Some(AlphaMode::Straight),
            current.transfer(),
        ),
        ConvertStep::LinearRgbToOklab => PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Oklab,
            None,
            TransferFunction::Unknown,
        )
        .with_primaries(current.primaries),
        ConvertStep::OklabToLinearRgb => PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        )
        .with_primaries(current.primaries),
        ConvertStep::LinearRgbaToOklaba => PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::OklabA,
            Some(AlphaMode::Straight),
            TransferFunction::Unknown,
        )
        .with_primaries(current.primaries),
        ConvertStep::OklabaToLinearRgba => PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            current.alpha(),
            TransferFunction::Linear,
        )
        .with_primaries(current.primaries),

        // Gamut matrix: same depth/layout/TF, but primaries change.
        // The actual target primaries are embedded in the matrix, not tracked
        // here — we mark them as Unknown since the step doesn't carry that info.
        // The final plan.to descriptor has the correct primaries.
        ConvertStep::GamutMatrixRgbF32(_) => PixelDescriptor::new(
            ChannelType::F32,
            current.layout(),
            current.alpha(),
            TransferFunction::Linear,
        ),
        ConvertStep::GamutMatrixRgbaF32(_) => PixelDescriptor::new(
            ChannelType::F32,
            current.layout(),
            current.alpha(),
            TransferFunction::Linear,
        ),
        // Fused steps: shape depends on FusedKind.
        ConvertStep::Fused { kind, .. } => {
            let (ch_type, transfer) = match kind {
                // u8 sRGB in, u8 sRGB out (same layout, same alpha).
                FusedKind::SrgbU8GamutRgb | FusedKind::SrgbU8GamutRgba => {
                    (ChannelType::U8, TransferFunction::Srgb)
                }
                FusedKind::SrgbU16GamutRgb => (ChannelType::U16, TransferFunction::Srgb),
                FusedKind::SrgbU8ToLinearF32Rgb => (ChannelType::F32, TransferFunction::Linear),
                FusedKind::LinearF32ToSrgbU8Rgb => (ChannelType::U8, TransferFunction::Srgb),
            };
            PixelDescriptor::new(ch_type, current.layout(), current.alpha(), transfer)
        }
        // F16↔F32 depth-only steps. No TF implication: same TF on both sides.
        ConvertStep::F16ToF32 => PixelDescriptor::new(
            ChannelType::F32,
            current.layout(),
            current.alpha(),
            current.transfer(),
        ),
        ConvertStep::F32ToF16 => PixelDescriptor::new(
            ChannelType::F16,
            current.layout(),
            current.alpha(),
            current.transfer(),
        ),
        // HDR steps. Both operate on linear-light F32 RGB and preserve the
        // layout/alpha/transfer/depth of the carrier. ToneMapBt2446A operates
        // in BT.2020 (planner-enforced); SoftCompressOklch operates in the
        // step's stored `primaries`. Neither step changes the descriptor
        // shape — they only update pixel values.
        #[cfg(feature = "hdr-experimental")]
        ConvertStep::ToneMapBt2446A { .. } => current,
        #[cfg(feature = "hdr-experimental")]
        ConvertStep::SoftCompressOklch { .. } => current,
    }
}

#[path = "convert_kernels.rs"]
mod convert_kernels;
use convert_kernels::apply_step_u8;
pub(crate) use convert_kernels::{hlg_eotf, hlg_oetf, pq_eotf, pq_oetf};

#[cfg(all(test, feature = "hdr-experimental"))]
mod hdr_plan_tests {
    //! Unit tests pinning the HDR-aware `ConvertPlan` against the same math
    //! the deleted `HdrToSdr::apply_strip` ran. Keeps the e2e ΔE2000 budget
    //! grounded in per-pixel parity rather than only the imazen-26 sample.
    use super::*;
    use crate::gamut::{apply_matrix_f32, conversion_matrix};
    use crate::hdr::{Bt2446A, SoftCompress};
    use crate::oklab;

    /// Reproduce the strip math of the deleted `HdrToSdr::apply_strip` for
    /// a BT.709 (linear) → BT.709 (linear) pipeline at 1000 nit source peak.
    fn reference_pipeline(input: [f32; 3]) -> [f32; 3] {
        let mut px = [input];
        // Scrub.
        for c in px[0].iter_mut() {
            if !c.is_finite() || *c < 0.0 {
                *c = 0.0;
            }
        }
        // Source primaries → BT.2020.
        let m_src = conversion_matrix(ColorPrimaries::Bt709, ColorPrimaries::Bt2020).unwrap();
        for p in px.iter_mut() {
            apply_matrix_f32(p, &m_src);
        }
        // BT.2446-A curve in BT.2020.
        Bt2446A::new(1000.0, 100.0).map_strip_simd(&mut px);
        // BT.2020 → target primaries.
        let m_dst = conversion_matrix(ColorPrimaries::Bt2020, ColorPrimaries::Bt709).unwrap();
        for p in px.iter_mut() {
            apply_matrix_f32(p, &m_dst);
        }
        // OKLch soft compress in target primaries.
        let m1 = oklab::rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();
        let m1_inv = oklab::lms_to_rgb_matrix(ColorPrimaries::Bt709).unwrap();
        let compressor = SoftCompress::from_matrices(&m1, &m1_inv, 0.96);
        compressor.apply_strip(&mut px);
        // Final clamp.
        for c in px[0].iter_mut() {
            if !c.is_finite() {
                *c = 0.0;
            } else {
                *c = c.clamp(0.0, 1.0);
            }
        }
        px[0]
    }

    /// Single-pixel sanity using the same entry the e2e test uses
    /// (`PixelBufferHdrConvertExt::convert_to_with_hdr_config`) — pin
    /// that the user-facing extension method routes through the same
    /// kernels the manual reference uses.
    #[test]
    fn pixel_buffer_hdr_convert_matches_reference_pipeline() {
        use crate::PixelBufferHdrConvertExt;
        use zenpixels::PixelBuffer;
        let src = PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
            ColorPrimaries::Bt709,
        );
        let to = PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
            ColorPrimaries::Bt709,
        );
        let hdr = HdrConfig {
            source_peak_nits: 1000.0,
            target_peak_nits: 100.0,
            gamut_knee: 0.96,
        };
        let inputs = [
            [0.0_f32, 0.0, 0.0],
            [0.18, 0.18, 0.18],
            [1.0, 1.0, 1.0],
            [0.5, 0.3, 0.1],
        ];
        for inp in inputs {
            let expected = reference_pipeline(inp);
            let bytes: Vec<u8> = bytemuck::cast_slice(&inp).to_vec();
            let buf = PixelBuffer::from_vec(bytes, 1, 1, src).unwrap();
            let out = buf.convert_to_with_hdr_config(to, hdr).expect("convert");
            let out_bytes = out.copy_to_contiguous_bytes();
            let got: &[f32] = bytemuck::cast_slice(&out_bytes);
            for k in 0..3 {
                let diff = (expected[k] - got[k]).abs();
                assert!(
                    diff < 5e-4,
                    "ext channel {k} for input {inp:?}: expected {} vs got {} (diff {})",
                    expected[k],
                    got[k],
                    diff,
                );
            }
        }
    }

    #[test]
    fn hdr_plan_matches_reference_pipeline_for_bt709_linear_targets() {
        let src = PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
            ColorPrimaries::Bt709,
        );
        let to = PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
            ColorPrimaries::Bt709,
        );
        let hdr = HdrConfig {
            source_peak_nits: 1000.0,
            target_peak_nits: 100.0,
            gamut_knee: 0.96,
        };
        let plan = ConvertPlan::new_with_hdr_config(src, to, hdr).expect("plan");
        let inputs = [
            [0.0_f32, 0.0, 0.0],
            [0.18, 0.18, 0.18],
            [1.0, 1.0, 1.0],
            [0.5, 0.3, 0.1],
            [0.9, 0.1, 0.05],
        ];
        for inp in inputs {
            let expected = reference_pipeline(inp);
            let bytes: Vec<u8> = bytemuck::cast_slice(&inp).to_vec();
            let mut out = vec![0u8; 12];
            convert_row(&plan, &bytes, &mut out, 1);
            let got_f: &[f32] = bytemuck::cast_slice(&out);
            for k in 0..3 {
                let diff = (expected[k] - got_f[k]).abs();
                assert!(
                    diff < 5e-4,
                    "channel {k} for input {inp:?}: expected {} vs got {} (diff {})",
                    expected[k],
                    got_f[k],
                    diff,
                );
            }
        }
    }
}
