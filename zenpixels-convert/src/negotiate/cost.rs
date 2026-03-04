//! Component cost functions + scoring + suitability penalties.

use crate::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction,
};

use super::{ConversionCost, ConvertIntent, Provenance};

// ---------------------------------------------------------------------------
// Internal scoring
// ---------------------------------------------------------------------------

/// Score a candidate target for ranking. Lower is better.
fn score_target(
    source: PixelDescriptor,
    provenance: Provenance,
    target: PixelDescriptor,
    consumer_cost: ConversionCost,
    intent: ConvertIntent,
) -> u32 {
    let our_cost = super::conversion_cost_with_provenance(source, target, provenance);
    let total_effort = our_cost.effort as u32 + consumer_cost.effort as u32;
    let total_loss =
        our_cost.loss as u32 + consumer_cost.loss as u32 + suitability_loss(target, intent) as u32;
    weighted_score(total_effort, total_loss, intent)
}

/// Find the best target from an iterator of (descriptor, consumer_cost) pairs.
pub(super) fn best_of(
    source: PixelDescriptor,
    provenance: Provenance,
    options: impl Iterator<Item = (PixelDescriptor, ConversionCost)>,
    intent: ConvertIntent,
) -> Option<PixelDescriptor> {
    let mut best: Option<(PixelDescriptor, u32)> = None;

    for (target, consumer_cost) in options {
        let score = score_target(source, provenance, target, consumer_cost, intent);
        match best {
            Some((_, best_score)) if score < best_score => best = Some((target, score)),
            None => best = Some((target, score)),
            _ => {}
        }
    }

    best.map(|(desc, _)| desc)
}

/// Blend effort and loss into a single ranking score based on intent.
///
/// - `Fastest`: effort matters 4x more than loss
/// - `LinearLight`/`Blend`: loss matters 4x more than effort
/// - `Perceptual`: loss matters 3x more than effort
pub(crate) fn weighted_score(effort: u32, loss: u32, intent: ConvertIntent) -> u32 {
    match intent {
        ConvertIntent::Fastest => effort * 4 + loss,
        ConvertIntent::LinearLight | ConvertIntent::Blend => effort + loss * 4,
        ConvertIntent::Perceptual => effort + loss * 3,
    }
}

/// How unsuitable a target format is for the given intent.
///
/// This is a quality-of-operation penalty, not a conversion penalty.
/// For example, u8 data processed with LinearLight resize produces
/// gamma-incorrect results — that's a quality loss independent of
/// how cheap the u8→u8 identity conversion is.
pub(crate) fn suitability_loss(target: PixelDescriptor, intent: ConvertIntent) -> u16 {
    match intent {
        ConvertIntent::Fastest => 0,
        ConvertIntent::LinearLight => linear_light_suitability(target),
        ConvertIntent::Blend => {
            let mut s = linear_light_suitability(target);
            // Straight alpha requires per-pixel division during compositing.
            // Calibrated: blending in straight alpha causes severe fringe artifacts
            // at semi-transparent edges (measured ΔE=17.2, High bucket).
            if target.layout().has_alpha() && target.alpha() == Some(AlphaMode::Straight) {
                s += 200;
            }
            s
        }
        ConvertIntent::Perceptual => perceptual_suitability(target),
    }
}

/// Suitability penalty for LinearLight operations (resize, blur).
/// f32 Linear is ideal; any gamma-encoded format produces gamma
/// darkening artifacts that dominate over quantization.
///
/// # Calibration notes (CIEDE2000 measurements)
///
/// **Non-linear (gamma-encoded) formats:**
/// Bilinear resize in sRGB measures p95 ΔE ≈ 13.7 regardless of bit depth
/// (u8=13.7, u16=13.7, f16=13.7, f32 sRGB=13.7).
/// Gamma darkening is the dominant error — precision barely matters.
///
/// **Linear formats:**
/// Only quantization matters. f32=0, f16=0.022, u8=0.213.
#[allow(unreachable_patterns)] // non_exhaustive: future variants
fn linear_light_suitability(target: PixelDescriptor) -> u16 {
    if target.transfer() == TransferFunction::Linear {
        // Linear space: only quantization error.
        match target.channel_type() {
            ChannelType::F32 => 0,
            ChannelType::F16 => 5, // 10 mantissa bits, measured ΔE=0.022
            ChannelType::U16 => 5, // 16 bits, negligible quantization
            ChannelType::U8 => 40, // severe banding in darks, measured ΔE=0.213
            _ => 50,
        }
    } else {
        // Non-linear (sRGB, BT.709, PQ, HLG): gamma darkening dominates.
        // All measure p95 ΔE ≈ 13.7 for resize regardless of precision.
        120
    }
}

/// Suitability penalty for perceptual operations (sharpening, color grading).
/// sRGB-encoded data is ideal; Linear f32 is slightly off.
fn perceptual_suitability(target: PixelDescriptor) -> u16 {
    if target.channel_type() == ChannelType::F32 && target.transfer() == TransferFunction::Linear {
        return 15;
    }
    if matches!(
        target.transfer(),
        TransferFunction::Pq | TransferFunction::Hlg
    ) {
        return 10;
    }
    0
}

// ---------------------------------------------------------------------------
// Component cost functions (effort + loss)
// ---------------------------------------------------------------------------

/// Cost of transfer function conversion.
pub(super) fn transfer_cost(from: TransferFunction, to: TransferFunction) -> ConversionCost {
    if from == to {
        return ConversionCost::ZERO;
    }
    match (from, to) {
        // Unknown → anything: relabeling, no actual math.
        (TransferFunction::Unknown, _) | (_, TransferFunction::Unknown) => {
            ConversionCost::new(1, 0)
        }

        // sRGB ↔ Linear: well-optimized EOTF/OETF, lossless in f32.
        (TransferFunction::Srgb, TransferFunction::Linear)
        | (TransferFunction::Linear, TransferFunction::Srgb) => ConversionCost::new(5, 0),

        // BT.709 ↔ sRGB/Linear: slightly different curve, near-lossless.
        (TransferFunction::Bt709, TransferFunction::Srgb)
        | (TransferFunction::Srgb, TransferFunction::Bt709)
        | (TransferFunction::Bt709, TransferFunction::Linear)
        | (TransferFunction::Linear, TransferFunction::Bt709) => ConversionCost::new(8, 0),

        // PQ/HLG ↔ anything else: expensive and lossy (range/gamut clipping).
        _ => ConversionCost::new(80, 300),
    }
}

/// Cost of channel depth conversion, considering the data's origin precision.
///
/// The `origin_depth` comes from [`Provenance`] and tells us the true
/// precision of the data. This matters because:
///
/// - `f32 → u8` with origin `U8`: the data was u8 JPEG decoded to f32
///   for processing. Round-trip back to u8 is lossless (±1 LSB).
/// - `f32 → u8` with origin `F32`: the data has true f32 precision
///   (e.g., EXR or HDR AVIF). Truncating to u8 destroys highlight detail.
///
/// **Rule:** If `target depth >= origin depth`, the narrowing has zero loss
/// because the target can represent everything the original data contained.
/// The effort cost is always based on the *current* depth conversion work.
pub(super) fn depth_cost(
    from: ChannelType,
    to: ChannelType,
    origin_depth: ChannelType,
) -> ConversionCost {
    if from == to {
        return ConversionCost::ZERO;
    }

    let effort = depth_effort(from, to);
    let loss = depth_loss(to, origin_depth);

    ConversionCost::new(effort, loss)
}

/// Computational effort for a depth conversion (independent of provenance).
#[allow(unreachable_patterns)] // non_exhaustive: future variants
fn depth_effort(from: ChannelType, to: ChannelType) -> u16 {
    match (from, to) {
        // Integer widen/narrow
        (ChannelType::U8, ChannelType::U16) | (ChannelType::U16, ChannelType::U8) => 10,
        // Float ↔ integer
        (ChannelType::U16, ChannelType::F32) | (ChannelType::F32, ChannelType::U16) => 25,
        (ChannelType::U8, ChannelType::F32) | (ChannelType::F32, ChannelType::U8) => 40,
        // F16 ↔ F32 (hardware or fast table conversion)
        (ChannelType::F16, ChannelType::F32) | (ChannelType::F32, ChannelType::F16) => 15,
        // F16 ↔ integer (via F32 intermediate)
        (ChannelType::F16, ChannelType::U8) | (ChannelType::U8, ChannelType::F16) => 30,
        (ChannelType::F16, ChannelType::U16) | (ChannelType::U16, ChannelType::F16) => 25,
        // remaining catch-all handles unknown future types
        _ => 100,
    }
}

/// Information loss when converting to `target_depth`, given the data's
/// `origin_depth`. If the target can represent the origin precision,
/// loss is zero.
#[allow(unreachable_patterns)] // non_exhaustive: future variants
fn depth_loss(target_depth: ChannelType, origin_depth: ChannelType) -> u16 {
    let target_bits = channel_bits(target_depth);
    let origin_bits = channel_bits(origin_depth);

    if target_bits >= origin_bits {
        // Target can hold everything the origin had — no loss.
        return 0;
    }

    // Target has less precision than the origin — lossy.
    //
    // Calibrated from CIEDE2000 measurements (perceptual_loss.rs).
    // In sRGB space, quantization to 8 bits produces ΔE < 0.5 (below JND)
    // because sRGB OETF provides perceptually uniform quantization.
    // The suitability_loss function handles the separate concern of
    // operating in a lower-precision format (gamma darkening in u8, etc).
    match (origin_depth, target_depth) {
        (ChannelType::U16, ChannelType::U8) => 10, // measured ΔE=0.14, sub-JND
        (ChannelType::F32, ChannelType::U8) => 10, // measured ΔE=0.14, sub-JND in sRGB
        (ChannelType::F32, ChannelType::U16) => 5, // 23→16 mantissa bits, negligible
        (ChannelType::F32, ChannelType::F16) => 20, // 23→10 mantissa bits, small loss
        (ChannelType::F16, ChannelType::U8) => 8,  // measured ΔE=0.000 (f16 >8 bits precision)
        (ChannelType::U16, ChannelType::F16) => 30, // 16→10 mantissa bits, moderate loss
        _ => 50,
    }
}

/// Nominal precision bits for a channel type (for ordering, not bit-exact).
///
/// F16 has 10 mantissa bits (~3.3 decimal digits) — between U8 (8 bits) and
/// U16 (16 bits).
#[allow(unreachable_patterns)] // non_exhaustive: future variants
pub(crate) fn channel_bits(ct: ChannelType) -> u16 {
    match ct {
        ChannelType::U8 => 8,
        ChannelType::F16 => 11, // 10 mantissa + 1 implicit
        ChannelType::U16 => 16,
        ChannelType::F32 => 32,
        _ => 0,
    }
}

/// Cost of color primaries (gamut) conversion.
///
/// Gamut hierarchy: BT.2020 ⊃ Display P3 ⊃ BT.709/sRGB.
///
/// Key principle: narrowing is only lossy if the data actually uses the
/// wider gamut. Provenance tracks whether the data originally came from
/// a narrower gamut (and hasn't been modified to use the wider one).
pub(super) fn primaries_cost(
    from: ColorPrimaries,
    to: ColorPrimaries,
    origin: ColorPrimaries,
) -> ConversionCost {
    if from == to {
        return ConversionCost::ZERO;
    }

    // Unknown ↔ anything: relabeling only, no actual math.
    if matches!(from, ColorPrimaries::Unknown) || matches!(to, ColorPrimaries::Unknown) {
        return ConversionCost::new(1, 0);
    }

    // Widening (e.g., sRGB → P3 → BT.2020): 3x3 matrix, lossless.
    if to.contains(from) {
        return ConversionCost::new(10, 0);
    }

    // Narrowing (e.g., BT.2020 → sRGB): check if origin fits in target.
    // If the data originally came from sRGB and was placed in BT.2020
    // without modifying colors, converting back to sRGB is near-lossless
    // (only numerical precision of the 3x3 matrix round-trip).
    if to.contains(origin) {
        return ConversionCost::new(10, 5);
    }

    // True gamut clipping: data uses the wider gamut and target is narrower.
    // Loss depends on how much wider the source is.
    match (from, to) {
        (ColorPrimaries::DisplayP3, ColorPrimaries::Bt709) => ConversionCost::new(15, 80),
        (ColorPrimaries::Bt2020, ColorPrimaries::DisplayP3) => ConversionCost::new(15, 100),
        (ColorPrimaries::Bt2020, ColorPrimaries::Bt709) => ConversionCost::new(15, 200),
        _ => ConversionCost::new(15, 150),
    }
}

/// Cost of layout conversion.
pub(super) fn layout_cost(from: ChannelLayout, to: ChannelLayout) -> ConversionCost {
    if from == to {
        return ConversionCost::ZERO;
    }
    match (from, to) {
        // Swizzle: cheap, lossless.
        (ChannelLayout::Bgra, ChannelLayout::Rgba) | (ChannelLayout::Rgba, ChannelLayout::Bgra) => {
            ConversionCost::new(5, 0)
        }

        // Add alpha: cheap, lossless (fill opaque).
        (ChannelLayout::Rgb, ChannelLayout::Rgba) | (ChannelLayout::Rgb, ChannelLayout::Bgra) => {
            ConversionCost::new(10, 0)
        }

        // Drop alpha: cheap but lossy (alpha channel destroyed).
        (ChannelLayout::Rgba, ChannelLayout::Rgb) | (ChannelLayout::Bgra, ChannelLayout::Rgb) => {
            ConversionCost::new(15, 50)
        }

        // Gray → RGB: replicate, lossless.
        (ChannelLayout::Gray, ChannelLayout::Rgb) => ConversionCost::new(8, 0),
        (ChannelLayout::Gray, ChannelLayout::Rgba) | (ChannelLayout::Gray, ChannelLayout::Bgra) => {
            ConversionCost::new(10, 0)
        }

        // Color → Gray: luma calculation, very lossy (color info destroyed).
        (ChannelLayout::Rgb, ChannelLayout::Gray)
        | (ChannelLayout::Rgba, ChannelLayout::Gray)
        | (ChannelLayout::Bgra, ChannelLayout::Gray) => ConversionCost::new(30, 500),

        // GrayAlpha → RGBA: replicate gray, lossless.
        (ChannelLayout::GrayAlpha, ChannelLayout::Rgba)
        | (ChannelLayout::GrayAlpha, ChannelLayout::Bgra) => ConversionCost::new(15, 0),

        // RGBA → GrayAlpha: luma + drop color, very lossy.
        (ChannelLayout::Rgba, ChannelLayout::GrayAlpha)
        | (ChannelLayout::Bgra, ChannelLayout::GrayAlpha) => ConversionCost::new(30, 500),

        // Gray ↔ GrayAlpha.
        (ChannelLayout::Gray, ChannelLayout::GrayAlpha) => ConversionCost::new(8, 0),
        (ChannelLayout::GrayAlpha, ChannelLayout::Gray) => ConversionCost::new(10, 50),

        // GrayAlpha → Rgb: replicate + drop alpha.
        (ChannelLayout::GrayAlpha, ChannelLayout::Rgb) => ConversionCost::new(12, 50),

        // RGB ↔ Oklab: matrix + cube root, lossless at f32.
        (ChannelLayout::Rgb, ChannelLayout::Oklab) | (ChannelLayout::Oklab, ChannelLayout::Rgb) => {
            ConversionCost::new(80, 0)
        }
        (ChannelLayout::Rgba, ChannelLayout::OklabA)
        | (ChannelLayout::OklabA, ChannelLayout::Rgba) => ConversionCost::new(80, 0),

        // Oklab ↔ alpha variants.
        (ChannelLayout::Oklab, ChannelLayout::OklabA) => ConversionCost::new(10, 0),
        (ChannelLayout::OklabA, ChannelLayout::Oklab) => ConversionCost::new(15, 50),

        // Cross-model with alpha changes.
        (ChannelLayout::Rgb, ChannelLayout::OklabA) => ConversionCost::new(90, 0),
        (ChannelLayout::OklabA, ChannelLayout::Rgb) => ConversionCost::new(90, 50),
        (ChannelLayout::Oklab, ChannelLayout::Rgba) => ConversionCost::new(90, 0),
        (ChannelLayout::Rgba, ChannelLayout::Oklab) => ConversionCost::new(90, 50),

        _ => ConversionCost::new(100, 500),
    }
}

/// Cost of alpha mode conversion.
pub(super) fn alpha_cost(
    from_alpha: Option<AlphaMode>,
    to_alpha: Option<AlphaMode>,
    from_layout: ChannelLayout,
    to_layout: ChannelLayout,
) -> ConversionCost {
    if !to_layout.has_alpha() || !from_layout.has_alpha() || from_alpha == to_alpha {
        return ConversionCost::ZERO;
    }
    match (from_alpha, to_alpha) {
        // Straight → Premul: per-pixel multiply, tiny rounding loss.
        (Some(AlphaMode::Straight), Some(AlphaMode::Premultiplied)) => ConversionCost::new(20, 5),
        // Premul → Straight: per-pixel divide, worse rounding at low alpha.
        (Some(AlphaMode::Premultiplied), Some(AlphaMode::Straight)) => ConversionCost::new(25, 10),
        _ => ConversionCost::ZERO,
    }
}
