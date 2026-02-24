//! Format negotiation — pick the best conversion target.
//!
//! The cost model separates **effort** (computational work) from **loss**
//! (information destroyed). The [`ConvertIntent`] controls how these axes
//! are weighted: `Fastest` prioritizes low effort, while `LinearLight` and
//! `Blend` prioritize low loss.
//!
//! Consumers that can perform conversions internally (e.g., a JPEG encoder
//! with a fused f32→u8+DCT path) can express this via [`FormatOption`]
//! with a custom [`ConversionCost`], so the negotiation picks their fast
//! path instead of doing a redundant conversion.

use zencodec_types::{AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Tracks where pixel data came from, so the cost model can distinguish
/// "f32 that was widened from u8 JPEG" (lossless round-trip back to u8)
/// from "f32 that was decoded from a 16-bit EXR" (lossy truncation to u8).
///
/// Without provenance, `depth_cost(f32 → u8)` always reports high loss.
/// With provenance, it can see that the data's *true* precision is u8,
/// so the round-trip is lossless.
///
/// # Future extensions
///
/// This struct will grow to include CICP primaries, matrix coefficients,
/// and codec type (lossy/lossless) — all metadata that describes the
/// data's origin characteristics rather than its current format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Provenance {
    /// The channel depth of the original source data.
    ///
    /// For a JPEG (u8 sRGB) decoded into f32 for resize, this is `U8`.
    /// For an EXR (f32) loaded directly, this is `F32`.
    /// For a 16-bit PNG, this is `U16`.
    pub origin_depth: ChannelType,
}

impl Provenance {
    /// Assume the descriptor's channel type *is* the true origin precision.
    ///
    /// This is the conservative default: if you don't know the data's history,
    /// assume its current depth is its true depth.
    #[inline]
    pub fn from_source(desc: PixelDescriptor) -> Self {
        Self {
            origin_depth: desc.channel_type,
        }
    }

    /// Create provenance with an explicit origin depth.
    ///
    /// Use this when the data has been widened from a known source depth.
    /// For example, a JPEG (u8) decoded into f32 for resize:
    ///
    /// ```rust,ignore
    /// let provenance = Provenance::with_origin_depth(ChannelType::U8);
    /// ```
    #[inline]
    pub const fn with_origin_depth(origin_depth: ChannelType) -> Self {
        Self { origin_depth }
    }
}

/// What the consumer plans to do with the converted pixels.
///
/// Shifts the format negotiation cost model to prefer formats
/// suited for the intended operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConvertIntent {
    /// Minimize conversion effort. Good for encoding — the codec
    /// already knows what it wants, just get there cheaply.
    #[default]
    Fastest,

    /// Pixel-accurate operations that need linear light:
    /// resize, blur, anti-aliasing, mipmap generation.
    /// Prefers f32 Linear. Straight alpha is fine.
    LinearLight,

    /// Compositing and blending (Porter-Duff, layer merge).
    /// Prefers f32 Linear with Premultiplied alpha.
    Blend,

    /// Perceptual adjustments: sharpening, contrast, saturation.
    /// Prefers sRGB space (perceptually uniform).
    Perceptual,
}

/// Two-axis conversion cost: computational effort vs. information loss.
///
/// These are independent concerns:
/// - A fast conversion can be very lossy (f32 HDR → u8 sRGB clamp).
/// - A slow conversion can be lossless (u8 sRGB → f32 Linear).
/// - A consumer's fused path can be fast with the same loss as our path.
///
/// [`ConvertIntent`] controls how the two axes are weighted for ranking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConversionCost {
    /// Computational work (cycles, not quality). Lower is faster.
    pub effort: u16,
    /// Information destroyed (precision, gamut, channels). Lower is more faithful.
    pub loss: u16,
}

impl ConversionCost {
    /// Zero cost — identity conversion.
    pub const ZERO: Self = Self { effort: 0, loss: 0 };

    /// Create a cost with explicit effort and loss.
    pub const fn new(effort: u16, loss: u16) -> Self {
        Self { effort, loss }
    }
}

impl core::ops::Add for ConversionCost {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            effort: self.effort.saturating_add(rhs.effort),
            loss: self.loss.saturating_add(rhs.loss),
        }
    }
}

/// A supported format with optional consumer-provided cost override.
///
/// Use this with [`best_match_with`] when the consumer can handle some
/// formats more efficiently than the default conversion path.
///
/// # Example
///
/// A JPEG encoder with a fast internal f32→u8 path:
///
/// ```rust,ignore
/// use zenpixels::{FormatOption, ConversionCost};
/// use zencodec_types::PixelDescriptor;
///
/// let options = &[
///     FormatOption::from(PixelDescriptor::RGB8_SRGB),     // native, zero cost
///     FormatOption::with_cost(
///         PixelDescriptor::RGBF32_LINEAR,
///         ConversionCost::new(5, 0),  // fast fused path, no extra loss
///     ),
/// ];
/// ```
#[derive(Clone, Copy, Debug)]
pub struct FormatOption {
    /// The pixel format the consumer can accept.
    pub descriptor: PixelDescriptor,
    /// Additional cost the consumer incurs to handle this format
    /// after we deliver it. Zero for native formats.
    pub consumer_cost: ConversionCost,
}

impl FormatOption {
    /// Create an option with explicit consumer cost.
    pub const fn with_cost(descriptor: PixelDescriptor, consumer_cost: ConversionCost) -> Self {
        Self {
            descriptor,
            consumer_cost,
        }
    }
}

impl From<PixelDescriptor> for FormatOption {
    fn from(descriptor: PixelDescriptor) -> Self {
        Self {
            descriptor,
            consumer_cost: ConversionCost::ZERO,
        }
    }
}

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

/// Pick the best conversion target from `supported` for the given source.
///
/// Returns `None` only if `supported` is empty.
///
/// This is the simple API — all consumer costs are assumed zero,
/// and provenance is inferred from the source descriptor (conservative).
/// Use [`best_match_with`] for consumer cost overrides, or
/// [`negotiate`] for full control over provenance and consumer costs.
pub fn best_match(
    source: PixelDescriptor,
    supported: &[PixelDescriptor],
    intent: ConvertIntent,
) -> Option<PixelDescriptor> {
    negotiate(
        source,
        Provenance::from_source(source),
        supported.iter().map(|&d| FormatOption::from(d)),
        intent,
    )
}

/// Pick the best conversion target from `options`, accounting for
/// consumer-provided cost overrides.
///
/// Returns `None` only if `options` is empty.
///
/// Each [`FormatOption`] specifies a format the consumer can accept
/// and what it costs the consumer to handle that format internally.
/// The total cost is `our_conversion + consumer_cost`, weighted by intent.
///
/// Provenance is inferred from the source descriptor. Use [`negotiate`]
/// when the data has been widened from a lower-precision origin.
pub fn best_match_with(
    source: PixelDescriptor,
    options: &[FormatOption],
    intent: ConvertIntent,
) -> Option<PixelDescriptor> {
    negotiate(
        source,
        Provenance::from_source(source),
        options.iter().copied(),
        intent,
    )
}

/// Fully-parameterized format negotiation.
///
/// This is the most flexible entry point: it takes explicit provenance
/// (so the cost model knows the data's true origin precision) and
/// consumer cost overrides (so fused conversion paths are accounted for).
///
/// # When to use
///
/// Use this when the current pixel format doesn't represent the data's
/// true precision. For example, a JPEG image (u8 sRGB) decoded into f32
/// for gamma-correct resize: the data is *currently* f32, but its origin
/// precision is u8. Converting back to u8 for JPEG encoding is lossless
/// (within ±1 LSB), and the cost model should reflect that.
///
/// ```rust,ignore
/// let provenance = Provenance::with_origin_depth(ChannelType::U8);
/// let target = negotiate(
///     current_f32_desc,
///     provenance,
///     encoder_options.iter().copied(),
///     ConvertIntent::Fastest,
/// );
/// ```
pub fn negotiate(
    source: PixelDescriptor,
    provenance: Provenance,
    options: impl Iterator<Item = FormatOption>,
    intent: ConvertIntent,
) -> Option<PixelDescriptor> {
    best_of(
        source,
        provenance,
        options.map(|o| (o.descriptor, o.consumer_cost)),
        intent,
    )
}

/// Recommend the ideal working format for a given intent, based on the source format.
///
/// Unlike [`best_match`], this isn't constrained to a list — it returns what
/// the consumer *should* be working in for optimal results.
///
/// Key principles:
/// - **Fastest** preserves the source format (identity).
/// - **LinearLight** upgrades to f32 Linear for gamma-correct operations.
/// - **Blend** upgrades to f32 Linear with premultiplied alpha.
/// - **Perceptual** keeps u8 sRGB for SDR-8 sources (LUT-fast), upgrades others to f32 sRGB.
/// - Never downgrades precision — a u16 source won't be recommended as u8.
pub fn ideal_format(source: PixelDescriptor, intent: ConvertIntent) -> PixelDescriptor {
    match intent {
        ConvertIntent::Fastest => source,

        ConvertIntent::LinearLight => {
            if source.channel_type == ChannelType::F32
                && source.transfer == TransferFunction::Linear
            {
                return source;
            }
            PixelDescriptor::new(
                ChannelType::F32,
                source.layout,
                source.alpha,
                TransferFunction::Linear,
            )
        }

        ConvertIntent::Blend => {
            let alpha = if source.layout.has_alpha() {
                AlphaMode::Premultiplied
            } else {
                source.alpha
            };
            if source.channel_type == ChannelType::F32
                && source.transfer == TransferFunction::Linear
                && source.alpha == alpha
            {
                return source;
            }
            PixelDescriptor::new(ChannelType::F32, source.layout, alpha, TransferFunction::Linear)
        }

        ConvertIntent::Perceptual => {
            let tier = precision_tier(source);
            match tier {
                PrecisionTier::Sdr8 => {
                    if source.transfer == TransferFunction::Srgb
                        || source.transfer == TransferFunction::Unknown
                    {
                        return source;
                    }
                    PixelDescriptor::new(
                        ChannelType::U8,
                        source.layout,
                        source.alpha,
                        TransferFunction::Srgb,
                    )
                }
                _ => PixelDescriptor::new(
                    ChannelType::F32,
                    source.layout,
                    source.alpha,
                    TransferFunction::Srgb,
                ),
            }
        }
    }
}

/// Compute the two-axis conversion cost for `from` → `to`.
///
/// This is the cost of *our* conversion kernels — it doesn't include
/// any consumer-side cost. Intent-independent.
///
/// Provenance is inferred from `from` (conservative: assumes current
/// depth is the true origin). Use [`conversion_cost_with_provenance`]
/// when the data has been widened from a lower-precision source.
pub fn conversion_cost(from: PixelDescriptor, to: PixelDescriptor) -> ConversionCost {
    conversion_cost_with_provenance(from, to, Provenance::from_source(from))
}

/// Compute the two-axis conversion cost with explicit provenance.
///
/// Like [`conversion_cost`], but uses the provided [`Provenance`] to
/// determine whether a depth narrowing is actually lossy. For example,
/// `f32 → u8` reports zero loss when `provenance.origin_depth == U8`,
/// because the data was originally u8 and the round-trip is lossless.
pub fn conversion_cost_with_provenance(
    from: PixelDescriptor,
    to: PixelDescriptor,
    provenance: Provenance,
) -> ConversionCost {
    transfer_cost(from.transfer, to.transfer)
        + depth_cost(from.channel_type, to.channel_type, provenance.origin_depth)
        + layout_cost(from.layout, to.layout)
        + alpha_cost(from.alpha, to.alpha, from.layout, to.layout)
}

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
    let our_cost = conversion_cost_with_provenance(source, target, provenance);
    let total_effort = our_cost.effort as u32 + consumer_cost.effort as u32;
    let total_loss = our_cost.loss as u32
        + consumer_cost.loss as u32
        + suitability_loss(target, intent) as u32;
    weighted_score(total_effort, total_loss, intent)
}

/// Find the best target from an iterator of (descriptor, consumer_cost) pairs.
fn best_of(
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
fn weighted_score(effort: u32, loss: u32, intent: ConvertIntent) -> u32 {
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
fn suitability_loss(target: PixelDescriptor, intent: ConvertIntent) -> u16 {
    match intent {
        ConvertIntent::Fastest => 0,
        ConvertIntent::LinearLight => linear_light_suitability(target),
        ConvertIntent::Blend => {
            let mut s = linear_light_suitability(target);
            // Straight alpha requires per-pixel division during compositing.
            if target.layout.has_alpha() && target.alpha == AlphaMode::Straight {
                s += 25;
            }
            s
        }
        ConvertIntent::Perceptual => perceptual_suitability(target),
    }
}

/// Suitability penalty for LinearLight operations (resize, blur).
/// f32 Linear is ideal; u8 gamma-encoded is worst.
fn linear_light_suitability(target: PixelDescriptor) -> u16 {
    match target.channel_type {
        ChannelType::F32 => {
            if target.transfer == TransferFunction::Linear {
                0
            } else {
                15
            }
        }
        ChannelType::U16 => 25,
        ChannelType::U8 => 40,
        _ => 50,
    }
}

/// Suitability penalty for perceptual operations (sharpening, color grading).
/// sRGB-encoded data is ideal; Linear f32 is slightly off.
fn perceptual_suitability(target: PixelDescriptor) -> u16 {
    if target.channel_type == ChannelType::F32 && target.transfer == TransferFunction::Linear {
        return 15;
    }
    if matches!(
        target.transfer,
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
fn transfer_cost(from: TransferFunction, to: TransferFunction) -> ConversionCost {
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
fn depth_cost(from: ChannelType, to: ChannelType, origin_depth: ChannelType) -> ConversionCost {
    if from == to {
        return ConversionCost::ZERO;
    }

    let effort = depth_effort(from, to);
    let loss = depth_loss(to, origin_depth);

    ConversionCost::new(effort, loss)
}

/// Computational effort for a depth conversion (independent of provenance).
fn depth_effort(from: ChannelType, to: ChannelType) -> u16 {
    match (from, to) {
        (ChannelType::U8, ChannelType::U16) | (ChannelType::U16, ChannelType::U8) => 10,
        (ChannelType::U16, ChannelType::F32) | (ChannelType::F32, ChannelType::U16) => 25,
        (ChannelType::U8, ChannelType::F32) | (ChannelType::F32, ChannelType::U8) => 40,
        _ => 100,
    }
}

/// Information loss when converting to `target_depth`, given the data's
/// `origin_depth`. If the target can represent the origin precision,
/// loss is zero.
fn depth_loss(target_depth: ChannelType, origin_depth: ChannelType) -> u16 {
    let target_bits = channel_bits(target_depth);
    let origin_bits = channel_bits(origin_depth);

    if target_bits >= origin_bits {
        // Target can hold everything the origin had — no loss.
        return 0;
    }

    // Target has less precision than the origin — lossy.
    match (origin_depth, target_depth) {
        (ChannelType::U16, ChannelType::U8) => 100,
        (ChannelType::F32, ChannelType::U8) => 300,
        (ChannelType::F32, ChannelType::U16) => 80,
        _ => 200,
    }
}

/// Nominal precision bits for a channel type (for ordering, not bit-exact).
fn channel_bits(ct: ChannelType) -> u16 {
    match ct {
        ChannelType::U8 => 8,
        ChannelType::U16 => 16,
        ChannelType::F32 => 32,
        _ => 0,
    }
}

/// Cost of layout conversion.
fn layout_cost(from: ChannelLayout, to: ChannelLayout) -> ConversionCost {
    if from == to {
        return ConversionCost::ZERO;
    }
    match (from, to) {
        // Swizzle: cheap, lossless.
        (ChannelLayout::Bgra, ChannelLayout::Rgba)
        | (ChannelLayout::Rgba, ChannelLayout::Bgra) => ConversionCost::new(5, 0),

        // Add alpha: cheap, lossless (fill opaque).
        (ChannelLayout::Rgb, ChannelLayout::Rgba)
        | (ChannelLayout::Rgb, ChannelLayout::Bgra) => ConversionCost::new(10, 0),

        // Drop alpha: cheap but lossy (alpha channel destroyed).
        (ChannelLayout::Rgba, ChannelLayout::Rgb)
        | (ChannelLayout::Bgra, ChannelLayout::Rgb) => ConversionCost::new(15, 50),

        // Gray → RGB: replicate, lossless.
        (ChannelLayout::Gray, ChannelLayout::Rgb) => ConversionCost::new(8, 0),
        (ChannelLayout::Gray, ChannelLayout::Rgba)
        | (ChannelLayout::Gray, ChannelLayout::Bgra) => ConversionCost::new(10, 0),

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

        _ => ConversionCost::new(100, 500),
    }
}

/// Cost of alpha mode conversion.
fn alpha_cost(
    from_alpha: AlphaMode,
    to_alpha: AlphaMode,
    from_layout: ChannelLayout,
    to_layout: ChannelLayout,
) -> ConversionCost {
    if !to_layout.has_alpha() || !from_layout.has_alpha() || from_alpha == to_alpha {
        return ConversionCost::ZERO;
    }
    match (from_alpha, to_alpha) {
        // Straight → Premul: per-pixel multiply, tiny rounding loss.
        (AlphaMode::Straight, AlphaMode::Premultiplied) => ConversionCost::new(20, 5),
        // Premul → Straight: per-pixel divide, worse rounding at low alpha.
        (AlphaMode::Premultiplied, AlphaMode::Straight) => ConversionCost::new(25, 10),
        (AlphaMode::None, _) | (_, AlphaMode::None) => ConversionCost::ZERO,
        _ => ConversionCost::ZERO,
    }
}

// ---------------------------------------------------------------------------
// Precision tier (used by ideal_format only)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum PrecisionTier {
    Sdr8 = 0,
    Sdr16 = 1,
    LinearF32 = 2,
    Hdr = 3,
}

fn precision_tier(desc: PixelDescriptor) -> PrecisionTier {
    if matches!(
        desc.transfer,
        TransferFunction::Pq | TransferFunction::Hlg
    ) {
        return PrecisionTier::Hdr;
    }
    match desc.channel_type {
        ChannelType::U8 => PrecisionTier::Sdr8,
        ChannelType::U16 => PrecisionTier::Sdr16,
        ChannelType::F32 => PrecisionTier::LinearF32,
        _ => PrecisionTier::Sdr8,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_wins() {
        let src = PixelDescriptor::RGB8_SRGB;
        let supported = &[PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGB8_SRGB];
        assert_eq!(
            best_match(src, supported, ConvertIntent::Fastest),
            Some(PixelDescriptor::RGB8_SRGB)
        );
    }

    #[test]
    fn empty_list_returns_none() {
        let src = PixelDescriptor::RGB8_SRGB;
        assert_eq!(best_match(src, &[], ConvertIntent::Fastest), None);
    }

    #[test]
    fn prefers_same_depth_over_cross_depth() {
        let src = PixelDescriptor::RGB8_SRGB;
        let supported = &[PixelDescriptor::RGBF32_LINEAR, PixelDescriptor::RGBA8_SRGB];
        assert_eq!(
            best_match(src, supported, ConvertIntent::Fastest),
            Some(PixelDescriptor::RGBA8_SRGB)
        );
    }

    #[test]
    fn bgra_rgba_swizzle_is_cheap() {
        let src = PixelDescriptor::BGRA8_SRGB;
        let supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB];
        assert_eq!(
            best_match(src, supported, ConvertIntent::Fastest),
            Some(PixelDescriptor::RGBA8_SRGB)
        );
    }

    #[test]
    fn gray_to_rgb_preferred_over_rgba() {
        let src = PixelDescriptor::GRAY8_SRGB;
        let supported = &[PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGB8_SRGB];
        assert_eq!(
            best_match(src, supported, ConvertIntent::Fastest),
            Some(PixelDescriptor::RGB8_SRGB)
        );
    }

    #[test]
    fn transfer_only_diff_is_cheap() {
        let src = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Rgb,
            AlphaMode::None,
            TransferFunction::Unknown,
        );
        let target = PixelDescriptor::RGB8_SRGB;
        let supported = &[target, PixelDescriptor::RGBF32_LINEAR];
        assert_eq!(
            best_match(src, supported, ConvertIntent::Fastest),
            Some(target)
        );
    }

    // Two-axis cost tests.

    #[test]
    fn conversion_cost_identity_is_zero() {
        let cost = conversion_cost(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGB8_SRGB);
        assert_eq!(cost, ConversionCost::ZERO);
    }

    #[test]
    fn widening_has_zero_loss() {
        let cost = conversion_cost(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBF32_LINEAR);
        assert_eq!(cost.loss, 0);
        assert!(cost.effort > 0);
    }

    #[test]
    fn narrowing_has_nonzero_loss() {
        let cost = conversion_cost(PixelDescriptor::RGBF32_LINEAR, PixelDescriptor::RGB8_SRGB);
        assert!(cost.loss > 0, "f32→u8 should report data loss");
        assert!(cost.effort > 0);
    }

    #[test]
    fn consumer_override_shifts_preference() {
        // Source is f32 Linear. Without consumer cost, we'd need to convert to u8.
        // With a consumer that accepts f32 cheaply, we skip our conversion.
        let src = PixelDescriptor::RGBF32_LINEAR;
        let options = &[
            FormatOption::from(PixelDescriptor::RGB8_SRGB),
            FormatOption::with_cost(PixelDescriptor::RGBF32_LINEAR, ConversionCost::new(5, 0)),
        ];
        // Even Fastest should prefer the zero-conversion path with low consumer cost.
        assert_eq!(
            best_match_with(src, options, ConvertIntent::Fastest),
            Some(PixelDescriptor::RGBF32_LINEAR)
        );
    }
}
