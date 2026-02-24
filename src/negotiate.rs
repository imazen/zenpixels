//! Format negotiation — pick the cheapest conversion target.

use zencodec_types::{AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};

/// What the consumer plans to do with the converted pixels.
///
/// Shifts the format negotiation cost model to prefer formats
/// suited for the intended operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConvertIntent {
    /// Minimize conversion cost. Good for encoding — the codec
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
    /// Prefers f32 in sRGB space (perceptually uniform).
    Perceptual,
}

/// Pick the best conversion target from `supported` for the given source,
/// considering what the consumer intends to do with the pixels.
///
/// Returns `None` only if `supported` is empty.
///
/// Preference order (modified by intent):
/// 1. Exact match (identical descriptor)
/// 2. Transfer-only difference (same layout + depth, different transfer)
/// 3. Same depth class, compatible layout (e.g. Rgb→Rgba add alpha, Bgra→Rgba swizzle)
/// 4. Cross-depth conversion (e.g. f32→u8)
///
/// Within each tier, formats that preserve more information score lower cost.
/// The `intent` shifts preferences — e.g. `LinearLight` prefers f32 Linear
/// targets even when u8 sRGB would be cheaper conversion-wise.
pub fn best_match(
    source: PixelDescriptor,
    supported: &[PixelDescriptor],
    intent: ConvertIntent,
) -> Option<PixelDescriptor> {
    if supported.is_empty() {
        return None;
    }

    let mut best: Option<(PixelDescriptor, u32)> = None;

    for &target in supported {
        let cost = conversion_cost(source, target, intent);
        match best {
            Some((_, best_cost)) if cost < best_cost => best = Some((target, cost)),
            None => best = Some((target, cost)),
            _ => {}
        }
    }

    best.map(|(desc, _)| desc)
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
            // Upgrade to f32 Linear, preserving layout.
            if source.channel_type == ChannelType::F32
                && source.transfer == TransferFunction::Linear
            {
                // Already f32 linear — keep as-is.
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
            // f32 Linear, with premultiplied alpha if source has alpha.
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
                    // u8 sRGB is optimal for SDR-8 perceptual ops (256-entry LUTs).
                    if source.transfer == TransferFunction::Srgb
                        || source.transfer == TransferFunction::Unknown
                    {
                        return source;
                    }
                    // BT.709 u8 → treat as sRGB (close enough for perceptual).
                    PixelDescriptor::new(
                        ChannelType::U8,
                        source.layout,
                        source.alpha,
                        TransferFunction::Srgb,
                    )
                }
                _ => {
                    // Higher precision sources → f32 sRGB for perceptual uniformity.
                    PixelDescriptor::new(
                        ChannelType::F32,
                        source.layout,
                        source.alpha,
                        TransferFunction::Srgb,
                    )
                }
            }
        }
    }
}

/// Compute a cost score for converting `from` → `to`, adjusted by intent.
///
/// Lower is cheaper. The base cost model uses tiers:
/// - 0: Identity (exact match)
/// - 1-9: Transfer-only difference
/// - 10-99: Same depth, layout change
/// - 100-999: Cross-depth conversion
/// - 1000+: Lossy conversions (alpha drop, color→gray)
///
/// Intent bonuses/penalties shift preferences without breaking tier boundaries
/// for unrelated conversions:
/// - `LinearLight`: f32 Linear gets bonus, u8 sRGB gets penalty
/// - `Blend`: same as LinearLight, plus premultiplied alpha bonus
/// - `Perceptual`: sRGB f32 gets bonus, u8 sRGB stays cheap for SDR-8
/// - Precision loss (e.g. u16→u8, HDR→SDR) is penalized in all intents
fn conversion_cost(from: PixelDescriptor, to: PixelDescriptor, intent: ConvertIntent) -> u32 {
    let mut cost = 0u32;

    // Base costs (all zero for exact match).
    cost += transfer_cost(from.transfer, to.transfer);
    cost += depth_cost(from.channel_type, to.channel_type);
    cost += layout_cost(from.layout, to.layout);
    cost += alpha_cost(from.alpha, to.alpha, from.layout, to.layout);

    // Precision loss penalty (applies to all intents).
    cost += precision_loss_penalty(from, to);

    // Intent-specific adjustments. Applied even for identity conversions
    // so that e.g. Perceptual can penalize f32 Linear in favor of f32 sRGB.
    cost = apply_intent_adjustment(cost, from, to, intent);

    cost
}

/// Precision tier of a source descriptor.
///
/// Higher tiers require more precision to represent faithfully.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum PrecisionTier {
    /// u8 + sRGB/BT.709/Unknown — standard 8-bit content.
    Sdr8 = 0,
    /// u16 + sRGB/BT.709 — high-precision SDR.
    Sdr16 = 1,
    /// f32 + Linear + sRGB gamut — working-space float.
    LinearF32 = 2,
    /// PQ/HLG or Rec.2020 gamut — HDR content.
    Hdr = 3,
}

/// Classify a descriptor into its precision tier.
fn precision_tier(desc: PixelDescriptor) -> PrecisionTier {
    // HDR transfer functions → HDR tier regardless of depth.
    if matches!(desc.transfer, TransferFunction::Pq | TransferFunction::Hlg) {
        return PrecisionTier::Hdr;
    }

    match desc.channel_type {
        ChannelType::U8 => PrecisionTier::Sdr8,
        ChannelType::U16 => PrecisionTier::Sdr16,
        // f32 is always high precision — changing transfer function (sRGB↔Linear)
        // doesn't lose data when the bit depth is 32-bit float.
        ChannelType::F32 => PrecisionTier::LinearF32,
        _ => PrecisionTier::Sdr8,
    }
}

/// Penalty for converting to a lower precision tier.
fn precision_loss_penalty(from: PixelDescriptor, to: PixelDescriptor) -> u32 {
    let src_tier = precision_tier(from);
    let dst_tier = precision_tier(to);

    if dst_tier >= src_tier {
        return 0;
    }

    // Scale penalty by how many tiers of precision we'd lose.
    let tier_gap = src_tier as u32 - dst_tier as u32;
    match tier_gap {
        1 => 200,  // e.g. u16→u8 or f32 linear→u16
        2 => 600,  // e.g. f32 linear→u8
        3 => 1200, // e.g. HDR→u8
        _ => 1500,
    }
}

/// Apply intent-specific cost adjustments.
///
/// Returns adjusted cost (saturating at 0 — costs never go negative).
fn apply_intent_adjustment(
    base_cost: u32,
    _from: PixelDescriptor,
    to: PixelDescriptor,
    intent: ConvertIntent,
) -> u32 {
    match intent {
        ConvertIntent::Fastest => base_cost,

        ConvertIntent::LinearLight => {
            let mut cost = base_cost;
            // Bonus for f32 Linear targets.
            if to.channel_type == ChannelType::F32 && to.transfer == TransferFunction::Linear {
                cost = cost.saturating_sub(40);
            }
            // Penalty for u8 targets (gamma-incorrect for resize/blur).
            if to.channel_type == ChannelType::U8 {
                cost = cost.saturating_add(40);
            }
            cost
        }

        ConvertIntent::Blend => {
            let mut cost = base_cost;
            // Same linear preference as LinearLight.
            if to.channel_type == ChannelType::F32 && to.transfer == TransferFunction::Linear {
                cost = cost.saturating_sub(40);
            }
            if to.channel_type == ChannelType::U8 {
                cost = cost.saturating_add(40);
            }
            // Bonus for premultiplied alpha.
            if to.layout.has_alpha() && to.alpha == AlphaMode::Premultiplied {
                cost = cost.saturating_sub(20);
            }
            // Penalty for straight alpha when target has alpha.
            if to.layout.has_alpha() && to.alpha == AlphaMode::Straight {
                cost = cost.saturating_add(20);
            }
            cost
        }

        ConvertIntent::Perceptual => {
            let mut cost = base_cost;
            // Bonus for f32 sRGB targets (perceptually uniform working space).
            if to.channel_type == ChannelType::F32
                && matches!(to.transfer, TransferFunction::Srgb | TransferFunction::Unknown)
            {
                cost = cost.saturating_sub(30);
            }
            // Linear f32 is slightly less ideal for perceptual ops.
            if to.channel_type == ChannelType::F32 && to.transfer == TransferFunction::Linear {
                cost = cost.saturating_add(10);
            }
            cost
        }
    }
}

/// Cost of transfer function conversion.
fn transfer_cost(from: TransferFunction, to: TransferFunction) -> u32 {
    if from == to {
        return 0;
    }

    match (from, to) {
        // Unknown can be treated as "whatever the target wants" — no actual math.
        (TransferFunction::Unknown, _) | (_, TransferFunction::Unknown) => 1,

        // sRGB ↔ Linear is the standard path, well-optimized.
        (TransferFunction::Srgb, TransferFunction::Linear)
        | (TransferFunction::Linear, TransferFunction::Srgb) => 3,

        // BT.709 is close to sRGB — similar cost.
        (TransferFunction::Bt709, TransferFunction::Srgb)
        | (TransferFunction::Srgb, TransferFunction::Bt709)
        | (TransferFunction::Bt709, TransferFunction::Linear)
        | (TransferFunction::Linear, TransferFunction::Bt709) => 5,

        // PQ/HLG are expensive (not yet supported).
        _ => 500,
    }
}

/// Cost of channel depth conversion.
fn depth_cost(from: ChannelType, to: ChannelType) -> u32 {
    if from == to {
        return 0;
    }

    match (from, to) {
        // u8 ↔ u16 is cheap integer math.
        (ChannelType::U8, ChannelType::U16) | (ChannelType::U16, ChannelType::U8) => 10,

        // u8 ↔ f32 involves transfer function (sRGB EOTF/OETF).
        (ChannelType::U8, ChannelType::F32) | (ChannelType::F32, ChannelType::U8) => 50,

        // u16 ↔ f32 is moderate.
        (ChannelType::U16, ChannelType::F32) | (ChannelType::F32, ChannelType::U16) => 30,

        _ => 100,
    }
}

/// Cost of layout conversion.
fn layout_cost(from: ChannelLayout, to: ChannelLayout) -> u32 {
    if from == to {
        return 0;
    }

    match (from, to) {
        // Bgra ↔ Rgba is a cheap swizzle (same channel count).
        (ChannelLayout::Bgra, ChannelLayout::Rgba)
        | (ChannelLayout::Rgba, ChannelLayout::Bgra) => 5,

        // Add alpha (Rgb → Rgba): cheap, just set alpha=255.
        (ChannelLayout::Rgb, ChannelLayout::Rgba)
        | (ChannelLayout::Rgb, ChannelLayout::Bgra) => 15,

        // Drop alpha (Rgba → Rgb): cheap copy.
        (ChannelLayout::Rgba, ChannelLayout::Rgb)
        | (ChannelLayout::Bgra, ChannelLayout::Rgb) => 20,

        // Gray → Rgb: replicate channel.
        (ChannelLayout::Gray, ChannelLayout::Rgb) => 10,
        (ChannelLayout::Gray, ChannelLayout::Rgba)
        | (ChannelLayout::Gray, ChannelLayout::Bgra) => 15,

        // Rgb → Gray: luma calculation.
        (ChannelLayout::Rgb, ChannelLayout::Gray)
        | (ChannelLayout::Rgba, ChannelLayout::Gray)
        | (ChannelLayout::Bgra, ChannelLayout::Gray) => 1000,

        // GrayAlpha ↔ Rgba.
        (ChannelLayout::GrayAlpha, ChannelLayout::Rgba)
        | (ChannelLayout::GrayAlpha, ChannelLayout::Bgra) => 20,
        (ChannelLayout::Rgba, ChannelLayout::GrayAlpha)
        | (ChannelLayout::Bgra, ChannelLayout::GrayAlpha) => 1000,

        // Gray ↔ GrayAlpha.
        (ChannelLayout::Gray, ChannelLayout::GrayAlpha) => 10,
        (ChannelLayout::GrayAlpha, ChannelLayout::Gray) => 15,

        // GrayAlpha → Rgb.
        (ChannelLayout::GrayAlpha, ChannelLayout::Rgb) => 15,

        // Anything else is expensive.
        _ => 2000,
    }
}

/// Cost of alpha mode conversion.
fn alpha_cost(
    from_alpha: AlphaMode,
    to_alpha: AlphaMode,
    from_layout: ChannelLayout,
    to_layout: ChannelLayout,
) -> u32 {
    // If target has no alpha, no alpha cost.
    if !to_layout.has_alpha() {
        return 0;
    }
    // If source has no alpha but target does, we'll fill with opaque — cheap.
    if !from_layout.has_alpha() {
        return 0;
    }

    if from_alpha == to_alpha {
        return 0;
    }

    match (from_alpha, to_alpha) {
        // Straight ↔ Premultiplied requires per-pixel multiply/divide.
        (AlphaMode::Straight, AlphaMode::Premultiplied)
        | (AlphaMode::Premultiplied, AlphaMode::Straight) => 30,

        // None → Straight/Premul is just "fill opaque" — already handled above.
        (AlphaMode::None, _) | (_, AlphaMode::None) => 0,

        _ => 0,
    }
}

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
        // RGBA8 should win over RGBF32 because same depth class.
        assert_eq!(
            best_match(src, supported, ConvertIntent::Fastest),
            Some(PixelDescriptor::RGBA8_SRGB)
        );
    }

    #[test]
    fn bgra_rgba_swizzle_is_cheap() {
        let src = PixelDescriptor::BGRA8_SRGB;
        let supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB];
        // RGBA8 should win (swizzle) over RGB8 (swizzle + drop alpha).
        assert_eq!(
            best_match(src, supported, ConvertIntent::Fastest),
            Some(PixelDescriptor::RGBA8_SRGB)
        );
    }

    #[test]
    fn gray_to_rgb_preferred_over_rgba() {
        let src = PixelDescriptor::GRAY8_SRGB;
        let supported = &[PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGB8_SRGB];
        // RGB8 is cheaper (no alpha to add).
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
        // RGB8_SRGB (transfer-only) should beat RGBF32 (cross-depth).
        assert_eq!(
            best_match(src, supported, ConvertIntent::Fastest),
            Some(target)
        );
    }
}
