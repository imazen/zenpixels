//! Format negotiation — pick the cheapest conversion target.

use zencodec_types::{AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};

/// Pick the cheapest conversion target from `supported` for the given source.
///
/// Returns `None` only if `supported` is empty.
///
/// Preference order:
/// 1. Exact match (identical descriptor)
/// 2. Transfer-only difference (same layout + depth, different transfer)
/// 3. Same depth class, compatible layout (e.g. Rgb→Rgba add alpha, Bgra→Rgba swizzle)
/// 4. Cross-depth conversion (e.g. f32→u8)
///
/// Within each tier, formats that preserve more information score lower cost.
pub fn best_match(source: PixelDescriptor, supported: &[PixelDescriptor]) -> Option<PixelDescriptor> {
    if supported.is_empty() {
        return None;
    }

    let mut best: Option<(PixelDescriptor, u32)> = None;

    for &target in supported {
        let cost = conversion_cost(source, target);
        match best {
            Some((_, best_cost)) if cost < best_cost => best = Some((target, cost)),
            None => best = Some((target, cost)),
            _ => {}
        }
    }

    best.map(|(desc, _)| desc)
}

/// Compute a cost score for converting `from` → `to`.
///
/// Lower is cheaper. The cost model uses tiers:
/// - 0: Identity (exact match)
/// - 1-9: Transfer-only difference
/// - 10-99: Same depth, layout change
/// - 100-999: Cross-depth conversion
/// - 1000+: Lossy conversions (alpha drop, color→gray)
fn conversion_cost(from: PixelDescriptor, to: PixelDescriptor) -> u32 {
    // Exact match.
    if from == to {
        return 0;
    }

    let mut cost = 0u32;

    // Transfer function cost.
    cost += transfer_cost(from.transfer, to.transfer);

    // Channel type (depth) cost.
    cost += depth_cost(from.channel_type, to.channel_type);

    // Layout cost.
    cost += layout_cost(from.layout, to.layout);

    // Alpha mode cost.
    cost += alpha_cost(from.alpha, to.alpha, from.layout, to.layout);

    cost
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
        assert_eq!(best_match(src, supported), Some(PixelDescriptor::RGB8_SRGB));
    }

    #[test]
    fn empty_list_returns_none() {
        let src = PixelDescriptor::RGB8_SRGB;
        assert_eq!(best_match(src, &[]), None);
    }

    #[test]
    fn prefers_same_depth_over_cross_depth() {
        let src = PixelDescriptor::RGB8_SRGB;
        let supported = &[PixelDescriptor::RGBF32_LINEAR, PixelDescriptor::RGBA8_SRGB];
        // RGBA8 should win over RGBF32 because same depth class.
        assert_eq!(
            best_match(src, supported),
            Some(PixelDescriptor::RGBA8_SRGB)
        );
    }

    #[test]
    fn bgra_rgba_swizzle_is_cheap() {
        let src = PixelDescriptor::BGRA8_SRGB;
        let supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB];
        // RGBA8 should win (swizzle) over RGB8 (swizzle + drop alpha).
        assert_eq!(
            best_match(src, supported),
            Some(PixelDescriptor::RGBA8_SRGB)
        );
    }

    #[test]
    fn gray_to_rgb_preferred_over_rgba() {
        let src = PixelDescriptor::GRAY8_SRGB;
        let supported = &[PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGB8_SRGB];
        // RGB8 is cheaper (no alpha to add).
        assert_eq!(
            best_match(src, supported),
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
        assert_eq!(best_match(src, supported), Some(target));
    }
}
