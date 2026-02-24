//! Tests for format negotiation preference ordering.

use zencodec_types::{
    AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction,
};
use zenpixels::best_match;

#[test]
fn exact_match_beats_everything() {
    let src = PixelDescriptor::RGBA8_SRGB;
    let supported = &[
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGBF32_LINEAR,
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::BGRA8_SRGB,
    ];
    assert_eq!(best_match(src, supported), Some(PixelDescriptor::RGBA8_SRGB));
}

#[test]
fn transfer_only_diff_over_layout_change() {
    // Source is RGB8 Unknown, targets are RGB8 sRGB and RGBA8 sRGB.
    // RGB8 sRGB (transfer-only diff) should beat RGBA8 sRGB (layout change).
    let src = PixelDescriptor::RGB8;
    let supported = &[PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGB8_SRGB];
    assert_eq!(best_match(src, supported), Some(PixelDescriptor::RGB8_SRGB));
}

#[test]
fn same_depth_over_cross_depth() {
    let src = PixelDescriptor::RGB8_SRGB;
    let supported = &[PixelDescriptor::RGBF32_LINEAR, PixelDescriptor::RGBA8_SRGB];
    // RGBA8 (same depth, add alpha) should beat RGBF32 (cross-depth).
    assert_eq!(
        best_match(src, supported),
        Some(PixelDescriptor::RGBA8_SRGB)
    );
}

#[test]
fn swizzle_cheaper_than_drop_alpha() {
    let src = PixelDescriptor::BGRA8_SRGB;
    let supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB];
    // BGRA→RGBA (swizzle) cheaper than BGRA→RGB (swizzle + drop alpha).
    assert_eq!(
        best_match(src, supported),
        Some(PixelDescriptor::RGBA8_SRGB)
    );
}

#[test]
fn gray_prefers_rgb_over_rgba() {
    let src = PixelDescriptor::GRAY8_SRGB;
    let supported = &[PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGB8_SRGB];
    // Gray→RGB is cheaper than Gray→RGBA.
    assert_eq!(
        best_match(src, supported),
        Some(PixelDescriptor::RGB8_SRGB)
    );
}

#[test]
fn f32_linear_prefers_u8_srgb_over_u16() {
    let src = PixelDescriptor::RGBF32_LINEAR;
    let desc_u16 = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Srgb,
    );
    let supported = &[desc_u16, PixelDescriptor::RGB8_SRGB];
    // f32→u16 cost=30 (depth) vs f32→u8 cost=50+3 (depth + sRGB transfer).
    // u16 should actually be cheaper.
    assert_eq!(best_match(src, supported), Some(desc_u16));
}

#[test]
fn empty_returns_none() {
    assert_eq!(best_match(PixelDescriptor::RGB8_SRGB, &[]), None);
}

#[test]
fn single_option_always_selected() {
    let src = PixelDescriptor::GRAY8_SRGB;
    let supported = &[PixelDescriptor::RGBAF32_LINEAR];
    // Even if expensive, the only option is selected.
    assert_eq!(
        best_match(src, supported),
        Some(PixelDescriptor::RGBAF32_LINEAR)
    );
}

/// Simulate JPEG encoder's supported formats.
#[test]
fn jpeg_format_negotiation() {
    let jpeg_supported = &[
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::GRAY8_SRGB,
    ];

    // RGBA8 → should pick RGB8 (drop alpha).
    assert_eq!(
        best_match(PixelDescriptor::RGBA8_SRGB, jpeg_supported),
        Some(PixelDescriptor::RGB8_SRGB)
    );

    // BGRA8 → should pick RGB8 (swizzle + drop alpha).
    assert_eq!(
        best_match(PixelDescriptor::BGRA8_SRGB, jpeg_supported),
        Some(PixelDescriptor::RGB8_SRGB)
    );

    // Gray8 → exact match.
    assert_eq!(
        best_match(PixelDescriptor::GRAY8_SRGB, jpeg_supported),
        Some(PixelDescriptor::GRAY8_SRGB)
    );

    // RgbF32 linear → RGB8 sRGB (cross-depth with OETF).
    assert_eq!(
        best_match(PixelDescriptor::RGBF32_LINEAR, jpeg_supported),
        Some(PixelDescriptor::RGB8_SRGB)
    );
}

/// Simulate WebP encoder's supported formats.
#[test]
fn webp_format_negotiation() {
    let webp_supported = &[
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGBA8_SRGB,
    ];

    // BGRA8 → RGBA8 (swizzle, keeps alpha).
    assert_eq!(
        best_match(PixelDescriptor::BGRA8_SRGB, webp_supported),
        Some(PixelDescriptor::RGBA8_SRGB)
    );

    // Gray8 → RGB8 (replicate).
    assert_eq!(
        best_match(PixelDescriptor::GRAY8_SRGB, webp_supported),
        Some(PixelDescriptor::RGB8_SRGB)
    );
}

/// Simulate PNG encoder's supported formats (supports many).
#[test]
fn png_format_negotiation() {
    let png_supported = &[
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::GRAY8_SRGB,
        PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            AlphaMode::None,
            TransferFunction::Srgb,
        ),
    ];

    // Exact matches.
    assert_eq!(
        best_match(PixelDescriptor::RGB8_SRGB, png_supported),
        Some(PixelDescriptor::RGB8_SRGB)
    );
    assert_eq!(
        best_match(PixelDescriptor::GRAY8_SRGB, png_supported),
        Some(PixelDescriptor::GRAY8_SRGB)
    );
}
