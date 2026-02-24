//! Tests for format negotiation preference ordering.

use zencodec_types::{
    AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction,
};
use zenpixels::{best_match, ideal_format, ConvertIntent};

// =========================================================================
// Existing tests (using ConvertIntent::Fastest — behavior preserved)
// =========================================================================

#[test]
fn exact_match_beats_everything() {
    let src = PixelDescriptor::RGBA8_SRGB;
    let supported = &[
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGBF32_LINEAR,
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::BGRA8_SRGB,
    ];
    assert_eq!(
        best_match(src, supported, ConvertIntent::Fastest),
        Some(PixelDescriptor::RGBA8_SRGB)
    );
}

#[test]
fn transfer_only_diff_over_layout_change() {
    let src = PixelDescriptor::RGB8;
    let supported = &[PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGB8_SRGB];
    assert_eq!(
        best_match(src, supported, ConvertIntent::Fastest),
        Some(PixelDescriptor::RGB8_SRGB)
    );
}

#[test]
fn same_depth_over_cross_depth() {
    let src = PixelDescriptor::RGB8_SRGB;
    let supported = &[PixelDescriptor::RGBF32_LINEAR, PixelDescriptor::RGBA8_SRGB];
    assert_eq!(
        best_match(src, supported, ConvertIntent::Fastest),
        Some(PixelDescriptor::RGBA8_SRGB)
    );
}

#[test]
fn swizzle_cheaper_than_drop_alpha() {
    let src = PixelDescriptor::BGRA8_SRGB;
    let supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB];
    assert_eq!(
        best_match(src, supported, ConvertIntent::Fastest),
        Some(PixelDescriptor::RGBA8_SRGB)
    );
}

#[test]
fn gray_prefers_rgb_over_rgba() {
    let src = PixelDescriptor::GRAY8_SRGB;
    let supported = &[PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGB8_SRGB];
    assert_eq!(
        best_match(src, supported, ConvertIntent::Fastest),
        Some(PixelDescriptor::RGB8_SRGB)
    );
}

#[test]
fn f32_linear_prefers_u16_over_u8_fastest() {
    let src = PixelDescriptor::RGBF32_LINEAR;
    let desc_u16 = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Srgb,
    );
    let supported = &[desc_u16, PixelDescriptor::RGB8_SRGB];
    // u16 should be cheaper (less precision loss).
    assert_eq!(
        best_match(src, supported, ConvertIntent::Fastest),
        Some(desc_u16)
    );
}

#[test]
fn empty_returns_none() {
    assert_eq!(
        best_match(PixelDescriptor::RGB8_SRGB, &[], ConvertIntent::Fastest),
        None
    );
}

#[test]
fn single_option_always_selected() {
    let src = PixelDescriptor::GRAY8_SRGB;
    let supported = &[PixelDescriptor::RGBAF32_LINEAR];
    assert_eq!(
        best_match(src, supported, ConvertIntent::Fastest),
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

    assert_eq!(
        best_match(PixelDescriptor::RGBA8_SRGB, jpeg_supported, ConvertIntent::Fastest),
        Some(PixelDescriptor::RGB8_SRGB)
    );
    assert_eq!(
        best_match(PixelDescriptor::BGRA8_SRGB, jpeg_supported, ConvertIntent::Fastest),
        Some(PixelDescriptor::RGB8_SRGB)
    );
    assert_eq!(
        best_match(PixelDescriptor::GRAY8_SRGB, jpeg_supported, ConvertIntent::Fastest),
        Some(PixelDescriptor::GRAY8_SRGB)
    );
    assert_eq!(
        best_match(PixelDescriptor::RGBF32_LINEAR, jpeg_supported, ConvertIntent::Fastest),
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

    assert_eq!(
        best_match(PixelDescriptor::BGRA8_SRGB, webp_supported, ConvertIntent::Fastest),
        Some(PixelDescriptor::RGBA8_SRGB)
    );
    assert_eq!(
        best_match(PixelDescriptor::GRAY8_SRGB, webp_supported, ConvertIntent::Fastest),
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

    assert_eq!(
        best_match(PixelDescriptor::RGB8_SRGB, png_supported, ConvertIntent::Fastest),
        Some(PixelDescriptor::RGB8_SRGB)
    );
    assert_eq!(
        best_match(PixelDescriptor::GRAY8_SRGB, png_supported, ConvertIntent::Fastest),
        Some(PixelDescriptor::GRAY8_SRGB)
    );
}

// =========================================================================
// Intent-aware best_match tests
// =========================================================================

#[test]
fn linear_light_prefers_f32_linear() {
    let src = PixelDescriptor::RGB8_SRGB;
    let supported = &[PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGBAF32_LINEAR];
    // LinearLight should prefer f32 Linear for gamma-correct resize.
    assert_eq!(
        best_match(src, supported, ConvertIntent::LinearLight),
        Some(PixelDescriptor::RGBAF32_LINEAR)
    );
}

#[test]
fn fastest_prefers_same_depth() {
    // Same scenario as above — Fastest should prefer staying in u8.
    let src = PixelDescriptor::RGB8_SRGB;
    let supported = &[PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGBAF32_LINEAR];
    assert_eq!(
        best_match(src, supported, ConvertIntent::Fastest),
        Some(PixelDescriptor::RGBA8_SRGB)
    );
}

#[test]
fn blend_prefers_premultiplied() {
    let src = PixelDescriptor::RGBA8_SRGB;
    let straight_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        AlphaMode::Straight,
        TransferFunction::Linear,
    );
    let premul_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        AlphaMode::Premultiplied,
        TransferFunction::Linear,
    );
    let supported = &[straight_f32, premul_f32];
    // Blend should prefer premultiplied.
    assert_eq!(
        best_match(src, supported, ConvertIntent::Blend),
        Some(premul_f32)
    );
}

#[test]
fn perceptual_prefers_srgb_f32() {
    let src = PixelDescriptor::RGBF32_LINEAR;
    let f32_srgb = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Srgb,
    );
    let supported = &[PixelDescriptor::RGBF32_LINEAR, f32_srgb];
    // Perceptual should prefer sRGB f32 over Linear f32.
    assert_eq!(
        best_match(src, supported, ConvertIntent::Perceptual),
        Some(f32_srgb)
    );
}

// =========================================================================
// Precision preservation tests
// =========================================================================

#[test]
fn u16_source_penalizes_u8_target() {
    let src = PixelDescriptor::RGB16_SRGB;
    let rgb16 = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Srgb,
    );
    let supported = &[PixelDescriptor::RGB8_SRGB, rgb16];
    // Any intent should prefer RGB16 to avoid truncation.
    assert_eq!(
        best_match(src, supported, ConvertIntent::Fastest),
        Some(rgb16)
    );
    assert_eq!(
        best_match(src, supported, ConvertIntent::LinearLight),
        Some(rgb16)
    );
}

#[test]
fn hdr_source_penalizes_sdr_target() {
    let src = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Pq,
    );
    let supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBF32_LINEAR];
    // HDR source → f32 Linear is much less lossy than u8 sRGB.
    assert_eq!(
        best_match(src, supported, ConvertIntent::Fastest),
        Some(PixelDescriptor::RGBF32_LINEAR)
    );
}

#[test]
fn sdr8_allows_u8_fast_path() {
    let src = PixelDescriptor::RGB8_SRGB;
    let supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBF32_LINEAR];
    // SDR-8 + Perceptual: u8 sRGB is fine (LUT-fast), should be preferred.
    assert_eq!(
        best_match(src, supported, ConvertIntent::Perceptual),
        Some(PixelDescriptor::RGB8_SRGB)
    );
}

// =========================================================================
// ideal_format tests
// =========================================================================

#[test]
fn ideal_format_fastest_identity() {
    // Fastest preserves format as-is for all source types.
    assert_eq!(
        ideal_format(PixelDescriptor::RGB8_SRGB, ConvertIntent::Fastest),
        PixelDescriptor::RGB8_SRGB
    );
    assert_eq!(
        ideal_format(PixelDescriptor::RGBF32_LINEAR, ConvertIntent::Fastest),
        PixelDescriptor::RGBF32_LINEAR
    );
    let pq = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Pq,
    );
    assert_eq!(ideal_format(pq, ConvertIntent::Fastest), pq);
}

#[test]
fn ideal_format_linear_light_sdr() {
    // u8 sRGB → f32 Linear.
    let result = ideal_format(PixelDescriptor::RGB8_SRGB, ConvertIntent::LinearLight);
    assert_eq!(result.channel_type, ChannelType::F32);
    assert_eq!(result.transfer, TransferFunction::Linear);
    assert_eq!(result.layout, ChannelLayout::Rgb);
}

#[test]
fn ideal_format_linear_light_already_f32_linear() {
    // Already f32 Linear → keep as-is.
    assert_eq!(
        ideal_format(PixelDescriptor::RGBF32_LINEAR, ConvertIntent::LinearLight),
        PixelDescriptor::RGBF32_LINEAR
    );
}

#[test]
fn ideal_format_linear_light_hdr() {
    // f32 PQ → f32 Linear (not HDR passthrough; we convert transfer).
    let pq = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Pq,
    );
    let result = ideal_format(pq, ConvertIntent::LinearLight);
    assert_eq!(result.channel_type, ChannelType::F32);
    assert_eq!(result.transfer, TransferFunction::Linear);
}

#[test]
fn ideal_format_blend_adds_premul() {
    // u8 sRGB straight alpha → f32 Linear premul.
    let result = ideal_format(PixelDescriptor::RGBA8_SRGB, ConvertIntent::Blend);
    assert_eq!(result.channel_type, ChannelType::F32);
    assert_eq!(result.transfer, TransferFunction::Linear);
    assert_eq!(result.alpha, AlphaMode::Premultiplied);
    assert_eq!(result.layout, ChannelLayout::Rgba);
}

#[test]
fn ideal_format_blend_no_alpha_source() {
    // RGB (no alpha) → f32 Linear, no premul needed.
    let result = ideal_format(PixelDescriptor::RGB8_SRGB, ConvertIntent::Blend);
    assert_eq!(result.channel_type, ChannelType::F32);
    assert_eq!(result.transfer, TransferFunction::Linear);
    assert_eq!(result.alpha, AlphaMode::None);
}

#[test]
fn ideal_format_perceptual_sdr8() {
    // u8 sRGB → keeps u8 sRGB (LUT-fast).
    assert_eq!(
        ideal_format(PixelDescriptor::RGB8_SRGB, ConvertIntent::Perceptual),
        PixelDescriptor::RGB8_SRGB
    );
}

#[test]
fn ideal_format_perceptual_u16() {
    // u16 sRGB → f32 sRGB (higher precision needs float).
    let result = ideal_format(PixelDescriptor::RGB16_SRGB, ConvertIntent::Perceptual);
    assert_eq!(result.channel_type, ChannelType::F32);
    assert_eq!(result.transfer, TransferFunction::Srgb);
}

#[test]
fn ideal_format_perceptual_hdr() {
    // f32 PQ → f32 sRGB (gamut clamp, but perceptually uniform).
    let pq = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        AlphaMode::None,
        TransferFunction::Pq,
    );
    let result = ideal_format(pq, ConvertIntent::Perceptual);
    assert_eq!(result.channel_type, ChannelType::F32);
    assert_eq!(result.transfer, TransferFunction::Srgb);
}
