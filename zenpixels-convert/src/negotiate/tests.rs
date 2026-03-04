//! Tests for format negotiation.

use super::*;
use crate::ChannelLayout;

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
        None,
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
