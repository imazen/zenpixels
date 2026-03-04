//! Tests for format negotiation preference ordering.

use zenpixels_convert::{AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};
use zenpixels_convert::{
    ConversionCost, ConvertIntent, FormatOption, Provenance, best_match, best_match_with,
    conversion_cost, conversion_cost_with_provenance, ideal_format, negotiate,
};

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
        None,
        TransferFunction::Srgb,
    );
    let supported = &[desc_u16, PixelDescriptor::RGB8_SRGB];
    // u16 has less loss than u8.
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
    let jpeg_supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::GRAY8_SRGB];

    assert_eq!(
        best_match(
            PixelDescriptor::RGBA8_SRGB,
            jpeg_supported,
            ConvertIntent::Fastest
        ),
        Some(PixelDescriptor::RGB8_SRGB)
    );
    assert_eq!(
        best_match(
            PixelDescriptor::BGRA8_SRGB,
            jpeg_supported,
            ConvertIntent::Fastest
        ),
        Some(PixelDescriptor::RGB8_SRGB)
    );
    assert_eq!(
        best_match(
            PixelDescriptor::GRAY8_SRGB,
            jpeg_supported,
            ConvertIntent::Fastest
        ),
        Some(PixelDescriptor::GRAY8_SRGB)
    );
    assert_eq!(
        best_match(
            PixelDescriptor::RGBF32_LINEAR,
            jpeg_supported,
            ConvertIntent::Fastest
        ),
        Some(PixelDescriptor::RGB8_SRGB)
    );
}

/// Simulate WebP encoder's supported formats.
#[test]
fn webp_format_negotiation() {
    let webp_supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB];

    assert_eq!(
        best_match(
            PixelDescriptor::BGRA8_SRGB,
            webp_supported,
            ConvertIntent::Fastest
        ),
        Some(PixelDescriptor::RGBA8_SRGB)
    );
    assert_eq!(
        best_match(
            PixelDescriptor::GRAY8_SRGB,
            webp_supported,
            ConvertIntent::Fastest
        ),
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
            None,
            TransferFunction::Srgb,
        ),
    ];

    assert_eq!(
        best_match(
            PixelDescriptor::RGB8_SRGB,
            png_supported,
            ConvertIntent::Fastest
        ),
        Some(PixelDescriptor::RGB8_SRGB)
    );
    assert_eq!(
        best_match(
            PixelDescriptor::GRAY8_SRGB,
            png_supported,
            ConvertIntent::Fastest
        ),
        Some(PixelDescriptor::GRAY8_SRGB)
    );
}

// =========================================================================
// Two-axis ConversionCost tests
// =========================================================================

#[test]
fn identity_cost_is_zero() {
    let cost = conversion_cost(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGB8_SRGB);
    assert_eq!(cost, ConversionCost::ZERO);
}

#[test]
fn widening_conversion_has_zero_loss() {
    // u8 sRGB → f32 Linear: effort > 0 (EOTF math), loss = 0 (lossless expansion).
    let cost = conversion_cost(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBF32_LINEAR);
    assert_eq!(cost.loss, 0, "widening should be lossless");
    assert!(cost.effort > 0, "conversion should have nonzero effort");
}

#[test]
fn narrowing_conversion_has_nonzero_loss() {
    // f32 Linear → u8 sRGB: lossy (quantization).
    let cost = conversion_cost(PixelDescriptor::RGBF32_LINEAR, PixelDescriptor::RGB8_SRGB);
    assert!(cost.loss > 0, "f32→u8 should report data loss");
    assert!(cost.effort > 0);
}

#[test]
fn swizzle_is_effort_only() {
    let cost = conversion_cost(PixelDescriptor::BGRA8_SRGB, PixelDescriptor::RGBA8_SRGB);
    assert!(cost.effort > 0);
    assert_eq!(cost.loss, 0, "swizzle is lossless");
}

#[test]
fn drop_alpha_reports_loss() {
    let cost = conversion_cost(PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGB8_SRGB);
    assert!(cost.loss > 0, "dropping alpha loses information");
}

#[test]
fn color_to_gray_reports_high_loss() {
    let cost = conversion_cost(PixelDescriptor::RGB8_SRGB, PixelDescriptor::GRAY8_SRGB);
    assert!(
        cost.loss >= 400,
        "RGB→Gray loses color information (loss={})",
        cost.loss
    );
}

#[test]
fn cost_is_additive() {
    let a = ConversionCost::new(10, 20);
    let b = ConversionCost::new(30, 40);
    assert_eq!(a + b, ConversionCost::new(40, 60));
}

#[test]
fn cost_add_saturates() {
    let a = ConversionCost::new(u16::MAX, u16::MAX);
    let b = ConversionCost::new(1, 1);
    assert_eq!(a + b, ConversionCost::new(u16::MAX, u16::MAX));
}

// =========================================================================
// Intent-aware best_match tests
// =========================================================================

#[test]
fn linear_light_prefers_f32_linear() {
    let src = PixelDescriptor::RGB8_SRGB;
    let supported = &[PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGBAF32_LINEAR];
    assert_eq!(
        best_match(src, supported, ConvertIntent::LinearLight),
        Some(PixelDescriptor::RGBAF32_LINEAR)
    );
}

#[test]
fn fastest_prefers_same_depth() {
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
        Some(AlphaMode::Straight),
        TransferFunction::Linear,
    );
    let premul_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        Some(AlphaMode::Premultiplied),
        TransferFunction::Linear,
    );
    let supported = &[straight_f32, premul_f32];
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
        None,
        TransferFunction::Srgb,
    );
    let supported = &[PixelDescriptor::RGBF32_LINEAR, f32_srgb];
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
        None,
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
        None,
        TransferFunction::Pq,
    );
    let supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBF32_LINEAR];
    assert_eq!(
        best_match(src, supported, ConvertIntent::Fastest),
        Some(PixelDescriptor::RGBF32_LINEAR)
    );
}

#[test]
fn sdr8_allows_u8_fast_path() {
    let src = PixelDescriptor::RGB8_SRGB;
    let supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBF32_LINEAR];
    assert_eq!(
        best_match(src, supported, ConvertIntent::Perceptual),
        Some(PixelDescriptor::RGB8_SRGB)
    );
}

// =========================================================================
// Consumer cost override tests (FormatOption / best_match_with)
// =========================================================================

#[test]
fn consumer_fused_f32_path_preferred() {
    // JPEG encoder with fast internal f32→u8+DCT path.
    // Source is f32 Linear. Without override, we'd convert to u8 ourselves.
    // With the fused path, the consumer handles f32 cheaply.
    let src = PixelDescriptor::RGBF32_LINEAR;
    let options = &[
        FormatOption::from(PixelDescriptor::RGB8_SRGB), // native, we convert
        FormatOption::with_cost(
            PixelDescriptor::RGBF32_LINEAR,
            ConversionCost::new(5, 0), // fast fused path, no extra loss
        ),
    ];
    assert_eq!(
        best_match_with(src, options, ConvertIntent::Fastest),
        Some(PixelDescriptor::RGBF32_LINEAR)
    );
}

#[test]
fn consumer_cost_sums_with_ours() {
    // Source is u8 sRGB. Consumer can accept f32 but it's expensive for them.
    // Our u8→f32 conversion + their high cost should lose to native u8.
    let src = PixelDescriptor::RGB8_SRGB;
    let options = &[
        FormatOption::from(PixelDescriptor::RGB8_SRGB),
        FormatOption::with_cost(
            PixelDescriptor::RGBF32_LINEAR,
            ConversionCost::new(200, 0), // consumer's internal handling is expensive
        ),
    ];
    // Native u8 should win — the consumer's f32 path is too expensive.
    assert_eq!(
        best_match_with(src, options, ConvertIntent::Fastest),
        Some(PixelDescriptor::RGB8_SRGB)
    );
}

#[test]
fn consumer_loss_matters_for_quality_intents() {
    // Two targets with identical effort but different consumer loss.
    let src = PixelDescriptor::RGBF32_LINEAR;
    let low_loss = FormatOption::with_cost(
        PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        ),
        ConversionCost::new(10, 5),
    );
    let high_loss = FormatOption::with_cost(
        PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        ),
        ConversionCost::new(10, 200),
    );
    // With same descriptor but different consumer loss, quality intent picks lower loss.
    // Note: same descriptor, so our conversion cost is the same. Only consumer_cost differs.
    let options = &[high_loss, low_loss];
    assert_eq!(
        best_match_with(src, options, ConvertIntent::LinearLight),
        Some(low_loss.descriptor) // both have same descriptor, but low_loss option wins
    );
}

#[test]
fn format_option_from_descriptor_has_zero_cost() {
    let opt = FormatOption::from(PixelDescriptor::RGB8_SRGB);
    assert_eq!(opt.consumer_cost, ConversionCost::ZERO);
}

// =========================================================================
// ideal_format tests
// =========================================================================

#[test]
fn ideal_format_fastest_identity() {
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
        None,
        TransferFunction::Pq,
    );
    assert_eq!(ideal_format(pq, ConvertIntent::Fastest), pq);
}

#[test]
fn ideal_format_linear_light_sdr() {
    let result = ideal_format(PixelDescriptor::RGB8_SRGB, ConvertIntent::LinearLight);
    assert_eq!(result.channel_type(), ChannelType::F32);
    assert_eq!(result.transfer(), TransferFunction::Linear);
    assert_eq!(result.layout(), ChannelLayout::Rgb);
}

#[test]
fn ideal_format_linear_light_already_f32_linear() {
    assert_eq!(
        ideal_format(PixelDescriptor::RGBF32_LINEAR, ConvertIntent::LinearLight),
        PixelDescriptor::RGBF32_LINEAR
    );
}

#[test]
fn ideal_format_linear_light_hdr() {
    let pq = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Pq,
    );
    let result = ideal_format(pq, ConvertIntent::LinearLight);
    assert_eq!(result.channel_type(), ChannelType::F32);
    assert_eq!(result.transfer(), TransferFunction::Linear);
}

#[test]
fn ideal_format_blend_adds_premul() {
    let result = ideal_format(PixelDescriptor::RGBA8_SRGB, ConvertIntent::Blend);
    assert_eq!(result.channel_type(), ChannelType::F32);
    assert_eq!(result.transfer(), TransferFunction::Linear);
    assert_eq!(result.alpha(), Some(AlphaMode::Premultiplied));
    assert_eq!(result.layout(), ChannelLayout::Rgba);
}

#[test]
fn ideal_format_blend_no_alpha_source() {
    let result = ideal_format(PixelDescriptor::RGB8_SRGB, ConvertIntent::Blend);
    assert_eq!(result.channel_type(), ChannelType::F32);
    assert_eq!(result.transfer(), TransferFunction::Linear);
    assert!(result.alpha().is_none());
}

#[test]
fn ideal_format_perceptual_sdr8() {
    assert_eq!(
        ideal_format(PixelDescriptor::RGB8_SRGB, ConvertIntent::Perceptual),
        PixelDescriptor::RGB8_SRGB
    );
}

#[test]
fn ideal_format_perceptual_u16() {
    let result = ideal_format(PixelDescriptor::RGB16_SRGB, ConvertIntent::Perceptual);
    assert_eq!(result.channel_type(), ChannelType::F32);
    assert_eq!(result.transfer(), TransferFunction::Srgb);
}

#[test]
fn ideal_format_perceptual_hdr() {
    let pq = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Pq,
    );
    let result = ideal_format(pq, ConvertIntent::Perceptual);
    assert_eq!(result.channel_type(), ChannelType::F32);
    assert_eq!(result.transfer(), TransferFunction::Srgb);
}

// =========================================================================
// Provenance tests — origin depth tracking
// =========================================================================

#[test]
fn u8_origin_f32_to_u8_reports_zero_loss() {
    // JPEG (u8 sRGB) → decoded to f32 for resize → encode back to u8.
    // The data's true precision is u8, so f32→u8 is lossless.
    let provenance = Provenance::with_origin_depth(ChannelType::U8);
    let cost = conversion_cost_with_provenance(
        PixelDescriptor::RGBF32_LINEAR,
        PixelDescriptor::RGB8_SRGB,
        provenance,
    );
    assert_eq!(
        cost.loss, 0,
        "u8→f32→u8 round-trip should report zero depth loss"
    );
    assert!(cost.effort > 0, "conversion still has effort cost");
}

#[test]
fn u16_origin_f32_to_u8_reports_loss() {
    // 16-bit PNG → decoded to f32 → encode to JPEG u8.
    // The data's true precision is u16, so f32→u8 is lossy.
    let provenance = Provenance::with_origin_depth(ChannelType::U16);
    let cost = conversion_cost_with_provenance(
        PixelDescriptor::RGBF32_LINEAR,
        PixelDescriptor::RGB8_SRGB,
        provenance,
    );
    assert!(cost.loss > 0, "u16→f32→u8 truncates precision");
}

#[test]
fn f32_origin_f32_to_u8_reports_high_loss() {
    // EXR (native f32) → encode to JPEG u8.
    // Perceptually calibrated: sRGB quantization produces ΔE < 0.5 (sub-JND),
    // so the loss value is low but still nonzero (distinguishing from u8-origin
    // round-trips which have zero loss).
    let provenance = Provenance::with_origin_depth(ChannelType::F32);
    let cost = conversion_cost_with_provenance(
        PixelDescriptor::RGBF32_LINEAR,
        PixelDescriptor::RGB8_SRGB,
        provenance,
    );
    assert!(
        cost.loss > 0,
        "f32→u8 with f32 origin should report nonzero loss (got {})",
        cost.loss
    );
    // Verify it's higher than the u8 round-trip (which is zero)
    let u8_prov = Provenance::with_origin_depth(ChannelType::U8);
    let u8_cost = conversion_cost_with_provenance(
        PixelDescriptor::RGBF32_LINEAR,
        PixelDescriptor::RGB8_SRGB,
        u8_prov,
    );
    assert!(
        cost.loss > u8_cost.loss,
        "f32 origin should have more loss than u8 origin"
    );
}

#[test]
fn u16_origin_f32_to_u16_reports_zero_loss() {
    // 16-bit PNG → f32 for processing → back to u16.
    let provenance = Provenance::with_origin_depth(ChannelType::U16);
    let f32_src = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    let u16_dst = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
    );
    let cost = conversion_cost_with_provenance(f32_src, u16_dst, provenance);
    assert_eq!(
        cost.loss, 0,
        "u16→f32→u16 round-trip should report zero depth loss"
    );
}

#[test]
fn provenance_without_narrowing_is_unchanged() {
    // u8 origin, already at u8 → no depth conversion at all.
    let provenance = Provenance::with_origin_depth(ChannelType::U8);
    let cost = conversion_cost_with_provenance(
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGB8_SRGB,
        provenance,
    );
    assert_eq!(cost, ConversionCost::ZERO);
}

#[test]
fn provenance_from_source_matches_old_behavior() {
    // Provenance::from_source(f32) should give the same result as
    // conversion_cost() — both report high loss for f32→u8.
    let src = PixelDescriptor::RGBF32_LINEAR;
    let dst = PixelDescriptor::RGB8_SRGB;
    let with_prov = conversion_cost_with_provenance(src, dst, Provenance::from_source(src));
    let without = conversion_cost(src, dst);
    assert_eq!(with_prov, without);
}

#[test]
fn provenance_shifts_negotiate_preference() {
    // Source is f32 Linear, but origin was u8 JPEG.
    // Consumer has a moderately expensive f32 path (effort=46).
    // Our f32→u8 conversion costs effort=45 (depth 40 + transfer 5), loss=10.
    //
    // Without provenance (f32 origin): f32→u8 has loss=10 (calibrated).
    //   f32→u8 score = 45*4 + 10 = 190
    //   f32→f32 score = 46*4 + 0  = 184  ← wins
    //
    // With u8 provenance: f32→u8 has loss=0 (round-trip lossless).
    //   f32→u8 score = 45*4 + 0 = 180  ← wins
    //   f32→f32 score = 46*4 + 0 = 184
    let src = PixelDescriptor::RGBF32_LINEAR;
    let provenance = Provenance::with_origin_depth(ChannelType::U8);
    let options = [
        FormatOption::from(PixelDescriptor::RGB8_SRGB), // our f32→u8 conversion, zero consumer cost
        FormatOption::with_cost(
            PixelDescriptor::RGBF32_LINEAR,
            ConversionCost::new(46, 0), // consumer accepts f32, slightly more effort than our conversion
        ),
    ];
    // Without provenance: f32→u8 has loss=10 → f32 path wins by effort.
    assert_eq!(
        best_match_with(src, &options, ConvertIntent::Fastest),
        Some(PixelDescriptor::RGBF32_LINEAR),
        "without provenance, f32 should win due to u8 loss penalty"
    );
    // With provenance: f32→u8 has loss=0 → u8 path is cheaper overall.
    assert_eq!(
        negotiate(
            src,
            provenance,
            options.iter().copied(),
            ConvertIntent::Fastest
        ),
        Some(PixelDescriptor::RGB8_SRGB),
        "with u8 provenance, u8 should win since round-trip is lossless"
    );
}
