//! Tests for format selection and preference ordering (`ConvertPlan::select`).

use zenpixels_convert::{
    AlphaMode, AlphaPolicy, CandidateTarget, ChannelLayout, ChannelType, ConversionSpec,
    ConvertError, ConvertIntent, ConvertOptions, ConvertPlan, DepthPolicy, LumaCoefficients,
    PixelDescriptor, Provenance, TransferFunction,
};
#[cfg(feature = "introspection")]
use zenpixels_convert::{ConversionCost, LossBucket};

fn test_options() -> ConvertOptions {
    ConvertOptions::retain_channels()
        .with_alpha_policy(AlphaPolicy::DiscardUnchecked)
        .with_depth_policy(DepthPolicy::Round)
        .with_luma(Some(LumaCoefficients::Bt709))
        .with_sdr_white(zenpixels::hdr::DiffuseWhite::BT2408)
}

fn test_spec() -> ConversionSpec<'static> {
    ConversionSpec::new(test_options())
}

fn select_desc(
    intent: ConvertIntent,
    src: PixelDescriptor,
    supported: &[PixelDescriptor],
) -> Option<PixelDescriptor> {
    let candidates: Vec<CandidateTarget<'_>> = supported
        .iter()
        .copied()
        .map(CandidateTarget::from)
        .collect();
    let spec = test_spec().with_intent(intent);
    ConvertPlan::select(src.encoding(), &candidates, &spec)
        .ok()
        .map(|s| s.plan().output_encoding().descriptor())
}

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
        select_desc(ConvertIntent::Fastest, src, supported),
        Some(PixelDescriptor::RGBA8_SRGB)
    );
}

#[test]
fn transfer_only_diff_over_layout_change_and_unknown_refusal() {
    let src = PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Bt709);
    let supported = &[PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGB8_SRGB];
    assert_eq!(
        select_desc(ConvertIntent::Fastest, src, supported),
        Some(PixelDescriptor::RGB8_SRGB)
    );

    // Unknown transfer without explicit relabeling refuses instead of guessing sRGB.
    let candidates: Vec<CandidateTarget<'_>> = supported
        .iter()
        .copied()
        .map(CandidateTarget::from)
        .collect();
    let err = ConvertPlan::select(PixelDescriptor::RGB8.encoding(), &candidates, &test_spec())
        .unwrap_err();
    assert!(matches!(
        err.error(),
        ConvertError::NoCandidate { index: 0, cause }
            if matches!(cause.as_ref(), ConvertError::UnknownTransferRelabelForbidden { .. })
    ));
}

#[test]
fn same_depth_over_cross_depth() {
    let src = PixelDescriptor::RGB8_SRGB;
    let supported = &[PixelDescriptor::RGBF32_LINEAR, PixelDescriptor::RGBA8_SRGB];
    assert_eq!(
        select_desc(ConvertIntent::Fastest, src, supported),
        Some(PixelDescriptor::RGBA8_SRGB)
    );
}

#[test]
fn swizzle_cheaper_than_drop_alpha() {
    let src = PixelDescriptor::BGRA8_SRGB;
    let supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB];
    assert_eq!(
        select_desc(ConvertIntent::Fastest, src, supported),
        Some(PixelDescriptor::RGBA8_SRGB)
    );
}

#[test]
fn gray_prefers_rgb_over_rgba() {
    let src = PixelDescriptor::GRAY8_SRGB;
    let supported = &[PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGB8_SRGB];
    assert_eq!(
        select_desc(ConvertIntent::Fastest, src, supported),
        Some(PixelDescriptor::RGB8_SRGB)
    );
}

#[test]
fn f32_linear_prefers_u16_over_u8_in_default_and_fastest() {
    let src = PixelDescriptor::RGBF32_LINEAR;
    let desc_u16 = PixelDescriptor::RGB16_SRGB;
    for candidates in [
        [
            CandidateTarget::from(PixelDescriptor::RGB8_SRGB),
            CandidateTarget::from(desc_u16),
        ],
        [
            CandidateTarget::from(desc_u16),
            CandidateTarget::from(PixelDescriptor::RGB8_SRGB),
        ],
    ] {
        let default_sel = ConvertPlan::select(src.encoding(), &candidates, &test_spec()).unwrap();
        assert_eq!(default_sel.plan().output_encoding().descriptor(), desc_u16);
    }
    assert_eq!(
        select_desc(
            ConvertIntent::Fastest,
            src,
            &[desc_u16, PixelDescriptor::RGB8_SRGB]
        ),
        Some(desc_u16)
    );
}

#[test]
fn empty_returns_empty_format_list_error() {
    let err =
        ConvertPlan::select(PixelDescriptor::RGB8_SRGB.encoding(), &[], &test_spec()).unwrap_err();
    assert_eq!(*err.error(), ConvertError::EmptyFormatList);
}

#[test]
fn single_option_always_selected() {
    let src = PixelDescriptor::GRAY8_SRGB;
    let supported = &[PixelDescriptor::RGBAF32_LINEAR];
    assert_eq!(
        select_desc(ConvertIntent::Fastest, src, supported),
        Some(PixelDescriptor::RGBAF32_LINEAR)
    );
}

#[test]
fn jpeg_format_negotiation() {
    let jpeg_supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::GRAY8_SRGB];

    assert_eq!(
        select_desc(
            ConvertIntent::Fastest,
            PixelDescriptor::RGBA8_SRGB,
            jpeg_supported
        ),
        Some(PixelDescriptor::RGB8_SRGB)
    );
    assert_eq!(
        select_desc(
            ConvertIntent::Fastest,
            PixelDescriptor::BGRA8_SRGB,
            jpeg_supported
        ),
        Some(PixelDescriptor::RGB8_SRGB)
    );
    assert_eq!(
        select_desc(
            ConvertIntent::Fastest,
            PixelDescriptor::GRAY8_SRGB,
            jpeg_supported
        ),
        Some(PixelDescriptor::GRAY8_SRGB)
    );
    assert_eq!(
        select_desc(
            ConvertIntent::Fastest,
            PixelDescriptor::RGBF32_LINEAR,
            jpeg_supported
        ),
        Some(PixelDescriptor::RGB8_SRGB)
    );
}

#[test]
fn webp_format_negotiation() {
    let webp_supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB];

    assert_eq!(
        select_desc(
            ConvertIntent::Fastest,
            PixelDescriptor::BGRA8_SRGB,
            webp_supported
        ),
        Some(PixelDescriptor::RGBA8_SRGB)
    );
    assert_eq!(
        select_desc(
            ConvertIntent::Fastest,
            PixelDescriptor::GRAY8_SRGB,
            webp_supported
        ),
        Some(PixelDescriptor::RGB8_SRGB)
    );
}

#[test]
fn png_format_negotiation() {
    let png_supported = &[
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::GRAY8_SRGB,
        PixelDescriptor::RGB16_SRGB,
    ];

    assert_eq!(
        select_desc(
            ConvertIntent::Fastest,
            PixelDescriptor::RGB8_SRGB,
            png_supported
        ),
        Some(PixelDescriptor::RGB8_SRGB)
    );
    assert_eq!(
        select_desc(
            ConvertIntent::Fastest,
            PixelDescriptor::GRAY8_SRGB,
            png_supported
        ),
        Some(PixelDescriptor::GRAY8_SRGB)
    );
}

#[cfg(feature = "introspection")]
#[test]
fn two_axis_conversion_cost_invariants() {
    assert_eq!(
        ConversionCost::between(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGB8_SRGB),
        ConversionCost::ZERO
    );
    let widen = ConversionCost::between(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBF32_LINEAR);
    assert_eq!(widen.loss(), 0);
    assert!(widen.effort() > 0);

    let narrow =
        ConversionCost::between(PixelDescriptor::RGBF32_LINEAR, PixelDescriptor::RGB8_SRGB);
    assert!(narrow.loss() > 0);
    assert!(narrow.effort() > 0);

    let swiz = ConversionCost::between(PixelDescriptor::BGRA8_SRGB, PixelDescriptor::RGBA8_SRGB);
    assert!(swiz.effort() > 0);
    assert_eq!(swiz.loss(), 0);

    let drop_a = ConversionCost::between(PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGB8_SRGB);
    assert!(drop_a.loss() > 0);

    let gray = ConversionCost::between(PixelDescriptor::RGB8_SRGB, PixelDescriptor::GRAY8_SRGB);
    assert!(gray.loss() >= 400);

    assert_eq!(
        ConversionCost::new(10, 20) + ConversionCost::new(30, 40),
        ConversionCost::new(40, 60)
    );
    assert_eq!(
        ConversionCost::new(u16::MAX, u16::MAX) + ConversionCost::new(1, 1),
        ConversionCost::new(u16::MAX, u16::MAX)
    );
}

#[test]
fn linear_light_prefers_f32_linear() {
    let src = PixelDescriptor::RGB8_SRGB;
    let supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBF32_LINEAR];
    assert_eq!(
        select_desc(ConvertIntent::LinearLight, src, supported),
        Some(PixelDescriptor::RGBF32_LINEAR)
    );
}

#[test]
fn fastest_prefers_same_depth() {
    let src = PixelDescriptor::RGB8_SRGB;
    let supported = &[PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGBAF32_LINEAR];
    assert_eq!(
        select_desc(ConvertIntent::Fastest, src, supported),
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
        select_desc(ConvertIntent::Blend, src, supported),
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
        select_desc(ConvertIntent::Perceptual, src, supported),
        Some(f32_srgb)
    );
}

#[test]
fn u16_source_penalizes_u8_target() {
    let src = PixelDescriptor::RGB16_SRGB;
    let rgb16 = PixelDescriptor::RGB16_SRGB;
    let supported = &[PixelDescriptor::RGB8_SRGB, rgb16];
    assert_eq!(
        select_desc(ConvertIntent::Fastest, src, supported),
        Some(rgb16)
    );
    assert_eq!(
        select_desc(ConvertIntent::LinearLight, src, supported),
        Some(rgb16)
    );
}

#[test]
fn hdr_source_picks_linear_over_unreachable_sdr_target() {
    let src = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Pq,
    );
    let supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBF32_LINEAR];
    assert_eq!(
        select_desc(ConvertIntent::Fastest, src, supported),
        Some(PixelDescriptor::RGBF32_LINEAR)
    );
}

#[test]
fn sdr8_allows_u8_fast_path() {
    let src = PixelDescriptor::RGB8_SRGB;
    let supported = &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBF32_LINEAR];
    assert_eq!(
        select_desc(ConvertIntent::Perceptual, src, supported),
        Some(PixelDescriptor::RGB8_SRGB)
    );
}

#[test]
fn consumer_fused_f32_path_preferred() {
    let src = PixelDescriptor::RGBF32_LINEAR;
    let candidates = [
        CandidateTarget::from(PixelDescriptor::RGB8_SRGB),
        CandidateTarget::from(PixelDescriptor::RGBF32_LINEAR).with_consumer_cost(5, 0),
    ];
    let sel = ConvertPlan::select(
        src.encoding(),
        &candidates,
        &test_spec().with_intent(ConvertIntent::Fastest),
    )
    .unwrap();
    assert_eq!(
        sel.plan().output_encoding().descriptor(),
        PixelDescriptor::RGBF32_LINEAR
    );
    assert_eq!(sel.candidate_index(), 1);
}

#[test]
fn consumer_cost_sums_with_ours() {
    let src = PixelDescriptor::RGB8_SRGB;
    let candidates = [
        CandidateTarget::from(PixelDescriptor::RGB8_SRGB),
        CandidateTarget::from(PixelDescriptor::RGBF32_LINEAR).with_consumer_cost(200, 0),
    ];
    let sel = ConvertPlan::select(
        src.encoding(),
        &candidates,
        &test_spec().with_intent(ConvertIntent::Fastest),
    )
    .unwrap();
    assert_eq!(
        sel.plan().output_encoding().descriptor(),
        PixelDescriptor::RGB8_SRGB
    );
    assert_eq!(sel.candidate_index(), 0);
}

#[test]
fn consumer_loss_matters_for_quality_intents() {
    let src = PixelDescriptor::RGBF32_LINEAR;
    let f32_srgb = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
    );
    let high_loss = CandidateTarget::from(f32_srgb).with_consumer_cost(10, 200);
    let low_loss = CandidateTarget::from(f32_srgb).with_consumer_cost(10, 5);
    let sel = ConvertPlan::select(
        src.encoding(),
        &[high_loss, low_loss],
        &test_spec().with_intent(ConvertIntent::LinearLight),
    )
    .unwrap();
    assert_eq!(sel.candidate_index(), 1);
}

#[cfg(feature = "introspection")]
#[test]
fn provenance_cost_invariants() {
    let u8_prov = Provenance::with_origin_depth(ChannelType::U8);
    let c_u8 = ConversionCost::between_with_provenance(
        PixelDescriptor::RGBF32_LINEAR,
        PixelDescriptor::RGB8_SRGB,
        u8_prov,
    );
    assert_eq!(c_u8.loss(), 0);
    assert!(c_u8.effort() > 0);

    let u16_prov = Provenance::with_origin_depth(ChannelType::U16);
    assert!(
        ConversionCost::between_with_provenance(
            PixelDescriptor::RGBF32_LINEAR,
            PixelDescriptor::RGB8_SRGB,
            u16_prov,
        )
        .loss()
            > 0
    );

    let f32_prov = Provenance::with_origin_depth(ChannelType::F32);
    let c_f32 = ConversionCost::between_with_provenance(
        PixelDescriptor::RGBF32_LINEAR,
        PixelDescriptor::RGB8_SRGB,
        f32_prov,
    );
    assert!(c_f32.loss() > c_u8.loss());

    let u16_dst = PixelDescriptor::RGB16_SRGB;
    assert_eq!(
        ConversionCost::between_with_provenance(PixelDescriptor::RGBF32_LINEAR, u16_dst, u16_prov)
            .loss(),
        0
    );

    assert_eq!(
        ConversionCost::between_with_provenance(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::RGB8_SRGB,
            u8_prov
        ),
        ConversionCost::ZERO
    );

    let src = PixelDescriptor::RGBF32_LINEAR;
    let dst = PixelDescriptor::RGB8_SRGB;
    assert_eq!(
        ConversionCost::between_with_provenance(src, dst, Provenance::from_source(src)),
        ConversionCost::between(src, dst)
    );
}

#[test]
fn provenance_shifts_select_preference() {
    let src = PixelDescriptor::RGBF32_LINEAR;
    let provenance = Provenance::with_origin_depth(ChannelType::U8);
    let candidates = [
        CandidateTarget::from(PixelDescriptor::RGB8_SRGB),
        CandidateTarget::from(PixelDescriptor::RGBF32_LINEAR).with_consumer_cost(46, 0),
    ];
    let without_prov = ConvertPlan::select(
        src.encoding(),
        &candidates,
        &test_spec()
            .with_intent(ConvertIntent::Fastest)
            .with_provenance(Provenance::from_source(src)),
    )
    .unwrap();
    assert_eq!(
        without_prov.plan().output_encoding().descriptor(),
        PixelDescriptor::RGBF32_LINEAR
    );

    let with_prov = ConvertPlan::select(
        src.encoding(),
        &candidates,
        &test_spec()
            .with_intent(ConvertIntent::Fastest)
            .with_provenance(provenance),
    )
    .unwrap();
    assert_eq!(
        with_prov.plan().output_encoding().descriptor(),
        PixelDescriptor::RGB8_SRGB
    );
}

#[cfg(feature = "introspection")]
#[test]
fn loss_buckets_follow_their_documented_ranges() {
    use LossBucket::{Clear, Imperceptible, Minor, Severe, SideBySide};
    for (loss, bucket) in [
        (0, Imperceptible),
        (10, Imperceptible),
        (11, SideBySide),
        (50, SideBySide),
        (51, Minor),
        (150, Minor),
        (151, Clear),
        (400, Clear),
        (401, Severe),
        (u16::MAX, Severe),
    ] {
        assert_eq!(
            ConversionCost::new(0, loss).loss_bucket(),
            bucket,
            "loss {loss}"
        );
    }
    assert!(Imperceptible < SideBySide && SideBySide < Minor && Minor < Clear && Clear < Severe);
}
