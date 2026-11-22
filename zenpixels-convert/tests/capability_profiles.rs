//! Integration tests for compile-time capability bitmasking (`caps` + `ConverterProfile`).

use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, Color, ColorPrimaries, PixelDescriptor as D,
    TransferFunction,
};
use zenpixels_convert::{
    AlphaPolicy, CandidateTarget, ConversionSpec, ConvertError, ConvertOptions, ConvertPlan,
    ConverterProfile, DepthPolicy, GamutPolicy, LumaCoefficients, caps,
};

fn permissive_opts() -> ConvertOptions {
    ConvertOptions::retain_channels()
        .with_alpha_policy(AlphaPolicy::DiscardUnchecked)
        .with_depth_policy(DepthPolicy::Round)
        .with_luma(Some(LumaCoefficients::Bt709))
}

fn rgbf32_srgb() -> D {
    D::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb)
}

fn rgbaf32_srgb() -> D {
    D::RGBAF32_LINEAR.with_transfer(TransferFunction::Srgb)
}

fn bgr8_srgb() -> D {
    D::new(
        ChannelType::U8,
        ChannelLayout::Bgr,
        None,
        TransferFunction::Srgb,
    )
}

fn assert_profile_matches_full<const CAPS: u64>(
    from: D,
    to: D,
    opts: ConvertOptions,
    src: &[u8],
    width: u32,
) {
    let spec = ConversionSpec::new(opts);
    let mut full = ConvertPlan::build(from.into(), to.into(), &spec)
        .unwrap_or_else(|e| panic!("full build failed for {from:?} -> {to:?}: {e}"))
        .prepare(width)
        .unwrap();
    let mut masked = ConverterProfile::<CAPS>::build(from.into(), to.into(), &spec)
        .unwrap_or_else(|e| panic!("masked build failed for {from:?} -> {to:?}: {e}"))
        .prepare(width)
        .unwrap();
    let out_len = width as usize * to.bytes_per_pixel();
    let mut expected = vec![0u8; out_len];
    let mut actual = vec![0u8; out_len];
    full.try_convert_row(src, &mut expected, width).unwrap();
    masked.try_convert_row(src, &mut actual, width).unwrap();
    assert_eq!(
        actual, expected,
        "byte mismatch for {from:?} -> {to:?} under CAPS=0x{CAPS:x}"
    );
}

fn assert_profile_rejects<const CAPS: u64>(from: D, to: D, opts: ConvertOptions) {
    let spec = ConversionSpec::new(opts);
    let err = ConverterProfile::<CAPS>::build(from.into(), to.into(), &spec)
        .expect_err(&format!("expected NoPath for {from:?} -> {to:?}"));
    assert!(
        matches!(err.error(), ConvertError::NoPath { .. }),
        "expected NoPath for {from:?} -> {to:?}, got {err:?}"
    );
}

#[test]
fn layout_u8_profile_matches_full_and_rejects_extras() {
    let opts = permissive_opts();
    let rgba8: Vec<u8> = (0..64).map(|i| (i * 17) as u8).collect();
    let rgb8: Vec<u8> = (0..48).map(|i| (i * 13) as u8).collect();

    assert_profile_matches_full::<{ caps::LAYOUT_U8 }>(
        D::RGBA8_SRGB,
        D::RGB8_SRGB,
        opts,
        &rgba8,
        16,
    );
    assert_profile_matches_full::<{ caps::LAYOUT_U8 }>(
        D::RGB8_SRGB,
        D::RGBA8_SRGB,
        opts,
        &rgb8,
        16,
    );
    assert_profile_matches_full::<{ caps::LAYOUT_U8 }>(
        D::RGBA8_SRGB,
        D::BGRA8_SRGB,
        opts,
        &rgba8,
        16,
    );
    assert_profile_matches_full::<{ caps::LAYOUT_U8 }>(
        D::RGB8_SRGB,
        D::BGRA8_SRGB,
        opts,
        &rgb8,
        16,
    );
    assert_profile_matches_full::<{ caps::LAYOUT_U8 }>(bgr8_srgb(), D::RGB8_SRGB, opts, &rgb8, 16);
    assert_profile_matches_full::<{ caps::LAYOUT_U8 }>(
        D::RGBA8_SRGB,
        D::RGBA8_SRGB,
        opts,
        &rgba8,
        16,
    );

    // Disallowed depths, gray, premul, matte, transfer, gamut.
    assert_profile_rejects::<{ caps::LAYOUT_U8 }>(D::RGBA16_SRGB, D::RGB16_SRGB, opts);
    assert_profile_rejects::<{ caps::LAYOUT_U8 }>(rgbaf32_srgb(), rgbf32_srgb(), opts);
    assert_profile_rejects::<{ caps::LAYOUT_U8 }>(D::RGB8_SRGB, D::GRAY8_SRGB, opts);
    assert_profile_rejects::<{ caps::LAYOUT_U8 }>(D::GRAY8_SRGB, D::RGB8_SRGB, opts);
    assert_profile_rejects::<{ caps::LAYOUT_U8 }>(
        D::RGBA8_SRGB,
        D::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied)),
        opts,
    );
    assert_profile_rejects::<{ caps::LAYOUT_U8 }>(
        D::RGBA8_SRGB,
        D::RGB8_SRGB,
        opts.with_alpha_policy(AlphaPolicy::CompositeOnto {
            background: Color::srgb8(255, 255, 255),
            reference_white: None,
        }),
    );
    assert_profile_rejects::<{ caps::LAYOUT_U8 }>(D::RGB8_SRGB, D::RGBF32_LINEAR, opts);
    assert_profile_rejects::<{ caps::LAYOUT_U8 }>(
        D::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3),
        D::RGB8_SRGB,
        opts,
    );

    #[cfg(feature = "oklab")]
    assert_profile_rejects::<{ caps::LAYOUT_U8 }>(D::RGBF32_LINEAR, D::OKLABF32, opts);

    #[cfg(feature = "hdr-transfers")]
    assert_profile_rejects::<{ caps::LAYOUT_U8 }>(
        D::RGBF32_LINEAR.with_transfer(TransferFunction::Pq),
        D::RGBF32_LINEAR,
        opts.with_sdr_white(zenpixels::hdr::DiffuseWhite::BT2408),
    );

    #[cfg(feature = "perceptual")]
    assert_profile_rejects::<{ caps::LAYOUT_U8 }>(
        rgbf32_srgb(),
        D::new(
            ChannelType::F32,
            ChannelLayout::Hsl,
            None,
            TransferFunction::Srgb,
        ),
        opts,
    );
}

#[test]
fn sdr_web_profile_matches_full_and_rejects_extras() {
    let opts = permissive_opts();
    let rgba8: Vec<u8> = (0..64).map(|i| (i * 19 + 7) as u8).collect();
    let rgb8: Vec<u8> = (0..48).map(|i| (i * 23 + 3) as u8).collect();
    let gray8: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
    let rgbf32: Vec<f32> = (0..48).map(|i| (i as f32) / 47.0).collect();
    let rgbaf32: Vec<f32> = (0..64).map(|i| (i as f32) / 63.0).collect();

    assert_profile_matches_full::<{ caps::SDR_WEB }>(D::RGBA8_SRGB, D::RGB8_SRGB, opts, &rgba8, 16);
    assert_profile_matches_full::<{ caps::SDR_WEB }>(D::RGB8_SRGB, D::GRAY8_SRGB, opts, &rgb8, 16);
    assert_profile_matches_full::<{ caps::SDR_WEB }>(
        D::GRAY8_SRGB,
        D::RGBA8_SRGB,
        opts,
        &gray8,
        16,
    );
    assert_profile_matches_full::<{ caps::SDR_WEB }>(
        D::RGBA8_SRGB,
        D::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied)),
        opts,
        &rgba8,
        16,
    );
    assert_profile_matches_full::<{ caps::SDR_WEB }>(
        D::RGBAF32_LINEAR,
        D::RGBAF32_LINEAR.with_alpha(Some(AlphaMode::Premultiplied)),
        opts,
        bytemuck::cast_slice(&rgbaf32),
        16,
    );
    let matte_opts = opts.with_alpha_policy(AlphaPolicy::CompositeOnto {
        background: Color::srgb8(240, 200, 160),
        reference_white: None,
    });
    assert_profile_matches_full::<{ caps::SDR_WEB }>(
        D::RGBA8_SRGB,
        D::RGB8_SRGB,
        matte_opts,
        &rgba8,
        16,
    );
    assert_profile_matches_full::<{ caps::SDR_WEB }>(
        rgbaf32_srgb(),
        rgbf32_srgb(),
        matte_opts,
        bytemuck::cast_slice(&rgbaf32),
        16,
    );
    assert_profile_matches_full::<{ caps::SDR_WEB }>(
        D::RGB8_SRGB,
        D::RGBF32_LINEAR,
        opts,
        &rgb8,
        16,
    );
    assert_profile_matches_full::<{ caps::SDR_WEB }>(
        D::RGBF32_LINEAR,
        D::RGB8_SRGB,
        opts,
        bytemuck::cast_slice(&rgbf32),
        16,
    );
    assert_profile_matches_full::<{ caps::SDR_WEB }>(
        rgbf32_srgb(),
        D::RGB8_SRGB,
        opts,
        bytemuck::cast_slice(&rgbf32),
        16,
    );
    assert_profile_matches_full::<{ caps::SDR_WEB }>(
        D::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3),
        D::RGB8_SRGB,
        opts,
        &rgb8,
        16,
    );
    assert_profile_matches_full::<{ caps::SDR_WEB }>(
        D::RGBF32_LINEAR.with_primaries(ColorPrimaries::DisplayP3),
        D::RGBF32_LINEAR,
        opts,
        bytemuck::cast_slice(&rgbf32),
        16,
    );

    // Disallowed in SDR_WEB: U16, F16, BT.709, Gamma22, extended sRGB, truncate, HDR, Oklab, perceptual.
    assert_profile_rejects::<{ caps::SDR_WEB }>(D::RGB16_SRGB, D::RGB8_SRGB, opts);
    assert_profile_rejects::<{ caps::SDR_WEB }>(D::RGBA16_SRGB, D::RGB16_SRGB, opts);
    assert_profile_rejects::<{ caps::SDR_WEB }>(D::RGBF16_LINEAR, D::RGBF32_LINEAR, opts);
    assert_profile_rejects::<{ caps::SDR_WEB }>(
        D::RGBF32_LINEAR,
        D::RGBF32_LINEAR.with_transfer(TransferFunction::Bt709),
        opts,
    );
    assert_profile_rejects::<{ caps::SDR_WEB }>(
        D::RGBF32_LINEAR,
        D::RGBF32_LINEAR.with_transfer(TransferFunction::Gamma22),
        opts,
    );
    assert_profile_rejects::<{ caps::SDR_WEB }>(
        rgbf32_srgb(),
        D::RGBF32_LINEAR,
        opts.with_gamut_policy(GamutPolicy::PreserveExtended),
    );
    assert_profile_rejects::<{ caps::SDR_WEB }>(
        rgbf32_srgb(),
        D::RGB8_SRGB,
        opts.with_depth_policy(DepthPolicy::Truncate),
    );

    #[cfg(feature = "oklab")]
    {
        assert_profile_rejects::<{ caps::SDR_WEB }>(D::RGBF32_LINEAR, D::OKLABF32, opts);
        assert_profile_rejects::<{ caps::SDR_WEB }>(
            D::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3),
            D::RGB8_SRGB,
            opts.with_gamut_policy(GamutPolicy::SoftCompress),
        );
    }

    #[cfg(feature = "hdr-transfers")]
    assert_profile_rejects::<{ caps::SDR_WEB }>(
        D::RGBF32_LINEAR.with_transfer(TransferFunction::Pq),
        D::RGBF32_LINEAR,
        opts.with_sdr_white(zenpixels::hdr::DiffuseWhite::BT2408),
    );

    #[cfg(feature = "perceptual")]
    assert_profile_rejects::<{ caps::SDR_WEB }>(
        rgbf32_srgb(),
        D::new(
            ChannelType::F32,
            ChannelLayout::Hsl,
            None,
            TransferFunction::Srgb,
        ),
        opts,
    );
}

#[test]
fn profile_select_skips_disallowed_candidates() {
    let opts = permissive_opts();
    let spec = ConversionSpec::new(opts);
    let targets = [
        CandidateTarget::from(D::RGB16_SRGB),
        CandidateTarget::from(D::RGBF32_LINEAR),
        CandidateTarget::from(D::RGB8_SRGB),
    ];
    let selected =
        ConverterProfile::<{ caps::LAYOUT_U8 }>::select(D::RGBA8_SRGB.into(), &targets, &spec)
            .expect("select should skip U16/F32 and pick RGB8_SRGB");
    assert_eq!(selected.candidate_index(), 2);
    assert_eq!(selected.plan().output_encoding().descriptor(), D::RGB8_SRGB);

    let mut worker = selected.into_plan().prepare(2).unwrap();
    let src = [10u8, 20, 30, 255, 40, 50, 60, 128];
    let mut dst = [0u8; 6];
    worker.try_convert_row(&src, &mut dst, 2).unwrap();
    assert_eq!(dst, [10, 20, 30, 40, 50, 60]);
}

#[test]
fn profile_plans_compose_with_then() {
    let opts = permissive_opts();
    let spec = ConversionSpec::new(opts);
    let first = ConverterProfile::<{ caps::SDR_WEB }>::build(
        D::RGBA8_SRGB.into(),
        D::RGBF32_LINEAR.into(),
        &spec,
    )
    .unwrap();
    let second = ConverterProfile::<{ caps::SDR_WEB }>::build(
        D::RGBF32_LINEAR.into(),
        D::BGRA8_SRGB.into(),
        &spec,
    )
    .unwrap();
    let composed = first.then(second).expect("matching endpoints");
    let mut worker = composed.prepare(4).unwrap();

    let src = [
        255u8, 0, 128, 200, 10, 20, 30, 40, 100, 150, 200, 255, 0, 0, 0, 0,
    ];
    let mut dst = [0u8; 16];
    worker.try_convert_row(&src, &mut dst, 4).unwrap();

    // Expected: RGB roundtrips sRGB U8 -> Linear F32 -> sRGB U8, then R<->B swap + opaque alpha.
    assert_eq!(
        dst,
        [
            128, 0, 255, 255, 30, 20, 10, 255, 200, 150, 100, 255, 0, 0, 0, 255
        ]
    );
}

#[test]
fn custom_capability_mask_combines_cleanly() {
    const SRGB_DEPTH_ONLY: u64 = caps::U8 | caps::F32 | caps::TF_SRGB;
    let opts = permissive_opts();
    let rgb8 = [0u8, 64, 128, 192, 255, 32];
    assert_profile_matches_full::<SRGB_DEPTH_ONLY>(D::RGB8_SRGB, D::RGBF32_LINEAR, opts, &rgb8, 2);
    assert_profile_rejects::<SRGB_DEPTH_ONLY>(D::RGBA8_SRGB, D::RGB8_SRGB, opts);
    assert_profile_rejects::<SRGB_DEPTH_ONLY>(D::GRAY8_SRGB, D::RGB8_SRGB, opts);
}

#[test]
fn profiles_capability_gates_attached_icc_and_cicp_contexts() {
    use zenpixels::{Cicp, ColorContext, PixelEncoding};
    const SDR_WEB_DESC_ONLY: u64 = caps::SDR_WEB & !caps::PROFILES;

    let opts = permissive_opts();
    let spec = ConversionSpec::new(opts);

    // Descriptor-only Display P3 -> sRGB still works identically without `PROFILES`
    // (native gamut matrix / fused matlut kernels do not require ICC/CMS).
    let p3_rgb8 = [255u8, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128];
    assert_profile_matches_full::<SDR_WEB_DESC_ONLY>(
        D::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3),
        D::RGB8_SRGB,
        opts,
        &p3_rgb8,
        4,
    );

    // Attached CICP/ICC ColorContext requires `caps::PROFILES`.
    let ctx = ColorContext::from_cicp(Cicp::SRGB);
    let src_with_ctx = PixelEncoding::new(D::RGBA8_SRGB, Some(&ctx));
    let dst_plain: PixelEncoding<'_> = D::RGB8_SRGB.into();

    // Cleared `PROFILES` refuses attached profiles with `NeedsCms`.
    let err_layout =
        ConverterProfile::<{ caps::LAYOUT_U8 }>::build(src_with_ctx, dst_plain, &spec).unwrap_err();
    assert!(
        matches!(err_layout.error(), ConvertError::NeedsCms { .. }),
        "expected NeedsCms on LAYOUT_U8 with attached profile, got {:?}",
        err_layout.error()
    );
    let err_sdr_desc =
        ConverterProfile::<SDR_WEB_DESC_ONLY>::build(src_with_ctx, dst_plain, &spec).unwrap_err();
    assert!(
        matches!(err_sdr_desc.error(), ConvertError::NeedsCms { .. }),
        "expected NeedsCms on SDR_WEB & !PROFILES with attached profile, got {:?}",
        err_sdr_desc.error()
    );

    // `SDR_WEB` (which includes `PROFILES`) accepts the same attached profile.
    let plan = ConverterProfile::<{ caps::SDR_WEB }>::build(src_with_ctx, dst_plain, &spec)
        .expect("SDR_WEB includes PROFILES and should accept CICP context");
    let mut worker = plan.prepare(2).unwrap();
    let src = [10u8, 20, 30, 255, 40, 50, 60, 255];
    let mut dst = [0u8; 6];
    worker.try_convert_row(&src, &mut dst, 2).unwrap();
    assert_eq!(dst, [10, 20, 30, 40, 50, 60]);
}

#[test]
fn layout_only_fast_planner_matches_full_across_u8_matrix() {
    const LAYOUT_GRAY_PREMUL_U8: u64 = caps::U8 | caps::LAYOUT | caps::GRAY | caps::PREMUL;

    let formats = [
        D::RGB8_SRGB,
        D::RGBA8_SRGB,
        D::BGRA8_SRGB,
        D::new(
            ChannelType::U8,
            ChannelLayout::Bgr,
            None,
            TransferFunction::Srgb,
        ),
        D::RGBX8_SRGB,
        D::BGRX8_SRGB,
        D::GRAY8_SRGB,
        D::GRAYA8_SRGB,
        D::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied)),
        D::BGRA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied)),
        D::RGBA8_SRGB.with_alpha(Some(AlphaMode::Opaque)),
    ];
    let opts = permissive_opts();

    for &from in &formats {
        for &to in &formats {
            let full = ConvertPlan::build(from.into(), to.into(), &ConversionSpec::new(opts));
            let fast = ConverterProfile::<LAYOUT_GRAY_PREMUL_U8>::build(
                from.into(),
                to.into(),
                &ConversionSpec::new(opts),
            );
            assert_eq!(
                full.is_ok(),
                fast.is_ok(),
                "admission mismatch for {from:?} -> {to:?}: full={full:?} fast={fast:?}"
            );
            if full.is_ok() {
                let src: Vec<u8> = (0..(4 * from.bytes_per_pixel()))
                    .map(|i| ((i * 37 + 13) % 256) as u8)
                    .collect();
                assert_profile_matches_full::<LAYOUT_GRAY_PREMUL_U8>(from, to, opts, &src, 4);
            }

            // Also verify `preserve_samples` admission parity.
            let full_exact =
                ConvertPlan::build(from.into(), to.into(), &ConversionSpec::preserve_samples());
            let fast_exact = ConverterProfile::<LAYOUT_GRAY_PREMUL_U8>::build(
                from.into(),
                to.into(),
                &ConversionSpec::preserve_samples(),
            );
            assert_eq!(
                full_exact.is_ok(),
                fast_exact.is_ok(),
                "preserve_samples mismatch for {from:?} -> {to:?}: full={full_exact:?} fast={fast_exact:?}"
            );
        }
    }

    // Policy error parity on layout-only path.
    let err = ConverterProfile::<{ caps::LAYOUT_U8 }>::build(
        D::RGBA8_SRGB.into(),
        D::RGB8_SRGB.into(),
        &ConversionSpec::new(ConvertOptions::retain_channels()),
    )
    .unwrap_err();
    assert!(matches!(err.error(), ConvertError::AlphaRemovalForbidden));

    let err_tf = ConverterProfile::<{ caps::LAYOUT_U8 }>::build(
        D::RGB8_SRGB.with_transfer(TransferFunction::Unknown).into(),
        D::RGB8_SRGB.into(),
        &ConversionSpec::new(opts),
    )
    .unwrap_err();
    assert!(matches!(
        err_tf.error(),
        ConvertError::UnknownTransferRelabelForbidden { .. }
    ));

    let err_facts = ConverterProfile::<{ caps::LAYOUT_U8 }>::build(
        D::RGBA8_SRGB.into(),
        D::RGB8_SRGB.into(),
        &ConversionSpec::new(opts).with_float_facts(zenpixels::sample::FloatFacts::NO_NAN),
    )
    .unwrap_err();
    assert!(matches!(
        err_facts.error(),
        ConvertError::FloatFactsUnsatisfied { .. }
    ));
}
