//! Integration tests for the universal fast-FMA scalar core, `const Accelerators`
//! overrides, `ConversionSpec::fast`, and the process-wide `accel` registry.
//!
//! Verifies:
//! 1. 0-bit parity (`Accelerators::SCALAR` == `Accelerators::ALL`) across all native
//!    conversion families (layout, gray, premul, matte, depth, sRGB/BT.709/Gamma22/PQ/HLG,
//!    gamut matrices, fused sRGB gamut, Oklab, `preserve_samples`, `ColorContext`).
//! 2. Zero `NoPath` surprises for every granular `Accelerators` preset (`SCALAR`, `LAYOUT`,
//!    `WEB_SDR`, `ALL`, `.with_*()`).
//! 3. `accel::enable_*()` process-wide registry progression and hermetic
//!    `with_accelerators(&Accelerators::SCALAR)` override.

use std::sync::Arc;
use zenpixels::hdr::DiffuseWhite;
use zenpixels::sample::FloatFacts;
use zenpixels::{
    AlphaMode, AlphaPolicy, ChannelLayout, ChannelType, Cicp, Color, ColorContext, ColorPrimaries,
    ConvertOptions, DepthPolicy, GamutPolicy, LumaCoefficients, PixelDescriptor as D,
    PixelEncoding, TransferFunction,
};
use zenpixels_convert::{Accelerators, ConversionSpec, ConvertPlan, accel};

fn perm_opts() -> ConvertOptions {
    ConvertOptions::retain_channels()
        .with_alpha_policy(AlphaPolicy::DiscardUnchecked)
        .with_depth_policy(DepthPolicy::Round)
        .with_luma(Some(LumaCoefficients::Bt709))
}

fn run_with(
    from: PixelEncoding<'_>,
    to: PixelEncoding<'_>,
    spec: &ConversionSpec<'_>,
    src: &[u8],
    width: u32,
) -> Vec<u8> {
    let mut conv = ConvertPlan::build(from, to, spec)
        .unwrap_or_else(|e| {
            panic!(
                "build failed {:?} -> {:?}: {e}",
                from.descriptor(),
                to.descriptor()
            )
        })
        .prepare(width)
        .unwrap();
    let mut dst = vec![0u8; width as usize * to.descriptor().bytes_per_pixel()];
    conv.try_convert_row(src, &mut dst, width).unwrap();
    dst
}

fn assert_bit_exact(
    label: &str,
    from: PixelEncoding<'_>,
    to: PixelEncoding<'_>,
    opts: ConvertOptions,
    src: &[u8],
    width: u32,
) {
    let scalar_spec = ConversionSpec::new(opts).with_accelerators(&Accelerators::SCALAR);
    let fast_spec = ConversionSpec::fast(opts);
    let web_spec = ConversionSpec::new(opts).with_accelerators(&Accelerators::WEB_SDR);
    let layout_spec = ConversionSpec::new(opts).with_accelerators(&Accelerators::LAYOUT);

    let want = run_with(from, to, &fast_spec, src, width);
    let got_scalar = run_with(from, to, &scalar_spec, src, width);
    let got_web = run_with(from, to, &web_spec, src, width);
    let got_layout = run_with(from, to, &layout_spec, src, width);

    assert_eq!(
        got_scalar, want,
        "{label}: Accelerators::SCALAR disagreed with Accelerators::ALL"
    );
    assert_eq!(
        got_web, want,
        "{label}: Accelerators::WEB_SDR disagreed with Accelerators::ALL"
    );
    assert_eq!(
        got_layout, want,
        "{label}: Accelerators::LAYOUT disagreed with Accelerators::ALL"
    );
}

#[test]
fn universal_scalar_matches_simd_bit_for_bit_across_all_families() {
    let opts = perm_opts();
    let w = 64u32;
    let wu = w as usize;
    let u8_4: Vec<u8> = (0..wu * 4).map(|i| ((i * 37 + 11) % 256) as u8).collect();
    let u8_3: Vec<u8> = (0..wu * 3).map(|i| ((i * 53 + 7) % 256) as u8).collect();
    let u16_3: Vec<u16> = (0..wu * 3)
        .map(|i| ((i * 1021 + 17) % 65536) as u16)
        .collect();
    let u16_4: Vec<u16> = (0..wu * 4)
        .map(|i| ((i * 997 + 13) % 65536) as u16)
        .collect();
    let f32_3: Vec<f32> = (0..wu * 3)
        .map(|i| (i as f32) / ((wu * 3 - 1) as f32))
        .collect();
    let f32_4: Vec<f32> = (0..wu * 4)
        .map(|i| (i as f32) / ((wu * 4 - 1) as f32))
        .collect();

    let srgb_f32 = D::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb);
    let bt709_f32 = D::RGBF32_LINEAR.with_transfer(TransferFunction::Bt709);
    let g22_f32 = D::RGBF32_LINEAR.with_transfer(TransferFunction::Gamma22);
    let pq_f32 = D::RGBF32_LINEAR
        .with_primaries(ColorPrimaries::Bt2020)
        .with_transfer(TransferFunction::Pq);
    let hlg_f32 = D::RGBF32_LINEAR
        .with_primaries(ColorPrimaries::Bt2020)
        .with_transfer(TransferFunction::Hlg);
    let lin_2020 = D::RGBF32_LINEAR.with_primaries(ColorPrimaries::Bt2020);
    let p3_u8 = D::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
    let p3_u16 = D::RGB16_SRGB.with_primaries(ColorPrimaries::DisplayP3);
    let p3_f32 = D::RGBF32_LINEAR.with_primaries(ColorPrimaries::DisplayP3);
    let adobe_u8 = D::RGB8_SRGB
        .with_primaries(ColorPrimaries::AdobeRgb)
        .with_transfer(TransferFunction::Gamma22);
    let bgr8 = D::new(
        ChannelType::U8,
        ChannelLayout::Bgr,
        None,
        TransferFunction::Srgb,
    );
    let rgba8_premul = D::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied));
    let rgba16_premul = D::RGBA16_SRGB.with_alpha(Some(AlphaMode::Premultiplied));
    let rgbaf32_premul = D::RGBAF32_LINEAR.with_alpha(Some(AlphaMode::Premultiplied));
    let matte_opts = opts.with_alpha_policy(AlphaPolicy::CompositeOnto {
        background: Color::srgb8(240, 200, 160),
        reference_white: None,
    });
    let hdr_opts = opts.with_sdr_white(DiffuseWhite::BT2408);
    let ext_opts = opts.with_gamut_policy(GamutPolicy::PreserveExtended);

    // Layout, gray, premul, matte
    assert_bit_exact(
        "RGBA8->RGB8",
        D::RGBA8_SRGB.into(),
        D::RGB8_SRGB.into(),
        opts,
        &u8_4,
        w,
    );
    assert_bit_exact(
        "RGB8->BGRA8",
        D::RGB8_SRGB.into(),
        D::BGRA8_SRGB.into(),
        opts,
        &u8_3,
        w,
    );
    assert_bit_exact(
        "BGR8->RGB8",
        bgr8.into(),
        D::RGB8_SRGB.into(),
        opts,
        &u8_3,
        w,
    );
    assert_bit_exact(
        "RGBA8->BGRA8",
        D::RGBA8_SRGB.into(),
        D::BGRA8_SRGB.into(),
        opts,
        &u8_4,
        w,
    );
    assert_bit_exact(
        "RGB8->GRAY8",
        D::RGB8_SRGB.into(),
        D::GRAY8_SRGB.into(),
        opts,
        &u8_3,
        w,
    );
    assert_bit_exact(
        "RGBA8 straight->premul",
        D::RGBA8_SRGB.into(),
        rgba8_premul.into(),
        opts,
        &u8_4,
        w,
    );
    assert_bit_exact(
        "RGBA8 premul->straight",
        rgba8_premul.into(),
        D::RGBA8_SRGB.into(),
        opts,
        &u8_4,
        w,
    );
    assert_bit_exact(
        "RGBA16 straight->premul",
        D::RGBA16_SRGB.into(),
        rgba16_premul.into(),
        opts,
        bytemuck::cast_slice(&u16_4),
        w,
    );
    assert_bit_exact(
        "RGBA16 premul->straight",
        rgba16_premul.into(),
        D::RGBA16_SRGB.into(),
        opts,
        bytemuck::cast_slice(&u16_4),
        w,
    );
    assert_bit_exact(
        "RGBAF32 straight->premul",
        D::RGBAF32_LINEAR.into(),
        rgbaf32_premul.into(),
        opts,
        bytemuck::cast_slice(&f32_4),
        w,
    );
    assert_bit_exact(
        "RGBAF32 premul->straight",
        rgbaf32_premul.into(),
        D::RGBAF32_LINEAR.into(),
        opts,
        bytemuck::cast_slice(&f32_4),
        w,
    );
    assert_bit_exact(
        "RGBA8->RGB8 matte sRGB",
        D::RGBA8_SRGB.into(),
        D::RGB8_SRGB.into(),
        matte_opts,
        &u8_4,
        w,
    );
    assert_bit_exact(
        "RGBA16->RGB16 matte sRGB",
        D::RGBA16_SRGB.into(),
        D::RGB16_SRGB.into(),
        matte_opts,
        bytemuck::cast_slice(&u16_4),
        w,
    );

    // Depth conversions
    assert_bit_exact(
        "RGB8->RGB16",
        D::RGB8_SRGB.into(),
        D::RGB16_SRGB.into(),
        opts,
        &u8_3,
        w,
    );
    assert_bit_exact(
        "RGB16->RGB8 round",
        D::RGB16_SRGB.into(),
        D::RGB8_SRGB.into(),
        opts,
        bytemuck::cast_slice(&u16_3),
        w,
    );
    assert_bit_exact(
        "RGB16->RGB8 trunc",
        D::RGB16_SRGB.into(),
        D::RGB8_SRGB.into(),
        opts.with_depth_policy(DepthPolicy::Truncate),
        bytemuck::cast_slice(&u16_3),
        w,
    );
    assert_bit_exact(
        "RGBF32->RGBF16",
        D::RGBF32_LINEAR.into(),
        D::RGBF16_LINEAR.into(),
        opts,
        bytemuck::cast_slice(&f32_3),
        w,
    );
    let rgb8_lin = D::RGB8_SRGB.with_transfer(TransferFunction::Linear);
    assert_bit_exact(
        "RGB8->RGBF32 naive",
        rgb8_lin.into(),
        D::RGBF32_LINEAR.into(),
        opts,
        &u8_3,
        w,
    );
    assert_bit_exact(
        "RGBF32->RGB8 naive",
        D::RGBF32_LINEAR.into(),
        rgb8_lin.into(),
        opts,
        bytemuck::cast_slice(&f32_3),
        w,
    );

    // SDR transfers
    assert_bit_exact(
        "RGB8 sRGB->RGBF32 lin",
        D::RGB8_SRGB.into(),
        D::RGBF32_LINEAR.into(),
        opts,
        &u8_3,
        w,
    );
    assert_bit_exact(
        "RGBF32 lin->RGB8 sRGB",
        D::RGBF32_LINEAR.into(),
        D::RGB8_SRGB.into(),
        opts,
        bytemuck::cast_slice(&f32_3),
        w,
    );
    assert_bit_exact(
        "RGBF32 sRGB->lin",
        srgb_f32.into(),
        D::RGBF32_LINEAR.into(),
        opts,
        bytemuck::cast_slice(&f32_3),
        w,
    );
    assert_bit_exact(
        "RGBF32 lin->sRGB",
        D::RGBF32_LINEAR.into(),
        srgb_f32.into(),
        opts,
        bytemuck::cast_slice(&f32_3),
        w,
    );
    assert_bit_exact(
        "RGBF32 lin->sRGB ext",
        D::RGBF32_LINEAR.into(),
        srgb_f32.into(),
        ext_opts,
        bytemuck::cast_slice(&f32_3),
        w,
    );
    assert_bit_exact(
        "RGBF32 BT709->lin",
        bt709_f32.into(),
        D::RGBF32_LINEAR.into(),
        opts,
        bytemuck::cast_slice(&f32_3),
        w,
    );
    assert_bit_exact(
        "RGBF32 lin->BT709",
        D::RGBF32_LINEAR.into(),
        bt709_f32.into(),
        opts,
        bytemuck::cast_slice(&f32_3),
        w,
    );
    assert_bit_exact(
        "RGBF32 Gamma22->lin",
        g22_f32.into(),
        D::RGBF32_LINEAR.into(),
        opts,
        bytemuck::cast_slice(&f32_3),
        w,
    );
    assert_bit_exact(
        "RGBF32 lin->Gamma22",
        D::RGBF32_LINEAR.into(),
        g22_f32.into(),
        opts,
        bytemuck::cast_slice(&f32_3),
        w,
    );

    // HDR transfers
    #[cfg(feature = "hdr-transfers")]
    {
        assert_bit_exact(
            "RGBF32 lin->PQ",
            lin_2020.into(),
            pq_f32.into(),
            hdr_opts,
            bytemuck::cast_slice(&f32_3),
            w,
        );
        assert_bit_exact(
            "RGB16 PQ->lin",
            D::RGB16_BT2100_PQ.into(),
            lin_2020.into(),
            hdr_opts,
            bytemuck::cast_slice(&u16_3),
            w,
        );
        assert_bit_exact(
            "RGB16 HLG->RGBF32 HLG",
            D::RGB16_BT2100_HLG.into(),
            hlg_f32.into(),
            opts,
            bytemuck::cast_slice(&u16_3),
            w,
        );
    }

    // Gamut conversions (fused U8/U16 + F32 matrix + cross-TF)
    assert_bit_exact(
        "P3 RGB8->sRGB RGB8",
        p3_u8.into(),
        D::RGB8_SRGB.into(),
        opts,
        &u8_3,
        w,
    );
    assert_bit_exact(
        "P3 RGB16->sRGB RGB16",
        p3_u16.into(),
        D::RGB16_SRGB.into(),
        opts,
        bytemuck::cast_slice(&u16_3),
        w,
    );
    assert_bit_exact(
        "P3 RGBF32->sRGB RGBF32",
        p3_f32.into(),
        D::RGBF32_LINEAR.into(),
        opts,
        bytemuck::cast_slice(&f32_3),
        w,
    );
    assert_bit_exact(
        "AdobeRGB8->sRGB8",
        adobe_u8.into(),
        D::RGB8_SRGB.into(),
        opts,
        &u8_3,
        w,
    );

    // Oklab
    #[cfg(feature = "oklab")]
    {
        assert_bit_exact(
            "RGBF32 lin->Oklab",
            D::RGBF32_LINEAR.into(),
            D::OKLABF32.into(),
            opts,
            bytemuck::cast_slice(&f32_3),
            w,
        );
        assert_bit_exact(
            "RGBAF32 lin->OklabA",
            D::RGBAF32_LINEAR.into(),
            D::OKLABAF32.into(),
            opts,
            bytemuck::cast_slice(&f32_4),
            w,
        );
    }

    // ColorContext (CICP Display P3 -> sRGB)
    let p3_ctx = Arc::new(ColorContext::from_cicp(Cicp::DISPLAY_P3));
    let src_ctx = PixelEncoding::new(p3_u8, Some(&p3_ctx));
    assert_bit_exact(
        "CICP P3 RGB8->sRGB RGB8",
        src_ctx,
        D::RGB8_SRGB.into(),
        opts,
        &u8_3,
        w,
    );
}

#[test]
fn float_facts_and_preserve_samples_consistent_between_scalar_and_fast() {
    let opts = perm_opts();
    let scalar_plan = ConvertPlan::build(
        D::RGB8_SRGB.into(),
        D::RGBF32_LINEAR.into(),
        &ConversionSpec::new(opts).with_accelerators(&Accelerators::SCALAR),
    )
    .unwrap();
    let fast_plan = ConvertPlan::build(
        D::RGB8_SRGB.into(),
        D::RGBF32_LINEAR.into(),
        &ConversionSpec::fast(opts),
    )
    .unwrap();
    assert_eq!(
        scalar_plan.output_float_facts(),
        fast_plan.output_float_facts()
    );
    assert!(scalar_plan.output_float_facts().contains(
        FloatFacts::NO_NAN
            | FloatFacts::NO_INFINITE
            | FloatFacts::NON_NEGATIVE
            | FloatFacts::AT_MOST_ONE
    ));

    // preserve_samples accepts reversible RGB8 -> RGBA16 and refuses RGBA8 -> RGB8 in both modes.
    for acc in [&Accelerators::SCALAR, &Accelerators::ALL] {
        let spec = ConversionSpec::preserve_samples().with_accelerators(acc);
        assert!(ConvertPlan::build(D::RGB8_SRGB.into(), D::RGBA16_SRGB.into(), &spec).is_ok());
        assert!(ConvertPlan::build(D::RGBA8_SRGB.into(), D::RGB8_SRGB.into(), &spec).is_err());
    }
}

#[test]
fn process_wide_registry_enables_cleanly_and_scalar_override_stays_hermetic() {
    let opts = perm_opts();
    let src: Vec<u8> = (0..64 * 3).map(|i| ((i * 29 + 3) % 256) as u8).collect();
    let p3_u8 = D::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);

    // Baseline before registering everything.
    let baseline = run_with(
        p3_u8.into(),
        D::RGB8_SRGB.into(),
        &ConversionSpec::new(opts),
        &src,
        64,
    );

    // Stepwise global registration: layout -> web_sdr -> all.
    accel::enable_layout();
    assert!(accel::enabled().has_layout());
    accel::enable_web_sdr();
    assert!(accel::enabled().has_srgb());
    assert!(accel::enabled().has_gamut());
    accel::enable_all();
    assert!(accel::enabled().is_all());

    let after_enable_all = run_with(
        p3_u8.into(),
        D::RGB8_SRGB.into(),
        &ConversionSpec::new(opts),
        &src,
        64,
    );
    let forced_scalar = run_with(
        p3_u8.into(),
        D::RGB8_SRGB.into(),
        &ConversionSpec::new(opts).with_accelerators(&Accelerators::SCALAR),
        &src,
        64,
    );
    assert_eq!(baseline, after_enable_all);
    assert_eq!(baseline, forced_scalar);
}
