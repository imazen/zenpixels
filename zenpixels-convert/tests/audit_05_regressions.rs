//! Comprehensive regression tests for zenpixels-convert 0.5 audit findings
//! (batches 1-3: planner ordering, OutputPlan validation, CMS policy/intent/alpha/gray,
//! exhaustive SIMD RGBA8 premul & F16 sNaN parity, BT.2100-3 narrow scaling,
//! RGBA/BGRA alpha parity, unpremul subnormal, matte-before-clip, HLG swizzle parity,
//! diffuse white rescale, Transfer::Unknown guard, Bt2446A baked peaks, F32->F16 facts,
//! scan_facts F16, YCbCr Cr validation, GainMap validation, orient color & 1x1 stride).

use std::sync::Arc;

use zenpixels_convert::{
    Accelerators, AlphaMode, AlphaPolicy, BitPacking, ChannelOrder, Cicp, Color, ColorOrigin,
    ColorSpace, ConvertError, ConvertPolicy, DepthPolicy, DiffuseWhite, F16Sample, FloatFacts,
    GamutPolicy, ImagePlan, ImageSpec, Orientation, OutputPlan, OutputSignaling, PeakSource,
    PixelBuf, PlanarView, PlaneRole, PlaneSpec, PlaneView, Primaries, RowPlan, SampleFacts,
    SampleLayout, SampleType, SignalRange, Transfer, YcbcrMatrix, apply_orientation,
    apply_orientation_into, convert_into, convert_to_buf, convert_to_buf_with_peak,
    reconstruct_planar_f32, scan_facts,
};

#[test]
fn exhaustive_256x256_rgba8_straight_to_premul_simd_matches_scalar_and_exact_oracle() {
    let mut src_bytes = Vec::with_capacity(256 * 256 * 4);
    for a in 0u16..=255 {
        for c in 0u16..=255 {
            src_bytes.push(c as u8);
            src_bytes.push(((c + 73) & 0xFF) as u8);
            src_bytes.push((255 - c) as u8);
            src_bytes.push(a as u8);
        }
    }
    let src_spec = ImageSpec::RGBA8_SRGB;
    let dst_spec = ImageSpec::RGBA8_SRGB.with_alpha(AlphaMode::Premultiplied);
    let src = PixelBuf::from_typed(src_bytes, 256, 256, src_spec).unwrap();

    for (depth_policy, round) in [(DepthPolicy::Round, true), (DepthPolicy::Truncate, false)] {
        let pol_scalar = ConvertPolicy::permissive()
            .with_depth(depth_policy)
            .with_accelerators(Accelerators::SCALAR);
        let pol_simd = ConvertPolicy::permissive()
            .with_depth(depth_policy)
            .with_accelerators(Accelerators::ALL);

        let out_scalar = convert_to_buf(&src.as_view(), dst_spec.clone(), &pol_scalar).unwrap();
        let out_simd = convert_to_buf(&src.as_view(), dst_spec.clone(), &pol_simd).unwrap();

        for y in 0..256u32 {
            let rs = out_scalar.as_view().row(y);
            let rv = out_simd.as_view().row(y);
            assert_eq!(rs, rv, "SIMD vs scalar mismatch at row {y} (round={round})");
            let a = y;
            for x in 0..256u32 {
                let i = (x as usize) * 4;
                let c0 = x;
                let c1 = (x + 73) & 0xFF;
                let c2 = 255 - x;
                let oracle = |c: u32| -> u8 {
                    if round {
                        ((c * a + 127) / 255) as u8
                    } else {
                        ((c * a) / 255) as u8
                    }
                };
                assert_eq!(rv[i], oracle(c0));
                assert_eq!(rv[i + 1], oracle(c1));
                assert_eq!(rv[i + 2], oracle(c2));
                assert_eq!(rv[i + 3], a as u8);
            }
        }
    }
}

#[test]
fn exhaustive_f16_including_signaling_nan_zero_bit_parity_simd_vs_scalar() {
    // 65,536 u16 patterns packed into 256x64 RGBAF16 (4 channels per pixel = 65536 f16s).
    let mut raw = Vec::with_capacity(65536 * 2);
    for bits in 0u16..=65535 {
        raw.extend_from_slice(&bits.to_ne_bytes());
    }
    let src_spec = ImageSpec::RGBAF16_LINEAR;
    let dst_spec = ImageSpec::RGBAF32_LINEAR;
    let src = PixelBuf::from_bytes(raw, 256, 64, src_spec).unwrap();

    let out_scalar = convert_to_buf(
        &src.as_view(),
        dst_spec.clone(),
        &ConvertPolicy::permissive().with_accelerators(Accelerators::SCALAR),
    )
    .unwrap();
    let out_simd = convert_to_buf(
        &src.as_view(),
        dst_spec,
        &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
    )
    .unwrap();

    for y in 0..64u32 {
        let rs = out_scalar.as_view().row_as::<u32>(y).unwrap();
        let rv = out_simd.as_view().row_as::<u32>(y).unwrap();
        assert_eq!(
            rs, rv,
            "F16->F32 bit disagreement between SCALAR and ALL on row {y}"
        );
    }

    // Explicitly verify signaling NaN 0x7c01 was quieted identically (qNaN bit 0x0040_0000 set).
    let idx = 0x7c01usize;
    let y = (idx / (256 * 4)) as u32;
    let c = idx % (256 * 4);
    let bits = out_scalar.as_view().row_as::<u32>(y).unwrap()[c];
    assert_ne!(
        bits & 0x0040_0000,
        0,
        "F16 sNaN 0x7c01 must quiet to qNaN in F32 (got 0x{bits:08x})"
    );
}

#[test]
fn narrow_range_u8_to_u16_and_p010_follow_bt2100_table_9() {
    // 1. RGB8 Narrow -> RGB16 Narrow: black 16 -> 4096 (16 << 8), white 235 -> 60160 (235 << 8).
    let narrow_rgb = ColorSpace::SRGB.with_range(SignalRange::Narrow);
    let src = PixelBuf::from_typed(
        vec![16u8, 128, 235],
        1,
        1,
        ImageSpec::new(SampleLayout::RGB8, narrow_rgb.clone()),
    )
    .unwrap();
    let dst_spec = ImageSpec::new(SampleLayout::RGB16, narrow_rgb);
    let out = convert_to_buf(&src.as_view(), dst_spec, &ConvertPolicy::permissive()).unwrap();
    let px = out.as_view().row_as::<u16>(0).unwrap();
    assert_eq!(px[0], 16 << 8, "narrow 8-bit black 16 must map to 4096");
    assert_eq!(px[2], 235 << 8, "narrow 8-bit white 235 must map to 60160");

    // 2. P010 (10-bit MSB in U16) Narrow -> RGBF32 Full: 10-bit black 64 -> 0.0, white 940 -> 1.0.
    let p010_layout = SampleLayout::RGB16.with_packing(BitPacking::P010);
    let p010_narrow = ImageSpec::new(
        p010_layout,
        ColorSpace::Rgb {
            primaries: Primaries::Bt2020,
            transfer: Transfer::Linear,
            range: SignalRange::Narrow,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let mut raw = Vec::new();
    for code10 in [64u16, 502u16, 940u16] {
        raw.extend_from_slice(&(code10 << 6).to_ne_bytes());
    }
    let src_p010 = PixelBuf::from_bytes(raw, 1, 1, p010_narrow).unwrap();
    let dst_f32 = ImageSpec::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: Primaries::Bt2020,
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let out_f32 =
        convert_to_buf(&src_p010.as_view(), dst_f32, &ConvertPolicy::permissive()).unwrap();
    let f = out_f32.as_view().row_as::<f32>(0).unwrap();
    assert!(
        (f[0] - 0.0).abs() < 1e-6,
        "P010 code 64 -> 0.0, got {}",
        f[0]
    );
    assert!(
        (f[1] - 0.5).abs() < 1e-6,
        "P010 code 502 -> 0.5, got {}",
        f[1]
    );
    assert!(
        (f[2] - 1.0).abs() < 1e-6,
        "P010 code 940 -> 1.0, got {}",
        f[2]
    );
}

#[test]
fn rgba_and_bgra_srgb_u8_to_lin_f32_alpha_and_fused_lin_to_srgb_zero_bit_parity() {
    let mut rgba_bytes = Vec::with_capacity(256 * 4);
    let mut bgra_bytes = Vec::with_capacity(256 * 4);
    for i in 0u16..=255 {
        let r = i as u8;
        let g = (255 - i) as u8;
        let b = ((i * 3) & 0xFF) as u8;
        let a = i as u8;
        rgba_bytes.extend_from_slice(&[r, g, b, a]);
        bgra_bytes.extend_from_slice(&[b, g, r, a]);
    }
    let src_rgba = PixelBuf::from_typed(rgba_bytes, 256, 1, ImageSpec::RGBA8_SRGB).unwrap();
    let src_bgra = PixelBuf::from_typed(bgra_bytes, 256, 1, ImageSpec::BGRA8_SRGB).unwrap();

    let pol_s = ConvertPolicy::permissive().with_accelerators(Accelerators::SCALAR);
    let pol_v = ConvertPolicy::permissive().with_accelerators(Accelerators::ALL);

    let r_s = convert_to_buf(&src_rgba.as_view(), ImageSpec::RGBAF32_LINEAR, &pol_s).unwrap();
    let r_v = convert_to_buf(&src_rgba.as_view(), ImageSpec::RGBAF32_LINEAR, &pol_v).unwrap();
    let b_s = convert_to_buf(&src_bgra.as_view(), ImageSpec::RGBAF32_LINEAR, &pol_s).unwrap();
    let b_v = convert_to_buf(&src_bgra.as_view(), ImageSpec::RGBAF32_LINEAR, &pol_v).unwrap();

    let rs = r_s.as_view().row_as::<u32>(0).unwrap();
    let rv = r_v.as_view().row_as::<u32>(0).unwrap();
    let bs = b_s.as_view().row_as::<u32>(0).unwrap();
    let bv = b_v.as_view().row_as::<u32>(0).unwrap();

    assert_eq!(rs, rv, "RGBA sRGB U8->LinF32 SCALAR vs ALL mismatch");
    assert_eq!(bs, bv, "BGRA sRGB U8->LinF32 SCALAR vs ALL mismatch");
    assert_eq!(
        rs, bs,
        "RGBA vs BGRA sRGB U8->LinF32 mismatch (alpha ULP drift)"
    );

    // Fused LinF32 -> SrgbU8 parity (RGBA, BGRA) including negative, >1, NaN.
    let mut lin_raw = Vec::new();
    let test_vals: [f32; 8] = [-0.5, 0.0, 0.0031308, 0.18, 0.5, 1.0, 1.5, f32::NAN];
    for i in 0..37usize {
        let r = test_vals[i % 8];
        let g = test_vals[(i + 2) % 8];
        let b = test_vals[(i + 5) % 8];
        let a = (i as f32) / 36.0;
        for v in [r, g, b, a] {
            lin_raw.extend_from_slice(&v.to_ne_bytes());
        }
    }
    let lin_buf = PixelBuf::from_bytes(lin_raw, 37, 1, ImageSpec::RGBAF32_LINEAR).unwrap();
    for dst_spec in [ImageSpec::RGBA8_SRGB, ImageSpec::BGRA8_SRGB] {
        let os = convert_to_buf(&lin_buf.as_view(), dst_spec.clone(), &pol_s).unwrap();
        let ov = convert_to_buf(&lin_buf.as_view(), dst_spec.clone(), &pol_v).unwrap();
        assert_eq!(
            os.as_view().row(0),
            ov.as_view().row(0),
            "fused LinF32->SrgbU8 mismatch for {:?}",
            dst_spec.layout
        );
    }
}

#[test]
fn unpremul_subnormal_alpha_no_nan_and_opaque_alpha_canonicalization() {
    // 1. Subnormal alpha in Unpremul must not produce NaN.
    let subnormal = f32::from_bits(1);
    let mut raw = Vec::new();
    for v in [0.0f32, subnormal, 1.0e-38, subnormal] {
        raw.extend_from_slice(&v.to_ne_bytes());
    }
    let premul_spec = ImageSpec::RGBAF32_LINEAR.with_alpha(AlphaMode::Premultiplied);
    let src = PixelBuf::from_bytes(raw, 1, 1, premul_spec).unwrap();
    let out = convert_to_buf(
        &src.as_view(),
        ImageSpec::RGBAF32_LINEAR,
        &ConvertPolicy::permissive(),
    )
    .unwrap();
    let px = out.as_view().row_as::<f32>(0).unwrap();
    for (i, &v) in px.iter().enumerate() {
        assert!(
            !v.is_nan(),
            "unpremul subnormal produced NaN at ch {i}: {px:?}"
        );
    }

    // 2. AlphaMode::Opaque canonicalizes junk alpha to 255 / 1.0 on BOTH byte fast path and strip path.
    let opaque_u8 = ImageSpec::RGBA8_SRGB.with_alpha(AlphaMode::Opaque);
    let src_op =
        PixelBuf::from_typed(vec![10u8, 20, 30, 0, 40, 50, 60, 17], 2, 1, opaque_u8).unwrap();
    let dst_straight = convert_to_buf(
        &src_op.as_view(),
        ImageSpec::RGBA8_SRGB,
        &ConvertPolicy::permissive(),
    )
    .unwrap();
    assert_eq!(
        dst_straight.as_view().row(0),
        &[10, 20, 30, 255, 40, 50, 60, 255]
    );

    let dst_lin = convert_to_buf(
        &src_op.as_view(),
        ImageSpec::RGBAF32_LINEAR,
        &ConvertPolicy::permissive(),
    )
    .unwrap();
    let f = dst_lin.as_view().row_as::<f32>(0).unwrap();
    assert_eq!(f[3], 1.0);
    assert_eq!(f[7], 1.0);
}

#[test]
fn output_plan_validates_spec_on_identity_and_signaling_contracts() {
    let p3_spec = ImageSpec::new(SampleLayout::RGB8, ColorSpace::DISPLAY_P3);
    let srgb_spec = ImageSpec::RGB8_SRGB;

    // 1. Identity plan rejects mismatched buffer/view spec!
    let id_plan = OutputPlan::build(
        srgb_spec.clone(),
        srgb_spec.clone(),
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputSignaling::Auto,
    )
    .unwrap();
    let wrong_buf = PixelBuf::new(2, 2, p3_spec.clone()).unwrap();
    assert!(id_plan.execute_buf(wrong_buf).is_err());

    // 2. SameAsOrigin rejected when color meaning changed (P3 -> sRGB), accepted when unchanged.
    assert!(
        OutputPlan::build(
            p3_spec.clone(),
            srgb_spec.clone(),
            &ConvertPolicy::permissive(),
            PeakSource::Measured,
            OutputSignaling::SameAsOrigin,
        )
        .is_err()
    );
    assert!(
        OutputPlan::build(
            p3_spec.clone(),
            p3_spec,
            &ConvertPolicy::permissive(),
            PeakSource::Measured,
            OutputSignaling::SameAsOrigin,
        )
        .is_ok()
    );

    // 3. CicpOnly rejected when CICP is unavailable (AdobeRgb has no standard CICP primaries!).
    let adobe_spec = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::SRGB.with_primaries(Primaries::AdobeRgb),
    );
    let err_cicp = OutputPlan::build(
        adobe_spec.clone(),
        adobe_spec,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputSignaling::CicpOnly,
    );
    assert!(matches!(
        err_cicp,
        Err(ref e) if matches!(e.error(), ConvertError::OutputSignalingUnavailable)
    ));

    // 4. Auto on Narrow range never attaches a full-range ICC profile.
    let narrow_spec = ImageSpec::RGB8_SRGB.with_range(SignalRange::Narrow);
    let narrow_plan = OutputPlan::build(
        narrow_spec.clone(),
        narrow_spec.clone(),
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputSignaling::Auto,
    )
    .unwrap();
    let narrow_buf = PixelBuf::new(1, 1, narrow_spec).unwrap();
    let ready: zenpixels_convert::EncodeReady<'static> =
        narrow_plan.execute_buf(narrow_buf).unwrap();
    assert!(
        ready.icc.is_none(),
        "narrow range Auto must not attach full-range ICC"
    );
    assert_eq!(ready.cicp.map(|c| c.full_range), Some(false));

    // 5. CustomIcc with GRAY header rejected on RGB8 target.
    let mut gray_icc = vec![0u8; 128];
    gray_icc[16..20].copy_from_slice(b"GRAY");
    gray_icc[36..40].copy_from_slice(b"acsp");
    let bad_custom = OutputPlan::build(
        srgb_spec.clone(),
        srgb_spec,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputSignaling::CustomIcc(Arc::from(gray_icc.as_slice())),
    );
    assert!(bad_custom.is_err());

    // 6. HDR PQ -> SDR sRGB clears stale HdrMetadata on EncodeReady.
    #[cfg(feature = "bt2446a")]
    {
        use zenpixels_convert::{HdrMetadata, Nits};
        let hdr_spec = ImageSpec::new(SampleLayout::RGBF32, ColorSpace::BT2100_PQ);
        let hdr_buf = PixelBuf::new(1, 1, hdr_spec.clone())
            .unwrap()
            .with_hdr(HdrMetadata {
                cll: Some(zenpixels_convert::ContentLightLevel::new(1000, 400)),
                mastering: None,
            });
        let pol = ConvertPolicy::display_sdr(DiffuseWhite::BT2408, Nits::new(100.0));
        let plan = OutputPlan::build(
            hdr_spec,
            ImageSpec::RGB8_SRGB,
            &pol,
            PeakSource::Fixed(Nits::new(1000.0)),
            OutputSignaling::Auto,
        )
        .unwrap();
        let ready = plan.execute_buf(hdr_buf).unwrap();
        assert_eq!(
            ready.hdr,
            HdrMetadata::default(),
            "HDR->SDR OutputPlan must strip stale HdrMetadata"
        );
        assert_eq!(ready.cicp, Some(Cicp::SRGB));
    }
}

#[test]
fn policy_lossless_gates_chroma_bitfield_alpha_f16_mantissa_and_unproven_float() {
    let lossless = ConvertPolicy::lossless();

    // 1. RGB8 -> Gray8 rejected (chroma loss).
    assert!(RowPlan::build(ImageSpec::RGB8_SRGB, ImageSpec::GRAY8_SRGB, &lossless, None).is_err());

    // 2. RGBA8 -> RGB10A2 rejected (8-bit alpha narrowed to 2-bit alpha!).
    let rgb10a2 = ImageSpec::new(SampleLayout::RGB10A2, ColorSpace::SRGB_ALPHA);
    assert!(RowPlan::build(ImageSpec::RGBA8_SRGB, rgb10a2, &lossless, None).is_err());

    // 3. RGB16 -> RGBF16 rejected (16-bit integer precision > 11-bit F16 mantissa!).
    let rgb16 = ImageSpec::new(SampleLayout::RGB16, ColorSpace::LINEAR_SRGB);
    let rgbf16 = ImageSpec::RGBF16_LINEAR;
    assert!(RowPlan::build(rgb16, rgbf16, &lossless, None).is_err());

    // 4. Unproven RGBF32 -> RGB8 rejected under GamutPolicy::Forbid, allowed with UNIT_INTERVAL.
    let forbid_gamut = ConvertPolicy::permissive().with_gamut(GamutPolicy::Forbid);
    assert!(
        RowPlan::build(
            ImageSpec::RGBF32_LINEAR,
            ImageSpec::RGB8_SRGB,
            &forbid_gamut,
            None
        )
        .is_err()
    );
    let proven_pol =
        forbid_gamut.with_source_facts(SampleFacts::from_float(FloatFacts::UNIT_INTERVAL));
    assert!(
        RowPlan::build(
            ImageSpec::RGBF32_LINEAR,
            ImageSpec::RGB8_SRGB,
            &proven_pol,
            None
        )
        .is_ok()
    );

    // 5. ImagePlan::execute rejects mismatched source spec BEFORE allocation.
    let img_plan = ImagePlan::build(
        ImageSpec::RGB8_SRGB,
        ImageSpec::RGBA8_SRGB,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
    .unwrap();
    let wrong_src = PixelBuf::new(2, 2, ImageSpec::RGBA8_SRGB).unwrap();
    assert!(img_plan.execute(&wrong_src).is_err());
}

#[test]
fn photometry_matte_before_clip_hlg_swizzle_parity_white_rescale_unknown_transfer_and_facts() {
    // 1. Matte compositing BEFORE gamut clipping: linear (r=2.0, g=0, b=0, a=0.5) over black -> r=1.0!
    // (If clipped before matte, r=1.0 * 0.5 = 0.5.)
    let mut raw = Vec::new();
    for v in [2.0f32, 0.0, 0.0, 0.5] {
        raw.extend_from_slice(&v.to_ne_bytes());
    }
    let src = PixelBuf::from_bytes(raw, 1, 1, ImageSpec::RGBAF32_LINEAR).unwrap();
    let pol = ConvertPolicy::permissive()
        .with_alpha(AlphaPolicy::CompositeOnto(Color::BLACK))
        .with_gamut(GamutPolicy::Clip);
    let out = convert_to_buf(&src.as_view(), ImageSpec::RGBF32_LINEAR, &pol).unwrap();
    let px = out.as_view().row_as::<f32>(0).unwrap();
    assert!(
        (px[0] - 1.0).abs() < 1e-5,
        "matte must happen before clip: expected r=1.0, got {}",
        px[0]
    );

    // 2. HlgMode::Scene without HlgRendering fails identically for same-order and swizzled RGB/BGR!
    let hlg_scene_rgb = ImageSpec::new(SampleLayout::RGBF32, ColorSpace::BT2100_HLG_SCENE);
    let lin2020_rgb = ImageSpec::new(
        SampleLayout::RGBF32,
        ColorSpace::LINEAR_SRGB.with_primaries(Primaries::Bt2020),
    );
    let lin2020_bgr = ImageSpec::new(
        SampleLayout::new(SampleType::F32, ChannelOrder::Bgr),
        lin2020_rgb.color.clone(),
    );
    assert!(
        RowPlan::build(
            hlg_scene_rgb.clone(),
            lin2020_rgb,
            &ConvertPolicy::permissive(),
            None
        )
        .is_err(),
        "HLG Scene -> Linear without HlgRendering must fail"
    );
    assert!(
        RowPlan::build(
            hlg_scene_rgb,
            lin2020_bgr,
            &ConvertPolicy::permissive(),
            None
        )
        .is_err(),
        "HLG Scene -> swizzled Linear without HlgRendering must also fail"
    );

    // 3. Relative-to-relative diffuse white rescale (100 nits -> 203 nits).
    let lin_100 = ImageSpec::RGBF32_LINEAR.with_white(DiffuseWhite::new(100.0));
    let lin_203 = ImageSpec::RGBF32_LINEAR.with_white(DiffuseWhite::new(203.0));
    let mut w_raw = Vec::new();
    for v in [1.0f32, 0.5, 0.25] {
        w_raw.extend_from_slice(&v.to_ne_bytes());
    }
    let src_100 = PixelBuf::from_bytes(w_raw, 1, 1, lin_100).unwrap();
    let out_203 =
        convert_to_buf(&src_100.as_view(), lin_203, &ConvertPolicy::permissive()).unwrap();
    let f = out_203.as_view().row_as::<f32>(0).unwrap();
    let ratio = 100.0 / 203.0;
    assert!(
        (f[0] - ratio).abs() < 1e-5,
        "expected {ratio}, got {}",
        f[0]
    );
    assert!((f[1] - 0.5 * ratio).abs() < 1e-5);

    // 4. Transfer::Unknown rejected on transfer change / linear math, allowed on layout copy.
    let unknown_rgb = ImageSpec::RGB8_SRGB.with_transfer(Transfer::Unknown);
    let unknown_bgr = ImageSpec::new(SampleLayout::BGR8, unknown_rgb.color.clone());
    assert!(
        RowPlan::build(
            unknown_rgb.clone(),
            unknown_bgr,
            &ConvertPolicy::permissive(),
            None
        )
        .is_ok()
    );
    assert!(
        RowPlan::build(
            unknown_rgb,
            ImageSpec::RGBF32_LINEAR,
            &ConvertPolicy::permissive(),
            None
        )
        .is_err()
    );

    // 5. F32 -> F16 must NOT certify UNIT_INTERVAL when values exceed 1.0 or are NaN/Inf.
    let mut big_f32 = Vec::new();
    for v in [2.5f32, f32::INFINITY, f32::NAN] {
        big_f32.extend_from_slice(&v.to_ne_bytes());
    }
    let src_big = PixelBuf::from_bytes(big_f32, 1, 1, ImageSpec::RGBF32_LINEAR).unwrap();
    let out_f16 = convert_to_buf(
        &src_big.as_view(),
        ImageSpec::RGBF16_LINEAR,
        &ConvertPolicy::permissive(),
    )
    .unwrap();
    assert!(
        !out_f16.facts().float.contains(FloatFacts::UNIT_INTERVAL),
        "F32->F16 with 2.5/inf/NaN must not claim UNIT_INTERVAL"
    );
    assert_eq!(out_f16.facts().depth, None);

    // 6. scan_facts on F16 detects semitransparent alpha, color, and FloatFacts!
    let mut f16_raw = Vec::new();
    for v in [0.25f32, 0.75, 0.5, 0.5] {
        f16_raw.extend_from_slice(&F16Sample::from_f32(v).to_bits().to_ne_bytes());
    }
    let f16_buf = PixelBuf::from_bytes(f16_raw, 1, 1, ImageSpec::RGBAF16_LINEAR).unwrap();
    let facts = scan_facts(&f16_buf.as_view());
    assert!(
        !facts.all_opaque,
        "F16 alpha 0.5 must not be marked all_opaque"
    );
    assert!(
        !facts.all_grayscale,
        "F16 (0.25, 0.75, 0.5) must not be marked all_grayscale"
    );
    assert!(facts.float.contains(FloatFacts::UNIT_INTERVAL));

    let mut f16_nan_raw = Vec::new();
    for bits in [0x7e00u16, 0x3c00, 0x3c00, 0x3c00] {
        f16_nan_raw.extend_from_slice(&bits.to_ne_bytes());
    }
    let f16_nan = PixelBuf::from_bytes(f16_nan_raw, 1, 1, ImageSpec::RGBAF16_LINEAR).unwrap();
    let nan_facts = scan_facts(&f16_nan.as_view());
    assert!(!nan_facts.float.contains(FloatFacts::NO_NAN));
}

#[cfg(feature = "bt2446a")]
#[test]
fn bt2446a_respects_baked_peaks_and_rejects_tiny_peaks() {
    use zenpixels_convert::{Bt2446A, Nits, ToneMapper};

    let mapper = Bt2446A::with_peaks(Nits::new(4000.0), Nits::new(100.0));
    assert_eq!(mapper.baked_peaks(), Some((4000.0, 100.0)));

    // Tiny peak < 1e-3 must fail cleanly rather than producing NaN.
    let src_spec = ImageSpec::new(SampleLayout::RGBF32, ColorSpace::BT2100_PQ);
    let tiny_pol = ConvertPolicy::display_sdr(DiffuseWhite::BT2408, Nits::new(100.0));
    assert!(
        RowPlan::build(
            src_spec.clone(),
            ImageSpec::RGBF32_LINEAR,
            &tiny_pol,
            Some(Nits::new(1e-6)),
        )
        .is_err(),
        "tiny source peak 1e-6 nits must be rejected cleanly"
    );

    // Pre-configured Bt2446A runs through RowPlan without needing PeakSource override,
    // and maps 4000 nits PQ to finite SDR output without double-scaling.
    let pol = ConvertPolicy::permissive()
        .with_sdr_white(DiffuseWhite::BT2408)
        .with_tone_mapper(Arc::new(mapper));
    let mut raw = Vec::new();
    // PQ code 0.9026 (~4000 nits)
    for _ in 0..3 {
        raw.extend_from_slice(&0.9026f32.to_ne_bytes());
    }
    let src = PixelBuf::from_bytes(raw, 1, 1, src_spec).unwrap();
    let out = convert_to_buf_with_peak(
        &src.as_view(),
        ImageSpec::RGBF32_LINEAR,
        &pol,
        PeakSource::Declared,
    )
    .unwrap();
    let f = out.as_view().row_as::<f32>(0).unwrap();
    assert!(
        f[0].is_finite() && f[0] > 0.8 && f[0] <= 1.05,
        "got {:?}",
        f
    );
}

#[cfg(all(feature = "cms-moxcms", feature = "cms-lcms", feature = "icc-db"))]
#[test]
fn cms_policy_lossless_alpha_gate_intent_opaque_and_moxcms_gray_srgb() {
    use zenpixels_convert::{Lcms, MoxCms, RenderingIntent, icc_profiles::SynthesizedIcc};

    let p3_icc: Arc<[u8]> = match SynthesizedIcc::for_cicp(Cicp::DISPLAY_P3) {
        SynthesizedIcc::Profile(p) => Arc::from(p.as_ref()),
        other => panic!("expected Profile for Display P3, got {other:?}"),
    };
    let src_p3_rgba = ImageSpec::new(
        SampleLayout::RGBA8,
        ColorSpace::Icc {
            profile: p3_icc.clone(),
            channels: 3,
            alpha: AlphaMode::Straight,
            intent: RenderingIntent::Perceptual,
        },
    );

    // 1. ConvertPolicy::lossless() with CMS must REJECT alpha drop (RGBA -> RGB) BEFORE CMS!
    let pol_lossless_mox = ConvertPolicy::lossless().with_cms(Arc::new(MoxCms));
    assert!(
        RowPlan::build(
            src_p3_rgba.clone(),
            ImageSpec::RGB8_SRGB,
            &pol_lossless_mox,
            None
        )
        .is_err()
    );
    let pol_lossless_lcms = ConvertPolicy::lossless().with_cms(Arc::new(Lcms));
    assert!(
        RowPlan::build(
            src_p3_rgba.clone(),
            ImageSpec::RGB8_SRGB,
            &pol_lossless_lcms,
            None
        )
        .is_err()
    );

    // 2. CMS rejects AlphaMode::Premultiplied.
    let src_p3_premul = ImageSpec::new(
        SampleLayout::RGBA8,
        ColorSpace::Icc {
            profile: p3_icc.clone(),
            channels: 3,
            alpha: AlphaMode::Premultiplied,
            intent: RenderingIntent::Perceptual,
        },
    );
    let pol_mox = ConvertPolicy::permissive()
        .with_icc_substitution(false)
        .with_cms(Arc::new(MoxCms));
    let pol_lcms = ConvertPolicy::permissive()
        .with_icc_substitution(false)
        .with_cms(Arc::new(Lcms));
    assert!(RowPlan::build(src_p3_premul.clone(), ImageSpec::RGBA8_SRGB, &pol_mox, None).is_err());
    assert!(RowPlan::build(src_p3_premul, ImageSpec::RGBA8_SRGB, &pol_lcms, None).is_err());

    // 3. CMS canonicalizes AlphaMode::Opaque junk alpha to 255 on BOTH MoxCms and Lcms.
    let src_p3_opaque = ImageSpec::new(
        SampleLayout::RGBA8,
        ColorSpace::Icc {
            profile: p3_icc.clone(),
            channels: 3,
            alpha: AlphaMode::Opaque,
            intent: RenderingIntent::Perceptual,
        },
    );
    let buf_op = PixelBuf::from_typed(vec![128u8, 64, 32, 0], 1, 1, src_p3_opaque).unwrap();
    let out_mox = convert_to_buf(&buf_op.as_view(), ImageSpec::RGBA8_SRGB, &pol_mox).unwrap();
    let out_lcms = convert_to_buf(&buf_op.as_view(), ImageSpec::RGBA8_SRGB, &pol_lcms).unwrap();
    assert_eq!(out_mox.as_view().row(0)[3], 255);
    assert_eq!(out_lcms.as_view().row(0)[3], 255);
    // A10: Verify non-neutral P3 RGB channels actually transform and agree between MoxCms & Lcms!
    let mox_px = out_mox.as_view().row(0);
    let lcms_px = out_lcms.as_view().row(0);
    for c in 0..3 {
        assert!(
            (i32::from(mox_px[c]) - i32::from(lcms_px[c])).abs() <= 1,
            "MoxCms {mox_px:?} vs Lcms {lcms_px:?} disagreed on channel {c}"
        );
    }
    assert!(
        mox_px[0] > 135 && mox_px[1] < 60 && mox_px[2] < 30,
        "expected Display P3 [128,64,32] -> sRGB saturated shift, got {mox_px:?}"
    );
    // A10: Relative vs Absolute intent between DCI-P3 (theater white 0.314, 0.351 — greenish)
    // and Display P3 (D65 0.3127, 0.3290) on mid-gray [180, 180, 180] (non-clipping!):
    // - Lcms RelativeColorimetric adapts DCI white to D65 -> neutral [~180, ~180, ~180].
    // - Lcms AbsoluteColorimetric preserves greenish DCI white in D65 (G > R, G > B+5)!
    // - MoxCms cleanly refuses AbsoluteColorimetric (moxcms 0.9 lacks media-white scaling)
    //   instead of silently substituting RelativeColorimetric!
    let dci_p3_icc: Arc<[u8]> = match SynthesizedIcc::for_cicp(Cicp::new(11, 13, 0, true)) {
        SynthesizedIcc::Profile(p) => Arc::from(p.as_ref()),
        other => panic!("expected DCI-P3 profile, got {other:?}"),
    };
    let src_dci_abs = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: dci_p3_icc.clone(),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::AbsoluteColorimetric,
        },
    );
    let src_dci_rel = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: dci_p3_icc,
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    let dst_p3_rel = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: p3_icc.clone(),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    let gray_abs_src = PixelBuf::from_typed(vec![180u8, 180, 180], 1, 1, src_dci_abs).unwrap();
    let gray_rel_src = PixelBuf::from_typed(vec![180u8, 180, 180], 1, 1, src_dci_rel).unwrap();
    let out_rel_mox =
        convert_to_buf(&gray_rel_src.as_view(), dst_p3_rel.clone(), &pol_mox).unwrap();
    let err_abs_mox =
        convert_to_buf(&gray_abs_src.as_view(), dst_p3_rel.clone(), &pol_mox).unwrap_err();
    assert_eq!(
        *err_abs_mox.error(),
        ConvertError::UnsupportedRoute,
        "MoxCms must refuse AbsoluteColorimetric instead of silently running Relative"
    );
    let out_rel_lcms =
        convert_to_buf(&gray_rel_src.as_view(), dst_p3_rel.clone(), &pol_lcms).unwrap();
    let out_abs_lcms = convert_to_buf(&gray_abs_src.as_view(), dst_p3_rel, &pol_lcms).unwrap();
    for (name, rel) in [
        ("MoxCms", out_rel_mox.as_view().row(0)),
        ("Lcms", out_rel_lcms.as_view().row(0)),
    ] {
        for c in 0..3 {
            assert!(
                (i32::from(rel[c]) - 180).abs() <= 1,
                "{name} RelativeColorimetric DCI->P3 must keep neutral gray ~180, got {rel:?}"
            );
        }
    }
    let abs_l = out_abs_lcms.as_view().row(0);
    assert!(
        abs_l[1] > abs_l[2] + 5 && abs_l[1] > abs_l[0] + 2,
        "Lcms AbsoluteColorimetric DCI->D65 must preserve greenish DCI white (G > R, G > B+5), got {abs_l:?}"
    );

    // A10b: Dual-intent LUT ICC profile (distinct A2B0 Perceptual -> reddish Lab vs
    // A2B1 RelativeColorimetric -> bluish Lab) proves BOTH MoxCms AND Lcms honor intent!
    let make_mft2_const_lab = |l: f64, a: f64, b: f64| -> Vec<u8> {
        let mut lut = Vec::new();
        lut.extend_from_slice(b"mft2");
        lut.extend_from_slice(&[0; 4]);
        lut.extend_from_slice(&[3, 3, 2, 0]);
        for row in 0..3 {
            for col in 0..3 {
                let one: i32 = if row == col { 0x10000 } else { 0 };
                lut.extend_from_slice(&one.to_be_bytes());
            }
        }
        lut.extend_from_slice(&2u16.to_be_bytes());
        lut.extend_from_slice(&2u16.to_be_bytes());
        for _ in 0..3 {
            lut.extend_from_slice(&0u16.to_be_bytes());
            lut.extend_from_slice(&0xFFFFu16.to_be_bytes());
        }
        let enc_l = (l * 652.8).round() as u16;
        let enc_a = ((a + 128.0) * 256.0).round() as u16;
        let enc_b = ((b + 128.0) * 256.0).round() as u16;
        for _ in 0..8 {
            lut.extend_from_slice(&enc_l.to_be_bytes());
            lut.extend_from_slice(&enc_a.to_be_bytes());
            lut.extend_from_slice(&enc_b.to_be_bytes());
        }
        for _ in 0..3 {
            lut.extend_from_slice(&0u16.to_be_bytes());
            lut.extend_from_slice(&0xFFFFu16.to_be_bytes());
        }
        lut
    };
    let a2b0 = make_mft2_const_lab(55.0, 50.0, 30.0); // Perceptual: strongly warm red
    let a2b1 = make_mft2_const_lab(55.0, -10.0, -50.0); // Relative: strongly cool blue
    let tag_start = 128usize + 4 + 2 * 12;
    let a2b0_off = tag_start;
    let a2b1_off = (a2b0_off + a2b0.len() + 3) & !3;
    let total_sz = a2b1_off + a2b1.len();
    let mut dual_icc = vec![0u8; total_sz];
    dual_icc[0..4].copy_from_slice(&(total_sz as u32).to_be_bytes());
    dual_icc[8..12].copy_from_slice(&0x0240_0000u32.to_be_bytes());
    dual_icc[12..16].copy_from_slice(b"mntr");
    dual_icc[16..20].copy_from_slice(b"RGB ");
    dual_icc[20..24].copy_from_slice(b"Lab ");
    dual_icc[36..40].copy_from_slice(b"acsp");
    for (i, v) in [0x0000_F6D6u32, 0x0001_0000, 0x0000_D32D]
        .iter()
        .enumerate()
    {
        dual_icc[68 + i * 4..72 + i * 4].copy_from_slice(&v.to_be_bytes());
    }
    dual_icc[128..132].copy_from_slice(&2u32.to_be_bytes());
    dual_icc[132..136].copy_from_slice(b"A2B0");
    dual_icc[136..140].copy_from_slice(&(a2b0_off as u32).to_be_bytes());
    dual_icc[140..144].copy_from_slice(&(a2b0.len() as u32).to_be_bytes());
    dual_icc[144..148].copy_from_slice(b"A2B1");
    dual_icc[148..152].copy_from_slice(&(a2b1_off as u32).to_be_bytes());
    dual_icc[152..156].copy_from_slice(&(a2b1.len() as u32).to_be_bytes());
    dual_icc[a2b0_off..a2b0_off + a2b0.len()].copy_from_slice(&a2b0);
    dual_icc[a2b1_off..a2b1_off + a2b1.len()].copy_from_slice(&a2b1);
    let dual_arc: Arc<[u8]> = Arc::from(dual_icc.as_slice());

    let src_perc = PixelBuf::from_typed(
        vec![128u8, 128, 128],
        1,
        1,
        ImageSpec::new(
            SampleLayout::RGB8,
            ColorSpace::Icc {
                profile: dual_arc.clone(),
                channels: 3,
                alpha: AlphaMode::None,
                intent: RenderingIntent::Perceptual,
            },
        ),
    )
    .unwrap();
    let src_rel = PixelBuf::from_typed(
        vec![128u8, 128, 128],
        1,
        1,
        ImageSpec::new(
            SampleLayout::RGB8,
            ColorSpace::Icc {
                profile: dual_arc,
                channels: 3,
                alpha: AlphaMode::None,
                intent: RenderingIntent::RelativeColorimetric,
            },
        ),
    )
    .unwrap();
    for (name, pol) in [("MoxCms", &pol_mox), ("Lcms", &pol_lcms)] {
        let p_out = convert_to_buf(&src_perc.as_view(), ImageSpec::RGB8_SRGB, pol).unwrap();
        let r_out = convert_to_buf(&src_rel.as_view(), ImageSpec::RGB8_SRGB, pol).unwrap();
        let pp = p_out.as_view().row(0);
        let rp = r_out.as_view().row(0);
        assert!(
            pp[0] > pp[2] + 80,
            "{name} Perceptual A2B0 must produce warm red, got {pp:?}"
        );
        assert!(
            rp[2] > rp[0] + 80,
            "{name} RelativeColorimetric A2B1 must produce cool blue, got {rp:?}"
        );
    }

    // 4. MoxCms Gray sRGB matches Lcms Gray sRGB closely (not gamma 2.2!).
    let mut gray_vals = Vec::with_capacity(256);
    for g in 0u8..=255 {
        gray_vals.push(g);
    }
    let src_gray = PixelBuf::from_typed(gray_vals, 256, 1, ImageSpec::GRAY8_SRGB).unwrap();
    let dst_p3 = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: p3_icc,
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    let g_mox = convert_to_buf(&src_gray.as_view(), dst_p3.clone(), &pol_mox).unwrap();
    let g_lcms = convert_to_buf(&src_gray.as_view(), dst_p3, &pol_lcms).unwrap();
    let max_diff = g_mox
        .as_view()
        .row(0)
        .iter()
        .zip(g_lcms.as_view().row(0))
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap();
    assert!(
        max_diff <= 2,
        "MoxCms vs Lcms Gray sRGB -> P3 max_diff={max_diff} (gamma 2.2 bug would give ~10+)"
    );
}

#[test]
fn ycbcr_gainmap_and_orient_safety_regressions() {
    // 1. YCbCr rejects mismatched Cr plane even when Y and Cb are valid.
    let y_spec = PlaneSpec::new(PlaneRole::Y, SampleType::U8);
    let cb_spec = PlaneSpec::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2);
    let cr_bad_spec = PlaneSpec::new(PlaneRole::Cr, SampleType::F32).with_subsampling(2, 2);
    let y_bytes = [128u8; 16];
    let cb_bytes = [128u8; 4];
    let cr_bytes = [0u8; 16];
    let pv_y = PlaneView::new(&y_bytes, 4, 4, 4, y_spec).unwrap();
    let pv_cb = PlaneView::new(&cb_bytes, 2, 2, 2, cb_spec).unwrap();
    let pv_cr = PlaneView::new(&cr_bytes, 2, 2, 8, cr_bad_spec).unwrap();
    let planar = PlanarView::new(4, 4, &[pv_y, pv_cb, pv_cr])
        .unwrap()
        .with_color(ColorSpace::SRGB)
        .with_matrix_coefficients(YcbcrMatrix::Bt709.to_matrix_coefficients());
    assert!(
        reconstruct_planar_f32(&planar).is_err(),
        "YCbCr must reject unsupported F32 Cr plane"
    );

    // 1b. YCbCr preserves full-res Alpha plane into RGBAF32 and rejects GainMap plane.
    let y_ok = PlaneView::new(
        &[235u8; 16],
        4,
        4,
        4,
        PlaneSpec::new(PlaneRole::Y, SampleType::U8),
    )
    .unwrap();
    let cb_ok = PlaneView::new(
        &[128u8; 4],
        2,
        2,
        2,
        PlaneSpec::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2),
    )
    .unwrap();
    let cr_ok = PlaneView::new(
        &[128u8; 4],
        2,
        2,
        2,
        PlaneSpec::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2),
    )
    .unwrap();
    let a_ok = PlaneView::new(
        &[128u8; 16],
        4,
        4,
        4,
        PlaneSpec::new(PlaneRole::Alpha, SampleType::U8),
    )
    .unwrap();
    let srgb_narrow_alpha = ColorSpace::Rgb {
        primaries: Primaries::Bt709,
        transfer: Transfer::Srgb,
        range: SignalRange::Narrow,
        alpha: AlphaMode::Straight,
        white: None,
    };
    let planar_alpha = PlanarView::new(4, 4, &[y_ok, cb_ok, cr_ok, a_ok])
        .unwrap()
        .with_color(srgb_narrow_alpha.clone())
        .with_matrix_coefficients(YcbcrMatrix::Bt709.to_matrix_coefficients());
    let rgba_out = reconstruct_planar_f32(&planar_alpha).unwrap();
    assert_eq!(rgba_out.spec().layout, SampleLayout::RGBAF32);
    assert_eq!(rgba_out.spec().color.alpha(), AlphaMode::Straight);
    let px_a = rgba_out.as_view().row_as::<f32>(0).unwrap();
    assert!((px_a[3] - 128.0 / 255.0).abs() < 1e-5);

    let gm_extra = PlaneView::new(
        &[0u8; 16],
        4,
        4,
        4,
        PlaneSpec::new(PlaneRole::GainMap, SampleType::U8),
    )
    .unwrap();
    let planar_gm = PlanarView::new(4, 4, &[y_ok, cb_ok, cr_ok, gm_extra])
        .unwrap()
        .with_color(srgb_narrow_alpha)
        .with_matrix_coefficients(YcbcrMatrix::Bt709.to_matrix_coefficients());
    assert!(
        reconstruct_planar_f32(&planar_gm).is_err(),
        "YCbCr must reject PlaneRole::GainMap rather than silently dropping it"
    );

    // 2. GainMap: preserves base DiffuseWhite, supports 3-ch & 1-ch extraction,
    // Reverse HDR-base signed weight, zero-weight identity bypass, negative offsets,
    // Alternate primaries, bit-packed Untagged gain map, and DualLayerHdrView!
    #[cfg(feature = "gainmap")]
    {
        use core::num::NonZeroU32;
        use zenpixels_convert::gainmap::{
            GainMapApplyConfig, apply_gain_map_with, extract_gain_map,
        };
        use zenpixels_convert::{
            DualLayerHdrBuf, GainMapChannels, GainMapColorSpace, GainMapComponent, GainMapMetadata,
            SignedFraction, UnsignedFraction, Unstoppable,
        };

        let base_spec = ImageSpec::RGB8_SRGB.with_white(DiffuseWhite::new(100.0));
        let sdr =
            PixelBuf::from_typed(vec![128u8, 128, 128, 200, 100, 50], 2, 1, base_spec).unwrap();
        let mut hdr_raw = Vec::new();
        for v in [0.5f32, 0.5, 0.5, 2.0, 0.5, 0.1] {
            hdr_raw.extend_from_slice(&v.to_ne_bytes());
        }
        let hdr_buf = PixelBuf::from_bytes(hdr_raw, 2, 1, ImageSpec::RGBF32_LINEAR).unwrap();
        let per_ch_meta = GainMapMetadata::new(
            GainMapChannels::PerChannel([
                GainMapComponent {
                    min_gain_stops: SignedFraction::ZERO,
                    max_gain_stops: SignedFraction::new(2, NonZeroU32::new(1).unwrap()),
                    gamma: UnsignedFraction::ONE,
                    base_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
                    alternate_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
                },
                GainMapComponent {
                    min_gain_stops: SignedFraction::new(-1, NonZeroU32::new(1).unwrap()),
                    max_gain_stops: SignedFraction::new(3, NonZeroU32::new(1).unwrap()),
                    gamma: UnsignedFraction::ONE,
                    base_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
                    alternate_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
                },
                GainMapComponent {
                    min_gain_stops: SignedFraction::new(1, NonZeroU32::new(2).unwrap()),
                    max_gain_stops: SignedFraction::new(4, NonZeroU32::new(1).unwrap()),
                    gamma: UnsignedFraction::ONE,
                    base_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
                    alternate_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
                },
            ]),
            UnsignedFraction::ZERO,
            UnsignedFraction::new(2, NonZeroU32::new(1).unwrap()),
            GainMapColorSpace::Base,
        )
        .unwrap();
        let dual = extract_gain_map(&sdr.as_view(), &hdr_buf.as_view(), per_ch_meta).unwrap();
        assert_eq!(dual.gain_map().spec().layout, SampleLayout::RGB8);
        assert_eq!(dual.metadata(), per_ch_meta);

        let applied = apply_gain_map_with(
            &dual.as_view(),
            &GainMapApplyConfig::from_headroom_stops(1.0),
            &Unstoppable,
        )
        .unwrap();
        assert_eq!(applied.spec().color.white(), Some(DiffuseWhite::new(100.0)));

        // Witness A (P0): Reverse / HDR-base signed weight: H_base=2, H_alt=0, B=1.0, L=2.0 -> 0.25 / 0.5 / 1.0
        let hdr_base = PixelBuf::from_bytes(
            [1.0f32, 1.0, 1.0]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
            1,
            1,
            ImageSpec::RGBF32_LINEAR,
        )
        .unwrap();
        let untagged_1 = ColorSpace::Untagged {
            channels: 1,
            alpha: AlphaMode::None,
        };
        let rev_gm = PixelBuf::from_typed(
            vec![255u8], // code 255 -> L = max_gain_stops = +2.0
            1,
            1,
            ImageSpec::new(SampleLayout::GRAY8, untagged_1.clone()),
        )
        .unwrap();
        let rev_meta = GainMapMetadata::new(
            GainMapChannels::Shared(GainMapComponent {
                min_gain_stops: SignedFraction::ZERO,
                max_gain_stops: SignedFraction::new(2, NonZeroU32::new(1).unwrap()),
                gamma: UnsignedFraction::ONE,
                base_offset: SignedFraction::ZERO,
                alternate_offset: SignedFraction::ZERO,
            }),
            UnsignedFraction::new(2, NonZeroU32::new(1).unwrap()),
            UnsignedFraction::ZERO,
            GainMapColorSpace::Base,
        )
        .unwrap();
        let rev_dual = DualLayerHdrBuf::new(hdr_base, rev_gm, rev_meta).unwrap();
        let at_h0 = apply_gain_map_with(
            &rev_dual.as_view(),
            &GainMapApplyConfig::from_headroom_stops(0.0),
            &Unstoppable,
        )
        .unwrap();
        let at_h1 = apply_gain_map_with(
            &rev_dual.as_view(),
            &GainMapApplyConfig::from_headroom_stops(1.0),
            &Unstoppable,
        )
        .unwrap();
        let at_h2 = apply_gain_map_with(
            &rev_dual.as_view(),
            &GainMapApplyConfig::from_headroom_stops(2.0),
            &Unstoppable,
        )
        .unwrap();
        assert!((at_h0.as_view().row_as::<f32>(0).unwrap()[0] - 0.25).abs() < 1e-5);
        assert!((at_h1.as_view().row_as::<f32>(0).unwrap()[0] - 0.5).abs() < 1e-5);
        assert!((at_h2.as_view().row_as::<f32>(0).unwrap()[0] - 1.0).abs() < 1e-5);

        // Witness B (P0): Zero-weight exact identity bypass with unequal offsets & negative offsets!
        let base_half = PixelBuf::from_bytes(
            [0.5f32, 0.5, 0.5]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
            1,
            1,
            ImageSpec::RGBF32_LINEAR,
        )
        .unwrap();
        let gm_10bit = PixelBuf::from_bytes(
            (512u16).to_ne_bytes().to_vec(),
            1,
            1,
            ImageSpec::new(
                SampleLayout::GRAY16.with_packing(BitPacking::Lsb { bits: 10 }),
                untagged_1.clone(),
            ),
        )
        .unwrap();
        let unequal_off_meta = GainMapMetadata::new(
            GainMapChannels::Shared(GainMapComponent {
                min_gain_stops: SignedFraction::ZERO,
                max_gain_stops: SignedFraction::new(2, NonZeroU32::new(1).unwrap()),
                gamma: UnsignedFraction::ONE,
                base_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
                alternate_offset: SignedFraction::new(-1, NonZeroU32::new(32).unwrap()),
            }),
            UnsignedFraction::ZERO,
            UnsignedFraction::new(2, NonZeroU32::new(1).unwrap()),
            GainMapColorSpace::Base,
        )
        .unwrap();
        let bypass_dual = DualLayerHdrBuf::new(base_half, gm_10bit, unequal_off_meta).unwrap();
        let at_zero = apply_gain_map_with(
            &bypass_dual.as_view(),
            &GainMapApplyConfig::from_headroom_stops(0.0)
                .with_policy(ConvertPolicy::working_linear()),
            &Unstoppable,
        )
        .unwrap();
        assert!(
            (at_zero.as_view().row_as::<f32>(0).unwrap()[0] - 0.5).abs() < 1e-6,
            "weight == 0 must return exact base 0.5 regardless of unequal/negative offsets!"
        );

        // Witness C (P0): GainMapColorSpace::Alternate converts to alternate primaries BEFORE gain,
        // and rejects missing alternate_color_space!
        let alt_meta = GainMapMetadata::new(
            GainMapChannels::Shared(GainMapComponent {
                min_gain_stops: SignedFraction::ZERO,
                max_gain_stops: SignedFraction::new(1, NonZeroU32::new(1).unwrap()),
                gamma: UnsignedFraction::ONE,
                base_offset: SignedFraction::ZERO,
                alternate_offset: SignedFraction::ZERO,
            }),
            UnsignedFraction::ZERO,
            UnsignedFraction::new(1, NonZeroU32::new(1).unwrap()),
            GainMapColorSpace::Alternate,
        )
        .unwrap();
        let srgb_red = PixelBuf::from_bytes(
            [1.0f32, 0.0, 0.0]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
            1,
            1,
            ImageSpec::RGBF32_LINEAR,
        )
        .unwrap();
        let full_gm = PixelBuf::from_typed(
            vec![255u8],
            1,
            1,
            ImageSpec::new(SampleLayout::GRAY8, untagged_1),
        )
        .unwrap();
        // B26: DualLayerHdrBuf::new refuses GainMapColorSpace::Alternate without alternate_color_space
        // at construction time; DualLayerHdrBuf::new_with_alternate_color supplies the required color space.
        assert!(
            DualLayerHdrBuf::new(srgb_red.clone(), full_gm.clone(), alt_meta).is_err(),
            "DualLayerHdrBuf::new must refuse GainMapColorSpace::Alternate without alternate_color_space"
        );
        let alt_with_p3 = DualLayerHdrBuf::new_with_alternate_color(
            srgb_red,
            full_gm,
            alt_meta,
            ColorSpace::DISPLAY_P3,
        )
        .unwrap();
        let alt_out = apply_gain_map_with(
            &alt_with_p3.as_view(),
            &GainMapApplyConfig::from_headroom_stops(1.0),
            &Unstoppable,
        )
        .unwrap();
        assert!(matches!(
            alt_out.spec().color,
            ColorSpace::Rgb {
                primaries: Primaries::DisplayP3,
                transfer: Transfer::Linear,
                ..
            }
        ));
        let alt_px = alt_out.as_view().row_as::<f32>(0).unwrap();
        // sRGB red (1,0,0) in Display P3 is ~(0.8225, 0.0332, 0.0171), times 2^1 = ~(1.645, 0.066, 0.034)
        assert!((alt_px[0] - 1.645).abs() < 0.01 && alt_px[1] > 0.05);
    }

    // 3. Orient: reject color mismatch and handle 1x1 buffers without stride overflow.
    let p3_src = PixelBuf::from_typed(
        vec![10u8, 20, 30],
        1,
        1,
        ImageSpec::new(SampleLayout::RGB8, ColorSpace::DISPLAY_P3),
    )
    .unwrap();
    let mut srgb_dst = PixelBuf::new(1, 1, ImageSpec::RGB8_SRGB).unwrap();
    assert!(
        apply_orientation_into(
            &p3_src.as_view(),
            Orientation::Rotate90,
            &mut srgb_dst.as_view_mut()
        )
        .is_err()
    );

    let one_px = PixelBuf::from_typed(vec![11u8, 22, 33], 1, 1, ImageSpec::RGB8_SRGB).unwrap();
    for orient in Orientation::ALL {
        let out = apply_orientation(&one_px.as_view(), orient).unwrap();
        assert_eq!(out.width(), 1);
        assert_eq!(out.height(), 1);
        assert_eq!(out.as_view().row(0), &[11, 22, 33]);
    }

    // 4. convert_into updates sink facts cleanly.
    let mut dst_buf = PixelBuf::new(1, 1, ImageSpec::RGBA8_SRGB).unwrap();
    convert_into(&one_px, &mut dst_buf, &ConvertPolicy::permissive()).unwrap();
    assert_eq!(dst_buf.as_view().row(0), &[11, 22, 33, 255]);
    assert!(dst_buf.facts().all_opaque);

    // Confirm ColorOrigin::from_cicp helper exists.
    let _orig = ColorOrigin::from_cicp(Cicp::SRGB);
}

#[test]
fn prefreeze_audit_signaling_lossless_negotiate_facts_xyz_gamma_regressions() {
    use zenpixels_convert::{ExactDepth, Gamma, Negotiate, WhitePoint};

    // 1. SameAsOrigin rejects stale P3 origin ICC/CICP on an sRGB buffer!
    let srgb = ImageSpec::RGB8_SRGB;
    let same_plan = OutputPlan::build(
        srgb.clone(),
        srgb.clone(),
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputSignaling::SameAsOrigin,
    )
    .unwrap();
    let stale_p3_origin =
        ColorOrigin::from_icc(Arc::from(zenpixels_convert::icc_profiles::DISPLAY_P3_V4));
    assert!(
        same_plan.resolve_signaling(&stale_p3_origin).is_err(),
        "SameAsOrigin must refuse stale Display P3 origin ICC on an sRGB buffer!"
    );
    let stale_cicp_origin = ColorOrigin::from_cicp(Cicp::DISPLAY_P3);
    assert!(
        same_plan.resolve_signaling(&stale_cicp_origin).is_err(),
        "SameAsOrigin must refuse stale Display P3 origin CICP on an sRGB buffer!"
    );
    let ok_srgb_origin = ColorOrigin::from_cicp(Cicp::SRGB);
    let (sig_cicp, sig_icc) = same_plan.resolve_signaling(&ok_srgb_origin).unwrap();
    assert_eq!(sig_cicp, Some(Cicp::SRGB));
    assert!(sig_icc.is_none());
    let _stream_row = same_plan.build_row_plan(None).unwrap();

    // 2. CustomIcc rejects arbitrary unrecognized RGB ICC on structured RGB destination!
    let mut dummy_rgb_icc = vec![0u8; 128];
    dummy_rgb_icc[16..20].copy_from_slice(b"RGB ");
    dummy_rgb_icc[36..40].copy_from_slice(b"acsp");
    assert!(
        OutputPlan::build(
            srgb.clone(),
            srgb.clone(),
            &ConvertPolicy::permissive(),
            PeakSource::Measured,
            OutputSignaling::CustomIcc(Arc::from(dummy_rgb_icc.as_slice())),
        )
        .is_err(),
        "CustomIcc must reject unrecognized ICC profile on structured RGB target!"
    );

    // 3. EmbedIcc works for structured ColorSpace::Gray (BT.709, Linear) when icc-db is enabled!
    #[cfg(feature = "icc-db")]
    for tf in [Transfer::Bt709, Transfer::Linear] {
        let gray_spec = ImageSpec::GRAY8_SRGB.with_transfer(tf);
        let plan = OutputPlan::build(
            gray_spec.clone(),
            gray_spec,
            &ConvertPolicy::permissive(),
            PeakSource::Measured,
            OutputSignaling::EmbedIcc,
        )
        .unwrap();
        let (_cicp, icc) = plan.resolve_signaling(&ColorOrigin::assumed()).unwrap();
        assert!(
            icc.is_some(),
            "EmbedIcc must synthesize Gray ICC for {tf:?}"
        );
    }

    // 4. ConvertPolicy::lossless() rejects Display P3 U8 -> sRGB U8 (bounded gamut clip!),
    // while allowing Display P3 F32 -> sRGB F32 (unbounded float PreserveExtended!).
    let p3_u8 = ImageSpec::new(SampleLayout::RGB8, ColorSpace::DISPLAY_P3);
    assert!(
        RowPlan::build(
            p3_u8,
            ImageSpec::RGB8_SRGB,
            &ConvertPolicy::lossless(),
            None
        )
        .is_err(),
        "lossless() must forbid Display P3 U8 -> sRGB U8 bounded clipping!"
    );
    let p3_f32 = ImageSpec::new(
        SampleLayout::RGBF32,
        ColorSpace::DISPLAY_P3.with_transfer(Transfer::Linear),
    );
    assert!(
        RowPlan::build(
            p3_f32,
            ImageSpec::RGBF32_LINEAR,
            &ConvertPolicy::lossless(),
            None
        )
        .is_ok(),
        "lossless() should allow unbounded F32 Display P3 -> sRGB (PreserveExtended)"
    );

    // 5. ImagePlan defers all_grayscale check to runtime buffer facts AND preserves identity facts!
    let mut gray_rgb_buf =
        PixelBuf::from_typed(vec![42u8, 42, 42, 99, 99, 99], 2, 1, srgb.clone()).unwrap();
    gray_rgb_buf.assume_facts(SampleFacts::NONE.with_all_grayscale(true));
    let plan_to_gray = ImagePlan::build(
        srgb.clone(),
        ImageSpec::GRAY8_SRGB,
        &ConvertPolicy::lossless(),
        PeakSource::Measured,
    )
    .unwrap();
    let out_g = plan_to_gray.execute(&gray_rgb_buf).unwrap();
    assert_eq!(out_g.as_view().row(0), &[42, 99]);

    let unverified_rgb =
        PixelBuf::from_typed(vec![255u8, 0, 0, 0, 255, 0], 2, 1, srgb.clone()).unwrap();
    assert!(plan_to_gray.execute(&unverified_rgb).is_err());

    let id_plan = ImagePlan::build(
        srgb.clone(),
        srgb.clone(),
        &ConvertPolicy::lossless(),
        PeakSource::Measured,
    )
    .unwrap();
    let copied = id_plan.execute(&gray_rgb_buf).unwrap();
    assert!(
        copied.facts().all_grayscale,
        "identity ImagePlan must preserve src buffer SampleFacts!"
    );

    // 6. Full-range U16 ExactDepth(8) must NOT bypass lossless narrow-range U16->U8 depth check!
    let u16_narrow = ImageSpec::RGB16_SRGB.with_range(SignalRange::Narrow);
    let u8_narrow = ImageSpec::RGB8_SRGB.with_range(SignalRange::Narrow);
    let pol_u8_proven = ConvertPolicy::lossless().with_source_facts(SampleFacts::from_depth(
        ExactDepth::new(SampleType::U16, SignalRange::Full, 8, 8).unwrap(),
    ));
    assert!(
        RowPlan::build(u16_narrow, u8_narrow, &pol_u8_proven, None).is_err(),
        "narrow-range U16->U8 must not bypass DepthPolicy::Forbid via full-range ExactDepth(8)"
    );

    // 7. Negotiate skips infeasible Transfer::Unknown gamut change & MoxCms-unsupported layouts!
    let unknown_tf_p3 = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::DISPLAY_P3.with_transfer(Transfer::Unknown),
    );
    let neg = Negotiate::select(
        &unknown_tf_p3,
        &[ImageSpec::RGB8_SRGB, unknown_tf_p3.clone()],
        &ConvertPolicy::permissive(),
    )
    .unwrap();
    assert_eq!(neg.spec, unknown_tf_p3);
    assert!(
        Negotiate::select(
            &unknown_tf_p3,
            &[ImageSpec::RGB8_SRGB],
            &ConvertPolicy::permissive()
        )
        .is_none()
    );

    #[cfg(feature = "cms-moxcms")]
    {
        use zenpixels_convert::{MoxCms, RenderingIntent};
        let p3_icc = Arc::<[u8]>::from(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
        let src_icc = ImageSpec::new(
            SampleLayout::RGB8,
            ColorSpace::Icc {
                profile: p3_icc,
                channels: 3,
                alpha: AlphaMode::None,
                intent: RenderingIntent::RelativeColorimetric,
            },
        );
        let pol_mox_nosub = ConvertPolicy::permissive()
            .with_icc_substitution(false)
            .with_cms(Arc::new(MoxCms));
        // CMS refuses OrderedDither and SoftCompress!
        assert!(
            RowPlan::build(
                src_icc.clone(),
                ImageSpec::RGB8_SRGB,
                &pol_mox_nosub.clone().with_depth(DepthPolicy::OrderedDither),
                None,
            )
            .is_err()
        );
        #[cfg(feature = "oklab")]
        assert!(
            RowPlan::build(
                src_icc.clone(),
                ImageSpec::RGB8_SRGB,
                &pol_mox_nosub.clone().with_gamut(GamutPolicy::SoftCompress),
                None,
            )
            .is_err()
        );
        // Negotiate skips BGR8 (unsupported by MoxCms direct transform) and picks RGB8!
        let picked = Negotiate::select(
            &src_icc,
            &[ImageSpec::BGR8_SRGB, ImageSpec::RGB8_SRGB],
            &pol_mox_nosub,
        )
        .unwrap();
        assert_eq!(picked.spec, ImageSpec::RGB8_SRGB);
    }

    // 8. Transfer::Gamma(DCI_2_6) & ColorSpace::Xyz <-> Rgb roundtrip!
    let dci_spec = ImageSpec::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: Primaries::DciP3,
            transfer: Transfer::Gamma(Gamma::DCI_2_6),
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let xyz_spec = ImageSpec::new(
        SampleLayout::XYZF32,
        ColorSpace::Xyz {
            white_point: WhitePoint::E,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let mut dci_buf = PixelBuf::new(2, 1, dci_spec.clone()).unwrap();
    dci_buf
        .as_view_mut()
        .row_as_mut::<f32>(0)
        .unwrap()
        .copy_from_slice(&[0.5, 0.25, 0.75, 1.0, 1.0, 1.0]);
    let in_xyz = convert_to_buf(
        &dci_buf.as_view(),
        xyz_spec,
        &ConvertPolicy::working_linear(),
    )
    .unwrap();
    let back_dci = convert_to_buf(
        &in_xyz.as_view(),
        dci_spec,
        &ConvertPolicy::working_linear(),
    )
    .unwrap();
    let orig_f = dci_buf.as_view().row_as::<f32>(0).unwrap();
    let back_f = back_dci.as_view().row_as::<f32>(0).unwrap();
    for i in 0..6 {
        assert!(
            (orig_f[i] - back_f[i]).abs() < 1e-4,
            "DCI-P3 Gamma 2.6 <-> XYZ(E) roundtrip drift at {i}: {} vs {}",
            orig_f[i],
            back_f[i]
        );
    }
}

#[test]
fn exhaustive_p3_srgb_option_a_exact_and_option_b_1lsb_parity() {
    // Runs ALL 16,777,216 RGB8 triplets in `--release`, and a dense 64^3 + edges grid in debug!
    let step: usize = if cfg!(debug_assertions) { 4 } else { 1 };

    for (label, src_prim, dst_prim, depth) in [
        (
            "P3->sRGB Round",
            Primaries::DisplayP3,
            Primaries::Bt709,
            DepthPolicy::Round,
        ),
        (
            "P3->sRGB Trunc",
            Primaries::DisplayP3,
            Primaries::Bt709,
            DepthPolicy::Truncate,
        ),
        (
            "sRGB->P3 Round",
            Primaries::Bt709,
            Primaries::DisplayP3,
            DepthPolicy::Round,
        ),
        (
            "BT2020->sRGB Round",
            Primaries::Bt2020,
            Primaries::Bt709,
            DepthPolicy::Round,
        ),
    ] {
        let src_spec = ImageSpec::new(
            SampleLayout::RGB8,
            ColorSpace::SRGB.with_primaries(src_prim),
        );
        let dst_spec = ImageSpec::new(
            SampleLayout::RGB8,
            ColorSpace::SRGB.with_primaries(dst_prim),
        );
        let plan_a_all = RowPlan::build(
            src_spec.clone(),
            dst_spec.clone(),
            &ConvertPolicy::permissive()
                .with_depth(depth)
                .with_accelerators(Accelerators::ALL),
            None,
        )
        .unwrap();
        let plan_a_scalar = RowPlan::build(
            src_spec.clone(),
            dst_spec.clone(),
            &ConvertPolicy::permissive()
                .with_depth(depth)
                .with_accelerators(Accelerators::SCALAR),
            None,
        )
        .unwrap();
        let plan_b_all = RowPlan::build(
            src_spec.clone(),
            dst_spec.clone(),
            &ConvertPolicy::permissive()
                .with_depth(depth)
                .with_approximate(true)
                .with_accelerators(Accelerators::ALL),
            None,
        )
        .unwrap();
        let plan_b_scalar = RowPlan::build(
            src_spec,
            dst_spec,
            &ConvertPolicy::permissive()
                .with_depth(depth)
                .with_approximate(true)
                .with_accelerators(Accelerators::SCALAR),
            None,
        )
        .unwrap();

        // Check a single plane of width 256*256 = 65536 pixels per R slice!
        let mut row_src = Vec::with_capacity(256 * 256 * 3);
        let mut out_a_all = vec![0u8; 256 * 256 * 3];
        let mut out_a_sca = vec![0u8; 256 * 256 * 3];
        let mut out_b_all = vec![0u8; 256 * 256 * 3];
        let mut out_b_sca = vec![0u8; 256 * 256 * 3];

        let mut max_err = 0u8;
        let mut diff_count = 0u64;
        let mut tested = 0u64;

        let mut r = 0usize;
        while r < 256 {
            row_src.clear();
            let mut g = 0usize;
            while g < 256 {
                let mut b = 0usize;
                while b < 256 {
                    row_src.push(r as u8);
                    row_src.push(g as u8);
                    row_src.push(b as u8);
                    b += step;
                }
                g += step;
            }
            let w = (row_src.len() / 3) as u32;
            let nbytes = row_src.len();
            plan_a_all
                .convert_row(&row_src, &mut out_a_all[..nbytes], w, 0)
                .unwrap();
            plan_a_scalar
                .convert_row(&row_src, &mut out_a_sca[..nbytes], w, 0)
                .unwrap();
            plan_b_all
                .convert_row(&row_src, &mut out_b_all[..nbytes], w, 0)
                .unwrap();
            plan_b_scalar
                .convert_row(&row_src, &mut out_b_sca[..nbytes], w, 0)
                .unwrap();

            assert_eq!(
                &out_a_all[..nbytes],
                &out_a_sca[..nbytes],
                "{label}: Option A ALL vs SCALAR 0-bit mismatch at r={r}"
            );
            assert_eq!(
                &out_b_all[..nbytes],
                &out_b_sca[..nbytes],
                "{label}: Option B Q14 ALL vs SCALAR 0-bit mismatch at r={r}"
            );
            for i in 0..nbytes {
                let d = out_a_all[i].abs_diff(out_b_all[i]);
                if d > 0 {
                    diff_count += 1;
                    if d > max_err {
                        max_err = d;
                    }
                }
            }
            tested += u64::from(w);
            r += step;
        }
        eprintln!(
            "{label}: tested={tested} triplets, Option A ALL==SCALAR 0-bit OK, Option B Q14 ALL==SCALAR 0-bit OK, max_err={max_err} LSB, diff_channels={diff_count}/{total_ch} ({:.2}%)",
            (diff_count as f64) * 100.0 / ((tested * 3) as f64),
            total_ch = tested * 3
        );
        assert!(
            max_err <= 1,
            "{label}: Option B Q14 max error {max_err} exceeded 1 LSB over {tested} RGB8 triplets!"
        );
    }

    // Also check all 6 U8 channel order combinations (RGB/BGR/RGBA/BGRA/RGBX/BGRX),
    // odd widths (1, 7, 17, 37), and Opaque alpha canonicalization on BOTH Option A & Option B!
    let orders = [
        (SampleLayout::RGB8, AlphaMode::None),
        (SampleLayout::BGR8, AlphaMode::None),
        (SampleLayout::RGBA8, AlphaMode::Straight),
        (SampleLayout::BGRA8, AlphaMode::Straight),
        (SampleLayout::RGBX8, AlphaMode::Undefined),
        (SampleLayout::BGRX8, AlphaMode::Undefined),
        (SampleLayout::RGBA8, AlphaMode::Opaque),
    ];
    for &(sl, sa) in &orders {
        for &(dl, da) in &orders {
            let src_spec = ImageSpec::new(sl, ColorSpace::DISPLAY_P3.with_alpha(sa));
            let dst_spec = ImageSpec::new(dl, ColorSpace::SRGB.with_alpha(da));
            let w = 37usize;
            let mut src_row = vec![0u8; w * sl.bytes_per_pixel()];
            for (i, b) in src_row.iter_mut().enumerate() {
                *b = ((i * 53 + 17) & 0xFF) as u8;
            }
            for approx in [false, true] {
                let pol_all = ConvertPolicy::permissive()
                    .with_approximate(approx)
                    .with_accelerators(Accelerators::ALL);
                let pol_sca = ConvertPolicy::permissive()
                    .with_approximate(approx)
                    .with_accelerators(Accelerators::SCALAR);
                let p_all =
                    RowPlan::build(src_spec.clone(), dst_spec.clone(), &pol_all, None).unwrap();
                let p_sca =
                    RowPlan::build(src_spec.clone(), dst_spec.clone(), &pol_sca, None).unwrap();
                let mut d_all = vec![0u8; w * dl.bytes_per_pixel()];
                let mut d_sca = vec![0u8; w * dl.bytes_per_pixel()];
                p_all
                    .convert_row(&src_row, &mut d_all, w as u32, 0)
                    .unwrap();
                p_sca
                    .convert_row(&src_row, &mut d_sca, w as u32, 0)
                    .unwrap();
                assert_eq!(
                    d_all, d_sca,
                    "srgb8_matrix parity mismatch approx={approx} {sl:?}->{dl:?}"
                );
                if dl.order.has_alpha_slot()
                    && (!sl.order.has_alpha_slot()
                        || sa == AlphaMode::Opaque
                        || da == AlphaMode::Opaque)
                {
                    for p in 0..w {
                        assert_eq!(
                            d_all[p * 4 + 3],
                            255,
                            "opaque fill failed for {sl:?}->{dl:?}"
                        );
                    }
                }
            }
        }
    }
}
