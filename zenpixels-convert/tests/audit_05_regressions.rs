//! Comprehensive regression tests for zenpixels-convert 0.5 audit findings
//! (batches 1-3: planner ordering, OutputPlan validation, CMS policy/intent/alpha/gray,
//! exhaustive SIMD RGBA8 premul & F16 sNaN parity, BT.2100-3 narrow scaling,
//! RGBA/BGRA alpha parity, unpremul subnormal, matte-before-clip, HLG swizzle parity,
//! diffuse white rescale, Transfer::Unknown guard, Bt2446A baked peaks, F32->F16 facts,
//! scan_facts F16, YCbCr Cr validation, GainMap validation, orient color & 1x1 stride).

use std::sync::Arc;

use zenpixels_convert::{
    Accelerators, AlphaMode, AlphaPolicy, BitPacking, ChannelOrder, Cicp, Color, ColorOrigin,
    ColorSpace, ConvertError, ConvertPolicy, DepthPolicy, DiffuseWhite, DualLayerHdr, F16Sample,
    FloatFacts, GainMapMetadata, GamutPolicy, ImagePlan, ImageSpec, Orientation, OutputPlan,
    OutputProfile, PeakSource, PixelBuf, PlanarView, PlaneRole, PlaneSpec, PlaneView, Primaries,
    RowPlan, SampleFacts, SampleLayout, SampleType, SignalRange, Transfer, YcbcrMatrix,
    apply_gain_map, apply_orientation, apply_orientation_into, convert_buf, convert_into,
    extract_gain_map, scan_facts, ycbcr_to_rgb_f32,
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
    let src_spec = ImageSpec::rgba8_srgb();
    let dst_spec = ImageSpec::rgba8_srgb().with_alpha(AlphaMode::Premultiplied);
    let src = PixelBuf::from_vec(src_bytes, 256, 256, src_spec).unwrap();

    for (depth_policy, round) in [(DepthPolicy::Round, true), (DepthPolicy::Truncate, false)] {
        let pol_scalar = ConvertPolicy::permissive()
            .with_depth(depth_policy)
            .with_accelerators(Accelerators::SCALAR);
        let pol_simd = ConvertPolicy::permissive()
            .with_depth(depth_policy)
            .with_accelerators(Accelerators::ALL);

        let out_scalar = convert_buf(
            &src.as_view(),
            dst_spec.clone(),
            &pol_scalar,
            PeakSource::Measured,
        )
        .unwrap();
        let out_simd = convert_buf(
            &src.as_view(),
            dst_spec.clone(),
            &pol_simd,
            PeakSource::Measured,
        )
        .unwrap();

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
    let src_spec = ImageSpec::rgbaf16_linear();
    let dst_spec = ImageSpec::rgbaf32_linear();
    let src = PixelBuf::from_bytes(raw, 256, 64, src_spec).unwrap();

    let out_scalar = convert_buf(
        &src.as_view(),
        dst_spec.clone(),
        &ConvertPolicy::permissive().with_accelerators(Accelerators::SCALAR),
        PeakSource::Measured,
    )
    .unwrap();
    let out_simd = convert_buf(
        &src.as_view(),
        dst_spec,
        &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
        PeakSource::Measured,
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
    let src = PixelBuf::from_vec(
        vec![16u8, 128, 235],
        1,
        1,
        ImageSpec::new(SampleLayout::RGB8, narrow_rgb.clone()),
    )
    .unwrap();
    let dst_spec = ImageSpec::new(SampleLayout::RGB16, narrow_rgb);
    let out = convert_buf(
        &src.as_view(),
        dst_spec,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
    .unwrap();
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
    let out_f32 = convert_buf(
        &src_p010.as_view(),
        dst_f32,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
    .unwrap();
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
    let src_rgba = PixelBuf::from_vec(rgba_bytes, 256, 1, ImageSpec::rgba8_srgb()).unwrap();
    let src_bgra = PixelBuf::from_vec(bgra_bytes, 256, 1, ImageSpec::bgra8_srgb()).unwrap();

    let pol_s = ConvertPolicy::permissive().with_accelerators(Accelerators::SCALAR);
    let pol_v = ConvertPolicy::permissive().with_accelerators(Accelerators::ALL);

    let r_s = convert_buf(
        &src_rgba.as_view(),
        ImageSpec::rgbaf32_linear(),
        &pol_s,
        PeakSource::Measured,
    )
    .unwrap();
    let r_v = convert_buf(
        &src_rgba.as_view(),
        ImageSpec::rgbaf32_linear(),
        &pol_v,
        PeakSource::Measured,
    )
    .unwrap();
    let b_s = convert_buf(
        &src_bgra.as_view(),
        ImageSpec::rgbaf32_linear(),
        &pol_s,
        PeakSource::Measured,
    )
    .unwrap();
    let b_v = convert_buf(
        &src_bgra.as_view(),
        ImageSpec::rgbaf32_linear(),
        &pol_v,
        PeakSource::Measured,
    )
    .unwrap();

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
    let lin_buf = PixelBuf::from_bytes(lin_raw, 37, 1, ImageSpec::rgbaf32_linear()).unwrap();
    for dst_spec in [ImageSpec::rgba8_srgb(), ImageSpec::bgra8_srgb()] {
        let os = convert_buf(
            &lin_buf.as_view(),
            dst_spec.clone(),
            &pol_s,
            PeakSource::Measured,
        )
        .unwrap();
        let ov = convert_buf(
            &lin_buf.as_view(),
            dst_spec.clone(),
            &pol_v,
            PeakSource::Measured,
        )
        .unwrap();
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
    let premul_spec = ImageSpec::rgbaf32_linear().with_alpha(AlphaMode::Premultiplied);
    let src = PixelBuf::from_bytes(raw, 1, 1, premul_spec).unwrap();
    let out = convert_buf(
        &src.as_view(),
        ImageSpec::rgbaf32_linear(),
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
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
    let opaque_u8 = ImageSpec::rgba8_srgb().with_alpha(AlphaMode::Opaque);
    let src_op =
        PixelBuf::from_vec(vec![10u8, 20, 30, 0, 40, 50, 60, 17], 2, 1, opaque_u8).unwrap();
    let dst_straight = convert_buf(
        &src_op.as_view(),
        ImageSpec::rgba8_srgb(),
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
    .unwrap();
    assert_eq!(
        dst_straight.as_view().row(0),
        &[10, 20, 30, 255, 40, 50, 60, 255]
    );

    let dst_lin = convert_buf(
        &src_op.as_view(),
        ImageSpec::rgbaf32_linear(),
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
    .unwrap();
    let f = dst_lin.as_view().row_as::<f32>(0).unwrap();
    assert_eq!(f[3], 1.0);
    assert_eq!(f[7], 1.0);
}

#[test]
fn output_plan_validates_spec_on_identity_and_signaling_contracts() {
    let p3_spec = ImageSpec::new(SampleLayout::RGB8, ColorSpace::DISPLAY_P3);
    let srgb_spec = ImageSpec::rgb8_srgb();

    // 1. Identity plan rejects mismatched buffer/view spec!
    let id_plan = OutputPlan::build(
        srgb_spec.clone(),
        srgb_spec.clone(),
        ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputProfile::Auto,
    )
    .unwrap();
    let wrong_buf = PixelBuf::new(2, 2, p3_spec.clone()).unwrap();
    assert!(id_plan.execute_buf(wrong_buf).is_err());

    // 2. SameAsOrigin rejected when color meaning changed (P3 -> sRGB), accepted when unchanged.
    assert!(
        OutputPlan::build(
            p3_spec.clone(),
            srgb_spec.clone(),
            ConvertPolicy::permissive(),
            PeakSource::Measured,
            OutputProfile::SameAsOrigin,
        )
        .is_err()
    );
    assert!(
        OutputPlan::build(
            p3_spec.clone(),
            p3_spec,
            ConvertPolicy::permissive(),
            PeakSource::Measured,
            OutputProfile::SameAsOrigin,
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
        ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputProfile::CicpOnly,
    );
    assert!(matches!(
        err_cicp,
        Err(ref e) if matches!(e.error(), ConvertError::OutputSignalingUnavailable)
    ));

    // 4. Auto on Narrow range never attaches a full-range ICC profile.
    let narrow_spec = ImageSpec::rgb8_srgb().with_range(SignalRange::Narrow);
    let narrow_plan = OutputPlan::build(
        narrow_spec.clone(),
        narrow_spec.clone(),
        ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputProfile::Auto,
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
        ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputProfile::CustomIcc(Arc::from(gray_icc.as_slice())),
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
            ImageSpec::rgb8_srgb(),
            pol,
            PeakSource::Fixed(Nits::new(1000.0)),
            OutputProfile::Auto,
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
    assert!(
        RowPlan::build(
            ImageSpec::rgb8_srgb(),
            ImageSpec::gray8_srgb(),
            &lossless,
            None
        )
        .is_err()
    );

    // 2. RGBA8 -> RGB10A2 rejected (8-bit alpha narrowed to 2-bit alpha!).
    let rgb10a2 = ImageSpec::new(SampleLayout::RGB10A2, ColorSpace::SRGB_ALPHA);
    assert!(RowPlan::build(ImageSpec::rgba8_srgb(), rgb10a2, &lossless, None).is_err());

    // 3. RGB16 -> RGBF16 rejected (16-bit integer precision > 11-bit F16 mantissa!).
    let rgb16 = ImageSpec::new(SampleLayout::RGB16, ColorSpace::LINEAR_SRGB);
    let rgbf16 = ImageSpec::rgbf16_linear();
    assert!(RowPlan::build(rgb16, rgbf16, &lossless, None).is_err());

    // 4. Unproven RGBF32 -> RGB8 rejected under GamutPolicy::Forbid, allowed with UNIT_INTERVAL.
    let forbid_gamut = ConvertPolicy::permissive().with_gamut(GamutPolicy::Forbid);
    assert!(
        RowPlan::build(
            ImageSpec::rgbf32_linear(),
            ImageSpec::rgb8_srgb(),
            &forbid_gamut,
            None
        )
        .is_err()
    );
    let proven_pol = forbid_gamut.with_facts(SampleFacts::from_float(FloatFacts::UNIT_INTERVAL));
    assert!(
        RowPlan::build(
            ImageSpec::rgbf32_linear(),
            ImageSpec::rgb8_srgb(),
            &proven_pol,
            None
        )
        .is_ok()
    );

    // 5. ImagePlan::execute rejects mismatched source spec BEFORE allocation.
    let img_plan = ImagePlan::build(
        ImageSpec::rgb8_srgb(),
        ImageSpec::rgba8_srgb(),
        ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
    .unwrap();
    let wrong_src = PixelBuf::new(2, 2, ImageSpec::rgba8_srgb()).unwrap();
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
    let src = PixelBuf::from_bytes(raw, 1, 1, ImageSpec::rgbaf32_linear()).unwrap();
    let pol = ConvertPolicy::permissive()
        .with_alpha(AlphaPolicy::CompositeOnto(Color::BLACK))
        .with_gamut(GamutPolicy::Clip);
    let out = convert_buf(
        &src.as_view(),
        ImageSpec::rgbf32_linear(),
        &pol,
        PeakSource::Measured,
    )
    .unwrap();
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
    let lin_100 = ImageSpec::rgbf32_linear().with_white(DiffuseWhite::new(100.0));
    let lin_203 = ImageSpec::rgbf32_linear().with_white(DiffuseWhite::new(203.0));
    let mut w_raw = Vec::new();
    for v in [1.0f32, 0.5, 0.25] {
        w_raw.extend_from_slice(&v.to_ne_bytes());
    }
    let src_100 = PixelBuf::from_bytes(w_raw, 1, 1, lin_100).unwrap();
    let out_203 = convert_buf(
        &src_100.as_view(),
        lin_203,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
    .unwrap();
    let f = out_203.as_view().row_as::<f32>(0).unwrap();
    let ratio = 100.0 / 203.0;
    assert!(
        (f[0] - ratio).abs() < 1e-5,
        "expected {ratio}, got {}",
        f[0]
    );
    assert!((f[1] - 0.5 * ratio).abs() < 1e-5);

    // 4. Transfer::Unknown rejected on transfer change / linear math, allowed on layout copy.
    let unknown_rgb = ImageSpec::rgb8_srgb().with_transfer(Transfer::Unknown);
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
            ImageSpec::rgbf32_linear(),
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
    let src_big = PixelBuf::from_bytes(big_f32, 1, 1, ImageSpec::rgbf32_linear()).unwrap();
    let out_f16 = convert_buf(
        &src_big.as_view(),
        ImageSpec::rgbf16_linear(),
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
    .unwrap();
    assert!(
        !out_f16.facts().float.contains(FloatFacts::UNIT_INTERVAL),
        "F32->F16 with 2.5/inf/NaN must not claim UNIT_INTERVAL"
    );
    assert_eq!(out_f16.facts().exact.effective_bits, 0);

    // 6. scan_facts on F16 detects semitransparent alpha, color, and FloatFacts!
    let mut f16_raw = Vec::new();
    for v in [0.25f32, 0.75, 0.5, 0.5] {
        f16_raw.extend_from_slice(&F16Sample::from_f32(v).to_bits().to_ne_bytes());
    }
    let f16_buf = PixelBuf::from_bytes(f16_raw, 1, 1, ImageSpec::rgbaf16_linear()).unwrap();
    let facts = scan_facts(&f16_buf.as_view());
    assert!(
        !facts.exact.all_opaque,
        "F16 alpha 0.5 must not be marked all_opaque"
    );
    assert!(
        !facts.exact.all_grayscale,
        "F16 (0.25, 0.75, 0.5) must not be marked all_grayscale"
    );
    assert!(facts.float.contains(FloatFacts::UNIT_INTERVAL));

    let mut f16_nan_raw = Vec::new();
    for bits in [0x7e00u16, 0x3c00, 0x3c00, 0x3c00] {
        f16_nan_raw.extend_from_slice(&bits.to_ne_bytes());
    }
    let f16_nan = PixelBuf::from_bytes(f16_nan_raw, 1, 1, ImageSpec::rgbaf16_linear()).unwrap();
    let nan_facts = scan_facts(&f16_nan.as_view());
    assert!(!nan_facts.float.contains(FloatFacts::NO_NAN));
}

#[cfg(feature = "bt2446a")]
#[test]
fn bt2446a_respects_baked_peaks_and_rejects_tiny_peaks() {
    use zenpixels_convert::{Bt2446A, Nits, ToneMapper};

    let mapper = Bt2446A::new(4000.0, 100.0);
    assert_eq!(mapper.baked_peaks(), Some((4000.0, 100.0)));

    // Tiny peak < 1e-3 must fail cleanly rather than producing NaN.
    let src_spec = ImageSpec::new(SampleLayout::RGBF32, ColorSpace::BT2100_PQ);
    let tiny_pol = ConvertPolicy::display_sdr(DiffuseWhite::BT2408, Nits::new(100.0));
    assert!(
        RowPlan::build(
            src_spec.clone(),
            ImageSpec::rgbf32_linear(),
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
    let out = convert_buf(
        &src.as_view(),
        ImageSpec::rgbf32_linear(),
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
            ImageSpec::rgb8_srgb(),
            &pol_lossless_mox,
            None
        )
        .is_err()
    );
    let pol_lossless_lcms = ConvertPolicy::lossless().with_cms(Arc::new(Lcms));
    assert!(
        RowPlan::build(
            src_p3_rgba.clone(),
            ImageSpec::rgb8_srgb(),
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
    assert!(
        RowPlan::build(
            src_p3_premul.clone(),
            ImageSpec::rgba8_srgb(),
            &pol_mox,
            None
        )
        .is_err()
    );
    assert!(RowPlan::build(src_p3_premul, ImageSpec::rgba8_srgb(), &pol_lcms, None).is_err());

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
    let buf_op = PixelBuf::from_vec(vec![128u8, 64, 32, 0], 1, 1, src_p3_opaque).unwrap();
    let out_mox = convert_buf(
        &buf_op.as_view(),
        ImageSpec::rgba8_srgb(),
        &pol_mox,
        PeakSource::Measured,
    )
    .unwrap();
    let out_lcms = convert_buf(
        &buf_op.as_view(),
        ImageSpec::rgba8_srgb(),
        &pol_lcms,
        PeakSource::Measured,
    )
    .unwrap();
    assert_eq!(out_mox.as_view().row(0)[3], 255);
    assert_eq!(out_lcms.as_view().row(0)[3], 255);

    // 4. MoxCms Gray sRGB matches Lcms Gray sRGB closely (not gamma 2.2!).
    let mut gray_vals = Vec::with_capacity(256);
    for g in 0u8..=255 {
        gray_vals.push(g);
    }
    let src_gray = PixelBuf::from_vec(gray_vals, 256, 1, ImageSpec::gray8_srgb()).unwrap();
    let dst_p3 = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: p3_icc,
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    let g_mox = convert_buf(
        &src_gray.as_view(),
        dst_p3.clone(),
        &pol_mox,
        PeakSource::Measured,
    )
    .unwrap();
    let g_lcms = convert_buf(&src_gray.as_view(), dst_p3, &pol_lcms, PeakSource::Measured).unwrap();
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
    let pv_y = PlaneView::new(y_spec, 4, 4, 4, &y_bytes).unwrap();
    let pv_cb = PlaneView::new(cb_spec, 2, 2, 2, &cb_bytes).unwrap();
    let pv_cr = PlaneView::new(cr_bad_spec, 2, 2, 8, &cr_bytes).unwrap();
    let planar = PlanarView::new(4, 4, &[pv_y, pv_cb, pv_cr]).unwrap();
    assert!(
        ycbcr_to_rgb_f32(
            &planar,
            YcbcrMatrix::Bt709,
            SignalRange::Full,
            Primaries::Bt709,
            Transfer::Srgb
        )
        .is_err(),
        "YCbCr must reject unsupported F32 Cr plane"
    );

    // 2. GainMap preserves base DiffuseWhite, rejects use_base_color_space=false,
    // and normalizes 1-channel metadata symmetrically across 3 channels.
    let base_spec = ImageSpec::rgb8_srgb().with_white(DiffuseWhite::new(100.0));
    let sdr = PixelBuf::from_vec(vec![128u8, 128, 128, 200, 100, 50], 2, 1, base_spec).unwrap();
    let mut hdr_raw = Vec::new();
    for v in [0.5f32, 0.5, 0.5, 2.0, 0.5, 0.1] {
        hdr_raw.extend_from_slice(&v.to_ne_bytes());
    }
    let hdr_buf = PixelBuf::from_bytes(hdr_raw, 2, 1, ImageSpec::rgbf32_linear()).unwrap();
    let asym_meta = GainMapMetadata {
        min_boost_stops: [0.0, -1.0, 0.5],
        max_boost_stops: [2.0, 3.0, 4.0],
        ..GainMapMetadata::symmetric(2.0)
    };
    let dual = extract_gain_map(&sdr.as_view(), &hdr_buf.as_view(), asym_meta).unwrap();
    assert_eq!(dual.metadata().min_boost_stops, [0.0; 3]);
    assert_eq!(dual.metadata().max_boost_stops, [2.0; 3]);

    let applied = apply_gain_map(&dual, 1.0).unwrap();
    assert_eq!(applied.spec().color.white(), Some(DiffuseWhite::new(100.0)));

    let mut alt_meta = dual.metadata();
    alt_meta.use_base_color_space = false;
    let (b, g, _) = dual.into_parts();
    let alt_dual = DualLayerHdr::new(b, g, alt_meta).unwrap();
    assert!(
        apply_gain_map(&alt_dual, 1.0).is_err(),
        "apply_gain_map must reject use_base_color_space == false"
    );

    // 3. Orient: reject color mismatch and handle 1x1 buffers without stride overflow.
    let p3_src = PixelBuf::from_vec(
        vec![10u8, 20, 30],
        1,
        1,
        ImageSpec::new(SampleLayout::RGB8, ColorSpace::DISPLAY_P3),
    )
    .unwrap();
    let mut srgb_dst = PixelBuf::new(1, 1, ImageSpec::rgb8_srgb()).unwrap();
    assert!(
        apply_orientation_into(
            &p3_src.as_view(),
            Orientation::Rotate90,
            &mut srgb_dst.as_view_mut()
        )
        .is_err()
    );

    let one_px = PixelBuf::from_vec(vec![11u8, 22, 33], 1, 1, ImageSpec::rgb8_srgb()).unwrap();
    for orient in Orientation::ALL {
        let out = apply_orientation(&one_px.as_view(), orient).unwrap();
        assert_eq!(out.width(), 1);
        assert_eq!(out.height(), 1);
        assert_eq!(out.as_view().row(0), &[11, 22, 33]);
    }

    // 4. convert_into updates sink facts cleanly.
    let mut dst_buf = PixelBuf::new(1, 1, ImageSpec::rgba8_srgb()).unwrap();
    convert_into(
        &one_px,
        &mut dst_buf,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
    .unwrap();
    assert_eq!(dst_buf.as_view().row(0), &[11, 22, 33, 255]);
    assert!(dst_buf.facts().exact.all_opaque);

    // Confirm ColorOrigin::from_cicp helper exists.
    let _orig = ColorOrigin::from_cicp(Cicp::SRGB);
}
