//! Integration tests for all 16 H.273 transfers (inverse accuracy + SCALAR/ALL parity),
//! OutputPlan HLG Display refusal, RGB10A2 Opaque canonicalization, Orientation propagation
//! & zero-copy bake_orientation_owned, lossless H.273 YCbCr matrices, siting-exact Bilinear
//! chroma upsampling, zero-alloc U8/U16/F32/BGR/Gray/PlanarRgb reconstruction, and
//! decoupled GainMapRowExtractor 3-ch + Shared metadata.

#[cfg(feature = "gainmap")]
use zenpixels_convert::gainmap::{
    GainMapApplyConfig, GainMapOrientation, GainMapRowExtractor, LinearInputChannels,
    apply_gain_map_oriented,
};
use zenpixels_convert::{
    Accelerators, AlphaMode, BandSizing, BandedBuf, ChromaSiting, Cicp, ColorOrigin, ColorSpace,
    ContentLightLevel, ConvertError, ConvertPolicy, DiffuseWhite, GamutPolicy, HdrMetadata,
    HlgMode, HlgRendering, ImagePlan, Orientation, OutputPlan, OutputSignaling, ParametricFormat,
    PeakSource, PixelBuf, PixelFormat, PixelRect, PlanarBuf, PlanarReconstructionOptions,
    PlaneFormat, PlaneRole, PlaneView, Primaries, RowPlan, SampleLayout, SampleType, SignalRange,
    Transfer, YcbcrMatrix, bake_orientation_owned, compact_in_place, compact_to_buf,
    reconstruct_planar_f32, reconstruct_planar_into, reconstruct_planar_to_buf,
};
#[cfg(feature = "gainmap")]
use zenpixels_convert::{
    GainMapChannels, GainMapColorSpace, GainMapComponent, GainMapMetadata, SignedFraction,
    UnsignedFraction,
};

#[test]
fn all_16_h273_transfers_inverse_accuracy_and_simd_parity() {
    let linear_spec = PixelFormat::RGBF32_LINEAR.with_white(DiffuseWhite::BT2408);
    let mut ramp = Vec::with_capacity(65 * 3);
    for i in 0..65 {
        let v = (i as f32) / 64.0;
        ramp.extend_from_slice(&[v, (v * 0.75).min(1.0), (1.0 - v).max(0.0)]);
    }

    let all_16_tc: [u8; 16] = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
    for tc in all_16_tc {
        let tf = Transfer::from_cicp(tc).expect("valid H.273 TC");
        #[cfg(not(feature = "hdr-transfers"))]
        if tf.is_hdr() {
            continue;
        }
        let prim = if tc == 18 {
            Primaries::Bt2020
        } else {
            Primaries::Bt709
        };
        let lin_for_tc = linear_spec.clone().with_primaries(prim);
        let enc_spec = PixelFormat::new(
            SampleLayout::RGBF32,
            ColorSpace::Rgb {
                primaries: prim,
                transfer: tf,
                range: SignalRange::Full,
                alpha: AlphaMode::None,
                white: Some(DiffuseWhite::BT2408),
            },
        );
        let src_buf = PixelBuf::from_typed(ramp.clone(), 65, 1, enc_spec.clone()).unwrap();

        let mut pol_scalar = ConvertPolicy::permissive()
            .with_gamut(GamutPolicy::PreserveExtended)
            .with_hlg_rendering(HlgRendering::REFERENCE_1000_NIT)
            .with_accelerators(Accelerators::SCALAR);
        pol_scalar.tone_map_hdr_to_sdr = false;
        let mut pol_all = ConvertPolicy::permissive()
            .with_gamut(GamutPolicy::PreserveExtended)
            .with_hlg_rendering(HlgRendering::REFERENCE_1000_NIT)
            .with_accelerators(Accelerators::ALL);
        pol_all.tone_map_hdr_to_sdr = false;

        // EOTF: encoded -> linear
        let lin_scalar =
            ImagePlan::build(&enc_spec, &lin_for_tc, &pol_scalar, PeakSource::Measured)
                .unwrap()
                .execute(&src_buf)
                .unwrap();
        let lin_all = ImagePlan::build(&enc_spec, &lin_for_tc, &pol_all, PeakSource::Measured)
            .unwrap()
            .execute(&src_buf)
            .unwrap();
        assert_eq!(
            lin_scalar.as_view().row(0),
            lin_all.as_view().row(0),
            "TC={tc} ({tf:?}) EOTF SCALAR vs ALL 0-bit parity"
        );

        // OETF: linear -> encoded
        let rt_scalar = ImagePlan::build(&lin_for_tc, &enc_spec, &pol_scalar, PeakSource::Measured)
            .unwrap()
            .execute(&lin_scalar)
            .unwrap();
        let rt_all = ImagePlan::build(&lin_for_tc, &enc_spec, &pol_all, PeakSource::Measured)
            .unwrap()
            .execute(&lin_all)
            .unwrap();
        assert_eq!(
            rt_scalar.as_view().row(0),
            rt_all.as_view().row(0),
            "TC={tc} ({tf:?}) OETF SCALAR vs ALL 0-bit parity"
        );

        // Check round-trip inverse accuracy (skip Log100/Log100Sqrt10 at exact 0.0 where log floor clamps).
        let rt_f: &[f32] = rt_all.as_view().row_as(0).unwrap();
        for (i, (&orig, &got)) in ramp.iter().zip(rt_f.iter()).enumerate() {
            if matches!(tf, Transfer::Log100 | Transfer::Log100Sqrt10) && orig < 0.02 {
                continue;
            }
            assert!(
                (orig - got).abs() < 2e-4,
                "TC={tc} ({tf:?}) idx={i}: orig={orig} got={got}"
            );
        }
    }

    // Extended negative values on Iec61966_2_4 (xvYCC, TC=11) and Bt1361 (TC=12) preserve sign!
    for tf in [Transfer::Iec61966_2_4, Transfer::Bt1361] {
        let enc = PixelFormat::RGBF32_LINEAR.with_transfer(tf);
        let neg = PixelBuf::from_typed(vec![-0.25_f32, -0.05, 0.5], 1, 1, enc.clone()).unwrap();
        let pol = ConvertPolicy::permissive().with_gamut(GamutPolicy::PreserveExtended);
        let lin = ImagePlan::build(
            &enc,
            &PixelFormat::RGBF32_LINEAR,
            &pol,
            PeakSource::Measured,
        )
        .unwrap()
        .execute(&neg)
        .unwrap();
        let lin_f: &[f32] = lin.as_view().row_as(0).unwrap();
        assert!(lin_f[0] < 0.0 && lin_f[1] < 0.0 && lin_f[2] > 0.0);
        let back = ImagePlan::build(
            &PixelFormat::RGBF32_LINEAR,
            &enc,
            &pol,
            PeakSource::Measured,
        )
        .unwrap()
        .execute(&lin)
        .unwrap();
        let back_f: &[f32] = back.as_view().row_as(0).unwrap();
        assert!((back_f[0] - (-0.25)).abs() < 1e-4);
        assert!((back_f[1] - (-0.05)).abs() < 1e-4);
        assert!((back_f[2] - 0.5).abs() < 1e-4);
    }
}

#[test]
fn output_plan_refuses_hlg_display_on_cicp_and_auto() {
    let hlg_display = PixelFormat::RGB16_BT2100_HLG_1000;
    let hlg_scene = PixelFormat::RGB16_BT2100_HLG_SCENE;
    let pol = ConvertPolicy::permissive();

    assert_eq!(
        *OutputPlan::build(
            &hlg_display,
            &hlg_display,
            &pol,
            PeakSource::Measured,
            OutputSignaling::CicpOnly,
        )
        .unwrap_err()
        .error(),
        ConvertError::OutputSignalingUnavailable
    );
    assert_eq!(
        *OutputPlan::build(
            &hlg_display,
            &hlg_display,
            &pol,
            PeakSource::Measured,
            OutputSignaling::Auto,
        )
        .unwrap_err()
        .error(),
        ConvertError::OutputSignalingUnavailable
    );
    assert!(
        OutputPlan::build(
            &hlg_scene,
            &hlg_scene,
            &pol,
            PeakSource::Measured,
            OutputSignaling::CicpOnly,
        )
        .is_ok()
    );
    assert!(
        OutputPlan::build(
            &hlg_scene,
            &hlg_scene,
            &pol,
            PeakSource::Measured,
            OutputSignaling::Auto,
        )
        .is_ok()
    );
    // Also verify HlgMode::Display variant explicitly.
    let custom_disp = PixelFormat::RGB16_BT2100_HLG_SCENE.with_transfer(Transfer::Hlg(
        HlgMode::Display(HlgRendering::REFERENCE_1000_NIT),
    ));
    assert_eq!(
        *OutputPlan::build(
            &custom_disp,
            &custom_disp,
            &pol,
            PeakSource::Measured,
            OutputSignaling::Auto,
        )
        .unwrap_err()
        .error(),
        ConvertError::OutputSignalingUnavailable
    );
}

#[test]
fn rgb10a2_unverified_opaque_canonicalization_and_orientation_propagation() {
    // 1. RGB10A2 with AlphaMode::Opaque and poisoned alpha bits (0 instead of 3) must canonicalize!
    let spec = PixelFormat::new(
        SampleLayout::RGB10A2,
        ColorSpace::SRGB.with_alpha(AlphaMode::Opaque),
    );
    let r = 511u32;
    let g = 256u32;
    let b = 1023u32;
    let poisoned_word: u32 = r | (g << 10) | (b << 20); // alpha = 0!
    let src = PixelBuf::from_pod_bytes_unchecked(vec![poisoned_word; 4], 4, 1, spec.clone())
        .unwrap()
        .with_orientation(Orientation::Rotate90);

    let row_plan = RowPlan::build(&spec, &spec, &ConvertPolicy::permissive(), None).unwrap();
    assert!(
        !row_plan.is_identity(),
        "unverified RGB10A2 Opaque must not use Memcpy"
    );

    let out = ImagePlan::build(
        &spec,
        &spec,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
    .unwrap()
    .execute(&src)
    .unwrap();
    assert_eq!(out.orientation(), Orientation::Rotate90);
    assert!(out.facts().all_opaque);
    let expected_word = poisoned_word | (3u32 << 30);
    let out_words: &[u32] = out.as_view().row_as(0).unwrap();
    assert_eq!(out_words, &[expected_word; 4]);

    // 2. Orientation propagates through BandedBuf conversion and load_bearing compaction.
    let mut dst_band = BandedBuf::new(4, 1, spec.clone(), BandSizing::MaxBandBytes(64)).unwrap();
    ImagePlan::build(
        &spec,
        &spec,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
    .unwrap()
    .execute_into(&src, &mut dst_band)
    .unwrap();
    assert_eq!(dst_band.orientation(), Orientation::Rotate90);

    let mut rgba_gray = PixelBuf::from_typed(
        vec![100u8, 100, 100, 255, 50, 50, 50, 255],
        2,
        1,
        ParametricFormat::RGBA8_SRGB,
    )
    .unwrap()
    .with_orientation(Orientation::Rotate270);
    let reduced = compact_to_buf(&rgba_gray.as_view()).unwrap().unwrap();
    assert_eq!(reduced.format().layout, SampleLayout::GRAY8);
    assert_eq!(reduced.orientation(), Orientation::Rotate270);
    assert!(
        compact_in_place(
            &mut rgba_gray,
            zenpixels_convert::AlphaCompaction::DropUnusedLanes,
        )
        .unwrap()
    );
    assert_eq!(rgba_gray.orientation(), Orientation::Rotate270);

    // 3. `bake_orientation_owned` is zero-copy on Identity and bakes + resets to Identity otherwise.
    let id_buf =
        PixelBuf::from_typed(vec![1u8, 2, 3, 4, 5, 6], 2, 1, ParametricFormat::RGB8_SRGB).unwrap();
    let orig_ptr = id_buf.as_view().as_strided_bytes().as_ptr();
    let baked_id = bake_orientation_owned(id_buf).unwrap();
    assert_eq!(baked_id.as_view().as_strided_bytes().as_ptr(), orig_ptr);
    assert_eq!(baked_id.orientation(), Orientation::Identity);

    let rot_buf = PixelBuf::from_typed(
        vec![10u8, 20, 30, 40, 50, 60],
        2,
        1,
        ParametricFormat::RGB8_SRGB,
    )
    .unwrap()
    .with_orientation(Orientation::Rotate90);
    let baked_rot = bake_orientation_owned(rot_buf).unwrap();
    assert_eq!(baked_rot.orientation(), Orientation::Identity);
    assert_eq!((baked_rot.width(), baked_rot.height()), (1, 2));
    assert_eq!(baked_rot.as_view().row(0), &[10, 20, 30]);
    assert_eq!(baked_rot.as_view().row(1), &[40, 50, 60]);
}

#[test]
fn ycbcr_lossless_matrices_bilinear_siting_planar_rgb_and_multi_targets() {
    // 1. All 9 supported H.273 Table 4 MC codes round-trip losslessly.
    for mc in [0u8, 1, 4, 5, 6, 7, 8, 9, 12] {
        let m = YcbcrMatrix::from_cicp_code(mc).expect("supported MC");
        assert_eq!(m.to_cicp_code(), mc);
    }
    assert!(YcbcrMatrix::Bt470Bg.same_coefficients(YcbcrMatrix::Bt601));

    // 2. YCgCo (MC=8) exact pure-primary reconstruction (Full range, 8-bit).
    // Pure green: R=0, G=1, B=0 -> Y=0.5 (~128), Cg=+0.5 (~255), Co=0 (~128).
    let y_data = [128u8];
    let cg_data = [255u8];
    let co_data = [128u8];
    let py = PlaneView::from_samples(
        &y_data,
        1,
        1,
        PlaneFormat::new(PlaneRole::Y, SampleType::U8),
    )
    .unwrap();
    let pcg = PlaneView::from_samples(
        &cg_data,
        1,
        1,
        PlaneFormat::new(PlaneRole::Cb, SampleType::U8),
    )
    .unwrap();
    let pco = PlaneView::from_samples(
        &co_data,
        1,
        1,
        PlaneFormat::new(PlaneRole::Cr, SampleType::U8),
    )
    .unwrap();
    let ycgco_view = zenpixels_convert::PlanarView::new(1, 1, &[py, pcg, pco])
        .unwrap()
        .with_color(ColorSpace::SRGB)
        .with_matrix_coefficients(zenpixels_convert::MatrixCoefficients::YCGCO);
    let ycgco_rgb = reconstruct_planar_to_buf(
        &ycgco_view,
        SampleLayout::RGB8,
        &PlanarReconstructionOptions::nearest(),
    )
    .unwrap();
    let px = ycgco_rgb.as_view().row(0);
    assert!(
        px[0] <= 2 && px[1] >= 253 && px[2] <= 2,
        "YCgCo green got {px:?}"
    );

    // 3. ChromaDerivedNcl (MC=12) derives Kr/Kb from DisplayP3 primaries!
    let cd_view = ycgco_view
        .clone()
        .with_color(ColorSpace::DISPLAY_P3)
        .with_matrix_coefficients(zenpixels_convert::MatrixCoefficients::CHROMA_DERIVED_NCL);
    let cd_f32 = reconstruct_planar_f32(&cd_view, &PlanarReconstructionOptions::nearest()).unwrap();
    assert_eq!(cd_f32.format().layout, SampleLayout::RGBF32);

    // 3b. MatrixCoefficients <-> YcbcrMatrix lossless interop.
    assert_eq!(
        YcbcrMatrix::from_matrix_coefficients(zenpixels_convert::MatrixCoefficients::BT709),
        Some(YcbcrMatrix::Bt709)
    );
    assert_eq!(
        YcbcrMatrix::from_matrix_coefficients(zenpixels_convert::MatrixCoefficients::ICTCP),
        None
    );
    assert_eq!(
        zenpixels_convert::MatrixCoefficients::from(YcbcrMatrix::Bt2020),
        zenpixels_convert::MatrixCoefficients::BT2020_NCL
    );

    // 4. Siting-exact Bilinear 4:2:0 upsampling (Center = 9:3:3:1, TopLeft, odd crop_origin, Unknown refused).
    let y_4x4 = [128u8; 16];
    // 2x2 chroma grid: top-left=0, top-right=255, bottom-left=0, bottom-right=255 (horizontal gradient)
    let cb_2x2 = [0u8, 255, 0, 255];
    let cr_2x2 = [128u8; 4];
    let y_p = PlaneView::from_samples(&y_4x4, 4, 4, PlaneFormat::new(PlaneRole::Y, SampleType::U8))
        .unwrap();
    let cb_p = PlaneView::from_samples(
        &cb_2x2,
        2,
        2,
        PlaneFormat::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2),
    )
    .unwrap();
    let cr_p = PlaneView::from_samples(
        &cr_2x2,
        2,
        2,
        PlaneFormat::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2),
    )
    .unwrap();

    // Unknown siting + Bilinear without fallback must fail!
    let unk_420 = zenpixels_convert::PlanarView::new(4, 4, &[y_p, cb_p, cr_p])
        .unwrap()
        .with_color(ColorSpace::SRGB)
        .with_matrix_coefficients(zenpixels_convert::MatrixCoefficients::BT709);
    assert_eq!(
        *reconstruct_planar_to_buf(
            &unk_420,
            SampleLayout::RGBF32,
            &PlanarReconstructionOptions::bilinear()
        )
        .unwrap_err()
        .error(),
        ConvertError::UnsupportedRoute
    );

    // With Center siting: x=0 -> c[0]=0, x=1 -> 0.75*0 + 0.25*255, x=2 -> 0.25*0 + 0.75*255, x=3 -> c[1]=255!
    let hdr = HdrMetadata {
        content_light_level: Some(ContentLightLevel::new(500, 100)),
        mastering: None,
    };
    let origin = ColorOrigin::from_cicp(Cicp::SRGB);
    let center_420 = unk_420
        .clone()
        .with_chroma_siting(ChromaSiting::Center)
        .with_hdr(hdr)
        .with_origin(&origin)
        .with_orientation(Orientation::Rotate90);
    let full_bi = reconstruct_planar_to_buf(
        &center_420,
        SampleLayout::RGBF32,
        &PlanarReconstructionOptions::bilinear(),
    )
    .unwrap();
    assert_eq!(full_bi.hdr(), hdr);
    assert_eq!(full_bi.origin(), &origin);
    assert_eq!(full_bi.orientation(), Orientation::Rotate90);

    let row0: &[f32] = full_bi.as_view().row_as(0).unwrap();
    // Blue channel increases monotonically across x = 0, 1, 2, 3 (strictly distinct intermediate taps!).
    assert!(row0[2] < row0[5] && row0[5] < row0[8] && row0[8] < row0[11]);

    // Odd crop origin (1, 1) bilinear MUST match full-frame bilinear cropped at (1, 1)!
    let odd_crop = center_420.crop(PixelRect::new(1, 1, 2, 2)).unwrap();
    let crop_bi = reconstruct_planar_to_buf(
        &odd_crop,
        SampleLayout::RGBF32,
        &PlanarReconstructionOptions::bilinear(),
    )
    .unwrap();
    for y in 0..2 {
        assert_eq!(
            crop_bi.as_view().row(y),
            full_bi
                .as_view()
                .crop(PixelRect::new(1, 1, 2, 2))
                .unwrap()
                .row(y),
            "odd-origin 4:2:0 bilinear row {y} must match full-frame bilinear + crop"
        );
    }

    // TopLeft siting differs from Center siting at x=1 (0.5/0.5 vs 0.75/0.25)!
    let tl_420 = unk_420.clone().with_chroma_siting(ChromaSiting::TopLeft);
    let tl_bi = reconstruct_planar_to_buf(
        &tl_420,
        SampleLayout::RGBF32,
        &PlanarReconstructionOptions::bilinear(),
    )
    .unwrap();
    let tl_r0: &[f32] = tl_bi.as_view().row_as(0).unwrap();
    assert!(
        (tl_r0[5] - row0[5]).abs() > 0.1,
        "TopLeft vs Center siting must produce distinct x=1"
    );

    // 5. Zero-alloc `reconstruct_planar_into` into BGR8, RGB8, RGBA8, GRAY8 + strict dst validation!
    let mut bgr8_buf = PixelBuf::new(4, 4, ParametricFormat::BGR8_SRGB).unwrap();
    reconstruct_planar_into(
        &center_420,
        &mut bgr8_buf.as_view_mut(),
        &PlanarReconstructionOptions::bilinear(),
    )
    .unwrap();
    let mut rgb8_buf = PixelBuf::new(4, 4, ParametricFormat::RGB8_SRGB).unwrap();
    reconstruct_planar_into(
        &center_420,
        &mut rgb8_buf.as_view_mut(),
        &PlanarReconstructionOptions::bilinear(),
    )
    .unwrap();
    // BGR8 matches RGB8 with R<->B swapped!
    for x in 0..4 {
        let bgr = &bgr8_buf.as_view().row(0)[x * 3..x * 3 + 3];
        let rgb = &rgb8_buf.as_view().row(0)[x * 3..x * 3 + 3];
        assert_eq!(bgr, &[rgb[2], rgb[1], rgb[0]]);
    }

    // `reconstruct_planar_into` accepts RGBA8_SRGB (Straight) when source has no alpha plane (fills 255),
    // and refuses mismatched destination color space (e.g., DisplayP3 or Linear) or untagged source!
    let mut rgba8_buf = PixelBuf::new(4, 4, PixelFormat::RGBA8_SRGB).unwrap();
    reconstruct_planar_into(
        &center_420,
        &mut rgba8_buf.as_view_mut(),
        &PlanarReconstructionOptions::bilinear(),
    )
    .unwrap();
    assert_eq!(rgba8_buf.row(0)[3], 255);

    let mut wrong_p3_dst = PixelBuf::new(
        4,
        4,
        PixelFormat::RGB8_SRGB.with_primaries(Primaries::DisplayP3),
    )
    .unwrap();
    assert_eq!(
        *reconstruct_planar_into(
            &center_420,
            &mut wrong_p3_dst.as_view_mut(),
            &PlanarReconstructionOptions::bilinear(),
        )
        .unwrap_err()
        .error(),
        ConvertError::UnsupportedRoute
    );
    let untagged_420 = zenpixels_convert::PlanarView::new(4, 4, &[y_p, cb_p, cr_p]).unwrap();
    assert_eq!(
        *reconstruct_planar_into(
            &untagged_420,
            &mut rgb8_buf.as_view_mut(),
            &PlanarReconstructionOptions::nearest(),
        )
        .unwrap_err()
        .error(),
        ConvertError::UnsupportedRoute
    );

    let rgb16_buf = reconstruct_planar_to_buf(
        &center_420,
        SampleLayout::RGB16,
        &PlanarReconstructionOptions::bilinear(),
    )
    .unwrap();
    assert_eq!(rgb16_buf.format().layout, SampleLayout::RGB16);

    // Monochrome Y -> GRAY8 zero-alloc!
    let mono_view = zenpixels_convert::PlanarView::new(4, 4, &[y_p])
        .unwrap()
        .with_color(ColorSpace::GRAY_SRGB);
    let mut gray8 = PixelBuf::new(4, 4, ParametricFormat::GRAY8_SRGB).unwrap();
    reconstruct_planar_into(
        &mono_view,
        &mut gray8.as_view_mut(),
        &PlanarReconstructionOptions::nearest(),
    )
    .unwrap();
    assert_eq!(gray8.as_view().row(0), &[128, 128, 128, 128]);

    // 6. Planar RGB (F32 R, G, B + Alpha) -> RGBAF32 and BGRA8!
    let p_rgb = PlanarBuf::new(
        2,
        1,
        &[
            PlaneFormat::new(PlaneRole::R, SampleType::F32),
            PlaneFormat::new(PlaneRole::G, SampleType::F32),
            PlaneFormat::new(PlaneRole::B, SampleType::F32),
            PlaneFormat::new(PlaneRole::Alpha, SampleType::F32),
        ],
    )
    .unwrap()
    .with_color(ColorSpace::LINEAR_SRGB_ALPHA);
    let mut p_rgb = p_rgb;
    p_rgb
        .plane_view_mut_by_role(PlaneRole::R)
        .unwrap()
        .row_as_mut::<f32>(0)
        .unwrap()
        .copy_from_slice(&[1.0, 0.25]);
    p_rgb
        .plane_view_mut_by_role(PlaneRole::G)
        .unwrap()
        .row_as_mut::<f32>(0)
        .unwrap()
        .copy_from_slice(&[0.5, 0.75]);
    p_rgb
        .plane_view_mut_by_role(PlaneRole::B)
        .unwrap()
        .row_as_mut::<f32>(0)
        .unwrap()
        .copy_from_slice(&[0.0, 1.0]);
    p_rgb
        .plane_view_mut_by_role(PlaneRole::Alpha)
        .unwrap()
        .row_as_mut::<f32>(0)
        .unwrap()
        .copy_from_slice(&[1.0, 0.5]);

    let out_f32 = reconstruct_planar_to_buf(
        &p_rgb.as_view(),
        SampleLayout::RGBAF32,
        &PlanarReconstructionOptions::nearest(),
    )
    .unwrap();
    assert_eq!(
        out_f32.as_view().row_as::<f32>(0).unwrap(),
        &[1.0, 0.5, 0.0, 1.0, 0.25, 0.75, 1.0, 0.5]
    );
    let out_bgra8 = reconstruct_planar_to_buf(
        &p_rgb.as_view(),
        SampleLayout::BGRA8,
        &PlanarReconstructionOptions::nearest(),
    )
    .unwrap();
    assert_eq!(
        out_bgra8.as_view().row(0),
        &[0, 128, 255, 255, 255, 191, 64, 128]
    );
    // Dropping alpha from planar RGB+A into RGB8 must refuse with AlphaLossForbidden!
    assert_eq!(
        *reconstruct_planar_to_buf(
            &p_rgb.as_view(),
            SampleLayout::RGB8,
            &PlanarReconstructionOptions::nearest()
        )
        .unwrap_err()
        .error(),
        ConvertError::AlphaLossForbidden
    );
    // And `reconstruct_planar_into` rejects premultiplied source written into Straight dst!
    let p_premul = p_rgb
        .as_view()
        .with_alpha_override(AlphaMode::Premultiplied);
    let mut dst_straight = PixelBuf::new(2, 1, PixelFormat::RGBAF32_LINEAR).unwrap();
    assert_eq!(
        *reconstruct_planar_into(
            &p_premul,
            &mut dst_straight.as_view_mut(),
            &PlanarReconstructionOptions::nearest(),
        )
        .unwrap_err()
        .error(),
        ConvertError::UnsupportedRoute
    );
}

#[cfg(feature = "gainmap")]
#[test]
fn gainmap_decoupled_3ch_shared_metadata_and_passthrough_orientation() {
    let comp = GainMapComponent {
        min_gain_stops: SignedFraction::try_new(0, 1).unwrap(),
        max_gain_stops: SignedFraction::try_new(2, 1).unwrap(),
        gamma: UnsignedFraction::try_new(1, 1).unwrap(),
        base_offset: SignedFraction::try_new(1, 64).unwrap(),
        alternate_offset: SignedFraction::try_new(1, 64).unwrap(),
    };
    let shared_meta = GainMapMetadata::new(
        GainMapChannels::Shared(comp),
        UnsignedFraction::try_new(0, 1).unwrap(),
        UnsignedFraction::try_new(2, 1).unwrap(),
        GainMapColorSpace::Base,
    )
    .unwrap();
    let per_ch_meta = GainMapMetadata::new(
        GainMapChannels::PerChannel([comp; 3]),
        UnsignedFraction::try_new(0, 1).unwrap(),
        UnsignedFraction::try_new(2, 1).unwrap(),
        GainMapColorSpace::Base,
    )
    .unwrap();

    // 3-channel gain map with 1-record `Shared` metadata must succeed!
    let mut ext_shared = GainMapRowExtractor::new(shared_meta, Primaries::Bt709).unwrap();
    let base = [0.5_f32, 0.5, 0.5, 0.25, 0.5, 1.0];
    let alt = [2.0_f32, 1.0, 0.5, 1.0, 1.0, 2.0];
    let mut dst_3ch = [0u8; 6];
    ext_shared
        .encode_row_linear_f32(
            &base,
            &alt,
            LinearInputChannels::Rgb,
            SampleLayout::RGB8,
            &mut dst_3ch,
        )
        .unwrap();
    // Pixel 0: R boost ~4x (2 stops -> ~255), G boost ~2x (1 stop -> ~128), B boost 1x (0 stops -> 0)
    assert!(dst_3ch[0] > 240 && dst_3ch[1] > 100 && dst_3ch[1] < 160 && dst_3ch[2] < 10);

    // 1-channel gain map with 3-record `PerChannel` metadata must be refused!
    let mut ext_per = GainMapRowExtractor::new(per_ch_meta, Primaries::Bt709).unwrap();
    let mut dst_1ch = [0u8; 2];
    assert_eq!(
        *ext_per
            .encode_row_linear_f32(
                &base,
                &alt,
                LinearInputChannels::Rgb,
                SampleLayout::GRAY8,
                &mut dst_1ch,
            )
            .unwrap_err()
            .error(),
        ConvertError::UnsupportedRoute
    );

    // `apply_gain_map_oriented` in `Passthrough` attaches `orient` onto the returned `PixelBuf`!
    let gm_spec = PixelFormat::new(
        SampleLayout::GRAY8,
        ColorSpace::Untagged {
            total_channels: 1,
            alpha: AlphaMode::None,
        },
    );
    let base_buf = PixelBuf::new_white(4, 2, ParametricFormat::RGB8_SRGB).unwrap();
    let gm_buf = PixelBuf::new(4, 2, gm_spec).unwrap();
    let dual = zenpixels_convert::DualLayerHdrBuf::new(base_buf, gm_buf, shared_meta)
        .unwrap()
        .with_orientation(Orientation::Rotate90);
    let cfg = GainMapApplyConfig::from_headroom_stops(2.0)
        .with_orientation(GainMapOrientation::Passthrough);
    let (out, out_orient) =
        apply_gain_map_oriented(&dual.as_view(), &cfg, &enough::Unstoppable).unwrap();
    assert_eq!(out_orient, Orientation::Rotate90);
    assert_eq!(out.orientation(), Orientation::Rotate90);
    assert_eq!(out.display_dimensions(), (2, 4));
}

/// Construct a complete, spec-conformant ICC v4.4 `mntr` `RGB ` matrix/TRC profile from
/// `CustomPrimaries` (using Annex 1 `to_icc_d50_matrix_f64()` for `rXYZ`/`gXYZ`/`bXYZ`)
/// plus a linear `curv` TRC, `wtpt` D50, and a 44-byte `cicp` custom-primaries tag!
fn build_matrix_trc_icc_with_custom_cicp(
    cp: zenpixels_convert::CustomPrimaries,
    tc: u8,
) -> Vec<u8> {
    let d50 = cp.to_icc_d50_matrix_f64().unwrap();
    let s15 = |v: f64| -> [u8; 4] {
        let q = if v >= 0.0 {
            (v * 65536.0 + 0.5) as i32
        } else {
            (v * 65536.0 - 0.5) as i32
        };
        q.to_be_bytes()
    };
    let xyz_tag = |col: usize| -> [u8; 20] {
        let mut b = [0u8; 20];
        b[0..4].copy_from_slice(b"XYZ ");
        b[8..12].copy_from_slice(&s15(d50[0][col]));
        b[12..16].copy_from_slice(&s15(d50[1][col]));
        b[16..20].copy_from_slice(&s15(d50[2][col]));
        b
    };
    let mut wtpt = [0u8; 20];
    wtpt[0..4].copy_from_slice(b"XYZ ");
    wtpt[8..12].copy_from_slice(&s15(0.9642));
    wtpt[12..16].copy_from_slice(&s15(1.0));
    wtpt[16..20].copy_from_slice(&s15(0.8249));

    // Linear TRC (`curv` with count = 0 = identity) or pure sRGB `para` funcType 3.
    let trc: Vec<u8> = if tc == 13 {
        let mut p = vec![0u8; 32];
        p[0..4].copy_from_slice(b"para");
        p[8..10].copy_from_slice(&3u16.to_be_bytes());
        p[12..16].copy_from_slice(&s15(2.4));
        p[16..20].copy_from_slice(&s15(1.0 / 1.055));
        p[20..24].copy_from_slice(&s15(0.055 / 1.055));
        p[24..28].copy_from_slice(&s15(1.0 / 12.92));
        p[28..32].copy_from_slice(&s15(0.04045));
        p
    } else {
        let mut c = vec![0u8; 12];
        c[0..4].copy_from_slice(b"curv");
        c
    };

    let cicp_tag =
        zenpixels_convert::CicpTag::with_custom_primaries(cp, tc, 0, true).to_tag_bytes_vec();
    let r_xyz = xyz_tag(0);
    let g_xyz = xyz_tag(1);
    let b_xyz = xyz_tag(2);

    // 8 tags: wtpt, rXYZ, gXYZ, bXYZ, rTRC, gTRC, bTRC (shared), cicp.
    let tag_count = 8u32;
    let table_end = 132 + (tag_count as usize) * 12;
    let mut body = Vec::new();
    let mut entries: Vec<([u8; 4], u32, u32)> = Vec::new();

    let push_tag = |entries: &mut Vec<([u8; 4], u32, u32)>,
                    body: &mut Vec<u8>,
                    sig: [u8; 4],
                    payload: &[u8]| {
        let off = (table_end + body.len()) as u32;
        body.extend_from_slice(payload);
        body.resize((body.len() + 3) & !3, 0);
        entries.push((sig, off, payload.len() as u32));
        off
    };

    push_tag(&mut entries, &mut body, *b"wtpt", &wtpt);
    push_tag(&mut entries, &mut body, *b"rXYZ", &r_xyz);
    push_tag(&mut entries, &mut body, *b"gXYZ", &g_xyz);
    push_tag(&mut entries, &mut body, *b"bXYZ", &b_xyz);
    let trc_off = push_tag(&mut entries, &mut body, *b"rTRC", &trc);
    entries.push((*b"gTRC", trc_off, trc.len() as u32));
    entries.push((*b"bTRC", trc_off, trc.len() as u32));
    push_tag(&mut entries, &mut body, *b"cicp", &cicp_tag);

    let total_size = (table_end + body.len()) as u32;
    let mut icc = vec![0u8; table_end];
    icc[0..4].copy_from_slice(&total_size.to_be_bytes());
    icc[8..12].copy_from_slice(&[4, 0x40, 0, 0]);
    icc[12..16].copy_from_slice(b"mntr");
    icc[16..20].copy_from_slice(b"RGB ");
    icc[20..24].copy_from_slice(b"XYZ ");
    icc[36..40].copy_from_slice(b"acsp");
    // PCS illuminant D50 (0.9642, 1.0, 0.8249) at bytes 68..80
    icc[68..72].copy_from_slice(&s15(0.9642));
    icc[72..76].copy_from_slice(&s15(1.0));
    icc[76..80].copy_from_slice(&s15(0.8249));
    icc[128..132].copy_from_slice(&tag_count.to_be_bytes());
    for (i, (sig, off, len)) in entries.into_iter().enumerate() {
        let base = 132 + i * 12;
        icc[base..base + 4].copy_from_slice(&sig);
        icc[base + 4..base + 8].copy_from_slice(&off.to_be_bytes());
        icc[base + 8..base + 12].copy_from_slice(&len.to_be_bytes());
    }
    icc.extend_from_slice(&body);
    icc
}

#[test]
fn icc_44byte_cicptype_interop_with_cms_load_bearing_and_gainmap() {
    use std::sync::Arc;
    use zenpixels_convert::{
        ChannelOrder, Cicp, CicpTag, CustomPrimaries, compact_to_buf, validate_icc_structure,
    };

    // 1. Build ACEScg (D60) and custom D65 ICC v4.4 profiles carrying 44-byte cicp tags.
    let acescg_icc = build_matrix_trc_icc_with_custom_cicp(CustomPrimaries::ACES_CG, 8);
    validate_icc_structure(&acescg_icc).unwrap();
    let acescg_tag = CicpTag::from_icc(&acescg_icc).unwrap();
    assert_eq!(acescg_tag.cicp(), Cicp::new(2, 8, 0, true));
    assert!(acescg_tag.custom_primaries().is_some());

    let custom_d65 =
        CustomPrimaries::new((0.65, 0.33), (0.28, 0.62), (0.15, 0.06), (0.3127, 0.3290)).unwrap();
    let d65_icc = build_matrix_trc_icc_with_custom_cicp(custom_d65, 13);
    validate_icc_structure(&d65_icc).unwrap();
    assert!(
        CicpTag::from_icc(&d65_icc)
            .unwrap()
            .custom_primaries()
            .is_some()
    );

    // 2. Load-bearing chroma collapse respects custom white point in 44-byte cicp tag:
    //    - Custom D65 neutral RGB ICC -> allowed to collapse to Gray!
    //    - ACEScg D60 neutral RGB ICC -> suppressed (stays RGB ICC, never shifts D60 to D65 Gray)!
    let d65_spec = PixelFormat::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::from(d65_icc.clone()),
            color_channels: 3,
            alpha: AlphaMode::None,
            intent: zenpixels_convert::RenderingIntent::RelativeColorimetric,
        },
    );
    let d65_gray_buf = PixelBuf::from_typed(vec![128u8; 4 * 3], 2, 2, d65_spec).unwrap();
    let reduced_d65 = compact_to_buf(&d65_gray_buf.as_view())
        .unwrap()
        .expect("custom D65 neutral ICC should collapse to Gray");
    assert_eq!(reduced_d65.format().layout.order, ChannelOrder::Gray);

    let acescg_spec_u8 = PixelFormat::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::from(acescg_icc.clone()),
            color_channels: 3,
            alpha: AlphaMode::None,
            intent: zenpixels_convert::RenderingIntent::RelativeColorimetric,
        },
    );
    let acescg_gray_buf = PixelBuf::from_typed(vec![128u8; 4 * 3], 2, 2, acescg_spec_u8).unwrap();
    assert!(
        compact_to_buf(&acescg_gray_buf.as_view())
            .unwrap()
            .is_none(),
        "ACEScg (D60) neutral RGB ICC must NOT collapse to D65 Gray"
    );

    // 3. CMS interop (MoxCms & Lcms): both parse the 44-byte cicp ICC profile cleanly AND match
    //    native `Primaries::Custom(ACES_CG)` -> linear sRGB!
    let native_acescg_spec = PixelFormat::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: Primaries::Custom(CustomPrimaries::ACES_CG),
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let icc_acescg_spec = PixelFormat::new(
        SampleLayout::RGBF32,
        ColorSpace::Icc {
            profile: Arc::from(acescg_icc),
            color_channels: 3,
            alpha: AlphaMode::None,
            intent: zenpixels_convert::RenderingIntent::RelativeColorimetric,
        },
    );
    let sample_pixels: Vec<f32> = vec![
        1.0, 1.0, 1.0, // D60 white -> must adapt to D65 white (1, 1, 1)
        0.5, 0.25, 0.1, // warm chromatic
        0.1, 0.4, 0.3, // cool chromatic
    ];
    let native_src =
        PixelBuf::from_typed(sample_pixels.clone(), 3, 1, native_acescg_spec.clone()).unwrap();
    #[cfg_attr(
        not(any(feature = "cms-moxcms", feature = "cms-lcms")),
        allow(unused_variables)
    )]
    let icc_src = PixelBuf::from_typed(sample_pixels, 3, 1, icc_acescg_spec).unwrap();

    let native_out = ImagePlan::build(
        &native_acescg_spec,
        &PixelFormat::RGBF32_LINEAR,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
    .unwrap()
    .execute(&native_src)
    .unwrap();
    #[cfg_attr(
        not(any(feature = "cms-moxcms", feature = "cms-lcms")),
        allow(unused_variables)
    )]
    let native_row: &[f32] = native_out.as_view().row_as(0).unwrap();

    #[cfg(feature = "cms-moxcms")]
    {
        let mox_pol = ConvertPolicy::permissive().with_cms(Arc::new(zenpixels_convert::MoxCms));
        let mox_out = ImagePlan::build(
            icc_src.format(),
            &PixelFormat::RGBF32_LINEAR,
            &mox_pol,
            PeakSource::Measured,
        )
        .unwrap()
        .execute(&icc_src)
        .unwrap();
        let mox_row: &[f32] = mox_out.as_view().row_as(0).unwrap();
        for (i, (&got, &want)) in mox_row.iter().zip(native_row).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "MoxCms 44B-cicp ICC mismatch at ch {i}: got {got}, want {want}"
            );
        }
    }

    #[cfg(feature = "cms-lcms")]
    {
        let lcms_pol = ConvertPolicy::permissive().with_cms(Arc::new(zenpixels_convert::Lcms));
        let lcms_out = ImagePlan::build(
            icc_src.format(),
            &PixelFormat::RGBF32_LINEAR,
            &lcms_pol,
            PeakSource::Measured,
        )
        .unwrap()
        .execute(&icc_src)
        .unwrap();
        let lcms_row: &[f32] = lcms_out.as_view().row_as(0).unwrap();
        for (i, (&got, &want)) in lcms_row.iter().zip(native_row).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "Lcms 44B-cicp ICC mismatch at ch {i}: got {got}, want {want}"
            );
        }

        // Also verify Lcms converting directly from `ColorSpace::Icc` -> `Primaries::Custom(ACES_CG)`
        // preserves true D60 white point (non-D65 white-point regression test!).
        let lcms_back = ImagePlan::build(
            icc_src.format(),
            &native_acescg_spec,
            &lcms_pol,
            PeakSource::Measured,
        )
        .unwrap()
        .execute(&icc_src)
        .unwrap();
        let back_row: &[f32] = lcms_back.as_view().row_as(0).unwrap();
        // White (1, 1, 1) in ACEScg ICC -> ACEScg structured RGB must stay (1, 1, 1)!
        for (c, &val) in back_row.iter().enumerate().take(3) {
            assert!(
                (val - 1.0).abs() < 1e-3,
                "Lcms ACEScg white ch {c}: got {val} (expected 1.0)"
            );
        }
    }

    // 4. Gain-map extraction resolves primaries from 44-byte custom-primaries cicp tag on ColorSpace::Icc!
    #[cfg(all(feature = "gainmap", feature = "cms-lcms"))]
    {
        let gm_cfg = zenpixels_convert::gainmap::GainMapExtractConfig::default()
            .with_scale_factor(1)
            .with_policy(ConvertPolicy::permissive().with_cms(Arc::new(zenpixels_convert::Lcms)));
        let base_sdr =
            PixelBuf::from_typed(vec![0.5_f32; 4 * 3], 2, 2, icc_src.format().clone()).unwrap();
        let alt_hdr = PixelBuf::from_typed(vec![1.0_f32; 4 * 3], 2, 2, native_acescg_spec).unwrap();
        let (gm_buf, _meta) = zenpixels_convert::gainmap::extract_gain_map_only(
            &base_sdr.as_view(),
            &alt_hdr.as_view(),
            &gm_cfg,
            &zenpixels_convert::Unstoppable,
        )
        .unwrap();
        assert_eq!(gm_buf.width(), 2);
        assert_eq!(gm_buf.height(), 2);
    }
}

#[test]
fn enough_stop_reexports_available_at_convert_root() {
    let stop: &dyn zenpixels_convert::Stop = &zenpixels_convert::Unstoppable;
    let check: Result<(), zenpixels_convert::StopReason> = stop.check();
    assert!(check.is_ok());
}

#[test]
fn bundle_1_and_2_policy_default_to_rgb8_orient_owned_and_alpha_compaction() {
    use zenpixels_convert::{
        AlphaCompaction, RowPlan, apply_orientation, apply_orientation_owned,
        bake_orientation_owned, compact_in_place, scan_facts, to_gray8_if_opaque,
        to_rgb8_if_opaque,
    };

    // M1: ConvertPolicy::default() is lossless (forbids depth, alpha, and range loss, preserves extended gamut).
    let def = ConvertPolicy::default();
    assert_eq!(def.depth, zenpixels_convert::DepthPolicy::Forbid);
    assert_eq!(def.alpha, zenpixels_convert::AlphaPolicy::Forbid);
    assert_eq!(def.gamut, zenpixels_convert::GamutPolicy::PreserveExtended);
    assert_eq!(def.range, zenpixels_convert::RangePolicy::Forbid);
    assert_eq!(
        *RowPlan::build(
            &PixelFormat::RGB16_SRGB,
            &PixelFormat::RGB8_SRGB,
            &def,
            None
        )
        .unwrap_err()
        .error(),
        ConvertError::DepthReductionForbidden
    );
    assert_eq!(
        *RowPlan::build(
            &PixelFormat::RGBA8_SRGB,
            &PixelFormat::RGB8_SRGB,
            &def,
            None
        )
        .unwrap_err()
        .error(),
        ConvertError::AlphaLossForbidden
    );
    assert_eq!(
        *RowPlan::build(
            PixelFormat::RGB8_SRGB.with_primaries(Primaries::DisplayP3),
            &zenpixels_convert::formats::RGB8_SRGB,
            &def,
            None,
        )
        .unwrap_err()
        .error(),
        ConvertError::DepthReductionForbidden
    );
    assert!(
        RowPlan::build(
            PixelFormat::RGBF32_LINEAR.with_primaries(Primaries::DisplayP3),
            &zenpixels_convert::formats::RGBF32_LINEAR,
            &def,
            None,
        )
        .is_ok(),
        "lossless default preserves extended floating-point gamut"
    );
    assert_eq!(
        *RowPlan::build(
            PixelFormat::RGB8_SRGB.with_range(SignalRange::Narrow),
            &zenpixels_convert::formats::RGB8_SRGB,
            &def,
            None,
        )
        .unwrap_err()
        .error(),
        ConvertError::RangeConversionForbidden
    );

    // M2: to_rgb8_if_opaque / to_gray8_if_opaque refuse unproven/semi-transparent alpha, succeed on proven opaque.
    let semi =
        PixelBuf::from_typed(vec![10u8, 20, 30, 128], 1, 1, PixelFormat::RGBA8_SRGB).unwrap();
    assert_eq!(
        *to_rgb8_if_opaque(&semi.as_view()).unwrap_err().error(),
        ConvertError::AlphaLossForbidden
    );
    assert_eq!(
        *to_gray8_if_opaque(&semi.as_view()).unwrap_err().error(),
        ConvertError::AlphaLossForbidden
    );
    let mut opaque_rgba =
        PixelBuf::from_typed(vec![10u8, 20, 30, 255], 1, 1, PixelFormat::RGBA8_SRGB).unwrap();
    assert_eq!(
        *to_rgb8_if_opaque(&opaque_rgba.as_view())
            .unwrap_err()
            .error(),
        ConvertError::AlphaLossForbidden
    );
    let facts = scan_facts(&opaque_rgba.as_view());
    opaque_rgba.assume_facts(facts);
    assert_eq!(
        to_rgb8_if_opaque(&opaque_rgba.as_view())
            .unwrap()
            .as_view()
            .row(0),
        &[10, 20, 30]
    );
    assert!(to_gray8_if_opaque(&opaque_rgba.as_view()).is_ok());

    // M7: apply_orientation_owned(rot90_buf, Identity) clears pending orientation to Identity
    // zero-copy, matching borrowed apply_orientation(&view, Identity).
    let rot90 = PixelBuf::from_typed(vec![1u8, 2, 3, 4, 5, 6], 2, 1, PixelFormat::RGB8_SRGB)
        .unwrap()
        .with_orientation(Orientation::Rotate90);
    let borrowed_id = apply_orientation(&rot90.as_view(), Orientation::Identity).unwrap();
    assert_eq!(borrowed_id.orientation(), Orientation::Identity);
    let rot90_for_owned = rot90.clone();
    let ptr_before = rot90_for_owned.as_view().row(0).as_ptr() as usize;
    let owned_id = apply_orientation_owned(rot90_for_owned, Orientation::Identity).unwrap();
    assert_eq!(owned_id.orientation(), Orientation::Identity);
    assert_eq!(
        owned_id.as_view().row(0).as_ptr() as usize,
        ptr_before,
        "apply_orientation_owned(_, Identity) must stay zero-copy while matching Identity metadata"
    );
    let baked = bake_orientation_owned(rot90).unwrap();
    assert_eq!(baked.orientation(), Orientation::Identity);
    assert_eq!((baked.width(), baked.height()), (1, 2));

    // S9: AlphaCompaction::{RetagOpaque, DropUnusedLanes} distinguish opaque retag vs lane drop.
    let mut b_retag = PixelBuf::from_typed(
        vec![10u8, 20, 30, 255, 40, 50, 60, 255],
        2,
        1,
        PixelFormat::RGBA8_SRGB,
    )
    .unwrap();
    let ptr_retag = b_retag.as_view().row(0).as_ptr() as usize;
    assert!(compact_in_place(&mut b_retag, AlphaCompaction::RetagOpaque).unwrap());
    assert_eq!(b_retag.format().layout, SampleLayout::RGBA8);
    assert_eq!(b_retag.format().color.alpha(), AlphaMode::Opaque);
    assert_eq!(b_retag.as_view().row(0).as_ptr() as usize, ptr_retag);

    let mut b_drop = PixelBuf::from_typed(
        vec![10u8, 20, 30, 255, 40, 50, 60, 255],
        2,
        1,
        PixelFormat::RGBA8_SRGB,
    )
    .unwrap();
    let ptr_drop = b_drop.as_view().row(0).as_ptr() as usize;
    assert!(compact_in_place(&mut b_drop, AlphaCompaction::DropUnusedLanes).unwrap());
    assert_eq!(b_drop.format(), &PixelFormat::RGB8_SRGB);
    assert_eq!(b_drop.as_view().row(0), &[10, 20, 30, 40, 50, 60]);
    assert_eq!(b_drop.as_view().row(0).as_ptr() as usize, ptr_drop);
}
