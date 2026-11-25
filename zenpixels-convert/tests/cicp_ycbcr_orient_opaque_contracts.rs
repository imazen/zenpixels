//! Integration tests for all 16 H.273 transfers (inverse accuracy + SCALAR/ALL parity),
//! OutputPlan HLG Display refusal, RGB10A2 Opaque canonicalization, Orientation propagation
//! & zero-copy bake_orientation_buf, lossless H.273 YCbCr matrices, siting-exact Bilinear
//! chroma upsampling, zero-alloc U8/U16/F32/BGR/Gray/PlanarRgb reconstruction, and
//! decoupled GainMapRowExtractor 3-ch + Shared metadata.

use zenpixels_convert::{
    Accelerators, AlphaMode, BandedBuf, ChromaSiting, Cicp, ColorOrigin, ColorSpace,
    ContentLightLevel, ConvertError, ConvertPolicy, DiffuseWhite, GamutPolicy, HdrMetadata,
    HlgMode, HlgRendering, ImagePlan, ImageSpec, Orientation, OutputPlan, OutputProfile,
    PeakSource, PixelBuf, PixelBufLoadBearingExt, PixelFormat, PixelViewLoadBearingExt, PlanarBuf,
    PlaneRole, PlaneSpec, PlaneView, Primaries, RowPlan, SampleLayout, SampleType, SignalRange,
    Transfer, YcbcrMatrix, YcbcrOptions, bake_orientation_buf, ycbcr_to_rgb_f32, ycbcr_to_rgb_into,
    ycbcr_to_rgb_with_options, ycbcr_to_rgb_with_options_into,
};
#[cfg(feature = "gainmap")]
use zenpixels_convert::{
    GainMapApplyConfig, GainMapChannels, GainMapColorSpace, GainMapComponent, GainMapMetadata,
    GainMapOrientation, GainMapRowExtractor, SignedFraction, UnsignedFraction,
    apply_gain_map_oriented,
};

#[test]
fn all_16_h273_transfers_inverse_accuracy_and_simd_parity() {
    let linear_spec = ImageSpec::RGBF32_LINEAR.with_white(DiffuseWhite::BT2408);
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
        let enc_spec = ImageSpec::new(
            SampleLayout::RGBF32,
            ColorSpace::Rgb {
                primaries: prim,
                transfer: tf,
                range: SignalRange::Full,
                alpha: AlphaMode::None,
                white: Some(DiffuseWhite::BT2408),
            },
        );
        let src_buf = PixelBuf::from_vec(ramp.clone(), 65, 1, enc_spec.clone()).unwrap();

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
        let lin_scalar = ImagePlan::build(
            &enc_spec,
            &lin_for_tc,
            pol_scalar.clone(),
            PeakSource::Measured,
        )
        .unwrap()
        .execute(&src_buf)
        .unwrap();
        let lin_all = ImagePlan::build(
            &enc_spec,
            &lin_for_tc,
            pol_all.clone(),
            PeakSource::Measured,
        )
        .unwrap()
        .execute(&src_buf)
        .unwrap();
        assert_eq!(
            lin_scalar.as_view().row(0),
            lin_all.as_view().row(0),
            "TC={tc} ({tf:?}) EOTF SCALAR vs ALL 0-bit parity"
        );

        // OETF: linear -> encoded
        let rt_scalar = ImagePlan::build(&lin_for_tc, &enc_spec, pol_scalar, PeakSource::Measured)
            .unwrap()
            .execute(&lin_scalar)
            .unwrap();
        let rt_all = ImagePlan::build(&lin_for_tc, &enc_spec, pol_all, PeakSource::Measured)
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
        let enc = ImageSpec::RGBF32_LINEAR.with_transfer(tf);
        let neg = PixelBuf::from_vec(vec![-0.25_f32, -0.05, 0.5], 1, 1, enc.clone()).unwrap();
        let pol = ConvertPolicy::permissive().with_gamut(GamutPolicy::PreserveExtended);
        let lin = ImagePlan::build(
            &enc,
            &ImageSpec::RGBF32_LINEAR,
            pol.clone(),
            PeakSource::Measured,
        )
        .unwrap()
        .execute(&neg)
        .unwrap();
        let lin_f: &[f32] = lin.as_view().row_as(0).unwrap();
        assert!(lin_f[0] < 0.0 && lin_f[1] < 0.0 && lin_f[2] > 0.0);
        let back = ImagePlan::build(&ImageSpec::RGBF32_LINEAR, &enc, pol, PeakSource::Measured)
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
    let hlg_display = ImageSpec::RGB16_BT2100_HLG_1000;
    let hlg_scene = ImageSpec::RGB16_BT2100_HLG_SCENE;
    let pol = ConvertPolicy::permissive();

    assert_eq!(
        *OutputPlan::build(
            &hlg_display,
            &hlg_display,
            pol.clone(),
            PeakSource::Measured,
            OutputProfile::CicpOnly,
        )
        .unwrap_err()
        .error(),
        ConvertError::OutputSignalingUnavailable
    );
    assert_eq!(
        *OutputPlan::build(
            &hlg_display,
            &hlg_display,
            pol.clone(),
            PeakSource::Measured,
            OutputProfile::Auto,
        )
        .unwrap_err()
        .error(),
        ConvertError::OutputSignalingUnavailable
    );
    assert!(
        OutputPlan::build(
            &hlg_scene,
            &hlg_scene,
            pol.clone(),
            PeakSource::Measured,
            OutputProfile::CicpOnly,
        )
        .is_ok()
    );
    assert!(
        OutputPlan::build(
            &hlg_scene,
            &hlg_scene,
            pol.clone(),
            PeakSource::Measured,
            OutputProfile::Auto,
        )
        .is_ok()
    );
    // Also verify HlgMode::Display variant explicitly.
    let custom_disp = ImageSpec::RGB16_BT2100_HLG_SCENE.with_transfer(Transfer::Hlg(
        HlgMode::Display(HlgRendering::REFERENCE_1000_NIT),
    ));
    assert_eq!(
        *OutputPlan::build(
            &custom_disp,
            &custom_disp,
            pol,
            PeakSource::Measured,
            OutputProfile::Auto,
        )
        .unwrap_err()
        .error(),
        ConvertError::OutputSignalingUnavailable
    );
}

#[test]
fn rgb10a2_unverified_opaque_canonicalization_and_orientation_propagation() {
    // 1. RGB10A2 with AlphaMode::Opaque and poisoned alpha bits (0 instead of 3) must canonicalize!
    let spec = ImageSpec::new(
        SampleLayout::RGB10A2,
        ColorSpace::SRGB.with_alpha(AlphaMode::Opaque),
    );
    let r = 511u32;
    let g = 256u32;
    let b = 1023u32;
    let poisoned_word: u32 = r | (g << 10) | (b << 20); // alpha = 0!
    let src = PixelBuf::from_raw_pod(vec![poisoned_word; 4], 4, 1, spec.clone())
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
        ConvertPolicy::permissive(),
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
    let mut dst_band = BandedBuf::new(4, 1, spec.clone(), 64).unwrap();
    ImagePlan::build(
        &spec,
        &spec,
        ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
    .unwrap()
    .execute_into(&src, &mut dst_band)
    .unwrap();
    assert_eq!(dst_band.orientation(), Orientation::Rotate90);

    let mut rgba_gray = PixelBuf::from_vec(
        vec![100u8, 100, 100, 255, 50, 50, 50, 255],
        2,
        1,
        PixelFormat::RGBA8_SRGB,
    )
    .unwrap()
    .with_orientation(Orientation::Rotate270);
    let reduced = rgba_gray
        .as_view()
        .reduce_to_load_bearing_format()
        .unwrap()
        .unwrap();
    assert_eq!(reduced.spec().layout, SampleLayout::GRAY8);
    assert_eq!(reduced.orientation(), Orientation::Rotate270);
    rgba_gray.reduce_to_load_bearing_format_in_place(true);
    assert_eq!(rgba_gray.orientation(), Orientation::Rotate270);

    // 3. `bake_orientation_buf` is zero-copy on Identity and bakes + resets to Identity otherwise.
    let id_buf =
        PixelBuf::from_vec(vec![1u8, 2, 3, 4, 5, 6], 2, 1, PixelFormat::RGB8_SRGB).unwrap();
    let orig_ptr = id_buf.as_view().as_strided_bytes().as_ptr();
    let baked_id = bake_orientation_buf(id_buf).unwrap();
    assert_eq!(baked_id.as_view().as_strided_bytes().as_ptr(), orig_ptr);
    assert_eq!(baked_id.orientation(), Orientation::Identity);

    let rot_buf = PixelBuf::from_vec(vec![10u8, 20, 30, 40, 50, 60], 2, 1, PixelFormat::RGB8_SRGB)
        .unwrap()
        .with_orientation(Orientation::Rotate90);
    let baked_rot = bake_orientation_buf(rot_buf).unwrap();
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
        PlaneSpec::new(PlaneRole::Y, SampleType::U8),
        1,
        1,
        1,
        &y_data,
    )
    .unwrap();
    let pcg = PlaneView::from_samples(
        PlaneSpec::new(PlaneRole::Cb, SampleType::U8),
        1,
        1,
        1,
        &cg_data,
    )
    .unwrap();
    let pco = PlaneView::from_samples(
        PlaneSpec::new(PlaneRole::Cr, SampleType::U8),
        1,
        1,
        1,
        &co_data,
    )
    .unwrap();
    let ycgco_view = zenpixels_convert::PlanarView::new(1, 1, &[py, pcg, pco])
        .unwrap()
        .with_color(ColorSpace::SRGB)
        .with_ycbcr_matrix_code(8);
    let ycgco_rgb =
        ycbcr_to_rgb_with_options(&ycgco_view, SampleLayout::RGB8, &YcbcrOptions::nearest())
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
        .with_ycbcr_matrix_code(12);
    let cd_f32 = ycbcr_to_rgb_f32(
        &cd_view,
        YcbcrMatrix::ChromaDerivedNcl,
        SignalRange::Full,
        Primaries::DisplayP3,
        Transfer::Srgb,
    )
    .unwrap();
    assert_eq!(cd_f32.spec().layout, SampleLayout::RGBF32);

    // 4. Siting-exact Bilinear 4:2:0 upsampling (Center = 9:3:3:1, TopLeft, odd crop_origin, Unknown refused).
    let y_4x4 = [128u8; 16];
    // 2x2 chroma grid: top-left=0, top-right=255, bottom-left=0, bottom-right=255 (horizontal gradient)
    let cb_2x2 = [0u8, 255, 0, 255];
    let cr_2x2 = [128u8; 4];
    let y_p = PlaneView::from_samples(
        PlaneSpec::new(PlaneRole::Y, SampleType::U8),
        4,
        4,
        4,
        &y_4x4,
    )
    .unwrap();
    let cb_p = PlaneView::from_samples(
        PlaneSpec::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2),
        2,
        2,
        2,
        &cb_2x2,
    )
    .unwrap();
    let cr_p = PlaneView::from_samples(
        PlaneSpec::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2),
        2,
        2,
        2,
        &cr_2x2,
    )
    .unwrap();

    // Unknown siting + Bilinear without fallback must fail!
    let unk_420 = zenpixels_convert::PlanarView::new(4, 4, &[y_p, cb_p, cr_p])
        .unwrap()
        .with_color(ColorSpace::SRGB)
        .with_ycbcr_matrix_code(1);
    assert_eq!(
        *ycbcr_to_rgb_with_options(&unk_420, SampleLayout::RGBF32, &YcbcrOptions::bilinear())
            .unwrap_err()
            .error(),
        ConvertError::UnsupportedRoute
    );

    // With Center siting: x=0 -> c[0]=0, x=1 -> 0.75*0 + 0.25*255, x=2 -> 0.25*0 + 0.75*255, x=3 -> c[1]=255!
    let hdr = HdrMetadata {
        cll: Some(ContentLightLevel::new(500, 100)),
        mastering: None,
    };
    let origin = ColorOrigin::from_cicp(Cicp::SRGB);
    let center_420 = unk_420
        .clone()
        .with_chroma_siting(ChromaSiting::Center)
        .with_hdr(hdr)
        .with_origin(&origin)
        .with_orientation(Orientation::Rotate90);
    let full_bi =
        ycbcr_to_rgb_with_options(&center_420, SampleLayout::RGBF32, &YcbcrOptions::bilinear())
            .unwrap();
    assert_eq!(full_bi.hdr(), hdr);
    assert_eq!(full_bi.origin(), &origin);
    assert_eq!(full_bi.orientation(), Orientation::Rotate90);

    let row0: &[f32] = full_bi.as_view().row_as(0).unwrap();
    // Blue channel increases monotonically across x = 0, 1, 2, 3 (strictly distinct intermediate taps!).
    assert!(row0[2] < row0[5] && row0[5] < row0[8] && row0[8] < row0[11]);

    // Odd crop origin (1, 1) bilinear MUST match full-frame bilinear cropped at (1, 1)!
    let odd_crop = center_420.crop(1, 1, 2, 2).unwrap();
    let crop_bi =
        ycbcr_to_rgb_with_options(&odd_crop, SampleLayout::RGBF32, &YcbcrOptions::bilinear())
            .unwrap();
    for y in 0..2 {
        assert_eq!(
            crop_bi.as_view().row(y),
            full_bi.as_view().crop(1, 1, 2, 2).unwrap().row(y),
            "odd-origin 4:2:0 bilinear row {y} must match full-frame bilinear + crop"
        );
    }

    // TopLeft siting differs from Center siting at x=1 (0.5/0.5 vs 0.75/0.25)!
    let tl_420 = unk_420.clone().with_chroma_siting(ChromaSiting::TopLeft);
    let tl_bi = ycbcr_to_rgb_with_options(&tl_420, SampleLayout::RGBF32, &YcbcrOptions::bilinear())
        .unwrap();
    let tl_r0: &[f32] = tl_bi.as_view().row_as(0).unwrap();
    assert!(
        (tl_r0[5] - row0[5]).abs() > 0.1,
        "TopLeft vs Center siting must produce distinct x=1"
    );

    // 5. Zero-alloc `ycbcr_to_rgb_into` & `ycbcr_to_rgb_with_options_into` into BGR8, BGRA8, RGB16, GRAY8!
    let mut bgr8_buf = PixelBuf::new(4, 4, PixelFormat::BGR8_SRGB).unwrap();
    ycbcr_to_rgb_with_options_into(
        &center_420,
        YcbcrMatrix::Bt709,
        SignalRange::Full,
        &YcbcrOptions::bilinear(),
        &mut bgr8_buf.as_view_mut(),
    )
    .unwrap();
    let mut rgb8_buf = PixelBuf::new(4, 4, PixelFormat::RGB8_SRGB).unwrap();
    ycbcr_to_rgb_with_options_into(
        &center_420,
        YcbcrMatrix::Bt709,
        SignalRange::Full,
        &YcbcrOptions::bilinear(),
        &mut rgb8_buf.as_view_mut(),
    )
    .unwrap();
    // BGR8 matches RGB8 with R<->B swapped!
    for x in 0..4 {
        let bgr = &bgr8_buf.as_view().row(0)[x * 3..x * 3 + 3];
        let rgb = &rgb8_buf.as_view().row(0)[x * 3..x * 3 + 3];
        assert_eq!(bgr, &[rgb[2], rgb[1], rgb[0]]);
    }

    let rgb16_buf =
        ycbcr_to_rgb_with_options(&center_420, SampleLayout::RGB16, &YcbcrOptions::bilinear())
            .unwrap();
    assert_eq!(rgb16_buf.spec().layout, SampleLayout::RGB16);

    // Monochrome Y -> GRAY8 zero-alloc!
    let mono_view = zenpixels_convert::PlanarView::new(4, 4, &[y_p])
        .unwrap()
        .with_color(ColorSpace::GRAY_SRGB);
    let mut gray8 = PixelBuf::new(4, 4, PixelFormat::GRAY8_SRGB).unwrap();
    ycbcr_to_rgb_into(
        &mono_view,
        YcbcrMatrix::Bt709,
        SignalRange::Full,
        &mut gray8.as_view_mut(),
    )
    .unwrap();
    assert_eq!(gray8.as_view().row(0), &[128, 128, 128, 128]);

    // 6. Planar RGB (F32 R, G, B + Alpha) -> RGBAF32 and RGBA8!
    let p_rgb = PlanarBuf::new(
        2,
        1,
        &[
            PlaneSpec::new(PlaneRole::R, SampleType::F32),
            PlaneSpec::new(PlaneRole::G, SampleType::F32),
            PlaneSpec::new(PlaneRole::B, SampleType::F32),
            PlaneSpec::new(PlaneRole::Alpha, SampleType::F32),
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

    let out_f32 = ycbcr_to_rgb_with_options(
        &p_rgb.as_view(),
        SampleLayout::RGBAF32,
        &YcbcrOptions::nearest(),
    )
    .unwrap();
    assert_eq!(
        out_f32.as_view().row_as::<f32>(0).unwrap(),
        &[1.0, 0.5, 0.0, 1.0, 0.25, 0.75, 1.0, 0.5]
    );
    let out_bgra8 = ycbcr_to_rgb_with_options(
        &p_rgb.as_view(),
        SampleLayout::BGRA8,
        &YcbcrOptions::nearest(),
    )
    .unwrap();
    assert_eq!(
        out_bgra8.as_view().row(0),
        &[0, 128, 255, 255, 255, 191, 64, 128]
    );
    // Dropping alpha from planar RGB+A into RGB8 must refuse with AlphaLossForbidden!
    assert_eq!(
        *ycbcr_to_rgb_with_options(
            &p_rgb.as_view(),
            SampleLayout::RGB8,
            &YcbcrOptions::nearest()
        )
        .unwrap_err()
        .error(),
        ConvertError::AlphaLossForbidden
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
    let shared_meta = GainMapMetadata::try_new(
        GainMapChannels::Shared(comp),
        UnsignedFraction::try_new(0, 1).unwrap(),
        UnsignedFraction::try_new(2, 1).unwrap(),
        GainMapColorSpace::Base,
    )
    .unwrap();
    let per_ch_meta = GainMapMetadata::try_new(
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
        .encode_row_linear_f32(&base, &alt, 3, SampleLayout::RGB8, &mut dst_3ch)
        .unwrap();
    // Pixel 0: R boost ~4x (2 stops -> ~255), G boost ~2x (1 stop -> ~128), B boost 1x (0 stops -> 0)
    assert!(dst_3ch[0] > 240 && dst_3ch[1] > 100 && dst_3ch[1] < 160 && dst_3ch[2] < 10);

    // 1-channel gain map with 3-record `PerChannel` metadata must be refused!
    let mut ext_per = GainMapRowExtractor::new(per_ch_meta, Primaries::Bt709).unwrap();
    let mut dst_1ch = [0u8; 2];
    assert_eq!(
        *ext_per
            .encode_row_linear_f32(&base, &alt, 3, SampleLayout::GRAY8, &mut dst_1ch)
            .unwrap_err()
            .error(),
        ConvertError::UnsupportedRoute
    );

    // `apply_gain_map_oriented` in `Passthrough` attaches `orient` onto the returned `PixelBuf`!
    let gm_spec = ImageSpec::new(
        SampleLayout::GRAY8,
        ColorSpace::Untagged {
            channels: 1,
            alpha: AlphaMode::None,
        },
    );
    let base_buf = PixelBuf::new_white(4, 2, PixelFormat::RGB8_SRGB).unwrap();
    let gm_buf = PixelBuf::new(4, 2, gm_spec).unwrap();
    let dual = zenpixels_convert::DualLayerHdr::new_with_alternate(
        base_buf,
        gm_buf,
        shared_meta,
        None,
        Orientation::Rotate90,
    )
    .unwrap();
    let cfg = GainMapApplyConfig::new(2.0).with_orientation(GainMapOrientation::Passthrough);
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
        ChannelOrder, Cicp, CicpTag, CustomPrimaries, PixelViewLoadBearingExt,
        validate_icc_structure,
    };

    // 1. Build ACEScg (D60) and custom D65 ICC v4.4 profiles carrying 44-byte cicp tags.
    let acescg_icc = build_matrix_trc_icc_with_custom_cicp(CustomPrimaries::ACES_CG, 8);
    validate_icc_structure(&acescg_icc).unwrap();
    let acescg_tag = CicpTag::from_icc(&acescg_icc).unwrap();
    assert_eq!(acescg_tag.cicp(), Cicp::new(2, 8, 0, true));
    assert!(acescg_tag.custom_primaries().is_some());

    let custom_d65 =
        CustomPrimaries::try_new((0.65, 0.33), (0.28, 0.62), (0.15, 0.06), (0.3127, 0.3290))
            .unwrap();
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
    let d65_spec = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::from(d65_icc.clone()),
            channels: 3,
            alpha: AlphaMode::None,
            intent: zenpixels_convert::RenderingIntent::RelativeColorimetric,
        },
    );
    let d65_gray_buf = PixelBuf::from_vec(vec![128u8; 4 * 3], 2, 2, d65_spec).unwrap();
    let reduced_d65 = d65_gray_buf
        .as_view()
        .reduce_to_load_bearing_format()
        .unwrap()
        .expect("custom D65 neutral ICC should collapse to Gray");
    assert_eq!(reduced_d65.spec().layout.order, ChannelOrder::Gray);

    let acescg_spec_u8 = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::from(acescg_icc.clone()),
            channels: 3,
            alpha: AlphaMode::None,
            intent: zenpixels_convert::RenderingIntent::RelativeColorimetric,
        },
    );
    let acescg_gray_buf = PixelBuf::from_vec(vec![128u8; 4 * 3], 2, 2, acescg_spec_u8).unwrap();
    assert!(
        acescg_gray_buf
            .as_view()
            .reduce_to_load_bearing_format()
            .unwrap()
            .is_none(),
        "ACEScg (D60) neutral RGB ICC must NOT collapse to D65 Gray"
    );

    // 3. CMS interop (MoxCms & Lcms): both parse the 44-byte cicp ICC profile cleanly AND match
    //    native `Primaries::Custom(ACES_CG)` -> linear sRGB!
    let native_acescg_spec = ImageSpec::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: Primaries::Custom(CustomPrimaries::ACES_CG),
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let icc_acescg_spec = ImageSpec::new(
        SampleLayout::RGBF32,
        ColorSpace::Icc {
            profile: Arc::from(acescg_icc),
            channels: 3,
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
        PixelBuf::from_vec(sample_pixels.clone(), 3, 1, native_acescg_spec.clone()).unwrap();
    let icc_src = PixelBuf::from_vec(sample_pixels, 3, 1, icc_acescg_spec).unwrap();

    let native_out = ImagePlan::build(
        &native_acescg_spec,
        &ImageSpec::RGBF32_LINEAR,
        ConvertPolicy::permissive(),
        PeakSource::default(),
    )
    .unwrap()
    .execute(&native_src)
    .unwrap();
    let native_row: &[f32] = native_out.as_view().row_as(0).unwrap();

    #[cfg(feature = "cms-moxcms")]
    {
        let mox_pol = ConvertPolicy::permissive().with_cms(Arc::new(zenpixels_convert::MoxCms));
        let mox_out = ImagePlan::build(
            icc_src.spec(),
            &ImageSpec::RGBF32_LINEAR,
            mox_pol,
            PeakSource::default(),
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
            icc_src.spec(),
            &ImageSpec::RGBF32_LINEAR,
            lcms_pol.clone(),
            PeakSource::default(),
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
            icc_src.spec(),
            &native_acescg_spec,
            lcms_pol,
            PeakSource::default(),
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
        let gm_cfg = zenpixels_convert::GainMapExtractConfig::default()
            .with_scale_factor(1)
            .with_policy(ConvertPolicy::permissive().with_cms(Arc::new(zenpixels_convert::Lcms)));
        let base_sdr =
            PixelBuf::from_vec(vec![0.5_f32; 4 * 3], 2, 2, icc_src.spec().clone()).unwrap();
        let alt_hdr = PixelBuf::from_vec(vec![1.0_f32; 4 * 3], 2, 2, native_acescg_spec).unwrap();
        let (gm_buf, _meta) = zenpixels_convert::extract_gain_map_only(
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
