//! Strict preservation, exact code-grid, load-bearing scan/reduction, signaling authority,
//! unknown-primaries luma, narrow-range bit-shift, float-facts, and CodeMap regression suite.
//!
//! Directly gates audit findings:
//! - B01 (lossless transfer conversion refusal)
//! - B02 (non-nesting LSB3->U8 refusal, 2/4->8 acceptance, U32 25-bit shift 0->1 exactness)
//! - B03 (non-opaque Straight<->Premul lossless refusal + opaque positive)
//! - B04 (cross-profile ICC lossless refusal + D01 same-profile opaque ICC without CMS)
//! - B05 (Negotiate feasibility parity)
//! - B06 / B07 / B08 (scan_facts + ImagePlan swizzle opaque reuse + CompactionReport in-place)
//! - B09 (Unknown primaries luma refusal vs explicit BT.709 / P3 red)
//! - B10 / B11 / B12 / B13 (OutputSignaling strict ICC validation, Gray Auto ICC, SameAsOrigin dual P3 & untagged, Gray CP=2/TC=8)
//! - B33 (narrow 12b->10b & U8->U16 BT.2100 Table 9)
//! - B34 (PQ->Linear Clip RGB vs BGR parity + no false AT_MOST_ONE / NO_SUBNORMALS)
//! - B42 (borrowed vs owned OutputPlan actual transformed pixel values + execute_view rejection)
//! - A02 (Adobe 563/256 vs Gamma22 2.2 at 0.5 + DCI Cinema TC17 at 0.5 & 1.0)
//! - D02 / D03 (SIMD scan predicates, CompactionReport, CodeMap & to_canonical)

extern crate alloc;
use alloc::sync::Arc;

use zenpixels_convert::{
    AlphaCompaction, AlphaMode, AlphaPolicy, BitPacking, ChannelOrder, Cicp, CodeMap, CodePlane,
    CodeRole, ColorOrigin, ColorSpace, CompactionReport, ContentLightLevel, ConvertError,
    ConvertPolicy, DepthPolicy, DiffuseWhite, FloatFacts, Gamma, HdrMetadata, ImagePlan, ImageSpec,
    LossKind, MasteringDisplay, Negotiate, Nits, OutputPlan, OutputSignaling, PeakSource, PixelBuf,
    Primaries, RangePolicy, RenderingIntent, RowPlan, SampleEncoding, SampleFacts, SampleLayout,
    SampleType, SignalRange, Transfer, analyze_compaction, compact_in_place, compact_to_buf,
    convert_to_buf_with_peak, scan_facts, to_canonical,
};

#[test]
fn b01_b02_b03_strict_lossless_refuses_non_injective_transfers_grids_and_reassociation() {
    // B01: sRGB8 [1,2,3] -> Linear8 [0,0,0] and Adobe-gamma -> sRGB8 MUST be refused under lossless()!
    let srgb8 = ImageSpec::RGB8_SRGB;
    let lin8 = ImageSpec::RGB8_SRGB.with_transfer(Transfer::Linear);
    let adobe_bt709 = ImageSpec::RGB8_SRGB.with_transfer(Transfer::AdobeRgb);
    let err_lin = RowPlan::build(
        srgb8.clone(),
        lin8.clone(),
        &ConvertPolicy::lossless(),
        None,
    )
    .unwrap_err();
    assert_eq!(*err_lin.error(), ConvertError::DepthReductionForbidden);
    let err_adobe = RowPlan::build(
        adobe_bt709.clone(),
        srgb8.clone(),
        &ConvertPolicy::lossless(),
        None,
    )
    .unwrap_err();
    assert_eq!(*err_adobe.error(), ConvertError::DepthReductionForbidden);

    // Negotiation MUST NOT report LossKind::None for sRGB8 -> Linear8!
    assert!(
        Negotiate::select(
            &srgb8,
            core::slice::from_ref(&lin8),
            &ConvertPolicy::lossless()
        )
        .is_none(),
        "Negotiate::select under lossless() must reject sRGB8 -> Linear8"
    );
    let perm_neg = Negotiate::select(
        &srgb8,
        core::slice::from_ref(&lin8),
        &ConvertPolicy::permissive(),
    )
    .unwrap();
    assert_ne!(perm_neg.cost.loss_rank, 0);
    let _ = LossKind::None;

    // Permissive witness actually transforms sRGB8 [1,2,3] -> Linear8 [0,0,0] (proving collision!).
    let perm_plan =
        RowPlan::build(srgb8.clone(), lin8, &ConvertPolicy::permissive(), None).unwrap();
    let mut out = [99u8; 3];
    perm_plan.convert_row(&[1, 2, 3], &mut out, 1, 0).unwrap();
    assert_eq!(out, [0, 0, 0]);

    // Same-transfer swizzle RGB8 -> BGR8 IS lossless!
    let bgr8 = ImageSpec::new(SampleLayout::BGR8, ColorSpace::SRGB);
    let swiz = RowPlan::build(srgb8.clone(), bgr8, &ConvertPolicy::lossless(), None).unwrap();
    let mut swiz_out = [0u8; 3];
    swiz.convert_row(&[1, 2, 3], &mut swiz_out, 1, 0).unwrap();
    assert_eq!(swiz_out, [3, 2, 1]);

    // B02: Full-range LSB3 -> U8 MUST be refused under lossless() (255 % 7 != 0),
    // while LSB4 -> U8 (255 % 15 == 0) succeeds exact!
    let lsb3 = ImageSpec::new(
        SampleLayout::new(SampleType::U8, ChannelOrder::Rgb)
            .with_packing(BitPacking::Lsb { bits: 3 }),
        ColorSpace::SRGB,
    );
    let lsb4 = ImageSpec::new(
        SampleLayout::new(SampleType::U8, ChannelOrder::Rgb)
            .with_packing(BitPacking::Lsb { bits: 4 }),
        ColorSpace::SRGB,
    );
    assert_eq!(
        *RowPlan::build(lsb3, srgb8.clone(), &ConvertPolicy::lossless(), None)
            .unwrap_err()
            .error(),
        ConvertError::DepthReductionForbidden
    );
    let p_lsb4 = RowPlan::build(lsb4, srgb8.clone(), &ConvertPolicy::lossless(), None).unwrap();
    let mut o4 = [0u8; 3];
    p_lsb4.convert_row(&[0, 1, 15], &mut o4, 1, 0).unwrap();
    assert_eq!(o4, [0, 17, 255]); // 1 * (255 / 15) = 17 exact!

    // B02: Gray U32 25-bit field shift 0 -> shift 1 preserves code 16_777_217 bit-exact!
    let g25_s0 = ImageSpec::new(
        SampleLayout::new(SampleType::U32, ChannelOrder::Gray).with_packing(
            BitPacking::Bitfields {
                bits: [25, 0, 0, 0],
                shifts: [0, 0, 0, 0],
            },
        ),
        ColorSpace::GRAY_SRGB,
    );
    let g25_s1 = ImageSpec::new(
        SampleLayout::new(SampleType::U32, ChannelOrder::Gray).with_packing(
            BitPacking::Bitfields {
                bits: [25, 0, 0, 0],
                shifts: [1, 0, 0, 0],
            },
        ),
        ColorSpace::GRAY_SRGB,
    );
    let p25 = RowPlan::build(g25_s0, g25_s1, &ConvertPolicy::lossless(), None).unwrap();
    let src_word: u32 = 16_777_217; // > 2^24 (would lose LSB if rounded through f32!)
    let mut dst_word: u32 = 0;
    p25.convert_row(
        &src_word.to_ne_bytes(),
        bytemuck::bytes_of_mut(&mut dst_word),
        1,
        0,
    )
    .unwrap();
    assert_eq!(dst_word >> 1, 16_777_217);

    // B03: Straight <-> Premultiplied on non-opaque RGBA8 MUST be refused under lossless(),
    // while proven-opaque RGBA8 Straight -> Premul succeeds losslessly!
    let straight = ImageSpec::RGBA8_SRGB;
    let premul = ImageSpec::RGBA8_SRGB.with_alpha(AlphaMode::Premultiplied);
    assert_eq!(
        *RowPlan::build(
            straight.clone(),
            premul.clone(),
            &ConvertPolicy::lossless(),
            None
        )
        .unwrap_err()
        .error(),
        ConvertError::DepthReductionForbidden
    );
    assert_eq!(
        *RowPlan::build(
            premul.clone(),
            straight.clone(),
            &ConvertPolicy::lossless(),
            None
        )
        .unwrap_err()
        .error(),
        ConvertError::DepthReductionForbidden
    );
    assert!(
        Negotiate::select(
            &straight,
            core::slice::from_ref(&premul),
            &ConvertPolicy::lossless()
        )
        .is_none()
    );
    let opaque_pol =
        ConvertPolicy::lossless().with_source_facts(SampleFacts::NONE.with_all_opaque(true));
    let op_plan = RowPlan::build(straight, premul, &opaque_pol, None).unwrap();
    let mut op_out = [0u8; 4];
    op_plan
        .convert_row(&[10, 20, 30, 255], &mut op_out, 1, 0)
        .unwrap();
    assert_eq!(op_out, [10, 20, 30, 255]);
}

#[test]
fn b04_d01_opaque_same_profile_icc_without_cms_and_cross_profile_lossless_refusal() {
    // D01: Same-profile opaque ICC (unrecognized bytes!) allows exact layout swizzle,
    // alpha insertion, and U8 -> U16 widening WITHOUT needing a CMS!
    let mut opaque_icc = vec![0u8; 256];
    opaque_icc[0..4].copy_from_slice(&256u32.to_be_bytes());
    opaque_icc[12..16].copy_from_slice(b"mntr");
    opaque_icc[16..20].copy_from_slice(b"RGB ");
    opaque_icc[20..24].copy_from_slice(b"XYZ ");
    opaque_icc[36..40].copy_from_slice(b"acsp");
    let icc_arc: Arc<[u8]> = Arc::from(opaque_icc.as_slice());

    let src_rgb8 = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: icc_arc.clone(),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::Perceptual,
        },
    );
    let dst_rgba8 = ImageSpec::new(
        SampleLayout::RGBA8,
        ColorSpace::Icc {
            profile: icc_arc.clone(),
            channels: 3,
            alpha: AlphaMode::Opaque,
            intent: RenderingIntent::Perceptual,
        },
    );
    let dst_rgb16 = ImageSpec::new(
        SampleLayout::RGB16,
        ColorSpace::Icc {
            profile: icc_arc,
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::Perceptual,
        },
    );
    let p_add_a = RowPlan::build(
        src_rgb8.clone(),
        dst_rgba8,
        &ConvertPolicy::lossless(),
        None,
    )
    .unwrap();
    let mut o_rgba = [0u8; 4];
    p_add_a
        .convert_row(&[11, 22, 33], &mut o_rgba, 1, 0)
        .unwrap();
    assert_eq!(o_rgba, [11, 22, 33, 255]);

    let p_widen = RowPlan::build(src_rgb8, dst_rgb16, &ConvertPolicy::lossless(), None).unwrap();
    let mut o_u16 = [0u16; 3];
    p_widen
        .convert_row(&[11, 22, 255], bytemuck::cast_slice_mut(&mut o_u16), 1, 0)
        .unwrap();
    assert_eq!(o_u16, [0x0B0B, 0x1616, 0xFFFF]);

    // B04: Cross-profile Display P3 ICC -> sRGB8 MUST be refused under lossless() with CMS!
    #[cfg(feature = "cms-moxcms")]
    {
        let p3_spec = ImageSpec::new(
            SampleLayout::RGB8,
            ColorSpace::Icc {
                profile: Arc::from(zenpixels_convert::icc_profiles::DISPLAY_P3_V4),
                channels: 3,
                alpha: AlphaMode::None,
                intent: RenderingIntent::RelativeColorimetric,
            },
        );
        let pol = ConvertPolicy::lossless()
            .with_icc_substitution(false)
            .with_cms(Arc::new(zenpixels_convert::MoxCms));
        assert_eq!(
            *RowPlan::build(p3_spec, ImageSpec::RGB8_SRGB, &pol, None)
                .unwrap_err()
                .error(),
            ConvertError::DepthReductionForbidden
        );
    }
}

#[test]
fn b06_b07_b08_d02_scan_facts_image_plan_reuse_and_load_bearing_reduction() {
    // B08: Cached ImagePlan RGBA8 -> BGRA8 swizzle preserves proven-opaque facts on image #1,
    // and MUST NOT lie `all_opaque=true` when reused on image #2 containing alpha=254!
    let plan = ImagePlan::build(
        ImageSpec::RGBA8_SRGB,
        ImageSpec::BGRA8_SRGB,
        &ConvertPolicy::lossless(),
        PeakSource::Measured,
    )
    .unwrap();

    let mut img1 = PixelBuf::from_typed(
        vec![1u8, 2, 3, 255, 4, 5, 6, 255],
        2,
        1,
        ImageSpec::RGBA8_SRGB,
    )
    .unwrap();
    let f1 = scan_facts(&img1.as_view());
    assert!(f1.all_opaque && f1.binary_alpha && !f1.all_grayscale);
    img1.assume_facts(f1);

    let out1 = plan.execute(&img1).unwrap();
    assert_eq!(out1.as_view().row(0), &[3, 2, 1, 255, 6, 5, 4, 255]);
    assert!(
        out1.facts().all_opaque,
        "ImagePlan swizzle must propagate proven all_opaque!"
    );

    let img2 = PixelBuf::from_typed(
        vec![1u8, 2, 3, 254, 4, 5, 6, 255],
        2,
        1,
        ImageSpec::RGBA8_SRGB,
    )
    .unwrap();
    let out2 = plan.execute(&img2).unwrap();
    assert_eq!(out2.as_view().row(0), &[3, 2, 1, 254, 6, 5, 4, 255]);
    assert!(
        !out2.facts().all_opaque,
        "reused ImagePlan must NOT falsely assert all_opaque on alpha=254!"
    );

    // B07: Poisoned `AlphaMode::Opaque` buffer with stored alpha=0 MUST scan as `all_opaque=false`!
    let poisoned = PixelBuf::from_typed(
        vec![10u8, 20, 30, 0],
        1,
        1,
        ImageSpec::RGBA8_SRGB.with_alpha(AlphaMode::Opaque),
    )
    .unwrap();
    let pf = scan_facts(&poisoned.as_view());
    assert!(
        !pf.all_opaque,
        "scan_facts must inspect stored alpha even on AlphaMode::Opaque"
    );

    // D02: CompactionReport + zero-alloc in-place compaction on RGBA16 -> Gray8!
    let mut buf16 = PixelBuf::from_typed(
        vec![
            0x4040u16, 0x4040, 0x4040, 0xFFFF, 0x8080, 0x8080, 0x8080, 0xFFFF, 0xC0C0, 0xC0C0,
            0xC0C0, 0xFFFF,
        ],
        3,
        1,
        ImageSpec::RGBA16_SRGB,
    )
    .unwrap();
    let report: CompactionReport = analyze_compaction(&buf16.as_view());
    assert_eq!(report.requires_alpha, Some(false));
    assert_eq!(report.requires_chroma, Some(false));
    assert_eq!(report.requires_u16_precision, Some(false));
    assert_eq!(report.candidate_spec(buf16.spec()), ImageSpec::GRAY8_SRGB);

    let ptr_before = buf16.as_view().row(0).as_ptr() as usize;
    compact_in_place(&mut buf16, AlphaCompaction::DropUnusedLanes);
    assert_eq!(buf16.spec(), &ImageSpec::GRAY8_SRGB);
    assert_eq!(buf16.as_view().row(0), &[0x40, 0x80, 0xC0]);
    assert_eq!(
        buf16.as_view().row(0).as_ptr() as usize,
        ptr_before,
        "in-place load-bearing reduction must reuse allocation without reallocating!"
    );

    // Narrow-range U16 MUST NOT be reduced to U8 via byte-replication!
    let narrow16 = PixelBuf::from_typed(
        vec![0x1010u16, 0x8080, 0xEBEB],
        1,
        1,
        ImageSpec::RGB16_SRGB.with_range(SignalRange::Narrow),
    )
    .unwrap();
    let nr = analyze_compaction(&narrow16.as_view());
    assert_eq!(nr.requires_u16_precision, None);
    assert_eq!(nr.candidate_spec(narrow16.spec()), *narrow16.spec());
}

#[test]
fn b09_unknown_primaries_luma_refusal_and_explicit_primaries() {
    // B09: Unknown primaries RGB -> Gray MUST refuse instead of silently assuming BT.709!
    let unknown_rgb = ImageSpec::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: Primaries::Unknown,
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let gray_lin = ImageSpec::GRAYF32_LINEAR;
    assert_eq!(
        *RowPlan::build(
            unknown_rgb.clone(),
            gray_lin.clone(),
            &ConvertPolicy::permissive(),
            None,
        )
        .unwrap_err()
        .error(),
        ConvertError::UnsupportedRoute
    );
    assert!(
        Negotiate::select(
            &unknown_rgb,
            core::slice::from_ref(&gray_lin),
            &ConvertPolicy::permissive()
        )
        .is_none()
    );

    // Explicit BT.709 red [1,0,0] -> 0.2126; Display P3 red [1,0,0] -> 0.22897456!
    let p_709 = RowPlan::build(
        ImageSpec::RGBF32_LINEAR,
        gray_lin.clone(),
        &ConvertPolicy::permissive(),
        None,
    )
    .unwrap();
    let p_p3 = RowPlan::build(
        ImageSpec::RGBF32_LINEAR.with_primaries(Primaries::DisplayP3),
        gray_lin,
        &ConvertPolicy::permissive(),
        None,
    )
    .unwrap();
    let red = [1.0f32, 0.0, 0.0];
    let mut g709 = [0.0f32; 1];
    let mut gp3 = [0.0f32; 1];
    p_709
        .convert_row(
            bytemuck::cast_slice(&red),
            bytemuck::cast_slice_mut(&mut g709),
            1,
            0,
        )
        .unwrap();
    p_p3.convert_row(
        bytemuck::cast_slice(&red),
        bytemuck::cast_slice_mut(&mut gp3),
        1,
        0,
    )
    .unwrap();
    assert!(
        (g709[0] - 0.2126).abs() < 1e-4,
        "BT.709 red luma={}",
        g709[0]
    );
    assert!((gp3[0] - 0.22897456).abs() < 1e-4, "P3 red luma={}", gp3[0]);
}

#[test]
fn b10_b11_b12_b13_b42_signaling_authority_strict_icc_and_owned_borrowed_pixels() {
    // B10: Empty, truncated, bad-size, DeviceLink, and wrong-data-space ICC MUST fail OutputPlan!
    let mut devicelink = vec![0u8; 132];
    devicelink[0..4].copy_from_slice(&132u32.to_be_bytes());
    devicelink[12..16].copy_from_slice(b"link");
    devicelink[16..20].copy_from_slice(b"RGB ");
    devicelink[36..40].copy_from_slice(b"acsp");
    for bad_icc in [vec![], vec![0u8; 64], devicelink] {
        let spec = ImageSpec::new(
            SampleLayout::RGB8,
            ColorSpace::Icc {
                profile: Arc::from(bad_icc.as_slice()),
                channels: 3,
                alpha: AlphaMode::None,
                intent: RenderingIntent::Perceptual,
            },
        );
        let plan = OutputPlan::build(
            spec.clone(),
            spec,
            &ConvertPolicy::permissive(),
            PeakSource::Measured,
            OutputSignaling::Auto,
        );
        assert!(
            plan.is_err(),
            "OutputPlan::Auto must reject structurally invalid/DeviceLink ICC"
        );
    }

    // B11: Known non-sRGB Gray (Linear Gray8 & Gamma 2.2 Gray8) under Auto MUST emit Gray ICC
    // (with `icc-db`) or fail with OutputSignalingUnavailable (never silently emit `(None, None)`)!
    for tf in [
        Transfer::Linear,
        Transfer::Gamma22,
        Transfer::Gamma(Gamma::new(2.4).unwrap()),
    ] {
        let gray_spec = ImageSpec::GRAY8_SRGB.with_transfer(tf);
        let res = OutputPlan::build(
            gray_spec.clone(),
            gray_spec,
            &ConvertPolicy::permissive(),
            PeakSource::Measured,
            OutputSignaling::Auto,
        );
        match res {
            Ok(plan) => {
                let (cicp, icc) = plan.resolve_signaling(&ColorOrigin::assumed()).unwrap();
                assert!(
                    cicp.is_some() || icc.is_some(),
                    "Auto must never silently return (None, None) for known Gray {tf:?}"
                );
                if let Some(bytes) = icc {
                    assert_eq!(&bytes[16..20], b"GRAY");
                }
            }
            Err(e) => {
                assert_eq!(*e.error(), ConvertError::OutputSignalingUnavailable);
            }
        }
    }

    // B12: SameAsOrigin on untagged assumed sRGB returns `(None, None)` (doesn't invent CICP!),
    // AND agreeing dual P3 ICC + P3 CICP on `ColorSpace::Icc` succeeds preserving both!
    let srgb_same = OutputPlan::build(
        ImageSpec::RGB8_SRGB,
        ImageSpec::RGB8_SRGB,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputSignaling::SameAsOrigin,
    )
    .unwrap();
    let (c_untagged, i_untagged) = srgb_same
        .resolve_signaling(&ColorOrigin::assumed())
        .unwrap();
    assert_eq!((c_untagged, i_untagged), (None, None));

    let p3_icc: Arc<[u8]> = Arc::from(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
    let p3_icc_spec = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: p3_icc.clone(),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    let p3_same = OutputPlan::build(
        p3_icc_spec.clone(),
        p3_icc_spec,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputSignaling::SameAsOrigin,
    )
    .unwrap();
    let dual_origin = ColorOrigin::from_icc_and_cicp(p3_icc.clone(), Cicp::DISPLAY_P3);
    let (dc, di) = p3_same.resolve_signaling(&dual_origin).unwrap();
    assert_eq!(dc, Some(Cicp::DISPLAY_P3));
    assert_eq!(di.as_deref(), Some(p3_icc.as_ref()));
    let conflict_origin = ColorOrigin::from_icc_and_cicp(p3_icc, Cicp::BT2100_PQ);
    assert!(p3_same.resolve_signaling(&conflict_origin).is_err());

    // B13: Gray CP=2 (unspecified primaries) with assigned TC=8 (Linear) synthesizes D65 Gray ICC!
    #[cfg(feature = "icc-db")]
    {
        use zenpixels_convert::icc_profiles::SynthesizedIcc;
        let gray_cp2 = SynthesizedIcc::for_gray_cicp(Cicp::new(2, 8, 0, true));
        let gray_cp1 = SynthesizedIcc::for_gray_cicp(Cicp::new(1, 8, 0, true));
        assert!(matches!(gray_cp2, SynthesizedIcc::Profile(_)));
        assert_eq!(gray_cp2, gray_cp1);
    }

    // B42: Borrowed vs owned OutputPlan checks ACTUAL transformed P3 -> sRGB pixels (not just tag!),
    // and execute_view rejects mismatched spec!
    let p3_spec = ImageSpec::new(SampleLayout::RGB8, ColorSpace::DISPLAY_P3);
    let out_plan = OutputPlan::build(
        p3_spec.clone(),
        ImageSpec::RGB8_SRGB,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputSignaling::Auto,
    )
    .unwrap();
    let src_buf = PixelBuf::from_typed(vec![200u8, 60, 40, 12, 124, 200], 2, 1, p3_spec).unwrap();
    let r_view = out_plan
        .execute_view(src_buf.as_view(), &ColorOrigin::assumed())
        .unwrap();
    let r_owned = out_plan.execute_buf(src_buf.clone()).unwrap();
    assert_eq!(
        r_view.pixels.as_view().row(0),
        r_owned.pixels.as_view().row(0)
    );
    assert_ne!(
        r_view.pixels.as_view().row(0),
        src_buf.as_view().row(0),
        "P3 -> sRGB OutputPlan must actually transform pixels, not pass through!"
    );
    let got = r_view.pixels.as_view().row(0);
    assert!(got[0] > 215 && got[1] < 55, "P3->sRGB pixel={got:?}");

    let wrong = PixelBuf::new(2, 1, ImageSpec::RGB8_SRGB).unwrap();
    assert!(
        out_plan
            .execute_view(wrong.as_view(), &ColorOrigin::assumed())
            .is_err()
    );
}

#[test]
fn a02_a01_b33_b34_transfers_narrow_bitshift_and_fast_row_float_facts() {
    // A02: Gamma22 (2.2) vs AdobeRgb (563/256) at 0.5 produce distinct binary64 reference values!
    let g22 = ImageSpec::RGBF32_LINEAR.with_transfer(Transfer::Gamma22);
    let adobe = ImageSpec::RGBF32_LINEAR.with_transfer(Transfer::AdobeRgb);
    let dci = ImageSpec::RGBF32_LINEAR.with_transfer(Transfer::DciCinema);
    let lin = ImageSpec::RGBF32_LINEAR;
    let half = [0.5f32, 0.5, 0.5];
    let one = [1.0f32, 1.0, 1.0];
    let mut o_g22 = [0f32; 3];
    let mut o_adobe = [0f32; 3];
    let mut o_dci_half = [0f32; 3];
    let mut o_dci_one = [0f32; 3];
    RowPlan::build(g22, lin.clone(), &ConvertPolicy::permissive(), None)
        .unwrap()
        .convert_row(
            bytemuck::cast_slice(&half),
            bytemuck::cast_slice_mut(&mut o_g22),
            1,
            0,
        )
        .unwrap();
    RowPlan::build(adobe, lin.clone(), &ConvertPolicy::permissive(), None)
        .unwrap()
        .convert_row(
            bytemuck::cast_slice(&half),
            bytemuck::cast_slice_mut(&mut o_adobe),
            1,
            0,
        )
        .unwrap();
    let dci_plan = RowPlan::build(dci, lin.clone(), &ConvertPolicy::permissive(), None).unwrap();
    dci_plan
        .convert_row(
            bytemuck::cast_slice(&half),
            bytemuck::cast_slice_mut(&mut o_dci_half),
            1,
            0,
        )
        .unwrap();
    dci_plan
        .convert_row(
            bytemuck::cast_slice(&one),
            bytemuck::cast_slice_mut(&mut o_dci_one),
            1,
            0,
        )
        .unwrap();
    assert!(
        (o_g22[0] - 0.5f32.powf(2.2)).abs() < 1e-4,
        "Gamma22(0.5)={}",
        o_g22[0]
    );
    assert!(
        (o_adobe[0] - 0.5f32.powf(563.0 / 256.0)).abs() < 1e-4,
        "Adobe(0.5)={}",
        o_adobe[0]
    );
    assert!(
        (o_adobe[0] - o_g22[0]).abs() > 1e-4,
        "Adobe and Gamma22 must differ at 0.5!"
    );
    assert!(
        (o_dci_one[0] - (52.37 / 48.0)).abs() < 1e-4,
        "DCI TC17(1.0)={}, expected ~1.091042",
        o_dci_one[0]
    );
    assert!(
        (o_dci_half[0] - (52.37 / 48.0) * 0.5f32.powf(2.6)).abs() < 1e-4,
        "DCI TC17(0.5)={}",
        o_dci_half[0]
    );

    // B33: Narrow-range BT.2100-3 Table 9 bit-shift preserves footroom & headroom!
    let n12 = ImageSpec::new(
        SampleLayout::new(SampleType::U16, ChannelOrder::Rgb)
            .with_packing(BitPacking::Lsb { bits: 12 }),
        ColorSpace::SRGB.with_range(SignalRange::Narrow),
    );
    let n10 = ImageSpec::new(
        SampleLayout::new(SampleType::U16, ChannelOrder::Rgb)
            .with_packing(BitPacking::Lsb { bits: 10 }),
        ColorSpace::SRGB.with_range(SignalRange::Narrow),
    );
    let p_12_10 = RowPlan::build(n12, n10, &ConvertPolicy::permissive(), None).unwrap();
    let mut o10 = [0u16; 3];
    p_12_10
        .convert_row(
            bytemuck::cast_slice(&[16u16, 274, 4000]),
            bytemuck::cast_slice_mut(&mut o10),
            1,
            0,
        )
        .unwrap();
    assert_eq!(o10, [4, 69, 1000]);

    // U8 narrow -> U16 narrow lossless widening: [15, 235, 236] -> [3840, 60160, 60416] (<< 8 exact!).
    let n8 = ImageSpec::RGB8_SRGB.with_range(SignalRange::Narrow);
    let n16 = ImageSpec::RGB16_SRGB.with_range(SignalRange::Narrow);
    let p_8_16 = RowPlan::build(n8, n16, &ConvertPolicy::lossless(), None).unwrap();
    let mut o16 = [0u16; 3];
    p_8_16
        .convert_row(&[15u8, 235, 236], bytemuck::cast_slice_mut(&mut o16), 1, 0)
        .unwrap();
    assert_eq!(o16, [3840, 60160, 60416]);

    // B34: PQ F32 RGB [1,1,1] -> Linear F32 RGB vs BGR under Clip MUST agree AND clip to [1,1,1],
    // while PreserveExtended MUST NOT claim AT_MOST_ONE!
    #[cfg(feature = "hdr-transfers")]
    {
        use zenpixels_convert::GamutPolicy;
        let pq_f32 = ImageSpec::new(SampleLayout::RGBF32, ColorSpace::BT2100_PQ);
        let lin_rgb = ImageSpec::RGBF32_LINEAR
            .with_primaries(Primaries::Bt2020)
            .with_white(DiffuseWhite::new(100.0));
        let lin_bgr = ImageSpec::new(
            SampleLayout::new(SampleType::F32, ChannelOrder::Bgr),
            lin_rgb.color.clone(),
        );
        let pol_clip = ConvertPolicy::permissive().with_gamut(GamutPolicy::Clip);
        let p_rgb = RowPlan::build(pq_f32.clone(), lin_rgb.clone(), &pol_clip, None).unwrap();
        let p_bgr = RowPlan::build(pq_f32.clone(), lin_bgr, &pol_clip, None).unwrap();
        let mut out_rgb = [0f32; 3];
        let mut out_bgr = [0f32; 3];
        p_rgb
            .convert_row(
                bytemuck::cast_slice(&one),
                bytemuck::cast_slice_mut(&mut out_rgb),
                1,
                0,
            )
            .unwrap();
        p_bgr
            .convert_row(
                bytemuck::cast_slice(&one),
                bytemuck::cast_slice_mut(&mut out_bgr),
                1,
                0,
            )
            .unwrap();
        assert_eq!(out_rgb, [1.0, 1.0, 1.0]);
        assert_eq!(out_bgr, [1.0, 1.0, 1.0]);

        let pol_ext = ConvertPolicy::permissive().with_gamut(GamutPolicy::PreserveExtended);
        let p_ext = RowPlan::build(pq_f32, lin_rgb, &pol_ext, None).unwrap();
        assert!(!p_ext.output_facts().float.contains(FloatFacts::AT_MOST_ONE));
        let mut out_ext = [0f32; 3];
        p_ext
            .convert_row(
                bytemuck::cast_slice(&one),
                bytemuck::cast_slice_mut(&mut out_ext),
                1,
                0,
            )
            .unwrap();
        assert!(
            (out_ext[0] - 100.0).abs() < 1e-2,
            "PQ(1.0) at 100 nits = 100.0, got {}",
            out_ext[0]
        );
    }

    // B34: Tiny white ratio producing subnormal output MUST NOT claim `FloatFacts::NO_SUBNORMALS`!
    let src_tiny = ImageSpec::RGB8_SRGB
        .with_transfer(Transfer::Linear)
        .with_white(DiffuseWhite::new(1e-35));
    let dst_huge = ImageSpec::RGBF32_LINEAR.with_white(DiffuseWhite::new(10_000.0));
    let p_sub = RowPlan::build(src_tiny, dst_huge, &ConvertPolicy::permissive(), None).unwrap();
    assert!(
        !p_sub
            .output_facts()
            .float
            .contains(FloatFacts::NO_SUBNORMALS),
        "extreme white downscale can produce subnormals and must not claim NO_SUBNORMALS"
    );
}

#[test]
fn d03_codemap_and_to_canonical_exhaustive() {
    let src10 = SampleEncoding::new(SampleType::U16, 10, 0).unwrap();
    let dst16 = SampleEncoding::new(SampleType::U16, 16, 0).unwrap();
    let map = CodeMap::new(src10, dst16, CodeRole::Intensity, SignalRange::Full).unwrap();
    let all10: Vec<u16> = (0..=1023).collect();
    let mut mapped = vec![0u16; 1024];
    map.map_words(&all10, &mut mapped).unwrap();
    for (c, &got) in mapped.iter().enumerate() {
        let want = ((c as u64 * 65535 + 511) / 1023) as u16;
        assert_eq!(got, want, "10->16 full mismatch at code {c}");
    }

    // Narrow 10-bit -> 12-bit intensity shift exact (`c << 2`).
    let n10 = SampleEncoding::new(SampleType::U16, 10, 0).unwrap();
    let n12 = SampleEncoding::new(SampleType::U16, 12, 0).unwrap();
    let nmap = CodeMap::new(n10, n12, CodeRole::Intensity, SignalRange::Narrow).unwrap();
    let mut nmapped = [0u16; 3];
    nmap.map_words(&[64, 512, 940], &mut nmapped).unwrap();
    assert_eq!(nmapped, [256, 2048, 3760]);

    // `to_canonical` promotes 10-bit full-range CodePlane into canonical U16 PixelViewMut!
    let words = [0u16, 512, 1023];
    let plane = CodePlane::from_u16(&words, 3, 1, 3, src10).unwrap();
    let mut dst_buf = PixelBuf::new(3, 1, ImageSpec::GRAY16_SRGB).unwrap();
    let (_, exact) =
        to_canonical(&[plane], SignalRange::Full, 0, 0, dst_buf.as_view_mut()).unwrap();
    assert_eq!(exact.color_bits(), 10);
    let dst_u16 = dst_buf.as_view().row_as::<u16>(0).unwrap();
    assert_eq!(dst_u16, &[0, ((512u64 * 65535 + 511) / 1023) as u16, 65535]);

    let _ = (
        AlphaPolicy::Forbid,
        DepthPolicy::Round,
        RangePolicy::Convert,
        Nits::new(100.0),
        |v, s: &ImageSpec, p, pk| convert_to_buf_with_peak(v, s, p, pk),
    );
}

#[test]
fn phase4a_icc_hdr_identity_load_bearing_origin_and_cicp_alias() {
    // H9: Identical-ICC identity / layout conversion MUST preserve `HdrMetadata` (CLL + mastering)!
    let p3_icc: Arc<[u8]> = Arc::from(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
    let icc_spec = ImageSpec::new(
        SampleLayout::RGB16,
        ColorSpace::Icc {
            profile: Arc::clone(&p3_icc),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::Perceptual,
        },
    );
    let hdr = HdrMetadata {
        cll: Some(ContentLightLevel::new(1000, 400)),
        mastering: Some(MasteringDisplay::new(
            [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
            [0.3127, 0.3290],
            0.005,
            1000.0,
        )),
    };
    let raw_nclx = Cicp::new(1, 1, 1, false);
    let orig = ColorOrigin::from_icc_and_cicp(Arc::clone(&p3_icc), raw_nclx);
    let src_buf = PixelBuf::from_typed(vec![1000u16, 2000, 3000], 1, 1, icc_spec.clone())
        .unwrap()
        .with_origin(orig.clone())
        .with_hdr(hdr);
    let copied = convert_to_buf_with_peak(
        &src_buf.as_view(),
        icc_spec.clone(),
        &ConvertPolicy::permissive(),
        PeakSource::Declared,
    )
    .unwrap();
    assert_eq!(
        copied.hdr(),
        hdr,
        "same-ICC identity conversion must preserve HDR metadata"
    );
    assert_eq!(
        copied.origin(),
        &orig,
        "same-ICC identity conversion must preserve origin"
    );
    assert_eq!(
        copied.as_view().row_as::<u16>(0).unwrap(),
        &[1000, 2000, 3000]
    );

    // Load-bearing RGB -> Gray collapse MUST preserve original source `ColorOrigin` provenance
    // and refuse collapsing non-D65 primaries (`DciP3`) to D65 Gray!
    let p3_rgb_spec = ImageSpec::RGB8_SRGB.with_primaries(Primaries::DisplayP3);
    let mut neutral_p3 = PixelBuf::from_typed(vec![128u8, 128, 128, 64, 64, 64], 2, 1, p3_rgb_spec)
        .unwrap()
        .with_origin(orig.clone())
        .with_hdr(hdr);
    let reduced_alloc = compact_to_buf(&neutral_p3.as_view())
        .unwrap()
        .expect("neutral Display P3 RGB reduces to Gray");
    assert_eq!(reduced_alloc.spec(), &ImageSpec::GRAY8_SRGB);
    assert_eq!(
        reduced_alloc.origin(),
        &orig,
        "compaction must preserve original file provenance untouched"
    );
    assert_eq!(reduced_alloc.hdr(), hdr);
    compact_in_place(&mut neutral_p3, AlphaCompaction::DropUnusedLanes);
    assert_eq!(neutral_p3.spec(), &ImageSpec::GRAY8_SRGB);
    assert_eq!(neutral_p3.origin(), &orig);

    // Non-D65 DCI-P3 RGB MUST NOT collapse to D65 Gray!
    let dci_spec = ImageSpec::RGB8_SRGB.with_primaries(Primaries::DciP3);
    let neutral_dci = PixelBuf::from_typed(vec![128u8, 128, 128], 1, 1, dci_spec.clone()).unwrap();
    let dci_rep = analyze_compaction(&neutral_dci.as_view());
    assert_eq!(dci_rep.candidate_spec(&dci_spec), dci_spec);
    assert!(
        compact_to_buf(&neutral_dci.as_view()).unwrap().is_none(),
        "DciP3 non-D65 white point must not collapse to D65 Gray"
    );

    // `OutputSignaling::SameAsOrigin` accepts semantically equivalent BT.709 transfer aliases
    // (`[9, 14, 0, 1]` BT.2020 10-bit transfer alias == `Transfer::Bt709` canonical `[9, 1, 0, 1]`)!
    let bt2020_709 = ImageSpec::RGB16_SRGB
        .with_primaries(Primaries::Bt2020)
        .with_transfer(Transfer::Bt709);
    let alias_cicp = Cicp::new(9, 14, 0, true);
    let alias_origin = ColorOrigin::from_cicp(alias_cicp);
    let plan = OutputPlan::build(
        bt2020_709.clone(),
        bt2020_709,
        &ConvertPolicy::permissive(),
        PeakSource::Declared,
        OutputSignaling::SameAsOrigin,
    )
    .unwrap();
    let (cicp_out, _icc_out) = plan.resolve_signaling(&alias_origin).expect(
        "SameAsOrigin must accept semantically equivalent CICP transfer alias TC 14 for Bt709",
    );
    assert_eq!(cicp_out, Some(alias_cicp));
}

#[test]
fn opus_review_p0_1_p0_2_p1_1_p1_2_p2_1_regressions() {
    // P0-1: Discarding premultiplied alpha under permissive() MUST unpremultiply consistently
    // across BOTH U8 -> RGB8 and U8 -> RGB16 (and -> RGBA8 Opaque)!
    let premul_spec = ImageSpec::RGBA8_SRGB.with_alpha(AlphaMode::Premultiplied);
    let premul_buf =
        PixelBuf::from_typed(vec![64u8, 0, 0, 128], 1, 1, premul_spec.clone()).unwrap();
    let out_rgb8 = convert_to_buf_with_peak(
        &premul_buf.as_view(),
        ImageSpec::RGB8_SRGB,
        &ConvertPolicy::permissive(),
        PeakSource::Declared,
    )
    .unwrap();
    let out_rgb16 = convert_to_buf_with_peak(
        &premul_buf.as_view(),
        ImageSpec::RGB16_SRGB,
        &ConvertPolicy::permissive(),
        PeakSource::Declared,
    )
    .unwrap();
    let r8 = out_rgb8.as_view().row(0)[0];
    let r16 = out_rgb16.as_view().row_as::<u16>(0).unwrap()[0];
    assert!(
        (127..=129).contains(&r8),
        "premul (64,0,0,128) -> RGB8 must unpremultiply (~128), got {r8}"
    );
    assert!(
        ((r16 as i32) - (r8 as i32 * 257)).abs() <= 257,
        "RGB8 ({r8}) and RGB16 ({r16}) must agree on unpremultiplied red"
    );
    let out_opaque = convert_to_buf_with_peak(
        &premul_buf.as_view(),
        ImageSpec::RGBA8_SRGB.with_alpha(AlphaMode::Opaque),
        &ConvertPolicy::permissive(),
        PeakSource::Declared,
    )
    .unwrap();
    assert!((127..=129).contains(&out_opaque.as_view().row(0)[0]));
    assert_eq!(out_opaque.as_view().row(0)[3], 255);

    // P0-2: Straight -> Premultiplied byte path MUST NOT preserve 1-bit `ExactDepth::color_bits`!
    let mut logo = PixelBuf::from_typed(
        vec![255u8, 0, 0, 128, 0, 255, 0, 255],
        2,
        1,
        ImageSpec::RGBA8_SRGB,
    )
    .unwrap();
    let measured = scan_facts(&logo.as_view());
    assert_eq!(measured.depth.unwrap().color_bits(), 1);
    logo.assume_facts(measured);
    let premul_out = convert_to_buf_with_peak(
        &logo.as_view(),
        ImageSpec::RGBA8_SRGB.with_alpha(AlphaMode::Premultiplied),
        &ConvertPolicy::permissive(),
        PeakSource::Declared,
    )
    .unwrap();
    let out_depth = premul_out.facts().depth.expect("U8 output has ExactDepth");
    assert_eq!(
        out_depth.color_bits(),
        8,
        "straight->premul rewrites color samples and must not claim color_bits=1"
    );
    assert!(
        out_depth.holds_for(&premul_out.as_view()).unwrap(),
        "attached ExactDepth must actually hold for output pixels!"
    );

    // P1-1: Unverified `Opaque -> Opaque` canonicalizes poisoned alpha=0 to 255 (both RGBA8 and GrayAlpha8),
    // while fresh `PixelBuf::new(..., Opaque)` keeps `all_opaque = true` and uses zero-copy `Memcpy`!
    let op_rgba_spec = ImageSpec::RGBA8_SRGB.with_alpha(AlphaMode::Opaque);
    let poisoned = PixelBuf::from_typed(vec![10u8, 20, 30, 0], 1, 1, op_rgba_spec.clone()).unwrap();
    let canon = convert_to_buf_with_peak(
        &poisoned.as_view(),
        op_rgba_spec.clone(),
        &ConvertPolicy::permissive(),
        PeakSource::Declared,
    )
    .unwrap();
    assert_eq!(canon.as_view().row(0), &[10, 20, 30, 255]);
    assert!(canon.facts().all_opaque);

    let op_ga_spec = ImageSpec::new(
        SampleLayout::new(SampleType::U8, ChannelOrder::GrayAlpha),
        ColorSpace::GRAY_SRGB.with_alpha(AlphaMode::Opaque),
    );
    let poisoned_ga = PixelBuf::from_typed(vec![77u8, 0], 1, 1, op_ga_spec.clone()).unwrap();
    let canon_ga = convert_to_buf_with_peak(
        &poisoned_ga.as_view(),
        op_ga_spec,
        &ConvertPolicy::permissive(),
        PeakSource::Declared,
    )
    .unwrap();
    assert_eq!(canon_ga.as_view().row(0), &[77, 255]);
    assert!(canon_ga.facts().all_opaque);

    let fresh_op = PixelBuf::new(1, 1, op_rgba_spec.clone()).unwrap();
    let id_plan = RowPlan::build(
        op_rgba_spec.clone(),
        op_rgba_spec,
        &ConvertPolicy::permissive().with_source_facts(fresh_op.facts()),
        None,
    )
    .unwrap();
    assert!(
        id_plan.is_identity(),
        "fresh PixelBuf::new(..., Opaque) has all_opaque=true and stays Memcpy"
    );

    // P1-2: Widening U8 3-bit grid {0, 36, ..., 255} to U16 MUST NOT falsely claim `color_bits = 3`!
    let mut u8_3bit = PixelBuf::from_typed(
        vec![0u8, 36, 73, 109, 146, 182, 219, 255],
        8,
        1,
        ImageSpec::GRAY8_SRGB,
    )
    .unwrap();
    let f3 = scan_facts(&u8_3bit.as_view());
    assert_eq!(f3.depth.unwrap().color_bits(), 3);
    u8_3bit.assume_facts(f3);
    let u16_from_3 = convert_to_buf_with_peak(
        &u8_3bit.as_view(),
        ImageSpec::GRAY16_SRGB,
        &ConvertPolicy::permissive(),
        PeakSource::Declared,
    )
    .unwrap();
    let d_u16_3 = u16_from_3.facts().depth.unwrap();
    assert_eq!(
        d_u16_3.color_bits(),
        8,
        "3-bit U8 grid does not nest under *257 into U16 3-bit grid"
    );
    assert!(d_u16_3.holds_for(&u16_from_3.as_view()).unwrap());

    // Conversely, 4-bit U8 grid {0, 17, 34, ..., 255} DOES nest in U16 (`4 | 8`)!
    let mut u8_4bit =
        PixelBuf::from_typed(vec![0u8, 17, 34, 51, 255], 5, 1, ImageSpec::GRAY8_SRGB).unwrap();
    let f4 = scan_facts(&u8_4bit.as_view());
    assert_eq!(f4.depth.unwrap().color_bits(), 4);
    u8_4bit.assume_facts(f4);
    let u16_from_4 = convert_to_buf_with_peak(
        &u8_4bit.as_view(),
        ImageSpec::GRAY16_SRGB,
        &ConvertPolicy::permissive(),
        PeakSource::Declared,
    )
    .unwrap();
    let d_u16_4 = u16_from_4.facts().depth.unwrap();
    assert_eq!(d_u16_4.color_bits(), 4);
    assert!(d_u16_4.holds_for(&u16_from_4.as_view()).unwrap());

    // P2-1: Reused sink clears stale ColorOrigin when color meaning changes!
    let p3_origin = ColorOrigin::from_cicp(Cicp::DISPLAY_P3);
    let mut reused_sink = PixelBuf::new(1, 1, ImageSpec::RGB8_SRGB)
        .unwrap()
        .with_origin(p3_origin.clone());
    let src_p3 = PixelBuf::from_typed(
        vec![200u8, 50, 50],
        1,
        1,
        ImageSpec::RGB8_SRGB.with_primaries(Primaries::DisplayP3),
    )
    .unwrap()
    .with_origin(p3_origin);
    let plan_p3_to_srgb = ImagePlan::build(
        src_p3.spec().clone(),
        reused_sink.spec().clone(),
        &ConvertPolicy::permissive(),
        PeakSource::Declared,
    )
    .unwrap();
    plan_p3_to_srgb
        .execute_into(&src_p3.as_view(), &mut reused_sink)
        .unwrap();
    assert_eq!(
        reused_sink.origin(),
        &ColorOrigin::assumed(),
        "reused sink must clear stale ColorOrigin when color meaning changed"
    );
}
