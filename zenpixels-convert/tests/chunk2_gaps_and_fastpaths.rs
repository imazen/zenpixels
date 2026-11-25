//! Integration tests verifying Chunk 2 gap closures and hot-path speed/parity invariants:
//! - C1: `LumaDomain::{Encoded, LinearLight}` (`(255,0,0)` -> 54 Encoded vs 127 LinearLight,
//!   12-variant `garb` SIMD vs scalar 0-bit parity, Display P3 deferred encoded OETF, Unknown transfer)
//! - G03: `OutputPlan` `AlphaMode::Opaque` canonicalization vs proven-opaque zero-copy
//! - G04: `OutputSignaling::CustomIcc` / `SameAsOrigin` rejects `MetadataOnly` ICC profiles
//! - G14: `Negotiate::select` scene-linear HDR precheck without tone mapper
//! - G02: `MoxCms` float MPE (`D2B*`/`B2D*`) refusal, unequal adjacent RGB TRC refusal, extended F32
//! - T2-3/G05: `apply_orientation_banded` / `apply_orientation_banded_into` across all 8 orientations & seams
//! - T2-6/G18: `measure_max_smoothed`, `LightLevelHistogram::{bins, try_merge}`
//! - T2-4/T2-5: `adapt_for_encode`, `to_rgb8`/`to_rgba8`/`to_gray8`/`to_bgra8`, `ycbcr_to_rgb_from_planar`
//! - T2-11: Exhaustive 65,536-pair `rgba8_straight_to_premul` & `lin_f32_to_srgb_u8` SIMD parity

use alloc::sync::Arc;
use zenpixels::icc::{IccIdentification, IdentificationUse};
use zenpixels::{
    AlphaMode, BandedBuf, ChannelOrder, ColorOrigin, ColorSpace, DiffuseWhite, FloatFacts,
    ImageSpec, Orientation, PeakSource, PixelBuf, PixelCow, PlanarBuf, PlaneBuf, PlaneRole,
    PlaneSpec, Primaries, RenderingIntent, SampleFacts, SampleLayout, SampleType, SignalRange,
    Transfer,
};
use zenpixels_convert::{
    Accelerators, ConvertError, ConvertPolicy, DepthPolicy, LightLevelHistogram, LightLevelMethod,
    LumaCoefficients, LumaDomain, Negotiate, OutputPlan, OutputSignaling,
    PlanarReconstructionOptions, RowPlan, YcbcrMatrix, adapt_for_encode, apply_orientation,
    apply_orientation_banded, apply_orientation_banded_into, measure_cll, measure_histogram,
    measure_max_smoothed, reconstruct_planar_f32, reconstruct_planar_to_buf, to_bgra8, to_gray8,
    to_rgb8, to_rgba8,
};

extern crate alloc;

#[test]
fn c1_luma_domain_encoded_default_vs_linear_light_and_simd_parity() {
    let rgb_srgb = ImageSpec::RGB8_SRGB;
    let gray_srgb = ImageSpec::GRAY8_SRGB;

    // Default policy uses LumaDomain::Encoded (0.2/0.4 JPEG/video Y' semantic).
    let def_pol = ConvertPolicy::permissive();
    assert_eq!(def_pol.luma_domain, LumaDomain::Encoded);

    let plan_enc = RowPlan::build(rgb_srgb.clone(), gray_srgb.clone(), &def_pol, None).unwrap();
    let plan_lin = RowPlan::build(
        rgb_srgb.clone(),
        gray_srgb.clone(),
        &ConvertPolicy::permissive().with_luma_domain(LumaDomain::LinearLight),
        None,
    )
    .unwrap();

    // Pure sRGB red (255, 0, 0): Encoded Y' = 54; LinearLight Y -> sRGB = 127!
    let red = [255u8, 0, 0];
    let mut out_enc = [0u8; 1];
    let mut out_lin = [0u8; 1];
    plan_enc.convert_row(&red, &mut out_enc, 1, 0).unwrap();
    plan_lin.convert_row(&red, &mut out_lin, 1, 0).unwrap();
    assert_eq!(out_enc[0], 54, "encoded BT.709 sRGB red must be 54");
    assert_eq!(out_lin[0], 127, "linear-light BT.709 sRGB red must be 127");

    // Neutral gray R=G=B must round-trip bit-exact for all 256 levels in Encoded mode.
    let mut grays = Vec::with_capacity(256 * 3);
    for v in 0u8..=255 {
        grays.extend_from_slice(&[v, v, v]);
    }
    let mut out_grays = vec![0u8; 256];
    plan_enc
        .convert_row(&grays, &mut out_grays, 256, 0)
        .unwrap();
    for (v, &g) in out_grays.iter().enumerate() {
        assert_eq!(g, v as u8, "neutral gray {v} drifted to {g}");
    }

    // 0-bit parity between SCALAR and ALL across all 12 RGB/BGR/RGBA/BGRA x BT709/BT601/BT2020 combos
    // and odd widths (1, 17, 63, 64, 65, 97).
    let orders = [
        (ChannelOrder::Rgb, AlphaMode::None),
        (ChannelOrder::Bgr, AlphaMode::None),
        (ChannelOrder::Rgba, AlphaMode::Straight),
        (ChannelOrder::Bgra, AlphaMode::Straight),
    ];
    let coeffs = [
        LumaCoefficients::Bt709,
        LumaCoefficients::Bt601,
        LumaCoefficients::Bt2020,
    ];
    for (order, alpha) in orders {
        let bpp = order.channels();
        let src_spec = ImageSpec::new(
            SampleLayout::new(SampleType::U8, order),
            ColorSpace::Rgb {
                primaries: Primaries::Bt709,
                transfer: Transfer::Srgb,
                range: SignalRange::Full,
                alpha,
                white: None,
            },
        );
        for coeff in coeffs {
            let p_sc = RowPlan::build(
                src_spec.clone(),
                gray_srgb.clone(),
                &ConvertPolicy::permissive()
                    .with_luma(coeff)
                    .with_accelerators(Accelerators::SCALAR),
                None,
            )
            .unwrap();
            let p_all = RowPlan::build(
                src_spec.clone(),
                gray_srgb.clone(),
                &ConvertPolicy::permissive()
                    .with_luma(coeff)
                    .with_accelerators(Accelerators::ALL),
                None,
            )
            .unwrap();
            for w in [1u32, 17, 63, 64, 65, 97] {
                let src: Vec<u8> = (0..(w as usize * bpp))
                    .map(|i| ((i * 73 + 19) & 0xFF) as u8)
                    .collect();
                let mut d_sc = vec![0u8; w as usize];
                let mut d_all = vec![0u8; w as usize];
                p_sc.convert_row(&src, &mut d_sc, w, 0).unwrap();
                p_all.convert_row(&src, &mut d_all, w, 0).unwrap();
                assert_eq!(
                    d_sc, d_all,
                    "SCALAR vs ALL mismatch for {order:?}/{coeff:?} width={w}"
                );
            }
        }
    }

    // Matching Transfer::Unknown succeeds in Encoded mode, fails in LinearLight mode.
    let unk_rgb = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Unknown,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let unk_gray = ImageSpec::new(
        SampleLayout::GRAY8,
        ColorSpace::Gray {
            transfer: Transfer::Unknown,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    assert!(
        RowPlan::build(
            unk_rgb.clone(),
            unk_gray.clone(),
            &ConvertPolicy::permissive(),
            None
        )
        .is_ok()
    );
    assert_eq!(
        *RowPlan::build(
            unk_rgb,
            unk_gray,
            &ConvertPolicy::permissive().with_luma_domain(LumaDomain::LinearLight),
            None,
        )
        .unwrap_err()
        .error(),
        ConvertError::UnsupportedRoute
    );

    // Cross-transfer Linear RGB -> sRGB Gray: Encoded applies OETF before luma (red = 54),
    // LinearLight computes luma in linear light before OETF (red = 127).
    let lin_rgb = ImageSpec::RGBF32_LINEAR;
    let p_cross_enc = RowPlan::build(
        lin_rgb.clone(),
        gray_srgb.clone(),
        &ConvertPolicy::permissive().with_luma_domain(LumaDomain::Encoded),
        None,
    )
    .unwrap();
    let p_cross_lin = RowPlan::build(
        lin_rgb,
        gray_srgb,
        &ConvertPolicy::permissive().with_luma_domain(LumaDomain::LinearLight),
        None,
    )
    .unwrap();
    let lin_red = [1.0f32, 0.0, 0.0];
    let mut g_ce = [0u8; 1];
    let mut g_cl = [0u8; 1];
    p_cross_enc
        .convert_row(bytemuck::cast_slice(&lin_red), &mut g_ce, 1, 0)
        .unwrap();
    p_cross_lin
        .convert_row(bytemuck::cast_slice(&lin_red), &mut g_cl, 1, 0)
        .unwrap();
    assert_eq!(g_ce[0], 54);
    assert_eq!(g_cl[0], 127);
}

#[test]
fn g03_output_plan_canonicalizes_unverified_opaque_alpha_and_stays_zero_copy_when_proven() {
    let spec = ImageSpec::RGBA8_SRGB.with_alpha(AlphaMode::Opaque);

    // Unverified poisoned Opaque alpha [9, 13, 17, 0] -> MUST canonicalize to [9, 13, 17, 255]!
    let mut poisoned = PixelBuf::from_typed(vec![9u8, 13, 17, 0], 1, 1, spec.clone()).unwrap();
    poisoned.assume_facts(SampleFacts::NONE);

    let plan = OutputPlan::build(
        spec.clone(),
        spec.clone(),
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputSignaling::Auto,
    )
    .unwrap();
    assert!(
        !plan.is_identity(),
        "unverified Opaque OutputPlan must not claim identity"
    );

    let ready_v = plan
        .execute_view(poisoned.as_view(), &ColorOrigin::assumed())
        .unwrap();
    assert!(matches!(ready_v.pixels, PixelCow::Owned(_)));
    assert_eq!(ready_v.pixels.as_view().row(0), &[9, 13, 17, 255]);

    let ready_b = plan.execute_buf(poisoned.clone()).unwrap();
    assert_eq!(ready_b.pixels.as_view().row(0), &[9, 13, 17, 255]);

    // Proven all_opaque on runtime view -> zero-copy Borrowed even from the same plan!
    let mut proven = PixelBuf::from_typed(vec![9u8, 13, 17, 255], 1, 1, spec.clone()).unwrap();
    proven.assume_facts(SampleFacts::NONE.with_all_opaque(true));
    let ready_zc = plan
        .execute_view(proven.as_view(), &ColorOrigin::assumed())
        .unwrap();
    assert!(
        matches!(ready_zc.pixels, PixelCow::Borrowed(_)),
        "proven all_opaque view must stay strictly zero-copy"
    );

    // Policy-proven all_opaque -> plan.is_identity() is true!
    let plan_proven = OutputPlan::build(
        spec.clone(),
        spec,
        &ConvertPolicy::permissive().with_source_facts(SampleFacts::NONE.with_all_opaque(true)),
        PeakSource::Measured,
        OutputSignaling::Auto,
    )
    .unwrap();
    assert!(plan_proven.is_identity());
}

#[test]
fn g04_output_plan_rejects_metadata_only_icc_for_structured_dst() {
    // Historical Illuminant-C Gray sRGB ICC profile (valid_use == MetadataOnly).
    let hex = [
        "000001a400000000044000006d6e74724752415958595a200000000000000000",
        "0000000061637370000000000000000000000000000000000000000000000000",
        "000000000000f6d6000100000000d32d00000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000",
        "0000000563686164000000c00000002c6b545243000000ec0000002077747074",
        "0000010c0000001464657363000001200000004a637072740000016c00000036",
        "736633320000000000010999000003e7fffff10c0000044e00010183fffffb27",
        "fffffce9000005420000b083706172610000000000030000000266660000f2a7",
        "00000d59000013d000000a5b58595a20000000000000fb240001000000012efd",
        "6d6c756300000000000000010000000c656e55530000002e0000001c00470072",
        "0061007900200048002e00320037003300200054004300310033002000430020",
        "0077006800690074006500006d6c756300000000000000010000000c656e5553",
        "0000001a0000001c005000750062006c0069006300200044006f006d00610069",
        "006e0000",
    ]
    .concat();
    let c_white_icc: Vec<u8> = hex
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect();
    let id = IccIdentification::from_icc(&c_white_icc).expect("recognized hash");
    assert_eq!(id.valid_use, IdentificationUse::MetadataOnly);

    // Must be rejected as CustomIcc for structured D65 Gray sRGB!
    let gray = ImageSpec::GRAY8_SRGB;
    let err = OutputPlan::build(
        gray.clone(),
        gray.clone(),
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputSignaling::CustomIcc(Arc::from(c_white_icc.as_slice())),
    )
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::OutputSignalingMismatch);

    // Must ALSO be rejected under SameAsOrigin if dst is structured D65 Gray sRGB!
    let plan_orig = OutputPlan::build(
        gray.clone(),
        gray.clone(),
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputSignaling::SameAsOrigin,
    )
    .unwrap();
    let buf = PixelBuf::new(2, 2, gray).unwrap();
    let c_origin = ColorOrigin::from_icc(Arc::from(c_white_icc.as_slice()));
    let err2 = plan_orig
        .execute_view(buf.as_view(), &c_origin)
        .unwrap_err();
    assert_eq!(*err2.error(), ConvertError::OutputSignalingMismatch);

    // But when destination IS ColorSpace::Icc with that exact profile, it succeeds!
    let icc_gray = ImageSpec::new(
        SampleLayout::GRAY8,
        ColorSpace::Icc {
            profile: Arc::from(c_white_icc.as_slice()),
            channels: 1,
            alpha: AlphaMode::None,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    assert!(
        OutputPlan::build(
            icc_gray.clone(),
            icc_gray,
            &ConvertPolicy::permissive(),
            PeakSource::Measured,
            OutputSignaling::CustomIcc(Arc::from(c_white_icc.as_slice())),
        )
        .is_ok()
    );
}

#[test]
fn g14_negotiate_scene_linear_hdr_precheck() {
    let lin_src = ImageSpec::RGBF32_LINEAR;
    let sdr_dst = ImageSpec::RGB8_SRGB;
    let pol_tm = ConvertPolicy::display_sdr(DiffuseWhite::BT2408, zenpixels::Nits::new(100.0));

    #[cfg(not(feature = "bt2446a"))]
    {
        // Unproven float range + tone_map_hdr_to_sdr + no bt2446a -> MUST NOT pick RGB8 sRGB!
        assert!(
            Negotiate::select(&lin_src, core::slice::from_ref(&sdr_dst), &pol_tm).is_none(),
            "unproven linear F32 with tone mapping and no bt2446a must reject SDR candidate"
        );
    }

    // When UNIT_INTERVAL is proven, tone mapping won't fire -> SDR candidate IS feasible!
    let pol_unit =
        pol_tm.with_source_facts(SampleFacts::NONE.with_float(FloatFacts::UNIT_INTERVAL));
    assert!(
        Negotiate::select(&lin_src, core::slice::from_ref(&sdr_dst), &pol_unit).is_some(),
        "unit-interval proven linear F32 must allow SDR candidate even without tone mapper"
    );
}

#[cfg(feature = "cms-moxcms")]
#[test]
fn g02_moxcms_float_mpe_and_unequal_trc_refusal_and_extended_f32() {
    use zenpixels_convert::{CmsBackend, MoxCms};

    // 1. Append a synthetic D2B1 tag to a real sRGB profile -> MoxCms refuses RelativeColorimetric
    //    source transform, while Perceptual (which checks D2B0) is unaffected.
    fn append_dummy_tag(base: &[u8], sig: &[u8; 4]) -> Vec<u8> {
        let tag_count = u32::from_be_bytes(base[128..132].try_into().unwrap()) as usize;
        let old_table_end = 132 + tag_count * 12;
        let new_table_end = old_table_end + 12;
        let mut out = Vec::with_capacity(base.len() + 28);
        out.extend_from_slice(&base[..128]);
        out.extend_from_slice(&((tag_count + 1) as u32).to_be_bytes());
        // Shift existing tag offsets by +12.
        for i in 0..tag_count {
            let off = 132 + i * 12;
            out.extend_from_slice(&base[off..off + 4]);
            let tag_off = u32::from_be_bytes(base[off + 4..off + 8].try_into().unwrap()) + 12;
            out.extend_from_slice(&tag_off.to_be_bytes());
            out.extend_from_slice(&base[off + 8..off + 12]);
        }
        // New tag entry pointing to the end of the file.
        let dummy_off = (base.len() + 12) as u32;
        out.extend_from_slice(sig);
        out.extend_from_slice(&dummy_off.to_be_bytes());
        out.extend_from_slice(&16u32.to_be_bytes());
        out.extend_from_slice(&base[old_table_end..]);
        out.extend_from_slice(&[0u8; 16]);
        let total = out.len() as u32;
        out[0..4].copy_from_slice(&total.to_be_bytes());
        assert_eq!(new_table_end, 132 + (tag_count + 1) * 12);
        out
    }

    let base_srgb = moxcms::ColorProfile::new_srgb().encode().unwrap();
    let with_d2b1 = append_dummy_tag(&base_srgb, b"D2B1");
    let src_rel = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::from(with_d2b1.as_slice()),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    let src_perc = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::from(with_d2b1.as_slice()),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::Perceptual,
        },
    );
    let dst_srgb = ImageSpec::RGB8_SRGB;
    assert!(!MoxCms.supports_specs(&src_rel, &dst_srgb));
    assert_eq!(
        *MoxCms
            .build_transform(&src_rel, &dst_srgb)
            .err()
            .unwrap()
            .error(),
        ConvertError::UnsupportedRoute
    );
    assert!(MoxCms.supports_specs(&src_perc, &dst_srgb));

    // 2. Destination B2D0 refusal for Perceptual.
    let with_b2d0 = append_dummy_tag(&base_srgb, b"B2D0");
    let dst_perc = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::from(with_b2d0.as_slice()),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::Perceptual,
        },
    );
    assert!(!MoxCms.supports_specs(&dst_srgb, &dst_perc));
    assert_eq!(
        *MoxCms
            .build_transform(&dst_srgb, &dst_perc)
            .err()
            .unwrap()
            .error(),
        ConvertError::UnsupportedRoute
    );

    // 3. RGB profile with rTRC == gTRC != bTRC must be refused.
    let mut unequal = moxcms::ColorProfile::new_srgb();
    unequal.blue_trc = Some(moxcms::ToneReprCurve::Parametric(vec![1.8]));
    let unequal_bytes = unequal.encode().unwrap();
    let src_unequal = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::from(unequal_bytes.as_slice()),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    assert!(!MoxCms.supports_specs(&src_unequal, &dst_srgb));
    assert_eq!(
        *MoxCms
            .build_transform(&src_unequal, &dst_srgb)
            .err()
            .unwrap()
            .error(),
        ConvertError::UnsupportedRoute
    );

    // 4. Extended-range F32 RGB is NOT clipped to [0, 1] on shared-curve RGB profiles!
    let mut lin_p3 = moxcms::ColorProfile::new_display_p3();
    let lin = moxcms::ToneReprCurve::Parametric(vec![1.0]);
    lin_p3.red_trc = Some(lin.clone());
    lin_p3.green_trc = Some(lin.clone());
    lin_p3.blue_trc = Some(lin);
    let lin_p3_bytes = lin_p3.encode().unwrap();
    let src_f32 = ImageSpec::new(
        SampleLayout::RGBF32,
        ColorSpace::Icc {
            profile: Arc::from(lin_p3_bytes.as_slice()),
            channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    let dst_f32 = ImageSpec::RGBF32_LINEAR;
    let tx = MoxCms.build_transform(&src_f32, &dst_f32).unwrap();
    let input = [1.5f32, 1.5, 1.5];
    let mut output = [0.0f32; 3];
    tx.transform_row(
        bytemuck::cast_slice(&input),
        bytemuck::cast_slice_mut(&mut output),
        1,
    )
    .unwrap();
    assert!(
        (output[0] - 1.5).abs() < 1e-3 && (output[1] - 1.5).abs() < 1e-3,
        "extended F32 value 1.5 must not clip to 1.0: got {output:?}"
    );
}

#[test]
fn t2_3_banded_orientation_matches_contiguous_bit_exact_all_8_orientations() {
    let orientations = [
        Orientation::Identity,
        Orientation::FlipH,
        Orientation::FlipV,
        Orientation::Rotate180,
        Orientation::Transpose,
        Orientation::Rotate90,
        Orientation::Rotate270,
        Orientation::Transverse,
    ];
    let specs = [
        ImageSpec::RGB8_SRGB,
        ImageSpec::RGBA8_SRGB,
        ImageSpec::RGB16_SRGB,
        ImageSpec::RGBAF32_LINEAR,
    ];
    let (w, h) = (37u32, 29u32);
    for spec in specs {
        let bpp = spec.layout.bytes_per_pixel();
        let mut contig = PixelBuf::new(w, h, spec.clone()).unwrap();
        let rb = contig.row_bytes();
        let mut view_mut = contig.as_view_mut();
        for y in 0..h {
            let row = view_mut.row_mut(y);
            for (x, b) in row.iter_mut().enumerate() {
                let i = (y as usize) * rb + x;
                *b = ((i.wrapping_mul(37) ^ (i >> 3) ^ 0x5A) & 0xFF) as u8;
            }
        }
        // Force uneven source bands (~7 rows/band -> 5 bands for h=29).
        let src_band_bytes = (w as usize) * bpp * 7;
        let banded_src = BandedBuf::from_view(&contig.as_view(), src_band_bytes).unwrap();
        assert!(banded_src.band_count() >= 4);

        for orient in orientations {
            let expected = apply_orientation(&contig.as_view(), orient).unwrap();
            // Use a DIFFERENT destination band height (~5 rows/band) to stress band seams!
            let (ow, oh) = orient.output_dimensions(w, h);
            let dst_band_bytes = (ow as usize) * bpp * 5;
            let got = apply_orientation_banded(&banded_src, orient, dst_band_bytes).unwrap();
            assert_eq!(got.width(), ow);
            assert_eq!(got.height(), oh);
            let got_contig = got.into_buf().unwrap();
            assert_eq!(
                got_contig.as_view().as_contiguous_bytes().unwrap(),
                expected.as_view().as_contiguous_bytes().unwrap(),
                "apply_orientation_banded mismatch for spec={:?} orient={:?}",
                spec.layout,
                orient
            );

            // Also verify zero-alloc `apply_orientation_banded_into` with 3-row dst bands.
            let mut dst_pre =
                BandedBuf::new(ow, oh, spec.clone(), (ow as usize) * bpp * 3).unwrap();
            apply_orientation_banded_into(&banded_src, orient, &mut dst_pre).unwrap();
            assert_eq!(
                dst_pre
                    .into_buf()
                    .unwrap()
                    .as_view()
                    .as_contiguous_bytes()
                    .unwrap(),
                expected.as_view().as_contiguous_bytes().unwrap(),
                "apply_orientation_banded_into mismatch for {:?}/{:?}",
                spec.layout,
                orient
            );
        }
    }
}

#[test]
fn t2_6_measure_max_smoothed_and_histogram_bins_try_merge() {
    let white = DiffuseWhite::new(100.0);
    let spec = ImageSpec::RGBF32_LINEAR.with_white(white);

    // 1. Single stuck hot pixel at 9000 nits (90.0 relative) in black 9x1 row:
    //    literal MaxCLL = 9000, smoothed MaxCLL = 9000 / 3 = 3000, MaxFALL = 1000!
    let mut row = vec![0.0f32; 9 * 3];
    row[4 * 3] = 90.0;
    let buf = PixelBuf::from_typed(row.clone(), 9, 1, spec.clone()).unwrap();
    let lit = measure_cll(&buf, white, None, LightLevelMethod::MaxRgb).unwrap();
    let sm = measure_max_smoothed(&buf, white, None, LightLevelMethod::MaxRgb).unwrap();
    assert_eq!(lit.max_content_light_level, 9000);
    assert_eq!(sm.max_content_light_level, 3000);
    assert_eq!(
        sm.max_frame_average_light_level,
        lit.max_frame_average_light_level
    );

    // 2. 3-pixel-wide bright bar at 9000 nits survives smoothed measurement at 9000!
    row[3 * 3] = 90.0;
    row[5 * 3] = 90.0;
    let buf3 = PixelBuf::from_typed(row, 9, 1, spec.clone()).unwrap();
    let sm3 = measure_max_smoothed(&buf3, white, None, LightLevelMethod::MaxRgb).unwrap();
    assert_eq!(sm3.max_content_light_level, 9000);

    // 3. LightLevelHistogram::bins() + try_merge() matches full-frame histogram exactly!
    let mut img = vec![0.0f32; 8 * 4 * 3];
    for p in 0..(8 * 4) {
        let v = if p % 5 == 0 { 0.0 } else { (p as f32) * 0.5 };
        img[p * 3] = v;
        img[p * 3 + 1] = v * 0.5;
        img[p * 3 + 2] = v * 0.25;
    }
    let full = PixelBuf::from_typed(img, 8, 4, spec).unwrap();
    let h_full = measure_histogram(&full, white, None, LightLevelMethod::MaxRgb).unwrap();
    let mut h_top = measure_histogram(
        &full.as_view().sub_rows(0, 2).unwrap(),
        white,
        None,
        LightLevelMethod::MaxRgb,
    )
    .unwrap();
    let h_bot = measure_histogram(
        &full.as_view().sub_rows(2, 2).unwrap(),
        white,
        None,
        LightLevelMethod::MaxRgb,
    )
    .unwrap();
    h_top.merge(&h_bot).unwrap();
    assert_eq!(h_top.bins(), h_full.bins());
    assert_eq!(h_top.total_pixels(), h_full.total_pixels());
    assert_eq!(h_top.zero_pixels(), h_full.zero_pixels());
    assert_eq!(h_top.max_nits(), h_full.max_nits());
    assert!((h_top.mean_nits() - h_full.mean_nits()).abs() < 1e-4);

    // Merging mismatched method must fail cleanly.
    let h_luma = LightLevelHistogram::new(LightLevelMethod::LuminanceBt2020);
    assert_eq!(
        *h_top.merge(&h_luma).unwrap_err().error(),
        ConvertError::UnsupportedRoute
    );
}

#[test]
fn t2_4_t2_5_adapt_for_encode_convenience_helpers_and_ycbcr_from_planar() {
    // 1. adapt_for_encode zero-copy identity + conversion.
    let buf = PixelBuf::from_typed(vec![10u8, 20, 30, 255], 1, 1, ImageSpec::RGBA8_SRGB).unwrap();
    let supported = [ImageSpec::RGB8_SRGB, ImageSpec::RGBA8_SRGB];
    let (neg, ready) = adapt_for_encode(
        buf.as_view(),
        buf.origin(),
        &supported,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputSignaling::Auto,
    )
    .unwrap();
    assert_eq!(neg.spec, ImageSpec::RGBA8_SRGB);
    assert!(matches!(ready.pixels, PixelCow::Borrowed(_)));

    // 2. to_rgb8 / to_rgba8 / to_gray8 / to_bgra8 convenience wrappers.
    // Unproven straight-alpha RGBA8 is refused by alpha-dropping helpers (DiscardIfOpaque):
    assert_eq!(
        *to_rgb8(&buf.as_view()).unwrap_err().error(),
        ConvertError::AlphaLossForbidden
    );
    assert_eq!(
        *to_gray8(&buf.as_view()).unwrap_err().error(),
        ConvertError::AlphaLossForbidden
    );
    let ra = to_rgba8(&buf.as_view()).unwrap();
    assert_eq!(ra.as_view().row(0), &[10, 20, 30, 255]);
    let bg = to_bgra8(&buf.as_view()).unwrap();
    assert_eq!(bg.as_view().row(0), &[30, 20, 10, 255]);
    // Once opacity is proven (scan_facts or into_opaque_verified), to_rgb8 and to_gray8 succeed:
    let mut proven_buf = buf.clone();
    let facts = zenpixels_convert::scan_facts(&proven_buf.as_view());
    proven_buf.assume_facts(facts);
    let r = to_rgb8(&proven_buf.as_view()).unwrap();
    assert_eq!(r.as_view().row(0), &[10, 20, 30]);
    let gr = to_gray8(&proven_buf.as_view()).unwrap();
    assert_eq!(
        gr.as_view().row(0),
        &[((10u16 * 54 + 20 * 183 + 30 * 19 + 128) >> 8) as u8]
    );

    // 3. YcbcrMatrix::from_cicp_code + to_cicp_code (lossless) + ycbcr_to_rgb_from_planar.
    assert_eq!(YcbcrMatrix::from_cicp_code(0), Some(YcbcrMatrix::Identity));
    assert_eq!(YcbcrMatrix::from_cicp_code(1), Some(YcbcrMatrix::Bt709));
    assert_eq!(YcbcrMatrix::from_cicp_code(5), Some(YcbcrMatrix::Bt470Bg));
    assert_eq!(YcbcrMatrix::from_cicp_code(6), Some(YcbcrMatrix::Bt601));
    assert!(YcbcrMatrix::Bt470Bg.same_coefficients(YcbcrMatrix::Bt601));
    assert_eq!(YcbcrMatrix::Bt470Bg.to_cicp_code(), 5);
    assert_eq!(YcbcrMatrix::Bt601.to_cicp_code(), 6);
    assert_eq!(YcbcrMatrix::from_cicp_code(9), Some(YcbcrMatrix::Bt2020));
    assert_eq!(YcbcrMatrix::from_cicp_code(2), None);

    let y_p = PlaneBuf::from_external(
        vec![235u8; 4],
        2,
        2,
        2,
        PlaneSpec::new(PlaneRole::Y, SampleType::U8),
    )
    .unwrap();
    let cb_p = PlaneBuf::from_external(
        vec![128u8],
        1,
        1,
        1,
        PlaneSpec::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2),
    )
    .unwrap();
    let cr_p = PlaneBuf::from_external(
        vec![128u8],
        1,
        1,
        1,
        PlaneSpec::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2),
    )
    .unwrap();
    let untagged_planar = PlanarBuf::from_planes(2, 2, vec![y_p, cb_p, cr_p]).unwrap();
    // Missing color/range/matrix metadata must refuse rather than guess!
    assert_eq!(
        *reconstruct_planar_f32(&untagged_planar.as_view())
            .unwrap_err()
            .error(),
        ConvertError::UnsupportedRoute
    );

    let tagged = untagged_planar
        .with_color(ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Bt709,
            range: SignalRange::Narrow,
            alpha: AlphaMode::None,
            white: None,
        })
        .with_matrix_coefficients_code(1);
    let from_meta = reconstruct_planar_f32(&tagged.as_view()).unwrap();
    let via_to_buf = reconstruct_planar_to_buf(
        &tagged.as_view(),
        SampleLayout::RGBF32,
        &PlanarReconstructionOptions::nearest(),
    )
    .unwrap();
    assert_eq!(
        from_meta.as_view().as_contiguous_bytes().unwrap(),
        via_to_buf.as_view().as_contiguous_bytes().unwrap()
    );
}

#[test]
fn t2_11_exhaustive_rgba8_premul_and_lin_f32_to_srgb_u8_parity() {
    // Exhaustive 65,536 (c, a) pairs for rgba8_straight_to_premul (both round=true and round=false).
    let src_spec = ImageSpec::RGBA8_SRGB;
    let dst_spec = ImageSpec::RGBA8_SRGB.with_alpha(AlphaMode::Premultiplied);
    let mut pairs = Vec::with_capacity(65536 * 4);
    for a in 0u8..=255 {
        for c in 0u8..=255 {
            pairs.extend_from_slice(&[c, c.wrapping_add(37), 255 - c, a]);
        }
    }
    for depth in [DepthPolicy::Round, DepthPolicy::Truncate] {
        let p_sc = RowPlan::build(
            src_spec.clone(),
            dst_spec.clone(),
            &ConvertPolicy::permissive()
                .with_depth(depth)
                .with_accelerators(Accelerators::SCALAR),
            None,
        )
        .unwrap();
        let p_all = RowPlan::build(
            src_spec.clone(),
            dst_spec.clone(),
            &ConvertPolicy::permissive()
                .with_depth(depth)
                .with_accelerators(Accelerators::ALL),
            None,
        )
        .unwrap();
        let mut d_sc = vec![0u8; pairs.len()];
        let mut d_all = vec![0u8; pairs.len()];
        p_sc.convert_row(&pairs, &mut d_sc, 65536, 0).unwrap();
        p_all.convert_row(&pairs, &mut d_all, 65536, 0).unwrap();
        assert_eq!(
            d_sc, d_all,
            "rgba8 premul SCALAR vs ALL mismatch for {depth:?}"
        );
    }

    // LinF32ToSrgbU8 exact (allow_approximate=false) and Option B approx (allow_approximate=true):
    // both MUST have 0-bit parity between SCALAR and ALL, and Option B must be within <= 1 LSB of exact!
    let src_f32 = ImageSpec::RGBAF32_LINEAR;
    let dst_u8 = ImageSpec::RGBA8_SRGB;
    let width = 1029u32; // Exercises 16-wide, 8-wide, and tail loops!
    let mut vals = Vec::with_capacity(width as usize * 4);
    for i in 0..width as usize {
        let t = (i as f32) / ((width - 1) as f32);
        vals.extend_from_slice(&[t, t * t, (t * 1.2 - 0.1), (i % 256) as f32 / 255.0]);
    }
    let src_bytes: &[u8] = bytemuck::cast_slice(&vals);

    let p_ex_sc = RowPlan::build(
        src_f32.clone(),
        dst_u8.clone(),
        &ConvertPolicy::permissive().with_accelerators(Accelerators::SCALAR),
        None,
    )
    .unwrap();
    let p_ex_all = RowPlan::build(
        src_f32.clone(),
        dst_u8.clone(),
        &ConvertPolicy::permissive().with_accelerators(Accelerators::ALL),
        None,
    )
    .unwrap();
    let p_ap_sc = RowPlan::build(
        src_f32.clone(),
        dst_u8.clone(),
        &ConvertPolicy::permissive()
            .with_approximate(true)
            .with_accelerators(Accelerators::SCALAR),
        None,
    )
    .unwrap();
    let p_ap_all = RowPlan::build(
        src_f32,
        dst_u8,
        &ConvertPolicy::permissive()
            .with_approximate(true)
            .with_accelerators(Accelerators::ALL),
        None,
    )
    .unwrap();

    let mut ex_sc = vec![0u8; width as usize * 4];
    let mut ex_all = vec![0u8; width as usize * 4];
    let mut ap_sc = vec![0u8; width as usize * 4];
    let mut ap_all = vec![0u8; width as usize * 4];
    p_ex_sc
        .convert_row(src_bytes, &mut ex_sc, width, 0)
        .unwrap();
    p_ex_all
        .convert_row(src_bytes, &mut ex_all, width, 0)
        .unwrap();
    p_ap_sc
        .convert_row(src_bytes, &mut ap_sc, width, 0)
        .unwrap();
    p_ap_all
        .convert_row(src_bytes, &mut ap_all, width, 0)
        .unwrap();

    assert_eq!(ex_sc, ex_all, "LinF32ToSrgbU8 exact SCALAR vs ALL mismatch");
    assert_eq!(
        ap_sc, ap_all,
        "LinF32ToSrgbU8 approx SCALAR vs ALL mismatch"
    );
    let max_err = ex_all
        .iter()
        .zip(ap_all.iter())
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap_or(0);
    assert!(
        max_err <= 1,
        "LinF32ToSrgbU8 Option B max_err={max_err} exceeded 1 LSB"
    );
}
