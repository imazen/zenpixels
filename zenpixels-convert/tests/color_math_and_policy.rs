//! Exact colorimetry, transfer accuracy, gamut ordering, policy refusal gates,
//! Oklab CIE XYZ D65 anchoring, PQ NaN flushing, Narrow<->Full, and Perceptual spaces.

use zenpixels_convert::{
    AlphaMode, AlphaPolicy, Color, ColorSpace, ConvertError, ConvertPolicy, DepthPolicy,
    DiffuseWhite, ExactDepth, GamutPolicy, HlgMode, HlgRendering, ImageSpec, Nits, PeakSource,
    PixelBuf, Primaries, RangePolicy, RowPlan, SampleFacts, SampleLayout, SignalRange, Transfer,
    convert_buf, scan_facts,
};

#[test]
fn linear_bt709_to_srgb_u8_display_p3_applies_gamut_matrix_before_clamping() {
    // Out-of-sRGB linear color in BT.709 with negative red (`R = -0.15, G = 0.85, B = 0.05`)
    // that maps INSIDE Display P3 gamut! If clamping happened before the 3x3 matrix,
    // R would be clipped to 0.0 and shift the resulting P3 color visibly.
    let src = ImageSpec::rgbf32_linear(); // Bt709
    let dst = ImageSpec::rgb8_srgb().with_primaries(Primaries::DisplayP3);

    let px_in = [-0.15f32, 0.85, 0.05];
    let expected_lin_p3 = Primaries::Bt709
        .gamut_matrix_to(Primaries::DisplayP3)
        .unwrap();
    let p3_r = expected_lin_p3[0][0] * px_in[0]
        + expected_lin_p3[0][1] * px_in[1]
        + expected_lin_p3[0][2] * px_in[2];
    assert!(
        p3_r > 0.01,
        "test vector must have positive Display P3 red after matrix: {p3_r}"
    );
    let expected_u8_r = (linear_srgb::precise::linear_to_srgb(p3_r) * 255.0 + 0.5) as u8;
    // Compare against wrong order (clamp before matrix):
    let wrong_r = expected_lin_p3[0][1] * px_in[1] + expected_lin_p3[0][2] * px_in[2];
    let wrong_u8_r = (linear_srgb::precise::linear_to_srgb(wrong_r) * 255.0 + 0.5) as u8;
    assert_ne!(expected_u8_r, wrong_u8_r);

    let plan = RowPlan::build(src, dst, &ConvertPolicy::permissive(), None).unwrap();
    let mut out = [0u8; 3];
    plan.convert_row(bytemuck::cast_slice(&px_in), &mut out, 1, 0)
        .unwrap();
    assert!((i16::from(out[0]) - i16::from(expected_u8_r)).abs() <= 1);
}

#[cfg(feature = "oklab")]
#[test]
fn display_p3_oklab_roundtrip_is_exact_and_primaries_anchored_via_xyz_d65() {
    let p3 = ImageSpec::rgb8_srgb().with_primaries(Primaries::DisplayP3);
    let to_ok = RowPlan::build(
        p3.clone(),
        ImageSpec::oklab_f32(),
        &ConvertPolicy::working_linear(),
        None,
    )
    .unwrap();
    let from_ok = RowPlan::build(
        ImageSpec::oklab_f32(),
        p3,
        &ConvertPolicy::permissive(),
        None,
    )
    .unwrap();

    let src = [0u8, 255, 128, 255, 0, 64, 42, 180, 210];
    let mut oklab = [0u8; 3 * 12];
    let mut back = [0u8; 9];
    to_ok.convert_row(&src, &mut oklab, 3, 0).unwrap();
    from_ok.convert_row(&oklab, &mut back, 3, 0).unwrap();
    assert_eq!(
        back, src,
        "DisplayP3 -> OKLABF32 -> DisplayP3 must round-trip bit-exact"
    );
}

#[cfg(feature = "hdr-transfers")]
#[test]
fn pq_flushes_rgb_nan_to_zero_while_leaving_alpha_untouched() {
    let pq_rgba = ImageSpec::new(
        SampleLayout::RGBAF32,
        ColorSpace::BT2100_PQ.with_alpha(AlphaMode::Straight),
    );
    let lin_rgba = ImageSpec::rgbaf32_linear()
        .with_primaries(Primaries::Bt2020)
        .with_white(DiffuseWhite::BT2408);

    let plan = RowPlan::build(pq_rgba, lin_rgba, &ConvertPolicy::working_linear(), None).unwrap();
    let alpha_bits = 0x7fc0_1234u32; // custom quiet NaN payload on alpha
    let src = [f32::NAN, 0.5f32, -0.25f32, f32::from_bits(alpha_bits)];
    let mut dst = [0.0f32; 4];
    plan.convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut dst),
        1,
        0,
    )
    .unwrap();

    assert_eq!(dst[0], 0.0, "PQ NaN red must flush to 0.0");
    assert!(dst[1].is_finite() && dst[1] > 0.0);
    assert_eq!(dst[2], 0.0, "PQ negative blue must flush to 0.0");
    assert_eq!(
        dst[3].to_bits(),
        alpha_bits,
        "alpha must pass through untouched"
    );
}

#[test]
fn depth_policy_truncate_vs_round_and_bayer_ordered_dither() {
    let src = ImageSpec::new(SampleLayout::GRAYF32, ColorSpace::GRAY_LINEAR);
    let dst = ImageSpec::new(SampleLayout::GRAY8, ColorSpace::GRAY_LINEAR);

    // 100.75 / 255.0 -> Round gives 101, Truncate gives 100!
    let val = [100.75f32 / 255.0];
    let mut out_round = [0u8; 1];
    let mut out_trunc = [0u8; 1];
    RowPlan::build(
        src.clone(),
        dst.clone(),
        &ConvertPolicy::permissive().with_depth(DepthPolicy::Round),
        None,
    )
    .unwrap()
    .convert_row(bytemuck::cast_slice(&val), &mut out_round, 1, 0)
    .unwrap();
    RowPlan::build(
        src.clone(),
        dst.clone(),
        &ConvertPolicy::permissive().with_depth(DepthPolicy::Truncate),
        None,
    )
    .unwrap()
    .convert_row(bytemuck::cast_slice(&val), &mut out_trunc, 1, 0)
    .unwrap();
    assert_eq!(out_round[0], 101);
    assert_eq!(out_trunc[0], 100);

    // OrderedDither across a flat 100.5/255.0 4x4 block produces ~half 100 and ~half 101!
    let plan_d = RowPlan::build(
        src,
        dst,
        &ConvertPolicy::permissive().with_depth(DepthPolicy::OrderedDither),
        None,
    )
    .unwrap();
    let flat = [100.5f32 / 255.0; 4];
    let mut c100 = 0;
    let mut c101 = 0;
    for y in 0..4 {
        let mut row = [0u8; 4];
        plan_d
            .convert_row(bytemuck::cast_slice(&flat), &mut row, 4, y)
            .unwrap();
        for b in row {
            match b {
                100 => c100 += 1,
                101 => c101 += 1,
                other => panic!("unexpected dither code {other}"),
            }
        }
    }
    assert_eq!(c100, 8);
    assert_eq!(c101, 8);
}

#[test]
fn policy_refusal_gates_and_sample_facts_unlock_lossless() {
    // DepthPolicy::Forbid refuses U16 -> U8 normally
    let err = RowPlan::build(
        ImageSpec::rgb16_srgb(),
        ImageSpec::rgb8_srgb(),
        &ConvertPolicy::lossless(),
        None,
    )
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::DepthReductionForbidden);

    // ...but succeeds when SampleFacts proves exact 8-bit full-range values!
    let facts_8 = SampleFacts::from_depth(
        ExactDepth::new(zenpixels_convert::SampleType::U16, SignalRange::Full, 8, 8).unwrap(),
    )
    .with_all_opaque(true);
    assert!(
        RowPlan::build(
            ImageSpec::rgb16_srgb(),
            ImageSpec::rgb8_srgb(),
            &ConvertPolicy::lossless().with_facts(facts_8),
            None
        )
        .is_ok()
    );

    // AlphaPolicy::Forbid refuses RGBA -> RGB
    let err = RowPlan::build(
        ImageSpec::rgba8_srgb(),
        ImageSpec::rgb8_srgb(),
        &ConvertPolicy::lossless(),
        None,
    )
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::AlphaLossForbidden);

    // AlphaPolicy::DiscardIfOpaque refuses unproven RGBA -> RGB, succeeds once scan_facts proves opacity!
    let mut rgba_buf = PixelBuf::new(4, 2, ImageSpec::rgba8_srgb()).unwrap();
    for y in 0..2 {
        for px in rgba_buf.as_view_mut().row_mut(y).as_chunks_mut::<4>().0 {
            px.copy_from_slice(&[100, 150, 200, 255]);
        }
    }
    let err = convert_buf(
        &rgba_buf.as_view(),
        ImageSpec::rgb8_srgb(),
        &ConvertPolicy::working_linear().with_alpha(AlphaPolicy::DiscardIfOpaque),
        PeakSource::Measured,
    )
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::AlphaLossForbidden);

    let scanned = scan_facts(&rgba_buf.as_view());
    assert!(scanned.all_opaque);
    rgba_buf.set_facts(scanned);
    let out = convert_buf(
        &rgba_buf.as_view(),
        ImageSpec::rgb8_srgb(),
        &ConvertPolicy::working_linear().with_alpha(AlphaPolicy::DiscardIfOpaque),
        PeakSource::Measured,
    )
    .unwrap();
    assert_eq!(&out.as_view().row(0)[0..3], &[100, 150, 200]);

    // RangePolicy::Forbid refuses Narrow <-> Full
    let err = RowPlan::build(
        ImageSpec::rgb8_srgb().with_range(SignalRange::Narrow),
        ImageSpec::rgb8_srgb(),
        &ConvertPolicy::lossless(),
        None,
    )
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::RangeConversionForbidden);

    // GamutPolicy::Forbid refuses P3 -> Bt709
    let err = RowPlan::build(
        ImageSpec::rgb8_srgb().with_primaries(Primaries::DisplayP3),
        ImageSpec::rgb8_srgb(),
        &ConvertPolicy::permissive().with_gamut(GamutPolicy::Forbid),
        None,
    )
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::GamutLossForbidden);

    // MissingHdrAnchor: PQ -> Linear without DiffuseWhite
    #[cfg(feature = "hdr-transfers")]
    {
        let err = RowPlan::build(
            ImageSpec::rgb16_bt2100_pq(),
            ImageSpec::rgbf32_linear().with_primaries(Primaries::Bt2020),
            &ConvertPolicy::permissive(),
            None,
        )
        .unwrap_err();
        assert_eq!(*err.error(), ConvertError::MissingHdrAnchor);

        // HlgRenderingRequired: HLG Scene -> Linear without HlgRendering
        let err = RowPlan::build(
            ImageSpec::rgb16_bt2100_hlg_scene(),
            ImageSpec::rgbf32_linear().with_primaries(Primaries::Bt2020),
            &ConvertPolicy::working_linear(),
            None,
        )
        .unwrap_err();
        assert_eq!(*err.error(), ConvertError::HlgRenderingRequired);
    }

    // CmsRequired: CMYK -> sRGB without CmsBackend
    let err = RowPlan::build(
        ImageSpec::cmyk8(None),
        ImageSpec::rgb8_srgb(),
        &ConvertPolicy::permissive(),
        None,
    )
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::CmsRequired);
}

#[test]
fn narrow_full_roundtrip_and_linear_matte_composite() {
    // Narrow U8 (16..=235) -> Full U8 (0..=255): 16 -> 0, 235 -> 255
    let narrow = ImageSpec::rgb8_srgb().with_range(SignalRange::Narrow);
    let full = ImageSpec::rgb8_srgb();
    let plan = RowPlan::build(
        narrow,
        full,
        &ConvertPolicy::permissive().with_range(RangePolicy::Convert),
        None,
    )
    .unwrap();
    let src = [16u8, 126, 235];
    let mut dst = [0u8; 3];
    plan.convert_row(&src, &mut dst, 1, 0).unwrap();
    assert_eq!(dst[0], 0);
    assert_eq!(dst[2], 255);
    assert!((i16::from(dst[1]) - 128).abs() <= 1);

    // 50% opaque white over linear black matte -> linear 0.5 -> sRGB ~188 (NOT naive gamma 128!)
    let matte_plan = RowPlan::build(
        ImageSpec::rgba8_srgb(),
        ImageSpec::rgb8_srgb(),
        &ConvertPolicy::permissive().with_alpha(AlphaPolicy::CompositeOnto(Color::BLACK)),
        None,
    )
    .unwrap();
    let mut out = [0u8; 3];
    matte_plan
        .convert_row(&[255, 255, 255, 128], &mut out, 1, 0)
        .unwrap();
    assert!(
        (i16::from(out[0]) - 188).abs() <= 1,
        "linear-light 50% white over black must be ~188, got {}",
        out[0]
    );
}

#[cfg(feature = "hdr-transfers")]
#[test]
fn pq_and_hlg_accuracy_bounds() {
    // A05: Exhaustive ALL 65,536 PQ U16 codes (not step_by(257)) plus independent binary64
    // SMPTE ST 2084 reference oracle in BOTH directions!
    fn st2084_eotf_nits(e: f64) -> f64 {
        const M1: f64 = 2610.0 / 16384.0;
        const M2: f64 = 2523.0 / 32.0;
        const C1: f64 = 3424.0 / 4096.0;
        const C2: f64 = 2413.0 / 128.0;
        const C3: f64 = 2392.0 / 128.0;
        let p = e.max(0.0).powf(1.0 / M2);
        let num = (p - C1).max(0.0);
        let den = C2 - C3 * p;
        10_000.0 * (num / den).powf(1.0 / M1)
    }

    let pq = ImageSpec::rgb16_bt2100_pq();
    let lin = ImageSpec::rgbf32_linear()
        .with_primaries(Primaries::Bt2020)
        .with_white(DiffuseWhite::BT2408);
    let to_lin = RowPlan::build(
        pq.clone(),
        lin.clone(),
        &ConvertPolicy::working_linear(),
        None,
    )
    .unwrap();
    let to_pq = RowPlan::build(lin.clone(), pq, &ConvertPolicy::working_linear(), None).unwrap();

    // Test every single code 0..=65535 (each pixel carries 3 distinct codes to catch lane crosstalk).
    let n_codes = 65_536usize;
    let w = n_codes as u32;
    let mut src_u16 = Vec::with_capacity(n_codes * 3);
    for i in 0..n_codes {
        src_u16.push(i as u16);
        src_u16.push((65_535 - i) as u16);
        src_u16.push(((i * 13) & 0xFFFF) as u16);
    }
    let mut lin_f32 = vec![0f32; n_codes * 3];
    let mut back_u16 = vec![0u16; n_codes * 3];
    to_lin
        .convert_row(
            bytemuck::cast_slice(&src_u16),
            bytemuck::cast_slice_mut(&mut lin_f32),
            w,
            0,
        )
        .unwrap();
    to_pq
        .convert_row(
            bytemuck::cast_slice(&lin_f32),
            bytemuck::cast_slice_mut(&mut back_u16),
            w,
            0,
        )
        .unwrap();

    let mut max_rt_err = 0i32;
    for i in 0..n_codes {
        let code = i as u16;
        let rt_err = (i32::from(code) - i32::from(back_u16[i * 3])).abs();
        if rt_err > max_rt_err {
            max_rt_err = rt_err;
        }
        // Check all 3 channels match for the same code (no lane crosstalk).
        assert_eq!(back_u16[i * 3], back_u16[(65_535 - i) * 3 + 1]);
        // Independent ST 2084 binary64 EOTF witness at key luminance checkpoints (0, 1, 2, knee, 203 nits, 1000, peak).
        let got_nits = f64::from(lin_f32[i * 3]) * 203.0;
        let ref_nits = st2084_eotf_nits(f64::from(code) / 65_535.0);
        if code == 0 {
            assert_eq!(got_nits, 0.0, "PQ code 0 must decode to exact 0.0 nits");
        } else if code == 1 {
            assert!(
                got_nits > 0.0 && (got_nits - ref_nits).abs() <= ref_nits * 5e-3 + 1e-10,
                "PQ code 1: got={got_nits} ref={ref_nits}"
            );
        } else {
            let tol = ref_nits * 2e-4 + 1e-6;
            assert!(
                (got_nits - ref_nits).abs() <= tol,
                "PQ EOTF drift at code {code}: got={got_nits} nits, ref={ref_nits} nits"
            );
        }
    }
    assert!(
        max_rt_err <= 1,
        "exhaustive 65,536-code PQ roundtrip exceeded ±1 u16: max_err={max_rt_err}"
    );
    assert_eq!(back_u16[0], 0);
    assert_eq!(back_u16[65_535 * 3], 65_535);

    // HLG BT.2100-3 with black lift Lb=0.05: code 0.0 decodes to exact 0.05 / 203.0!
    let hlg_lift = HlgRendering::new(Nits::new(1000.0), 0.05, 1.2).unwrap();
    let hlg_spec = ImageSpec::new(
        SampleLayout::RGBF32,
        ColorSpace::BT2100_HLG_SCENE.with_transfer(Transfer::Hlg(HlgMode::Display(hlg_lift))),
    );
    let p_hlg = RowPlan::build(hlg_spec, lin, &ConvertPolicy::working_linear(), None).unwrap();
    let black = [0.0f32; 3];
    let mut out = [0.0f32; 3];
    p_hlg
        .convert_row(
            bytemuck::cast_slice(&black),
            bytemuck::cast_slice_mut(&mut out),
            1,
            0,
        )
        .unwrap();
    let want = 0.05 / 203.0;
    assert!(
        (out[0] - want).abs() < 1e-6,
        "HLG black lift: got {}, want {want}",
        out[0]
    );
}

#[cfg(feature = "perceptual")]
#[test]
fn perceptual_spaces_roundtrip_accurately() {
    use zenpixels_convert::{ChannelOrder, PerceptualSpace, SampleType};
    let lin = ImageSpec::rgbf32_linear();
    let sample = [0.6f32, 0.35, 0.2];
    for space in [
        PerceptualSpace::Oklch,
        PerceptualSpace::CieLab,
        PerceptualSpace::CieLch,
        PerceptualSpace::Jzazbz,
        PerceptualSpace::Jzczhz,
        PerceptualSpace::Hsl,
        PerceptualSpace::Hsv,
    ] {
        let p_spec = ImageSpec::new(
            SampleLayout::new(SampleType::F32, ChannelOrder::Perceptual3),
            ColorSpace::Perceptual {
                space,
                alpha: AlphaMode::None,
            },
        );
        let fwd = RowPlan::build(
            lin.clone(),
            p_spec.clone(),
            &ConvertPolicy::working_linear(),
            None,
        )
        .unwrap();
        let inv =
            RowPlan::build(p_spec, lin.clone(), &ConvertPolicy::working_linear(), None).unwrap();
        let mut mid = [0.0f32; 3];
        let mut back = [0.0f32; 3];
        fwd.convert_row(
            bytemuck::cast_slice(&sample),
            bytemuck::cast_slice_mut(&mut mid),
            1,
            0,
        )
        .unwrap();
        inv.convert_row(
            bytemuck::cast_slice(&mid),
            bytemuck::cast_slice_mut(&mut back),
            1,
            0,
        )
        .unwrap();
        for c in 0..3 {
            assert!(
                (back[c] - sample[c]).abs() < 1e-3,
                "{space:?} roundtrip ch {c}: got {} want {}",
                back[c],
                sample[c]
            );
        }
    }
}
